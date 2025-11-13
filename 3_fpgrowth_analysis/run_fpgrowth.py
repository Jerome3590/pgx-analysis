# run_fpgrowth.py

import os
import sys
import argparse
import re
import json

# Set root of project (e.g., /home/pgx3874/pgx-analysis)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if project_root not in sys.path:
    sys.path.append(project_root)

# Project utilities
from helpers_1997_13.common_imports import (
    s3_client,
    S3_BUCKET,
    List,
    ClientError,
    pd
)

# logging utilities are available in helpers_1997_13.logging_utils (use within functions as needed)

from helpers_1997_13.s3_utils import (
    s3_exists,
    parse_s3_path,
    parse_path_params,
    get_output_paths, 
    save_to_s3_parquet, 
    save_to_s3_json
)

from helpers_1997_13.constants import (
    RICHMOND_ZIP_CODES,
    EXCLUDED_CODES,
    TOP_K,
    BASE_PATH_COHORT,
    BASE_PATH_FEATURES,
    METRIC_COLUMNS
)

from helpers_1997_13.aws_utils import (
    get_instance_id,
    notify_error,
    send_email
)

from helpers_1997_13.duckdb_utils import (
    get_duckdb_connection,
)

from helpers_1997_13.data_utils import (
    is_excluded_code,
    handle_empty_filtered_cohort
)

from helpers_1997_13.cohort_utils import (
    get_cohort_paths,
    check_cohort_needs_processing
)

from helpers_1997_13.fpgrowth_utils import (
    extract_tokens,
    run_fpgrowth_drug_token_with_fallback,
    generate_rules_from_itemsets,
    convert_frozensets,
    clean_rules_dataframe,
    expand_pattern_metrics,
    save_feature_artifacts
)

from helpers_1997_13.drug_utils import (
    encode_drug_name,
    save_drug_encoding_map
)

from helpers_1997_13.visualization_utils import create_network_visualization


def feature_engineer(cohort_name, age_band, event_year, paths, logger, bucket_name="pgxdatalake"):
    try:
        cohort_file_path = get_cohort_paths(cohort_name, age_band, event_year, bucket_name)

        if not cohort_file_path or not s3_exists(cohort_file_path):
            logger.warning(f"Cohort data not found at {cohort_file_path}")
            return None

        # Use DuckDB to read the actual file
        con = get_duckdb_connection(logger=logger)
        df = con.execute(f"SELECT * FROM read_parquet('{cohort_file_path}')").df()
        df.columns = [col.lower() for col in df.columns]

        if "drug_name" in df.columns:
            df["drug_tokens"] = df.apply(lambda row: extract_tokens(row, token_type="drug", logger=logger), axis=1)
        else:
            logger.error("Missing 'drug_name' column needed to extract 'drug_tokens'")
            return None

        df["cohort_name"] = cohort_name
        df["age_band"] = age_band
        df["event_year"] = event_year

        logger.info(f"Loaded {len(df)} records for {cohort_name}/{age_band}/{event_year}")

        df, itemsets, rules = run_fpgrowth_drug_token_with_fallback(
            df, cohort_name, age_band, event_year, paths, logger, support_start=0.01, TOP_K=100
        )

        fallback_needed = (
            itemsets.empty or
            (isinstance(itemsets.get("itemsets"), pd.Series) and itemsets.get("itemsets").iloc[0] == "No itemsets generated") or
            rules.empty or
            (isinstance(rules.get("antecedents"), pd.Series) and rules.get("antecedents").iloc[0] == "No rules generated")
        )

        if rules.empty or itemsets.empty:
            logger.info("ℹ️ No valid itemsets/rules generated — returned with placeholder outputs")

        # === Feature Encode Drug Names ===
        encoding_map = {}
        try:
            logger.info("Encoding drug names with FpGrowth metrics and linguistic features...")
            for _, row in itemsets.iterrows():
                for drug in row["itemsets"]:
                    encoding = encode_drug_name(drug, {
                        "support": row.get("support", 0),
                        "confidence": row.get("confidence", 0),
                        "certainty": row.get("certainty", 0)
                    }, logger)
                    encoding_map[drug] = encoding

            df["drug_encoding"] = df["drug_tokens"].apply(
                lambda tokens: [encoding_map.get(d, "X000000000000000") for d in tokens] if isinstance(tokens, list) else []
            )

            df["encoded_drug_name"] = df["drug_encoding"]

            save_drug_encoding_map(encoding_map, cohort_name, age_band, event_year, logger=logger)

        except Exception as e:
            logger.error(f"Error during drug encoding: {e}")

        # === Expand Pattern Metrics and Save Manifest ===
        manifest = {}
        try:
            df, manifest = expand_pattern_metrics(
                df, itemsets, rules, cohort_name, age_band, event_year,
                s3_bucket=bucket_name, encoding_map=encoding_map, logger=logger
            )
        except Exception as e:
            logger.error(f"Error in expand_pattern_metrics: {e}")

        # === Optimized Network Visualization Handling ===
        try:
            if not rules.empty:
                itemsets_counts = {
                    item: row["support"]
                    for _, row in itemsets.iterrows()
                    for item in convert_frozensets(row["itemsets"])
                }

                if isinstance(rules, pd.DataFrame):
                    rules_clean = pd.DataFrame(convert_frozensets(rules))
                    rules_clean = clean_rules_dataframe(rules_clean, logger)

                    net_result = create_network_visualization(
                        rules_df=rules_clean,
                        title=f"{cohort_name.upper()} Drug Network: Age {age_band} Year {event_year}",
                        cohort_name=cohort_name,
                        age_band=age_band,
                        event_year=event_year,
                        itemsets_counts=itemsets_counts,
                        logger=logger
                    )
                    logger.info(f"✓ Network visualization created with {net_result['num_nodes']} nodes and {net_result['num_edges']} edges.")
                else:
                    logger.info("Skipping network visualization: Rules object is not a valid DataFrame.")
            else:
                logger.info("Skipping network visualization: No rules available")
        except Exception as e:
            logger.error(f"✗ Error generating network visualization: {e}")

        # === Save Outputs ===
        try:
            itemsets_json_safe = convert_frozensets(itemsets.to_dict(orient='records'))
            rules_json_safe = convert_frozensets(rules.to_dict(orient='records'))

            partition_cols = ["cohort_name", "age_band", "event_year"]

            save_to_s3_parquet(df, paths['fpgrowth_features_parquet'], logger, partition_cols=partition_cols)
            save_to_s3_json(itemsets_json_safe, paths['itemsets_json'], logger, partition_cols=partition_cols)
            save_to_s3_json(rules_json_safe, paths['rules_json'], logger, partition_cols=partition_cols)
            save_to_s3_json(manifest, paths['features_manifest_json'], logger, partition_cols=partition_cols)

            logger.info(f"✓ Feature engineering completed for {cohort_name}/{age_band}/{event_year}")
        except Exception as e:
            logger.error(f"✗ Error saving FpGrowth outputs for {cohort_name}/{age_band}/{event_year}: {e}")

        return df

    except Exception as e:
        logger.error(f"Error in feature_engineer: {e}")
        return None


def process_cohort_feature_engineer(cohort_name, age_band, event_year, paths, logger, bucket_name="pgxdatalake"):
                        
    try:
        cohort_path = get_cohort_paths(cohort_name, age_band, event_year, bucket_name)
        if not cohort_path:
            logger.warning(f"\u2717 Unable to locate cohort path for {cohort_name}/{age_band}/{event_year}")
            return

        logger.info(f"\u2192 Found cohort path: {cohort_path}")

        needs_info = check_cohort_needs_processing(cohort_path, bucket_name, logger)
        if not needs_info:
            logger.warning(f"\u2717 Could not determine if cohort needs processing: {cohort_path}")
            return

        if not needs_info["needs_processing"]:
            logger.info(f"\u2713 Skipping {cohort_name}/{age_band}/{event_year} - already processed")
            return

        df = feature_engineer(cohort_name, age_band, event_year, paths, logger, bucket_name)
        if df is not None:
            logger.info(f"\u2713 Successfully processed {cohort_name}/{age_band}/{event_year}")
        else:
            logger.warning(f"\u2192 No data to process for {cohort_name}/{age_band}/{event_year}")

    except Exception as e:
        logger.error(f"\u2717 Error in process_cohort: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run FpGrowth feature engineering on cohort data")
    parser.add_argument("--cohort_name", type=str, help="Cohort type (e.g., ed_non_opioid)")
    parser.add_argument("--age-band", type=str, help="Age band (e.g., 45-54)")
    parser.add_argument("--event-year", type=str, help="Event year (e.g., 2020)")
    parser.add_argument("--path", type=str, help="Full S3 path to cohort.parquet")
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--support", type=float, default=0.01)
    parser.add_argument("--top-k", type=int, default=25)

    args = parser.parse_args()

    # Resolve metadata
    if args.path:
        params = parse_path_params(args.path)
        if not params:
            print("✗ Unable to parse cohort metadata from provided --path")
            sys.exit(1)
        cohort_name = params["cohort_name"]
        age_band = params["age_band"]
        event_year = params["event_year"]
    elif args.cohort_name and args.age_band and args.event_year:
        cohort_name = args.cohort_name
        age_band = args.age_band
        event_year = args.event_year
    else:
        print("✗ Must provide either --path or all of --cohort_name, --age-band, and --event-year")
        sys.exit(1)

    # Only now is it safe to set up logger
    logger = get_logger("cohort_feature_engineer", age_band, event_year)
    logger.info(f"→ Processing: {age_band} | {event_year}")

    paths = get_output_paths(cohort_name, age_band, event_year, bucket_name="pgxdatalake")

    process_cohort_feature_engineer(
        cohort_name=cohort_name,
        age_band=age_band,
        event_year=event_year,
        paths=paths,
        logger=logger,
        bucket_name="pgxdatalake"
    )
