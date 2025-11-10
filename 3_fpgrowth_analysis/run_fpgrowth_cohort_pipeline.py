import os
import sys
import logging
import time
import psutil
from io import BytesIO
import pandas as pd
from typing import Union, Dict
from typing import Dict, Union
from concurrent.futures import ProcessPoolExecutor, as_completed

# Project path config (match group pipeline style)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from helpers.constants import (
    AGE_BANDS, EVENT_YEARS, TOP_K, MIN_SUPPORT_THRESHOLD, MIN_CONFIDENCE_MEDIUM, TIMEOUT_SECONDS
)
from helpers.duckdb_utils import setup_duckdb_environment

from helpers.fpgrowth_utils import (
    run_fpgrowth_drug_token_with_fallback,
    convert_frozensets,
)

from helpers.s3_utils import (
    get_output_paths,
    save_to_s3_json,
    save_to_s3_parquet,
    s3_exists,
    get_cohort_parquet_path,
)

from helpers.visualization_utils import create_network_visualization


def create_cohort_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def log_usage(logger: logging.Logger, stage: str) -> None:
    memory = psutil.virtual_memory()
    cpu = psutil.cpu_percent(interval=0.1)
    logger.info(f"{stage} | Memory: {memory.percent:.1f}% | CPU: {cpu:.1f}%")


def load_curated_cohort(cohort_name: str, age_band: str, event_year: str, logger: logging.Logger) -> pd.DataFrame:
    """Load curated cohort parquet from GOLD cohorts_clean."""
    con = setup_duckdb_environment(logger)
    cohort_path = get_cohort_parquet_path(cohort_name, age_band, event_year)
    df = con.execute(f"SELECT mi_person_key, drug_name FROM read_parquet('{cohort_path}')").df()
    con.close()
    return df


def run_single_cohort(job: dict) -> tuple:
    """Process a single cohort-age-year job end-to-end and save outputs to GOLD."""
    cohort = job['cohort']
    age_band = job['age_band']
    event_year = str(job['event_year'])

    logger = create_cohort_logger(f"cohort_fpgrowth_{cohort}_{age_band}_{event_year}")

    try:
        # Allow dynamic threshold and timeout
        min_support = job.get('min_support_threshold', MIN_SUPPORT_THRESHOLD)
        timeout_seconds = job.get('timeout_seconds', TIMEOUT_SECONDS)
        t0 = time.time()
        log_usage(logger, f"Start {cohort}/{age_band}/{event_year}")
        logger.info(f"Loading curated cohort data for {cohort}/{age_band}/{event_year}")
        df = load_curated_cohort(cohort, age_band, event_year, logger)

        # Log distinct counts for diagnostics
        n_persons = df['mi_person_key'].nunique()
        n_drugs = df['drug_name'].nunique()
        logger.info(f"[DIAG] Distinct mi_person_keys: {n_persons}")
        logger.info(f"[DIAG] Distinct drug_names: {n_drugs}")

        if df.empty:
            logger.warning(f"No data for {cohort}/{age_band}/{event_year}")
            return (cohort, age_band, event_year, False, "No data")

        logger.info(f"Building transactions as tokens for {cohort}/{age_band}/{event_year}")
        grouped = (
            df.groupby("mi_person_key")["drug_name"]
              .agg(lambda rows: sorted({
                  f"drug_{str(d).strip().lower()}"
                  for d in rows if pd.notnull(d) and str(d).strip()
              }))
              .reset_index()
              .rename(columns={"drug_name": "drug_tokens"})
        )

        logger.info(f"Getting output paths for {cohort}/{age_band}/{event_year}")
        paths = get_output_paths(cohort.lower(), age_band, event_year)

        logger.info(f"Running FP-Growth for {cohort}/{age_band}/{event_year}")

        # Raise thresholds to reduce rule count and increase significance
        _, itemsets, rules = run_fpgrowth_drug_token_with_fallback(
            grouped, cohort.lower(), age_band, event_year, paths, logger,
            support_start=min_support, TOP_K=TOP_K, min_confidence=MIN_CONFIDENCE_MEDIUM, timeout=timeout_seconds
        )

        logger.info(f"FP-Growth: {len(itemsets)} itemsets generated for {cohort}/{age_band}/{event_year}")
        logger.info(f"FP-Growth: {len(rules)} rules generated for {cohort}/{age_band}/{event_year}")
        if rules.empty:
            logger.warning(f"FP-Growth: No rules generated for {cohort}/{age_band}/{event_year}")

        logger.info(f"Saving itemsets and rules JSON for {cohort}/{age_band}/{event_year}")
        itemsets_json = convert_frozensets(itemsets.to_dict(orient="records"), logger)
        rules_json = convert_frozensets(rules.to_dict(orient="records"), logger)
        save_to_s3_json(itemsets_json, paths["itemsets_json"], logger)
        save_to_s3_json(rules_json, paths["rules_json"], logger)

        logger.info(f"Loading additional data and building features for {cohort}/{age_band}/{event_year}")
        # Cohort-specific patterns only; skipping global pattern labeling

        try:
            logger.info(f"Building cohort-specific drug encoding and FP-Growth metrics for {cohort}/{age_band}/{event_year}")
            unique_drugs = sorted(set(d for d in df['drug_name'].dropna().astype(str)))
            support_map = {drug: 0.0 for drug in unique_drugs}
            confidence_vals = {drug: [] for drug in unique_drugs}
            certainty_vals = {drug: [] for drug in unique_drugs}

            for _, row in itemsets.iterrows():
                items = row.get('itemsets', [])
                sup = float(row.get('support', 0.0)) if row.get('support', None) is not None else 0.0
                for tok in items if isinstance(items, (list, set, tuple)) else []:
                    if isinstance(tok, str) and tok.startswith('drug_'):
                        dn = tok.removeprefix('drug_')
                        if dn in support_map:
                            support_map[dn] = max(support_map[dn], sup)

            for _, row in rules.iterrows():
                conf = float(row.get('confidence', 0.0)) if row.get('confidence', None) is not None else 0.0
                cert = float(row.get('certainty', 0.0)) if row.get('certainty', None) is not None else 0.0
                for col in ('antecedents', 'consequents'):
                    vals = row.get(col, [])
                    if isinstance(vals, (list, set, tuple)):
                        for tok in vals:
                            if isinstance(tok, str) and tok.startswith('drug_'):
                                dn = tok.removeprefix('drug_')
                                if dn in confidence_vals:
                                    confidence_vals[dn].append(conf)
                                if dn in certainty_vals:
                                    certainty_vals[dn].append(cert)

            import glob, json
            # Build cohort-specific drug metrics from scratch (do not use global CSV)
            logger.info(f"Building cohort-specific drug metrics for {cohort}/{age_band}/{event_year}")
            drug_metrics = []
            # Use itemsets and rules to build metrics for each drug in this cohort
            for _, item in itemsets.iterrows():
                original_itemset = item['itemsets']
                drugs = list(original_itemset)
                for drug in drugs:
                    drug_record = {
                        'drug_name': drug,
                        'support': item['support'],
                        'itemsets': original_itemset,
                        'frequency': len(drugs)
                    }
                    drug_metrics.append(drug_record)
            for _, rule in rules.iterrows():
                original_antecedents = rule['antecedents']
                drugs = list(original_antecedents)
                for drug in drugs:
                    existing_record = next((r for r in drug_metrics if r['drug_name'] == drug), None)
                    if existing_record:
                        existing_record.update({
                            'confidence': rule['confidence'],
                            'lift': rule['lift'],
                            'antecedents': rule['antecedents'],
                            'consequents': rule['consequents'],
                            'certainty': rule['certainty']
                        })
                    else:
                        drug_record = {
                            'drug_name': drug,
                            'confidence': rule['confidence'],
                            'lift': rule['lift'],
                            'antecedents': rule['antecedents'],
                            'consequents': rule['consequents'],
                            'certainty': rule['certainty']
                        }
                        drug_metrics.append(drug_record)
            # Convert to DataFrame for downstream merging
            drug_metrics_df = pd.DataFrame(drug_metrics)

            # Skipping loading of global drug patterns JSON; cohort-specific patterns only

            logger.info(f"Building rule counts and trend placeholders for {cohort}/{age_band}/{event_year}")
            num_rules_map = {drug: 0 for drug in unique_drugs}
            num_drugs_in_rules_map = {drug: 0 for drug in unique_drugs}
            for _, row in rules.iterrows():
                drugs_in_rule = set()
                for col in ("antecedents", "consequents"):
                    vals = row.get(col, [])
                    if isinstance(vals, (list, set, tuple)):
                        for tok in vals:
                            if isinstance(tok, str) and tok.startswith("drug_"):
                                dn = tok.removeprefix("drug_")
                                drugs_in_rule.add(dn)
                for dn in drugs_in_rule:
                    if dn in num_rules_map:
                        num_rules_map[dn] += 1
                        num_drugs_in_rules_map[dn] += len(drugs_in_rule)

            trend_map = {drug: None for drug in unique_drugs}
            trend_dir_map = {drug: None for drug in unique_drugs}

            # 5. Merge all info for output (no global encoding dependency)
            records = []
            for i, drug in enumerate(unique_drugs):
                rec = {"drug_name": drug, "cohort_encoded_id": i}
                # Add linguistics if available
                if not drug_metrics_df.empty:
                    row = drug_metrics_df[drug_metrics_df["drug_name"] == drug]
                    if not row.empty:
                        for col in row.columns:
                            if col != "drug_name":
                                rec[col] = row.iloc[0][col]
                # Add cohort-specific FP-Growth metrics
                rec["support"] = support_map.get(drug, 0.0)
                rec["num_rules"] = num_rules_map.get(drug, 0)
                rec["num_drugs_in_rules"] = num_drugs_in_rules_map.get(drug, 0)
                rec["trend"] = trend_map.get(drug)
                rec["trend_direction"] = trend_dir_map.get(drug)
                records.append(rec)

            encoding_df = pd.DataFrame(records)
            save_to_s3_parquet(encoding_df, paths['drug_encoding_parquet'], logger)

        except Exception as enc_e:
            logger.warning(f"Failed to build/save cohort encoding map: {enc_e}")

        # Network visualization (log success, no HTML output)
        try:
            if not rules.empty:
                if "certainty" not in rules.columns:
                    rules["certainty"] = rules["confidence"] if "confidence" in rules.columns else 0.0
                create_network_visualization(
                    rules_df=rules,
                    title=f"{cohort} {age_band} {event_year} Drug Network",
                    cohort_name=cohort.lower(),
                    age_band=age_band,
                    event_year=event_year,
                    itemsets_counts=None,
                    logger=logger,
                )
                logger.info(f"Network visualization created and saved to S3 for {cohort}/{age_band}/{event_year}")
        except Exception as viz_e:
            logger.warning(f"Network visualization failed for {cohort}/{age_band}/{event_year}: {viz_e}")

        log_usage(logger, f"End {cohort}/{age_band}/{event_year}")
        logger.info(f"✓ Completed {cohort}/{age_band}/{event_year} in {time.time() - t0:.2f}s")
        return (cohort, age_band, event_year, True, "Success")
    except Exception as e:
        logger.error(f"✗ {cohort}/{age_band}/{event_year} failed: {e}")
        log_usage(logger, f"End (failed) {cohort}/{age_band}/{event_year}")
        return (cohort, age_band, event_year, False, str(e))


def execute_cohort_fpgrowth_pipeline(cohort=None) -> None:
    logger = create_cohort_logger("cohort_parallel_fpgrowth")

    # Accept single cohort, list, or default
    if cohort is None:
        target_cohorts = ["opioid_ed", "non_opioid_ed"]
    elif isinstance(cohort, str):
        target_cohorts = [cohort]
    else:
        target_cohorts = list(cohort)


    # Check S3 for already processed jobs (rules_json as marker)
    jobs = []
    for c in target_cohorts:
        for age_band in AGE_BANDS:
            for year in EVENT_YEARS:
                paths = get_output_paths(c, age_band, year)
                if not s3_exists(paths["rules_json"]):
                    job = {"cohort": c, "age_band": age_band, "event_year": year}
                    if hasattr(execute_cohort_fpgrowth_pipeline, 'min_support_threshold') and execute_cohort_fpgrowth_pipeline.min_support_threshold is not None:
                        job['min_support_threshold'] = execute_cohort_fpgrowth_pipeline.min_support_threshold
                    if hasattr(execute_cohort_fpgrowth_pipeline, 'timeout_seconds') and execute_cohort_fpgrowth_pipeline.timeout_seconds is not None:
                        job['timeout_seconds'] = execute_cohort_fpgrowth_pipeline.timeout_seconds
                    jobs.append(job)

    logger.info(f"🚀 Submitting {len(jobs)} missing cohort jobs with process pools")

    max_retries = 2
    attempt = 0
    failed = jobs
    # Only retry the specific cohort/age_band/year jobs that failed in the previous attempt
    while failed and attempt <= max_retries:
        logger.info(f"[RETRY] Attempt {attempt+1} for {len(failed)} jobs")
        results: list[tuple] = []
        with ProcessPoolExecutor(max_workers=32) as executor:
            future_to_job = {executor.submit(run_single_cohort, job): job for job in failed}
            for future in as_completed(future_to_job):
                job = future_to_job[future]
                logger.info(f"[POOL] Starting new job: {job['cohort']}/{job['age_band']}/{job['event_year']} after previous job completed.")
                try:
                    results.append(future.result(timeout=1800))
                except Exception as err:
                    logger.error(f"Job failed {job}: {err}")
                    results.append((job["cohort"], job["age_band"], job["event_year"], False, str(err)))

        succeeded = sum(1 for r in results if r[3])
        # Only retry jobs that failed in this attempt
        failed = [
            {"cohort": r[0], "age_band": r[1], "event_year": r[2]}
            for r in results if not r[3]
        ]
        logger.info(f"🎉 Cohort FP-Growth completed: {succeeded} / {len(results)} succeeded on attempt {attempt+1}.")
        if failed:
            logger.info("Failed jobs:")
            for f in failed[:20]:
                logger.info(f"  {f['cohort']}/{f['age_band']}/{f['event_year']}")
        attempt += 1



import argparse
parser = argparse.ArgumentParser(description="Run FP-Growth cohort pipeline.")
parser.add_argument("--cohort", type=str, nargs="?", default=None, help="Cohort name or comma-separated list of cohorts to process (default: all)")
parser.add_argument("--min_support_threshold", type=float, default=None, help="Minimum support threshold for FP-Growth fallback (overrides default)")
parser.add_argument("--timeout_seconds", type=int, default=None, help="Timeout in seconds for each FP-Growth fallback attempt (overrides default)")
args = parser.parse_args()
cohort_arg = args.cohort
min_support_arg = args.min_support_threshold
timeout_arg = args.timeout_seconds
if cohort_arg is not None and "," in cohort_arg:
    cohort_arg = [c.strip() for c in cohort_arg.split(",")]

# Attach dynamic args to function for job propagation
execute_cohort_fpgrowth_pipeline.min_support_threshold = min_support_arg
execute_cohort_fpgrowth_pipeline.timeout_seconds = timeout_arg
execute_cohort_fpgrowth_pipeline(cohort=cohort_arg)
