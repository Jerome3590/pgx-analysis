#!/usr/bin/env python3
"""
Build final training feature table for a given cohort and age band.

This script merges, for a specified `(cohort_name, age_band)`:
- Base target patient list from `model_data`
- PGx pharmacogenomics features (allele frequencies, drug-gene mappings)

NOTE: BupaR, FP-Growth, and DTW features are NOT included to avoid target leakage.
These features are used only for visualization/dashboard purposes, not for model training.

Outputs a patient-level CSV and Parquet (one row per `mi_person_key`) under:
  `6_final_model/outputs/{cohort_name}/{age_band_fname}/{cohort_name}_{age_band_fname}_train_final_features.csv`
  `6_final_model/outputs/{cohort_name}/{age_band_fname}/inputs/model_train/final_features.parquet`
"""

import argparse
from pathlib import Path

import duckdb
import pandas as pd


def build_final_features(project_root: Path, cohort_name: str, age_band: str) -> None:
    """
    Build final features for a specific cohort and age band.

    Parameters
    ----------
    project_root : Path
        Project root directory.
    cohort_name : str
        Cohort identifier, e.g. "opioid_ed" or "non_opioid_ed".
    age_band : str
        Age band string, e.g. "0-12", "13-24", "65-74".
    """
    age_band_fname = age_band.replace("-", "_")

    # ------------------------------------------------------------------
    # Source 1: Base target patient list from model_data
    # ------------------------------------------------------------------
    model_data_path = (
        project_root
        / "model_data"
        / f"cohort_name={cohort_name}"
        / f"age_band={age_band}"
        / "model_events.parquet"
    )

    if not model_data_path.exists():
        raise FileNotFoundError(f"model_data parquet not found: {model_data_path}")

    con = duckdb.connect()
    # Get both target and control patients
    # Convert Path to string with forward slashes for cross-platform compatibility (Windows/Linux)
    model_data_path_str = str(model_data_path).replace('\\', '/')
    base_df = con.execute(
        f"""
        SELECT DISTINCT mi_person_key, target
        FROM read_parquet('{model_data_path_str}')
        WHERE target IN (0, 1)
        """
    ).df()
    con.close()

    # Ensure mi_person_key is string type
    base_df['mi_person_key'] = base_df['mi_person_key'].astype(str)
    
    n_target = len(base_df[base_df['target'] == 1])
    n_control = len(base_df[base_df['target'] == 0])
    print(f"[INFO] Loaded {n_target} target patients and {n_control} control patients from {model_data_path}")
    print(f"[INFO] Total: {len(base_df)} patients")
    print(f"[INFO] NOTE: BupaR, FP-Growth, and DTW features are excluded to avoid target leakage.")
    print(f"[INFO] These features are used only for visualization/dashboard purposes.")

    # ------------------------------------------------------------------
    # Source 2: Item features (CPT, ICD, Drug Name binary indicators)
    # ------------------------------------------------------------------
    # Load aggregated feature importance to get list of important codes/drugs
    fi_csv = (
        project_root
        / "3_feature_importance"
        / "outputs"
        / cohort_name
        / age_band
        / f"{cohort_name}_{age_band_fname}_aggregated_feature_importance.csv"
    )
    
    # Fallback to alternative location
    if not fi_csv.exists():
        fi_csv = (
            project_root
            / "3_feature_importance"
            / "outputs"
            / cohort_name
            / f"{cohort_name}_{age_band_fname}_aggregated_feature_importance.csv"
        )
    
    # Try downloading from S3 if not found locally
    if not fi_csv.exists():
        try:
            import boto3
            from botocore.exceptions import ClientError
            
            s3_client = boto3.client("s3")
            bucket = "pgxdatalake"
            s3_key = f"gold/feature_importance/{cohort_name}/{age_band}/{cohort_name}_{age_band_fname}_aggregated_feature_importance.csv"
            
            print(f"[INFO] Feature importance CSV not found locally. Downloading from S3: s3://{bucket}/{s3_key}")
            
            fi_csv = (
                project_root
                / "3_feature_importance"
                / "outputs"
                / cohort_name
                / age_band
                / f"{cohort_name}_{age_band_fname}_aggregated_feature_importance.csv"
            )
            fi_csv.parent.mkdir(parents=True, exist_ok=True)
            
            s3_client.download_file(bucket, s3_key, str(fi_csv))
            print(f"[OK] Downloaded feature importance CSV to {fi_csv}")
        except (ImportError, ClientError, Exception) as e:
            print(f"[WARNING] Feature importance CSV not found locally and S3 download failed: {e}")
            fi_csv = None
    
    item_features_df = None
    if fi_csv and fi_csv.exists():
        # Load feature importance to get list of important codes/drugs
        fi_df = pd.read_csv(fi_csv)
        important_features = fi_df['feature'].tolist()
        
        # Filter to item_* features only
        important_items = [f.replace('item_', '') for f in important_features if f.startswith('item_')]
        
        print(f"[INFO] Creating binary indicators for {len(important_items)} important codes/drugs from feature importance")
        
        # Create binary indicators for each important code/drug
        # Use a more efficient approach: load all events and create features in pandas
        con = duckdb.connect()
        
        # Load all relevant columns from model_data
        # Convert Path to string with forward slashes for cross-platform compatibility (Windows/Linux)
        model_data_path_str = str(model_data_path).replace('\\', '/')
        events_df = con.execute(
            f"""
            SELECT 
                CAST(mi_person_key AS VARCHAR) AS mi_person_key,
                procedure_code,
                cpt_mod_1_code,
                cpt_mod_2_code,
                primary_icd_diagnosis_code,
                two_icd_diagnosis_code,
                three_icd_diagnosis_code,
                four_icd_diagnosis_code,
                five_icd_diagnosis_code,
                six_icd_diagnosis_code,
                seven_icd_diagnosis_code,
                eight_icd_diagnosis_code,
                nine_icd_diagnosis_code,
                ten_icd_diagnosis_code,
                two_icd_procedure_code,
                three_icd_procedure_code,
                four_icd_procedure_code,
                five_icd_procedure_code,
                six_icd_procedure_code,
                seven_icd_procedure_code,
                eight_icd_procedure_code,
                nine_icd_procedure_code,
                ten_icd_procedure_code,
                drug_name
            FROM read_parquet('{model_data_path_str}')
            """
        ).df()
        con.close()
        
        # Create a set of all codes/drugs from all columns
        all_codes = set()
        code_columns = [
            'procedure_code', 'cpt_mod_1_code', 'cpt_mod_2_code',
            'primary_icd_diagnosis_code', 'two_icd_diagnosis_code', 'three_icd_diagnosis_code',
            'four_icd_diagnosis_code', 'five_icd_diagnosis_code', 'six_icd_diagnosis_code',
            'seven_icd_diagnosis_code', 'eight_icd_diagnosis_code', 'nine_icd_diagnosis_code',
            'ten_icd_diagnosis_code', 'two_icd_procedure_code', 'three_icd_procedure_code',
            'four_icd_procedure_code', 'five_icd_procedure_code', 'six_icd_procedure_code',
            'seven_icd_procedure_code', 'eight_icd_procedure_code', 'nine_icd_procedure_code',
            'ten_icd_procedure_code', 'drug_name'
        ]
        
        for col in code_columns:
            if col in events_df.columns:
                all_codes.update(events_df[col].dropna().unique())
        
        # Create binary indicators for each important item (more efficient: build all at once)
        item_feature_dict = {}
        
        for item in important_items:
            item_feature_name = f"item_{item}"
            
            # Check if patient has this code/drug in any column
            mask = pd.Series(False, index=events_df.index)
            for col in code_columns:
                if col in events_df.columns:
                    mask |= (events_df[col] == item)
            
            # Get patients who have this code/drug
            patients_with_item = set(events_df.loc[mask, 'mi_person_key'].unique())
            
            # Store binary indicator for later concatenation
            item_feature_dict[item_feature_name] = base_df['mi_person_key'].isin(patients_with_item).astype(int)
        
        # Create DataFrame from all item features at once (avoids fragmentation)
        if item_feature_dict:
            item_features_df = pd.DataFrame(item_feature_dict)
            item_features_df.insert(0, 'mi_person_key', base_df['mi_person_key'].values)
            n_item_features = len(item_feature_dict)
            print(f"[INFO] Created {n_item_features} item_* binary features")
        else:
            item_features_df = None
    else:
        print(f"[WARNING] Feature importance CSV not found. Skipping item_* feature creation.")
        print(f"[WARNING] Expected location: {fi_csv}")

    # ------------------------------------------------------------------
    # Source 3: PGx features (REQUIRED - no target leakage)
    # ------------------------------------------------------------------
    # Try multiple possible paths (current structure and legacy)
    pgx_csv = (
        project_root
        / "5_pgx_analysis"
        / "outputs"
        / "feature_engineering"
        / f"pgx_added_features_{cohort_name}_{age_band_fname}.csv"
    )

    # Fallback to 5c_pgx_analysis if 5_pgx_analysis doesn't exist
    if not pgx_csv.exists():
        pgx_csv = (
            project_root
            / "5c_pgx_analysis"
            / "outputs"
            / "feature_engineering"
            / f"pgx_added_features_{cohort_name}_{age_band_fname}.csv"
        )

    # Fallback to legacy 7_pgx_analysis path
    if not pgx_csv.exists():
        pgx_csv = (
            project_root
            / "7_pgx_analysis"
            / "outputs"
            / "feature_engineering"
            / f"pgx_added_features_{cohort_name}_{age_band_fname}.csv"
        )

    pgx_df = None
    if pgx_csv.exists():
        pgx_df = pd.read_csv(pgx_csv)
        # Ensure mi_person_key is string type
        pgx_df['mi_person_key'] = pgx_df['mi_person_key'].astype(str)
        print(f"[INFO] Loaded PGx features for {len(pgx_df)} patients ({len(pgx_df.columns) - 1} features)")
    else:
        # Try downloading from S3 if not found locally
        try:
            import boto3
            from botocore.exceptions import ClientError

            s3_client = boto3.client("s3")
            bucket = "pgxdatalake"
            s3_key = f"gold/pgx_features/{cohort_name}/{age_band}/pgx_added_features_{cohort_name}_{age_band_fname}.csv"

            print(f"[INFO] PGx features not found locally. Downloading from S3: s3://{bucket}/{s3_key}")

            # Download to 5_pgx_analysis path (standard location)
            pgx_csv = (
                project_root
                / "5_pgx_analysis"
                / "outputs"
                / "feature_engineering"
                / f"pgx_added_features_{cohort_name}_{age_band_fname}.csv"
            )
            pgx_csv.parent.mkdir(parents=True, exist_ok=True)

            s3_client.download_file(bucket, s3_key, str(pgx_csv))
            print(f"[OK] Downloaded PGx features to {pgx_csv}")

            pgx_df = pd.read_csv(pgx_csv)
            pgx_df['mi_person_key'] = pgx_df['mi_person_key'].astype(str)
            print(f"[INFO] Loaded PGx features for {len(pgx_df)} patients ({len(pgx_df.columns) - 1} features)")
        except (ImportError, ClientError, Exception) as e:
            print(f"[ERROR] PGx features not found locally and S3 download failed: {e}")
            print("[ERROR] Expected locations:")
            pgx_path_1 = project_root / "5_pgx_analysis" / "outputs" / "feature_engineering" / f"pgx_added_features_{cohort_name}_{age_band_fname}.csv"
            pgx_path_2 = project_root / "5c_pgx_analysis" / "outputs" / "feature_engineering" / f"pgx_added_features_{cohort_name}_{age_band_fname}.csv"
            s3_path = f"s3://pgxdatalake/gold/pgx_features/{cohort_name}/{age_band}/pgx_added_features_{cohort_name}_{age_band_fname}.csv"
            print(f"  - {pgx_path_1}")
            print(f"  - {pgx_path_2}")
            print(f"  - {s3_path}")
            raise FileNotFoundError(
                f"PGx features are required but not found. Checked local paths and S3: {s3_path}"
            )
    
    # PGx features are required
    if pgx_df is None:
        raise FileNotFoundError(
            f"PGx features are required but could not be loaded. "
            f"Please ensure PGx features exist locally or in S3."
        )

    # ------------------------------------------------------------------
    # Merge features on mi_person_key (item_* + PGx + base_df)
    # ------------------------------------------------------------------
    # Ensure base_df mi_person_key is string type
    base_df['mi_person_key'] = base_df['mi_person_key'].astype(str)
    
    # Start with base_df
    merged = base_df.copy()
    
    # Add item_* features if available
    if item_features_df is not None:
        item_features_df['mi_person_key'] = item_features_df['mi_person_key'].astype(str)
        merged = merged.merge(item_features_df, on="mi_person_key", how="left")
        # Fill NaN with 0 for item_* features (patient doesn't have the code/drug)
        item_cols = [c for c in item_features_df.columns if c.startswith('item_')]
        for col in item_cols:
            if col in merged.columns:
                merged[col] = merged[col].fillna(0).astype(int)
    
    # Add PGx features
    # Ensure PGx dataframe has mi_person_key as string
    # Also drop 'target' column from PGx dataframe (keep only from base_df)
    pgx_df['mi_person_key'] = pgx_df['mi_person_key'].astype(str)
    if 'target' in pgx_df.columns:
        pgx_df = pgx_df.drop(columns=['target'])
    
    merged = merged.merge(pgx_df, on="mi_person_key", how="left")
    
    # Clean up any duplicate target columns (from merges with suffixes)
    if 'target_x' in merged.columns:
        # Keep target_x if it exists, rename to target
        if 'target' in merged.columns:
            merged = merged.drop(columns=['target'])
        merged = merged.rename(columns={'target_x': 'target'})
    elif 'target_y' in merged.columns:
        # Keep target_y if it exists, rename to target
        if 'target' in merged.columns:
            merged = merged.drop(columns=['target'])
        merged = merged.rename(columns={'target_y': 'target'})
    
    # Ensure target column exists (should be from base_df)
    if 'target' not in merged.columns:
        print("[WARNING] Target column missing after merge, adding from base_df")
        merged = merged.merge(base_df[['mi_person_key', 'target']], on="mi_person_key", how="left")

    out_dir = project_root / "6_final_model" / "outputs" / cohort_name / age_band_fname
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path_csv = out_dir / f"{cohort_name}_{age_band_fname}_train_final_features.csv"
    out_path_parquet = out_dir / "inputs" / "model_train" / "final_features.parquet"

    # Save CSV (for backward compatibility)
    print(f"[INFO] Writing final feature table to CSV: {out_path_csv} ({len(merged)} rows, {len(merged.columns)} columns)")
    merged.to_csv(out_path_csv, index=False)
    
    # Save Parquet (preferred format for downstream steps)
    print(f"[INFO] Writing final feature table to Parquet: {out_path_parquet}")
    out_path_parquet.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(out_path_parquet, index=False, compression='snappy', engine='pyarrow')
    print("[INFO] Done.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build final patient-level feature table for a cohort/age_band, "
            "combining model_data targets with PGx features only. "
            "BupaR, FP-Growth, and DTW features are excluded to avoid target leakage."
        )
    )
    parser.add_argument(
        "--project-root",
        type=str,
        default=".",
        help="Project root path (default: current directory)",
    )
    parser.add_argument(
        "--cohort-name",
        type=str,
        default="opioid_ed",
        help="Cohort name (e.g. opioid_ed, non_opioid_ed). Default: opioid_ed",
    )
    parser.add_argument(
        "--age-band",
        type=str,
        default="0-12",
        help="Age band string, e.g. '0-12', '13-24', '65-74'. Default: 0-12",
    )
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    build_final_features(project_root, cohort_name=args.cohort_name, age_band=args.age_band)


if __name__ == "__main__":
    main()


