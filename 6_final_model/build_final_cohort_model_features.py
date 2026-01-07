#!/usr/bin/env python3
"""
Build final training feature table for a given cohort and age band.

This script merges, for a specified `(cohort_name, age_band)`:
- Base target patient list from `model_data`
- FP-Growth features (itemsets, rules, metrics)
- BupaR sequence and time-to-event features
- DTW trajectory features (prototype DTW distances)
- PGx pharmacogenomics features (allele frequencies, drug-gene mappings)

Outputs a patient-level CSV and Parquet (one row per `mi_person_key`) under:
  `6_final_model/outputs/{cohort_name}/{age_band_fname}/{cohort_name}_{age_band_fname}_train_final_features.csv`
  `6_final_model/outputs/{cohort_name}/{age_band_fname}/inputs/model_train/final_features.parquet`

Currently supported cohorts:
- `opioid_ed`  – expects BupaR files with F1120 naming (`pre_f1120`, `post_f1120`, `time_to_f1120`)
- `non_opioid_ed` – expects BupaR files with HCG naming (`pre_hcg`, `time_to_hcg`)
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
    base_df = con.execute(
        f"""
        SELECT DISTINCT mi_person_key, target
        FROM read_parquet('{model_data_path}')
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

    # ------------------------------------------------------------------
    # Source 2: BupaR patient features (pre/post + time-to-event)
    # ------------------------------------------------------------------
    bupar_root = (
        project_root
        / "5a_bupaR_analysis"
        / "outputs"
        / cohort_name
        / age_band_fname
        / "features"
    )

    # Filenames depend on cohort
    if cohort_name == "opioid_ed":
        pre_bupar_csv = (
            bupar_root
            / f"{cohort_name}_{age_band_fname}_train_target_pre_f1120_patient_features_bupar.csv"
        )
        post_bupar_csv = (
            bupar_root
            / f"{cohort_name}_{age_band_fname}_train_target_post_f1120_patient_features_bupar.csv"
        )
        time_to_bupar_csv = (
            bupar_root
            / f"{cohort_name}_{age_band_fname}_train_target_time_to_f1120_features_bupar.csv"
        )
    elif cohort_name == "non_opioid_ed":
        pre_bupar_csv = (
            bupar_root
            / f"{cohort_name}_{age_band_fname}_train_target_pre_hcg_patient_features_bupar.csv"
        )
        # Polypharmacy pipeline does not define post-HCG features (descriptive only)
        post_bupar_csv = None
        time_to_bupar_csv = (
            bupar_root
            / f"{cohort_name}_{age_band_fname}_train_target_time_to_hcg_features_bupar.csv"
        )
    else:
        raise ValueError(
            f"Unsupported cohort_name for BupaR feature merging: {cohort_name}"
        )

    if not pre_bupar_csv.exists():
        raise FileNotFoundError(f"Pre-target BupaR features not found: {pre_bupar_csv}")
    if time_to_bupar_csv is None or not time_to_bupar_csv.exists():
        raise FileNotFoundError(
            f"Time-to-event BupaR features not found: {time_to_bupar_csv}"
        )

    pre_df = pd.read_csv(pre_bupar_csv)
    time_to_df = pd.read_csv(time_to_bupar_csv)
    post_df = None

    if post_bupar_csv is not None and post_bupar_csv.exists():
        post_df = pd.read_csv(post_bupar_csv)
    elif cohort_name == "opioid_ed":
        # For opioid_ed we expect post-F1120; for safety, require it
        raise FileNotFoundError(f"Post-target BupaR features not found: {post_bupar_csv}")

    # In BupaR outputs, the ID column may be case_id or mi_person_key
    # Use feature engineering files which should have mi_person_key
    bupar_features_csv = (
        project_root
        / "5a_bupaR_analysis"
        / "outputs"
        / "feature_engineering"
        / f"bupaR_added_features_{cohort_name}_{age_band_fname}.csv"
    )
    
    if bupar_features_csv.exists():
        # Use consolidated BupaR features file if available
        bupar_features_df = pd.read_csv(bupar_features_csv)
        bupar_features_df['mi_person_key'] = bupar_features_df['mi_person_key'].astype(str)
        print(f"[INFO] Using consolidated BupaR features file ({len(bupar_features_df.columns) - 1} features)")
        # Replace individual dataframes with consolidated one
        pre_df = bupar_features_df.copy()
        post_df = None  # Already included in consolidated file
        time_to_df = pd.DataFrame(columns=['mi_person_key'])  # Placeholder
    else:
        # Fallback to individual BupaR files
        if "case_id" in pre_df.columns:
            pre_df = pre_df.rename(columns={"case_id": "mi_person_key"})
        if post_df is not None and "case_id" in post_df.columns:
            post_df = post_df.rename(columns={"case_id": "mi_person_key"})
        if "case_id" in time_to_df.columns:
            time_to_df = time_to_df.rename(columns={"case_id": "mi_person_key"})

    msg = (
        f"[INFO] Loaded BupaR pre-target features for {len(pre_df)} patients, "
        f"time-to-event features for {len(time_to_df)} patients"
    )
    if post_df is not None:
        msg += f", post-target features for {len(post_df)} patients"
    print(msg)

    # ------------------------------------------------------------------
    # Source 3: DTW trajectory features (prototype distances)
    # ------------------------------------------------------------------
    # Try feature engineering directory first (new structure)
    dtw_csv = (
        project_root
        / "6_dtw_analysis"
        / "outputs"
        / "feature_engineering"
        / f"dtw_added_features_{cohort_name}_{age_band_fname}.csv"
    )
    
    # Fallback to old structure if new doesn't exist
    if not dtw_csv.exists():
        dtw_root = (
            project_root
            / "6_dtw_analysis"
            / "outputs"
            / cohort_name
            / age_band_fname
            / "features"
        )
        dtw_csv = dtw_root / f"{cohort_name}_{age_band_fname}_train_target_dtw_features.csv"

    if not dtw_csv.exists():
        raise FileNotFoundError(f"DTW features not found: {dtw_csv}")

    dtw_df = pd.read_csv(dtw_csv)
    # Ensure mi_person_key column exists (may be case_id in old format)
    if "case_id" in dtw_df.columns and "mi_person_key" not in dtw_df.columns:
        dtw_df = dtw_df.rename(columns={"case_id": "mi_person_key"})
    print(f"[INFO] Loaded DTW features for {len(dtw_df)} patients ({len(dtw_df.columns) - 1} features)")

    # ------------------------------------------------------------------
    # Source 4: FP-Growth features
    # ------------------------------------------------------------------
    fpgrowth_csv = (
        project_root
        / "4_fpgrowth_analysis"
        / "outputs"
        / "feature_engineering"
        / f"fpgrowth_added_features_{cohort_name}_{age_band_fname}.csv"
    )

    fpgrowth_df = None
    if fpgrowth_csv.exists():
        fpgrowth_df = pd.read_csv(fpgrowth_csv)
        # Ensure mi_person_key is string type
        fpgrowth_df['mi_person_key'] = fpgrowth_df['mi_person_key'].astype(str)
        print(f"[INFO] Loaded FP-Growth features for {len(fpgrowth_df)} patients ({len(fpgrowth_df.columns) - 1} features)")
    else:
        print(f"[WARNING] FP-Growth features not found: {fpgrowth_csv}")

    # ------------------------------------------------------------------
    # Source 5: PGx features
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
            print(f"✓ Downloaded PGx features to {pgx_csv}")

            pgx_df = pd.read_csv(pgx_csv)
            pgx_df['mi_person_key'] = pgx_df['mi_person_key'].astype(str)
            print(f"[INFO] Loaded PGx features for {len(pgx_df)} patients ({len(pgx_df.columns) - 1} features)")
        except (ImportError, ClientError, Exception) as e:
            print(f"[WARNING] PGx features not found locally and S3 download failed: {e}")
            print("[WARNING] Expected locations:")
            pgx_path_1 = project_root / "5_pgx_analysis" / "outputs" / "feature_engineering" / f"pgx_added_features_{cohort_name}_{age_band_fname}.csv"
            pgx_path_2 = project_root / "5c_pgx_analysis" / "outputs" / "feature_engineering" / f"pgx_added_features_{cohort_name}_{age_band_fname}.csv"
            s3_path = f"s3://pgxdatalake/gold/pgx_features/{cohort_name}/{age_band}/pgx_added_features_{cohort_name}_{age_band_fname}.csv"
            print(f"  - {pgx_path_1}")
            print(f"  - {pgx_path_2}")
            print(f"  - {s3_path}")

    # ------------------------------------------------------------------
    # Merge all features on mi_person_key
    # ------------------------------------------------------------------
    # Ensure base_df mi_person_key is string type
    base_df['mi_person_key'] = base_df['mi_person_key'].astype(str)
    
    # Ensure all other dataframes have mi_person_key as string
    # Also drop 'target' column from feature dataframes (keep only from base_df)
    if pre_df is not None:
        pre_df['mi_person_key'] = pre_df['mi_person_key'].astype(str)
        if 'target' in pre_df.columns:
            pre_df = pre_df.drop(columns=['target'])
    if post_df is not None:
        post_df['mi_person_key'] = post_df['mi_person_key'].astype(str)
        if 'target' in post_df.columns:
            post_df = post_df.drop(columns=['target'])
    if time_to_df is not None:
        time_to_df['mi_person_key'] = time_to_df['mi_person_key'].astype(str)
        if 'target' in time_to_df.columns:
            time_to_df = time_to_df.drop(columns=['target'])
    if dtw_df is not None:
        dtw_df['mi_person_key'] = dtw_df['mi_person_key'].astype(str)
        if 'target' in dtw_df.columns:
            dtw_df = dtw_df.drop(columns=['target'])
    if fpgrowth_df is not None:
        fpgrowth_df['mi_person_key'] = fpgrowth_df['mi_person_key'].astype(str)
        if 'target' in fpgrowth_df.columns:
            fpgrowth_df = fpgrowth_df.drop(columns=['target'])
    if pgx_df is not None:
        pgx_df['mi_person_key'] = pgx_df['mi_person_key'].astype(str)
        if 'target' in pgx_df.columns:
            pgx_df = pgx_df.drop(columns=['target'])

    merged = base_df.merge(pre_df, on="mi_person_key", how="left")
    if post_df is not None:
        merged = merged.merge(post_df, on="mi_person_key", how="left", suffixes=("_pre", "_post"))
    merged = merged.merge(time_to_df, on="mi_person_key", how="left")
    merged = merged.merge(dtw_df, on="mi_person_key", how="left")
    
    # Add FP-Growth features
    if fpgrowth_df is not None:
        merged = merged.merge(fpgrowth_df, on="mi_person_key", how="left")
    
    # Add PGx features
    if pgx_df is not None:
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
            "combining model_data targets with BupaR and DTW features."
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


