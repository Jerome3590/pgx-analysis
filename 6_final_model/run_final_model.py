"""
Final-model training script that merges **within-cohort** model_data and
feature-engineering tables for a given (cohort, age_band), then fits a classifier
(CPU on Linux, GPU on Windows if available).

This is intended to be a fast, reproducible analogue of the smoke-test workflow,
using locally built and downloaded artifacts:

- 4_model_data/cohort_name={cohort}/age_band={age_band}/model_events.parquet
  (cases + within-cohort controls, with an event-level `target` column)
- feature_engineering/from_s3/{4_fpgrowth,5_bupar,6_dtw,7_pgx}/{cohort}/{age_band}/*_added_features_*.csv
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Optional
import json
import os

import duckdb
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    log_loss,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

# Optuna HPO (multi-objective Recall + AUC-PR)
N_MCCV_HPO = 5
N_OPTUNA_TRIALS = 50
RANDOM_STATE = 1997

# Matplotlib for visualizations (set backend before importing pyplot)
import matplotlib
if os.environ.get('DISPLAY') is None:
    matplotlib.use('Agg')  # Use non-interactive backend on headless systems
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402

# Ensure py_helpers is on the path when running as a script
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from py_helpers.fe_monitor import (  # noqa: E402
    detect_runtime_environment,
    function_block,
    module_block,
    step_block,
    mirror_log_to_s3,
)
from py_helpers.constants import age_band_to_fname, DRUG_NAMES_EXCLUDED_MODEL_TRAINING, FEATURE_SUBSTRINGS_EXCLUDED
from py_helpers.env_utils import get_mc_cv_n_runs, get_data_root, get_model_data_root, get_workflow_python_bin, is_linux
from py_helpers.event_density_utils import (
    DENSITY_BINS as _DENSITY_BINS,
    compute_bin_thresholds as _compute_bin_thresholds,
    assign_n_event_bins as _assign_n_event_bins,
    save_thresholds as _save_thresholds,
    default_threshold_cache_path as _threshold_cache_path,
)

try:
    from py_helpers.common_imports import s3_client, S3_BUCKET  # noqa: E402
except ImportError:
    import boto3  # noqa: E402
    s3_client = boto3.client("s3")
    S3_BUCKET = "pgxdatalake"
from py_helpers.categorical_encoding import (  # noqa: E402
    encode_cpt_series,
    encode_drug_name_series,
    encode_icd_series,
)


def _load_feature_table(path: Path, required: bool = True) -> pd.DataFrame:
    """
    Load a CSV feature table if it exists; return empty DataFrame if missing
    and required=False.
    """
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Required feature file not found: {path}")
        print(f"Feature file not found (skipping): {path}")
        return pd.DataFrame()
    print(f"Loading features from {path}")
    return pd.read_csv(path)


def remove_target_leakage_features(df: pd.DataFrame, cohort: str, age_band: str) -> pd.DataFrame:
    """
    Remove target-leakage features based on naming conventions and data validation,
    matching the comprehensive logic in 6_final_model/remove_target_leakage.py.

    This drops:
      - Columns starting with 'post_'
      - Columns containing 'time_to' / 'time_to_'
      - Time-window features with suffixes like '_30d', '_90d', '_180d'
        (except those with 'interval' in the name)
      - Datetime helper columns: 'target_time', 'first_time'
      - DTW-derived features (any column with 'dtw' in its name)
      - Trajectory/sequence/itemset (defensive only; feature engineering never generates these)
      - Any feature whose name contains 'F1120'
      - Non-predictive markers/confounders (SUBOXONE, BUPRENORPHINE, F1123)
      - For non_opioid_ed cohort: ICD and CPT features (polypharmacy uses drugs only)
      - item_* features with post-target events (validated against event data)
    """
    cols = list(df.columns)
    leakage: set[str] = set()

    print(f"\n[INFO] Removing target leakage features for {cohort}/{age_band}")
    print(f"[INFO] Original dataset: {len(df)} patients, {len(df.columns)} columns")

    # 1. Post-event features (calculated AFTER target event)
    post_features = [c for c in cols if c.startswith("post_")]
    leakage.update(post_features)
    if post_features:
        print(f"\n[INFO] Post-event features (TARGET LEAKAGE): {len(post_features)}")
        for f in post_features[:10]:
            print(f"  - {f}")
        if len(post_features) > 10:
            print(f"  ... and {len(post_features) - 10} more")

    # 2. Time-to-target features (reference the target event)
    time_to_features = [
        c for c in cols if "time_to" in c.lower() or "time_to_" in c.lower()
    ]
    leakage.update(time_to_features)
    if time_to_features:
        print(f"\n[INFO] Time-to-target features (TARGET LEAKAGE): {len(time_to_features)}")
        for f in time_to_features[:10]:
            print(f"  - {f}")
        if len(time_to_features) > 10:
            print(f"  ... and {len(time_to_features) - 10} more")

    # 2b. Time-window features that reference target event (30d, 90d, 180d before target)
    # NOTE: Time intervals BETWEEN consecutive events (e.g., drug_interval_mean) are OK
    time_window_features = [
        c
        for c in cols
        if any(x in c for x in ["_30d", "_90d", "_180d"])
        and "interval" not in c.lower()
    ]
    leakage.update(time_window_features)
    if time_window_features:
        print(f"\n[INFO] Time-window features referencing target (TARGET LEAKAGE): {len(time_window_features)}")
        for f in time_window_features[:10]:
            print(f"  - {f}")
        if len(time_window_features) > 10:
            print(f"  ... and {len(time_window_features) - 10} more")

    # Note: Time interval features (between consecutive events) are KEPT - they're predictive
    interval_features = [c for c in cols if "interval" in c.lower()]
    if interval_features:
        print(f"\n[INFO] Time interval features (KEPT - predictive): {len(interval_features)}")
        for f in interval_features[:5]:
            print(f"  - {f}")
        if len(interval_features) > 5:
            print(f"  ... and {len(interval_features) - 5} more")

    # 3. Target time, first time, and cohort target-date columns (not features; must not be used for training)
    # Step 4 writes first_f1120_date / first_o11_p_date; exclude legacy names if present.
    datetime_features = [
        c for c in (
            "target_time", "first_time",
            "first_f1120_date", "first_o11_p_date",
            "first_opioid_ed_date", "first_ed_non_opioid_date",  # legacy
        )
        if c in cols
    ]
    leakage.update(datetime_features)

    # 4. DTW features (REMOVED - used for protocol filtering, not as features)
    dtw_features = [c for c in cols if "dtw" in c.lower()]
    leakage.update(dtw_features)
    if dtw_features:
        print(f"\n[INFO] DTW features found: {len(dtw_features)}")
        print("[INFO] DTW features are REMOVED - DTW is used for protocol filtering, not feature engineering")
        print("[INFO] DTW captures standard care protocols that both targets and controls follow")
        print("[INFO] Sequence information comes from BupaR, not DTW")
        for f in dtw_features[:10]:
            print(f"  - {f}")
        if len(dtw_features) > 10:
            print(f"  ... and {len(dtw_features) - 10} more")

    # 4b. Trajectory / sequence / itemset (defensive only; feature engineering never generates these—only n_events, item_*, PGx)
    traj_seq_itemset = [
        c for c in cols
        if "trajectory" in c.lower() or "sequence" in c.lower() or "itemset" in c.lower()
    ]
    leakage.update(traj_seq_itemset)
    if traj_seq_itemset:
        print(f"\n[INFO] Trajectory/sequence/itemset columns found (unexpected): {len(traj_seq_itemset)}")
        print("[INFO] Removed defensively; feature engineering does not produce these.")
        for f in traj_seq_itemset[:10]:
            print(f"  - {f}")
        if len(traj_seq_itemset) > 10:
            print(f"  ... and {len(traj_seq_itemset) - 10} more")

    # Remove initial leakage features
    safe_features = [c for c in cols if c not in leakage]
    df_clean = df[safe_features].copy()

    # Verify no F1120 in feature names (should be excluded during feature engineering)
    f1120_features = [c for c in df_clean.columns if "F1120" in c.upper()]
    if f1120_features:
        print(f"\n[WARNING] Found {len(f1120_features)} features with F1120 in name:")
        for f in f1120_features:
            print(f"  - {f}")
        print("[INFO] These should be removed - F1120 must be excluded from final model features")
        safe_features = [c for c in safe_features if c not in f1120_features]
        leakage.update(f1120_features)
        df_clean = df[safe_features].copy()

    # 5. Remove non-predictive markers/confounders
    excluded_markers = [
        "item_drug_SUBOXONE",  # Treatment medication - marker, not predictive
        "item_drug_BUPRENORPHINE_HCL",  # Treatment medication - marker, not predictive
        "item_drug_BUPRENORPHINE_HCL_NALOXON",  # Treatment medication - marker, not predictive
        "item_icd_F1123",  # Opioid dependence ICD code - marker, not predictive
    ]
    found_excluded = [c for c in df_clean.columns if c in excluded_markers]
    if found_excluded:
        print(f"\n[INFO] Removing {len(found_excluded)} non-predictive markers/confounders:")
        for f in found_excluded:
            print(f"  - {f}")
        safe_features = [c for c in safe_features if c not in found_excluded]
        leakage.update(found_excluded)
        df_clean = df[safe_features].copy()

    # 6. For non_opioid_ed cohort: remove ICD and CPT features (polypharmacy uses drugs only)
    if "non_opioid" in cohort.lower() or "ed_non_opioid" in cohort.lower():
        item_icd_features_to_remove = [c for c in df_clean.columns if c.startswith("item_icd_")]
        item_cpt_features_to_remove = [c for c in df_clean.columns if c.startswith("item_cpt_")]
        if item_icd_features_to_remove or item_cpt_features_to_remove:
            print(f"\n[INFO] For non_opioid_ed cohort: Removing ICD and CPT features (polypharmacy uses drugs only)")
            print(f"  Removing {len(item_icd_features_to_remove)} ICD features and {len(item_cpt_features_to_remove)} CPT features")
            safe_features = [c for c in safe_features if c not in item_icd_features_to_remove and c not in item_cpt_features_to_remove]
            leakage.update(item_icd_features_to_remove)
            leakage.update(item_cpt_features_to_remove)
            df_clean = df[safe_features].copy()

    # 7. Validate item_* features for post-target leakage (drugs and ICD codes)
    print(f"\n[INFO] Validating item_* features for post-target leakage...")
    item_drug_features = [c for c in df_clean.columns if c.startswith("item_drug_")]
    item_icd_features = [c for c in df_clean.columns if c.startswith("item_icd_")]
    item_cpt_features = [c for c in df_clean.columns if c.startswith("item_cpt_")]

    post_target_item_features = []

    if item_drug_features or item_icd_features or item_cpt_features:
        # Check underlying event data for post-target leakage (canonical location only)
        model_data_root = get_model_data_root()
        model_data_path = (
            model_data_root
            / f"cohort_name={cohort}"
            / f"age_band={age_band}"
            / "model_events.parquet"
        )
        if not model_data_path.exists():
            model_data_path = (
                model_data_root
                / f"cohort_name={cohort}"
                / f"age_band={age_band}"
                / "model_events_no_protocols.parquet"
            )

        if model_data_path.exists():
            try:
                # Determine target date field (must exist in model_events.parquet; Step 4 uses canonical names)
                if "opioid" in cohort.lower():
                    target_date_field = "first_f1120_date"
                else:
                    target_date_field = "first_o11_p_date"

                con = duckdb.connect()
                model_data_path_str = str(model_data_path).replace("\\", "/")

                # model_events.parquet has target date from Step 4: first_f1120_date / first_o11_p_date.
                # Step 2 already constrains target events to a 21-day window; temporal filtering is done there.
                parquet_cols = [
                    row[0]
                    for row in con.execute(
                        f"DESCRIBE SELECT * FROM read_parquet('{model_data_path_str}')"
                    ).fetchall()
                ]
                if target_date_field not in parquet_cols:
                    print(
                        f"  [INFO] model_events.parquet has no column '{target_date_field}'; "
                        "skipping post-target leakage validation (best-effort)."
                    )
                    con.close()
                else:
                    # Check each item feature for post-target events
                    for feature_name in item_drug_features + item_icd_features + item_cpt_features:
                        # Extract the code/drug name from feature name
                        if feature_name.startswith("item_drug_"):
                            code_name = feature_name.replace("item_drug_", "")
                            code_column = "drug_name"
                        elif feature_name.startswith("item_icd_"):
                            code_name = feature_name.replace("item_icd_", "")
                            code_column = None  # Will check all ICD columns
                        elif feature_name.startswith("item_cpt_"):
                            code_name = feature_name.replace("item_cpt_", "")
                            code_column = "procedure_code"
                        else:
                            continue

                        # Get patients who have this feature = 1
                        patients_with_feature = df_clean[df_clean[feature_name] == 1]["mi_person_key"].astype(str).unique().tolist()

                        if not patients_with_feature or len(patients_with_feature) == 0:
                            continue

                        # Sanitize code_name for SQL (escape single quotes)
                        sanitized_code_name = code_name.replace("'", "''")

                        # Limit to reasonable batch size for query
                        max_batch_size = 1000
                        post_target_found = False

                        for i in range(0, len(patients_with_feature), max_batch_size):
                            batch = patients_with_feature[i:i + max_batch_size]
                            # Sanitize patient IDs for SQL (escape single quotes)
                            sanitized_patients = [p.replace("'", "''") for p in batch]
                            patient_list = ",".join([f"'{p}'" for p in sanitized_patients])

                            # Check if any of these patients have this code/drug AFTER target event
                            if code_column == "drug_name":
                                query = f"""
                                SELECT COUNT(*) as post_target_count
                                FROM read_parquet('{model_data_path_str}')
                                WHERE CAST(mi_person_key AS VARCHAR) IN ({patient_list})
                                  AND drug_name = '{sanitized_code_name}'
                                  AND {target_date_field} IS NOT NULL
                                  AND event_date IS NOT NULL
                                  AND CAST(event_date AS TIMESTAMP) >= CAST({target_date_field} AS TIMESTAMP)
                                """
                            elif code_column == "procedure_code":
                                query = f"""
                                SELECT COUNT(*) as post_target_count
                                FROM read_parquet('{model_data_path_str}')
                                WHERE CAST(mi_person_key AS VARCHAR) IN ({patient_list})
                                  AND procedure_code = '{sanitized_code_name}'
                                  AND {target_date_field} IS NOT NULL
                                  AND event_date IS NOT NULL
                                  AND CAST(event_date AS TIMESTAMP) >= CAST({target_date_field} AS TIMESTAMP)
                                """
                            else:
                                # Check all ICD diagnosis columns
                                query = f"""
                                SELECT COUNT(*) as post_target_count
                                FROM read_parquet('{model_data_path_str}')
                                WHERE CAST(mi_person_key AS VARCHAR) IN ({patient_list})
                                  AND (
                                    primary_icd_diagnosis_code = '{sanitized_code_name}'
                                    OR two_icd_diagnosis_code = '{sanitized_code_name}'
                                    OR three_icd_diagnosis_code = '{sanitized_code_name}'
                                    OR four_icd_diagnosis_code = '{sanitized_code_name}'
                                    OR five_icd_diagnosis_code = '{sanitized_code_name}'
                                    OR six_icd_diagnosis_code = '{sanitized_code_name}'
                                    OR seven_icd_diagnosis_code = '{sanitized_code_name}'
                                    OR eight_icd_diagnosis_code = '{sanitized_code_name}'
                                    OR nine_icd_diagnosis_code = '{sanitized_code_name}'
                                    OR ten_icd_diagnosis_code = '{sanitized_code_name}'
                                  )
                                  AND {target_date_field} IS NOT NULL
                                  AND event_date IS NOT NULL
                                  AND CAST(event_date AS TIMESTAMP) >= CAST({target_date_field} AS TIMESTAMP)
                                """

                            result = con.execute(query).df()
                            post_target_count = result.iloc[0]["post_target_count"] if len(result) > 0 else 0

                            if post_target_count > 0:
                                post_target_found = True
                                if feature_name not in post_target_item_features:
                                    post_target_item_features.append(feature_name)
                                    print(f"  [WARNING] {feature_name}: {post_target_count} post-target events found (TARGET LEAKAGE)")
                                break  # Found leakage, no need to check more batches

                        if post_target_found:
                            continue  # Move to next feature

                    con.close()

            except Exception as e:
                print(f"  [WARNING] Could not validate item_* features against event data: {e}")
                print(f"  [INFO] Skipping post-target validation (this is a best-effort check)")
        else:
            print(f"  [INFO] Model data file not found for validation: {model_data_path}")
            print(f"  [INFO] Skipping post-target validation (this is a best-effort check)")

    if post_target_item_features:
        print(f"\n[WARNING] Found {len(post_target_item_features)} item_* features with post-target events:")
        for f in post_target_item_features:
            print(f"  - {f}")
        print("[INFO] These features may include post-target events and should be removed")
        safe_features = [c for c in safe_features if c not in post_target_item_features]
        leakage.update(post_target_item_features)
        df_clean = df[safe_features].copy()
    else:
        print(f"  [OK] No post-target leakage detected in item_* features")

    # Verify important predictive features are preserved
    sequence_features = [c for c in df_clean.columns if "sequence" in c.lower() or "trace" in c.lower()]
    interval_features_kept = [c for c in safe_features if "interval" in c.lower()]
    fpgrowth_features_kept = [c for c in safe_features if any(x in c for x in ["itemset", "rule", "support", "confidence", "lift"])]

    print(f"\n[INFO] Preserving important predictive features:")
    print(f"  Sequence features (top/rare): {len([c for c in sequence_features if c in safe_features])}")
    print(f"  Time interval features (between events): {len(interval_features_kept)}")
    print(f"  FP-Growth features (itemsets/rules): {len(fpgrowth_features_kept)}")

    print(f"\n[INFO] Removing {len(leakage)} leakage features")
    print(f"[INFO] Clean dataset: {len(df_clean)} patients, {len(df_clean.columns)} columns")
    print(f"[INFO] All features are from events BEFORE F1120 (excluding F1120 and everything after)")

    return df_clean


def _validate_s3_file_has_controls(s3_path: str) -> dict:
    """
    Validate that an S3 parquet file contains both cases (target=1) and controls (target=0).
    Uses DuckDB's S3 support to query without downloading the entire file.
    
    Returns:
        dict with keys: has_controls (bool), n_cases (int), n_controls (int), error (str or None)
    """
    import duckdb
    con = duckdb.connect()
    try:
        result = con.execute(
            f"""
            SELECT 
                COUNT(*) FILTER (WHERE target = 1) AS n_cases,
                COUNT(*) FILTER (WHERE target = 0) AS n_controls
            FROM read_parquet('{s3_path}')
            """
        ).fetchone()
        
        n_cases = result[0] if result else 0
        n_controls = result[1] if result else 0
        has_controls = n_controls > 0
        
        return {
            "has_controls": has_controls,
            "n_cases": n_cases,
            "n_controls": n_controls,
            "error": None,
        }
    except Exception as e:
        return {
            "has_controls": False,
            "n_cases": 0,
            "n_controls": 0,
            "error": str(e),
        }
    finally:
        con.close()


def _resolve_model_events_path(cohort: str, age_band: str) -> Path:
    """
    Resolve the path to model_events.parquet using the single canonical location.

    Uses py_helpers.env_utils.get_model_data_root() (one location for efficiency).
    If not found locally, tries S3 gold/cohorts_model_data/ and downloads to the
    canonical path.
    """
    model_data_root = get_model_data_root()
    canonical_path = (
        model_data_root
        / f"cohort_name={cohort}"
        / f"age_band={age_band}"
        / "model_events.parquet"
    )
    candidates = [canonical_path]
    download_dest = canonical_path

    # Check canonical location
    for path in candidates:
        if path.exists():
            print(f"Found model_events.parquet at: {path}")
            return path

    # Log which paths we checked
    print("Model data not found locally. Checked paths:")
    for path in candidates:
        print(f"  - {path} (exists: {path.exists()})")

    # If not found locally, try downloading from S3
    s3_key_candidates = [
        f"gold/cohorts_model_data/cohort_name={cohort}/age_band={age_band}/model_events.parquet",
        f"gold/model_data/cohort_name={cohort}/age_band={age_band}/model_events.parquet",
        f"gold/model_data/{cohort}/{age_band}/model_events.parquet",
    ]

    download_dest.parent.mkdir(parents=True, exist_ok=True)

    for s3_key in s3_key_candidates:
        try:
            # Check if file exists in S3
            s3_client.head_object(Bucket=S3_BUCKET, Key=s3_key)
            s3_path = f"s3://{S3_BUCKET}/{s3_key}"
            
            # Validate controls BEFORE downloading (using DuckDB S3 support)
            print(f"Checking S3 file for controls: {s3_path}")
            validation_result = _validate_s3_file_has_controls(s3_path)
            
            if validation_result.get("error"):
                print(f"Warning: Could not validate S3 file {s3_path}: {validation_result['error']}")
                print("Proceeding with download and will validate after...")
            elif not validation_result.get("has_controls", False):
                print(
                    f"ERROR: S3 file {s3_path} is missing controls! "
                    f"Cases: {validation_result.get('n_cases', 0)}, Controls: {validation_result.get('n_controls', 0)}"
                )
                print(
                    f"This file should be regenerated with controls. "
                    f"Please run: python 4_model_data/create_model_data.py --cohort {cohort} --age-band {age_band}"
                )
                print("Skipping this S3 file and trying next candidate...")
                continue  # Skip this S3 file, try next candidate
            
            # Download the file
            print(f"Downloading model_events.parquet from S3: {s3_path}")
            print(f"Downloading to: {download_dest}")
            obj = s3_client.get_object(Bucket=S3_BUCKET, Key=s3_key)
            with open(download_dest, 'wb') as f:
                f.write(obj['Body'].read())
            print(f"Saved to: {download_dest}")
            
            # Validate again after download (double-check)
            import duckdb
            con = duckdb.connect()
            try:
                result = con.execute(
                    f"""
                    SELECT 
                        COUNT(*) FILTER (WHERE target = 1) AS n_cases,
                        COUNT(*) FILTER (WHERE target = 0) AS n_controls
                    FROM read_parquet('{download_dest}')
                    """
                ).fetchone()
                
                n_cases = result[0] if result else 0
                n_controls = result[1] if result else 0
                if n_controls == 0:
                    print(
                        f"ERROR: Downloaded file is missing controls! "
                        f"Cases: {n_cases}, Controls: {n_controls}"
                    )
                    print(
                        f"This file should be regenerated with controls. "
                        f"Please run: python 4_model_data/create_model_data.py --cohort {cohort} --age-band {age_band}"
                    )
                    # Delete the invalid file
                    download_dest.unlink()
                    print("Deleted invalid file. Trying next S3 candidate...")
                    continue
                else:
                    print(f"Validation passed: {n_cases} cases, {n_controls} controls")
            finally:
                con.close()
            
            return download_dest
        except Exception as e:
            print(f"S3 key not found or error: {s3_key} - {e}")
            continue

    # If all checks failed, raise error with helpful message
    error_msg = (
        f"Model data not found for cohort={cohort}, age_band={age_band}.\n"
        "Checked locations:\n"
    )
    for path in candidates:
        error_msg += f"  - {path} (exists: {path.exists()})\n"
    error_msg += "\nS3 locations checked:\n"
    for s3_key in s3_key_candidates:
        error_msg += f"  - s3://{S3_BUCKET}/{s3_key}\n"
    raise FileNotFoundError(error_msg)


def _create_aggregated_feature_importance_visualizations(
    cohort: str, age_band: str, out_base: Path
) -> None:
    """
    Create bar chart and heatmap visualizations from aggregated feature importance CSV.
    Resolves path via shared logic (3a outputs, gold/feature_importance, S3).
    """
    age_band_fname = age_band.replace("-", "_")
    filename = f"{cohort}_{age_band_fname}_aggregated_feature_importance.csv"

    # Resolve aggregated FI path: same order as Step 3b / workflow (3a outputs, from_s3, gold, S3)
    agg_csv_path = None
    try:
        from py_helpers.feature_importance_eda_utils import resolve_aggregated_fi_path
        agg_csv_path = resolve_aggregated_fi_path(cohort, age_band, PROJECT_ROOT)
    except Exception:
        pass
    if agg_csv_path is None:
        # Local sync location (workflow syncs S3 gold/feature_importance here)
        data_root = get_data_root()
        gold_fi = data_root / "gold" / "feature_importance" / cohort / age_band / filename
        if gold_fi.exists():
            agg_csv_path = gold_fi
    if agg_csv_path is None:
        # Legacy: 3a outputs and 3_feature_importance paths
        candidates = [
            PROJECT_ROOT / "3a_feature_importance" / "outputs" / cohort / filename,
            PROJECT_ROOT / "3a_feature_importance" / "outputs" / cohort / age_band_fname / filename,
            PROJECT_ROOT / "3a_feature_importance" / "from_s3" / "by_cohort" / cohort / age_band_fname / filename,
            PROJECT_ROOT / "3_feature_importance" / "outputs" / cohort / filename,
            PROJECT_ROOT / "3_feature_importance" / "from_s3" / "by_cohort" / cohort / filename,
        ]
        for p in candidates:
            if p.exists():
                agg_csv_path = p
                break

    if agg_csv_path is None or not agg_csv_path.exists():
        print(f"[WARNING] Aggregated feature importance CSV not found (checked 3a outputs, gold/feature_importance, S3).")
        print("Skipping visualization generation.")
        return
    
    try:
        df = pd.read_csv(agg_csv_path)
        
        # Ensure required columns exist
        if "feature" not in df.columns:
            print(f"[WARNING] 'feature' column not found in {agg_csv_path}")
            return
        
        # Use importance_scaled if available, otherwise importance_normalized, otherwise importance
        importance_col = None
        for col in ["importance_scaled", "importance_normalized", "importance"]:
            if col in df.columns:
                importance_col = col
                break
        
        if importance_col is None:
            print(f"[WARNING] No importance column found in {agg_csv_path}")
            return
        
        # Create plots directory
        plots_dir = out_base / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        
        # Get top 50 features
        top50 = df.nlargest(50, importance_col).copy()
        top50 = top50.sort_values(importance_col, ascending=True)  # For horizontal bar chart
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.dpi'] = 300
        plt.rcParams['savefig.dpi'] = 300
        
        # ============================================================================
        # PLOT 1: Top 50 Features Bar Chart
        # ============================================================================
        print(f"\nCreating top 50 features bar chart...")
        fig, ax = plt.subplots(figsize=(12, 14))
        
        bars = ax.barh(range(len(top50)), top50[importance_col].values, 
                       color='steelblue', alpha=0.8)
        ax.set_yticks(range(len(top50)))
        ax.set_yticklabels(top50['feature'].values, fontsize=8)
        ax.set_xlabel('Importance Score', fontsize=12)
        ax.set_ylabel('Feature', fontsize=12)
        ax.set_title(
            f'Top 50 Features by {importance_col.replace("_", " ").title()}\n'
            f'{cohort} / {age_band}',
            fontsize=14, fontweight='bold'
        )
        ax.invert_yaxis()
        ax.grid(axis='x', linestyle='--', alpha=0.3)
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, top50[importance_col].values)):
            ax.text(val, i, f' {val:.3f}', va='center', fontsize=7)
        
        plt.tight_layout()
        bar_chart_path = plots_dir / f"{cohort}_{age_band_fname}_top50_features_bar_chart.png"
        plt.savefig(bar_chart_path, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"✓ Saved bar chart: {bar_chart_path}")
        
        # ============================================================================
        # PLOT 2: Heatmap (if we have multiple importance metrics or model counts)
        # ============================================================================
        print(f"Creating feature importance heatmap...")
        
        # Prepare data for heatmap
        # If we have n_models or multiple importance columns, create a heatmap
        heatmap_data = top50[['feature', importance_col]].copy()
        
        # Add rank for visualization
        heatmap_data['rank'] = range(1, len(heatmap_data) + 1)
        
        # Create a pivot-style visualization showing top features
        # For a simple heatmap, we'll show importance values as a heatmap
        fig, ax = plt.subplots(figsize=(14, 12))
        
        # Create a matrix where rows are features and columns are importance
        # We'll show top 50 features in a heatmap format
        heatmap_matrix = top50[[importance_col]].T.values
        
        # Create heatmap
        sns.heatmap(
            heatmap_matrix,
            yticklabels=[importance_col.replace("_", " ").title()],
            xticklabels=top50['feature'].values,
            annot=False,
            fmt='.3f',
            cmap='YlOrRd',
            cbar_kws={'label': 'Importance Score'},
            ax=ax
        )
        
        ax.set_title(
            f'Feature Importance Heatmap (Top 50)\n{cohort} / {age_band}',
            fontsize=14, fontweight='bold'
        )
        ax.set_xlabel('Feature', fontsize=12)
        ax.set_ylabel('Importance Metric', fontsize=12)
        
        # Rotate x-axis labels for readability
        plt.setp(ax.get_xticklabels(), rotation=90, ha='right', fontsize=6)
        
        plt.tight_layout()
        heatmap_path = plots_dir / f"{cohort}_{age_band_fname}_feature_importance_heatmap.png"
        plt.savefig(heatmap_path, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"✓ Saved heatmap: {heatmap_path}")
        
        # Upload to S3 if available
        try:
            import subprocess
            import shutil
            aws_cmd = shutil.which("aws")
            if aws_cmd:
                s3_plots_base = f"s3://pgxdatalake/gold/final_model/{cohort}/{age_band}/plots/"
                for plot_file in [bar_chart_path, heatmap_path]:
                    s3_path = f"{s3_plots_base}{plot_file.name}"
                    result = subprocess.run(
                        [aws_cmd, 's3', 'cp', str(plot_file), s3_path],
                        capture_output=True, text=True, timeout=60
                    )
                    if result.returncode == 0:
                        print(f"✓ Uploaded to S3: {s3_path}")
        except Exception as e:
            print(f"[WARNING] Could not upload plots to S3: {e}")
        
        print(f"\nVisualizations saved to: {plots_dir}")
        
    except Exception as e:
        print(f"[WARNING] Error creating visualizations: {e}")
        import traceback
        traceback.print_exc()


def _load_aggregated_feature_importance_codes(cohort: str, age_band: str, top_n: int = None) -> List[tuple[str, str]]:
    """
    Load refined feature importance codes (drug/ICD/CPT) from Step 3b (cohort_feature_importance).
    
    This function now uses the Step 3b refined feature importance files, which include
    leakage filtering and refinement from BupaR post-target analysis.
    
    Args:
        cohort: Cohort name
        age_band: Age band
        top_n: Maximum number of top features to return (default: None = no limit)
               If None, returns all features sorted by importance.
               If set, limits to top_n to prevent memory/SQL issues.
    
    Returns:
        List of item codes (drug names, ICD codes, CPT codes) from refined FI CSV,
        sorted by importance_scaled (descending), optionally limited to top_n.
    """
    age_band_fname = age_band.replace("-", "_")
    
    # REQUIRED: Step 3b refined feature importance (removes target leakage)
    # No fallback - Step 3b must run before Step 6
    refined_csv_path = (
        PROJECT_ROOT
        / "3b_feature_importance_eda"
        / "outputs"
        / cohort
        / age_band_fname
        / f"{cohort}_{age_band_fname}_cohort_feature_importance.csv"
    )
    
    # Try S3 download if not found locally
    if not refined_csv_path.exists():
        age_band_fname_s3 = age_band.replace("-", "_")
        s3_key = (
            f"gold/feature_importance/{cohort}/{age_band}/"
            f"{cohort}_{age_band_fname_s3}_cohort_feature_importance.csv"
        )
        s3_path = f"s3://{S3_BUCKET}/{s3_key}"
        
        print(f"[INFO] Step 3b refined feature importance not found locally.")
        print(f"[INFO] Checking S3: {s3_path}")
        
        try:
            # Check if file exists in S3
            s3_client.head_object(Bucket=S3_BUCKET, Key=s3_key)
            print(f"[INFO] Found in S3. Downloading...")
            refined_csv_path.parent.mkdir(parents=True, exist_ok=True)
            obj = s3_client.get_object(Bucket=S3_BUCKET, Key=s3_key)
            with open(refined_csv_path, 'wb') as f:
                f.write(obj['Body'].read())
            print(f"[INFO] Successfully downloaded from S3: {refined_csv_path}")
        except s3_client.exceptions.ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', 'Unknown')
            if error_code == '404':
                print(f"[WARN] File not found in S3: {s3_path}")
            else:
                print(f"[WARN] S3 check failed (error: {error_code}): {e}")
        except Exception as e:
            print(f"[WARN] S3 download failed: {e}")
    
    # REQUIRED: Step 3b refined feature importance must exist (no fallback)
    if not refined_csv_path.exists():
        raise FileNotFoundError(
            f"Step 3b refined feature importance CSV is REQUIRED (removes target leakage) but not found for {cohort}/{age_band}.\n"
            f"Expected location: {refined_csv_path}\n"
            f"S3 location: s3://{S3_BUCKET}/gold/feature_importance/{cohort}/{age_band}/{cohort}_{age_band_fname}_cohort_feature_importance.csv\n"
            f"Step 3b must run before Step 6 to produce refined features with leakage filtering.\n"
            f"Run: python 3b_feature_importance_eda/run_feature_importance_eda.py --cohort {cohort} --age-band {age_band}"
        )
    
    csv_path = refined_csv_path
    print(f"\n[INFO] Using Step 3b refined feature importance (leakage-filtered): {csv_path}")
    
    df = pd.read_csv(csv_path)
    if "feature" not in df.columns:
        raise ValueError(f"'feature' column not found in {csv_path}")
    
    # Log file info for verification
    print(f"[INFO] Loaded {len(df)} features from Step 3b refined feature importance")
    if "importance_scaled" in df.columns:
        print(f"[INFO] Feature importance range: min={df['importance_scaled'].min():.6f}, max={df['importance_scaled'].max():.6f}, mean={df['importance_scaled'].mean():.6f}")
    
    # Check for potential leakage indicators in feature names
    leakage_indicators = ['post', 'f1120', 'target', 'leakage']
    potential_leakage = df[df['feature'].str.lower().str.contains('|'.join(leakage_indicators), na=False)]
    if len(potential_leakage) > 0:
        print(f"[WARN] Found {len(potential_leakage)} features with potential leakage indicators:")
        for feat in potential_leakage['feature'].head(10):
            print(f"[WARN]   - {feat}")
        if len(potential_leakage) > 10:
            print(f"[WARN]   ... and {len(potential_leakage) - 10} more")
    
    # Resolve importance column (Step 3b may use importance_scaled_by_model_sum or importance_mean)
    importance_col = None
    for col in (
        "importance_scaled",
        "importance_normalized",
        "importance_scaled_by_model_sum",
        "importance_mean",
    ):
        if col in df.columns:
            importance_col = col
            break
    if importance_col:
        # Filter to only features with importance > 0 (or > small epsilon)
        initial_count = len(df)
        df = df[df[importance_col] > 1e-10].copy()
        filtered_count = len(df)
        if filtered_count < initial_count:
            print(f"[INFO] Filtered out {initial_count - filtered_count} features with zero/negative importance")
            print(f"[INFO] Keeping {filtered_count} features with importance > 0")
    else:
        print(f"[WARNING] No importance column found (tried importance_scaled, importance_normalized, importance_scaled_by_model_sum, importance_mean). Not filtering by importance.")

    # For non_opioid_ed we only use drug features; when Step 3b provides code_type, keep only drug
    if ("non_opioid" in cohort.lower() or "ed_non_opioid" in cohort.lower()) and "code_type" in df.columns:
        before = len(df)
        df = df[df["code_type"].astype(str).str.strip().str.lower() == "drug"].copy()
        if len(df) < before:
            print(f"[INFO] For non_opioid_ed (drug only): kept {len(df)} drug features, dropped {before - len(df)} non-drug from Step 3b.")
    
    # Remove duplicate features (keep first occurrence, which should be highest importance after sorting)
    initial_count = len(df)
    df = df.drop_duplicates(subset=["feature"], keep="first")
    if len(df) < initial_count:
        print(f"[INFO] Removed {initial_count - len(df)} duplicate features")
    
    # Sort by importance (use first available column)
    if importance_col:
        df = df.sort_values(importance_col, ascending=False)
    else:
        print(f"[WARNING] No importance column found. Using row order.")
    
    # Create a mapping of feature -> importance for sorting
    if importance_col:
        feature_importance_map = dict(zip(df["feature"], df[importance_col]))
    else:
        # No importance column - use row order (df is already sorted)
        feature_importance_map = {feat: -idx for idx, feat in enumerate(df["feature"])}

    # Use code_type from Step 3b when present (single source of truth for drug/icd/cpt)
    use_csv_code_type = "code_type" in df.columns
    if use_csv_code_type:
        df["_code_type_lower"] = df["code_type"].astype(str).str.strip().str.lower()
        print(f"[INFO] Using code_type from Step 3b for feature classification (drug/icd/cpt)")
    
    # Parse features to extract code and type, preserving importance for sorting
    # Format: item_{type}_{code} or item_{code} (Step 3a); Step 3b may add code_type column
    # Exclude drugs in DRUG_NAMES_EXCLUDED_MODEL_TRAINING (Narcan, etc.) case-insensitively
    excluded_drugs_lower = {z.lower() for z in DRUG_NAMES_EXCLUDED_MODEL_TRAINING}
    parsed_features = []
    n_excluded_drugs = 0
    for _, row in df.iterrows():
        feature_str = str(row["feature"])
        importance = feature_importance_map.get(feature_str, 0)
        if use_csv_code_type:
            ctype = row.get("_code_type_lower", "drug")
            if ctype not in ("drug", "icd", "cpt"):
                continue
        else:
            ctype = None
        
        if feature_str.startswith("item_drug_"):
            code = feature_str.replace("item_drug_", "", 1)
            ftype = ctype if ctype else "drug"
        elif feature_str.startswith("item_icd_"):
            code = feature_str.replace("item_icd_", "", 1)
            ftype = ctype if ctype else "icd"
        elif feature_str.startswith("item_cpt_"):
            code = feature_str.replace("item_cpt_", "", 1)
            ftype = ctype if ctype else "cpt"
        elif feature_str.startswith("item_"):
            code = feature_str.replace("item_", "", 1)
            if ctype:
                ftype = ctype
            elif any(c.isalpha() for c in code[:3]) and len(code) > 5:
                ftype = "drug"
            elif code.replace(".", "").replace("-", "").isdigit() or (len(code) <= 10 and any(c.isdigit() for c in code)):
                ftype = "icd"
            else:
                ftype = "drug"
        else:
            code = feature_str
            ftype = ctype if ctype else "drug"
        if ftype == "drug" and (code or "").strip().lower() in excluded_drugs_lower:
            n_excluded_drugs += 1
            continue
        if any((sub.lower() in (code or "").lower()) for sub in FEATURE_SUBSTRINGS_EXCLUDED):
            n_excluded_drugs += 1
            continue
        parsed_features.append((ftype, code, importance))
    if n_excluded_drugs:
        print(f"[INFO] Excluded {n_excluded_drugs} feature(s) (excluded drugs + substrings e.g. syringe)")
    
    # Sort by importance (descending)
    parsed_features.sort(key=lambda x: x[2], reverse=True)
    
    # Remove duplicate (type, code) pairs - keep the one with highest importance
    seen = {}
    for ftype, code, importance in parsed_features:
        key = (ftype, code)
        if key not in seen or seen[key][2] < importance:
            seen[key] = (ftype, code, importance)
    
    parsed_features = list(seen.values())
    parsed_features.sort(key=lambda x: x[2], reverse=True)  # Re-sort after deduplication
    
    if top_n is not None and len(parsed_features) > top_n:
        parsed_features = parsed_features[:top_n]
        print(f"[INFO] Limited to top {top_n} features from {len(seen)} unique features with signal")
    else:
        print(f"[INFO] Loaded {len(parsed_features)} unique aggregated feature importance codes with importance > 0")
    
    # Return as list of tuples: (type, code) - drop importance
    return [(ftype, code) for ftype, code, _ in parsed_features]


def build_final_features(cohort: str, age_band: str) -> pd.DataFrame:
    """
    Build final feature matrix using aggregated patient-level features + PGx features only.
    
    We do not calculate trajectory, sequence, or itemset features. Only:
    - n_events (count of events per patient from model_events)
    - item_* binary features (drug/ICD/CPT from aggregated feature importance)
    - PGx features (e.g. pgx_num_drugs, pgx_num_cpic_drugs from 5_pgx_analysis)
    BupaR, DTW, FP-Growth are for dashboard visualizations only, not for model training.
    
    Inputs:
      - 4_model_data/cohort_name={cohort}/age_band={age_band}/model_events.parquet
        (event-level cases + controls with `target` column)
      - 5_pgx_analysis/.../{cohort}/{age_band}/pgx_added_features_*.csv

    The assembled dataset includes BOTH:
      - case patients (`target=1`) from the cohort model_events.parquet
      - within-cohort control patients (`target=0`) from the same file

    Event-level model_data is aggregated to patient-level using DuckDB before
    merging with PGx feature table.
    """
    age_band_fname = age_band_to_fname(age_band)

    # Base model_data for target cohort (event-level; collapse to patient-level via DuckDB)
    events_path = _resolve_model_events_path(cohort, age_band)

    print(f"Loading model data (cases + controls) from {events_path}")
    # Create a single DuckDB connection and view for reuse across all queries
    # This avoids repeated parquet scans and can reduce runtime by 20-30%
    con = duckdb.connect()
    try:
        # Create a view so we can reference it multiple times without re-reading the parquet
        con.execute(f"CREATE OR REPLACE VIEW events_view AS SELECT * FROM read_parquet('{events_path}')")
        
        # Check if target column exists, if not create it based on F1120 presence
        # F1120 (opioid dependence) indicates a target case (target=1)
        columns_info = con.execute("DESCRIBE events_view").df()
        has_target_column = 'target' in columns_info['column_name'].values
        
        if not has_target_column:
            print("[WARN] Target column not found in model_events.parquet")
            print("[INFO] Creating target column based on F1120 presence (F1120 = target=1)")
            # Create target column: 1 if patient has F1120 in any ICD diagnosis column, 0 otherwise
            con.execute(f"""
                CREATE OR REPLACE VIEW events_view AS
                SELECT 
                    *,
                    CASE 
                        WHEN primary_icd_diagnosis_code LIKE '%F1120%'
                          OR two_icd_diagnosis_code LIKE '%F1120%'
                          OR three_icd_diagnosis_code LIKE '%F1120%'
                          OR four_icd_diagnosis_code LIKE '%F1120%'
                          OR five_icd_diagnosis_code LIKE '%F1120%'
                          OR six_icd_diagnosis_code LIKE '%F1120%'
                          OR seven_icd_diagnosis_code LIKE '%F1120%'
                          OR eight_icd_diagnosis_code LIKE '%F1120%'
                          OR nine_icd_diagnosis_code LIKE '%F1120%'
                          OR ten_icd_diagnosis_code LIKE '%F1120%'
                        THEN 1
                        ELSE 0
                    END AS target
                FROM read_parquet('{events_path}')
            """)
            print("[OK] Target column created successfully")
        
        # Aggregate event-level data to one row per patient with label
        # Use MAX(target) to handle patients with mixed targets (prefer case=1 if any event is case)
        # This ensures each patient appears only once
        grouped = con.execute(
            """
            SELECT
                CAST(mi_person_key AS VARCHAR) AS mi_person_key,
                CAST(MAX(target) AS INTEGER)   AS target,
                COUNT(*)                       AS n_events
            FROM events_view
            WHERE target IN (0, 1)
            GROUP BY mi_person_key
            """
        ).df()

        # Ensure binary labels
        grouped["target"] = grouped["target"].astype(int).clip(lower=0, upper=1)

        # n_event_bin: compute from n_events (P25/P50/P95 → low/medium/high/extreme).
        # Saved as JSON so DTW, FP-Growth, BupaR, and inference use the identical cut-points.
        _thresholds = _compute_bin_thresholds(grouped["n_events"])
        grouped["n_event_bin"] = _assign_n_event_bins(grouped["n_events"], _thresholds)
        _bin_ord = {b: i for i, b in enumerate(_DENSITY_BINS)}
        grouped["n_event_bin_ordinal"] = grouped["n_event_bin"].map(_bin_ord).fillna(0).astype(int)
        _tcache = _threshold_cache_path(PROJECT_ROOT, cohort, age_band)
        _tcache.parent.mkdir(parents=True, exist_ok=True)
        _save_thresholds({**_thresholds, "cohort": cohort, "age_band": age_band}, _tcache)
        print(f"[INFO] n_event_bin thresholds saved: {_tcache}")
        print(f"[INFO] n_event_bin distribution: {grouped['n_event_bin'].value_counts().to_dict()}")

        # Debug: Print class distribution
        target_counts = grouped["target"].value_counts()
        print(f"Class distribution after aggregation:")
        print(f"  Cases (target=1): {target_counts.get(1, 0)}")
        print(f"  Controls (target=0): {target_counts.get(0, 0)}")
        if len(target_counts) < 2:
            print(f"  WARNING: Only one class present! All targets = {target_counts.index[0]}")

        # ------------------------------------------------------------------
        # Create binary features for aggregated feature importance codes
        # ------------------------------------------------------------------
        # Load aggregated FI codes from Step 3 and create binary indicators
        # (1 if patient has code, 0 otherwise). 
        # - For XGBoost: Used as numeric features (required)
        # - For CatBoost: Marked as categorical features for better performance
        #   (CatBoost handles categorical features natively and performs better
        #    when binary features are treated as categorical)
        try:
            # Load ALL aggregated feature importance codes (no limit)
            # User requested all drugs be included regardless of count
            important_codes = _load_aggregated_feature_importance_codes(cohort, age_band, top_n=None)
        except (FileNotFoundError, ValueError) as e:
            print(f"[WARNING] Could not load aggregated FI codes: {e}")
            print("Will create binary features from all codes in model_events.parquet")
            important_codes = None
        
        # Reuse the same connection and view for column inspection
        events_sample = con.execute("SELECT * FROM events_view LIMIT 1").df()
        available_cols = events_sample.columns.tolist()
        
        # Build binary features for each important code
        # Use a safer approach: create a lookup table and join instead of embedding values in SQL
        binary_feature_exprs = []
        
        if important_codes and len(important_codes) > 0:
            # Use a safer approach: create temporary lookup tables and use JOINs
            # This completely avoids SQL injection and special character issues
            
            # Build lookup tables for each column type
            # important_codes is now a list of tuples: (type, code)
            drug_codes = []
            icd_codes = []
            cpt_codes = []
            excluded_drugs_lower = {z.lower() for z in DRUG_NAMES_EXCLUDED_MODEL_TRAINING}
            excluded_substrings_lower = [z.lower() for z in FEATURE_SUBSTRINGS_EXCLUDED]
            
            for code_type, code in important_codes:
                code_str = str(code)
                # Create safe feature name (replace all special chars with underscore)
                code_safe = code_str.replace(' ', '_').replace('-', '_').replace('.', '_').replace('/', '_').replace('&', '_').replace('(', '_').replace(')', '_').replace('[', '_').replace(']', '_').replace('{', '_').replace('}', '_').replace('*', '_').replace('+', '_').replace('=', '_').replace('|', '_').replace('^', '_').replace('%', '_').replace('!', '_').replace('@', '_').replace('#', '_').replace('$', '_').replace('"', '_').replace("'", '_').replace('\\', '_')
                
                # Only create features for the code type specified
                if code_type == "drug" and "drug_name" in available_cols:
                    if any(sub in (code_str or "").lower() for sub in excluded_substrings_lower):
                        continue
                    drug_codes.append((code_str, f"item_drug_{code_safe}"))
                elif code_type == "icd":
                    icd_cols = [c for c in available_cols if 'icd_diagnosis_code' in c.lower()]
                    if icd_cols:
                        icd_codes.append((code_str, f"item_icd_{code_safe}"))
                elif code_type == "cpt" and "procedure_code" in available_cols:
                    cpt_codes.append((code_str, f"item_cpt_{code_safe}"))
            
            # Fallback for non_opioid_ed: if Step 3b yielded no drug codes, use distinct drugs from model_events
            # so we do not end up with only n_events + PGx (e.g. non_opioid_ed 75-84 with 3 features)
            if (
                ("non_opioid" in cohort.lower() or "ed_non_opioid" in cohort.lower())
                and not drug_codes
                and "drug_name" in available_cols
            ):
                print(
                    "[WARN] Step 3b yielded no drug codes for this cohort/age_band; "
                    "using distinct drug_name from model_events as fallback."
                )
                distinct_drugs_df = con.execute(
                    """
                    SELECT DISTINCT drug_name
                    FROM events_view
                    WHERE drug_name IS NOT NULL AND TRIM(CAST(drug_name AS VARCHAR)) <> ''
                    """
                ).df()
                for drug in distinct_drugs_df["drug_name"].unique():
                    drug_str = str(drug).strip()
                    if not drug_str or drug_str.lower() in excluded_drugs_lower:
                        continue
                    if any(sub in drug_str.lower() for sub in excluded_substrings_lower):
                        continue
                    code_safe = drug_str.replace(" ", "_").replace("-", "_").replace(".", "_").replace("/", "_").replace("&", "_").replace("(", "_").replace(")", "_").replace("[", "_").replace("]", "_").replace("{", "_").replace("}", "_").replace("*", "_").replace("+", "_").replace("=", "_").replace("|", "_").replace("^", "_").replace("%", "_").replace('"', "_").replace("'", "_").replace("\\", "_")
                    drug_codes.append((drug_str, f"item_drug_{code_safe}"))
                if drug_codes:
                    print(f"[INFO] Fallback: added {len(drug_codes)} drug features from model_events.")

            # Create temporary lookup tables using executemany (parameterized, safe)
            if drug_codes:
                con.execute("CREATE TEMP TABLE IF NOT EXISTS drug_code_lookup(code_value VARCHAR, feature_name VARCHAR)")
                con.executemany("INSERT INTO drug_code_lookup VALUES (?, ?)", drug_codes)
            
            if icd_codes:
                con.execute("CREATE TEMP TABLE IF NOT EXISTS icd_code_lookup(code_value VARCHAR, feature_name VARCHAR)")
                con.executemany("INSERT INTO icd_code_lookup VALUES (?, ?)", icd_codes)
            
            if cpt_codes:
                con.execute("CREATE TEMP TABLE IF NOT EXISTS cpt_code_lookup(code_value VARCHAR, feature_name VARCHAR)")
                con.executemany("INSERT INTO cpt_code_lookup VALUES (?, ?)", cpt_codes)
            
            # Build feature expressions - we'll use these in the SQL query below
            # The actual SQL will use JOINs to lookup tables, not these expressions directly
            all_feature_names = []
            if drug_codes:
                all_feature_names.extend([name for _, name in drug_codes])
            if icd_codes:
                all_feature_names.extend([name for _, name in icd_codes])
            if cpt_codes:
                all_feature_names.extend([name for _, name in cpt_codes])
            
            # Store feature names for use in SQL query
            binary_feature_exprs = all_feature_names
        else:
            # Fallback: create binary features for all distinct codes
            print("Creating binary features for all distinct codes in data...")
            excluded_drugs_lower = {z.lower() for z in DRUG_NAMES_EXCLUDED_MODEL_TRAINING}
            excluded_substrings_lower = [z.lower() for z in FEATURE_SUBSTRINGS_EXCLUDED]
            
            if "drug_name" in available_cols:
                distinct_drugs = con.execute(
                    """
                    SELECT DISTINCT drug_name
                    FROM events_view
                    WHERE drug_name IS NOT NULL AND TRIM(drug_name) <> ''
                    """
                ).df()
                for drug in distinct_drugs["drug_name"].unique():
                    drug_str = str(drug)
                    if drug_str.strip().lower() in excluded_drugs_lower:
                        continue
                    if any(sub in drug_str.lower() for sub in excluded_substrings_lower):
                        continue
                    drug_escaped = drug_str.replace("\\", "\\\\").replace("'", "''")
                    drug_safe = drug_str.replace(' ', '_').replace('-', '_').replace('.', '_').replace('/', '_').replace('&', '_').replace('(', '_').replace(')', '_').replace('[', '_').replace(']', '_').replace('{', '_').replace('}', '_').replace('*', '_').replace('+', '_').replace('=', '_').replace('|', '_').replace('^', '_').replace('%', '_').replace('"', '_').replace("'", '_').replace('\\', '_')
                    feature_name = f"item_drug_{drug_safe}"
                    binary_feature_exprs.append(
                        f"MAX(CASE WHEN CAST(drug_name AS VARCHAR) = CAST('{drug_escaped}' AS VARCHAR) THEN 1 ELSE 0 END) AS {feature_name}"
                    )
            
            if "primary_icd_diagnosis_code" in available_cols:
                distinct_icd = con.execute(
                    """
                    SELECT DISTINCT primary_icd_diagnosis_code
                    FROM events_view
                    WHERE primary_icd_diagnosis_code IS NOT NULL AND TRIM(primary_icd_diagnosis_code) <> ''
                    """
                ).df()
                for icd in distinct_icd["primary_icd_diagnosis_code"].unique():
                    icd_str = str(icd)
                    icd_escaped = icd_str.replace("\\", "\\\\").replace("'", "''")
                    icd_safe = icd_str.replace('.', '_').replace('-', '_').replace('/', '_').replace('&', '_').replace('(', '_').replace(')', '_').replace('[', '_').replace(']', '_').replace('{', '_').replace('}', '_').replace('*', '_').replace('+', '_').replace('=', '_').replace('|', '_').replace('^', '_').replace('%', '_').replace('"', '_').replace("'", '_').replace('\\', '_')
                    feature_name = f"item_icd_{icd_safe}"
                    binary_feature_exprs.append(
                        f"MAX(CASE WHEN CAST(primary_icd_diagnosis_code AS VARCHAR) = CAST('{icd_escaped}' AS VARCHAR) THEN 1 ELSE 0 END) AS {feature_name}"
                    )
            
            if "procedure_code" in available_cols:
                distinct_cpt = con.execute(
                    """
                    SELECT DISTINCT procedure_code
                    FROM events_view
                    WHERE procedure_code IS NOT NULL AND TRIM(procedure_code) <> ''
                    """
                ).df()
                for cpt in distinct_cpt["procedure_code"].unique():
                    cpt_str = str(cpt)
                    cpt_escaped = cpt_str.replace("\\", "\\\\").replace("'", "''")
                    cpt_safe = cpt_str.replace('.', '_').replace('-', '_').replace('/', '_').replace('&', '_').replace('(', '_').replace(')', '_').replace('[', '_').replace(']', '_').replace('{', '_').replace('}', '_').replace('*', '_').replace('+', '_').replace('=', '_').replace('|', '_').replace('^', '_').replace('%', '_').replace('"', '_').replace("'", '_').replace('\\', '_')
                    feature_name = f"item_cpt_{cpt_safe}"
                    binary_feature_exprs.append(
                        f"MAX(CASE WHEN CAST(procedure_code AS VARCHAR) = CAST('{cpt_escaped}' AS VARCHAR) THEN 1 ELSE 0 END) AS {feature_name}"
                    )
        
        if binary_feature_exprs:
            # Build SQL using JOINs to lookup tables (completely avoids SQL injection)
            # Use a CTE to match codes, then pivot to binary features
            sql_parts = ["CAST(mi_person_key AS VARCHAR) AS mi_person_key"]
            
            # Build UNION of all matched codes
            union_parts = []
            
            if drug_codes:
                union_parts.append("""
                    SELECT DISTINCT
                        CAST(e.mi_person_key AS VARCHAR) AS mi_person_key,
                        l.feature_name
                    FROM events_view e
                    INNER JOIN drug_code_lookup l ON CAST(e.drug_name AS VARCHAR) = l.code_value
                """)
            
            if icd_codes:
                icd_cols_list = [c for c in available_cols if 'icd_diagnosis_code' in c.lower()]
                if icd_cols_list:
                    icd_conditions = " OR ".join([f"CAST(e.{col} AS VARCHAR) = l.code_value" for col in icd_cols_list])
                    union_parts.append(f"""
                        SELECT DISTINCT
                            CAST(e.mi_person_key AS VARCHAR) AS mi_person_key,
                            l.feature_name
                        FROM events_view e
                        INNER JOIN icd_code_lookup l ON ({icd_conditions})
                    """)
            
            if cpt_codes:
                union_parts.append("""
                    SELECT DISTINCT
                        CAST(e.mi_person_key AS VARCHAR) AS mi_person_key,
                        l.feature_name
                    FROM events_view e
                    INNER JOIN cpt_code_lookup l ON CAST(e.procedure_code AS VARCHAR) = l.code_value
                """)
            
            if union_parts:
                # Process features in batches to avoid SQL query size limits and floating point exceptions
                # With 11,058 features, a single query with 11,058 CASE WHEN expressions is too large
                all_features = binary_feature_exprs  # These are the feature names we stored above
                batch_size = 500  # Process 500 features at a time
                n_batches = (len(all_features) + batch_size - 1) // batch_size
                
                print(f"Processing {len(all_features)} binary features in {n_batches} batches of {batch_size}...")
                
                # Process features in batches
                all_binary_feats_dfs = []
                for batch_idx in range(n_batches):
                    start_idx = batch_idx * batch_size
                    end_idx = min(start_idx + batch_size, len(all_features))
                    batch_features = all_features[start_idx:end_idx]
                    
                    print(f"  Processing batch {batch_idx + 1}/{n_batches} ({len(batch_features)} features)...")
                    
                    # Build feature expressions for this batch
                    feature_exprs = []
                    for feature_name in batch_features:
                        # Escape feature_name for SQL (it's safe since we created it, but be defensive)
                        feature_name_escaped = feature_name.replace("'", "''")
                        feature_exprs.append(
                            f"MAX(CASE WHEN matched.feature_name = '{feature_name_escaped}' THEN 1 ELSE 0 END) AS {feature_name_escaped}"
                        )
                    
                    sql = f"""
                        WITH matched AS (
                            {' UNION ALL '.join(union_parts)}
                        )
                        SELECT
                            CAST(e.mi_person_key AS VARCHAR) AS mi_person_key,
                            {', '.join(feature_exprs)}
                        FROM events_view e
                        LEFT JOIN matched ON CAST(e.mi_person_key AS VARCHAR) = matched.mi_person_key
                        GROUP BY e.mi_person_key
                    """
                    
                    try:
                        batch_df = con.execute(sql).df()
                        batch_df["mi_person_key"] = batch_df["mi_person_key"].astype(str)
                        
                        # Fill NaN values with 0
                        batch_feat_cols = [col for col in batch_df.columns if col != "mi_person_key"]
                        batch_df[batch_feat_cols] = batch_df[batch_feat_cols].fillna(0).astype(int)
                        
                        all_binary_feats_dfs.append(batch_df)
                    except Exception as sql_error:
                        print(f"[ERROR] Failed to create binary features for batch {batch_idx + 1}: {sql_error}")
                        import traceback
                        traceback.print_exc()
                        raise  # Re-raise to fail the step
                
                # Merge all batches together
                print(f"Merging {len(all_binary_feats_dfs)} batches...")
                binary_feats_df = grouped[["mi_person_key"]].copy()  # Start with patient keys
                
                for batch_df in all_binary_feats_dfs:
                    binary_feats_df = binary_feats_df.merge(
                        batch_df, on="mi_person_key", how="left"
                    )
                
                # Fill NaN values in merged binary features with 0
                binary_feat_cols = [col for col in binary_feats_df.columns if col != "mi_person_key"]
                binary_feats_df[binary_feat_cols] = binary_feats_df[binary_feat_cols].fillna(0).astype(int)
                
                # Merge binary features with grouped data
                grouped = grouped.merge(binary_feats_df, on="mi_person_key", how="left")
                
                # Fill NaN values in merged binary features with 0
                for col in binary_feat_cols:
                    if col in grouped.columns:
                        grouped[col] = grouped[col].fillna(0).astype(int)
                
                print(f"✅ Created {len(binary_feature_exprs)} binary features from aggregated FI codes (processed in {n_batches} batches)")
            else:
                print("[WARNING] No binary features to create")
        else:
            print("[WARNING] No binary features created")
    except Exception as e:
        print(f"[WARNING] Error creating binary features: {e}")
        import traceback
        traceback.print_exc()
    finally:
        con.close()

    # ------------------------------------------------------------------
    # PGx Feature Table (from Step 5)
    # ------------------------------------------------------------------
    # Step 5 adds PGx features - load them here for final model training
    # Note: BupaR, DTW, and FP-Growth are now used for dashboard visualizations only
    
    # Check multiple locations for PGx features from Step 5 (canonical: 5_pgx_analysis/outputs)
    pgx_path_candidates = [
        PROJECT_ROOT / "5_pgx_analysis" / "outputs" / "feature_engineering" / f"pgx_added_features_{cohort}_{age_band_fname}.csv",
    ]
    data_root = get_data_root()
    pgx_path_candidates.append(
        data_root / "5_pgx_analysis" / "outputs" / "feature_engineering" / f"pgx_added_features_{cohort}_{age_band_fname}.csv",
    )
    
    pgx_path = None
    for candidate in pgx_path_candidates:
        if candidate.exists():
            pgx_path = candidate
            break
    
    # If not found locally, try downloading from S3
    if pgx_path is None:
        # Primary S3 location: gold/pgx_features/
        s3_key_candidates = [
            f"gold/pgx_features/{cohort}/{age_band}/pgx_added_features_{cohort}_{age_band_fname}.csv",
            # Legacy S3 location: gold/feature_engineering/7_pgx/
            f"gold/feature_engineering/7_pgx/{cohort}/{age_band}/pgx_added_features_{cohort}_{age_band_fname}.csv",
        ]
        
        # Try to download from S3 to primary local location
        download_dest = pgx_path_candidates[0]
        download_dest.parent.mkdir(parents=True, exist_ok=True)
        
        for s3_key in s3_key_candidates:
            try:
                # Check if file exists in S3
                s3_client.head_object(Bucket=S3_BUCKET, Key=s3_key)
                s3_path = f"s3://{S3_BUCKET}/{s3_key}"
                
                print(f"PGx features not found locally. Downloading from S3: {s3_path}")
                s3_client.download_file(S3_BUCKET, s3_key, str(download_dest))
                print(f"✓ Downloaded PGx features to {download_dest}")
                pgx_path = download_dest
                break
            except s3_client.exceptions.ClientError as e:
                if e.response["Error"]["Code"] in ["404", "NoSuchKey"]:
                    continue
                print(f"Warning: Could not check/download PGx features from {s3_key}: {e}")
                continue
            except Exception as e:
                print(f"Warning: Error downloading PGx features from {s3_key}: {e}")
                continue
    
    # Default to primary location if none found (will be checked by _load_feature_table)
    if pgx_path is None:
        pgx_path = pgx_path_candidates[0]

    pgx = _load_feature_table(pgx_path, required=False)

    if "mi_person_key" in pgx.columns:
        pgx["mi_person_key"] = pgx["mi_person_key"].astype(str)

    # Merge aggregated patient-level features with PGx features only
    final = grouped.copy()
    if not pgx.empty:
        print(f"Merging PGx features ({pgx.shape[1] - 1} columns).")
        final = final.merge(pgx, on="mi_person_key", how="left")
    else:
        print(f"No PGx features found for {cohort}, {age_band} (continuing without PGx features).")

    # Drop any patients with missing target
    final = final.dropna(subset=["target"])

    # Apply target-leakage removal rules before returning the feature matrix.
    final = remove_target_leakage_features(final, cohort=cohort, age_band=age_band)
    
    # Validate: Check for duplicate column names (excluding merge key)
    duplicate_cols = final.columns[final.columns.duplicated()].tolist()
    if duplicate_cols:
        raise ValueError(
            f"Duplicate feature columns detected after merging feature tables for {cohort}/{age_band}: {duplicate_cols}. "
            f"This will cause issues in downstream processing. Please ensure each feature table has unique column names."
        )
    
    # Validate: Ensure feature column names are unique
    feature_cols = [c for c in final.columns if c not in ("mi_person_key", "target")]
    if len(feature_cols) != len(set(feature_cols)):
        duplicates = [col for col in feature_cols if feature_cols.count(col) > 1]
        unique_duplicates = list(set(duplicates))
        raise ValueError(
            f"Duplicate feature names detected in final feature matrix for {cohort}/{age_band}: {unique_duplicates}. "
            f"Total features: {len(feature_cols)}, Unique features: {len(set(feature_cols))}. "
            f"This will cause issues in downstream processing."
        )
    
    return final


def _generate_mc_splits(X, y, n_splits: int, test_size: float = 0.3, random_state: int = 1997):
    """Yield (X_train, X_test, y_train, y_test) for each MC split."""
    for split_idx in range(n_splits):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=random_state + split_idx
        )
        yield X_train, X_test, y_train, y_test


def _build_model_from_trial(trial, model_type: str, device: str, nthread: int, cat_feature_indices: list):
    """Build one of XGBClassifier, XGBRFClassifier, or CatBoostClassifier from Optuna trial."""
    import xgboost as xgb  # type: ignore
    if model_type == "xgb":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 600),
            "max_depth": trial.suggest_int("max_depth", 4, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "gamma": trial.suggest_float("gamma", 1e-8, 1.0, log=True),
        }
        clf = xgb.XGBClassifier(
            **params,
            tree_method="hist",
            device=device,
            objective="binary:logistic",
            eval_metric="logloss",
            n_jobs=nthread,
            random_state=RANDOM_STATE,
        )
        return clf
    if model_type == "xgb_rf":
        params = {
            "n_estimators": trial.suggest_int("n_estimators_rf", 200, 600),
            "max_depth": trial.suggest_int("max_depth_rf", 4, 10),
            "subsample": trial.suggest_float("subsample_rf", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree_rf", 0.6, 1.0),
        }
        clf = xgb.XGBRFClassifier(
            **params,
            tree_method="hist",
            device=device,
            objective="binary:logistic",
            eval_metric="logloss",
            n_jobs=nthread,
            random_state=RANDOM_STATE,
        )
        return clf
    # cat
    from catboost import CatBoostClassifier  # type: ignore
    params = {
        "iterations": trial.suggest_int("iterations", 300, 800),
        "learning_rate": trial.suggest_float("learning_rate_cat", 0.02, 0.2, log=True),
        "depth": trial.suggest_int("depth", 4, 10),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
    }
    clf = CatBoostClassifier(
        **params,
        loss_function="Logloss",
        eval_metric="Logloss",
        grow_policy="SymmetricTree",
        random_seed=RANDOM_STATE,
        verbose=False,
        cat_features=cat_feature_indices,
    )
    return clf


def _select_best_trial_from_pareto(study, strategy: str = "auprc", recall_min: float = 0.7):
    """Pick one trial from study.best_trials (Pareto front). strategy: auprc | recall | recall_threshold."""
    best_trials = study.best_trials
    if not best_trials:
        return None
    # values: (mean_recall, mean_auprc) at indices 0, 1
    if strategy == "recall_threshold":
        qualified = [t for t in best_trials if t.values[0] is not None and t.values[0] >= recall_min]
        if qualified:
            return max(qualified, key=lambda t: t.values[1])
        return max(best_trials, key=lambda t: t.values[1])
    if strategy == "recall":
        return max(best_trials, key=lambda t: t.values[0] if t.values and t.values[0] is not None else -1)
    return max(best_trials, key=lambda t: t.values[1] if t.values and len(t.values) > 1 and t.values[1] is not None else -1)


def _default_xgb_params():
    return {
        "n_estimators": 500,
        "max_depth": 6,
        "learning_rate": 0.05,
        "min_child_weight": 1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0,
        "gamma": 1e-8,
    }


def _default_xgb_rf_params():
    return {
        "n_estimators_rf": 500,
        "max_depth_rf": 6,
        "subsample_rf": 0.8,
        "colsample_bytree_rf": 0.8,
    }


def _default_cat_params():
    return {
        "iterations": 500,
        "learning_rate_cat": 0.05,
        "depth": 6,
        "l2_leaf_reg": 3.0,
    }


def _build_model_from_params(params: dict, model_type: str, device: str, nthread: int, cat_feature_indices: list):
    """Build classifier from a params dict (e.g. from trial.params). model_type: xgb | xgb_rf | cat."""
    import xgboost as xgb  # type: ignore
    if model_type == "xgb":
        return xgb.XGBClassifier(
            n_estimators=params.get("n_estimators", 500),
            max_depth=params.get("max_depth", 6),
            learning_rate=params.get("learning_rate", 0.05),
            min_child_weight=params.get("min_child_weight", 1),
            subsample=params.get("subsample", 0.8),
            colsample_bytree=params.get("colsample_bytree", 0.8),
            reg_lambda=params.get("reg_lambda", 1.0),
            gamma=params.get("gamma", 1e-8),
            tree_method="hist",
            device=device,
            objective="binary:logistic",
            eval_metric="logloss",
            n_jobs=nthread,
            random_state=RANDOM_STATE,
        )
    if model_type == "xgb_rf":
        return xgb.XGBRFClassifier(
            n_estimators=params.get("n_estimators_rf", 500),
            max_depth=params.get("max_depth_rf", 6),
            subsample=params.get("subsample_rf", 0.8),
            colsample_bytree=params.get("colsample_bytree_rf", 0.8),
            tree_method="hist",
            device=device,
            objective="binary:logistic",
            eval_metric="logloss",
            n_jobs=nthread,
            random_state=RANDOM_STATE,
        )
    from catboost import CatBoostClassifier  # type: ignore
    return CatBoostClassifier(
        iterations=params.get("iterations", 500),
        learning_rate=params.get("learning_rate_cat", 0.05),
        depth=params.get("depth", 6),
        l2_leaf_reg=params.get("l2_leaf_reg", 3.0),
        loss_function="Logloss",
        eval_metric="Logloss",
        grow_policy="SymmetricTree",
        random_seed=RANDOM_STATE,
        verbose=False,
        cat_features=cat_feature_indices,
    )


def _recompute_selection_from_summary_df(summary_df: pd.DataFrame) -> tuple:
    """From a model_metrics_summary DataFrame, compute selected_model, best_xgb_variant, best_pr_auc, best_recall, selection_reason (same rule: AUC-PR then Recall)."""
    _names = {"xgb": "XGBoost", "xgb_rf": "XGBoost RF", "catboost": "CatBoost", "ensemble": "Ensemble"}
    name_to_internal = {"XGBoost": "xgb", "XGBoost_RF": "xgb_rf", "CatBoost": "catboost", "Ensemble": "ensemble"}
    rows = summary_df[summary_df["model"].isin(name_to_internal.keys())].copy()
    if rows.empty:
        return ("xgb", "xgb", 0.0, 0.0, "No candidates in summary")
    rows = rows.sort_values(by=["pr_auc_mean", "recall_mean"], ascending=[False, False])
    first = rows.iloc[0]
    selected_model = name_to_internal[first["model"]]
    best_pr_auc = float(first["pr_auc_mean"])
    best_recall = float(first["recall_mean"])
    xgb_row = summary_df[summary_df["model"] == "XGBoost"]
    xgb_rf_row = summary_df[summary_df["model"] == "XGBoost_RF"]
    if not xgb_row.empty and not xgb_rf_row.empty:
        xgb_pr = float(xgb_row["pr_auc_mean"].iloc[0])
        xgb_r = float(xgb_row["recall_mean"].iloc[0])
        rf_pr = float(xgb_rf_row["pr_auc_mean"].iloc[0])
        rf_r = float(xgb_rf_row["recall_mean"].iloc[0])
        best_xgb_variant = "xgb" if (xgb_pr > rf_pr or (xgb_pr == rf_pr and xgb_r >= rf_r)) else "xgb_rf"
    else:
        best_xgb_variant = "xgb"
    selection_reason = (
        f"{_names[selected_model]} selected by 25-run MCCV (AUC-PR then Recall): AUC-PR={best_pr_auc:.4f}, Recall={best_recall:.4f}"
    )
    return (selected_model, best_xgb_variant, best_pr_auc, best_recall, selection_reason)


def train_and_evaluate(
    df: pd.DataFrame,
    cohort: str,
    age_band: str,
    n_runs: int | None = None,
    bin_name: str | None = None,
) -> None:
    """
    Train XGBoost (CPU on Linux, GPU on Windows if available) and CatBoost on the assembled feature table,
    optionally using Monte-Carlo CV with `n_runs` stratified train/test splits.

    When n_runs > 1, metrics (AUC, PR-AUC, LogLoss, recall) are aggregated across
    runs for:
      - XGBoost (boosting)
      - CatBoost (if available)
      - Simple ensemble of XGBoost + CatBoost (probability average)

    Idempotent: if model_metrics_summary.csv and final model artifacts already exist,
    only selection is recomputed from the summary (AUC-PR then Recall) and metadata/CSV
    are updated; no retraining.

    bin_name : when provided (e.g. 'low', 'medium', 'high', 'extreme'), outputs are
               written to a per-bin subdirectory:
               outputs/{cohort}/{age_band_fname}/bin_models/{bin_name}/
    """
    # Pre-compute age_band_fname and out_base once so they are available throughout.
    age_band_fname = age_band_to_fname(age_band)
    if bin_name is not None:
        out_base = PROJECT_ROOT / "6_final_model" / "outputs" / cohort / age_band_fname / "bin_models" / bin_name
    else:
        out_base = PROJECT_ROOT / "6_final_model" / "outputs" / cohort / age_band_fname
    # Separate features and label.
    # n_event_bin is a string label used for per-bin filtering; n_event_bin_ordinal is the
    # numeric model feature (0=low, 1=medium, 2=high, 3=extreme).
    _EXCLUDE_FROM_FEATURES = {"mi_person_key", "target", "n_event_bin"}
    feature_cols: List[str] = [
        c for c in df.columns if c not in _EXCLUDE_FROM_FEATURES
    ]
    
    # Validate: Ensure feature column names are unique
    if len(feature_cols) != len(set(feature_cols)):
        duplicates = [col for col in feature_cols if feature_cols.count(col) > 1]
        unique_duplicates = list(set(duplicates))
        raise ValueError(
            f"Duplicate feature names detected in training data for {cohort}/{age_band}: {unique_duplicates}. "
            f"Total features: {len(feature_cols)}, Unique features: {len(set(feature_cols))}. "
            f"This will cause issues in model training and downstream processing."
        )
    
    # Identify categorical features for CatBoost (binary item_* features)
    # CatBoost performs better when binary features are treated as categorical
    categorical_feature_names = [c for c in feature_cols if c.startswith('item_')]
    
    # Keep numeric feature columns for XGBoost (all features including binary)
    # CatBoost can handle both numeric and categorical
    numeric_feature_cols = [
        c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c]) or c in categorical_feature_names
    ]
    
    # Also include non-numeric features that CatBoost can handle as categorical
    # (though we're primarily using binary item_* features)
    for c in feature_cols:
        if c not in numeric_feature_cols and c.startswith('item_'):
            # Binary features should be numeric (0/1), but if they're strings, include them
            numeric_feature_cols.append(c)
    
    if not numeric_feature_cols:
        raise ValueError("No feature columns available for training.")

    if len(numeric_feature_cols) < len(feature_cols):
        dropped = sorted(set(feature_cols) - set(numeric_feature_cols))
        print(
            "Dropping non-numeric feature columns:\n"
            + ", ".join(dropped)
        )

    # Replace inf/-inf with NaN, then fill remaining NaNs with 0 for robustness
    X = df[numeric_feature_cols].replace([float("inf"), float("-inf")], pd.NA)
    X = X.fillna(0)
    y = df["target"].astype(int)
    
    # Get categorical feature indices for CatBoost (indices in X, not in original df)
    # CatBoost performs better when binary features are treated as categorical
    cat_feature_indices = [
        i for i, col in enumerate(numeric_feature_cols) 
        if col in categorical_feature_names
    ]
    
    if cat_feature_indices:
        print(f"Marking {len(cat_feature_indices)} binary features (item_*) as categorical for CatBoost")

    # ------------------------------------------------------------------
    # Class distribution diagnostics
    # ------------------------------------------------------------------
    def _counts(series: pd.Series) -> str:
        vc = series.value_counts().to_dict()
        total = int(series.shape[0])
        parts = []
        for cls in sorted(vc.keys()):
            cnt = int(vc[cls])
            frac = cnt / total if total > 0 else 0.0
            label = "control" if cls == 0 else "target"
            parts.append(f"{label}={cnt} ({frac:.3f})")
        return f"n={total}; " + ", ".join(parts)

    print("\nClass distribution (overall):")
    print("  " + _counts(y))

    # If there is only one class overall, training a classifier is not meaningful.
    if y.nunique() < 2:
        print(
            "\nOnly one class present in the assembled data; "
            "skipping model training for this cohort/age_band."
        )
        return

    # ------------------------------------------------------------------
    # Idempotent selection-only path: if summary CSV and final models exist, just correct selection and exit
    # ------------------------------------------------------------------
    summary_csv_path = out_base / f"{cohort}_{age_band_fname}_model_metrics_summary.csv"
    _s3_bin_infix = f"bin_models/{bin_name}/" if bin_name else ""
    s3_summary_csv = f"s3://pgxdatalake/gold/final_model/{cohort}/{age_band}/{_s3_bin_infix}{cohort}_{age_band_fname}_model_metrics_summary.csv"
    model_json_dir = out_base / "final_model_json"
    xgb_json_path = model_json_dir / f"{cohort}_{age_band_fname}_best_xgboost_model.json"
    cb_cbm_path = model_json_dir / f"{cohort}_{age_band_fname}_best_catboost_model.cbm"
    cb_joblib_path = out_base / "models" / "catboost.joblib"

    def _try_load_summary_csv():
        if summary_csv_path.exists():
            return pd.read_csv(summary_csv_path)
        try:
            from py_helpers.checkpoint_utils import check_s3_output_exists
            if check_s3_output_exists(s3_summary_csv):
                summary_csv_path.parent.mkdir(parents=True, exist_ok=True)
                import subprocess
                subprocess.run(["aws", "s3", "cp", s3_summary_csv, str(summary_csv_path)], check=True)
                return pd.read_csv(summary_csv_path)
        except Exception:
            pass
        return None

    existing_summary = _try_load_summary_csv()
    has_catboost_in_summary = existing_summary is not None and "CatBoost" in existing_summary["model"].values
    has_catboost_artifact = cb_cbm_path.exists() or cb_joblib_path.exists()
    skip_retrain = (
        existing_summary is not None
        and xgb_json_path.exists()
        and (has_catboost_artifact if has_catboost_in_summary else True)
    )
    if skip_retrain:
        selected_model, best_xgb_variant, best_pr_auc, best_recall, selection_reason = _recompute_selection_from_summary_df(existing_summary)
        _names = {"xgb": "XGBoost", "xgb_rf": "XGBoost RF", "catboost": "CatBoost", "ensemble": "Ensemble"}
        # Update "selected" column in summary to match
        def _selected_for_row(row):
            return row["model"] == _names.get(selected_model, selected_model)
        existing_summary["selected"] = existing_summary.apply(_selected_for_row, axis=1)
        existing_summary.to_csv(summary_csv_path, index=False)
        try:
            from py_helpers.checkpoint_utils import upload_file_to_s3
            upload_file_to_s3(summary_csv_path, s3_summary_csv, check_exists=False)
        except Exception:
            pass
        metadata_path = out_base / f"{cohort}_{age_band_fname}_model_selection_metadata.json"
        s3_metadata = f"s3://pgxdatalake/gold/final_model/{cohort}/{age_band}/{_s3_bin_infix}{cohort}_{age_band_fname}_model_selection_metadata.json"
        selection_metadata = {
            "selected_model": selected_model,
            "best_xgb_variant": best_xgb_variant,
            "best_pr_auc": best_pr_auc,
            "best_recall": best_recall,
            "selection_reason": selection_reason,
        }
        if metadata_path.exists():
            with open(metadata_path) as f:
                existing_meta = json.load(f)
            selection_metadata["xgb_recall_mean"] = existing_meta.get("xgb_recall_mean")
            selection_metadata["xgb_pr_auc_mean"] = existing_meta.get("xgb_pr_auc_mean")
            selection_metadata["xgb_rf_recall_mean"] = existing_meta.get("xgb_rf_recall_mean")
            selection_metadata["xgb_rf_pr_auc_mean"] = existing_meta.get("xgb_rf_pr_auc_mean")
            selection_metadata["optuna_used"] = existing_meta.get("optuna_used", False)
            if existing_meta.get("optuna_best_params") is not None:
                selection_metadata["optuna_best_params"] = existing_meta["optuna_best_params"]
            for k in ("catboost_recall_mean", "catboost_pr_auc_mean", "catboost_auc_mean", "catboost_logloss_mean"):
                if k in existing_meta:
                    selection_metadata[k] = existing_meta[k]
        with open(metadata_path, "w") as f:
            json.dump(selection_metadata, f, indent=2)
        try:
            from py_helpers.checkpoint_utils import upload_file_to_s3
            upload_file_to_s3(metadata_path, s3_metadata, check_exists=False)
        except Exception:
            pass
        print(f"\n[IDEMPOTENT] Existing models found; selection corrected from summary (no retrain).")
        print(f"Selected: {_names.get(selected_model, selected_model).upper()} (AUC-PR={best_pr_auc:.4f}, Recall={best_recall:.4f})")
        return

    # Prepare containers for MC metrics
    # Track XGBoost and XGBoost RF separately for model selection
    model_names = ["xgb", "xgb_rf", "catboost", "ensemble"]
    metrics = {
        m: {"auc": [], "pr_auc": [], "logloss": [], "recall": []} for m in model_names
    }

    # ------------------------------------------------------------------
    # Model selection: prefer GPU XGBoost if available
    # ------------------------------------------------------------------
    use_xgb = False
    xgb_import_error = None
    try:
        import xgboost as xgb  # type: ignore

        use_xgb = True
    except Exception as e:
        use_xgb = False
        xgb_import_error = str(e)

    try:
        from catboost import CatBoostClassifier  # type: ignore

        have_catboost = True
        print(f"[INFO] CatBoost is available - will run for all {n_runs} MC CV splits")
    except Exception:
        have_catboost = False
        print(f"[INFO] CatBoost not available - only running XGBoost models for {n_runs} MC CV splits")
        print(f"[INFO] To install CatBoost: pip install catboost")

    if not use_xgb:
        error_msg = "XGBoost is required for the final model."
        if xgb_import_error:
            error_msg += f" Import error: {xgb_import_error}"
        error_msg += "\n\nTo install XGBoost, run: pip install xgboost"
        raise ImportError(error_msg)

    from py_helpers.env_utils import get_cpu_cores  # local import to avoid cycles
    nthread = get_cpu_cores()
    device = "cpu" if is_linux() else "cuda"

    print(f"\n[INFO] Starting Monte-Carlo CV with {n_runs} splits (n_jobs={nthread} cores)")
    print(f"[INFO] Models to run: XGBoost, XGBoost RF" + (", CatBoost" if have_catboost else ""))

    last_run_artifacts = {}
    optuna_used = False
    optuna_best_params = None
    optuna_selected_model = None  # model type Optuna chose (used for which full-data refit gets Optuna params)

    # OOF probability accumulators for Platt scaling (logistic regression on OOF probas vs actuals).
    # Each MC test fold is out-of-fold: the model predicting it was never trained on those rows.
    # Concatenated OOF predictions = the ideal data for fitting a second-stage calibrator.
    oof_preds: dict = {
        "xgb":      {"y_proba": [], "y_true": []},
        "xgb_rf":   {"y_proba": [], "y_true": []},
        "catboost": {"y_proba": [], "y_true": []},
    }
    selected_model = None
    best_xgb_variant = None
    selection_reason = ""
    best_pr_auc = 0.0
    best_recall = 0.0

    try:
        import optuna  # type: ignore
        optuna_available = True
    except Exception:
        optuna_available = False

    if optuna_available and n_runs:
        model_types = ["xgb", "xgb_rf"]
        if have_catboost:
            model_types.append("cat")
        hpo_splits = list(_generate_mc_splits(X, y, N_MCCV_HPO, 0.3, RANDOM_STATE))
        print(f"[INFO] Optuna HPO: {N_OPTUNA_TRIALS} trials, {N_MCCV_HPO} splits per trial (Recall + AUC-PR)")

        def _optuna_objective(trial):
            model_type = trial.suggest_categorical("model_type", model_types)
            recalls, auprcs = [], []
            for (X_tr, X_te, y_tr, y_te) in hpo_splits:
                clf = _build_model_from_trial(trial, model_type, device, nthread, cat_feature_indices)
                try:
                    clf.fit(X_tr, y_tr)
                except Exception:
                    if hasattr(clf, "set_params"):
                        clf.set_params(tree_method="hist")
                        if "device" in clf.get_params():
                            clf.set_params(device="cpu")
                    clf.fit(X_tr, y_tr)
                y_proba = clf.predict_proba(X_te)[:, 1]
                recalls.append(recall_score(y_te, (y_proba >= 0.5).astype(int), zero_division=0))
                auprcs.append(average_precision_score(y_te, y_proba))
            mean_recall = float(np.mean(recalls))
            mean_auprc = float(np.mean(auprcs))
            trial.set_user_attr("mean_recall", mean_recall)
            trial.set_user_attr("mean_auprc", mean_auprc)
            return (mean_recall, mean_auprc)

        study = optuna.create_study(directions=["maximize", "maximize"])
        study.optimize(_optuna_objective, n_trials=N_OPTUNA_TRIALS)
        best_trial = _select_best_trial_from_pareto(study, strategy="auprc")
        if best_trial is not None:
            _optuna_model = best_trial.params["model_type"]
            selected_model = "catboost" if _optuna_model == "cat" else _optuna_model
            optuna_selected_model = selected_model  # remember for full-data refit (Optuna params only for this type)
            optuna_best_params = best_trial.params
            optuna_used = True
            if selected_model in ("xgb", "xgb_rf"):
                best_xgb_variant = selected_model
            else:
                xgb_trials = [t for t in study.best_trials if t.params.get("model_type") in ("xgb", "xgb_rf")]
                if xgb_trials:
                    best_xgb_trial = max(xgb_trials, key=lambda t: t.values[1] if t.values and len(t.values) > 1 else -1)
                    best_xgb_variant = best_xgb_trial.params["model_type"]
                else:
                    best_xgb_variant = "xgb"
            # Run full n_runs MCCV with selected model (Optuna params) and others (defaults) to fill metrics
            for run_idx in range(n_runs):
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.3, stratify=y, random_state=RANDOM_STATE + run_idx
                )
                seed = RANDOM_STATE + run_idx
                # XGB
                xgb_params = optuna_best_params if selected_model == "xgb" else _default_xgb_params()
                xgb_clf = _build_model_from_params(xgb_params, "xgb", device, nthread, cat_feature_indices)
                xgb_clf.set_params(random_state=seed)
                try:
                    xgb_clf.fit(X_train, y_train)
                except Exception:
                    xgb_clf.set_params(tree_method="hist", device="cpu")
                    xgb_clf.fit(X_train, y_train)
                y_proba_xgb = xgb_clf.predict_proba(X_test)[:, 1]
                metrics["xgb"]["auc"].append(roc_auc_score(y_test, y_proba_xgb))
                metrics["xgb"]["pr_auc"].append(average_precision_score(y_test, y_proba_xgb))
                metrics["xgb"]["logloss"].append(log_loss(y_test, y_proba_xgb))
                metrics["xgb"]["recall"].append(recall_score(y_test, (y_proba_xgb >= 0.5).astype(int)))
                # XGB RF
                xgb_rf_params = optuna_best_params if selected_model == "xgb_rf" else _default_xgb_rf_params()
                xgb_rf_clf = _build_model_from_params(xgb_rf_params, "xgb_rf", device, nthread, cat_feature_indices)
                xgb_rf_clf.set_params(random_state=seed + 1000)
                try:
                    xgb_rf_clf.fit(X_train, y_train)
                except Exception:
                    xgb_rf_clf.set_params(tree_method="hist", device="cpu")
                    xgb_rf_clf.fit(X_train, y_train)
                y_proba_xgb_rf = xgb_rf_clf.predict_proba(X_test)[:, 1]
                metrics["xgb_rf"]["auc"].append(roc_auc_score(y_test, y_proba_xgb_rf))
                metrics["xgb_rf"]["pr_auc"].append(average_precision_score(y_test, y_proba_xgb_rf))
                metrics["xgb_rf"]["logloss"].append(log_loss(y_test, y_proba_xgb_rf))
                metrics["xgb_rf"]["recall"].append(recall_score(y_test, (y_proba_xgb_rf >= 0.5).astype(int)))
                y_proba_cb = None
                if have_catboost:
                    cb_params = optuna_best_params if selected_model == "cat" else _default_cat_params()
                    cb_clf = _build_model_from_params(cb_params, "cat", device, nthread, cat_feature_indices)
                    cb_train_dir = PROJECT_ROOT / "6_final_model" / "outputs" / cohort / age_band_fname / "catboost_info"
                    cb_train_dir.mkdir(parents=True, exist_ok=True)
                    cb_clf.set_params(train_dir=str(cb_train_dir))
                    try:
                        cb_clf.fit(X_train, y_train)
                        y_proba_cb = cb_clf.predict_proba(X_test)[:, 1]
                        metrics["catboost"]["auc"].append(roc_auc_score(y_test, y_proba_cb))
                        metrics["catboost"]["pr_auc"].append(average_precision_score(y_test, y_proba_cb))
                        metrics["catboost"]["logloss"].append(log_loss(y_test, y_proba_cb))
                        metrics["catboost"]["recall"].append(recall_score(y_test, (y_proba_cb >= 0.5).astype(int)))
                    except Exception as e:
                        print(f"[WARN] CatBoost run {run_idx + 1} failed: {e}")
                # Ensemble
                if y_proba_cb is not None:
                    y_proba_xgb_best = y_proba_xgb if metrics["xgb"]["recall"][-1] >= metrics["xgb_rf"]["recall"][-1] else y_proba_xgb_rf
                    y_proba_ens = 0.5 * y_proba_xgb_best + 0.5 * y_proba_cb
                    metrics["ensemble"]["auc"].append(roc_auc_score(y_test, y_proba_ens))
                    metrics["ensemble"]["pr_auc"].append(average_precision_score(y_test, y_proba_ens))
                    metrics["ensemble"]["logloss"].append(log_loss(y_test, y_proba_ens))
                    metrics["ensemble"]["recall"].append(recall_score(y_test, (y_proba_ens >= 0.5).astype(int)))
                else:
                    if metrics["xgb"]["recall"][-1] >= metrics["xgb_rf"]["recall"][-1]:
                        metrics["ensemble"]["auc"].append(metrics["xgb"]["auc"][-1])
                        metrics["ensemble"]["pr_auc"].append(metrics["xgb"]["pr_auc"][-1])
                        metrics["ensemble"]["logloss"].append(metrics["xgb"]["logloss"][-1])
                        metrics["ensemble"]["recall"].append(metrics["xgb"]["recall"][-1])
                    else:
                        metrics["ensemble"]["auc"].append(metrics["xgb_rf"]["auc"][-1])
                        metrics["ensemble"]["pr_auc"].append(metrics["xgb_rf"]["pr_auc"][-1])
                        metrics["ensemble"]["logloss"].append(metrics["xgb_rf"]["logloss"][-1])
                        metrics["ensemble"]["recall"].append(metrics["xgb_rf"]["recall"][-1])
                # Accumulate OOF predictions for Platt calibration
                oof_preds["xgb"]["y_proba"].append(y_proba_xgb)
                oof_preds["xgb"]["y_true"].append(y_test.values if hasattr(y_test, "values") else y_test)
                oof_preds["xgb_rf"]["y_proba"].append(y_proba_xgb_rf)
                oof_preds["xgb_rf"]["y_true"].append(y_test.values if hasattr(y_test, "values") else y_test)
                if y_proba_cb is not None:
                    oof_preds["catboost"]["y_proba"].append(y_proba_cb)
                    oof_preds["catboost"]["y_true"].append(y_test.values if hasattr(y_test, "values") else y_test)
                if (run_idx + 1) % 5 == 0 or run_idx == 0:
                    print(f"[MC {run_idx + 1}/{n_runs}] Optuna 25-split MCCV in progress...")
            # Selection will be overwritten below from 25-run MCCV means so CSV and selected agree

    if not optuna_used:
        # Legacy path: fixed hyperparameters, select by AUC-PR then Recall
        for run_idx in range(n_runs):
            # MC split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, stratify=y, random_state=42 + run_idx
            )

            print(f"\n[MC {run_idx + 1}/{n_runs}] Class distribution (train):")
            print("  " + _counts(y_train))
            print(f"[MC {run_idx + 1}/{n_runs}] Class distribution (test):")
            print("  " + _counts(y_test))

            models_to_train = ["XGBoost", "XGBoost RF"]
            if have_catboost:
                models_to_train.append("CatBoost")
            print(
                f"\n[MC {run_idx + 1}/{n_runs}] Training {', '.join(models_to_train)} for "
                f"cohort={cohort}, age_band={age_band} with "
                f"{X_train.shape[0]} train and {X_test.shape[0]} test rows, "
                f"{X_train.shape[1]} numeric features."
            )

            # Train XGBoost (boosting)
            xgb_clf = xgb.XGBClassifier(
                n_estimators=500,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                tree_method="hist",
                device=device,
                objective="binary:logistic",
                eval_metric="logloss",
                n_jobs=nthread,
                random_state=42 + run_idx,
            )

            try:
                xgb_clf.fit(X_train, y_train)
            except Exception:
                # Fallback to CPU if CUDA fails (shouldn't happen on Linux)
                print(
                    "\nXGBoost CUDA device not available; "
                    "falling back to CPU hist tree_method."
                )
                xgb_clf.set_params(tree_method="hist")
                if "device" in xgb_clf.get_params():
                    xgb_clf.set_params(device="cpu")
                xgb_clf.fit(X_train, y_train)

            # XGBoost metrics
            y_proba_xgb = xgb_clf.predict_proba(X_test)[:, 1]
            y_pred_xgb = (y_proba_xgb >= 0.5).astype(int)
            metrics["xgb"]["auc"].append(roc_auc_score(y_test, y_proba_xgb))
            metrics["xgb"]["pr_auc"].append(average_precision_score(y_test, y_proba_xgb))
            metrics["xgb"]["logloss"].append(log_loss(y_test, y_proba_xgb))
            metrics["xgb"]["recall"].append(recall_score(y_test, y_pred_xgb))

            # Train XGBoost RF (random forest)
            xgb_rf_clf = xgb.XGBRFClassifier(
                n_estimators=500,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                tree_method="hist",
                device=device,
                objective="binary:logistic",
                eval_metric="logloss",
                n_jobs=nthread,
                random_state=42 + run_idx + 1000,  # Different seed for RF
            )

            try:
                xgb_rf_clf.fit(X_train, y_train)
            except Exception:
                # Fallback to CPU if CUDA fails
                print(
                    "\nXGBoost RF CUDA device not available; "
                    "falling back to CPU hist tree_method."
                )
                xgb_rf_clf.set_params(tree_method="hist")
                if "device" in xgb_rf_clf.get_params():
                    xgb_rf_clf.set_params(device="cpu")
                xgb_rf_clf.fit(X_train, y_train)

            # XGBoost RF metrics
            y_proba_xgb_rf = xgb_rf_clf.predict_proba(X_test)[:, 1]
            y_pred_xgb_rf = (y_proba_xgb_rf >= 0.5).astype(int)
            metrics["xgb_rf"]["auc"].append(roc_auc_score(y_test, y_proba_xgb_rf))
            metrics["xgb_rf"]["pr_auc"].append(average_precision_score(y_test, y_proba_xgb_rf))
            metrics["xgb_rf"]["logloss"].append(log_loss(y_test, y_proba_xgb_rf))
            metrics["xgb_rf"]["recall"].append(recall_score(y_test, y_pred_xgb_rf))

            y_proba_cb = None
            if have_catboost:
                # Scope CatBoost's internal training artifacts (catboost_info) to a
                # cohort/age-band specific directory under 6_final_model outputs,
                # instead of writing to the project root.
                cb_train_dir = (
                    PROJECT_ROOT
                    / "6_final_model"
                    / "outputs"
                    / cohort
                    / age_band_fname
                    / "catboost_info"
                )
                cb_train_dir.mkdir(parents=True, exist_ok=True)

                cb_clf = CatBoostClassifier(
                    iterations=500,
                    learning_rate=0.05,
                    depth=6,
                    loss_function="Logloss",
                    eval_metric="Logloss",
                    grow_policy="SymmetricTree",  # enforce oblivious trees
                    random_seed=42 + run_idx,
                    verbose=False,
                    train_dir=str(cb_train_dir),
                    cat_features=cat_feature_indices,  # Mark binary features as categorical for better performance
                )
                try:
                    cb_clf.fit(X_train, y_train)
                    y_proba_cb = cb_clf.predict_proba(X_test)[:, 1]
                    y_pred_cb = (y_proba_cb >= 0.5).astype(int)
                    metrics["catboost"]["auc"].append(
                        roc_auc_score(y_test, y_proba_cb)
                    )
                    metrics["catboost"]["pr_auc"].append(
                        average_precision_score(y_test, y_proba_cb)
                    )
                    metrics["catboost"]["logloss"].append(
                        log_loss(y_test, y_proba_cb)
                    )
                    metrics["catboost"]["recall"].append(
                        recall_score(y_test, y_pred_cb)
                    )
                except Exception as e:
                    print(f"\nCatBoost training failed in run {run_idx + 1}; skipping. {e}")

            # Ensemble: Use best XGBoost variant (will be selected after MC-CV) + CatBoost
            # For now, use XGBoost (will be replaced by best variant after selection)
            if y_proba_cb is not None:
                # Use best performing XGBoost variant for ensemble
                # Compare XGBoost vs XGBoost RF for this run
                if metrics["xgb"]["recall"][-1] >= metrics["xgb_rf"]["recall"][-1]:
                    y_proba_xgb_best = y_proba_xgb
                else:
                    y_proba_xgb_best = y_proba_xgb_rf

                y_proba_ens = 0.5 * y_proba_xgb_best + 0.5 * y_proba_cb
                y_pred_ens = (y_proba_ens >= 0.5).astype(int)
                metrics["ensemble"]["auc"].append(roc_auc_score(y_test, y_proba_ens))
                metrics["ensemble"]["pr_auc"].append(
                    average_precision_score(y_test, y_proba_ens)
                )
                metrics["ensemble"]["logloss"].append(log_loss(y_test, y_proba_ens))
                metrics["ensemble"]["recall"].append(recall_score(y_test, y_pred_ens))
            else:
                # Mirror best XGBoost variant metrics when ensemble is unavailable
                if metrics["xgb"]["recall"][-1] >= metrics["xgb_rf"]["recall"][-1]:
                    metrics["ensemble"]["auc"].append(metrics["xgb"]["auc"][-1])
                    metrics["ensemble"]["pr_auc"].append(metrics["xgb"]["pr_auc"][-1])
                    metrics["ensemble"]["logloss"].append(metrics["xgb"]["logloss"][-1])
                    metrics["ensemble"]["recall"].append(metrics["xgb"]["recall"][-1])
                else:
                    metrics["ensemble"]["auc"].append(metrics["xgb_rf"]["auc"][-1])
                    metrics["ensemble"]["pr_auc"].append(metrics["xgb_rf"]["pr_auc"][-1])
                    metrics["ensemble"]["logloss"].append(metrics["xgb_rf"]["logloss"][-1])
                    metrics["ensemble"]["recall"].append(metrics["xgb_rf"]["recall"][-1])

            # Print metrics for all models
            print(
                f"[MC {run_idx + 1}/{n_runs}] "
                f"XGB AUC={metrics['xgb']['auc'][-1]:.4f}, PR-AUC={metrics['xgb']['pr_auc'][-1]:.4f}, "
                f"Recall={metrics['xgb']['recall'][-1]:.4f} | "
                f"XGB-RF AUC={metrics['xgb_rf']['auc'][-1]:.4f}, PR-AUC={metrics['xgb_rf']['pr_auc'][-1]:.4f}, "
                f"Recall={metrics['xgb_rf']['recall'][-1]:.4f}",
                end=""
            )

            # Add CatBoost metrics if available
            if have_catboost and metrics.get("catboost") and metrics["catboost"].get("auc") and len(metrics["catboost"]["auc"]) > 0:
                print(
                    f" | CatBoost AUC={metrics['catboost']['auc'][-1]:.4f}, PR-AUC={metrics['catboost']['pr_auc'][-1]:.4f}, "
                    f"Recall={metrics['catboost']['recall'][-1]:.4f}"
                )
            else:
                print()  # Newline if CatBoost not available

            # Accumulate OOF predictions for Platt calibration
            oof_preds["xgb"]["y_proba"].append(y_proba_xgb)
            oof_preds["xgb"]["y_true"].append(y_test.values if hasattr(y_test, "values") else y_test)
            oof_preds["xgb_rf"]["y_proba"].append(y_proba_xgb_rf)
            oof_preds["xgb_rf"]["y_true"].append(y_test.values if hasattr(y_test, "values") else y_test)
            if y_proba_cb is not None:
                oof_preds["catboost"]["y_proba"].append(y_proba_cb)
                oof_preds["catboost"]["y_true"].append(y_test.values if hasattr(y_test, "values") else y_test)

            # Save artifacts from last run for detailed reporting and importances
            if run_idx == n_runs - 1:
                last_run_artifacts = {
                    "xgb_clf": xgb_clf,
                    "xgb_rf_clf": xgb_rf_clf,
                    "X_train": X_train,
                    "X_test": X_test,
                    "y_train": y_train,
                    "y_test": y_test,
                    "y_pred_xgb": y_pred_xgb,
                    "y_proba_xgb": y_proba_xgb,
                    "y_proba_xgb_rf": y_proba_xgb_rf,
                    "y_proba_cb": y_proba_cb,
                }

    # Aggregate metrics across runs
    print("\n=== Monte-Carlo CV Summary (n_runs={}) ===".format(n_runs))
    for model_key in ["xgb", "xgb_rf", "catboost", "ensemble"]:
        vals = metrics[model_key]
        if not vals["auc"]:
            continue
        print(f"\nModel: {model_key}")
        print(
            "  AUC:     mean={:.4f} std={:.4f}".format(
                float(np.mean(vals["auc"])), float(np.std(vals["auc"], ddof=0))
            )
        )
        print(
            "  PR-AUC:  mean={:.4f} std={:.4f}".format(
                float(np.mean(vals["pr_auc"])),
                float(np.std(vals["pr_auc"], ddof=0)),
            )
        )
        print(
            "  LogLoss: mean={:.4f} std={:.4f}".format(
                float(np.mean(vals["logloss"])),
                float(np.std(vals["logloss"], ddof=0)),
            )
        )
        print(
            "  Recall:  mean={:.4f} std={:.4f}".format(
                float(np.mean(vals["recall"])),
                float(np.std(vals["recall"], ddof=0)),
            )
        )

    # Detailed reports from last run (Best XGBoost variant, CatBoost, Ensemble)
    if last_run_artifacts:
        xgb_clf = last_run_artifacts["xgb_clf"]
        xgb_rf_clf = last_run_artifacts.get("xgb_rf_clf")
        X_train = last_run_artifacts["X_train"]
        X_test = last_run_artifacts["X_test"]
        y_train = last_run_artifacts["y_train"]
        y_test = last_run_artifacts["y_test"]
        y_pred_xgb = last_run_artifacts["y_pred_xgb"]
        y_proba_xgb = last_run_artifacts["y_proba_xgb"]
        y_proba_xgb_rf = last_run_artifacts.get("y_proba_xgb_rf")
        y_proba_cb = last_run_artifacts["y_proba_cb"]

        # Determine which XGBoost variant performed better in last run
        xgb_last_recall = metrics["xgb"]["recall"][-1] if metrics["xgb"]["recall"] else 0.0
        xgb_rf_last_recall = metrics["xgb_rf"]["recall"][-1] if metrics["xgb_rf"]["recall"] else 0.0
        
        if xgb_last_recall >= xgb_rf_last_recall:
            best_y_pred_last = y_pred_xgb
            best_y_proba_last = y_proba_xgb
            best_variant_name = "XGBoost"
        else:
            best_y_pred_last = (y_proba_xgb_rf >= 0.5).astype(int) if y_proba_xgb_rf is not None else y_pred_xgb
            best_y_proba_last = y_proba_xgb_rf if y_proba_xgb_rf is not None else y_proba_xgb
            best_variant_name = "XGBoost RF"

        print(f"\n=== Detailed metrics from last MC run (Best: {best_variant_name}) ===")
        print(f"\nClassification report ({best_variant_name}):")
        print(classification_report(y_test, best_y_pred_last, digits=3))

        if y_proba_cb is not None:
            from catboost import CatBoostClassifier  # type: ignore

            # Recompute CatBoost metrics for last run (already trained)
            y_proba_cb = y_proba_cb
            y_pred_cb = (y_proba_cb >= 0.5).astype(int)
            cb_auc = roc_auc_score(y_test, y_proba_cb)
            cb_pr_auc = average_precision_score(y_test, y_proba_cb)
            cb_ll = log_loss(y_test, y_proba_cb)

            print("\n=== Detailed metrics from last MC run (CatBoost) ===")
            print(f"AUC:     {cb_auc:.4f}")
            print(f"PR-AUC:  {cb_pr_auc:.4f}")
            print(f"LogLoss: {cb_ll:.4f}")
            print("\nClassification report (CatBoost):")
            print(classification_report(y_test, y_pred_cb, digits=3))

            # Use best XGBoost variant for ensemble
            y_proba_ens = 0.5 * best_y_proba_last + 0.5 * y_proba_cb
            y_pred_ens = (y_proba_ens >= 0.5).astype(int)
            ens_auc = roc_auc_score(y_test, y_proba_ens)
            ens_pr_auc = average_precision_score(y_test, y_proba_ens)
            ens_ll = log_loss(y_test, y_proba_ens)

            print("\n=== Detailed metrics from last MC run (Ensemble) ===")
            print(f"AUC:     {ens_auc:.4f}")
            print(f"PR-AUC:  {ens_pr_auc:.4f}")
            print(f"LogLoss: {ens_ll:.4f}")
            print("\nClassification report (Ensemble):")
            print(classification_report(y_test, y_pred_ens, digits=3))

    # ------------------------------------------------------------------
    # Model Selection: XGBoost, XGBoost RF, CatBoost — Primary = AUC-PR, Tie-break = Recall
    # ------------------------------------------------------------------
    print("\n=== Model Selection (Recall and AUC-PR) ===")
    
    # Calculate mean metrics across MC runs
    xgb_recall_mean = float(np.mean(metrics["xgb"]["recall"])) if metrics["xgb"]["recall"] else 0.0
    xgb_pr_auc_mean = float(np.mean(metrics["xgb"]["pr_auc"])) if metrics["xgb"]["pr_auc"] else 0.0
    xgb_rf_recall_mean = float(np.mean(metrics["xgb_rf"]["recall"])) if metrics["xgb_rf"]["recall"] else 0.0
    xgb_rf_pr_auc_mean = float(np.mean(metrics["xgb_rf"]["pr_auc"])) if metrics["xgb_rf"]["pr_auc"] else 0.0
    
    # Calculate CatBoost mean metrics (if available)
    cb_recall_mean = None
    cb_pr_auc_mean = None
    cb_auc_mean = None
    cb_logloss_mean = None
    if have_catboost and metrics.get("catboost") and metrics["catboost"].get("recall"):
        cb_recall_mean = float(np.mean(metrics["catboost"]["recall"]))
        cb_pr_auc_mean = float(np.mean(metrics["catboost"]["pr_auc"]))
        cb_auc_mean = float(np.mean(metrics["catboost"]["auc"]))
        cb_logloss_mean = float(np.mean(metrics["catboost"]["logloss"]))
    
    print(f"XGBoost:      Recall={xgb_recall_mean:.4f}, AUC-PR={xgb_pr_auc_mean:.4f}")
    print(f"XGBoost RF:   Recall={xgb_rf_recall_mean:.4f}, AUC-PR={xgb_rf_pr_auc_mean:.4f}")
    if cb_recall_mean is not None:
        print(f"CatBoost:     Recall={cb_recall_mean:.4f}, AUC-PR={cb_pr_auc_mean:.4f}, AUC={cb_auc_mean:.4f}, LogLoss={cb_logloss_mean:.4f}")
    
    # Select best model from 25-run MCCV means (AUC-PR then Recall) so CSV "selected" column matches
    _names = {"xgb": "XGBoost", "xgb_rf": "XGBoost RF", "catboost": "CatBoost", "ensemble": "Ensemble"}
    candidates = [
        ("xgb", xgb_pr_auc_mean, xgb_recall_mean),
        ("xgb_rf", xgb_rf_pr_auc_mean, xgb_rf_recall_mean),
    ]
    if cb_pr_auc_mean is not None and cb_recall_mean is not None:
        candidates.append(("catboost", cb_pr_auc_mean, cb_recall_mean))
    if metrics.get("ensemble") and metrics["ensemble"].get("pr_auc"):
        ens_pr_auc = float(np.mean(metrics["ensemble"]["pr_auc"]))
        ens_recall = float(np.mean(metrics["ensemble"]["recall"]))
        candidates.append(("ensemble", ens_pr_auc, ens_recall))
    candidates.sort(key=lambda t: (-t[1], -t[2]))
    selected_model = candidates[0][0]
    best_pr_auc = candidates[0][1]
    best_recall = candidates[0][2]
    if xgb_pr_auc_mean > xgb_rf_pr_auc_mean or (xgb_pr_auc_mean == xgb_rf_pr_auc_mean and xgb_recall_mean >= xgb_rf_recall_mean):
        best_xgb_variant = "xgb"
    else:
        best_xgb_variant = "xgb_rf"
    selection_reason = (
        f"{_names[selected_model]} selected by 25-run MCCV (AUC-PR then Recall): AUC-PR={best_pr_auc:.4f}, Recall={best_recall:.4f}"
    )
    if optuna_used and optuna_selected_model is not None and optuna_selected_model != selected_model:
        selection_reason += f" [Optuna had chosen {_names.get(optuna_selected_model, optuna_selected_model)}]"
    print(f"\nSelected: {_names[selected_model].upper()}")
    print(f"Reason: {selection_reason}")
    # Best XGBoost variant is always trained and saved for FFA rule analysis (even when CatBoost is selected)
    print(f"Best XGBoost variant for FFA: {best_xgb_variant.upper()}")
    
    # Save model selection metadata
    out_base.mkdir(parents=True, exist_ok=True)
    
    selection_metadata = {
        "selected_model": selected_model,
        "best_xgb_variant": best_xgb_variant,
        "xgb_recall_mean": xgb_recall_mean,
        "xgb_pr_auc_mean": xgb_pr_auc_mean,
        "xgb_rf_recall_mean": xgb_rf_recall_mean,
        "xgb_rf_pr_auc_mean": xgb_rf_pr_auc_mean,
        "selection_reason": selection_reason,
    }
    
    # Add CatBoost metrics if available
    if cb_recall_mean is not None:
        selection_metadata.update({
            "catboost_recall_mean": cb_recall_mean,
            "catboost_pr_auc_mean": cb_pr_auc_mean,
            "catboost_auc_mean": cb_auc_mean,
            "catboost_logloss_mean": cb_logloss_mean,
        })
    selection_metadata["best_pr_auc"] = best_pr_auc
    selection_metadata["best_recall"] = best_recall
    if optuna_used and optuna_best_params is not None:
        selection_metadata["optuna_used"] = True
        selection_metadata["optuna_best_params"] = optuna_best_params
    else:
        selection_metadata["optuna_used"] = False

    metadata_path = out_base / f"{cohort}_{age_band_fname}_model_selection_metadata.json"
    s3_metadata = f"s3://pgxdatalake/gold/final_model/{cohort}/{age_band}/{_s3_bin_infix}{cohort}_{age_band_fname}_model_selection_metadata.json"
    
    # Helper function for idempotent model saving with S3 upload
    def save_model_idempotent(local_path: Path, s3_path: str, save_func, *save_args, **save_kwargs):
        """Save model file idempotently: check S3 first, then local, then save and upload."""
        try:
            from py_helpers.checkpoint_utils import check_s3_output_exists, upload_file_to_s3
            # Check S3 first
            if check_s3_output_exists(s3_path):
                print(f"[INFO] Model already exists in S3: {s3_path}; skipping save.")
                # Download from S3 if not present locally
                if not local_path.exists():
                    print(f"[INFO] Downloading from S3 to {local_path}...")
                    import subprocess
                    subprocess.run(["aws", "s3", "cp", s3_path, str(local_path)], check=True)
                return False  # Already exists
        except ImportError:
            pass  # Fallback to local-only if checkpoint_utils not available
        
        # Check local file
        if local_path.exists():
            print(f"[INFO] Model already exists locally: {local_path}; skipping save.")
            # Upload to S3 if not present there
            try:
                from py_helpers.checkpoint_utils import upload_file_to_s3
                upload_file_to_s3(local_path, s3_path)
            except ImportError:
                pass
            return False  # Already exists
        
        # Save locally
        local_path.parent.mkdir(parents=True, exist_ok=True)
        save_func()
        print(f"Saved model to {local_path}")
        
        # Upload to S3
        try:
            from py_helpers.checkpoint_utils import upload_file_to_s3
            if upload_file_to_s3(local_path, s3_path):
                print(f"Uploaded to S3: {s3_path}")
        except ImportError:
            pass  # S3 upload is optional
        
        return True  # Newly saved
    
    def save_metadata():
        with open(metadata_path, "w") as f:
            json.dump(selection_metadata, f, indent=2)
    
    save_model_idempotent(metadata_path, s3_metadata, save_metadata)
    print(f"Saved model selection metadata to {metadata_path}")
    
    # Create per-run MC CV results CSV
    mc_cv_results = []
    for run_idx in range(n_runs):
        # XGBoost metrics
        if run_idx < len(metrics["xgb"]["recall"]):
            mc_cv_results.append({
                "split": run_idx + 1,
                "model": "XGBoost",
                "recall": metrics["xgb"]["recall"][run_idx],
                "pr_auc": metrics["xgb"]["pr_auc"][run_idx],
                "auc": metrics["xgb"]["auc"][run_idx],
                "logloss": metrics["xgb"]["logloss"][run_idx],
            })
        
        # XGBoost RF metrics
        if run_idx < len(metrics["xgb_rf"]["recall"]):
            mc_cv_results.append({
                "split": run_idx + 1,
                "model": "XGBoost_RF",
                "recall": metrics["xgb_rf"]["recall"][run_idx],
                "pr_auc": metrics["xgb_rf"]["pr_auc"][run_idx],
                "auc": metrics["xgb_rf"]["auc"][run_idx],
                "logloss": metrics["xgb_rf"]["logloss"][run_idx],
            })
        
        # CatBoost metrics (if available)
        if have_catboost and run_idx < len(metrics.get("catboost", {}).get("recall", [])):
            mc_cv_results.append({
                "split": run_idx + 1,
                "model": "CatBoost",
                "recall": metrics["catboost"]["recall"][run_idx],
                "pr_auc": metrics["catboost"]["pr_auc"][run_idx],
                "auc": metrics["catboost"]["auc"][run_idx],
                "logloss": metrics["catboost"]["logloss"][run_idx],
            })
        
        # Ensemble metrics (if available)
        if run_idx < len(metrics.get("ensemble", {}).get("recall", [])):
            mc_cv_results.append({
                "split": run_idx + 1,
                "model": "Ensemble",
                "recall": metrics["ensemble"]["recall"][run_idx],
                "pr_auc": metrics["ensemble"]["pr_auc"][run_idx],
                "auc": metrics["ensemble"]["auc"][run_idx],
                "logloss": metrics["ensemble"]["logloss"][run_idx],
            })
    
    # Save MC CV results CSV
    mc_cv_csv_path = out_base / f"{cohort}_{age_band_fname}_mc_cv_results.csv"
    s3_mc_cv_csv = f"s3://pgxdatalake/gold/final_model/{cohort}/{age_band}/{_s3_bin_infix}{cohort}_{age_band_fname}_mc_cv_results.csv"
    
    def save_mc_cv_csv():
        mc_cv_df = pd.DataFrame(mc_cv_results)
        mc_cv_df.to_csv(mc_cv_csv_path, index=False)
    
    save_model_idempotent(mc_cv_csv_path, s3_mc_cv_csv, save_mc_cv_csv)
    print(f"\nSaved MC CV per-run results CSV to {mc_cv_csv_path}")
    print(f"  Total rows: {len(mc_cv_results)} (across {n_runs} splits and all models)")
    
    # Create summary CSV with metrics for all models
    summary_data = []
    
    # XGBoost metrics
    summary_data.append({
        "model": "XGBoost",
        "recall_mean": xgb_recall_mean,
        "pr_auc_mean": xgb_pr_auc_mean,
        "auc_mean": float(np.mean(metrics["xgb"]["auc"])) if metrics["xgb"]["auc"] else None,
        "logloss_mean": float(np.mean(metrics["xgb"]["logloss"])) if metrics["xgb"]["logloss"] else None,
        "n_runs": len(metrics["xgb"]["recall"]) if metrics["xgb"]["recall"] else 0,
        "selected": selected_model == "xgb"
    })
    
    # XGBoost RF metrics
    summary_data.append({
        "model": "XGBoost_RF",
        "recall_mean": xgb_rf_recall_mean,
        "pr_auc_mean": xgb_rf_pr_auc_mean,
        "auc_mean": float(np.mean(metrics["xgb_rf"]["auc"])) if metrics["xgb_rf"]["auc"] else None,
        "logloss_mean": float(np.mean(metrics["xgb_rf"]["logloss"])) if metrics["xgb_rf"]["logloss"] else None,
        "n_runs": len(metrics["xgb_rf"]["recall"]) if metrics["xgb_rf"]["recall"] else 0,
        "selected": selected_model == "xgb_rf"
    })
    
    # CatBoost metrics (if available)
    if cb_recall_mean is not None:
        summary_data.append({
            "model": "CatBoost",
            "recall_mean": cb_recall_mean,
            "pr_auc_mean": cb_pr_auc_mean,
            "auc_mean": cb_auc_mean,
            "logloss_mean": cb_logloss_mean,
            "n_runs": len(metrics["catboost"]["recall"]) if metrics.get("catboost") and metrics["catboost"].get("recall") else 0,
            "selected": selected_model == "catboost"
        })
    
    # Ensemble metrics (if available)
    if metrics.get("ensemble") and metrics["ensemble"].get("recall"):
        ensemble_recall_mean = float(np.mean(metrics["ensemble"]["recall"]))
        ensemble_pr_auc_mean = float(np.mean(metrics["ensemble"]["pr_auc"]))
        ensemble_auc_mean = float(np.mean(metrics["ensemble"]["auc"])) if metrics["ensemble"]["auc"] else None
        ensemble_logloss_mean = float(np.mean(metrics["ensemble"]["logloss"])) if metrics["ensemble"]["logloss"] else None
        
        summary_data.append({
            "model": "Ensemble",
            "recall_mean": ensemble_recall_mean,
            "pr_auc_mean": ensemble_pr_auc_mean,
            "auc_mean": ensemble_auc_mean,
            "logloss_mean": ensemble_logloss_mean,
            "n_runs": len(metrics["ensemble"]["recall"]),
            "selected": selected_model == "ensemble"
        })
    
    # Create DataFrame and save to CSV
    summary_df = pd.DataFrame(summary_data)
    summary_csv_path = out_base / f"{cohort}_{age_band_fname}_model_metrics_summary.csv"
    s3_summary_csv = f"s3://pgxdatalake/gold/final_model/{cohort}/{age_band}/{_s3_bin_infix}{cohort}_{age_band_fname}_model_metrics_summary.csv"
    
    def save_summary_csv():
        summary_df.to_csv(summary_csv_path, index=False)
    
    save_model_idempotent(summary_csv_path, s3_summary_csv, save_summary_csv)
    print(f"\nSaved model metrics summary CSV to {summary_csv_path}")
    print(summary_df.to_string(index=False))

    # ------------------------------------------------------------------
    # Fit Platt calibration from OOF predictions (sigmoid: LogReg on OOF probas vs actuals)
    # ------------------------------------------------------------------
    # The MC test-fold predictions are out-of-fold (OOF) for the base models.
    # Fitting a logistic regression on the concatenated OOF probabilities → actual labels
    # gives a second-stage calibrator (Platt scaling) that corrects systematic over/under-
    # prediction so dashboard risk scores reflect observed event rates.
    # Saved as calibration_{model_type}.joblib; Lambda loads these at inference time.
    # ------------------------------------------------------------------
    from sklearn.linear_model import LogisticRegression as _LR
    import numpy as _np

    _cal_models_dir = out_base / "models"
    _cal_models_dir.mkdir(parents=True, exist_ok=True)
    _cal_diag: dict = {}

    for _mkey, _mname in [("xgb", "xgboost"), ("xgb_rf", "xgboost_rf"), ("catboost", "catboost")]:
        _probas = oof_preds[_mkey]["y_proba"]
        _trues  = oof_preds[_mkey]["y_true"]
        if not _probas:
            continue
        _p_all = _np.concatenate(_probas).reshape(-1, 1)
        _t_all = _np.concatenate(_trues)
        if len(_np.unique(_t_all)) < 2:
            print(f"[CALIB] Skipping {_mkey}: only one class in OOF labels.")
            continue
        _cal = _LR(C=1.0, solver="lbfgs", max_iter=1000)
        _cal.fit(_p_all, _t_all)
        _cal_path = _cal_models_dir / f"calibration_{_mname}.joblib"
        joblib.dump(_cal, _cal_path)
        print(f"[CALIB] {_mname}: fitted Platt calibrator on {len(_t_all)} OOF samples → {_cal_path}")
        # Calibration diagnostics: mean raw vs mean calibrated probability
        _cal_proba = _cal.predict_proba(_p_all)[:, 1]
        _cal_diag[_mname] = {
            "n_oof_samples": int(len(_t_all)),
            "mean_raw_proba": float(_np.mean(_p_all)),
            "mean_calibrated_proba": float(_np.mean(_cal_proba)),
            "observed_rate": float(_np.mean(_t_all)),
            "calibrator_coef": float(_cal.coef_[0][0]),
            "calibrator_intercept": float(_cal.intercept_[0]),
        }
        _diff = abs(_np.mean(_cal_proba) - float(_np.mean(_t_all)))
        print(
            f"[CALIB] {_mname}: raw mean={_np.mean(_p_all):.4f} → "
            f"calibrated mean={_np.mean(_cal_proba):.4f} (observed rate={_np.mean(_t_all):.4f}, "
            f"residual={_diff:.4f})"
        )

    _cal_diag_path = _cal_models_dir / "calibration_diagnostics.json"
    with open(_cal_diag_path, "w") as _fh:
        json.dump(_cal_diag, _fh, indent=2)
    print(f"[CALIB] Diagnostics saved → {_cal_diag_path}")

    # ------------------------------------------------------------------
    # Train final models on full data and export best models
    # ------------------------------------------------------------------
    import xgboost as xgb  # type: ignore

    from py_helpers.env_utils import get_cpu_cores  # local import to avoid cycles
    nthread = get_cpu_cores()
    
    # Determine device: CPU on Linux, CUDA on Windows (if available)
    device = "cpu" if is_linux() else "cuda"

    # Train best XGBoost variant on full data (use Optuna params only when Optuna tuned that variant)
    xgb_params = optuna_best_params if (optuna_used and optuna_selected_model == "xgb") else _default_xgb_params()
    xgb_rf_params = optuna_best_params if (optuna_used and optuna_selected_model == "xgb_rf") else _default_xgb_rf_params()
    if best_xgb_variant == "xgb":
        xgb_final = _build_model_from_params(xgb_params, "xgb", device, nthread, cat_feature_indices)
        xgb_final.set_params(random_state=RANDOM_STATE)
    else:  # xgb_rf
        xgb_final = _build_model_from_params(xgb_rf_params, "xgb_rf", device, nthread, cat_feature_indices)
        xgb_final.set_params(random_state=RANDOM_STATE)

    try:
        xgb_final.fit(X, y)
    except Exception:
        # Fallback to CPU if CUDA fails (shouldn't happen on Linux)
        xgb_final.set_params(tree_method="hist")
        if "device" in xgb_final.get_params():
            xgb_final.set_params(device="cpu")
        xgb_final.fit(X, y)

    # Train the other XGBoost variant so we can save both for the dashboard (catboost + xgboost + xgboost_rf)
    if best_xgb_variant == "xgb":
        xgb_tree_final = xgb_final
        xgb_rf_final = _build_model_from_params(_default_xgb_rf_params(), "xgb_rf", device, nthread, cat_feature_indices)
        xgb_rf_final.set_params(random_state=RANDOM_STATE)
    else:
        xgb_rf_final = xgb_final
        xgb_tree_final = _build_model_from_params(_default_xgb_params(), "xgb", device, nthread, cat_feature_indices)
        xgb_tree_final.set_params(random_state=RANDOM_STATE)
    for other_model in (xgb_rf_final if best_xgb_variant == "xgb" else xgb_tree_final,):
        try:
            other_model.fit(X, y)
        except Exception:
            other_model.set_params(tree_method="hist")
            if "device" in other_model.get_params():
                other_model.set_params(device="cpu")
            other_model.fit(X, y)

    # Export BEST XGBoost model JSON for FFA rule analysis (always saved, even when selected_model is CatBoost)
    model_json_dir = out_base / "final_model_json"
    model_json_dir.mkdir(parents=True, exist_ok=True)
    xgb_json_path = (
        model_json_dir
        / f"{cohort}_{age_band_fname}_best_xgboost_model.json"
    )
    s3_xgb_json = f"s3://pgxdatalake/gold/final_model/{cohort}/{age_band}/{_s3_bin_infix}{cohort}_{age_band_fname}_best_xgboost_model.json"
    
    booster = xgb_final.get_booster()
    # Use text dump format so the existing XGBoostSymbolicExplainer parser
    # (_parse_xgboost_tree_dump) can consume the trees.
    tree_dumps = booster.get_dump(dump_format="text")
    # Normalize model_type for FFA compatibility: "xgb" -> "xgboost", "xgb_rf" -> "xgboost_rf"
    normalized_model_type = "xgboost" if best_xgb_variant == "xgb" else ("xgboost_rf" if best_xgb_variant == "xgb_rf" else best_xgb_variant)
    ffa_model_json = {
        "model_type": normalized_model_type,
        "variant": best_xgb_variant,  # Keep original variant name for reference
        "feature_names": numeric_feature_cols,
        "trees": tree_dumps,
        "selection_metadata": selection_metadata,
    }
    
    def save_xgb_json():
        with open(xgb_json_path, "w") as f:
            json.dump(ffa_model_json, f, indent=2)
    
    save_model_idempotent(xgb_json_path, s3_xgb_json, save_xgb_json)
    print(f"\nSaved BEST XGBoost model JSON ({best_xgb_variant}) to {xgb_json_path} (for FFA rule analysis)")

    # XGBoost feature importances (from full-data model)
    if hasattr(xgb_final, "feature_importances_"):
        importances = xgb_final.feature_importances_
        fi_df = pd.DataFrame(
            {
                "feature": numeric_feature_cols,
                "importance": importances,
            }
        )
        fi_df = fi_df.sort_values("importance", ascending=False)

        fi_path = out_base / f"{cohort}_{age_band_fname}_xgboost_feature_importance.csv"
        _bin_s3_infix = f"bin_models/{bin_name}/" if bin_name else ""
        s3_fi_path = f"s3://pgxdatalake/gold/final_model/{cohort}/{age_band}/{_bin_s3_infix}{cohort}_{age_band_fname}_xgboost_feature_importance.csv"
        
        def save_fi():
            fi_df.to_csv(fi_path, index=False)
        
        save_model_idempotent(fi_path, s3_fi_path, save_fi)
        print(
            f"\nSaved XGBoost feature importances to {fi_path} "
            f"(top 10 features shown below)."
        )
        print(fi_df.head(10).to_string(index=False))
        
        # Create aggregated feature importance visualizations (bar chart and heatmap)
        try:
            _create_aggregated_feature_importance_visualizations(cohort, age_band, out_base)
        except Exception as e:
            print(f"[WARNING] Could not create visualizations: {e}")
            import traceback
            traceback.print_exc()

    # Train CatBoost on full data and export BEST CatBoost binary (for SHAP)
    try:
        from catboost import CatBoostClassifier  # type: ignore

        cb_params = optuna_best_params if (optuna_used and optuna_selected_model == "catboost") else _default_cat_params()
        cb_final = _build_model_from_params(cb_params, "cat", device, nthread, cat_feature_indices)
        cb_final.set_params(random_seed=RANDOM_STATE)
        cb_final.fit(X, y)

        # Save BEST CatBoost model as binary (.cbm) for SHAP analysis
        cb_binary_path = (
            model_json_dir
            / f"{cohort}_{age_band_fname}_best_catboost_model.cbm"
        )
        s3_cb_cbm = f"s3://pgxdatalake/gold/final_model/{cohort}/{age_band}/{_s3_bin_infix}{cohort}_{age_band_fname}_best_catboost_model.cbm"
        
        def save_cb_cbm():
            cb_final.save_model(str(cb_binary_path), format="cbm")
        
        save_model_idempotent(cb_binary_path, s3_cb_cbm, save_cb_cbm)
        print(f"Saved BEST CatBoost model binary to {cb_binary_path} (for SHAP analysis)")

        # Also save JSON for reference
        cb_json_path = (
            model_json_dir
            / f"{cohort}_{age_band_fname}_best_catboost_model.json"
        )
        s3_cb_json = f"s3://pgxdatalake/gold/final_model/{cohort}/{age_band}/{_s3_bin_infix}{cohort}_{age_band_fname}_best_catboost_model.json"
        
        def save_cb_json():
            cb_final.save_model(str(cb_json_path), format="json")
        
        save_model_idempotent(cb_json_path, s3_cb_json, save_cb_json)
        print(f"Saved BEST CatBoost model JSON to {cb_json_path}")

        # CatBoost feature importances (from full-data final model)
        try:
            if hasattr(cb_final, "get_feature_importance"):
                cb_importances = cb_final.get_feature_importance()
            else:
                cb_importances = getattr(cb_final, "feature_importances_", None)
            if cb_importances is not None and len(cb_importances) == len(numeric_feature_cols):
                cb_fi_df = pd.DataFrame(
                    {"feature": numeric_feature_cols, "importance": cb_importances}
                )
                cb_fi_df = cb_fi_df.sort_values("importance", ascending=False)
                cb_fi_path = out_base / f"{cohort}_{age_band_fname}_catboost_feature_importance.csv"
                _bin_s3_infix_cb = f"bin_models/{bin_name}/" if bin_name else ""
                s3_cb_fi_path = f"s3://pgxdatalake/gold/final_model/{cohort}/{age_band}/{_bin_s3_infix_cb}{cohort}_{age_band_fname}_catboost_feature_importance.csv"
                def save_cb_fi():
                    cb_fi_df.to_csv(cb_fi_path, index=False)
                save_model_idempotent(cb_fi_path, s3_cb_fi_path, save_cb_fi)
                print(f"\nSaved CatBoost feature importances to {cb_fi_path} (top 10 below).")
                print(cb_fi_df.head(10).to_string(index=False))
        except Exception as e:
            print(f"[WARNING] Could not save CatBoost feature importances: {e}")

        # Also save binary/joblib models for deployment (step 10 dashboard)
        models_dir = out_base / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        xgb_joblib_path = models_dir / "xgboost.joblib"
        cb_joblib_path = models_dir / "catboost.joblib"
        
        s3_xgb_joblib = f"s3://pgxdatalake/gold/final_model/{cohort}/{age_band}/{_s3_bin_infix}xgboost.joblib"
        s3_cb_joblib = f"s3://pgxdatalake/gold/final_model/{cohort}/{age_band}/{_s3_bin_infix}catboost.joblib"
        
        def save_xgb_joblib():
            # Fix base_score before saving to ensure SHAP compatibility
            # XGBoost sometimes serializes base_score as '[1.6610055E-1]' which SHAP can't parse
            model_to_save = xgb_final
            if hasattr(xgb_final, 'get_booster'):
                booster = xgb_final.get_booster()
                try:
                    config = booster.save_config()
                    config_dict = json.loads(config)
                    learner = config_dict.get('learner')
                    if isinstance(learner, list) and len(learner) > 0:
                        learner = learner[0]
                    if learner and 'learner_model_param' in learner:
                        base_score = learner['learner_model_param'].get('base_score', '')
                        if isinstance(base_score, str) and base_score.startswith('[') and base_score.endswith(']'):
                            # Fix base_score format
                            import ast
                            parsed = ast.literal_eval(base_score)
                            if isinstance(parsed, list) and len(parsed) > 0:
                                fixed_score = float(parsed[0])
                                learner['learner_model_param']['base_score'] = str(fixed_score)
                                if isinstance(config_dict['learner'], list):
                                    config_dict['learner'][0] = learner
                                else:
                                    config_dict['learner'] = learner
                                
                                # Reload config into booster
                                booster.load_config(json.dumps(config_dict))
                                
                                # Force persistence: save to temp file and reload into new model
                                import tempfile
                                import os
                                import xgboost as xgb
                                
                                with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_file:
                                    tmp_path = tmp_file.name
                                
                                # Save fixed model to temp file
                                booster.save_model(tmp_path)
                                
                                # Create new model object with fixed booster
                                model_to_save = xgb.XGBClassifier()
                                model_to_save.load_model(tmp_path)
                                
                                # Clean up temp file
                                try:
                                    os.unlink(tmp_path)
                                except:
                                    pass
                                
                                print(f"✅ Fixed base_score from {base_score} to {fixed_score} before saving joblib")
                except Exception as e:
                    print(f"⚠️  Warning: Could not fix base_score before saving: {e}. SHAP may fail.")
            
            joblib.dump(model_to_save, xgb_joblib_path)
        
        save_model_idempotent(xgb_joblib_path, s3_xgb_joblib, save_xgb_joblib)
        print(f"Saved deployment-ready XGBoost model to {xgb_joblib_path}")
        
        # Also save native XGBoost booster binary model for SHAP (more reliable than joblib)
        xgb_binary_model_path = models_dir / "xgboost_model.ubj"
        s3_xgb_binary_model = f"s3://pgxdatalake/gold/final_model/{cohort}/{age_band}/{_s3_bin_infix}xgboost_model.ubj"
        
        def save_xgb_binary_model():
            # Use XGBoost final model's booster to save native binary format (UBJ)
            # This is what SHAP's TreeExplainer expects and avoids base_score parsing issues
            model_source = xgb_final
            if hasattr(model_source, 'get_booster'):
                booster = model_source.get_booster()
                booster.save_model(str(xgb_binary_model_path))
            elif hasattr(xgb_final, 'get_booster'):
                booster = xgb_final.get_booster()
                booster.save_model(str(xgb_binary_model_path))
            else:
                raise ValueError("Cannot save XGBoost binary model: no booster available")
        
        save_model_idempotent(xgb_binary_model_path, s3_xgb_binary_model, save_xgb_binary_model)
        print(f"Saved native XGBoost booster binary model to {xgb_binary_model_path} (for SHAP)")
        
        # Also save native CatBoost binary model for SHAP (consistent with XGBoost)
        cb_binary_model_path = models_dir / "catboost_model.cbm"
        s3_cb_binary_model = f"s3://pgxdatalake/gold/final_model/{cohort}/{age_band}/{_s3_bin_infix}catboost_model.cbm"
        
        def save_cb_binary_model():
            # Save CatBoost in native binary format (.cbm) for SHAP
            # This is CatBoost's native format and works directly with SHAP
            cb_final.save_model(str(cb_binary_model_path), format="cbm")
        
        save_model_idempotent(cb_binary_model_path, s3_cb_binary_model, save_cb_binary_model)
        print(f"Saved native CatBoost binary model to {cb_binary_model_path} (for SHAP)")
        
        def save_cb_joblib():
            cb_final.save_model(str(cb_joblib_path))
        
        save_model_idempotent(cb_joblib_path, s3_cb_joblib, save_cb_joblib)
        print(f"Saved deployment-ready CatBoost model to {cb_joblib_path}")

        # ------------------------------------------------------------------
        # Mirror final cohort-level models into 6_final_model/model_outputs
        # for FFA, SHAP, and future prediction consumption.
        # ------------------------------------------------------------------
        model_outputs_base = (
            PROJECT_ROOT
            / "6_final_model"
            / "model_outputs"
            / cohort
            / age_band_fname
        )
        model_outputs_base.mkdir(parents=True, exist_ok=True)

        # BEST XGBoost JSON (for FFA analysis) - mirror to model_outputs
        xgb_json_out = (
            model_outputs_base
            / f"{cohort}_{age_band_fname}_best_xgboost_model.json"
        )
        def save_xgb_json_out():
            with open(xgb_json_out, "w") as f:
                json.dump(ffa_model_json, f, indent=2)
        
        save_model_idempotent(xgb_json_out, s3_xgb_json, save_xgb_json_out)

        # BEST CatBoost binary (.cbm) for SHAP analysis - mirror to model_outputs
        cb_cbm_out = (
            model_outputs_base
            / f"{cohort}_{age_band_fname}_best_catboost_model.cbm"
        )
        def save_cb_cbm_out():
            cb_final.save_model(str(cb_cbm_out), format="cbm")
        
        save_model_idempotent(cb_cbm_out, s3_cb_cbm, save_cb_cbm_out)

        # Also save CatBoost JSON for reference - mirror to model_outputs
        cb_json_out = (
            model_outputs_base
            / f"{cohort}_{age_band_fname}_best_catboost_model.json"
        )
        def save_cb_json_out():
            cb_final.save_model(str(cb_json_out), format="json")
        
        save_model_idempotent(cb_json_out, s3_cb_json, save_cb_json_out)

        print(f"Saved final model artifacts for {cohort} / {age_band} to {model_outputs_base}")

        # Save checkpoint with all S3 outputs
        try:
            from py_helpers.checkpoint_utils import save_step_checkpoint

            s3_outputs = [
                s3_xgb_json,
                s3_cb_cbm,
                s3_cb_json,
                s3_metadata,
                s3_xgb_joblib,
                s3_cb_joblib,
                s3_fi_path,
            ]

            # Save checkpoint
            save_step_checkpoint(
                step_name="6_final_model",
                cohort=cohort,
                age_band=age_band,
                metadata={
                    "best_xgb_variant": best_xgb_variant,
                    "n_runs": n_runs,
                },
                output_paths=s3_outputs,
            )
        except ImportError:
            pass  # Checkpoint saving is optional
    except ImportError as e:
        print(f"CatBoost not installed; skipping CatBoost model save. {e}")
    except Exception as e:
        print(f"CatBoost failed to train or save; skipping CatBoost model export. Error: {e}")
        import traceback
        traceback.print_exc()


def mirror_bin_artifacts_to_aggregate_root(
    cohort: str,
    age_band: str,
    preferred_bin: str = "medium",
) -> None:
    """
    After per-bin training only, copy one bin's artifact tree to the cohort-level
    aggregate directory so prepare_models.py, Lambda, and S3 paths that expect
    outputs/{cohort}/{age_band}/ (not bin_models/) keep working.

    Preference order: *preferred_bin* first, then remaining _DENSITY_BINS order.
    """
    import shutil

    age_band_fname = age_band_to_fname(age_band)
    agg = PROJECT_ROOT / "6_final_model" / "outputs" / cohort / age_band_fname
    bin_root = agg / "bin_models"
    order = [preferred_bin] + [b for b in _DENSITY_BINS if b != preferred_bin]
    src_bin = None
    for b in order:
        meta = bin_root / b / f"{cohort}_{age_band_fname}_model_selection_metadata.json"
        if meta.exists():
            src_bin = b
            break
    if src_bin is None:
        print(
            "[WARN] mirror_bin_artifacts_to_aggregate_root: no per-bin model metadata "
            f"found under {bin_root}; aggregate root not filled."
        )
        return

    src = bin_root / src_bin
    for item in src.iterdir():
        dest = agg / item.name
        if dest.exists():
            if dest.is_dir():
                shutil.rmtree(dest)
            else:
                dest.unlink()
        if item.is_dir():
            shutil.copytree(item, dest)
        else:
            shutil.copy2(item, dest)

    mo = PROJECT_ROOT / "6_final_model" / "model_outputs" / cohort / age_band_fname
    mo.mkdir(parents=True, exist_ok=True)
    fj = agg / "final_model_json"
    if fj.exists():
        for name in (
            f"{cohort}_{age_band_fname}_best_xgboost_model.json",
            f"{cohort}_{age_band_fname}_best_catboost_model.cbm",
            f"{cohort}_{age_band_fname}_best_catboost_model.json",
        ):
            p = fj / name
            if p.exists():
                shutil.copy2(p, mo / name)

    print(
        f"[INFO] Mirrored bin '{src_bin}' artifacts to aggregate root for "
        f"{cohort}/{age_band} (deploy / prepare_models)."
    )


def train_per_bin(
    df: pd.DataFrame,
    cohort: str,
    age_band: str,
    n_runs: int | None = None,
    min_total: int = 50,
    min_per_class: int = 10,
) -> None:
    """
    Train a separate model for each n_event_bin (low / medium / high / extreme).

    For each bin, the subset of *df* where n_event_bin == bin is extracted and
    passed to train_and_evaluate() with outputs going to:
      outputs/{cohort}/{age_band_fname}/bin_models/{bin_name}/

    Bins with fewer than *min_total* patients or fewer than *min_per_class* in
    either class are skipped with a warning.

    Requires that build_final_features() has already been called (so df contains
    the 'n_event_bin' string column and 'n_event_bin_ordinal' numeric feature).
    """
    if "n_event_bin" not in df.columns:
        print(
            "[WARN] train_per_bin: 'n_event_bin' column not found in feature matrix. "
            "Re-run build_final_features() first (Step 6)."
        )
        return

    print(f"\n{'='*60}")
    print(f"Per-bin model training: {cohort} / {age_band}")
    print(f"  Full dataset: {len(df)} patients")
    print(f"  Bin distribution: {df['n_event_bin'].value_counts().to_dict()}")
    print(f"{'='*60}")

    for bin_name in _DENSITY_BINS:
        bin_df = df[df["n_event_bin"] == bin_name].copy()
        n_total = len(bin_df)
        n_cases = int((bin_df["target"] == 1).sum())
        n_controls = int((bin_df["target"] == 0).sum())

        print(f"\n--- Bin: {bin_name} | {n_total} patients ({n_cases} cases, {n_controls} controls) ---")

        if n_total < min_total or n_cases < min_per_class or n_controls < min_per_class:
            print(
                f"  [SKIP] Insufficient data for per-bin model "
                f"(need >={min_total} total, >={min_per_class} per class)."
            )
            continue

        train_and_evaluate(bin_df, cohort, age_band, n_runs=n_runs, bin_name=bin_name)

    print(f"\n{'='*60}")
    print(f"Per-bin training complete: {cohort} / {age_band}")
    print(f"{'='*60}\n")


def main() -> None:
    import logging

    parser = argparse.ArgumentParser(
        description="Build final features and train a baseline model for a cohort/age_band."
    )
    parser.add_argument("--cohort", required=True, help="Cohort name, e.g. opioid_ed")
    parser.add_argument("--age_band", required=True, help="Age band, e.g. 0-12")
    parser.add_argument(
        "--n_runs",
        type=int,
        default=None,
        help="Number of Monte-Carlo CV runs (default: auto-detect from environment, 3 on EC2, 1 on Windows)",
    )
    parser.add_argument(
        "--train-mode",
        choices=["per_bin", "aggregate", "both"],
        default="per_bin",
        help="per_bin (default): train density-bin models only; mirror one bin to aggregate outputs for deploy. "
        "aggregate: single cohort-wide model only (legacy). both: cohort-wide then per-bin subdirs.",
    )
    args = parser.parse_args()

    # Simple logger + log file for final model
    logs_dir = PROJECT_ROOT / "logs" / "6_final_model"
    logs_dir.mkdir(parents=True, exist_ok=True)
    age_band_fname = age_band_to_fname(args.age_band)
    log_path = logs_dir / f"final_model_{args.cohort}_{age_band_fname}.log"

    logger = logging.getLogger(f"final_model.{args.cohort}.{age_band_fname}")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s",
        )
        fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    logger.propagate = False

    env = detect_runtime_environment(PROJECT_ROOT)
    logger.info(
        "Runtime environment: os=%s logical_cores=%s ram_gb=%s fast_root=%s",
        env.os_name,
        env.logical_cores,
        env.ram_gb,
        env.fast_root,
    )

    # Idempotency check: Check local files first, then S3
    age_band_fname_check = age_band_to_fname(args.age_band)
    out_base_check = PROJECT_ROOT / "6_final_model" / "outputs" / args.cohort / age_band_fname_check
    
    # Define expected local output paths
    local_outputs = {
        "metadata": out_base_check / f"{args.cohort}_{age_band_fname_check}_model_selection_metadata.json",
        "xgb_json": out_base_check / "final_model_json" / f"{args.cohort}_{age_band_fname_check}_best_xgboost_model.json",
        "cb_cbm": out_base_check / "final_model_json" / f"{args.cohort}_{age_band_fname_check}_best_catboost_model.cbm",
        "xgb_joblib": out_base_check / "models" / "xgboost.joblib",
        "cb_joblib": out_base_check / "models" / "catboost.joblib",
        "fi_csv": out_base_check / f"{args.cohort}_{age_band_fname_check}_xgboost_feature_importance.csv",
        "features_csv": out_base_check / f"{args.cohort}_{age_band_fname_check}_train_final_features_no_leakage.csv",
        # Also check model_outputs copies (needed by FFA/SHAP)
        "model_outputs_xgb_json": PROJECT_ROOT / "6_final_model" / "model_outputs" / args.cohort / age_band_fname_check / f"{args.cohort}_{age_band_fname_check}_best_xgboost_model.json",
    }
    
    # Check if all local outputs exist
    # In per_bin mode, cohort-level files are only a mirrored snapshot — check per-bin files instead.
    if args.train_mode == "per_bin":
        from py_helpers.event_density_utils import DENSITY_BINS as _CHECK_DENSITY_BINS
        all_local_exist = (
            local_outputs["features_csv"].exists()
            and all(
                (out_base_check / "bin_models" / b / "models" / "xgboost.joblib").exists()
                for b in _CHECK_DENSITY_BINS
            )
        )
    else:
        all_local_exist = all(path.exists() for path in local_outputs.values())

    if all_local_exist:
        logger.info(f"Step 6 outputs already exist locally for {args.cohort}/{args.age_band}; skipping regeneration.")
        logger.info(f"  Found {len(local_outputs)} output files")
        
        # Ensure model_outputs copies exist (needed by FFA/SHAP even if idempotent)
        model_outputs_base = PROJECT_ROOT / "6_final_model" / "model_outputs" / args.cohort / age_band_fname_check
        model_outputs_base.mkdir(parents=True, exist_ok=True)
        
        # Copy model JSON files to model_outputs if they don't exist there
        import shutil
        xgb_json_source = local_outputs["xgb_json"]
        xgb_json_dest = model_outputs_base / f"{args.cohort}_{age_band_fname_check}_best_xgboost_model.json"
        if xgb_json_source.exists() and not xgb_json_dest.exists():
            shutil.copy2(xgb_json_source, xgb_json_dest)
            logger.info(f"Copied XGBoost JSON to model_outputs: {xgb_json_dest}")
        
        cb_cbm_source = local_outputs["cb_cbm"]
        cb_cbm_dest = model_outputs_base / f"{args.cohort}_{age_band_fname_check}_best_catboost_model.cbm"
        if cb_cbm_source.exists() and not cb_cbm_dest.exists():
            shutil.copy2(cb_cbm_source, cb_cbm_dest)
            logger.info(f"Copied CatBoost CBM to model_outputs: {cb_cbm_dest}")
        
        # Still try to sync to S3 if not already there (idempotent upload)
        try:
            from py_helpers.checkpoint_utils import upload_file_to_s3, save_step_checkpoint
            
            s3_outputs = []
            s3_base = f"s3://pgxdatalake/gold/final_model/{args.cohort}/{args.age_band}"
            
            # Upload each file if not already in S3
            if local_outputs["metadata"].exists():
                s3_path = f"{s3_base}/{args.cohort}_{age_band_fname_check}_model_selection_metadata.json"
                if upload_file_to_s3(local_outputs["metadata"], s3_path, logger):
                    s3_outputs.append(s3_path)
            
            if local_outputs["xgb_json"].exists():
                s3_path = f"{s3_base}/{args.cohort}_{age_band_fname_check}_best_xgboost_model.json"
                if upload_file_to_s3(local_outputs["xgb_json"], s3_path, logger):
                    s3_outputs.append(s3_path)
            
            if local_outputs["cb_cbm"].exists():
                s3_path = f"{s3_base}/{args.cohort}_{age_band_fname_check}_best_catboost_model.cbm"
                if upload_file_to_s3(local_outputs["cb_cbm"], s3_path, logger):
                    s3_outputs.append(s3_path)
            
            if local_outputs["xgb_joblib"].exists():
                s3_path = f"{s3_base}/xgboost.joblib"
                if upload_file_to_s3(local_outputs["xgb_joblib"], s3_path, logger):
                    s3_outputs.append(s3_path)
            
            if local_outputs["cb_joblib"].exists():
                s3_path = f"{s3_base}/catboost.joblib"
                if upload_file_to_s3(local_outputs["cb_joblib"], s3_path, logger):
                    s3_outputs.append(s3_path)
            
            if local_outputs["fi_csv"].exists():
                s3_path = f"{s3_base}/{args.cohort}_{age_band_fname_check}_xgboost_feature_importance.csv"
                if upload_file_to_s3(local_outputs["fi_csv"], s3_path, logger):
                    s3_outputs.append(s3_path)
            
            if local_outputs["features_csv"].exists():
                s3_path = f"{s3_base}/{args.cohort}_{age_band_fname_check}_train_final_features_no_leakage.csv"
                if upload_file_to_s3(local_outputs["features_csv"], s3_path, logger):
                    s3_outputs.append(s3_path)
            
            # Save checkpoint if outputs uploaded
            if s3_outputs:
                save_step_checkpoint(
                    step_name="6_final_model",
                    cohort=args.cohort,
                    age_band=args.age_band,
                    metadata={"n_outputs": len(s3_outputs)},
                    output_paths=s3_outputs,
                )
        except ImportError:
            pass  # S3 sync is optional
        
        return
    
    # Check S3 for existing outputs (fallback if local doesn't exist)
    try:
        from py_helpers.checkpoint_utils import check_step_outputs_exist, check_step_checkpoint_exists
        
        if args.train_mode == "per_bin":
            from py_helpers.event_density_utils import DENSITY_BINS as _CHECK_DENSITY_BINS
            s3_output_paths = [
                f"s3://pgxdatalake/gold/final_model/{args.cohort}/{args.age_band}/bin_models/{b}/xgboost.joblib"
                for b in _CHECK_DENSITY_BINS
            ]
        else:
            s3_output_paths = [
                f"s3://pgxdatalake/gold/final_model/{args.cohort}/{args.age_band}/{args.cohort}_{age_band_fname_check}_best_xgboost_model.json",
                f"s3://pgxdatalake/gold/final_model/{args.cohort}/{args.age_band}/{args.cohort}_{age_band_fname_check}_best_catboost_model.cbm",
                f"s3://pgxdatalake/gold/final_model/{args.cohort}/{args.age_band}/{args.cohort}_{age_band_fname_check}_model_selection_metadata.json",
            ]

        if check_step_outputs_exist(s3_output_paths, logger) or check_step_checkpoint_exists("6_final_model", args.cohort, args.age_band, logger):
            logger.info(f"Step 6 outputs exist in S3 for {args.cohort}/{args.age_band}; downloading to local.")
            
            # Download from S3 to local
            try:
                import boto3
                s3_client = boto3.client("s3")
                S3_BUCKET = "pgxdatalake"
                s3_base_key = f"gold/final_model/{args.cohort}/{args.age_band}"
                
                # Download each file
                out_base_check.mkdir(parents=True, exist_ok=True)
                (out_base_check / "final_model_json").mkdir(parents=True, exist_ok=True)
                (out_base_check / "models").mkdir(parents=True, exist_ok=True)
                
                # Download metadata
                s3_key = f"{s3_base_key}/{args.cohort}_{age_band_fname_check}_model_selection_metadata.json"
                try:
                    s3_client.download_file(S3_BUCKET, s3_key, str(local_outputs["metadata"]))
                    logger.info(f"Downloaded {local_outputs['metadata']} from S3")
                except Exception as e:
                    logger.debug(f"Could not download metadata: {e}")
                
                # Download XGBoost JSON
                s3_key = f"{s3_base_key}/{args.cohort}_{age_band_fname_check}_best_xgboost_model.json"
                try:
                    s3_client.download_file(S3_BUCKET, s3_key, str(local_outputs["xgb_json"]))
                    logger.info(f"Downloaded {local_outputs['xgb_json']} from S3")
                except Exception as e:
                    logger.debug(f"Could not download XGBoost JSON: {e}")
                
                # Download CatBoost CBM
                s3_key = f"{s3_base_key}/{args.cohort}_{age_band_fname_check}_best_catboost_model.cbm"
                try:
                    s3_client.download_file(S3_BUCKET, s3_key, str(local_outputs["cb_cbm"]))
                    logger.info(f"Downloaded {local_outputs['cb_cbm']} from S3")
                except Exception as e:
                    logger.debug(f"Could not download CatBoost CBM: {e}")
                
                # Download joblib files if they exist
                s3_key = f"{s3_base_key}/xgboost.joblib"
                try:
                    s3_client.download_file(S3_BUCKET, s3_key, str(local_outputs["xgb_joblib"]))
                    logger.info(f"Downloaded {local_outputs['xgb_joblib']} from S3")
                except Exception as e:
                    logger.debug(f"Could not download XGBoost joblib: {e}")
                
                s3_key = f"{s3_base_key}/catboost.joblib"
                try:
                    s3_client.download_file(S3_BUCKET, s3_key, str(local_outputs["cb_joblib"]))
                    logger.info(f"Downloaded {local_outputs['cb_joblib']} from S3")
                except Exception as e:
                    logger.debug(f"Could not download CatBoost joblib: {e}")
                
                # Download feature importance CSV if it exists
                s3_key = f"{s3_base_key}/{args.cohort}_{age_band_fname_check}_xgboost_feature_importance.csv"
                try:
                    s3_client.download_file(S3_BUCKET, s3_key, str(local_outputs["fi_csv"]))
                    logger.info(f"Downloaded {local_outputs['fi_csv']} from S3")
                except Exception as e:
                    logger.debug(f"Could not download feature importance CSV: {e}")
                
                # Download features CSV (needed by Step 8 SHAP analysis)
                s3_key = f"{s3_base_key}/{args.cohort}_{age_band_fname_check}_train_final_features_no_leakage.csv"
                try:
                    s3_client.download_file(S3_BUCKET, s3_key, str(local_outputs["features_csv"]))
                    logger.info(f"Downloaded {local_outputs['features_csv']} from S3")
                except Exception as e:
                    logger.debug(f"Could not download features CSV: {e}")
                
                # Check if we got the essential files (including features CSV needed by Step 8)
                essential_files = [
                    local_outputs["metadata"], 
                    local_outputs["xgb_json"], 
                    local_outputs["cb_cbm"],
                    local_outputs["features_csv"]  # Required by Step 8 SHAP analysis
                ]
                if all(path.exists() for path in essential_files):
                    logger.info(f"Step 6 outputs downloaded from S3; skipping regeneration.")
                    # Ensure model_outputs copies exist (needed by FFA/SHAP)
                    model_outputs_base = PROJECT_ROOT / "6_final_model" / "model_outputs" / args.cohort / age_band_fname_check
                    model_outputs_base.mkdir(parents=True, exist_ok=True)
                    import shutil
                    xgb_json_source = local_outputs["xgb_json"]
                    xgb_json_dest = model_outputs_base / f"{args.cohort}_{age_band_fname_check}_best_xgboost_model.json"
                    if xgb_json_source.exists() and not xgb_json_dest.exists():
                        shutil.copy2(xgb_json_source, xgb_json_dest)
                        logger.info(f"Copied XGBoost JSON to model_outputs: {xgb_json_dest}")
                    return
                else:
                    logger.warning(f"Some essential files missing after S3 download. Will regenerate.")
                    missing_files = [f for f in essential_files if not f.exists()]
                    logger.warning(f"Missing files: {[str(f) for f in missing_files]}")
            except Exception as e:
                logger.warning(f"Could not download from S3: {e}. Will regenerate outputs.")
    except ImportError:
        pass  # Fallback to local-only if checkpoint_utils not available

    with function_block("final_model", "run_final_model", logger=logger):
        with step_block("final_model", "build_final_features", logger=logger):
            df = build_final_features(args.cohort, args.age_band)
        if df.empty:
            logger.info(
                "No data assembled for cohort=%s, age_band=%s.",
                args.cohort,
                args.age_band,
            )
            return

        # Persist leakage-filtered final feature table for downstream FFA analysis
        age_band_fname = age_band_to_fname(args.age_band)
        features_dir = (
            PROJECT_ROOT / "6_final_model" / "outputs" / args.cohort / age_band_fname
        )
        features_dir.mkdir(parents=True, exist_ok=True)
        features_path = (
            features_dir
            / f"{args.cohort}_{age_band_fname}_train_final_features_no_leakage.csv"
        )
        df.to_csv(features_path, index=False)
        logger.info("Saved final features (no leakage) to %s", features_path)

        if args.n_runs is not None:
            n_runs = args.n_runs
            logger.info("Using explicit n_runs=%s from command-line argument", n_runs)
        else:
            n_runs = get_mc_cv_n_runs()
            logger.info(
                "Auto-selected n_runs=%s based on environment (CPU cores, memory)",
                n_runs,
            )
        logger.info("train-mode=%s", args.train_mode)

        if args.train_mode in ("aggregate", "both"):
            with step_block("final_model", "train_and_evaluate", logger=logger):
                train_and_evaluate(df, args.cohort, args.age_band, n_runs=n_runs)

        if args.train_mode in ("per_bin", "both"):
            with step_block("final_model", "train_per_bin", logger=logger):
                train_per_bin(df, args.cohort, args.age_band, n_runs=n_runs)

        if args.train_mode == "per_bin":
            mirror_bin_artifacts_to_aggregate_root(args.cohort, args.age_band)

        # Upload train/test to S3 (required for SHAP and FFA analysis; not optional)
        prepare_script = PROJECT_ROOT / "6_final_model" / "prepare_train_test_s3.py"
        if prepare_script.exists():
            logger.info("Uploading model training input data to S3 (required for SHAP/FFA)...")
            subprocess.run(
                [str(get_workflow_python_bin()), str(prepare_script), "--cohort-name", args.cohort, "--age-band", args.age_band, "--project-root", str(PROJECT_ROOT)],
                check=True,
                cwd=PROJECT_ROOT,
            )
            logger.info("Model training input data uploaded to S3 successfully.")
        else:
            raise RuntimeError(
                "prepare_train_test_s3.py not found; model training input data must be uploaded to S3 for SHAP/FFA analysis."
            )

    # Mirror log to pgx-repository/6_final_model_log (best-effort)
    try:
        mirror_log_to_s3(
            feature_step="6_final_model",
            cohort=args.cohort,
            age_band=args.age_band,
            log_path=log_path,
            logger=logger,
        )
    except Exception:
        # Silent best-effort; log is still available locally
        pass


if __name__ == "__main__":
    with module_block("final_model"):
        main()


