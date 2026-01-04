"""
Final-model training script that merges **within-cohort** model_data and
feature-engineering tables for a given (cohort, age_band), then fits a classifier
(CPU on Linux, GPU on Windows if available).

This is intended to be a fast, reproducible analogue of the smoke-test workflow,
using locally built and downloaded artifacts:

- 4a_model_data/cohort_name={cohort}/age_band={age_band}/model_events.parquet
  (cases + within-cohort controls, with an event-level `target` column)
- feature_engineering/from_s3/{4_fpgrowth,5_bupar,6_dtw,7_pgx}/{cohort}/{age_band}/*_added_features_*.csv
"""

import argparse
import sys
from pathlib import Path
from typing import List
import json

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
from py_helpers.constants import age_band_to_fname
from py_helpers.env_utils import get_mc_cv_n_runs, get_data_root, is_linux

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


def remove_target_leakage_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove obvious target-leakage features based on naming conventions,
    mirroring the logic in the legacy remove_target_leakage.py script.

    This drops:
      - Columns starting with 'post_'
      - Columns containing 'time_to' / 'time_to_'
      - Time-window features with suffixes like '_30d', '_90d', '_180d'
        (except those with 'interval' in the name)
      - Datetime helper columns: 'target_time', 'first_time'
      - DTW-derived features (any column with 'dtw' in its name)
      - Any feature whose name contains 'F1120'
    """
    cols = list(df.columns)
    leakage: set[str] = set()

    # Post-event features
    post_features = [c for c in cols if c.startswith("post_")]
    leakage.update(post_features)

    # Time-to-target features
    time_to_features = [
        c for c in cols if "time_to" in c.lower() or "time_to_" in c.lower()
    ]
    leakage.update(time_to_features)

    # Time-window features referencing target (but not generic intervals)
    time_window_features = [
        c
        for c in cols
        if any(x in c for x in ["_30d", "_90d", "_180d"])
        and "interval" not in c.lower()
    ]
    leakage.update(time_window_features)

    # Datetime helper columns
    datetime_features = [c for c in ("target_time", "first_time") if c in cols]
    leakage.update(datetime_features)

    # DTW features – used for protocol filtering, not as model inputs
    dtw_features = [c for c in cols if "dtw" in c.lower()]
    leakage.update(dtw_features)

    # Any feature explicitly referencing F1120
    f1120_features = [c for c in cols if "F1120" in c.upper()]
    leakage.update(f1120_features)

    if leakage:
        kept = [c for c in cols if c not in leakage]
        print(
            "Removing potential target-leakage features:\n  "
            + ", ".join(sorted(leakage))
        )
        return df[kept].copy()

    return df


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
    Resolve the path to model_events.parquet, checking multiple locations.

    Priority on Linux/EC2:
    1. get_data_root()/4a_model_data/... (/mnt/nvme/4a_model_data/...)
    2. PROJECT_ROOT/4a_model_data/... (fallback)
    3. Try downloading from S3 to get_data_root() if not found locally

    Priority on Windows:
    1. PROJECT_ROOT/4a_model_data/... (Windows/local dev)
    2. get_data_root()/4a_model_data/... (fallback)
    3. Try downloading from S3 to PROJECT_ROOT if not found locally

    Returns:
        Path to model_events.parquet file
    """
    data_root = get_data_root()
    is_linux_system = is_linux()

    # Build candidate paths - prioritize data root on Linux, project root on Windows
    if is_linux_system:
        # On Linux/EC2: prioritize /mnt/nvme
        candidates = [
            data_root / "4a_model_data" / f"cohort_name={cohort}" / f"age_band={age_band}" / "model_events.parquet",
            PROJECT_ROOT / "4a_model_data" / f"cohort_name={cohort}" / f"age_band={age_band}" / "model_events.parquet",
        ]
        # Download destination: prefer data root on Linux
        download_dest = candidates[0]
    else:
        # On Windows: prioritize project root
        candidates = [
            PROJECT_ROOT / "4a_model_data" / f"cohort_name={cohort}" / f"age_band={age_band}" / "model_events.parquet",
            data_root / "4a_model_data" / f"cohort_name={cohort}" / f"age_band={age_band}" / "model_events.parquet",
        ]
        # Download destination: prefer project root on Windows
        download_dest = candidates[0]

    # Check each candidate
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
                    f"Please run: python 4a_model_data/create_model_data.py --cohort {cohort} --age-band {age_band}"
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
                        f"Please run: python 4a_model_data/create_model_data.py --cohort {cohort} --age-band {age_band}"
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


def build_final_features(cohort: str, age_band: str) -> pd.DataFrame:
    """
    Build final feature matrix using aggregated patient-level features + PGx features only.
    
    NEW WORKFLOW:
    - Uses aggregated patient-level features (drug/ICD/CPT encodings) directly (no additional encoding)
    - Only adds PGx features (BupaR, DTW, FP-Growth moved to dashboard visualizations only)
    
    Inputs:
      - 4a_model_data/cohort_name={cohort}/age_band={age_band}/model_events.parquet
        (event-level cases + controls with `target` column)
      - 5c_pgx_analysis/.../{cohort}/{age_band}/pgx_added_features_*.csv

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
    con = duckdb.connect()
    # Aggregate event-level data to one row per patient with label
    # Use MAX(target) to handle patients with mixed targets (prefer case=1 if any event is case)
    # This ensures each patient appears only once
    grouped = con.execute(
        f"""
        SELECT
            CAST(mi_person_key AS VARCHAR) AS mi_person_key,
            CAST(MAX(target) AS INTEGER)   AS target,
            COUNT(*)                       AS n_events
        FROM read_parquet('{events_path}')
        WHERE target IN (0, 1)
        GROUP BY mi_person_key
        """
    ).df()
    con.close()

    # Ensure binary labels
    grouped["target"] = grouped["target"].astype(int).clip(lower=0, upper=1)
    
    # Debug: Print class distribution
    target_counts = grouped["target"].value_counts()
    print(f"Class distribution after aggregation:")
    print(f"  Cases (target=1): {target_counts.get(1, 0)}")
    print(f"  Controls (target=0): {target_counts.get(0, 0)}")
    if len(target_counts) < 2:
        print(f"  WARNING: Only one class present! All targets = {target_counts.index[0]}")

    # ------------------------------------------------------------------
    # Event-level categorical encoding (drug / ICD / CPT) aggregated
    # to patient-level numeric features.
    # ------------------------------------------------------------------
    # We push the heavy work into DuckDB over Parquet so that even very large
    # event tables remain tractable. We only bring the small, unique code
    # sets into pandas for numeric encoding.
    events_cols = [
        "mi_person_key",
        "drug_name",
        "primary_icd_diagnosis_code",
        "procedure_code",
    ]
    con = duckdb.connect()
    try:
        events_sample = con.execute(
            f"SELECT * FROM read_parquet('{events_path}') LIMIT 1"
        ).df()
        available_cols = [c for c in events_cols if c in events_sample.columns]

        if available_cols:
            try:
                per_patient_parts: list[pd.DataFrame] = []

                # Drug-name based features: prefer precomputed numeric encodings
                # from the 6a drug codebook; if missing, build from distinct
                # drug_name values once and optionally persist.
                if "drug_name" in available_cols:
                    age_band_fname = age_band_to_fname(age_band)
                    codebook_path = (
                        PROJECT_ROOT
                        / "6_final_model"
                        / "outputs"
                        / cohort
                        / age_band_fname
                        / f"{cohort}_{age_band_fname}_drug_codebook.csv"
                    )

                    if codebook_path.exists():
                        codebook_df = pd.read_csv(codebook_path)
                    else:
                        distinct_drugs = con.execute(
                            f"""
                            SELECT DISTINCT drug_name AS drug_name_raw
                            FROM read_parquet('{events_path}')
                            WHERE drug_name IS NOT NULL
                              AND TRIM(drug_name) <> ''
                            """
                        ).df()
                        if not distinct_drugs.empty:
                            drug_enc = encode_drug_name_series(
                                distinct_drugs["drug_name_raw"], prefix="drug"
                            )
                            codebook_df = pd.concat(
                                [distinct_drugs.reset_index(drop=True), drug_enc],
                                axis=1,
                            )
                            codebook_path.parent.mkdir(parents=True, exist_ok=True)
                            codebook_df.to_csv(codebook_path, index=False)
                        else:
                            codebook_df = pd.DataFrame()

                    if not codebook_df.empty:
                        non_feature_cols = {
                            "drug_id",
                            "drug_name_raw",
                            "drug_name_normalized",
                        }
                        drug_feat_cols = [
                            c for c in codebook_df.columns if c not in non_feature_cols
                        ]
                        con.register(
                            "drug_codebook",
                            codebook_df[["drug_name_raw"] + drug_feat_cols],
                        )
                        agg_exprs = []
                        for col in drug_feat_cols:
                            agg_exprs.append(f"AVG(dc.{col}) AS mean_{col}")
                            agg_exprs.append(f"MAX(dc.{col}) AS max_{col}")
                        sql = f"""
                            SELECT
                                CAST(e.mi_person_key AS VARCHAR) AS mi_person_key,
                                {', '.join(agg_exprs)}
                            FROM read_parquet('{events_path}') e
                            LEFT JOIN drug_codebook dc
                              ON e.drug_name = dc.drug_name_raw
                            WHERE e.drug_name IS NOT NULL
                              AND TRIM(e.drug_name) <> ''
                            GROUP BY mi_person_key
                        """
                        drug_agg_df = con.execute(sql).df()
                        drug_agg_df["mi_person_key"] = drug_agg_df[
                            "mi_person_key"
                        ].astype(str)
                        per_patient_parts.append(
                            drug_agg_df.set_index("mi_person_key")
                        )

                # Primary ICD diagnosis code features: build a small codebook on
                # distinct codes, then join and aggregate in DuckDB.
                if "primary_icd_diagnosis_code" in available_cols:
                    distinct_icd = con.execute(
                        f"""
                        SELECT DISTINCT primary_icd_diagnosis_code AS code_raw
                        FROM read_parquet('{events_path}')
                        WHERE primary_icd_diagnosis_code IS NOT NULL
                          AND TRIM(primary_icd_diagnosis_code) <> ''
                        """
                    ).df()
                    if not distinct_icd.empty:
                        icd_enc = encode_icd_series(
                            distinct_icd["code_raw"], prefix="icd_primary"
                        )
                        icd_codebook = pd.concat(
                            [distinct_icd.reset_index(drop=True), icd_enc], axis=1
                        )
                        icd_feat_cols = [
                            c for c in icd_codebook.columns if c != "code_raw"
                        ]
                        con.register(
                            "icd_codebook",
                            icd_codebook[["code_raw"] + icd_feat_cols],
                        )
                        agg_exprs = []
                        for col in icd_feat_cols:
                            agg_exprs.append(f"AVG(ic.{col}) AS mean_{col}")
                            agg_exprs.append(f"MAX(ic.{col}) AS max_{col}")
                        sql = f"""
                            SELECT
                                CAST(e.mi_person_key AS VARCHAR) AS mi_person_key,
                                {', '.join(agg_exprs)}
                            FROM read_parquet('{events_path}') e
                            LEFT JOIN icd_codebook ic
                              ON e.primary_icd_diagnosis_code = ic.code_raw
                            WHERE e.primary_icd_diagnosis_code IS NOT NULL
                              AND TRIM(e.primary_icd_diagnosis_code) <> ''
                            GROUP BY mi_person_key
                        """
                        icd_agg_df = con.execute(sql).df()
                        icd_agg_df["mi_person_key"] = icd_agg_df["mi_person_key"].astype(
                            str
                        )
                        per_patient_parts.append(
                            icd_agg_df.set_index("mi_person_key")
                        )

                # CPT / procedure code features: same pattern as ICD.
                if "procedure_code" in available_cols:
                    distinct_cpt = con.execute(
                        f"""
                        SELECT DISTINCT procedure_code AS code_raw
                        FROM read_parquet('{events_path}')
                        WHERE procedure_code IS NOT NULL
                          AND TRIM(procedure_code) <> ''
                        """
                    ).df()
                    if not distinct_cpt.empty:
                        cpt_enc = encode_cpt_series(
                            distinct_cpt["code_raw"], prefix="cpt_base"
                        )
                        cpt_codebook = pd.concat(
                            [distinct_cpt.reset_index(drop=True), cpt_enc], axis=1
                        )
                        cpt_feat_cols = [
                            c for c in cpt_codebook.columns if c != "code_raw"
                        ]
                        con.register(
                            "cpt_codebook",
                            cpt_codebook[["code_raw"] + cpt_feat_cols],
                        )
                        agg_exprs = []
                        for col in cpt_feat_cols:
                            agg_exprs.append(f"AVG(cp.{col}) AS mean_{col}")
                            agg_exprs.append(f"MAX(cp.{col}) AS max_{col}")
                        sql = f"""
                            SELECT
                                CAST(e.mi_person_key AS VARCHAR) AS mi_person_key,
                                {', '.join(agg_exprs)}
                            FROM read_parquet('{events_path}') e
                            LEFT JOIN cpt_codebook cp
                              ON e.procedure_code = cp.code_raw
                            WHERE e.procedure_code IS NOT NULL
                              AND TRIM(e.procedure_code) <> ''
                            GROUP BY mi_person_key
                        """
                        cpt_agg_df = con.execute(sql).df()
                        cpt_agg_df["mi_person_key"] = cpt_agg_df["mi_person_key"].astype(
                            str
                        )
                        per_patient_parts.append(
                            cpt_agg_df.set_index("mi_person_key")
                        )

                if per_patient_parts:
                    code_feats = pd.concat(per_patient_parts, axis=1).reset_index()
                    code_feats = code_feats.rename(columns={"index": "mi_person_key"})
                    grouped = grouped.merge(code_feats, on="mi_person_key", how="left")
            except MemoryError:
                print(
                    "Skipping event-level categorical encoding (drug/ICD/CPT) for "
                    f"{cohort}, {age_band} due to MemoryError; relying on "
                    "FP-Growth, BupaR, DTW, and PGx feature tables instead."
                )
    finally:
        con.close()

    # ------------------------------------------------------------------
    # PGx Feature Table (ONLY feature engineering step)
    # ------------------------------------------------------------------
    # Note: BupaR, DTW, and FP-Growth are now used for dashboard visualizations only
    # OS-aware path resolution: check multiple locations
    data_root = get_data_root()
    
    local_base_candidates = [
        PROJECT_ROOT / "5_feature_engineering" / "feature_engineering_outputs",
        data_root / "5_feature_engineering" / "feature_engineering_outputs",
    ]
    local_base = next((p for p in local_base_candidates if p.exists()), local_base_candidates[0])
    
    s3_base_candidates = [
        PROJECT_ROOT / "5_feature_engineering" / "from_s3" / "feature_engineering_outputs",
        data_root / "5_feature_engineering" / "from_s3" / "feature_engineering_outputs",
    ]
    s3_base = next((p for p in s3_base_candidates if p.exists()), s3_base_candidates[0])

    def _first_existing(primary: Path, fallback: Path) -> Path:
        """Return first existing path, or primary if neither exists."""
        if primary.exists():
            return primary
        if fallback.exists():
            return fallback
        # Return primary as default (will be checked by _load_feature_table)
        return primary

    # Only load PGx features (other feature engineering steps moved to dashboard)
    pgx_path = _first_existing(
        local_base
        / "7_pgx"
        / cohort
        / age_band
        / f"pgx_added_features_{cohort}_{age_band_fname}.csv",
        s3_base
        / "7_pgx"
        / cohort
        / age_band
        / f"pgx_added_features_{cohort}_{age_band_fname}.csv",
    )

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
    final = remove_target_leakage_features(final)
    return final


def train_and_evaluate(
    df: pd.DataFrame, cohort: str, age_band: str, n_runs: int | None = None
) -> None:
    """
    Train XGBoost (CPU on Linux, GPU on Windows if available) and CatBoost on the assembled feature table,
    optionally using Monte-Carlo CV with `n_runs` stratified train/test splits.

    When n_runs > 1, metrics (AUC, PR-AUC, LogLoss, recall) are aggregated across
    runs for:
      - XGBoost (boosting)
      - CatBoost (if available)
      - Simple ensemble of XGBoost + CatBoost (probability average)
    """
    # Pre-compute age_band_fname once so it is available throughout this function.
    age_band_fname = age_band_to_fname(age_band)
    # Separate features and label
    feature_cols: List[str] = [
        c for c in df.columns if c not in ("mi_person_key", "target")
    ]
    # Keep only numeric feature columns to avoid dtype issues (e.g., ISO timestamps)
    numeric_feature_cols = [
        c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])
    ]
    if not numeric_feature_cols:
        raise ValueError("No numeric feature columns available for training.")

    if len(numeric_feature_cols) < len(feature_cols):
        dropped = sorted(set(feature_cols) - set(numeric_feature_cols))
        print(
            "Dropping non-numeric feature columns (not suitable for RandomForest):\n"
            + ", ".join(dropped)
        )

    # Replace inf/-inf with NaN, then fill remaining NaNs with 0 for robustness
    X = df[numeric_feature_cols].replace([float("inf"), float("-inf")], pd.NA)
    X = X.fillna(0)
    y = df["target"].astype(int)

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
    try:
        import xgboost as xgb  # type: ignore

        use_xgb = True
    except Exception:
        use_xgb = False

    try:
        from catboost import CatBoostClassifier  # type: ignore

        have_catboost = True
    except Exception:
        have_catboost = False

    if not use_xgb:
        raise ImportError("XGBoost is required for the final model.")

    last_run_artifacts = {}

    for run_idx in range(n_runs):
        # MC split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, stratify=y, random_state=42 + run_idx
        )

        print(f"\n[MC {run_idx + 1}/{n_runs}] Class distribution (train):")
        print("  " + _counts(y_train))
        print(f"[MC {run_idx + 1}/{n_runs}] Class distribution (test):")
        print("  " + _counts(y_test))

        print(
            f"\n[MC {run_idx + 1}/{n_runs}] Training XGBoost and XGBoost RF (CPU on Linux, GPU on Windows if available) for "
            f"cohort={cohort}, age_band={age_band} with "
            f"{X_train.shape[0]} train and {X_test.shape[0]} test rows, "
            f"{X_train.shape[1]} numeric features."
        )

        from py_helpers.env_utils import get_xgb_cpu_nthread  # local import to avoid cycles
        nthread = get_xgb_cpu_nthread()
        
        # Determine device: CPU on Linux, CUDA on Windows (if available)
        device = "cpu" if is_linux() else "cuda"

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

        print(
            f"[MC {run_idx + 1}/{n_runs}] "
            f"XGB AUC={metrics['xgb']['auc'][-1]:.4f}, PR-AUC={metrics['xgb']['pr_auc'][-1]:.4f}, "
            f"Recall={metrics['xgb']['recall'][-1]:.4f} | "
            f"XGB-RF AUC={metrics['xgb_rf']['auc'][-1]:.4f}, PR-AUC={metrics['xgb_rf']['pr_auc'][-1]:.4f}, "
            f"Recall={metrics['xgb_rf']['recall'][-1]:.4f}"
        )

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
    # Model Selection: Compare XGBoost vs XGBoost RF using Recall and AUC-PR
    # ------------------------------------------------------------------
    print("\n=== Model Selection (Recall and AUC-PR) ===")
    
    # Calculate mean metrics across MC runs
    xgb_recall_mean = float(np.mean(metrics["xgb"]["recall"])) if metrics["xgb"]["recall"] else 0.0
    xgb_pr_auc_mean = float(np.mean(metrics["xgb"]["pr_auc"])) if metrics["xgb"]["pr_auc"] else 0.0
    xgb_rf_recall_mean = float(np.mean(metrics["xgb_rf"]["recall"])) if metrics["xgb_rf"]["recall"] else 0.0
    xgb_rf_pr_auc_mean = float(np.mean(metrics["xgb_rf"]["pr_auc"])) if metrics["xgb_rf"]["pr_auc"] else 0.0
    
    print(f"XGBoost:      Recall={xgb_recall_mean:.4f}, AUC-PR={xgb_pr_auc_mean:.4f}")
    print(f"XGBoost RF:   Recall={xgb_rf_recall_mean:.4f}, AUC-PR={xgb_rf_pr_auc_mean:.4f}")
    
    # Select best XGBoost variant: Primary = Recall, Secondary = AUC-PR
    if xgb_recall_mean > xgb_rf_recall_mean:
        best_xgb_variant = "xgb"
        selection_reason = f"XGBoost selected due to higher recall ({xgb_recall_mean:.4f} vs {xgb_rf_recall_mean:.4f})"
    elif xgb_recall_mean < xgb_rf_recall_mean:
        best_xgb_variant = "xgb_rf"
        selection_reason = f"XGBoost RF selected due to higher recall ({xgb_rf_recall_mean:.4f} vs {xgb_recall_mean:.4f})"
    else:
        # Tie on recall, use AUC-PR as tiebreaker
        if xgb_pr_auc_mean >= xgb_rf_pr_auc_mean:
            best_xgb_variant = "xgb"
            selection_reason = f"XGBoost selected due to tie on recall ({xgb_recall_mean:.4f}) and higher AUC-PR ({xgb_pr_auc_mean:.4f} vs {xgb_rf_pr_auc_mean:.4f})"
        else:
            best_xgb_variant = "xgb_rf"
            selection_reason = f"XGBoost RF selected due to tie on recall ({xgb_rf_recall_mean:.4f}) and higher AUC-PR ({xgb_rf_pr_auc_mean:.4f} vs {xgb_pr_auc_mean:.4f})"
    
    print(f"\nSelected: {best_xgb_variant.upper()}")
    print(f"Reason: {selection_reason}")
    
    # Save model selection metadata
    age_band_fname = age_band_to_fname(age_band)
    out_base = PROJECT_ROOT / "6_final_model" / "outputs" / cohort / age_band_fname
    out_base.mkdir(parents=True, exist_ok=True)
    
    selection_metadata = {
        "best_xgb_variant": best_xgb_variant,
        "xgb_recall_mean": xgb_recall_mean,
        "xgb_pr_auc_mean": xgb_pr_auc_mean,
        "xgb_rf_recall_mean": xgb_rf_recall_mean,
        "xgb_rf_pr_auc_mean": xgb_rf_pr_auc_mean,
        "selection_reason": selection_reason,
    }
    
    metadata_path = out_base / f"{cohort}_{age_band_fname}_model_selection_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(selection_metadata, f, indent=2)
    print(f"Saved model selection metadata to {metadata_path}")

    # ------------------------------------------------------------------
    # Train final models on full data and export best models
    # ------------------------------------------------------------------
    import xgboost as xgb  # type: ignore

    from py_helpers.env_utils import get_xgb_cpu_nthread  # local import to avoid cycles
    nthread = get_xgb_cpu_nthread()
    
    # Determine device: CPU on Linux, CUDA on Windows (if available)
    device = "cpu" if is_linux() else "cuda"

    # Train best XGBoost variant on full data
    if best_xgb_variant == "xgb":
        xgb_final = xgb.XGBClassifier(
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
            random_state=1997,
        )
    else:  # xgb_rf
        xgb_final = xgb.XGBRFClassifier(
            n_estimators=500,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method="hist",
            device=device,
            objective="binary:logistic",
            eval_metric="logloss",
            n_jobs=nthread,
            random_state=1997,
        )
    
    try:
        xgb_final.fit(X, y)
    except Exception:
        # Fallback to CPU if CUDA fails (shouldn't happen on Linux)
        xgb_final.set_params(tree_method="hist")
        if "device" in xgb_final.get_params():
            xgb_final.set_params(device="cpu")
        xgb_final.fit(X, y)

    # Export BEST XGBoost model JSON (FFA-friendly: trees + feature_names)
    model_json_dir = out_base / "final_model_json"
    model_json_dir.mkdir(parents=True, exist_ok=True)
    xgb_json_path = (
        model_json_dir
        / f"{cohort}_{age_band_fname}_best_xgboost_model.json"
    )
    booster = xgb_final.get_booster()
    # Use text dump format so the existing XGBoostSymbolicExplainer parser
    # (_parse_xgboost_tree_dump) can consume the trees.
    tree_dumps = booster.get_dump(dump_format="text")
    ffa_model_json = {
        "model_type": best_xgb_variant,
        "variant": best_xgb_variant,
        "feature_names": numeric_feature_cols,
        "trees": tree_dumps,
        "selection_metadata": selection_metadata,
    }
    with open(xgb_json_path, "w") as f:
        json.dump(ffa_model_json, f)
    print(f"\nSaved BEST XGBoost model JSON ({best_xgb_variant}) to {xgb_json_path}")

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
        fi_df.to_csv(fi_path, index=False)
        print(
            f"\nSaved XGBoost feature importances to {fi_path} "
            f"(top 10 features shown below)."
        )
        print(fi_df.head(10).to_string(index=False))

    # Train CatBoost on full data and export BEST CatBoost binary (for SHAP)
    try:
        from catboost import CatBoostClassifier  # type: ignore

        cb_final = CatBoostClassifier(
            iterations=500,
            learning_rate=0.05,
            depth=6,
            loss_function="Logloss",
            eval_metric="Logloss",
            grow_policy="SymmetricTree",
            random_seed=1997,
            verbose=False,
        )
        cb_final.fit(X, y)

        # Save BEST CatBoost model as binary (.cbm) for SHAP analysis
        cb_binary_path = (
            model_json_dir
            / f"{cohort}_{age_band_fname}_best_catboost_model.cbm"
        )
        cb_final.save_model(str(cb_binary_path), format="cbm")
        print(f"Saved BEST CatBoost model binary to {cb_binary_path} (for SHAP analysis)")
        
        # Also save JSON for reference
        cb_json_path = (
            model_json_dir
            / f"{cohort}_{age_band_fname}_best_catboost_model.json"
        )
        cb_final.save_model(str(cb_json_path), format="json")
        print(f"Saved BEST CatBoost model JSON to {cb_json_path}")

        # Also save binary/joblib models for deployment (step 10 dashboard)
        models_dir = out_base / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        xgb_joblib_path = models_dir / "xgboost.joblib"
        cb_joblib_path = models_dir / "catboost.joblib"
        joblib.dump(xgb_final, xgb_joblib_path)
        cb_final.save_model(str(cb_joblib_path))
        print(f"Saved deployment-ready XGBoost model to {xgb_joblib_path}")
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

        # BEST XGBoost JSON (for FFA analysis)
        xgb_json_out = (
            model_outputs_base
            / f"{cohort}_{age_band_fname}_best_xgboost_model.json"
        )
        with open(xgb_json_out, "w") as f:
            json.dump(ffa_model_json, f)

        # BEST CatBoost binary (.cbm) for SHAP analysis
        cb_cbm_out = (
            model_outputs_base
            / f"{cohort}_{age_band_fname}_best_catboost_model.cbm"
        )
        cb_final.save_model(str(cb_cbm_out), format="cbm")
        
        # Also save CatBoost JSON for reference
        cb_json_out = (
            model_outputs_base
            / f"{cohort}_{age_band_fname}_best_catboost_model.json"
        )
        cb_final.save_model(str(cb_json_out), format="json")

        print(f"Saved final model artifacts for {cohort} / {age_band} to {model_outputs_base}")
    except Exception as e:
        print(f"CatBoost not available or failed to train final model; skipping JSON export. {e}")


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
    args = parser.parse_args()

    # Simple logger + log file for final model
    logs_dir = PROJECT_ROOT / "logs" / "final_model"
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
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    logger.propagate = False

    env = detect_runtime_environment(PROJECT_ROOT)
    logger.info(
        "Runtime environment: os=%s logical_cores=%s ram_gb=%s fast_root=%s",
        env.os_name,
        env.logical_cores,
        env.ram_gb,
        env.fast_root,
    )

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

        with step_block("final_model", "train_and_evaluate", logger=logger):
            n_runs = args.n_runs if args.n_runs is not None else get_mc_cv_n_runs()
            train_and_evaluate(df, args.cohort, args.age_band, n_runs=n_runs)

    # Mirror log to pgx-repository/final_model_log (best-effort)
    try:
        mirror_log_to_s3(
            feature_step="final_model",
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


