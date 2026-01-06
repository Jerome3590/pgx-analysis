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
    1. PROJECT_ROOT/4a_model_data/... (EBS-backed, persists across reboots)
    2. get_data_root()/4a_model_data/... (/mnt/nvme/4a_model_data/..., fast but ephemeral)
    3. Try downloading from S3 to PROJECT_ROOT if not found locally

    Priority on Windows:
    1. PROJECT_ROOT/4a_model_data/... (Windows/local dev)
    2. get_data_root()/4a_model_data/... (fallback)
    3. Try downloading from S3 to PROJECT_ROOT if not found locally

    Returns:
        Path to model_events.parquet file
    """
    data_root = get_data_root()
    is_linux_system = is_linux()

    # Build candidate paths - prioritize EBS-backed storage (PROJECT_ROOT) to persist across reboots
    # NVMe is fast but ephemeral, so we check it second as a fallback
    if is_linux_system:
        # On Linux/EC2: prioritize EBS-backed PROJECT_ROOT (persists), then NVMe (fast but ephemeral)
        candidates = [
            PROJECT_ROOT / "4a_model_data" / f"cohort_name={cohort}" / f"age_band={age_band}" / "model_events.parquet",
            data_root / "4a_model_data" / f"cohort_name={cohort}" / f"age_band={age_band}" / "model_events.parquet",
        ]
        # Download destination: prefer EBS-backed PROJECT_ROOT on Linux (persists across reboots)
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


def _create_aggregated_feature_importance_visualizations(
    cohort: str, age_band: str, out_base: Path
) -> None:
    """
    Create bar chart and heatmap visualizations from aggregated feature importance CSV.
    
    Args:
        cohort: Cohort name
        age_band: Age band
        out_base: Output directory base path
    """
    age_band_fname = age_band.replace("-", "_")
    
    # Try to load aggregated feature importance CSV from Step 3
    agg_csv_path = (
        PROJECT_ROOT
        / "3_feature_importance"
        / "outputs"
        / cohort
        / f"{cohort}_{age_band_fname}_aggregated_feature_importance.csv"
    )
    
    # Fallback: try S3 download location
    if not agg_csv_path.exists():
        agg_csv_path = (
            PROJECT_ROOT
            / "3_feature_importance"
            / "from_s3"
            / "by_cohort"
            / cohort
            / f"{cohort}_{age_band_fname}_aggregated_feature_importance.csv"
        )
    
    if not agg_csv_path.exists():
        print(f"[WARNING] Aggregated feature importance CSV not found: {agg_csv_path}")
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
    Load aggregated feature importance codes (drug/ICD/CPT) from Step 3.
    
    Args:
        cohort: Cohort name
        age_band: Age band
        top_n: Maximum number of top features to return (default: None = no limit)
               If None, returns all features sorted by importance.
               If set, limits to top_n to prevent memory/SQL issues.
    
    Returns:
        List of item codes (drug names, ICD codes, CPT codes) from aggregated FI CSV,
        sorted by importance_scaled (descending), optionally limited to top_n.
    """
    age_band_fname = age_band.replace("-", "_")
    
    # Try local path first
    agg_csv_path = (
        PROJECT_ROOT
        / "3_feature_importance"
        / "outputs"
        / cohort
        / f"{cohort}_{age_band_fname}_aggregated_feature_importance.csv"
    )
    
    # Fallback: try S3 download location
    if not agg_csv_path.exists():
        agg_csv_path = (
            PROJECT_ROOT
            / "3_feature_importance"
            / "from_s3"
            / "by_cohort"
            / cohort
            / f"{cohort}_{age_band_fname}_aggregated_feature_importance.csv"
        )
    
    if not agg_csv_path.exists():
        raise FileNotFoundError(
            f"Aggregated feature importance CSV not found for {cohort}/{age_band}. "
            f"Expected at: {agg_csv_path}"
        )
    
    df = pd.read_csv(agg_csv_path)
    if "feature" not in df.columns:
        raise ValueError(f"'feature' column not found in {agg_csv_path}")
    
    # Sort by importance_scaled if available, otherwise by importance_normalized
    if "importance_scaled" in df.columns:
        df = df.sort_values("importance_scaled", ascending=False)
    elif "importance_normalized" in df.columns:
        df = df.sort_values("importance_normalized", ascending=False)
    else:
        # If no importance column, just take first N unique features
        print(f"[WARNING] No importance_scaled or importance_normalized column found. Using first {top_n} features.")
    
    # Extract codes and preserve type information (item_drug_, item_icd_, item_cpt_)
    # Features from Step 3 are already prefixed with type: item_drug_AMOXICILLIN, item_icd_E11.9, etc.
    # Create a mapping of feature -> importance for sorting
    importance_col = "importance_scaled" if "importance_scaled" in df.columns else ("importance_normalized" if "importance_normalized" in df.columns else None)
    if importance_col:
        feature_importance_map = dict(zip(df["feature"], df[importance_col]))
    else:
        # No importance column - use row order (df is already sorted)
        feature_importance_map = {feat: -idx for idx, feat in enumerate(df["feature"])}
    
    # Parse features to extract code and type, preserving importance for sorting
    # Format: item_{type}_{code} or just {code} (fallback)
    parsed_features = []
    for feature in df["feature"]:
        feature_str = str(feature)
        importance = feature_importance_map.get(feature_str, 0)
        
        if feature_str.startswith("item_drug_"):
            code = feature_str.replace("item_drug_", "", 1)
            parsed_features.append(("drug", code, importance))
        elif feature_str.startswith("item_icd_"):
            code = feature_str.replace("item_icd_", "", 1)
            parsed_features.append(("icd", code, importance))
        elif feature_str.startswith("item_cpt_"):
            code = feature_str.replace("item_cpt_", "", 1)
            parsed_features.append(("cpt", code, importance))
        elif feature_str.startswith("item_"):
            # Generic item_ prefix without type - extract code
            code = feature_str.replace("item_", "", 1)
            # Try to infer type from code format (heuristic)
            if any(c.isalpha() for c in code[:3]) and len(code) > 5:
                # Likely a drug name (longer, alphabetic)
                parsed_features.append(("drug", code, importance))
            elif code.replace(".", "").replace("-", "").isdigit() or (len(code) <= 10 and any(c.isdigit() for c in code)):
                # Likely ICD or CPT (shorter, contains digits)
                parsed_features.append(("icd", code, importance))
            else:
                # Unknown type - default to drug
                parsed_features.append(("drug", code, importance))
        else:
            # No prefix - assume drug by default
            parsed_features.append(("drug", feature_str, importance))
    
    # Sort by importance (descending) and apply limit if specified
    parsed_features.sort(key=lambda x: x[2], reverse=True)
    
    if top_n is not None and len(parsed_features) > top_n:
        parsed_features = parsed_features[:top_n]
        print(f"[INFO] Limited to top {top_n} features from {len(df)} total features")
    else:
        print(f"[INFO] Loaded all {len(parsed_features)} aggregated feature importance codes (no limit)")
    
    # Return as list of tuples: (type, code) - drop importance
    return [(ftype, code) for ftype, code, _ in parsed_features]


def build_final_features(cohort: str, age_band: str) -> pd.DataFrame:
    """
    Build final feature matrix using aggregated patient-level features + PGx features only.
    
    NEW WORKFLOW:
    - Uses aggregated patient-level features (drug/ICD/CPT encodings) directly (no additional encoding)
    - Only adds PGx features (BupaR, DTW, FP-Growth moved to dashboard visualizations only)
    
    Inputs:
      - 4a_model_data/cohort_name={cohort}/age_band={age_band}/model_events.parquet
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
            
            for code_type, code in important_codes:
                code_str = str(code)
                # Create safe feature name (replace all special chars with underscore)
                code_safe = code_str.replace(' ', '_').replace('-', '_').replace('.', '_').replace('/', '_').replace('&', '_').replace('(', '_').replace(')', '_').replace('[', '_').replace(']', '_').replace('{', '_').replace('}', '_').replace('*', '_').replace('+', '_').replace('=', '_').replace('|', '_').replace('^', '_').replace('%', '_').replace('!', '_').replace('@', '_').replace('#', '_').replace('$', '_').replace('"', '_').replace("'", '_').replace('\\', '_')
                
                # Only create features for the code type specified
                if code_type == "drug" and "drug_name" in available_cols:
                    drug_codes.append((code_str, f"item_drug_{code_safe}"))
                elif code_type == "icd":
                    icd_cols = [c for c in available_cols if 'icd_diagnosis_code' in c.lower()]
                    if icd_cols:
                        icd_codes.append((code_str, f"item_icd_{code_safe}"))
                elif code_type == "cpt" and "procedure_code" in available_cols:
                    cpt_codes.append((code_str, f"item_cpt_{code_safe}"))
            
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
    
    # Check multiple locations for PGx features from Step 5
    pgx_path_candidates = [
        # Primary location: Step 5 outputs
        PROJECT_ROOT / "5_pgx_analysis" / "outputs" / "feature_engineering" / f"pgx_added_features_{cohort}_{age_band_fname}.csv",
        # Legacy location: feature_engineering_outputs
        PROJECT_ROOT / "5_feature_engineering" / "feature_engineering_outputs" / "7_pgx" / cohort / age_band / f"pgx_added_features_{cohort}_{age_band_fname}.csv",
        # S3 download location
        PROJECT_ROOT / "5_feature_engineering" / "from_s3" / "feature_engineering_outputs" / "7_pgx" / cohort / age_band / f"pgx_added_features_{cohort}_{age_band_fname}.csv",
    ]
    
    # Also check data root locations
    data_root = get_data_root()
    pgx_path_candidates.extend([
        data_root / "5_pgx_analysis" / "outputs" / "feature_engineering" / f"pgx_added_features_{cohort}_{age_band_fname}.csv",
        data_root / "5_feature_engineering" / "feature_engineering_outputs" / "7_pgx" / cohort / age_band / f"pgx_added_features_{cohort}_{age_band_fname}.csv",
    ])
    
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
    final = remove_target_leakage_features(final)
    
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
    except Exception:
        have_catboost = False

    if not use_xgb:
        error_msg = "XGBoost is required for the final model."
        if xgb_import_error:
            error_msg += f" Import error: {xgb_import_error}"
        error_msg += "\n\nTo install XGBoost, run: pip install xgboost"
        raise ImportError(error_msg)

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
    s3_metadata = f"s3://pgxdatalake/gold/final_model/{cohort}/{age_band}/{cohort}_{age_band_fname}_model_selection_metadata.json"
    
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
    s3_xgb_json = f"s3://pgxdatalake/gold/final_model/{cohort}/{age_band}/{cohort}_{age_band_fname}_best_xgboost_model.json"
    
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
        s3_fi_path = f"s3://pgxdatalake/gold/final_model/{cohort}/{age_band}/{cohort}_{age_band_fname}_xgboost_feature_importance.csv"
        
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

        cb_final = CatBoostClassifier(
            iterations=500,
            learning_rate=0.05,
            depth=6,
            loss_function="Logloss",
            eval_metric="Logloss",
            grow_policy="SymmetricTree",
            random_seed=1997,
            verbose=False,
            cat_features=cat_feature_indices,  # Mark binary features as categorical for better performance
        )
        cb_final.fit(X, y)

        # Save BEST CatBoost model as binary (.cbm) for SHAP analysis
        cb_binary_path = (
            model_json_dir
            / f"{cohort}_{age_band_fname}_best_catboost_model.cbm"
        )
        s3_cb_cbm = f"s3://pgxdatalake/gold/final_model/{cohort}/{age_band}/{cohort}_{age_band_fname}_best_catboost_model.cbm"
        
        def save_cb_cbm():
            cb_final.save_model(str(cb_binary_path), format="cbm")
        
        save_model_idempotent(cb_binary_path, s3_cb_cbm, save_cb_cbm)
        print(f"Saved BEST CatBoost model binary to {cb_binary_path} (for SHAP analysis)")

        # Also save JSON for reference
        cb_json_path = (
            model_json_dir
            / f"{cohort}_{age_band_fname}_best_catboost_model.json"
        )
        s3_cb_json = f"s3://pgxdatalake/gold/final_model/{cohort}/{age_band}/{cohort}_{age_band_fname}_best_catboost_model.json"
        
        def save_cb_json():
            cb_final.save_model(str(cb_json_path), format="json")
        
        save_model_idempotent(cb_json_path, s3_cb_json, save_cb_json)
        print(f"Saved BEST CatBoost model JSON to {cb_json_path}")

        # Also save binary/joblib models for deployment (step 10 dashboard)
        models_dir = out_base / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        xgb_joblib_path = models_dir / "xgboost.joblib"
        cb_joblib_path = models_dir / "catboost.joblib"
        
        s3_xgb_joblib = f"s3://pgxdatalake/gold/final_model/{cohort}/{age_band}/xgboost.joblib"
        s3_cb_joblib = f"s3://pgxdatalake/gold/final_model/{cohort}/{age_band}/catboost.joblib"
        
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
        s3_xgb_binary_model = f"s3://pgxdatalake/gold/final_model/{cohort}/{age_band}/xgboost_model.ubj"
        
        def save_xgb_binary_model():
            # Use the fixed model's booster to save native binary format (UBJ)
            # This is what SHAP's TreeExplainer expects and avoids base_score parsing issues
            # Binary format is faster and more reliable than JSON
            # Prefer model_to_save (which may have been fixed) over xgb_final
            model_source = model_to_save if 'model_to_save' in locals() else xgb_final
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
        s3_cb_binary_model = f"s3://pgxdatalake/gold/final_model/{cohort}/{age_band}/catboost_model.cbm"
        
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

        with step_block("final_model", "train_and_evaluate", logger=logger):
            if args.n_runs is not None:
                n_runs = args.n_runs
                logger.info(f"Using explicit n_runs={n_runs} from command-line argument")
            else:
                n_runs = get_mc_cv_n_runs()
                logger.info(f"Auto-selected n_runs={n_runs} based on environment (CPU cores, memory)")
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


