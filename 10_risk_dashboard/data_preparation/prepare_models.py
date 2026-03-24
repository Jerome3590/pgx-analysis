#!/usr/bin/env python3
"""
Prepare models and feature schemas for Lambda deployment.

This script:
1. Loads models from 6_final_model/outputs/{cohort}/{age_band_fname}/models/
2. Extracts feature schemas from 6_final_model/outputs/.../ train CSVs
3. Writes to 10_risk_dashboard/outputs/models/ (used by prepare_lambda_dir.py and Docker)
4. Creates feature_schema.json per cohort/age_band

Usage:
    python prepare_models.py --cohort opioid_ed
    python prepare_models.py --cohort non_opioid_ed
    python prepare_models.py --all
"""

import os
import sys
import json
import argparse
import logging
import subprocess
import importlib.util
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import joblib
import pandas as pd

try:
    import duckdb
    DUCKDB_AVAILABLE = True
except ImportError:
    DUCKDB_AVAILABLE = False

# Add project root to path (script is in 10_risk_dashboard/data_preparation/)
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

MODEL_LIBS_AVAILABLE = (
    importlib.util.find_spec("catboost") is not None
    and importlib.util.find_spec("xgboost") is not None
)
if not MODEL_LIBS_AVAILABLE:
    print("Warning: Model libraries not available. Some operations may fail.")

# Configuration
# Use refactored final-model outputs (step 6) as the canonical source.
FINAL_MODEL_DIR = PROJECT_ROOT / '6_final_model' / 'outputs'
OUTPUT_DIR = PROJECT_ROOT / '10_risk_dashboard' / 'outputs' / 'models'  # For Docker container build
S3_MODEL_PREFIX = 'gold/dashboard/models'  # Optional S3 backup location

# Age bands (each cohort has all age bands; from py_helpers.constants)
from py_helpers.constants import REQUIRED_COHORTS  # noqa: E402
OPIOID_ED_AGE_BANDS = REQUIRED_COHORTS["opioid_ed"]
POLYPHARMACY_AGE_BANDS = REQUIRED_COHORTS["non_opioid_ed"]


def load_model(cohort: str, age_band: str, model_type: str) -> Optional[Any]:
    """Load model from final_model directory."""
    age_band_fname = age_band.replace("-", "_")
    model_dir = FINAL_MODEL_DIR / cohort / age_band_fname / 'models'
    
    if model_type == 'catboost':
        # Prefer binary/joblib object saved by 6_final_model_selection/run_final_model.py
        joblib_path = model_dir / 'catboost.joblib'
        if joblib_path.exists():
            if MODEL_LIBS_AVAILABLE:
                from catboost import CatBoostClassifier
                model = CatBoostClassifier()
                model.load_model(str(joblib_path))
                return model
            return joblib.load(joblib_path)
        # Fallback: try legacy combined joblib if present
        legacy_joblib = model_dir / f'{cohort}_{age_band_fname}_final_model.joblib'
        if legacy_joblib.exists():
            return joblib.load(legacy_joblib)
    elif model_type == 'xgboost':
        # Prefer joblib XGBoost model saved by 6_final_model_selection/run_final_model.py
        joblib_path = model_dir / 'xgboost.joblib'
        if joblib_path.exists():
            return joblib.load(joblib_path)
        # Fallback: JSON path (used by some older tooling)
        json_path = FINAL_MODEL_DIR / cohort / age_band_fname / 'final_model_json' / f'{cohort}_{age_band_fname}_final_model_xgboost.json'
        if json_path.exists():
            return json_path
    elif model_type == 'xgboost_rf':
        # Step 6 saves only the best XGBoost variant as xgboost.joblib; xgboost_rf is not saved when best is xgb
        joblib_path = model_dir / 'xgboost_rf.joblib'
        if joblib_path.exists():
            return joblib.load(joblib_path)
        json_path = FINAL_MODEL_DIR / cohort / age_band_fname / 'final_model_json' / f'{cohort}_{age_band_fname}_final_model_xgboost_rf.json'
        if json_path.exists():
            return json_path
        legacy_joblib = model_dir / f'{cohort}_{age_band_fname}_final_model.joblib'
        if legacy_joblib.exists():
            return joblib.load(legacy_joblib)
    
    return None


def calculate_model_weights(cohort: str, age_band: str) -> Dict[str, float]:
    """
    Choose the best model per cohort/age_band based on MC-CV performance.

    Uses composite score: 0.5 * PR-AUC + 0.5 * (1/(1+logloss)). The model with
    the highest composite score gets weight 1.0; all others get 0.0. We always
    use the best model for the respective cohort (and age_band).

    Returns:
        {
            'catboost': 0.0 or 1.0,
            'xgboost': 0.0 or 1.0,
            'xgboost_rf': 0.0 or 1.0
        }
    """
    age_band_fname = age_band.replace("-", "_")
    default_models = ['catboost', 'xgboost', 'xgboost_rf']
    mc_cv_path = FINAL_MODEL_DIR / cohort / age_band_fname / f'{cohort}_{age_band_fname}_mc_cv_results.csv'

    if not mc_cv_path.exists():
        print(f"Warning: MC-CV results not found: {mc_cv_path}")
        print("  Using equal weights (1.0 each)")
        return {m: 1.0 / len(default_models) for m in default_models}

    df = pd.read_csv(mc_cv_path)
    csv_to_internal = {'XGBoost': 'xgboost', 'XGBoost_RF': 'xgboost_rf', 'CatBoost': 'catboost'}

    model_scores = {}
    for csv_name, internal_name in csv_to_internal.items():
        model_data = df[df['model'] == csv_name]
        if len(model_data) == 0:
            continue
        mean_logloss = model_data['logloss'].mean()
        mean_pr_auc = model_data['pr_auc'].mean()
        normalized_logloss_score = 1 / (1 + mean_logloss)
        composite_score = 0.5 * mean_pr_auc + 0.5 * normalized_logloss_score
        model_scores[internal_name] = {
            'mean_logloss': mean_logloss,
            'mean_pr_auc': mean_pr_auc,
            'composite_score': composite_score
        }

    if not model_scores:
        print("Warning: No MC-CV data, using equal weights")
        return {m: 1.0 / len(default_models) for m in default_models}

    # Check if Ensemble was selected in model_metrics_summary.csv
    summary_path = FINAL_MODEL_DIR / cohort / age_band_fname / f'{cohort}_{age_band_fname}_model_metrics_summary.csv'
    ensemble_selected = False
    if summary_path.exists():
        try:
            summary_df = pd.read_csv(summary_path)
            ens_row = summary_df[summary_df['model'] == 'Ensemble']
            if not ens_row.empty:
                sel_val = str(ens_row['selected'].iloc[0]).strip().lower()
                ensemble_selected = sel_val in ('true', '1', 'yes')
        except Exception:
            pass

    if ensemble_selected:
        # Proportional weights from composite scores so all component models contribute
        total_score = sum(s['composite_score'] for s in model_scores.values())
        weights = {m: (model_scores[m]['composite_score'] / total_score if m in model_scores else 0.0)
                   for m in default_models}
        print(f"  Ensemble selected for {cohort}/{age_band}: using proportional weights")
    else:
        best_model = max(model_scores.keys(), key=lambda m: model_scores[m]['composite_score'])
        weights = {m: 1.0 if m == best_model else 0.0 for m in default_models}
        print(f"  Best model for {cohort}/{age_band}: {best_model} (composite_score: {model_scores[best_model]['composite_score']:.4f})")

    for m in model_scores:
        if m not in weights:
            weights[m] = 0.0

    for model in default_models:
        w = weights.get(model, 0.0)
        if model in model_scores:
            print(f"    {model}: {w:.3f} (composite_score: {model_scores[model]['composite_score']:.4f})")
        else:
            print(f"    {model}: {w:.3f} (no MC-CV data)")
    return weights


def _resolve_train_data_path(cohort: str, age_band_fname: str) -> Tuple[Optional[Path], str]:
    """Resolve training data path: prefer Parquet (efficient), fallback to CSV. Returns (path, 'parquet'|'csv') or (None, '')."""
    parquet_path = FINAL_MODEL_DIR / cohort / age_band_fname / "inputs" / "model_train" / "final_features.parquet"
    csv_path = FINAL_MODEL_DIR / cohort / age_band_fname / f"{cohort}_{age_band_fname}_train_final_features_no_leakage.csv"
    if parquet_path.exists():
        return parquet_path, "parquet"
    if csv_path.exists():
        return csv_path, "csv"
    return None, ""


def _extract_feature_schema_duckdb(data_path: Path, data_format: str) -> Dict[str, Any]:
    """
    Use DuckDB for efficient reads and aggregations on Parquet or CSV.
    Returns feature_names, defaults (medians), patient_bucket_thresholds, n_samples.
    Path is passed as parameter to avoid injection; table/source is our own path.
    """
    exclude_cols = {"mi_person_key", "target", "event_year", "cohort_name", "age_band"}
    path_str = str(data_path.resolve())

    con = duckdb.connect(":memory:")
    # Use parameterized query for path; reader is fixed (read_parquet or read_csv_auto)
    reader = "read_parquet(?)" if data_format == "parquet" else "read_csv_auto(?)"
    params = [path_str]

    try:
        # Column names and types (path passed as param; reader is read_parquet(?) or read_csv_auto(?))
        desc = con.execute(f"DESCRIBE SELECT * FROM {reader}", params).fetchall()  # nosec B608
        columns = [row[0] for row in desc]
        types = {row[0]: row[1] for row in desc}
        feature_names = [c for c in columns if c not in exclude_cols]

        # Row count (efficient)
        n_samples = con.execute(f"SELECT count(*) FROM {reader}", params).fetchone()[0]  # nosec B608

        # Bucket percentiles (33rd/67th) for n_events, n_drugs — single pass over data
        patient_bucket_thresholds = {}
        bucket_vars = ["n_events", "n_drugs"]
        for var in bucket_vars:
            if var not in columns:
                continue
            try:
                # var from bucket_vars; path in params. Reader and var are from our code, not user input.
                q = f"SELECT quantile_cont(\"{var}\", 0.33) AS q33, quantile_cont(\"{var}\", 0.67) AS q67 FROM {reader}"
                row = con.execute(q, params).fetchone()  # nosec B608
                if row and row[0] is not None and row[1] is not None:
                    patient_bucket_thresholds[var] = {"low_medium": float(row[0]), "medium_high": float(row[1])}
            except Exception:
                pass
        if patient_bucket_thresholds:
            print(f"  Patient bucket thresholds: {list(patient_bucket_thresholds.keys())} (DuckDB)", flush=True)

        # Defaults: median per numeric feature
        defaults = {}
        numeric_features = [
            c for c in feature_names
            if types.get(c, "").upper() in ("INTEGER", "BIGINT", "DOUBLE", "FLOAT", "REAL")
        ]
        if numeric_features:
            median_exprs = ", ".join(f'median("{c}") AS "{c}"' for c in numeric_features)
            try:
                med_df = con.execute(f"SELECT {median_exprs} FROM {reader}", params).fetchdf()  # nosec B608
                for c in numeric_features:
                    if c in med_df.columns and pd.notna(med_df[c].iloc[0]):
                        defaults[c] = float(med_df[c].iloc[0])
                    else:
                        defaults[c] = 0.0
            except Exception:
                pass
        for c in feature_names:
            if c not in defaults:
                defaults[c] = 0.0

        return {
            "feature_names": feature_names,
            "defaults": defaults,
            "patient_bucket_thresholds": patient_bucket_thresholds,
            "n_samples": n_samples,
        }
    finally:
        con.close()


def extract_feature_schema(cohort: str, age_band: str) -> Dict[str, Any]:
    """
    Extract feature schema from training data. Prefers Parquet + DuckDB for efficient reads and transforms.
    Falls back to CSV (with DuckDB if available, else pandas).
    """
    age_band_fname = age_band.replace("-", "_")
    data_path, data_format = _resolve_train_data_path(cohort, age_band_fname)

    if data_path is None:
        print("Warning: Training data not found (checked parquet and CSV).", flush=True)
        return {"features": [], "defaults": {}, "model_weights": {}}

    print(f"  Using {'Parquet' if data_format == 'parquet' else 'CSV'}: {data_path.name}", flush=True)

    if DUCKDB_AVAILABLE:
        try:
            schema = _extract_feature_schema_duckdb(data_path, data_format)
            model_weights = calculate_model_weights(cohort, age_band)
            out = {
                "features": schema["feature_names"],
                "defaults": schema["defaults"],
                "model_weights": model_weights,
                "n_features": len(schema["feature_names"]),
                "n_samples": schema["n_samples"],
            }
            if schema["patient_bucket_thresholds"]:
                out["patient_bucket_thresholds"] = schema["patient_bucket_thresholds"]
            return out
        except Exception as e:
            print(f"  Warning: DuckDB path failed ({e}), falling back to pandas.", flush=True)

    # Fallback: pandas on CSV (or parquet via pandas)
    if data_format == "parquet":
        df = pd.read_parquet(data_path)
    else:
        df = pd.read_csv(data_path, nrows=100_000)
    exclude_cols = {"mi_person_key", "target", "event_year", "cohort_name", "age_band"}
    feature_names = [c for c in df.columns if c not in exclude_cols]
    defaults = {}
    for c in feature_names:
        if c in df.columns and df[c].dtype in ["int64", "float64"]:
            defaults[c] = float(df[c].median())
        else:
            defaults[c] = 0.0
    bucket_vars = ["n_events", "n_drugs"]
    patient_bucket_thresholds = {}
    for var in bucket_vars:
        if var in df.columns:
            try:
                patient_bucket_thresholds[var] = {
                    "low_medium": float(df[var].quantile(0.33)),
                    "medium_high": float(df[var].quantile(0.67)),
                }
            except Exception:
                pass
    model_weights = calculate_model_weights(cohort, age_band)
    out = {
        "features": feature_names,
        "defaults": defaults,
        "model_weights": model_weights,
        "n_features": len(feature_names),
        "n_samples": len(df),
    }
    if patient_bucket_thresholds:
        out["patient_bucket_thresholds"] = patient_bucket_thresholds
    return out


def save_model(model: Any, output_path: Path, model_type: str):
    """Save model to output directory."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if model_type == 'catboost':
        from catboost import CatBoostClassifier
        if isinstance(model, CatBoostClassifier):
            model.save_model(str(output_path))
        else:
            joblib.dump(model, output_path)
    else:
        if isinstance(model, Path):
            # Copy JSON file
            import shutil
            shutil.copy(model, output_path)
        else:
            joblib.dump(model, output_path)
    
    print(f"  Saved {model_type} model to: {output_path}")


def _run_2019_distribution_script(cohort: str) -> None:
    """Run prepare_risk_distribution_2019.py for this cohort (idempotent). No-op if script fails."""
    script = Path(__file__).resolve().parent / "prepare_risk_distribution_2019.py"
    if not script.exists():
        return
    print(f"  Running 2019 risk distribution script for {cohort} (timeout 600s)...", flush=True)
    try:
        subprocess.run(
            [sys.executable, str(script), "--cohort", cohort],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=600,
        )
        print("  Completed 2019 distribution script.", flush=True)
    except subprocess.TimeoutExpired:
        print("  Note: 2019 distribution script timed out after 600s.", flush=True)
    except Exception as e:
        print(f"  Note: 2019 distribution script skipped ({e}).", flush=True)


def _prepare_one_age_band(cohort: str, age_band: str) -> Tuple[str, str, bool]:
    """
    Process one (cohort, age_band): extract schema, write JSON, load/save models.
    Used by ProcessPoolExecutor; must be top-level for pickling.
    Returns (cohort, age_band, success).
    """
    age_band_fname = age_band.replace("-", "_")
    output_dir = OUTPUT_DIR / cohort / age_band_fname
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_schema = extract_feature_schema(cohort, age_band)
    n_features = feature_schema.get("n_features", len(feature_schema.get("features", [])))
    if n_features == 0:
        return (cohort, age_band, False)
    schema_path = output_dir / "feature_schema.json"
    with open(schema_path, "w") as f:
        json.dump(feature_schema, f, indent=2)
    for model_type in ("catboost", "xgboost", "xgboost_rf"):
        model = load_model(cohort, age_band, model_type)
        if model is None:
            continue
        model_path = output_dir / f"{model_type}.joblib"
        if isinstance(model, Path):
            model_path = output_dir / f"{model_type}.json"
            import shutil
            shutil.copy(model, model_path)
        else:
            save_model(model, model_path, model_type)
    # Copy n_event_bin_thresholds.json (written by run_final_model.py; Lambda uses it to assign bin labels)
    thresholds_src = FINAL_MODEL_DIR / cohort / age_band_fname / "n_event_bin_thresholds.json"
    if thresholds_src.exists():
        import shutil as _shutil
        _shutil.copy2(thresholds_src, output_dir / "n_event_bin_thresholds.json")
        print(f"  Copied n_event_bin_thresholds.json -> {output_dir}")
    else:
        print(f"  [WARN] n_event_bin_thresholds.json not found at {thresholds_src}; skipping (run run_final_model.py first)")
    # Copy Platt calibration models (calibration_{model_type}.joblib + diagnostics JSON)
    # These are written to models/ subdir by run_final_model.py Platt calibration block.
    import shutil as _shutil2
    cal_src_dir = FINAL_MODEL_DIR / cohort / age_band_fname / "models"
    for _cal_fname in ("calibration_xgboost.joblib", "calibration_xgboost_rf.joblib", "calibration_catboost.joblib", "calibration_diagnostics.json"):
        _cal_src = cal_src_dir / _cal_fname
        if _cal_src.exists():
            _shutil2.copy2(_cal_src, output_dir / _cal_fname)
            print(f"  Copied {_cal_fname} -> {output_dir}")
        else:
            print(f"  [WARN] {_cal_fname} not found at {_cal_src}; skipping (run run_final_model.py first)")

    # Copy per-bin models: bin_models/{bin_name}/models/ → output_dir/bin_models/{bin_name}/
    # Lambda routes inference to these when the patient's n_event_bin is known (falls back to full-cohort if absent).
    import shutil as _shutil3
    _DENSITY_BINS = ("low", "medium", "high", "extreme")
    for _bin in _DENSITY_BINS:
        _bin_src_models = FINAL_MODEL_DIR / cohort / age_band_fname / "bin_models" / _bin / "models"
        if not _bin_src_models.exists():
            continue
        _bin_dst = output_dir / "bin_models" / _bin
        _bin_dst.mkdir(parents=True, exist_ok=True)
        # Main model files
        for _mtype in ("catboost", "xgboost", "xgboost_rf"):
            for _fname in (f"{_mtype}.joblib", f"{_mtype}.json"):
                _src = _bin_src_models / _fname
                if _src.exists():
                    _shutil3.copy2(_src, _bin_dst / _fname)
                    print(f"  Copied bin_models/{_bin}/{_fname} -> {_bin_dst}")
                    break  # prefer joblib; skip json if joblib present
        # Per-bin calibration files
        for _cal_fname in ("calibration_xgboost.joblib", "calibration_xgboost_rf.joblib", "calibration_catboost.joblib"):
            _src = _bin_src_models / _cal_fname
            if _src.exists():
                _shutil3.copy2(_src, _bin_dst / _cal_fname)
                print(f"  Copied bin_models/{_bin}/{_cal_fname} -> {_bin_dst}")
        # Per-bin feature importance CSVs (one level up from models/ subdir)
        _bin_root = FINAL_MODEL_DIR / cohort / age_band_fname / "bin_models" / _bin
        for _fi_fname in (f"{cohort}_{age_band_fname}_xgboost_feature_importance.csv",
                          f"{cohort}_{age_band_fname}_catboost_feature_importance.csv"):
            _src = _bin_root / _fi_fname
            if _src.exists():
                _shutil3.copy2(_src, _bin_dst / _fi_fname)
                print(f"  Copied bin_models/{_bin}/{_fi_fname} -> {_bin_dst}")

    return (cohort, age_band, True)


def prepare_models_for_cohort(cohort: str, age_bands: List[str]):
    """Prepare models for a cohort using all available cores (parallel over age_bands)."""
    n_workers = max(1, os.cpu_count() or 1)
    print(f"\n{'='*60}", flush=True)
    print(f"Preparing models for {cohort} ({len(age_bands)} age bands, {n_workers} workers)", flush=True)
    print(f"{'='*60}", flush=True)

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_prepare_one_age_band, cohort, ab): ab for ab in age_bands}
        for future in as_completed(futures):
            age_band = futures[future]
            try:
                c, ab, ok = future.result()
                if ok:
                    print(f"  Complete: {cohort}/{ab}", flush=True)
                else:
                    print(f"  Skip (no features): {cohort}/{ab}", flush=True)
            except Exception as e:
                print(f"  Error {cohort}/{age_band}: {e}", flush=True)

    _run_2019_distribution_script(cohort)
    print(f"\n{'='*60}")
    print(f"Model preparation complete for {cohort}")
    print(f"{'='*60}")


def _upload_one_cohort_age_to_s3(cohort: str, age_band: str) -> None:
    """Upload one cohort/age_band directory to S3. Includes per-bin bin_models/ recursively."""
    import boto3
    s3_client = boto3.client("s3")
    bucket = "pgxdatalake"
    prefix = "gold/dashboard/models"
    age_band_fname = age_band.replace("-", "_")
    local_dir = OUTPUT_DIR / cohort / age_band_fname
    if not local_dir.exists():
        return
    base_s3 = f"{prefix}/{cohort}/{age_band_fname}"
    for file_path in local_dir.rglob("*"):
        if file_path.is_file():
            rel = file_path.relative_to(local_dir)
            s3_key = f"{base_s3}/{rel.as_posix()}"
            s3_client.upload_file(str(file_path), bucket, s3_key)


def upload_to_s3(cohort: str, age_bands: List[str]):
    """Upload prepared models to S3 (parallel over age_bands)."""
    try:
        import boto3  # noqa: F401
    except ImportError:
        print("boto3 not available, skipping S3 upload")
        return
    n_workers = max(1, os.cpu_count() or 1)
    print("\nUploading models to S3 (parallel)...")
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(_upload_one_cohort_age_to_s3, cohort, ab) for ab in age_bands]
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"  S3 upload error: {e}")
    print("S3 upload complete!")


def main():
    parser = argparse.ArgumentParser(
        description='Prepare models for Lambda container deployment (ECR)',
        epilog='Models will be placed in models/ directory for Docker build. '
               'Use --upload-s3 to also upload to S3 as backup/fallback.'
    )
    parser.add_argument('--cohort', choices=['opioid_ed', 'non_opioid_ed'],
                       help='Cohort to process')
    parser.add_argument('--all', action='store_true',
                       help='Process all cohorts')
    parser.add_argument('--upload-s3', action='store_true',
                       help='Also upload to S3 as backup/fallback (optional)')
    parser.add_argument('--force', action='store_true',
                       help='Clear S3 checkpoint (9_dashboard_models) so workflow Step 3 will re-run')
    
    args = parser.parse_args()
    
    if args.force:
        try:
            from py_helpers.checkpoint_utils import delete_step_checkpoint
            logger = logging.getLogger(__name__)
            if delete_step_checkpoint("9_dashboard_models", "all", "all", logger=logger):
                print("Cleared checkpoint: 9_dashboard_models (workflow Step 3 will re-run)")
        except Exception as e:
            print(f"Warning: could not clear checkpoint: {e}")

    if args.all:
        cohorts = [
            ('opioid_ed', OPIOID_ED_AGE_BANDS),
            ('non_opioid_ed', POLYPHARMACY_AGE_BANDS)
        ]
    elif args.cohort:
        if args.cohort == 'opioid_ed':
            cohorts = [('opioid_ed', OPIOID_ED_AGE_BANDS)]
        else:
            cohorts = [('non_opioid_ed', POLYPHARMACY_AGE_BANDS)]
    else:
        parser.print_help()
        return
    
    print("\n" + "="*60, flush=True)
    print("Preparing models for Lambda Container (ECR) deployment", flush=True)
    print("="*60, flush=True)
    print("Models will be placed in: models/", flush=True)
    print("This directory will be copied into Docker container image", flush=True)
    print("="*60 + "\n", flush=True)
    
    for cohort, age_bands in cohorts:
        prepare_models_for_cohort(cohort, age_bands)
        
        if args.upload_s3:
            print(f"\nUploading {cohort} models to S3 (backup)...")
            upload_to_s3(cohort, age_bands)
    
    print("\n" + "="*60)
    print("✓ Model preparation complete!")
    print("="*60)
    print("\nNext steps:")
    print("  1. Review models/ directory structure")
    print("  2. Build Docker image: docker build -t pgx-risk-dashboard .")
    print("  3. Push to ECR: ./docker_build.sh")
    print("  4. Create Lambda function from container image")
    print("="*60)


if __name__ == '__main__':
    main()

