#!/usr/bin/env python3
"""
Prepare all required outputs for Lambda Docker build.

This script downloads/prepares all required files and organizes them in lambda_dir/
for inclusion in the Docker container image.

Usage:
    python prepare_lambda_dir.py [--download-s3] [--source-local]
"""

import os
import sys
import argparse
import shutil
import subprocess
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Dict, Optional

# Add project root to path
# This script is in 10_risk_dashboard/deployment/
# Project root is 3 levels up
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Configuration
LAMBDA_DIR = PROJECT_ROOT / "10_risk_dashboard" / "lambda_dir"
# Local EC2 path: models written by 6_final_model training pipeline.
# Only available on the EC2 instance where training was run.
MODELS_SOURCE = PROJECT_ROOT / "10_risk_dashboard" / "outputs" / "models"
MODELS_EC2_SOURCE = PROJECT_ROOT / "6_final_model" / "outputs"
METADATA_SOURCE = PROJECT_ROOT / "10_risk_dashboard" / "outputs" / "metadata"
DATA_SOURCE = PROJECT_ROOT / "10_risk_dashboard" / "outputs" / "cpic"

# Required cohorts and age bands (each cohort has all age bands)
from py_helpers.constants import REQUIRED_COHORTS
from py_helpers.env_utils import get_workflow_python_bin

# S3 paths
S3_BUCKET = "pgxdatalake"
S3_MODELS_PREFIX = "gold/dashboard/models"
S3_METADATA_PREFIX = "gold/dashboard/metadata"
S3_DATA_PREFIX = "gold/dashboard/data"


def log(message: str):
    """Print log message."""
    print(f"[INFO] {message}")


def error(message: str):
    """Print error message."""
    print(f"[ERROR] {message}", file=sys.stderr)


def download_from_s3(s3_key: str, local_path: Path, bucket: str = S3_BUCKET) -> bool:
    """Download file from S3."""
    try:
        import boto3
        s3_client = boto3.client("s3")
        local_path.parent.mkdir(parents=True, exist_ok=True)
        s3_client.download_file(bucket, s3_key, str(local_path))
        return True
    except ImportError:
        error("boto3 not available. Cannot download from S3.")
        return False
    except Exception as e:
        error(f"Failed to download s3://{bucket}/{s3_key}: {e}")
        return False


def copy_file(source: Path, dest: Path) -> bool:
    """Copy file from source to destination."""
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)
        return True
    except Exception as e:
        error(f"Failed to copy {source} -> {dest}: {e}")
        return False


def _local_source_for(cohort: str, age_band_fname: str) -> Optional[Path]:
    """Return the first local EC2 model directory that exists, or None."""
    candidates = [
        MODELS_SOURCE / cohort / age_band_fname,
        MODELS_EC2_SOURCE / cohort / age_band_fname,
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def _prepare_models_one_cohort_age(
    cohort: str, age_band: str, download_s3: bool
) -> tuple:
    """Prepare one cohort/age_band: S3 first, local EC2 fallback, fail if neither.

    Returns (cohort, age_band, success, missing_reason).
    """
    age_band_fname = age_band.replace("-", "_")
    dest_dir = LAMBDA_DIR / "models" / cohort / age_band_fname
    dest_dir.mkdir(parents=True, exist_ok=True)
    _DENSITY_BINS = ("low", "medium", "high", "extreme")
    s3_prefix = f"{S3_MODELS_PREFIX}/{cohort}/{age_band_fname}"
    model_files = [
        "catboost.joblib", "xgboost.joblib", "xgboost.json", "feature_schema.json",
        "risk_distribution_2019.json", "n_event_bin_thresholds.json",
        "calibration_xgboost.joblib", "calibration_xgboost_rf.joblib",
        "calibration_catboost.joblib", "calibration_diagnostics.json",
    ]
    bin_model_files = [
        "catboost.joblib", "xgboost.joblib", "xgboost_rf.joblib",
        "calibration_xgboost.joblib", "calibration_xgboost_rf.joblib", "calibration_catboost.joblib",
    ]

    local_src = _local_source_for(cohort, age_band_fname)

    def _get(s3_key: str, local_path: Path, local_fallback: Optional[Path]) -> bool:
        """S3 first (unless download_s3=False); local EC2 fallback; return True if obtained."""
        if download_s3 and download_from_s3(s3_key, local_path):
            return True
        if local_fallback and local_fallback.exists():
            return copy_file(local_fallback, local_path)
        return False

    # Flat model files
    got_schema = False
    for model_file in model_files:
        local_fallback = (local_src / model_file) if local_src else None
        ok = _get(f"{s3_prefix}/{model_file}", dest_dir / model_file, local_fallback)
        if model_file == "feature_schema.json" and ok:
            got_schema = True

    # Per-bin model files
    for _bin in _DENSITY_BINS:
        for _fname in bin_model_files:
            local_fallback = (local_src / "bin_models" / _bin / _fname) if local_src else None
            _get(
                f"{s3_prefix}/bin_models/{_bin}/{_fname}",
                dest_dir / "bin_models" / _bin / _fname,
                local_fallback,
            )

    if not got_schema:
        reason = "not in S3" + (" and local EC2 path not found" if local_src is None else f" and not in {local_src}")
        return (cohort, age_band, False, reason)
    return (cohort, age_band, True, "")


def prepare_models(download_s3: bool = True) -> bool:
    """Prepare models: S3 first, local EC2 fallback. Fails if feature_schema missing."""
    log("Preparing models (S3 primary, local EC2 fallback)...")
    if not download_s3:
        log("  NOTE: --no-s3 set; skipping S3 and using local EC2 paths only.")
    models_dest = LAMBDA_DIR / "models"
    models_dest.mkdir(parents=True, exist_ok=True)
    tasks = [
        (cohort, age_band)
        for cohort, age_bands in REQUIRED_COHORTS.items()
        for age_band in age_bands
    ]
    n_workers = max(1, os.cpu_count() or 1)
    success = True
    failures = []
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(_prepare_models_one_cohort_age, c, ab, download_s3): (c, ab)
            for c, ab in tasks
        }
        for future in as_completed(futures):
            c, ab = futures[future]
            try:
                _, _, ok, reason = future.result()
                if not ok:
                    failures.append((c, ab, reason))
                    success = False
            except Exception as e:
                failures.append((c, ab, str(e)))
                success = False
    if not success:
        error("Models not found for the following cohort/age_band combinations:")
        for c, ab, reason in failures:
            error(f"  {c}/{ab}: {reason}")
        error("")
        error("This build script must be run where model artifacts are available:")
        error("  1. S3 (pgxdatalake/gold/dashboard/models/): upload via prepare_models.py --upload-s3")
        error("  2. Local EC2 path (6_final_model/outputs/): run training pipeline (Step 6) on EC2 first")
        error("")
        error("If running locally without EC2 training outputs, run the Docker build on EC2.")
    return success


def run_generate_metrics(download_s3: bool = False) -> bool:
    """Run generate_metrics.py so outputs/metadata/model_performance_metrics.json exists (for ECR bundle)."""
    log("Generating model performance metrics...")
    metrics_script = PROJECT_ROOT / "10_risk_dashboard" / "data_preparation" / "generate_metrics.py"
    if not metrics_script.exists():
        log("  generate_metrics.py not found; skipping metrics bundle.")
        return True
    cmd = [sys.executable, str(metrics_script)]
    if download_s3:
        cmd.append("--download-s3")
    try:
        r = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True, timeout=60)
        if r.returncode != 0:
            log(f"  Warning: generate_metrics.py exited {r.returncode} (metrics may be missing in image).")
            if r.stderr:
                log(f"  {r.stderr.strip()[:200]}")
            return True  # non-fatal
        log("  [OK] model_performance_metrics.json written to outputs/metadata/")
        return True
    except Exception as e:
        log(f"  Warning: could not run generate_metrics.py: {e}")
        return True  # non-fatal


def prepare_metadata(download_s3: bool = False) -> bool:
    """Prepare metadata directory."""
    log("Preparing metadata...")
    
    metadata_dest = LAMBDA_DIR / "metadata"
    metadata_dest.mkdir(parents=True, exist_ok=True)
    
    success = True
    
    for cohort in REQUIRED_COHORTS.keys():
        metadata_file = f"metadata_{cohort}.json"
        source_path = METADATA_SOURCE / metadata_file
        dest_path = metadata_dest / metadata_file
        
        if download_s3:
            # Download from S3
            log(f"  Downloading {cohort} metadata from S3...")
            s3_key = f"{S3_METADATA_PREFIX}/{metadata_file}"
            if download_from_s3(s3_key, dest_path):
                log(f"    [OK] Downloaded {metadata_file}")
            else:
                # Try local fallback
                if source_path.exists():
                    copy_file(source_path, dest_path)
                    log(f"    [OK] Copied {metadata_file} from local")
                else:
                    error(f"  Metadata not found: {metadata_file}")
                    success = False
        else:
            # Copy from local
            if not source_path.exists():
                error(f"  Metadata not found: {source_path}")
                success = False
                continue
            
            log(f"  Copying {cohort} metadata...")
            if copy_file(source_path, dest_path):
                log(f"    [OK] Copied {metadata_file}")
            else:
                success = False
    
    # Copy model performance metrics (optional; Lambda prefers S3 at gold/dashboard/metadata/model_performance_metrics.json)
    metrics_file = "model_performance_metrics.json"
    metrics_source = METADATA_SOURCE / metrics_file
    if metrics_source.exists():
        if copy_file(metrics_source, metadata_dest / metrics_file):
            log(f"  [OK] Copied {metrics_file} (fallback; Lambda serves from S3 when available)")
    
    return success


def prepare_data(download_s3: bool = False) -> bool:
    """Prepare data directory (CPIC Excel and Parquet for Lambda)."""
    log("Preparing data (CPIC Excel + Parquet)...")

    data_dest = LAMBDA_DIR / "data"
    data_dest.mkdir(parents=True, exist_ok=True)

    cpic_xlsx = "cpic_gene-drug_pairs.xlsx"
    cpic_parquet = "cpic_gene-drug_pairs.parquet"
    source_xlsx = DATA_SOURCE / cpic_xlsx
    source_parquet = DATA_SOURCE / cpic_parquet
    dest_xlsx = data_dest / cpic_xlsx
    dest_parquet = data_dest / cpic_parquet

    if download_s3:
        log("  Downloading CPIC data from S3...")
        if download_from_s3(f"{S3_DATA_PREFIX}/{cpic_xlsx}", dest_xlsx):
            log(f"    [OK] Downloaded {cpic_xlsx}")
        if download_from_s3(f"{S3_DATA_PREFIX}/{cpic_parquet}", dest_parquet):
            log(f"    [OK] Downloaded {cpic_parquet}")
        if dest_xlsx.exists():
            return True
        if source_xlsx.exists():
            copy_file(source_xlsx, dest_xlsx)
            log(f"    [OK] Copied {cpic_xlsx} from local")
            return True

    # Copy from local; if missing, try to run prepare_cpic_data once
    if not source_xlsx.exists():
        log("  CPIC file missing; running prepare_cpic_data.py...")
        try:
            r = subprocess.run(
                [str(get_workflow_python_bin()), str(PROJECT_ROOT / "10_risk_dashboard" / "data_preparation" / "prepare_cpic_data.py")],
                cwd=str(PROJECT_ROOT),
                capture_output=True,
                text=True,
                timeout=120,
            )
            if r.returncode != 0:
                error(f"  CPIC data not found: {source_xlsx}")
                if r.stderr:
                    error(f"  prepare_cpic_data: {r.stderr.strip()[:200]}")
                error("  Run: python 10_risk_dashboard/data_preparation/prepare_cpic_data.py")
                return False
            if not source_xlsx.exists():
                error(f"  CPIC data still not found after prepare_cpic_data: {source_xlsx}")
                return False
            log("  [OK] prepare_cpic_data produced CPIC file")
        except Exception as e:
            error(f"  CPIC data not found: {source_xlsx}")
            error(f"  Failed to run prepare_cpic_data: {e}")
            error("  Run: python 10_risk_dashboard/data_preparation/prepare_cpic_data.py")
            return False

    log("  Copying CPIC data...")
    ok = copy_file(source_xlsx, dest_xlsx)
    if ok:
        log(f"    [OK] Copied {cpic_xlsx}")
    if source_parquet.exists():
        if copy_file(source_parquet, dest_parquet):
            log(f"    [OK] Copied {cpic_parquet} (DuckDB will use this in Lambda)")
    return ok


def verify_lambda_dir() -> bool:
    """Verify that lambda_dir has all required files."""
    log("Verifying lambda_dir contents...")
    
    issues = []
    
    # Check models
    models_dir = LAMBDA_DIR / "models"
    if not models_dir.exists():
        issues.append("models/ directory missing")
    else:
        for cohort, age_bands in REQUIRED_COHORTS.items():
            for age_band in age_bands:
                age_band_fname = age_band.replace("-", "_")
                cohort_dir = models_dir / cohort / age_band_fname
                
                if not cohort_dir.exists():
                    issues.append(f"models/{cohort}/{age_band_fname}/ missing")
                    continue
                
                # Check for required files
                required_files = ["feature_schema.json"]
                for req_file in required_files:
                    if not (cohort_dir / req_file).exists():
                        issues.append(f"models/{cohort}/{age_band_fname}/{req_file} missing")
                
                # Check for at least one model file
                model_files = list(cohort_dir.glob("*.joblib")) + list(cohort_dir.glob("*.json"))
                if not model_files:
                    issues.append(f"models/{cohort}/{age_band_fname}/ has no model files")
    
    # Check metadata
    metadata_dir = LAMBDA_DIR / "metadata"
    if not metadata_dir.exists():
        issues.append("metadata/ directory missing")
    else:
        for cohort in REQUIRED_COHORTS.keys():
            metadata_file = metadata_dir / f"metadata_{cohort}.json"
            if not metadata_file.exists():
                issues.append(f"metadata/metadata_{cohort}.json missing")
    
    # Check data
    data_dir = LAMBDA_DIR / "data"
    if not data_dir.exists():
        issues.append("data/ directory missing")
    else:
        cpic_xlsx = data_dir / "cpic_gene-drug_pairs.xlsx"
        cpic_parquet = data_dir / "cpic_gene-drug_pairs.parquet"
        if not cpic_xlsx.exists() and not cpic_parquet.exists():
            issues.append("data/ missing cpic_gene-drug_pairs.xlsx or cpic_gene-drug_pairs.parquet")
    
    if issues:
        error("Verification failed:")
        for issue in issues:
            error(f"  - {issue}")
        return False
    
    log("[OK] All required files present in lambda_dir/")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Prepare all required outputs for Lambda Docker build"
    )
    parser.add_argument(
        "--no-s3",
        action="store_true",
        help="Skip S3 and use only local EC2 paths (6_final_model/outputs/). Fails if local paths absent."
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify lambda_dir contents, don't prepare"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Clear S3 checkpoints (9_dashboard_metadata, 9_dashboard_models) so workflow Steps 1a and 3 will re-run"
    )
    
    args = parser.parse_args()
    
    if args.force:
        try:
            from py_helpers.checkpoint_utils import delete_step_checkpoint
            import logging
            _log = logging.getLogger(__name__)
            for step in ("9_dashboard_metadata", "9_dashboard_models"):
                if delete_step_checkpoint(step, "all", "all", logger=_log):
                    log(f"Cleared checkpoint: {step}")
        except Exception as e:
            error(f"Could not clear checkpoints: {e}")

    log("=" * 60)
    log("Preparing Lambda Directory for Docker Build")
    log("=" * 60)
    log("")
    
    if args.verify_only:
        success = verify_lambda_dir()
        sys.exit(0 if success else 1)

    download_s3 = not args.no_s3  # S3 is default; --no-s3 disables it

    # Create lambda_dir
    LAMBDA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Run metrics and metadata in parallel (independent), then models and data
    success = True
    with ThreadPoolExecutor(max_workers=2) as executor:
        f_metrics = executor.submit(run_generate_metrics, download_s3)
        f_meta = executor.submit(prepare_metadata, download_s3)
        success = f_metrics.result() and f_meta.result()
    success &= prepare_models(download_s3=download_s3)
    success &= prepare_data(download_s3=download_s3)
    
    log("")
    log("=" * 60)
    
    if success:
        log("[OK] Lambda directory preparation complete!")
        log("")
        log("Directory structure:")
        log(f"  {LAMBDA_DIR}/")
        log("  +-- models/")
        log("  |   +-- opioid_ed/")
        log("  |   |   +-- {age_band}/")
        log("  |   +-- non_opioid_ed/")
        log("  |       +-- {age_band}/")
        log("  +-- metadata/")
        log("  |   +-- metadata_opioid_ed.json")
        log("  |   +-- metadata_non_opioid_ed.json")
        log("  |   +-- model_performance_metrics.json")
        log("  +-- data/")
        log("      +-- cpic_gene-drug_pairs.xlsx (+ .parquet if prepared)")
        log("")
        log("Next steps:")
        log("  1. Verify: python prepare_lambda_dir.py --verify-only")
        log("  2. Build Docker: cd 10_risk_dashboard && docker build -t pgx-risk-dashboard .")
        log("  3. Push to ECR: ./docker_build.sh")
    else:
        error("Lambda directory preparation failed!")
        error("Please check errors above and ensure all required outputs are available.")
        error("")
        error("Prerequisites (run from project root):")
        error("  1. Models:  python 10_risk_dashboard/data_preparation/prepare_models.py --all")
        error("             (requires 6_final_model outputs)")
        error("  2. CPIC:   python 10_risk_dashboard/data_preparation/prepare_cpic_data.py")
        error("  3. Metadata: generated by workflow Step 1a (generate_metadata.py)")
        sys.exit(1)


if __name__ == "__main__":
    main()
