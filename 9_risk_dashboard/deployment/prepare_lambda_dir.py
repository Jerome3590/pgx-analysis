#!/usr/bin/env python3
"""
Prepare all required outputs for Lambda Docker build.

This script downloads/prepares all required files and organizes them in lambda_dir/
for inclusion in the Docker container image.

Usage:
    python prepare_lambda_dir.py [--download-s3] [--source-local]
"""

import sys
import argparse
import shutil
import json
from pathlib import Path
from typing import List, Dict, Optional

# Add project root to path
# This script is in 9_risk_dashboard/deployment/
# Project root is 3 levels up
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Configuration
# Note: With new structure, Dockerfile copies directly from outputs/
# This script is kept for backward compatibility or manual preparation
LAMBDA_DIR = PROJECT_ROOT / "9_risk_dashboard" / "lambda_dir"
MODELS_SOURCE = PROJECT_ROOT / "9_risk_dashboard" / "outputs" / "models"
METADATA_SOURCE = PROJECT_ROOT / "9_risk_dashboard" / "outputs" / "metadata"
DATA_SOURCE = PROJECT_ROOT / "9_risk_dashboard" / "outputs" / "cpic"

# Required cohorts and age bands
REQUIRED_COHORTS = {
    "opioid_ed": ["13-24", "25-44", "45-54", "55-64"],
    "non_opioid_ed": ["65-74", "75-84", "85-94"],
}

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


def prepare_models(download_s3: bool = False) -> bool:
    """Prepare models directory."""
    log("Preparing models...")
    
    models_dest = LAMBDA_DIR / "models"
    models_dest.mkdir(parents=True, exist_ok=True)
    
    success = True
    
    for cohort, age_bands in REQUIRED_COHORTS.items():
        for age_band in age_bands:
            age_band_fname = age_band.replace("-", "_")
            
            # Source paths
            source_dir = MODELS_SOURCE / cohort / age_band_fname
            dest_dir = models_dest / cohort / age_band_fname
            
            if download_s3:
                # Download from S3
                log(f"  Downloading {cohort}/{age_band} models from S3...")
                s3_prefix = f"{S3_MODELS_PREFIX}/{cohort}/{age_band_fname}"
                
                # Download model files
                for model_file in ["catboost.joblib", "xgboost.joblib", "xgboost.json", "feature_schema.json"]:
                    s3_key = f"{s3_prefix}/{model_file}"
                    local_path = dest_dir / model_file
                    if download_from_s3(s3_key, local_path):
                        log(f"    ✓ Downloaded {model_file}")
                    else:
                        # Try local fallback
                        local_source = source_dir / model_file
                        if local_source.exists():
                            copy_file(local_source, local_path)
                            log(f"    ✓ Copied {model_file} from local")
            else:
                # Copy from local
                if not source_dir.exists():
                    error(f"  Models not found: {source_dir}")
                    success = False
                    continue
                
                log(f"  Copying {cohort}/{age_band} models...")
                dest_dir.mkdir(parents=True, exist_ok=True)
                
                # Copy all files from source
                for file_path in source_dir.glob("*"):
                    if file_path.is_file():
                        dest_path = dest_dir / file_path.name
                        if copy_file(file_path, dest_path):
                            log(f"    ✓ Copied {file_path.name}")
                        else:
                            success = False
    
    return success


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
                log(f"    ✓ Downloaded {metadata_file}")
            else:
                # Try local fallback
                if source_path.exists():
                    copy_file(source_path, dest_path)
                    log(f"    ✓ Copied {metadata_file} from local")
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
                log(f"    ✓ Copied {metadata_file}")
            else:
                success = False
    
    return success


def prepare_data(download_s3: bool = False) -> bool:
    """Prepare data directory (CPIC Excel file)."""
    log("Preparing data (CPIC Excel)...")
    
    data_dest = LAMBDA_DIR / "data"
    data_dest.mkdir(parents=True, exist_ok=True)
    
    cpic_file = "cpic_gene-drug_pairs.xlsx"
    source_path = DATA_SOURCE / cpic_file
    dest_path = data_dest / cpic_file
    
    if download_s3:
        # Download from S3
        log(f"  Downloading CPIC data from S3...")
        s3_key = f"{S3_DATA_PREFIX}/{cpic_file}"
        if download_from_s3(s3_key, dest_path):
            log(f"    ✓ Downloaded {cpic_file}")
            return True
        else:
            # Try local fallback
            if source_path.exists():
                copy_file(source_path, dest_path)
                log(f"    ✓ Copied {cpic_file} from local")
                return True
    
    # Copy from local
    if not source_path.exists():
        error(f"  CPIC data not found: {source_path}")
        error("  Run: python prepare_cpic_data.py")
        return False
    
    log(f"  Copying CPIC data...")
    if copy_file(source_path, dest_path):
        log(f"    ✓ Copied {cpic_file}")
        return True
    
    return False


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
        cpic_file = data_dir / "cpic_gene-drug_pairs.xlsx"
        if not cpic_file.exists():
            issues.append("data/cpic_gene-drug_pairs.xlsx missing")
    
    if issues:
        error("Verification failed:")
        for issue in issues:
            error(f"  - {issue}")
        return False
    
    log("✓ All required files present in lambda_dir/")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Prepare all required outputs for Lambda Docker build"
    )
    parser.add_argument(
        "--download-s3",
        action="store_true",
        help="Download files from S3 (with local fallback)"
    )
    parser.add_argument(
        "--source-local",
        action="store_true",
        default=True,
        help="Use local files as source (default)"
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify lambda_dir contents, don't prepare"
    )
    
    args = parser.parse_args()
    
    log("=" * 60)
    log("Preparing Lambda Directory for Docker Build")
    log("=" * 60)
    log("")
    
    if args.verify_only:
        success = verify_lambda_dir()
        sys.exit(0 if success else 1)
    
    # Create lambda_dir
    LAMBDA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Prepare all components
    success = True
    
    success &= prepare_models(download_s3=args.download_s3)
    success &= prepare_metadata(download_s3=args.download_s3)
    success &= prepare_data(download_s3=args.download_s3)
    
    log("")
    log("=" * 60)
    
    if success:
        log("✓ Lambda directory preparation complete!")
        log("")
        log("Directory structure:")
        log(f"  {LAMBDA_DIR}/")
        log("  ├── models/")
        log("  │   ├── opioid_ed/")
        log("  │   │   └── {age_band}/")
        log("  │   └── non_opioid_ed/")
        log("  │       └── {age_band}/")
        log("  ├── metadata/")
        log("  │   ├── metadata_opioid_ed.json")
        log("  │   └── metadata_non_opioid_ed.json")
        log("  └── data/")
        log("      └── cpic_gene-drug_pairs.xlsx")
        log("")
        log("Next steps:")
        log("  1. Verify: python prepare_lambda_dir.py --verify-only")
        log("  2. Build Docker: cd 9_risk_dashboard && docker build -t pgx-risk-dashboard .")
        log("  3. Push to ECR: ./docker_build.sh")
    else:
        error("Lambda directory preparation failed!")
        error("Please check errors above and ensure all required outputs are available.")
        sys.exit(1)


if __name__ == "__main__":
    main()
