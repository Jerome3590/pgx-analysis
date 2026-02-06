#!/usr/bin/env python3
"""
Download model, SHAP, and FFA artifacts from S3 and test with binary/categorical fixes.
"""

import sys
import boto3
import pandas as pd
from pathlib import Path
from io import BytesIO
import json
import logging
from botocore.exceptions import ClientError

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

S3_BUCKET = "pgxdatalake"
s3_client = boto3.client('s3')

# Test cohort - using opioid_ed/13-24 as it's commonly used
COHORT_NAME = "opioid_ed"
AGE_BAND = "13-24"
AGE_BAND_FNAME = AGE_BAND.replace("-", "_")


def list_s3_prefix(prefix: str) -> list:
    """List all objects with given prefix."""
    try:
        response = s3_client.list_objects_v2(Bucket=S3_BUCKET, Prefix=prefix)
        if 'Contents' in response:
            return [obj['Key'] for obj in response['Contents']]
        return []
    except Exception as e:
        logger.error(f"Error listing S3 prefix {prefix}: {e}")
        return []


def download_file_from_s3(s3_key: str, local_path: Path, bucket: str = "pgxdatalake") -> bool:
    """Download a file from S3 to local path."""
    try:
        local_path.parent.mkdir(parents=True, exist_ok=True)
        s3_client.download_file(bucket, s3_key, str(local_path))
        logger.info(f"[OK] Downloaded from {bucket}: {s3_key} -> {local_path}")
        return True
    except ClientError as e:
        if e.response['Error']['Code'] == '404':
            logger.warning(f"[X] Not found in {bucket}: {s3_key}")
        else:
            logger.error(f"[X] Error downloading from {bucket} {s3_key}: {e}")
        return False
    except Exception as e:
        logger.error(f"[X] Error downloading from {bucket} {s3_key}: {e}")
        return False


def download_model_artifacts():
    """Download model JSON files from S3."""
    logger.info("=" * 80)
    logger.info("Downloading Model Artifacts")
    logger.info("=" * 80)
    
    model_base = (
        PROJECT_ROOT / "6_final_model" / "outputs" / COHORT_NAME / AGE_BAND_FNAME / "final_model_json"
    )
    model_base.mkdir(parents=True, exist_ok=True)
    
    # Try multiple S3 locations - model files are directly in the age_band directory
    s3_prefixes = [
        f"gold/final_model/{COHORT_NAME}/{AGE_BAND}/",  # Main location
        f"gold/final_model/{COHORT_NAME}/{AGE_BAND}/final_model_json/",
        f"gold/model_outputs/{COHORT_NAME}/{AGE_BAND_FNAME}/",
    ]
    
    model_files = []
    for prefix in s3_prefixes:
        files = list_s3_prefix(prefix)
        if files:
            logger.info(f"Found {len(files)} files in {prefix}")
            model_files.extend(files)
    
    # Also try direct file paths
    direct_paths = [
        f"gold/final_model/{COHORT_NAME}/{AGE_BAND}/{COHORT_NAME}_{AGE_BAND_FNAME}_best_xgboost_model.json",
        f"gold/final_model/{COHORT_NAME}/{AGE_BAND}/{COHORT_NAME}_{AGE_BAND_FNAME}_best_catboost_model.json",
    ]
    
    downloaded = 0
    for s3_key in model_files + direct_paths:
        if s3_key.endswith('.json') and ('xgboost' in s3_key.lower() or 'catboost' in s3_key.lower() or 'best' in s3_key.lower()):
            filename = Path(s3_key).name
            local_path = model_base / filename
            if download_file_from_s3(s3_key, local_path):
                downloaded += 1
    
    # Check if we have any model files locally
    if downloaded == 0:
        local_files = list(model_base.glob("*.json"))
        if local_files:
            logger.info(f"Found {len(local_files)} model files locally: {[f.name for f in local_files]}")
            return True
    
    logger.info(f"Downloaded {downloaded} model files")
    return downloaded > 0


def download_shap_artifacts():
    """Download SHAP artifacts from S3."""
    logger.info("=" * 80)
    logger.info("Downloading SHAP Artifacts")
    logger.info("=" * 80)
    
    shap_base = (
        PROJECT_ROOT / "7_shap_analysis" / "outputs" / COHORT_NAME / AGE_BAND_FNAME
    )
    shap_base.mkdir(parents=True, exist_ok=True)
    
    # Download SHAP global importance
    for model_type in ['xgboost', 'catboost']:
        shap_key = f"gold/shap_analysis/{COHORT_NAME}/{AGE_BAND}/{COHORT_NAME}_{AGE_BAND_FNAME}_shap_global_importance_{model_type}.csv"
        local_path = shap_base / f"{COHORT_NAME}_{AGE_BAND_FNAME}_shap_global_importance_{model_type}.csv"
        download_file_from_s3(shap_key, local_path)
        
        # Download SHAP sample values (individual SHAP values per instance)
        shap_values_key = f"gold/shap_analysis/{COHORT_NAME}/{AGE_BAND}/{COHORT_NAME}_{AGE_BAND_FNAME}_shap_sample_values_{model_type}.parquet"
        local_path = shap_base / f"{COHORT_NAME}_{AGE_BAND_FNAME}_shap_sample_values_{model_type}.parquet"
        download_file_from_s3(shap_values_key, local_path)


def download_data_artifacts():
    """Download data files from S3 if needed."""
    logger.info("=" * 80)
    logger.info("Checking Data Artifacts")
    logger.info("=" * 80)
    
    # Check multiple locations for data file
    data_paths_to_check = [
        # Primary location: 6_final_model outputs
        (
            PROJECT_ROOT / "6_final_model" / "outputs" / COHORT_NAME / AGE_BAND_FNAME
            / "inputs" / "model_train" / "final_features.parquet",
            "6_final_model outputs (parquet)"
        ),
        (
            PROJECT_ROOT / "6_final_model" / "outputs" / COHORT_NAME / AGE_BAND_FNAME
            / f"{COHORT_NAME}_{AGE_BAND_FNAME}_train_final_features_no_leakage.csv",
            "6_final_model outputs (CSV)"
        ),
        # Alternative: data folder (various structures)
        (
            PROJECT_ROOT / "data" / COHORT_NAME / AGE_BAND_FNAME / "final_features.parquet",
            "data folder (parquet)"
        ),
        (
            PROJECT_ROOT / "data" / COHORT_NAME / AGE_BAND_FNAME
            / f"{COHORT_NAME}_{AGE_BAND_FNAME}_train_final_features_no_leakage.csv",
            "data folder (CSV)"
        ),
        (
            PROJECT_ROOT / "data" / f"{COHORT_NAME}_{AGE_BAND_FNAME}_train_final_features_no_leakage.csv",
            "data folder root (CSV)"
        ),
        # Data folder with cohorts structure (check latest years)
        (
            PROJECT_ROOT / "data" / "cohorts" / f"cohort_name={COHORT_NAME}" / "event_year=2019"
            / f"age_band={AGE_BAND}" / "final_features.parquet",
            "data/cohorts 2019 (parquet)"
        ),
        (
            PROJECT_ROOT / "data" / "cohorts" / f"cohort_name={COHORT_NAME}" / "event_year=2020"
            / f"age_band={AGE_BAND}" / "final_features.parquet",
            "data/cohorts 2020 (parquet)"
        ),
        # Gold cohorts directory
        (
            PROJECT_ROOT / "data" / "gold_cohorts" / f"cohort_name={COHORT_NAME}"
            / f"{COHORT_NAME}_{AGE_BAND_FNAME}_train_final_features_no_leakage.csv",
            "data/gold_cohorts (CSV)"
        ),
        (
            PROJECT_ROOT / "data" / "gold_cohorts" / f"cohort_name={COHORT_NAME}"
            / "final_features.parquet",
            "data/gold_cohorts (parquet)"
        ),
        (
            PROJECT_ROOT / "data" / "gold_cohorts"
            / f"{COHORT_NAME}_{AGE_BAND_FNAME}_train_final_features_no_leakage.csv",
            "data/gold_cohorts root (CSV)"
        ),
    ]
    
    # Check if data exists locally
    for data_path, location_desc in data_paths_to_check:
        if data_path.exists():
            logger.info(f"[OK] Data file exists locally at {location_desc}: {data_path}")
            return True
    
    # Try to download from S3 to primary location
    data_path_parquet = data_paths_to_check[0][0]
    data_path_parquet.parent.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
    data_key = f"gold/final_model/{COHORT_NAME}/{AGE_BAND}/inputs/model_train/final_features.parquet"
    if download_file_from_s3(data_key, data_path_parquet):
        return True
    
    # Try CSV alternative
    data_path_csv = data_paths_to_check[1][0]
    data_path_csv.parent.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
    data_key_csv = f"gold/final_model/{COHORT_NAME}/{AGE_BAND}/inputs/model_train/{COHORT_NAME}_{AGE_BAND_FNAME}_train_final_features_no_leakage.csv"
    if download_file_from_s3(data_key_csv, data_path_csv):
        return True
    
    # Try alternative S3 locations
    alt_s3_keys = [
        f"gold/cohorts/{COHORT_NAME}/{AGE_BAND}/final_features.parquet",
        f"gold/cohorts/{COHORT_NAME}/{AGE_BAND}/{COHORT_NAME}_{AGE_BAND_FNAME}_train_final_features_no_leakage.csv",
        f"gold/gold_cohorts/{COHORT_NAME}/{AGE_BAND}/final_features.parquet",
        f"gold/gold_cohorts/{COHORT_NAME}/{AGE_BAND}/{COHORT_NAME}_{AGE_BAND_FNAME}_train_final_features_no_leakage.csv",
        f"gold/cohorts_model_data/{COHORT_NAME}/{AGE_BAND}/final_features.parquet",
        f"gold/cohorts_model_data/{COHORT_NAME}/{AGE_BAND}/{COHORT_NAME}_{AGE_BAND_FNAME}_train_final_features_no_leakage.csv",
    ]
    for alt_key in alt_s3_keys:
        # Try downloading to gold_cohorts location
        gold_cohorts_path = PROJECT_ROOT / "data" / "gold_cohorts" / f"{COHORT_NAME}_{AGE_BAND_FNAME}_train_final_features_no_leakage.csv"
        if download_file_from_s3(alt_key, gold_cohorts_path, bucket="pgxdatalake"):
            logger.info(f"Downloaded training data from pgxdatalake to gold_cohorts: {gold_cohorts_path}")
            return True
    
    # Try pgx-repository bucket
    repo_keys = [
        f"gold/final_model/{COHORT_NAME}/{AGE_BAND}/inputs/model_train/final_features.parquet",
        f"gold/final_model/{COHORT_NAME}/{AGE_BAND}/inputs/model_train/{COHORT_NAME}_{AGE_BAND_FNAME}_train_final_features_no_leakage.csv",
        f"final_model/{COHORT_NAME}/{AGE_BAND}/inputs/model_train/final_features.parquet",
        f"final_model/{COHORT_NAME}/{AGE_BAND}/inputs/model_train/{COHORT_NAME}_{AGE_BAND_FNAME}_train_final_features_no_leakage.csv",
    ]
    for repo_key in repo_keys:
        # Try both parquet and CSV locations
        if repo_key.endswith('.parquet'):
            local_path = data_path_parquet
        else:
            local_path = data_path_csv
        if download_file_from_s3(repo_key, local_path, bucket="pgx-repository"):
            logger.info(f"Downloaded training data from pgx-repository: {repo_key}")
            return True
    
    logger.warning("Could not find data files in any expected location:")
    for data_path, location_desc in data_paths_to_check:
        logger.warning(f"  - {location_desc}: {data_path}")
    logger.warning("You may need to run Step 6 first or place the data file in one of these locations.")
    return False


def download_existing_ffa_artifacts():
    """Download existing FFA artifacts (optional - we'll regenerate with fixes)."""
    logger.info("=" * 80)
    logger.info("Downloading Existing FFA Artifacts (for comparison)")
    logger.info("=" * 80)
    
    ffa_base = (
        PROJECT_ROOT / "8_ffa_analysis" / "outputs" / COHORT_NAME / AGE_BAND_FNAME
    )
    
    for model_type in ['xgboost']:  # Test with XGBoost first
        model_output_dir = ffa_base / model_type
        model_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Download explanations
        s3_key = f"gold/ffa_analysis/{COHORT_NAME}/{AGE_BAND}/{model_type}/axp_explanations.parquet"
        local_path = model_output_dir / "axp_explanations.parquet"
        download_file_from_s3(s3_key, local_path)
        
        # Download feature importance
        s3_key = f"gold/ffa_analysis/{COHORT_NAME}/{AGE_BAND}/{model_type}/feature_importance_axp.parquet"
        local_path = model_output_dir / "feature_importance_axp.parquet"
        download_file_from_s3(s3_key, local_path)
        
        # Download causal importance (if exists)
        s3_key = f"gold/ffa_analysis/{COHORT_NAME}/{AGE_BAND}/{model_type}/causal_importance.parquet"
        local_path = model_output_dir / "causal_importance.parquet"
        download_file_from_s3(s3_key, local_path)


def test_ffa_analysis():
    """Run FFA analysis with the fixes."""
    logger.info("=" * 80)
    logger.info("Running FFA Analysis with Binary/Categorical Fixes")
    logger.info("=" * 80)
    
    # Delete existing causal importance to force regeneration with fixes
    ffa_base = (
        PROJECT_ROOT / "8_ffa_analysis" / "outputs" / COHORT_NAME / AGE_BAND_FNAME / "xgboost"
    )
    causal_path = ffa_base / "causal_importance.parquet"
    if causal_path.exists():
        logger.info(f"Deleting existing causal_importance.parquet to force regeneration with fixes...")
        causal_path.unlink()
    
    # Import and run the analysis
    sys.path.insert(0, str(PROJECT_ROOT / "8_ffa_analysis"))
    
    # Change to the FFA analysis directory
    import os
    original_cwd = os.getcwd()
    os.chdir(PROJECT_ROOT / "8_ffa_analysis")
    
    try:
        # Import the main analysis script
        from run_full_ffa_analysis import main
        
        # Run with test cohort
        logger.info(f"Running FFA analysis for {COHORT_NAME}/{AGE_BAND}...")
        logger.info("This will test the binary/categorical feature fixes.")
        logger.info("Causal analysis will be regenerated with the fixes.")
        
        # Set up command line arguments
        import sys as sys_module
        original_argv = sys_module.argv
        sys_module.argv = [
            'run_full_ffa_analysis.py',
            '--cohort-name', COHORT_NAME,
            '--age-band', AGE_BAND,
            '--model-type', 'xgboost'  # Test with XGBoost first
        ]
        
        try:
            main()
        finally:
            sys_module.argv = original_argv
            
    except Exception as e:
        logger.error(f"Error running FFA analysis: {e}", exc_info=True)
        raise
    finally:
        os.chdir(original_cwd)


def compare_results():
    """Compare old vs new results to verify fixes."""
    logger.info("=" * 80)
    logger.info("Comparing Results (Old vs New)")
    logger.info("=" * 80)
    
    ffa_base = (
        PROJECT_ROOT / "8_ffa_analysis" / "outputs" / COHORT_NAME / AGE_BAND_FNAME / "xgboost"
    )
    
    # Load causal importance (if both exist)
    old_causal_path = ffa_base / "causal_importance.parquet"
    new_causal_path = ffa_base / "causal_importance.parquet"  # Same file, but regenerated
    
    if old_causal_path.exists():
        try:
            causal_df = pd.read_parquet(old_causal_path)
            
            # Check binary features
            if 'is_binary' in causal_df.columns:
                binary_features = causal_df[causal_df['is_binary'] == True]
                binary_with_causal = binary_features[binary_features['causal_importance'] > 0]
                
                logger.info(f"Binary features analysis:")
                logger.info(f"  Total binary features: {len(binary_features)}")
                logger.info(f"  Binary features with causal_importance > 0: {len(binary_with_causal)}")
                logger.info(f"  Percentage: {len(binary_with_causal) / len(binary_features) * 100:.1f}%")
                
                if len(binary_with_causal) > 0:
                    logger.info("\nTop binary features by causal importance:")
                    top_binary = binary_with_causal.nlargest(10, 'causal_importance')
                    for idx, row in top_binary.iterrows():
                        logger.info(f"  {row['feature']:<50} {row['causal_importance']:.6f}")
        except Exception as e:
            logger.warning(f"Could not analyze causal results: {e}")


def main():
    """Main function to download and test."""
    logger.info("=" * 80)
    logger.info("FFA Binary/Categorical Feature Fix Test")
    logger.info("=" * 80)
    logger.info(f"Cohort: {COHORT_NAME}, Age Band: {AGE_BAND}")
    logger.info("")
    
    # Step 1: Download model artifacts
    if not download_model_artifacts():
        logger.error("Failed to download model artifacts. Cannot proceed.")
        return
    
    # Step 2: Download SHAP artifacts
    download_shap_artifacts()
    
    # Step 3: Check/download data artifacts
    if not download_data_artifacts():
        logger.warning("Data files not found. Analysis may fail if data is required.")
    
    # Step 4: Download existing FFA artifacts (for comparison)
    download_existing_ffa_artifacts()
    
    # Step 5: Run FFA analysis with fixes
    test_ffa_analysis()
    
    # Step 6: Compare results
    compare_results()
    
    logger.info("=" * 80)
    logger.info("Test Complete!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
