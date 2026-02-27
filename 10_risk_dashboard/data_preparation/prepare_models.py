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

import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import joblib
import pandas as pd
import numpy as np

# Add project root to path
# This script is in 10_risk_dashboard/data_preparation/
# Project root is 3 levels up
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from catboost import CatBoostClassifier
    import xgboost as xgb
    MODEL_LIBS_AVAILABLE = True
except ImportError:
    MODEL_LIBS_AVAILABLE = False
    print("Warning: Model libraries not available. Some operations may fail.")

# Configuration
# Use refactored final-model outputs (step 6) as the canonical source.
FINAL_MODEL_DIR = PROJECT_ROOT / '6_final_model' / 'outputs'
OUTPUT_DIR = PROJECT_ROOT / '10_risk_dashboard' / 'outputs' / 'models'  # For Docker container build
S3_MODEL_PREFIX = 'gold/dashboard/models'  # Optional S3 backup location

# Age bands (each cohort has all age bands; from py_helpers.constants)
from py_helpers.constants import REQUIRED_COHORTS
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
    Calculate model weights based on MC-CV performance metrics.
    
    Uses composite score: 0.5 * PR-AUC + 0.5 * (1/(1+logloss))
    Weights are normalized to sum to 1.0.
    
    Returns:
        {
            'catboost': weight,
            'xgboost': weight,
            'xgboost_rf': weight
        }
    """
    age_band_fname = age_band.replace("-", "_")
    # Step 6 saves MC-CV results at cohort/age_band_fname/ (not under models/)
    mc_cv_path = FINAL_MODEL_DIR / cohort / age_band_fname / f'{cohort}_{age_band_fname}_mc_cv_results.csv'
    
    if not mc_cv_path.exists():
        print(f"Warning: MC-CV results not found: {mc_cv_path}")
        print("  Using equal weights (1.0 each)")
        return {
            'catboost': 1.0,
            'xgboost': 1.0,
            'xgboost_rf': 1.0
        }
    
    # Load MC-CV results
    df = pd.read_csv(mc_cv_path)
    
    # Calculate composite scores for each model
    model_scores = {}
    for model_name in ['catboost', 'xgboost', 'xgboost_rf']:
        model_data = df[df['model'] == model_name]
        
        if len(model_data) == 0:
            print(f"Warning: No MC-CV results for {model_name}")
            continue
        
        mean_logloss = model_data['logloss'].mean()
        mean_pr_auc = model_data['pr_auc'].mean()
        
        # Normalize logloss: 1 / (1 + logloss) - higher is better
        normalized_logloss_score = 1 / (1 + mean_logloss)
        
        # PR-AUC is already in [0, 1], higher is better
        normalized_pr_auc_score = mean_pr_auc
        
        # Composite score: 0.5 * PR-AUC + 0.5 * normalized_logloss
        composite_score = 0.5 * normalized_pr_auc_score + 0.5 * normalized_logloss_score
        
        model_scores[model_name] = {
            'mean_logloss': mean_logloss,
            'mean_pr_auc': mean_pr_auc,
            'composite_score': composite_score
        }
    
    # Normalize weights to sum to 1.0
    total_score = sum(s['composite_score'] for s in model_scores.values())
    
    if total_score == 0:
        print("Warning: All composite scores are zero, using equal weights")
        return {
            'catboost': 1.0 / len(model_scores),
            'xgboost': 1.0 / len(model_scores),
            'xgboost_rf': 1.0 / len(model_scores)
        }
    
    weights = {
        model: model_scores[model]['composite_score'] / total_score
        for model in model_scores.keys()
    }
    
    # Ensure all three models have weights (fill missing with 0)
    for model in ['catboost', 'xgboost', 'xgboost_rf']:
        if model not in weights:
            weights[model] = 0.0
    
    print(f"  Model weights (based on composite score):")
    for model, weight in weights.items():
        if model in model_scores:
            score = model_scores[model]['composite_score']
            print(f"    {model}: {weight:.4f} (composite_score: {score:.4f})")
        else:
            print(f"    {model}: {weight:.4f} (no MC-CV data)")
    
    return weights


def extract_feature_schema(cohort: str, age_band: str) -> Dict[str, Any]:
    """
    Extract feature schema from training data.
    
    Returns feature names and default values.
    """
    age_band_fname = age_band.replace("-", "_")
    
    # Try to load training data
    train_data_path = FINAL_MODEL_DIR / cohort / age_band_fname / f'{cohort}_{age_band_fname}_train_final_features_no_leakage.csv'
    
    if not train_data_path.exists():
        print(f"Warning: Training data not found: {train_data_path}")
        return {'features': [], 'defaults': {}, 'model_weights': {}}
    
    # Load a sample of training data
    df = pd.read_csv(train_data_path, nrows=1000)
    
    # Get feature names (exclude target columns)
    exclude_cols = ['mi_person_key', 'target', 'event_year', 'cohort_name', 'age_band']
    feature_names = [col for col in df.columns if col not in exclude_cols]
    
    # Calculate default values (medians for numeric, 0 for binary)
    defaults = {}
    for feature in feature_names:
        if feature in df.columns:
            if df[feature].dtype in ['int64', 'float64']:
                defaults[feature] = float(df[feature].median())
            else:
                defaults[feature] = 0.0
    
    # Calculate model weights based on MC-CV performance
    model_weights = calculate_model_weights(cohort, age_band)
    
    return {
        'features': feature_names,
        'defaults': defaults,
        'model_weights': model_weights,
        'n_features': len(feature_names),
        'n_samples': len(df)
    }


def save_model(model: Any, output_path: Path, model_type: str):
    """Save model to output directory."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if model_type == 'catboost':
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


def prepare_models_for_cohort(cohort: str, age_bands: List[str]):
    """Prepare models for a cohort."""
    print(f"\n{'='*60}")
    print(f"Preparing models for {cohort}")
    print(f"{'='*60}")
    
    for age_band in age_bands:
        print(f"\nProcessing {age_band}...")
        age_band_fname = age_band.replace("-", "_")
        output_dir = OUTPUT_DIR / cohort / age_band_fname
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Extract feature schema and model weights
        print("  Extracting feature schema and model weights...")
        feature_schema = extract_feature_schema(cohort, age_band)
        
         # If no features were found (e.g., training data missing for this age_band),
        # skip model preparation for this band.
        n_features = feature_schema.get('n_features', len(feature_schema.get('features', [])))
        if n_features == 0:
            print(f"  No features found for {cohort} / {age_band} (training data missing or empty). Skipping.")
            continue

        schema_path = output_dir / 'feature_schema.json'
        with open(schema_path, 'w') as f:
            json.dump(feature_schema, f, indent=2)
        print(f"  Saved feature schema ({n_features} features)")
        if 'model_weights' in feature_schema:
            print(f"  Model weights included in schema")
        
        # Load and save models
        model_types = ['catboost', 'xgboost', 'xgboost_rf']
        for model_type in model_types:
            print(f"  Loading {model_type} model...")
            model = load_model(cohort, age_band, model_type)
            
            if model is None:
                # Step 6 saves only the best XGBoost variant; xgboost_rf is expected missing when best is xgb
                msg = "skipping (optional; pipeline saves only best XGBoost variant)" if model_type == "xgboost_rf" else "not found, skipping"
                print(f"    {model_type} {msg}")
                continue
            
            model_path = output_dir / f'{model_type}.joblib'
            if isinstance(model, Path):
                # For JSON models, save as JSON
                model_path = output_dir / f'{model_type}.json'
                import shutil
                shutil.copy(model, model_path)
            else:
                save_model(model, model_path, model_type)
        
        print(f"  Complete: {output_dir}")
    
    print(f"\n{'='*60}")
    print(f"Model preparation complete for {cohort}")
    print(f"{'='*60}")


def upload_to_s3(cohort: str, age_bands: List[str]):
    """Upload prepared models to S3."""
    try:
        import boto3
        s3_client = boto3.client('s3')
        bucket = 'pgxdatalake'
        prefix = 'gold/dashboard/models'
        
        print(f"\nUploading models to S3...")
        
        for age_band in age_bands:
            age_band_fname = age_band.replace("-", "_")
            local_dir = OUTPUT_DIR / cohort / age_band_fname
            
            if not local_dir.exists():
                continue
            
            for file_path in local_dir.glob('*'):
                if file_path.is_file():
                    s3_key = f"{prefix}/{cohort}/{age_band_fname}/{file_path.name}"
                    s3_client.upload_file(
                        str(file_path),
                        bucket,
                        s3_key
                    )
                    print(f"  Uploaded: s3://{bucket}/{s3_key}")
        
        print("S3 upload complete!")
        
    except ImportError:
        print("boto3 not available, skipping S3 upload")
    except Exception as e:
        print(f"Failed to upload to S3: {e}")


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
    
    print("\n" + "="*60)
    print("Preparing models for Lambda Container (ECR) deployment")
    print("="*60)
    print("Models will be placed in: models/")
    print("This directory will be copied into Docker container image")
    print("="*60 + "\n")
    
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

