#!/usr/bin/env python3
"""
Run feature importance analysis for a single cohort/age-band combination

This script is idempotent - it will skip models that already have results in S3.
If you need to re-run a specific model, delete its result file from S3 first.

Configuration:
- Permutation importance uses n_repeats=3 (default) for optimal speed/accuracy balance
  (40% faster than n_repeats=5 with minimal impact on statistical stability)

Outputs:
- Individual model results: 3_feature_importance/outputs/{cohort}_{age_band}_{method}_feature_importance.csv
- Aggregated results: 3_feature_importance/outputs/{cohort}_{age_band}_aggregated_feature_importance.csv
- Constant features: 3_feature_importance/outputs/{cohort}_{age_band}_constant_features.csv

All results are also uploaded to S3:
- s3://pgxdatalake/gold/feature_importance/{cohort}/{age_band}/
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from py_helpers.feature_importance_utils import run_cohort_analysis

# Configuration
COHORT_NAME = "non_opioid_ed"
AGE_BAND = "65-74"
TRAIN_YEARS = [2016, 2017, 2018]
TEST_YEAR = 2019
N_SPLITS = 25
TRAIN_PROP = 0.8
SCALING_METRIC = "recall"
DEBUG_MODE = False

# Model parameters (core ensemble: CatBoost, XGBoost, XGBoost RF)
MODEL_PARAMS = {
    'catboost': {
        'iterations': 500,
        'learning_rate': 0.1,
        'depth': 6,
        'verbose': False,
        'random_seed': 42,
    },
    'xgboost': {
        'max_depth': 6,
        'learning_rate': 0.1,
        'n_estimators': 250,
        'subsample': 1.0,
        'colsample_bytree': 1.0,
        'random_seed': 42,
        'n_jobs': 2,  # Use 2 threads per model (with 28 workers × 2 threads = 56 threads on 32 cores, acceptable with early stopping)
        'tree_method': 'hist',  # Faster than exact, more accurate than approx
        'early_stopping_rounds': 10,  # Enable early stopping for faster training
    },
    'xgboost_rf': {
        'max_depth': 6,
        'learning_rate': 0.1,
        'n_estimators': 250,
        'subsample': 0.8,
        'max_features': None,
        'random_seed': 42,
        'n_jobs': 2,  # Use 2 threads per model (with 28 workers × 2 threads = 56 threads on 32 cores, acceptable with early stopping)
        'tree_method': 'hist',  # Faster than exact, more accurate than approx
        'early_stopping_rounds': 10,  # Enable early stopping for faster training
    },
}

# Set up parallel processing
import multiprocessing
# Optimized for EC2: 32 cores, 1TB RAM
# Use 28 workers (leave 4 cores for system/OS overhead)
# With 1TB RAM, we can handle high parallelism without memory concerns
N_WORKERS = max(1, multiprocessing.cpu_count() - 4)

print(f"Running feature importance analysis:")
print(f"  Cohort: {COHORT_NAME}")
print(f"  Age Band: {AGE_BAND}")
print(f"  Train Years: {TRAIN_YEARS}")
print(f"  Test Year: {TEST_YEAR}")
print(f"  MC-CV Splits: {N_SPLITS}")
print(f"  Workers: {N_WORKERS}")
print(f"  Output Directory: 3_feature_importance/outputs")
print("Note: This script is idempotent - models with existing results in S3 will be skipped.")
print()

# Run analysis
result = run_cohort_analysis(
    cohort_name=COHORT_NAME,
    age_band=AGE_BAND,
    train_years=TRAIN_YEARS,
    test_year=TEST_YEAR,
    n_splits=N_SPLITS,
    train_prop=TRAIN_PROP,
    n_workers=N_WORKERS,
    scaling_metric=SCALING_METRIC,
    model_params=MODEL_PARAMS,
    debug_mode=DEBUG_MODE,
    output_dir='3_feature_importance/outputs'
)

# Check results
if result.get('status') == 'success':
    print("[SUCCESS] Analysis complete!")
    print(f"  Aggregated output: {result.get('output_file', 'N/A')}")
    print(f"  Features analyzed: {result.get('n_features', 'N/A')}")
    print(f"  Individual model results saved to: 3_feature_importance/outputs/")
    print(f"  All results uploaded to: s3://pgxdatalake/gold/feature_importance/{COHORT_NAME}/{AGE_BAND}/")
elif result.get('status') == 'skipped':
    print(f"[SKIPPED] Analysis skipped: {result.get('reason', 'Unknown reason')}")
else:
    print(f"[ERROR] Analysis failed: {result.get('error', 'Unknown error')}")
    sys.exit(1)
