#!/usr/bin/env python3
"""
Run feature importance analysis for a single cohort/age-band combination

This script is idempotent - it will skip models that already have results in S3.
If you need to re-run a specific model, delete its result file from S3 first.

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

from helpers_1997_13.feature_importance_utils import run_cohort_analysis

# Configuration
COHORT_NAME = "non_opioid_ed"
AGE_BAND = "65-74"
TRAIN_YEARS = [2016, 2017, 2018]
TEST_YEAR = 2019
N_SPLITS = 50
TRAIN_PROP = 0.8
SCALING_METRIC = "recall"
DEBUG_MODE = False

# Model parameters
MODEL_PARAMS = {
    'catboost': {
        'iterations': 500,
        'learning_rate': 0.1,
        'depth': 6,
        'verbose': False,
        'random_seed': 42
    },
    'random_forest': {
        'ntree': 500,
        'mtry': None,
        'nodesize': 1,
        'maxnodes': None,
        'random_seed': 42
    },
    'xgboost': {
        'max_depth': 6,
        'learning_rate': 0.1,
        'n_estimators': 500,
        'subsample': 1.0,
        'colsample_bytree': 1.0,
        'random_seed': 42
    },
    'xgboost_rf': {
        'max_depth': 6,
        'learning_rate': 0.1,
        'n_estimators': 500,
        'subsample': 0.8,
        'max_features': None,
        'random_seed': 42
    },
    'lightgbm': {
        'n_estimators': 500,
        'learning_rate': 0.1,
        'num_leaves': 31,
        'feature_fraction': 1.0,
        'bagging_fraction': 1.0,
        'bagging_freq': 0,
        'random_seed': 42
    },
    'extratrees': {
        'n_estimators': 500,
        'max_features': None,
        'min_samples_leaf': 1,
        'max_depth': None,
        'random_seed': 42
    },
    'logistic_regression': {
        'penalty': 'l2',
        'C': 1.0,
        'solver': 'lbfgs',
        'max_iter': 1000,
        'random_seed': 42
    },
    'linearsvc': {
        'penalty': 'l2',
        'C': 1.0,
        'loss': 'squared_hinge',
        'max_iter': 1000,
        'dual': True,
        'random_seed': 42
    },
    'elasticnet': {
        'C': 1.0,
        'l1_ratio': 0.5,
        'max_iter': 1000,
        'random_seed': 42
    },
    'lasso': {
        'C': 1.0,
        'max_iter': 1000,
        'random_seed': 42
    }
}

# Set up parallel processing
import multiprocessing
N_WORKERS = max(1, multiprocessing.cpu_count() - 2)

print(f"Running feature importance analysis:")
print(f"  Cohort: {COHORT_NAME}")
print(f"  Age Band: {AGE_BAND}")
print(f"  Train Years: {TRAIN_YEARS}")
print(f"  Test Year: {TEST_YEAR}")
print(f"  MC-CV Splits: {N_SPLITS}")
print(f"  Workers: {N_WORKERS}")
print(f"  Output Directory: 3_feature_importance/outputs")
print(f\"\nNote: This script is idempotent - models with existing results in S3 will be skipped.\")
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
    print(f\"\\n[SUCCESS] Analysis complete!\")
    print(f\"  Aggregated output: {result.get('output_file', 'N/A')}\")
    print(f\"  Features analyzed: {result.get('n_features', 'N/A')}\")
    print(f\"\\n  Individual model results saved to: 3_feature_importance/outputs/\")
    print(f\"  All results uploaded to: s3://pgxdatalake/gold/feature_importance/{COHORT_NAME}/{AGE_BAND}/\")
elif result.get('status') == 'skipped':
    print(f\"\\n[SKIPPED] Analysis skipped: {result.get('reason', 'Unknown reason')}\")
else:
    print(f\"\\n[ERROR] Analysis failed: {result.get('error', 'Unknown error')}\")
    sys.exit(1)
