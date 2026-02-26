#!/usr/bin/env python3
"""Check if required inputs exist for Step 8."""

import sys
import io
from pathlib import Path

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

PROJECT_ROOT = Path('.')
cohort = 'opioid_ed'
age_band = '13-24'
age_band_fname = age_band.replace('-', '_')

print('Checking required inputs for Step 8...')
print('=' * 80)

# Check model files (Step 6 writes best_xgboost_model.json and best_catboost_model.cbm/.json)
model_json_base = PROJECT_ROOT / '6_final_model' / 'outputs' / cohort / age_band_fname / 'final_model_json'
print(f'\nModel JSON base: {model_json_base}')
if model_json_base.exists():
    print('  [OK] Exists')
    xgb_json = model_json_base / f'{cohort}_{age_band_fname}_best_xgboost_model.json'
    if xgb_json.exists():
        print(f'  [OK] XGBoost JSON: {xgb_json}')
    else:
        print(f'  [MISSING] XGBoost JSON: {xgb_json}')
    cb_cbm = model_json_base / f'{cohort}_{age_band_fname}_best_catboost_model.cbm'
    if cb_cbm.exists():
        print(f'  [OK] CatBoost binary: {cb_cbm}')
    else:
        print(f'  [MISSING] CatBoost binary: {cb_cbm}')
else:
    print('  [MISSING] Model JSON base')

# Check data files
data_parquet = PROJECT_ROOT / '6_final_model' / 'outputs' / cohort / age_band_fname / 'inputs' / 'model_train' / 'final_features.parquet'
data_csv = PROJECT_ROOT / '6_final_model' / 'outputs' / cohort / age_band_fname / f'{cohort}_{age_band_fname}_train_final_features_no_leakage.csv'
print(f'\nData files:')
if data_parquet.exists():
    print(f'  [OK] Parquet: {data_parquet}')
elif data_csv.exists():
    print(f'  [OK] CSV: {data_csv}')
else:
    print(f'  [MISSING] Both: {data_parquet} or {data_csv}')

# Check SHAP outputs
shap_dir = PROJECT_ROOT / '7_shap_analysis' / 'outputs' / cohort / age_band_fname
shap_global = shap_dir / f'{cohort}_{age_band_fname}_shap_global_importance_xgboost.csv'
shap_samples = shap_dir / f'{cohort}_{age_band_fname}_shap_sample_values_xgboost.parquet'
print(f'\nSHAP outputs:')
if shap_global.exists():
    print(f'  [OK] Global importance: {shap_global}')
else:
    print(f'  [MISSING] Global importance: {shap_global}')
if shap_samples.exists():
    print(f'  [OK] Sample values: {shap_samples}')
else:
    print(f'  [MISSING] Sample values: {shap_samples}')

print()
print('=' * 80)
