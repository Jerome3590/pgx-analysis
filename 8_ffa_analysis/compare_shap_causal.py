#!/usr/bin/env python3
"""
Compare SHAP feature importance with Causal importance to assess grouping impact.
"""

import sys
import pandas as pd
from io import BytesIO
import boto3
from botocore.exceptions import ClientError
from pathlib import Path

# Fix Windows encoding
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

s3_client = boto3.client('s3')
bucket = 'pgxdatalake'

cohort = 'opioid_ed'
age_band = '13-24'
age_band_fname = age_band.replace('-', '_')

print('Comparing SHAP vs Causal Importance')
print('=' * 80)
print(f'Cohort: {cohort}, Age Band: {age_band}')
print()

# Load SHAP importance
shap_key = f'gold/shap_analysis/{cohort}/{age_band}/{cohort}_{age_band_fname}_shap_global_importance_xgboost.csv'
print('Loading SHAP importance...')
try:
    obj = s3_client.get_object(Bucket=bucket, Key=shap_key)
    shap_df = pd.read_csv(BytesIO(obj['Body'].read()))
    print(f'  Loaded {len(shap_df)} features')
except Exception as e:
    print(f'  ERROR: {e}')
    shap_df = None

# Load causal importance
causal_key = f'gold/ffa_analysis/{cohort}/{age_band}/xgboost/causal_importance.parquet'
print('Loading causal importance...')
try:
    obj = s3_client.get_object(Bucket=bucket, Key=causal_key)
    causal_df = pd.read_parquet(BytesIO(obj['Body'].read()))
    print(f'  Loaded {len(causal_df)} features')
except Exception as e:
    print(f'  ERROR: {e}')
    causal_df = None

if shap_df is None or causal_df is None:
    print('ERROR: Could not load both datasets')
    sys.exit(1)

# Check SHAP column names
print('SHAP columns:', shap_df.columns.tolist())
print('Causal columns:', causal_df.columns.tolist())
print()

# Find the importance column in SHAP (could be 'importance', 'shap_value', 'mean_abs_shap', etc.)
shap_importance_col = None
for col in ['importance', 'shap_value', 'mean_abs_shap', 'mean(|SHAP value|)', 'shap_importance']:
    if col in shap_df.columns:
        shap_importance_col = col
        break

if shap_importance_col is None:
    # Use first numeric column
    numeric_cols = shap_df.select_dtypes(include=['float64', 'int64']).columns
    if len(numeric_cols) > 0:
        shap_importance_col = numeric_cols[0]
    else:
        print('ERROR: Could not find importance column in SHAP data')
        sys.exit(1)

print(f'Using SHAP column: {shap_importance_col}')
print()

# Merge on feature name
merged = pd.merge(
    shap_df[['feature', shap_importance_col]].rename(columns={shap_importance_col: 'shap_importance'}),
    causal_df[['feature', 'causal_importance', 'is_binary']],
    on='feature',
    how='outer'
).fillna(0.0)

print()
print('=' * 80)
print('COMPARISON SUMMARY')
print('=' * 80)
print()

# Separate by feature type
drug_features = merged[merged['feature'].str.startswith('item_drug_')]
icd_features = merged[merged['feature'].str.startswith('item_icd_')]
pgx_features = merged[merged['feature'].str.startswith('pgx_')]
other_features = merged[~merged['feature'].str.startswith('item_') & ~merged['feature'].str.startswith('pgx_')]

print('Drug Features:')
print(f'  Total: {len(drug_features)}')
shap_gt0 = len(drug_features[drug_features['shap_importance'] > 0])
causal_gt0 = len(drug_features[drug_features['causal_importance'] > 0])
both_gt0 = len(drug_features[(drug_features['shap_importance'] > 0) & (drug_features['causal_importance'] > 0)])
shap_gt0_causal_eq0 = len(drug_features[(drug_features['shap_importance'] > 0) & (drug_features['causal_importance'] == 0)])
print(f'  SHAP > 0: {shap_gt0}')
print(f'  Causal > 0: {causal_gt0}')
print(f'  Both > 0: {both_gt0}')
print(f'  SHAP > 0 but Causal = 0: {shap_gt0_causal_eq0}')
print()

print('ICD Features:')
print(f'  Total: {len(icd_features)}')
shap_gt0_icd = len(icd_features[icd_features['shap_importance'] > 0])
causal_gt0_icd = len(icd_features[icd_features['causal_importance'] > 0])
both_gt0_icd = len(icd_features[(icd_features['shap_importance'] > 0) & (icd_features['causal_importance'] > 0)])
shap_gt0_causal_eq0_icd = len(icd_features[(icd_features['shap_importance'] > 0) & (icd_features['causal_importance'] == 0)])
print(f'  SHAP > 0: {shap_gt0_icd}')
print(f'  Causal > 0: {causal_gt0_icd}')
print(f'  Both > 0: {both_gt0_icd}')
print(f'  SHAP > 0 but Causal = 0: {shap_gt0_causal_eq0_icd}')
print()

print('PGx Features:')
print(f'  Total: {len(pgx_features)}')
shap_gt0_pgx = len(pgx_features[pgx_features['shap_importance'] > 0])
causal_gt0_pgx = len(pgx_features[pgx_features['causal_importance'] > 0])
both_gt0_pgx = len(pgx_features[(pgx_features['shap_importance'] > 0) & (pgx_features['causal_importance'] > 0)])
print(f'  SHAP > 0: {shap_gt0_pgx}')
print(f'  Causal > 0: {causal_gt0_pgx}')
print(f'  Both > 0: {both_gt0_pgx}')
print()

print('Other Features:')
print(f'  Total: {len(other_features)}')
shap_gt0_other = len(other_features[other_features['shap_importance'] > 0])
causal_gt0_other = len(other_features[other_features['causal_importance'] > 0])
both_gt0_other = len(other_features[(other_features['shap_importance'] > 0) & (other_features['causal_importance'] > 0)])
print(f'  SHAP > 0: {shap_gt0_other}')
print(f'  Causal > 0: {causal_gt0_other}')
print(f'  Both > 0: {both_gt0_other}')
print()

print('=' * 80)
print('TOP 20 FEATURES BY SHAP IMPORTANCE')
print('=' * 80)
top_shap = merged.nlargest(20, 'shap_importance')[['feature', 'shap_importance', 'causal_importance', 'is_binary']]
for idx, row in top_shap.iterrows():
    feat_type = 'binary' if row['is_binary'] else 'continuous'
    causal_val = row['causal_importance']
    causal_str = f'{causal_val:.6f}' if pd.notna(causal_val) else 'N/A'
    print(f'  {row["feature"]:<50} SHAP: {row["shap_importance"]:>10.6f}  Causal: {causal_str:>10} ({feat_type})')
print()

print('=' * 80)
print('TOP 20 FEATURES BY CAUSAL IMPORTANCE')
print('=' * 80)
top_causal = merged.nlargest(20, 'causal_importance')[['feature', 'shap_importance', 'causal_importance', 'is_binary']]
for idx, row in top_causal.iterrows():
    feat_type = 'binary' if row['is_binary'] else 'continuous'
    shap_val = row['shap_importance']
    shap_str = f'{shap_val:.6f}' if pd.notna(shap_val) else 'N/A'
    print(f'  {row["feature"]:<50} SHAP: {shap_str:>10}  Causal: {row["causal_importance"]:>10.6f} ({feat_type})')
print()

print('=' * 80)
print('FEATURES WITH HIGH SHAP BUT ZERO CAUSAL')
print('=' * 80)
high_shap_zero_causal = merged[
    (merged['shap_importance'] > 0.01) & 
    (merged['causal_importance'] == 0.0)
].nlargest(20, 'shap_importance')[['feature', 'shap_importance', 'causal_importance', 'is_binary']]

if len(high_shap_zero_causal) > 0:
    print(f'Found {len(high_shap_zero_causal)} features with SHAP > 0.01 but Causal = 0.0')
    print('Top examples:')
    for idx, row in high_shap_zero_causal.iterrows():
        feat_type = 'binary' if row['is_binary'] else 'continuous'
        print(f'  {row["feature"]:<50} SHAP: {row["shap_importance"]:>10.6f}  Causal: {row["causal_importance"]:>10.6f} ({feat_type})')
else:
    print('No features found with high SHAP but zero causal importance')
print()

# Correlation
print('=' * 80)
print('CORRELATION ANALYSIS')
print('=' * 80)
# Filter to features present in both
both_present = merged[(merged['shap_importance'] > 0) | (merged['causal_importance'] > 0)]
if len(both_present) > 1:
    correlation = both_present['shap_importance'].corr(both_present['causal_importance'])
    print(f'Correlation between SHAP and Causal Importance: {correlation:.4f}')
    print()
    print('Interpretation:')
    if correlation > 0.7:
        print('  Strong positive correlation - SHAP and Causal agree')
    elif correlation > 0.3:
        print('  Moderate correlation - Some agreement')
    elif correlation > -0.3:
        print('  Weak correlation - Different signals')
    else:
        print('  Negative correlation - Opposite signals')
else:
    print('Not enough data for correlation analysis')
