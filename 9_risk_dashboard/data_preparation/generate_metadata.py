#!/usr/bin/env python3
"""
Generate metadata files for dashboard from feature importance CSVs.

This script extracts valid codes (ICD, CPT, Drug) from feature importance files
and creates metadata JSON files for each cohort/age_band combination.

Usage:
    python generate_metadata.py --cohort opioid_ed
    python generate_metadata.py --cohort non_opioid_ed
    python generate_metadata.py --all
"""

import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any
import pandas as pd

# Add project root to path
# This script is in 9_risk_dashboard/data_preparation/
# Project root is 3 levels up
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from py_helpers.env_utils import get_data_root

# Configuration: prefer DATA_ROOT/gold/feature_importance (NVMe), then project outputs
def _fi_roots():
    """Roots for feature importance (Step 3b then Step 3a), NVMe first."""
    data_root = get_data_root()
    return [
        data_root / "gold" / "feature_importance",
        PROJECT_ROOT / "3b_feature_importance_eda" / "outputs",
    ]

def _aggregated_fi_roots():
    """Roots for aggregated feature importance (Step 3a), NVMe first."""
    data_root = get_data_root()
    return [
        data_root / "gold" / "feature_importance",
        PROJECT_ROOT / "3a_feature_importance" / "outputs",
    ]

FEATURE_IMPORTANCE_DIR = PROJECT_ROOT / '3b_feature_importance_eda' / 'outputs'
AGGREGATED_FEATURE_IMPORTANCE_DIR = PROJECT_ROOT / '3a_feature_importance' / 'outputs'
FINAL_MODEL_DIR = PROJECT_ROOT / '6_final_model' / 'outputs'
OUTPUT_DIR = PROJECT_ROOT / '9_risk_dashboard' / 'outputs' / 'metadata'

# Age bands for each cohort
OPIOID_ED_AGE_BANDS = ["13-24", "25-44", "45-54", "55-64"]
POLYPHARMACY_AGE_BANDS = ["65-74", "75-84", "85-94"]

# Code type prefixes
DRUG_PREFIX = "item_"
ICD_PREFIX = "item_"
CPT_PREFIX = "item_"


def parse_feature_name(feature: str) -> tuple[str, str]:
    """
    Parse feature name to extract code type and code.
    Handles both "item_<code>" format and raw code (no prefix).
    Returns: (code_type, code); code_type is 'drug', 'icd', 'cpt', or 'other'.
    """
    if feature is None or (isinstance(feature, float) and pd.isna(feature)):
        return ('other', '')
    feature = str(feature).strip()
    if not feature:
        return ('other', '')

    # Remove item_ prefix if present (Step 3a/4 model data convention)
    if feature.startswith("item_"):
        code = feature[5:].strip()
    else:
        code = feature

    if not code:
        return ('other', feature)

    # CPT: all digits (e.g. 99284, 80305)
    if code.isdigit():
        return ('cpt', code)
    # ICD: starts with letter, then digits/dots (e.g. F1120, R51, G89.12)
    if code[0].isalpha() and len(code) >= 2:
        rest = code[1:].replace('.', '').replace('-', '')
        if rest.isdigit():
            return ('icd', code)
        # Letter + alphanumeric could be ICD or drug; short codes like R51 -> icd
        if len(code) <= 5 and code.isalnum():
            return ('icd', code)
        # Longer or mixed -> treat as drug (e.g. AMOXICILLIN, SUBOXONE)
        return ('drug', code)
    # Numeric with possible suffix -> cpt
    if code.replace('.', '').isdigit():
        return ('cpt', code)
    # Default: treat as drug (e.g. drug names, mixed alphanumeric)
    return ('drug', code)


def load_feature_importance(cohort: str, age_band: str) -> pd.DataFrame:
    """Load feature importance CSV for a cohort/age_band.
    
    Prioritizes Step 3b cohort_feature_importance (refined). Checks NVMe then project.
    Falls back to Step 3 aggregated_feature_importance (same order).
    """
    age_band_fname = age_band.replace("-", "_")
    step3b_filename = f"{cohort}_{age_band_fname}_cohort_feature_importance.csv"
    aggregated_filename = f"{cohort}_{age_band_fname}_aggregated_feature_importance.csv"

    # Step 3b refined: try each root (NVMe then project)
    for base in _fi_roots():
        step3b_filepath = base / cohort / age_band_fname / step3b_filename
        if step3b_filepath.exists():
            print(f"Loading Step 3b refined features: {step3b_filepath}")
            return pd.read_csv(step3b_filepath)

    # Fallback: Step 3 aggregated
    for base in _aggregated_fi_roots():
        aggregated_filepath = base / cohort / age_band_fname / aggregated_filename
        if aggregated_filepath.exists():
            print(f"Loading Step 3 aggregated features (fallback): {aggregated_filepath}")
            return pd.read_csv(aggregated_filepath)

    print(f"Warning: Feature importance not found for {cohort}/{age_band} in Step 3b or Step 3 roots")
    return pd.DataFrame()


def extract_codes_from_features(df: pd.DataFrame, top_n: int = 100) -> Dict[str, List[Dict[str, Any]]]:
    """
    Extract codes from feature importance DataFrame.
    Supports Step 3b cohort_feature_importance and Step 3 aggregated CSVs (various column names).
    
    Returns:
        {
            'drugs': [{'code': '...', 'display': '...', 'importance': ...}, ...],
            'icds': [...],
            'cpts': [...]
        }
    """
    codes = {
        'drugs': [],
        'icds': [],
        'cpts': []
    }
    if df.empty:
        return codes

    # Ensure 'feature' column exists (Step 3b may use first column or "Feature" with capital F)
    feature_col_name = next((c for c in df.columns if c.strip().lower() == 'feature'), None)
    if feature_col_name and feature_col_name != 'feature':
        df = df.rename(columns={feature_col_name: 'feature'})
    elif 'feature' not in df.columns and len(df.columns) >= 1:
        df = df.rename(columns={df.columns[0]: 'feature'})

    # Resolve importance column (Step 3b: importance_scaled_by_model_sum, importance_mean; Step 3a: scaled_importance_mean, etc.)
    sort_col = None
    for col in (
        'importance_scaled_by_model_sum',
        'importance_mean',
        'scaled_importance_mean',
        'importance_scaled',
        'importance_normalized',
        'importance',
    ):
        if col in df.columns:
            sort_col = col
            break
    if sort_col is None:
        # Fallback: any column with 'importance' in the name, or second column
        imp_cols = [c for c in df.columns if 'importance' in c.lower()]
        if imp_cols:
            sort_col = imp_cols[0]
        elif len(df.columns) >= 2:
            sort_col = df.columns[1]
    if sort_col is None:
        print("Warning: No importance column found")
        return codes

    df_sorted = df.nlargest(top_n, sort_col)
    if 'feature' not in df_sorted.columns:
        print("Warning: No 'feature' column found")
        return codes

    for _, row in df_sorted.iterrows():
        feature = row['feature']
        try:
            importance = float(row[sort_col])
        except (TypeError, ValueError):
            importance = 0.0
        if pd.isna(importance):
            importance = 0.0
        
        code_type, code = parse_feature_name(feature)
        if not code or code_type == 'other':
            continue
        # Exclude F1120 from ICD codes (it's the target, not an input)
        if code_type == 'icd' and code.upper() == 'F1120':
            continue
        if code_type in codes:
            # Create display name (clean up code)
            display = code.replace('_', ' ').title()
            
            codes[code_type].append({
                'code': code,
                'display': display,
                'importance': importance,
                'feature_name': feature
            })
    
    # Sort each list by importance (descending)
    for code_type in codes:
        codes[code_type].sort(key=lambda x: x['importance'], reverse=True)
    
    return codes


def generate_metadata_for_cohort(cohort: str, age_bands: List[str]) -> Dict[str, Any]:
    """Generate metadata for a cohort."""
    metadata = {
        'cohort': cohort,
        'age_bands': age_bands,
        'codes': {}
    }
    
    for age_band in age_bands:
        print(f"Processing {cohort} / {age_band}...")
        
        # Load feature importance
        df = load_feature_importance(cohort, age_band)
        
        if df.empty:
            print(f"  No data found for {age_band}")
            metadata['codes'][age_band] = {
                'drugs': [],
                'icds': [],
                'cpts': []
            }
            continue
        
        # Extract codes
        codes = extract_codes_from_features(df, top_n=200)
        
        metadata['codes'][age_band] = codes
        
        print(f"  Found {len(codes['drugs'])} drugs, {len(codes['icds'])} ICDs, {len(codes['cpts'])} CPTs")
    
    return metadata


def save_metadata(metadata: Dict[str, Any], output_dir: Path):
    """Save metadata to JSON file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cohort = metadata['cohort']
    filename = f"metadata_{cohort}.json"
    filepath = output_dir / filename
    
    with open(filepath, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Saved metadata to: {filepath}")
    
    # Also save to S3 if boto3 is available
    try:
        import boto3
        s3_client = boto3.client('s3')
        bucket = 'pgxdatalake'
        key = f'gold/dashboard/metadata/{filename}'
        
        s3_client.put_object(
            Bucket=bucket,
            Key=key,
            Body=json.dumps(metadata, indent=2),
            ContentType='application/json'
        )
        print(f"Uploaded to S3: s3://{bucket}/{key}")
    except ImportError:
        print("boto3 not available, skipping S3 upload")
    except Exception as e:
        print(f"Failed to upload to S3: {e}")


def main():
    parser = argparse.ArgumentParser(description='Generate dashboard metadata files')
    parser.add_argument('--cohort', choices=['opioid_ed', 'non_opioid_ed'], 
                       help='Cohort to process')
    parser.add_argument('--all', action='store_true', 
                       help='Process all cohorts')
    
    args = parser.parse_args()
    
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
    
    # OUTPUT_DIR is 9_risk_dashboard/outputs/metadata (created on first save)
    print(f"Output directory: {OUTPUT_DIR.resolve()}")
    for cohort, age_bands in cohorts:
        print(f"\n{'='*60}")
        print(f"Generating metadata for {cohort}")
        print(f"{'='*60}")
        
        metadata = generate_metadata_for_cohort(cohort, age_bands)
        save_metadata(metadata, OUTPUT_DIR)
    
    print(f"\n{'='*60}")
    print("Metadata generation complete!")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()

