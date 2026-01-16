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

# Configuration
# Use Step 3b cohort_feature_importance files (refined features)
FEATURE_IMPORTANCE_DIR = PROJECT_ROOT / '3b_feature_importance_eda' / 'outputs'
# Fallback to Step 3 aggregated feature importance if Step 3b files not available
AGGREGATED_FEATURE_IMPORTANCE_DIR = PROJECT_ROOT / '3_feature_importance' / 'outputs'
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
    
    Returns: (code_type, code)
    code_type: 'drug', 'icd', 'cpt', or 'other'
    """
    # Remove item_ prefix
    if feature.startswith("item_"):
        code = feature[5:]  # Remove "item_"
    else:
        return ('other', feature)
    
    # Try to determine code type
    # ICD codes typically start with letters (F1120, R51, etc.)
    # CPT codes are typically numeric (80305, 99213, etc.)
    # Drug names are typically uppercase letters/numbers
    
    # Check if it's an ICD code (starts with letter)
    if code and code[0].isalpha():
        # Common ICD patterns: F1120, R51, G9012, etc.
        if len(code) >= 3 and (code[0].isalpha() and code[1:].replace('.', '').isdigit()):
            return ('icd', code)
        # Could also be a drug name
        return ('drug', code)
    
    # Check if it's a CPT code (numeric)
    if code.isdigit():
        return ('cpt', code)
    
    # Default to drug name
    return ('drug', code)


def load_feature_importance(cohort: str, age_band: str) -> pd.DataFrame:
    """Load feature importance CSV for a cohort/age_band.
    
    Prioritizes Step 3b cohort_feature_importance files (refined features).
    Falls back to Step 3 aggregated_feature_importance if Step 3b files not available.
    """
    age_band_fname = age_band.replace("-", "_")
    
    # Try Step 3b cohort_feature_importance first (refined features)
    step3b_filename = f"{cohort}_{age_band_fname}_cohort_feature_importance.csv"
    step3b_filepath = FEATURE_IMPORTANCE_DIR / cohort / age_band_fname / step3b_filename
    
    if step3b_filepath.exists():
        print(f"Loading Step 3b refined features: {step3b_filepath}")
        df = pd.read_csv(step3b_filepath)
        return df
    
    # Fallback to Step 3 aggregated_feature_importance
    aggregated_filename = f"{cohort}_{age_band_fname}_aggregated_feature_importance.csv"
    aggregated_filepath = AGGREGATED_FEATURE_IMPORTANCE_DIR / cohort / age_band_fname / aggregated_filename
    
    if aggregated_filepath.exists():
        print(f"Loading Step 3 aggregated features (fallback): {aggregated_filepath}")
        df = pd.read_csv(aggregated_filepath)
        return df
    
    print(f"Warning: Feature importance file not found in Step 3b or Step 3: {step3b_filepath} or {aggregated_filepath}")
    return pd.DataFrame()


def extract_codes_from_features(df: pd.DataFrame, top_n: int = 100) -> Dict[str, List[Dict[str, Any]]]:
    """
    Extract codes from feature importance DataFrame.
    
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
    
    # Sort by importance and take top N
    if 'importance_scaled' in df.columns:
        sort_col = 'importance_scaled'
    elif 'importance_normalized' in df.columns:
        sort_col = 'importance_normalized'
    elif 'importance' in df.columns:
        sort_col = 'importance'
    else:
        print("Warning: No importance column found")
        return codes
    
    df_sorted = df.nlargest(top_n, sort_col)
    
    for _, row in df_sorted.iterrows():
        feature = row['feature']
        importance = float(row[sort_col])
        
        code_type, code = parse_feature_name(feature)
        
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

