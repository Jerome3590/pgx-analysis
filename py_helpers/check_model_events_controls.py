#!/usr/bin/env python3
"""Check which model_events.parquet files in S3 have controls."""

import subprocess
import sys
import tempfile
import os
import pandas as pd

def check_s3_file_controls(s3_path, profile='mushin'):
    """Check if an S3 parquet file has controls."""
    # Create temp file
    with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as tmp:
        temp_file = tmp.name
    
    try:
        # Download file
        result = subprocess.run(
            ['aws', 's3', 'cp', s3_path, temp_file, '--profile', profile],
            capture_output=True, text=True
        )
        
        if result.returncode != 0:
            return {'error': 'Download failed', 'stderr': result.stderr}
        
        # Check controls using pandas
        import pandas as pd
        df = pd.read_parquet(temp_file)
        
        if 'target' not in df.columns:
            return {'error': 'No target column found'}
        
        n_controls = int((df['target'] == 0).sum())
        n_cases = int((df['target'] == 1).sum())
        
        return {
            'has_controls': n_controls > 0,
            'n_controls': n_controls,
            'n_cases': n_cases
        }
    finally:
        # Clean up
        if os.path.exists(temp_file):
            os.unlink(temp_file)


def main():
    age_bands = ['13-24', '25-44', '45-54', '55-64', '65-74', '75-84', '85-114']
    
    print('=== Checking model_events.parquet files in S3 for controls ===')
    print('')
    
    # Import get_cohort_slug to determine slug based on age band
    # Age bands < 65 use "opioid" slug, >= 65 use "polypharmacy" slug
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from py_helpers.constants import get_cohort_slug
    
    results = {}
    for age_band in age_bands:
        # Get cohort slug based on age band: "opioid" for < 65, "polypharmacy" for >= 65
        cohort_slug = get_cohort_slug(age_band)
        s3_path = f's3://pgxdatalake/gold/cohorts/input_model_data/cohort_name={cohort_slug}/age_band={age_band}/model_events.parquet'
        
        # Check if file exists first
        ls_result = subprocess.run(
            ['aws', 's3', 'ls', s3_path, '--profile', 'mushin'],
            capture_output=True, text=True
        )
        
        if ls_result.returncode != 0:
            results[age_band] = {'error': 'File not found'}
            print(f'{cohort_slug}/{age_band}: [NOT FOUND]')
            continue
        
        print(f'Checking {cohort_slug}/{age_band}...', end=' ', flush=True)
        result = check_s3_file_controls(s3_path)
        
        if 'error' in result:
            results[age_band] = result
            print(f'[ERROR] {result["error"]}')
        else:
            results[age_band] = result
            status = '[OK]' if result['has_controls'] else '[NO CONTROLS]'
            print(f'{status} Controls: {result["n_controls"]:,}, Cases: {result["n_cases"]:,}')
    
    print('')
    print('=== Summary ===')
    print('')
    
    valid = [ab for ab, r in results.items() if r.get('has_controls', False)]
    invalid = [ab for ab, r in results.items() if not r.get('has_controls', False) and 'error' not in r]
    missing = [ab for ab, r in results.items() if 'error' in r and r.get('error') == 'File not found']
    
    print(f'Valid (with controls): {len(valid)}')
    for ab in valid:
        r = results[ab]
        cohort_slug = get_cohort_slug(ab)
        print(f'  ✓ {cohort_slug}/{ab}: {r["n_controls"]:,} controls, {r["n_cases"]:,} cases')
    
    if invalid:
        print(f'')
        print(f'Invalid (no controls): {len(invalid)}')
        for ab in invalid:
            r = results[ab]
            cohort_slug = get_cohort_slug(ab)
            print(f'  ✗ {cohort_slug}/{ab}: {r.get("n_controls", 0):,} controls, {r.get("n_cases", 0):,} cases')
    
    if missing:
        print(f'')
        print(f'Missing: {len(missing)}')
        for ab in missing:
            cohort_slug = get_cohort_slug(ab)
            print(f'  - {cohort_slug}/{ab}')


if __name__ == '__main__':
    main()
