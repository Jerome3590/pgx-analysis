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
    cohort = 'opioid_ed'
    age_bands = ['13-24', '25-44', '45-54', '55-64', '65-74', '75-84', '85-94']
    
    print('=== Checking model_events.parquet files in S3 for controls ===')
    print('')
    
    s3_base = f's3://pgxdatalake/gold/cohorts_model_data/cohort_name={cohort}/age_band='
    
    results = {}
    for age_band in age_bands:
        s3_path = f'{s3_base}{age_band}/model_events.parquet'
        
        # Check if file exists first
        ls_result = subprocess.run(
            ['aws', 's3', 'ls', s3_path, '--profile', 'mushin'],
            capture_output=True, text=True
        )
        
        if ls_result.returncode != 0:
            results[age_band] = {'error': 'File not found'}
            print(f'{cohort}/{age_band}: [NOT FOUND]')
            continue
        
        print(f'Checking {cohort}/{age_band}...', end=' ', flush=True)
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
        print(f'  ✓ {cohort}/{ab}: {r["n_controls"]:,} controls, {r["n_cases"]:,} cases')
    
    if invalid:
        print(f'')
        print(f'Invalid (no controls): {len(invalid)}')
        for ab in invalid:
            r = results[ab]
            print(f'  ✗ {cohort}/{ab}: {r.get("n_controls", 0):,} controls, {r.get("n_cases", 0):,} cases')
    
    if missing:
        print(f'')
        print(f'Missing: {len(missing)}')
        for ab in missing:
            print(f'  - {cohort}/{ab}')


if __name__ == '__main__':
    main()
