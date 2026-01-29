#!/usr/bin/env python3
"""Check which cohort.parquet files in S3 have controls (is_target_case column)."""

import subprocess
import sys
import tempfile
import os
import pandas as pd

def check_s3_cohort_file_controls(s3_path, profile='mushin'):
    """Check if an S3 cohort.parquet file has controls."""
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
        df = pd.read_parquet(temp_file)
        
        if 'is_target_case' not in df.columns:
            return {'error': 'No is_target_case column found'}
        
        n_controls = int((df['is_target_case'] == 0).sum())
        n_cases = int((df['is_target_case'] == 1).sum())
        
        return {
            'has_controls': n_controls > 0,
            'n_controls': n_controls,
            'n_cases': n_cases,
            'total': len(df)
        }
    except Exception as e:
        return {'error': str(e)}
    finally:
        # Clean up
        if os.path.exists(temp_file):
            os.unlink(temp_file)


def main():
    cohort = 'opioid_ed'
    age_bands = ['13-24', '25-44', '45-54', '55-64', '65-74', '75-84', '85-94']
    event_years = [2016, 2017, 2018, 2019]
    
    print('=== Checking cohort.parquet files in S3 for controls ===')
    print('')
    
    s3_base = f's3://pgxdatalake/gold/cohorts_F1120/cohort_name={cohort}/event_year='
    
    results = {}
    for age_band in age_bands:
        results[age_band] = {}
        for year in event_years:
            s3_path = f'{s3_base}{year}/age_band={age_band}/cohort.parquet'
            
            # Check if file exists first
            ls_result = subprocess.run(
                ['aws', 's3', 'ls', s3_path, '--profile', 'mushin'],
                capture_output=True, text=True
            )
            
            if ls_result.returncode != 0:
                results[age_band][year] = {'error': 'File not found'}
                continue
            
            print(f'Checking {cohort}/{age_band}/{year}...', end=' ', flush=True)
            result = check_s3_cohort_file_controls(s3_path)
            
            if 'error' in result:
                results[age_band][year] = result
                print(f'[ERROR] {result["error"]}')
            else:
                results[age_band][year] = result
                status = '[OK]' if result['has_controls'] else '[NO CONTROLS]'
                print(f'{status} Controls: {result["n_controls"]:,}, Cases: {result["n_cases"]:,}, Total: {result["total"]:,}')
    
    print('')
    print('=== Summary by Age Band ===')
    print('')
    
    for age_band in age_bands:
        valid_years = [y for y in event_years if results[age_band].get(y, {}).get('has_controls', False)]
        invalid_years = [y for y in event_years if y in results[age_band] and not results[age_band][y].get('has_controls', False) and 'error' not in results[age_band][y]]
        missing_years = [y for y in event_years if results[age_band].get(y, {}).get('error') == 'File not found']
        
        if valid_years:
            print(f'{cohort}/{age_band}:')
            print(f'  Valid (with controls): {len(valid_years)} years - {valid_years}')
            for year in valid_years:
                r = results[age_band][year]
                print(f'    {year}: {r["n_controls"]:,} controls, {r["n_cases"]:,} cases')
            if invalid_years:
                print(f'  Invalid (no controls): {invalid_years}')
            if missing_years:
                print(f'  Missing: {missing_years}')
            print('')


if __name__ == '__main__':
    main()
