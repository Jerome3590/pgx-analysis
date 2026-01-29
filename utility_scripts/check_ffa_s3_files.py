#!/usr/bin/env python3
"""Check FFA analysis files in S3 for visualization readiness."""

import sys
import subprocess
from collections import defaultdict

sys.stdout.reconfigure(encoding='utf-8')

cohorts = {
    'opioid_ed': ['13-24', '25-44', '45-54', '55-64'],
    'non_opioid_ed': ['65-74', '75-84', '85-94']
}

expected_files = ['axp_explanations.parquet', 'feature_importance_axp.parquet', 
                  'causal_importance.parquet', 'interaction_analysis.parquet']

# Get S3 file list
try:
    result = subprocess.run(
        ['aws', 's3', 'ls', 's3://pgxdatalake/gold/ffa_analysis/', '--profile', 'mushin', '--recursive'],
        capture_output=True,
        text=True
    )
    s3_files = [line.split()[-1] for line in result.stdout.split('\n') if line.strip()]
except Exception as e:
    print(f"Error listing S3 files: {e}")
    sys.exit(1)

# Parse S3 files
found = defaultdict(lambda: defaultdict(set))

for f in s3_files:
    parts = f.split('/')
    if len(parts) >= 5 and parts[-2] == 'xgboost':
        cohort = parts[2]
        age_band = parts[3]
        filename = parts[-1]
        if cohort in cohorts and age_band in cohorts[cohort]:
            found[cohort][age_band].add(filename)

# Check status
print('='*80)
print('FFA ANALYSIS FILES STATUS IN S3')
print('='*80)
print()

all_complete = True
missing_summary = []

for cohort, age_bands in cohorts.items():
    print(f'{cohort.upper()}:')
    for age_band in age_bands:
        print(f'  {age_band}:')
        missing = []
        for exp_file in expected_files:
            if exp_file in found[cohort][age_band]:
                print(f'    [OK] {exp_file}')
            else:
                print(f'    [MISSING] {exp_file}')
                missing.append(exp_file)
                all_complete = False
        
        if missing:
            missing_summary.append(f'{cohort}/{age_band}: {", ".join(missing)}')
            print(f'    -> Missing {len(missing)} file(s)')
        else:
            print(f'    -> All 4 files present')
        print()

print('='*80)
if all_complete:
    print('STATUS: ALL FILES PRESENT FOR ALL COHORTS')
    print('All visualization files are available in S3!')
else:
    print('STATUS: SOME FILES MISSING')
    print()
    print('Missing files:')
    for item in missing_summary:
        print(f'  - {item}')
print('='*80)
