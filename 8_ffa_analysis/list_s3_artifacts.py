#!/usr/bin/env python3
"""List available artifacts in S3 for testing."""

import boto3
from botocore.exceptions import ClientError

s3_client = boto3.client('s3')
bucket = 'pgxdatalake'

cohort = 'opioid_ed'
age_band = '13-24'
age_band_fname = age_band.replace('-', '_')

# List prefixes to check
prefixes = [
    f"gold/final_model/{cohort}/",
    f"gold/shap_analysis/{cohort}/",
    f"gold/ffa_analysis/{cohort}/",
    f"gold/model_outputs/{cohort}/",
]

print("Searching S3 for available artifacts...")
print("=" * 80)

for prefix in prefixes:
    print(f"\nPrefix: {prefix}")
    try:
        # First check subdirectories
        response = s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix, Delimiter='/', MaxKeys=100)
        
        if 'CommonPrefixes' in response:
            print(f"  Subdirectories:")
            for cp in response['CommonPrefixes'][:5]:
                print(f"    {cp['Prefix']}")
                # List files in subdirectory
                sub_response = s3_client.list_objects_v2(Bucket=bucket, Prefix=cp['Prefix'], MaxKeys=20)
                if 'Contents' in sub_response:
                    for obj in sub_response['Contents'][:5]:
                        print(f"      {obj['Key']}")
        
        # Also check for files directly in prefix
        response_no_delim = s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix, MaxKeys=20)
        if 'Contents' in response_no_delim:
            print(f"  Files in prefix:")
            for obj in response_no_delim['Contents'][:10]:
                print(f"    {obj['Key']}")
    except Exception as e:
        print(f"  Error: {e}")

# Specifically check for model files
print("\n" + "=" * 80)
print("Checking for model JSON files:")
print("=" * 80)

model_paths = [
    f"gold/final_model/{cohort}/{age_band}/final_model_json/",
    f"gold/final_model/{cohort}/{age_band}/",
]

for model_path in model_paths:
    print(f"\nChecking: {model_path}")
    try:
        response = s3_client.list_objects_v2(Bucket=bucket, Prefix=model_path, MaxKeys=50)
        if 'Contents' in response:
            for obj in response['Contents']:
                if '.json' in obj['Key']:
                    print(f"  ✓ {obj['Key']}")
    except Exception as e:
        print(f"  Error: {e}")
