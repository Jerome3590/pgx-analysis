#!/usr/bin/env python3
"""
Check S3 for cohort feature importance files and verify Step 4a configuration.
"""

import sys
from pathlib import Path

try:
    import boto3
    from botocore.exceptions import ClientError
except ImportError:
    print("ERROR: boto3 not installed. Install with: pip install boto3")
    sys.exit(1)

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

S3_BUCKET = "pgxdatalake"
S3_PREFIX = "gold/feature_importance/"

# Expected cohorts and age bands
EXPECTED_COHORTS = {
    "opioid_ed": ["13-24", "25-44", "45-54", "55-64"],
    "non_opioid_ed": ["65-74", "75-84", "85-94"]
}

def check_s3_files():
    """Check S3 for cohort feature importance files."""
    s3_client = boto3.client('s3')
    
    print("=" * 80)
    print("Checking S3 for Cohort Feature Importance Files")
    print("=" * 80)
    print(f"S3 Bucket: {S3_BUCKET}")
    print(f"S3 Prefix: {S3_PREFIX}")
    print()
    
    found_files = {}
    missing_files = []
    
    # Check each expected cohort/age_band combination
    for cohort, age_bands in EXPECTED_COHORTS.items():
        print(f"\nCohort: {cohort}")
        print("-" * 80)
        
        for age_band in age_bands:
            age_band_fname = age_band.replace("-", "_")
            s3_key = f"{S3_PREFIX}{cohort}/{age_band}/{cohort}_{age_band_fname}_cohort_feature_importance.csv"
            
            try:
                # Check if file exists
                response = s3_client.head_object(Bucket=S3_BUCKET, Key=s3_key)
                size_mb = response['ContentLength'] / (1024 * 1024)
                last_modified = response['LastModified']
                
                found_files[(cohort, age_band)] = {
                    'size_mb': size_mb,
                    'last_modified': last_modified,
                    's3_key': s3_key
                }
                
                print(f"  [OK] {age_band}: Found ({size_mb:.2f} MB, modified: {last_modified.strftime('%Y-%m-%d %H:%M:%S')})")
                print(f"     s3://{S3_BUCKET}/{s3_key}")
                
            except ClientError as e:
                if e.response['Error']['Code'] == '404':
                    missing_files.append((cohort, age_band, s3_key))
                    print(f"  [MISSING] {age_band}: NOT FOUND")
                else:
                    print(f"  [WARN] {age_band}: Error checking - {e}")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"[OK] Found: {len(found_files)} files")
    print(f"[MISSING] Missing: {len(missing_files)} files")
    
    if missing_files:
        print("\nMissing files:")
        for cohort, age_band, s3_key in missing_files:
            print(f"  - {cohort}/{age_band}: s3://{S3_BUCKET}/{s3_key}")
    
    # Check Step 4a configuration
    print("\n" + "=" * 80)
    print("Step 4a Configuration Check")
    print("=" * 80)
    
    step4a_script = PROJECT_ROOT / "4a_model_data" / "create_model_data.py"
    if step4a_script.exists():
        print(f"[OK] Step 4a script found: {step4a_script}")
        
        # Check if it has S3 download function
        with open(step4a_script, 'r') as f:
            content = f.read()
            if 'download_cohort_feature_importance_from_s3' in content:
                print("[OK] Step 4a has S3 download function")
            else:
                print("[WARN] Step 4a may not have S3 download function")
            
            if 'gold/feature_importance' in content:
                print("[OK] Step 4a configured to use S3 path: gold/feature_importance/")
            else:
                print("[WARN] Step 4a may not be configured for S3 paths")
    else:
        print(f"[ERROR] Step 4a script not found: {step4a_script}")
    
    return found_files, missing_files

if __name__ == "__main__":
    try:
        found, missing = check_s3_files()
        
        if missing:
            print("\n" + "=" * 80)
            print("RECOMMENDATIONS")
            print("=" * 80)
            print("To upload missing files, run Step 3b for each missing cohort/age_band:")
            for cohort, age_band, _ in missing:
                print(f"  python 3b_feature_importance_eda/run_step_3b.py --cohort {cohort} --age-band {age_band}")
            print("\nOr download from S3 if they exist elsewhere:")
            print("  python 4a_model_data/create_model_data.py --download-from-s3")
    except Exception as e:
        print(f"\n[ERROR] Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
