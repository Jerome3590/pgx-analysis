#!/usr/bin/env bash
# Clear ALL pipeline outputs and checkpoints for all cohorts
# This will force a complete pipeline restart from scratch

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "========================================"
echo "CLEARING ALL PIPELINE OUTPUTS"
echo "========================================"
echo ""
echo "This will delete:"
echo "  - All local outputs (Steps 3-8)"
echo "  - All S3 outputs (Steps 3-8)"
echo "  - All S3 checkpoints"
echo "  - All time logs"
echo ""
read -p "Are you sure you want to continue? (yes/no): " confirm
if [ "$confirm" != "yes" ]; then
    echo "Aborted."
    exit 1
fi

# Step 1: Clear Steps 6, 7, 8 (Models, SHAP, FFA)
echo ""
echo "[1/6] Clearing Steps 6, 7, 8 (Models, SHAP, FFA)..."
bash utility_scripts/clear_models.sh --all --s3

# Step 2: Clear Step 5 (PGx)
echo ""
echo "[2/6] Clearing Step 5 (PGx Analysis)..."
# Get all cohorts and age bands
COHORTS=("opioid_ed" "non_opioid_ed")
AGE_BANDS=("0-12" "13-24" "25-44" "45-54" "55-64" "65-74" "75-84" "85-94")

for cohort in "${COHORTS[@]}"; do
    for age_band in "${AGE_BANDS[@]}"; do
        python utility_scripts/clear_pgx_step5_outputs.py --cohort "$cohort" --age-band "$age_band" 2>/dev/null || true
    done
done

# Step 3: Clear Step 4b (DTW Filter outputs)
echo ""
echo "[3/6] Clearing Step 4b (DTW Filter outputs)..."
# Clear local outputs
rm -rf 4b_event_filter/outputs/*
rm -rf 4a_model_data/cohort_name=*/age_band=*/model_events_no_protocols.parquet 2>/dev/null || true

# Clear S3 outputs
python -c "
import boto3
s3_client = boto3.client('s3')
bucket = 'pgxdatalake'
prefixes = [
    'gold/model_data/cohort_name=',
    'gold/cohorts_model_data/cohort_name=',
]

for prefix in prefixes:
    paginator = s3_client.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=bucket, Prefix=prefix)
    objects_to_delete = []
    for page in pages:
        if 'Contents' in page:
            for obj in page['Contents']:
                if 'model_events_no_protocols.parquet' in obj['Key']:
                    objects_to_delete.append({'Key': obj['Key']})
    if objects_to_delete:
        s3_client.delete_objects(Bucket=bucket, Delete={'Objects': objects_to_delete})
        print(f'  Deleted {len(objects_to_delete)} filtered model data files')
" 2>/dev/null || echo "  (S3 clearing skipped - may need AWS credentials)"

# Step 4: Clear Step 4a (Model Data)
echo ""
echo "[4/6] Clearing Step 4a (Model Data)..."
bash utility_scripts/clean_model_data.sh 2>/dev/null || true

# Step 5: Clear Step 3 (Feature Importance)
echo ""
echo "[5/6] Clearing Step 3 (Feature Importance)..."
# Clear local outputs
rm -rf 3_feature_importance/outputs/*
rm -rf 3_feature_importance/from_s3/*

# Clear S3 outputs
python -c "
import boto3
s3_client = boto3.client('s3')
bucket = 'pgxdatalake'
prefix = 'gold/feature_importance/'
paginator = s3_client.get_paginator('list_objects_v2')
pages = paginator.paginate(Bucket=bucket, Prefix=prefix)
objects_to_delete = []
for page in pages:
    if 'Contents' in page:
        for obj in page['Contents']:
            objects_to_delete.append({'Key': obj['Key']})
if objects_to_delete:
    s3_client.delete_objects(Bucket=bucket, Delete={'Objects': objects_to_delete})
    print(f'  Deleted {len(objects_to_delete)} feature importance files')
" 2>/dev/null || echo "  (S3 clearing skipped - may need AWS credentials)"

# Step 6: Clear ALL S3 checkpoints
echo ""
echo "[6/6] Clearing ALL S3 checkpoints..."
python -c "
import sys
try:
    import boto3
    s3_client = boto3.client('s3')
    checkpoint_bucket = 'pgx-repository'
    prefix = 'pipeline_checkpoints/'
    paginator = s3_client.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=checkpoint_bucket, Prefix=prefix)
    objects_to_delete = []
    for page in pages:
        if 'Contents' in page:
            for obj in page['Contents']:
                objects_to_delete.append({'Key': obj['Key']})
    if objects_to_delete:
        # Delete in batches of 1000
        for i in range(0, len(objects_to_delete), 1000):
            batch = objects_to_delete[i:i+1000]
            s3_client.delete_objects(Bucket=checkpoint_bucket, Delete={'Objects': batch})
        print(f'  ✓ Deleted {len(objects_to_delete)} checkpoint files')
    else:
        print('  ✓ No checkpoints found')
except ImportError:
    print('  ✗ boto3 not installed - skipping S3 checkpoint clearing')
    sys.exit(1)
except Exception as e:
    print(f'  ✗ Error clearing S3 checkpoints: {e}')
    sys.exit(1)
" 2>&1 || echo "  (S3 clearing skipped - check AWS credentials and boto3 installation)"

# Clear time logs
echo ""
echo "Clearing time logs..."
rm -f utility_scripts/time_log.json 2>/dev/null || true

echo ""
echo "========================================"
echo "PIPELINE CLEARED - READY FOR FRESH START"
echo "========================================"
echo ""
echo "Next steps:"
echo "  1. Run: bash utility_scripts/run_all_cohorts_workflow.sh"
echo "  2. Or run individual cohorts: bash utility_scripts/run_cohort_workflow.sh <cohort> <age_band>"
