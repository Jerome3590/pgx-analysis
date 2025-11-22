#!/bin/bash
# Optimized FPGrowth execution for HP Omen (14 cores, 32GB RAM, RTX 3080 Ti)

echo "=========================================="
echo "FPGrowth Analysis - Optimized for Omen"
echo "=========================================="
echo ""
echo "Hardware:"
echo "  CPU: 14 cores / 20 threads"
echo "  RAM: 32GB (12.8GB available)"
echo "  GPU: RTX 3080 Ti (16GB) - Not used"
echo ""
echo "Configuration:"
echo "  Global: Sequential (handles full dataset)"
echo "  Cohort: 10 parallel workers"
echo ""

# Check if mlxtend is available
python -c "import mlxtend" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "⚠️  mlxtend not found in current Python"
    echo "    Installing packages in Jupyter kernel..."
    python -m pip install --user mlxtend pandas numpy scipy scikit-learn matplotlib --quiet
fi

echo "Starting execution..."
echo ""

# Step 1: Global FPGrowth
echo "=========================================="
echo "STEP 1: Global FPGrowth Analysis"  
echo "=========================================="
echo "Processing: drug_name, icd_code, cpt_code"
echo "Estimated time: 30-60 minutes"
echo ""

jupyter nbconvert \
  --to notebook \
  --execute \
  --ExecutePreprocessor.timeout=7200 \
  --output executed_global_fpgrowth.ipynb \
  3_fpgrowth_analysis/global_fpgrowth_feature_importance.ipynb

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Global analysis complete!"
    echo ""
else
    echo ""
    echo "✗ Global analysis failed - check executed_global_fpgrowth.ipynb for errors"
    exit 1
fi

# Step 2: Cohort FPGrowth
echo "=========================================="
echo "STEP 2: Cohort-Specific FPGrowth Analysis"
echo "=========================================="
echo "Processing: 90 cohorts × 3 item types = 270 jobs"
echo "Parallel workers: 10"
echo "Estimated time: 2-4 hours"
echo ""

jupyter nbconvert \
  --to notebook \
  --execute \
  --ExecutePreprocessor.timeout=18000 \
  --output executed_cohort_fpgrowth.ipynb \
  3_fpgrowth_analysis/cohort_fpgrowth_feature_importance.ipynb

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Cohort analysis complete!"
    echo ""
else
    echo ""
    echo "✗ Cohort analysis failed - check executed_cohort_fpgrowth.ipynb for errors"
    exit 1
fi

echo "=========================================="
echo "ANALYSIS COMPLETE!"
echo "=========================================="
echo ""
echo "Results saved to S3:"
echo "  s3://pgxdatalake/gold/fpgrowth/global/{drug_name,icd_code,cpt_code}/"
echo "  s3://pgxdatalake/gold/fpgrowth/cohort/{drug_name,icd_code,cpt_code}/"
echo ""
echo "Executed notebooks (with outputs/errors):"
echo "  3_fpgrowth_analysis/executed_global_fpgrowth.ipynb"
echo "  3_fpgrowth_analysis/executed_cohort_fpgrowth.ipynb"
echo ""
echo "✓ Done!"

