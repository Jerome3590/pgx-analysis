#!/bin/bash
# Complete FPGrowth Analysis Workflow

echo "========================================="
echo "FPGrowth Analysis - Full Workflow"
echo "========================================="
echo ""

# Step 1: Global Analysis
echo "Step 1: Running Global FPGrowth Analysis..."
echo "This will process drug_name, icd_code, and cpt_code"
echo ""
jupyter nbconvert --to notebook --execute \
  3_fpgrowth_analysis/global_fpgrowth_feature_importance.ipynb \
  --output executed_global_fpgrowth.ipynb

echo ""
echo "✓ Global analysis complete!"
echo ""

# Step 2: Cohort Analysis  
echo "Step 2: Running Cohort-Specific FPGrowth Analysis..."
echo "This will process all cohort combinations"
echo ""
jupyter nbconvert --to notebook --execute \
  3_fpgrowth_analysis/cohort_fpgrowth_feature_importance.ipynb \
  --output executed_cohort_fpgrowth.ipynb

echo ""
echo "✓ Cohort analysis complete!"
echo ""

echo "========================================="
echo "Analysis Complete!"
echo "========================================="
echo ""
echo "Results saved to S3:"
echo "  - gold/fpgrowth/global/{drug_name,icd_code,cpt_code}/"
echo "  - gold/fpgrowth/cohort/{drug_name,icd_code,cpt_code}/"
echo ""
echo "Executed notebooks saved locally:"
echo "  - 3_fpgrowth_analysis/executed_global_fpgrowth.ipynb"
echo "  - 3_fpgrowth_analysis/executed_cohort_fpgrowth.ipynb"
echo ""
