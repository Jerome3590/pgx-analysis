# Workflow Testing and Validation

**Testing Framework for Sequential Analysis Workflow**

This document describes the workflow testing framework used to validate the sequential analysis pipeline, ensuring each step completes successfully before proceeding to the next.

---

## Overview

The workflow follows a **sequential execution pattern** where each analysis step must be completed (all outputs and plots generated) before proceeding to the next step:

```
3a_feature_importance → 4_model_data → 5_pgx_analysis → 6_final_model → 7_shap_analysis → 8_ffa_analysis → 9_risk_dashboard
```
FP-Growth, BupaR, and DTW are used for dashboard visualizations only (`9_risk_dashboard/visualizations/`).

Each step has:
- **Output Manifests**: Documented expected outputs in each step's README
- **Validation Scripts**: Tools to verify completion
- **Completion Criteria**: Clear definition of "complete"

---

## Test Example: Cohort 1, Age Band 0-12

**Date:** December 9, 2025  
**Step:** 3a_feature_importance  
**Test Status:** ✅ Complete

### Test Results

#### Data Files Status: ✅ COMPLETE (5/5)

All required data files exist in `outputs/`:

- ✅ `opioid_ed_0_12_aggregated_feature_importance.csv` (69,835 bytes)
- ✅ `opioid_ed_0_12_catboost_feature_importance.csv` (50,812 bytes)
- ✅ `opioid_ed_0_12_xgboost_feature_importance.csv` (40,433 bytes)
- ✅ `opioid_ed_0_12_xgboost_rf_feature_importance.csv` (40,921 bytes)
- ✅ `opioid_ed_0_12_constant_features.csv` (4,875 bytes)

**Source:** Files downloaded from S3 (idempotency check found existing results)

#### Visualization Files Status: ✅ COMPLETE (4/4)

All plots generated in `outputs/plots/`:

- ✅ `opioid_ed_0-12_2019_top50_features.png`
- ✅ `opioid_ed_0-12_2019_top50_with_recall.png`
- ✅ `opioid_ed_0-12_2019_normalized_vs_scaled.png`
- ✅ `opioid_ed_0-12_2019_category_distribution.png`

---

## Workflow Validation

### ✅ Idempotency Test: PASSED
- Script correctly detected existing results in S3
- Downloaded aggregated file instead of recomputing
- Individual model files available in S3
- No redundant computation performed

### ✅ Output Structure: PASSED
- `outputs/` directory exists
- `outputs/plots/` directory created automatically
- Files follow naming convention from manifest
- Matches standard output structure framework

### ✅ Manifest Compliance: PASSED
- All data files match manifest expectations
- All visualization files match manifest expectations
- File naming conventions followed
- File sizes reasonable

### ✅ Sequential Execution: PASSED
- Step 3 completed successfully
- All outputs validated
- Ready to proceed to Step 4

---

## Validation Tools

### Feature Importance Validation

**Script:** `3a_feature_importance/validate_outputs.py`

**Usage:**
```bash
python 3a_feature_importance/validate_outputs.py opioid_ed 0_12
```

**Output:**
```
======================================================================
Step 3: Feature Importance - Output Validation
======================================================================

Cohort: opioid_ed, Age Band: 0_12
Output Directory: 3a_feature_importance/outputs
Plots Directory: 3a_feature_importance/outputs/plots

Data Files Status:
----------------------------------------------------------------------
[OK] opioid_ed_0_12_aggregated_feature_importance.csv (69,835 bytes)
[OK] opioid_ed_0_12_catboost_feature_importance.csv (50,812 bytes)
...

Visualization Files Status:
----------------------------------------------------------------------
[OK] opioid_ed_0-12_2019_top50_features.png (400,695 bytes)
...

======================================================================
Summary
======================================================================
Data Files: 5/5 complete
Plots: 4/4 complete

[SUCCESS] Step 3 COMPLETE - Ready to proceed to Step 4
```

---

## Issues Identified and Resolved

### 1. Missing Recall Columns in Aggregated File
- **Issue**: Aggregated file missing `mc_cv_recall_mean` and `mc_cv_recall_std` columns
- **Resolution**: Updated aggregation function to include these columns
- **Status**: ✅ Fixed

### 2. Visualization Script Path Issues
- **Issue**: R script auto-execution when sourced causes path resolution problems
- **Resolution**: Created Python visualization script with better path handling
- **Status**: ✅ Fixed

### 3. Unicode Display Issues
- **Issue**: Windows console encoding issues with checkmark characters
- **Resolution**: Updated validation script to use ASCII-compatible status indicators
- **Status**: ✅ Fixed

### 4. Cross-Platform Compatibility
- **Issue**: Scripts needed to work on both Linux EC2 and Windows
- **Resolution**: Implemented platform detection and automatic backend selection
- **Status**: ✅ Fixed

---

## Workflow Framework Validation

### ✅ Sequential Execution
Framework enforces completion before proceeding:
- Each step must complete all outputs before next step begins
- Validation scripts check completion status
- Clear error messages if step incomplete

### ✅ Output Manifests
Manifests documented in each step's README:
- Expected data files with patterns
- Expected visualization files with patterns
- File descriptions and requirements
- Completion checklists

### ✅ Validation Scripts
Tools created for checking completion:
- `3a_feature_importance/validate_outputs.py` - Feature importance validation
- Can be extended for other steps

### ✅ Directory Structure
Standard structure followed:
- `{analysis_folder}/outputs/` for data files
- `{analysis_folder}/outputs/plots/` for visualization files
- Consistent across all analysis steps

---

## Recommendations

### 1. Automated Validation
- Integrate validation checks into main workflow scripts
- Fail fast if outputs missing or incomplete
- Provide clear error messages

### 2. Completion Tracking
- Consider adding completion markers (e.g., `.complete` files)
- Track completion status in metadata files
- Enable resumption of interrupted workflows

### 3. Cross-Step Validation
- Verify inputs from previous steps exist
- Check data format compatibility
- Validate data quality before processing

### 4. Documentation
- Keep manifests up-to-date as outputs change
- Document any deviations from standard structure
- Include troubleshooting guides

---

## Testing Checklist

For each analysis step, verify:

- [ ] All expected data files exist
- [ ] All expected visualization files exist
- [ ] Files match naming patterns in manifest
- [ ] File sizes are reasonable (not empty, not suspiciously large)
- [ ] Validation script passes
- [ ] Outputs uploaded to S3 (if applicable)
- [ ] README completion checklist satisfied
- [ ] Ready to proceed to next step

---

## Related Documentation

- [`docs/README_output_structure.md`](README_output_structure.md) - Standard output structure framework
- [`3a_feature_importance/README.md`](../3a_feature_importance/README.md) - Step 3 analysis documentation
- [`docs/README_feature_importance.md`](README_feature_importance.md) - Feature importance analysis guide
- [`docs/README_feature_importance_visualization.md`](README_feature_importance_visualization.md) - Visualization guide

---

**Last Updated:** December 9, 2025  
**Test Framework Status:** ✅ Validated

