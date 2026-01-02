# Workflow Completion Action Plan

**Generated:** 2025-01-02  
**Status:** Active - Ready for Execution

---

## Overview

This document provides a step-by-step action plan to complete the remaining workflow steps for all cohorts.

### Current Status Summary

- ✅ **Fully Complete (3 cohorts):** `opioid_ed` 13-24, 25-44, 45-54
- ⏳ **Partially Complete (5 cohorts):** `opioid_ed` 0-12, 55-64; `non_opioid_ed` 65-74, 75-84, 85-94

---

## Priority Order

### Priority 1: Production Cohorts (non_opioid_ed)
These are the primary production cohorts for the risk dashboard:
- `non_opioid_ed / 65-74`
- `non_opioid_ed / 75-84`
- `non_opioid_ed / 85-94`

### Priority 2: Remaining opioid_ed Cohorts
- `opioid_ed / 55-64` (production cohort)
- `opioid_ed / 0-12` (test/smoke test cohort)

---

## Detailed Action Plan by Cohort

### Cohort 1: non_opioid_ed / 65-74

**Current Status:** Steps 3-4a complete  
**Remaining Steps:** 4b, 5a-5d, 6-8

#### Step 4b: DTW Protocol Filtering
```bash
python 4b_dtw_filter/filter_protocol_events.py \
  --cohort-name non_opioid_ed \
  --age-band 65-74
```
**Expected Output:** `4a_model_data/cohort_name=non_opioid_ed/age_band=65-74/model_events_no_protocols.parquet`

#### Step 4c: Extreme Density Cohort Split (Optional but Recommended)
```bash
python 5b_fpgrowth_analysis/extract_extreme_density_cohort.py \
  --cohort-name non_opioid_ed \
  --age-band 65-74
```
**Expected Outputs:**
- `4a_model_data/cohort_name=non_opioid_ed_extreme_density/age_band=65-74/model_events.parquet`
- `4a_model_data/cohort_name=non_opioid_ed/age_band=65-74/model_events.parquet` (rewritten without extreme patients)

#### Step 5b: FP-Growth Analysis
```bash
python 5b_fpgrowth_analysis/run_analysis.py \
  --cohort-name non_opioid_ed \
  --age-band 65-74
```
**Or use notebook:**
```bash
# Open: 5b_fpgrowth_analysis/fpgrowth_cohort_runner.ipynb
# Configure: COHORTS_TO_RUN = ["non_opioid_ed"]
# Configure: AGE_BANDS_TO_RUN = ["65-74"]
```
**Expected Outputs:**
- `5_feature_engineering/feature_engineering_outputs/4_fpgrowth/non_opioid_ed/65-74/fpgrowth_added_features_non_opioid_ed_65_74.csv`
- `5b_fpgrowth_analysis/outputs/...` (itemsets, rules, plots)

#### Step 5a: BupaR Process Mining
```bash
python 5a_bupaR_analysis/run_analysis.py \
  --cohort-name non_opioid_ed \
  --age-band 65-74
```
**Or use notebook:**
```bash
# Open: 5a_bupaR_analysis/bupar_cohort_runner.ipynb
# Configure: COHORTS_TO_RUN = ["non_opioid_ed"]
# Configure: AGE_BANDS_TO_RUN = ["65-74"]
```
**Expected Outputs:**
- `5_feature_engineering/feature_engineering_outputs/5_bupar/non_opioid_ed/65-74/bupaR_added_features_non_opioid_ed_65_74.csv`
- `5a_bupaR_analysis/outputs/non_opioid_ed/65_74/...` (features, plots)

#### Step 5c: PGx Feature Engineering
```bash
python 5c_pgx_analysis/run_analysis.py \
  --cohort-name non_opioid_ed \
  --age-band 65-74
```
**Or use notebook:**
```bash
# Open: 5c_pgx_analysis/pgx_cohort_runner.ipynb
# Configure: COHORTS_TO_RUN = ["non_opioid_ed"]
# Configure: AGE_BANDS_TO_RUN = ["65-74"]
```
**Expected Outputs:**
- `5_feature_engineering/feature_engineering_outputs/7_pgx/non_opioid_ed/65-74/pgx_added_features_non_opioid_ed_65_74.csv`

#### Step 5d: DTW Trajectory Features
```bash
# Step 1: Create DTW features
python 5d_dtw_analysis/create_dtw_features.py \
  --cohort non_opioid_ed \
  --age_band 65-74

# Step 2: Add DTW features to model data
python 5d_dtw_analysis/add_dtw_features_to_model_data.py \
  --cohort-name non_opioid_ed \
  --age-band 65-74
```
**Or use notebook:**
```bash
# Open: 5d_dtw_analysis/dtw_cohort_runner.ipynb
# Configure: COHORTS_TO_RUN = ["non_opioid_ed"]
# Configure: AGE_BANDS_TO_RUN = ["65-74"]
```
**Expected Outputs:**
- `5_feature_engineering/feature_engineering_outputs/6_dtw/non_opioid_ed/65-74/dtw_added_features_non_opioid_ed_65_74.csv`

#### Step 6: Final Model Training
```bash
python 6b_final_model_selection/run_final_model.py \
  --cohort non_opioid_ed \
  --age_band 65-74
```
**Or use notebook:**
```bash
# Open: 6b_final_model_selection/final_model_cohort_runner.ipynb
# Configure: COHORTS_TO_RUN = ["non_opioid_ed"]
# Configure: AGE_BANDS_TO_RUN = ["65-74"]
```
**Expected Outputs:**
- `6_final_model/model_outputs/non_opioid_ed/65_74/non_opioid_ed_65_74_final_model_xgboost.json`
- `6_final_model/model_outputs/non_opioid_ed/65_74/non_opioid_ed_65_74_final_model_catboost.cbm`

#### Step 7: FFA Analysis
```bash
python 7_ffa_analysis/run_full_ffa_analysis.py \
  --cohort-name non_opioid_ed \
  --age-band 65-74
```
**Or use notebook:**
```bash
# Open: 7_ffa_analysis/ffa_cohort_runner.ipynb
# Configure: COHORTS_TO_RUN = ["non_opioid_ed"]
# Configure: AGE_BANDS_TO_RUN = ["65-74"]
```
**Expected Outputs:**
- `7_ffa_analysis/outputs/non_opioid_ed/65_74/xgboost/analysis_summary.json`
- `7_ffa_analysis/outputs/non_opioid_ed/65_74/xgboost/feature_importance_axp.csv`

#### Step 8: SHAP Analysis
```bash
python 8_shap_analysis/run_shap_analysis.py \
  --cohort non_opioid_ed \
  --age_band 65-74
```
**Or use notebook:**
```bash
# Open: 8_shap_analysis/shap_cohort_runner.ipynb
# Configure: COHORTS_TO_RUN = ["non_opioid_ed"]
# Configure: AGE_BANDS_TO_RUN = ["65-74"]
```
**Expected Outputs:**
- `8_shap_analysis/outputs/non_opioid_ed/65_74/non_opioid_ed_65_74_shap_global_importance_xgboost.csv`
- `8_shap_analysis/outputs/non_opioid_ed/65_74/non_opioid_ed_65_74_shap_sample_values_xgboost.parquet`

---

### Cohort 2: non_opioid_ed / 75-84

**Same steps as 65-74, replace age-band:**
- All commands use `--age-band 75-84` or `AGE_BANDS_TO_RUN = ["75-84"]`

---

### Cohort 3: non_opioid_ed / 85-94

**Same steps as 65-74, replace age-band:**
- All commands use `--age-band 85-94` or `AGE_BANDS_TO_RUN = ["85-94"]`

---

### Cohort 4: opioid_ed / 55-64

**Current Status:** Steps 3-4a, 4b, 4c complete  
**Remaining Steps:** 5a-5d, 6-8

**Note:** This cohort already has:
- ✅ `model_events.parquet`
- ✅ `model_events_no_protocols.parquet`
- ✅ Extreme density split completed

**Follow Steps 5a-5d, 6-8 from Cohort 1 above, using:**
- `--cohort-name opioid_ed`
- `--age-band 55-64`

---

### Cohort 5: opioid_ed / 0-12

**Current Status:** Steps 3-4a complete  
**Remaining Steps:** 4b, 5a-5d, 6-8

**Note:** This is a test/smoke test cohort. Follow all steps from Cohort 1, using:
- `--cohort-name opioid_ed`
- `--age-band 0-12`

---

## Batch Execution Strategy

### Option 1: Use Cohort Runner Notebooks (Recommended)

Each step has a dedicated cohort runner notebook that can process multiple cohorts/age bands:

1. **FP-Growth:** `5b_fpgrowth_analysis/fpgrowth_cohort_runner.ipynb`
2. **BupaR:** `5a_bupaR_analysis/bupar_cohort_runner.ipynb`
3. **PGx:** `5c_pgx_analysis/pgx_cohort_runner.ipynb`
4. **DTW:** `5d_dtw_analysis/dtw_cohort_runner.ipynb`
5. **Final Model:** `6b_final_model_selection/final_model_cohort_runner.ipynb`
6. **FFA:** `7_ffa_analysis/ffa_cohort_runner.ipynb`
7. **SHAP:** `8_shap_analysis/shap_cohort_runner.ipynb`

**Configuration Example:**
```python
# In each notebook, set:
COHORTS_TO_RUN = ["non_opioid_ed", "opioid_ed"]
AGE_BANDS_TO_RUN = ["65-74", "75-84", "85-94", "55-64", "0-12"]
```

### Option 2: Use Main Pipeline Notebook

**Open:** `pgx_cohort_pipeline.ipynb`

This notebook provides:
- Status checking functions
- Pipeline execution functions
- Batch status overview

**Example usage:**
```python
# Check status for all cohorts
print_batch_status_table()

# Run full pipeline for a specific cohort
run_full_pipeline("non_opioid_ed", "65-74")

# Run from a specific step
run_full_pipeline("non_opioid_ed", "65-74", start_from="5a")
```

---

## Step Dependencies

### Critical Dependencies

1. **Step 4a → Step 4b:** Model data must exist before protocol filtering
2. **Step 4b → Step 5a-5d:** Protocol-filtered data preferred (but can use unfiltered)
3. **Step 5b → Step 5a:** FP-Growth itemsets used by BupaR (target-only itemsets)
4. **Step 5a-5d → Step 6:** All feature engineering must complete before final model
5. **Step 6 → Step 7-8:** Final model must exist before FFA/SHAP analysis

### Optional Dependencies

- **Step 4c (Extreme Density Split):** Can be run before or after Step 4b, but should be before Step 5b
- **Step 5d (DTW Features):** Can run independently, but benefits from protocol-filtered data

---

## Execution Order Recommendations

### For Production Cohorts (non_opioid_ed)

**Day 1: Data Preparation**
1. Run Step 4b (DTW Protocol Filtering) for all 3 cohorts
2. Run Step 4c (Extreme Density Split) for all 3 cohorts

**Day 2: Feature Engineering**
1. Run Step 5b (FP-Growth) for all 3 cohorts
2. Run Step 5a (BupaR) for all 3 cohorts
3. Run Step 5c (PGx) for all 3 cohorts
4. Run Step 5d (DTW) for all 3 cohorts

**Day 3: Modeling & Analysis**
1. Run Step 6 (Final Model) for all 3 cohorts
2. Run Step 7 (FFA) for all 3 cohorts
3. Run Step 8 (SHAP) for all 3 cohorts

### For Remaining opioid_ed Cohorts

**Follow same pattern as production cohorts, but can run in parallel or sequentially.**

---

## Verification Checklist

After completing each step, verify:

### Step 4b Verification
- [ ] `model_events_no_protocols.parquet` exists
- [ ] File size is reasonable (not empty)
- [ ] Can read parquet file successfully

### Step 4c Verification
- [ ] Extreme density cohort parquet exists
- [ ] `extreme_density_patients_{age_band}.csv` exists
- [ ] Main cohort parquet has been rewritten (check modification time)

### Step 5b Verification
- [ ] `fpgrowth_added_features_{cohort}_{age_band}.csv` exists
- [ ] Itemsets JSON files exist in `5b_fpgrowth_analysis/outputs/`
- [ ] Plots exist in `5_feature_engineering/feature_engineering_outputs/4_fpgrowth/{cohort}/{age_band}/plots/`

### Step 5a Verification
- [ ] `bupaR_added_features_{cohort}_{age_band}.csv` exists
- [ ] Feature files exist in `5a_bupaR_analysis/outputs/{cohort}/{age_band}/features/`
- [ ] Plots exist in `5_feature_engineering/feature_engineering_outputs/5_bupar/{cohort}/{age_band}/plots/`

### Step 5c Verification
- [ ] `pgx_added_features_{cohort}_{age_band}.csv` exists
- [ ] Feature files exist in `5c_pgx_analysis/outputs/`

### Step 5d Verification
- [ ] `dtw_added_features_{cohort}_{age_band}.csv` exists
- [ ] Feature files exist in `5d_dtw_analysis/outputs/`
- [ ] Plots exist (if generated)

### Step 6 Verification
- [ ] `{cohort}_{age_band}_final_model_xgboost.json` exists
- [ ] `{cohort}_{age_band}_final_model_catboost.cbm` exists
- [ ] Model metrics JSON exists

### Step 7 Verification
- [ ] `analysis_summary.json` exists in `7_ffa_analysis/outputs/{cohort}/{age_band}/xgboost/`
- [ ] `feature_importance_axp.csv` exists
- [ ] `causal_importance.csv` exists

### Step 8 Verification
- [ ] `{cohort}_{age_band}_shap_global_importance_xgboost.csv` exists
- [ ] `{cohort}_{age_band}_shap_global_importance_catboost.csv` exists
- [ ] `{cohort}_{age_band}_shap_sample_values_xgboost.parquet` exists
- [ ] `{cohort}_{age_band}_shap_sample_values_catboost.parquet` exists

---

## Known Issues & Notes

### DTW Features (Step 5d)

**Issue:** DTW features are missing for all cohorts, including completed ones.

**Investigation Needed:**
1. Check if DTW features were intentionally skipped for some cohorts
2. Verify DTW script requirements (may need protocol-filtered data)
3. Check if DTW features are optional for final model

**Action:** Investigate why DTW features weren't generated for completed cohorts (13-24, 25-44, 45-44) before running for new cohorts.

### Extreme Density Cohorts

**Current Status:** Only `opioid_ed_extreme_density / 55-64` exists.

**Recommendation:** 
- Run Step 4c for all production cohorts (non_opioid_ed 65-74, 75-84, 85-94)
- Consider running for opioid_ed 13-24, 25-44, 45-54 if not already done

### Resource Considerations

**Memory-Intensive Steps:**
- Step 5b (FP-Growth): Consider running on EC2 for large cohorts
- Step 6 (Final Model): May benefit from GPU if available
- Step 8 (SHAP): Can be memory-intensive for large feature sets

**Time Estimates (per cohort) - Steps 4b-8 Only:**

**Note:** These estimates are for Steps 4b-8 only. Step 3 (Feature Importance) is already complete for all cohorts and took ~6.5-7.5 hours per cohort when run.

**Time varies significantly by cohort size:**

| Cohort Size | Step 4b | Step 4c | Step 5b | Step 5a | Step 5c | Step 5d | Step 6 | Step 7 | Step 8 | **Total** |
|-------------|---------|---------|---------|---------|---------|---------|--------|--------|--------|-----------|
| **Small** (0-12: 2K events) | 5 min | 5 min | 15 min | 10 min | 5 min | 10 min | 15 min | 30 min | 30 min | **~2 hours** |
| **Medium** (13-24: 436K events) | 20 min | 10 min | 1-2 hrs | 30 min | 15 min | 30 min | 30 min | 1-2 hrs | 1-2 hrs | **~5-7 hours** |
| **Large** (25-44: 4.6M events) | 30 min | 15 min | 3-5 hrs | 1-2 hrs | 30 min | 1-2 hrs | 1-2 hrs | 2-3 hrs | 2-3 hrs | **~12-18 hours** |
| **Very Large** (55-64: 3.2M events) | 30 min | 15 min | 2-4 hrs | 1-2 hrs | 30 min | 1-2 hrs | 1-2 hrs | 2-3 hrs | 2-3 hrs | **~10-15 hours** |
| **Production** (65-74: 2.9M events) | 30 min | 15 min | 2-4 hrs | 1-2 hrs | 30 min | 1-2 hrs | 1-2 hrs | 2-3 hrs | 2-3 hrs | **~10-15 hours** |

**Key Factors:**
- **Cohort size** (event count) is the primary driver of runtime
- **FP-Growth (Step 5b)** is typically the longest step for large cohorts (2-5 hours)
- **FFA (Step 7)** and **SHAP (Step 8)** are computationally intensive (2-3 hours each for large cohorts)
- **Small cohorts** (0-12) complete much faster (~2 hours total)
- **Parallel execution** possible where dependencies allow (e.g., Steps 5a, 5c, 5d can run in parallel after Step 5b)

**Production Cohorts (non_opioid_ed 65-74, 75-84, 85-94):**
- **65-74**: ~10-15 hours (2.9M training events)
- **75-84**: ~8-12 hours (1.2M training events) 
- **85-94**: ~4-6 hours (274K training events)

**Note:** Actual runtime depends on:
- System resources (CPU cores, RAM, GPU availability)
- Whether running on EC2 vs local Windows
- Network speed for S3 operations
- Whether steps are run sequentially or in parallel

---

## Progress Tracking

Update `status/WORKFLOW_STATUS.md` after completing each step for each cohort.

**Template for updates:**
```markdown
### {Date} - {Cohort} / {Age Band} - Step {X} Complete
- ✅ Step {X}: {Step Name}
- Command: `{command used}`
- Outputs: {list key outputs}
- Notes: {any issues or observations}
```

---

## Quick Reference: All Commands

### non_opioid_ed / 65-74 (Production Priority 1)

```bash
# Step 4b
python 4b_dtw_filter/filter_protocol_events.py --cohort-name non_opioid_ed --age-band 65-74

# Step 4c
python 5b_fpgrowth_analysis/extract_extreme_density_cohort.py --cohort-name non_opioid_ed --age-band 65-74

# Step 5b
python 5b_fpgrowth_analysis/run_analysis.py --cohort-name non_opioid_ed --age-band 65-74

# Step 5a
python 5a_bupaR_analysis/run_analysis.py --cohort-name non_opioid_ed --age-band 65-74

# Step 5c
python 5c_pgx_analysis/run_analysis.py --cohort-name non_opioid_ed --age-band 65-74

# Step 5d
python 5d_dtw_analysis/create_dtw_features.py --cohort non_opioid_ed --age_band 65-74
python 5d_dtw_analysis/add_dtw_features_to_model_data.py --cohort-name non_opioid_ed --age-band 65-74

# Step 6
python 6b_final_model_selection/run_final_model.py --cohort non_opioid_ed --age_band 65-74

# Step 7
python 7_ffa_analysis/run_full_ffa_analysis.py --cohort-name non_opioid_ed --age-band 65-74

# Step 8
python 8_shap_analysis/run_shap_analysis.py --cohort non_opioid_ed --age_band 65-74
```

**Repeat for 75-84 and 85-94 (replace age-band in all commands).**

---

## Next Steps

1. **Review this plan** and adjust priorities if needed
2. **Start with Priority 1:** non_opioid_ed / 65-74
3. **Use cohort runner notebooks** for batch processing where possible
4. **Update WORKFLOW_STATUS.md** as steps complete
5. **Investigate DTW features** issue before running Step 5d

---

**Last Updated:** 2025-01-02  
**Status:** Ready for execution
