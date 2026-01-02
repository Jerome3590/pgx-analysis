# DTW Analysis Pipeline Troubleshooting Guide

## Overview

The DTW (Dynamic Time Warping) analysis pipeline has two steps:
1. **`create_dtw_features.py`** - Creates DTW trajectory features from patient sequences
2. **`add_dtw_features_to_model_data.py`** - Merges DTW features into final feature file

## Known Issues

### 1. Missing DTW Features for All Cohorts ⚠️

**Status:** DTW features are missing for all cohorts, including completed ones.

**Documented in:** `status/WORKFLOW_ACTION_PLAN.md`

**Possible Causes:**
- DTW analysis was intentionally skipped
- Scripts failed silently
- Missing dependencies or prerequisites
- Path resolution issues

**Action Required:** Investigate why DTW features weren't generated before running on EC2.

---

### 2. Missing Dependencies

#### dtaidistance Package

**Error:**
```
dtaidistance package not available. Install with: pip install dtaidistance
```

**Fix:**
```bash
pip install dtaidistance
```

**Note:** This package is required for DTW distance calculations.

---

### 3. Missing Prerequisites

#### FP-Growth Itemsets Required

**Issue:** DTW analysis depends on FP-Growth itemsets to filter allowed activity codes.

**Required Files:**
- `5b_fpgrowth_analysis/outputs/{cohort}/target/{age_band}/train/drug_name_itemsets_target_only.json`
- `5b_fpgrowth_analysis/outputs/{cohort}/target/{age_band}/train/icd_code_itemsets_target_only.json`
- `5b_fpgrowth_analysis/outputs/{cohort}/target/{age_band}/train/cpt_code_itemsets_target_only.json`
- `5b_fpgrowth_analysis/outputs/{cohort}/target/{age_band}/train/medical_code_itemsets_target_only.json`

**Path Resolution:**
The script constructs paths based on:
- `base_cohort` (e.g., `opioid_ed` for `opioid_ed_extreme_density`)
- `split_type` (default: `"target"`)
- `age_band_fname` (e.g., `"0_12"` for `"0-12"`)
- `event_year` (default: `"train"`)

**Check:**
```bash
# Verify FP-Growth itemsets exist
ls -la 5b_fpgrowth_analysis/outputs/{cohort}/target/{age_band}/train/*itemsets*.json
```

**Action:** Ensure Step 5b (FP-Growth) is complete before running DTW.

---

#### Model Data Required

**Required Files:**
- `4a_model_data/cohort_name={cohort}/age_band={age_band}/model_events_no_protocols.parquet` (preferred)
- OR `4a_model_data/cohort_name={cohort}/age_band={age_band}/model_events.parquet` (fallback)

**Check:**
```bash
# Verify model data exists
ls -la 4a_model_data/cohort_name={cohort}/age_band={age_band}/model_events*.parquet
```

**Action:** Ensure Steps 4a (Model Data) and 4b (DTW Protocol Filter) are complete.

---

### 4. Path Resolution Issues

#### Complex Path Construction

**Issue:** The script constructs FP-Growth output paths using multiple variables:
- `base_cohort` extraction (handles `_extreme_density` suffix)
- `split_type` (default: `"target"`)
- `age_band_fname` (converts `"0-12"` to `"0_12"`)
- `event_year` (default: `"train"`)

**Example Path:**
```
5b_fpgrowth_analysis/outputs/opioid_ed/target/0_12/train/drug_name_itemsets_target_only.json
```

**Potential Issues:**
- Age band format mismatch (`"0-12"` vs `"0_12"`)
- Split type mismatch (`"target"` vs `"combined"`)
- Event year mismatch (`"train"` vs `"2019"`)
- Base cohort extraction for extreme-density cohorts

**Debug:**
Add logging to `create_dtw_features.py` to print constructed paths:
```python
logger.info(f"FP-Growth output directory: {fpgrowth_output_dir}")
logger.info(f"Looking for itemsets in: {itemsets_files}")
```

---

### 5. Target Leakage Concerns

**Documented in:** `docs/CrossStep_Development/README_target_leakage.md`

**Issue:** Original DTW implementation had target leakage:
- Prototypes were based on target patients' trajectories ending at target event
- Trajectories were truncated at target event
- Control patients don't have target events, causing inconsistent features

**Current Status:** The script uses cutoff dates:
- **Target patients:** Use F1120 date as cutoff (events before F1120)
- **Control patients:** Use reference date (first event date)

**Verification Needed:** Ensure cutoff date logic prevents leakage.

---

### 6. Two-Step Process Complexity

**Issue:** DTW requires two sequential steps:
1. `create_dtw_features.py` → Creates `dtw_features_{cohort}_{age_band}.csv`
2. `add_dtw_features_to_model_data.py` → Creates `dtw_added_features_{cohort}_{age_band}.csv`

**Failure Points:**
- Step 1 fails → Step 2 can't find input file
- Step 2 fails → Final features not created
- No clear error if Step 1 partially completes

**Check Intermediate File:**
```bash
# Verify Step 1 output exists
ls -la 5d_dtw_analysis/outputs/feature_engineering/dtw_features_{cohort}_{age_band}.csv
```

---

### 7. Empty Trajectories

**Issue:** If no trajectories are extracted, the script returns an empty DataFrame.

**Possible Causes:**
- No allowed codes from FP-Growth itemsets
- Model data has no matching events
- Cutoff dates exclude all events
- Item type filtering too restrictive

**Debug:**
```python
# Check allowed codes count
logger.info(f"Total allowed codes: {len(allowed_codes)}")

# Check trajectory extraction
logger.info(f"Extracted {len(patient_trajectories)} trajectories for {item_type}")
```

---

### 8. Memory Issues

**Issue:** DTW distance calculations can be memory-intensive for large cohorts.

**Symptoms:**
- Script hangs or crashes
- Out of memory errors
- Slow execution

**Mitigation:**
- Reduce number of prototypes (`--n_prototypes 3` instead of `5`)
- Process smaller cohorts first
- Use protocol-filtered data (smaller dataset)
- Consider sampling for initial testing

---

## Pre-Flight Checklist

Before running DTW analysis on EC2, verify:

- [ ] **dtaidistance package installed**
  ```bash
  pip list | grep dtaidistance
  ```

- [ ] **Step 5b (FP-Growth) complete**
  ```bash
  ls 5b_fpgrowth_analysis/outputs/{cohort}/target/{age_band}/train/*itemsets*.json
  ```

- [ ] **Step 4a (Model Data) complete**
  ```bash
  ls 4a_model_data/cohort_name={cohort}/age_band={age_band}/model_events*.parquet
  ```

- [ ] **Step 4b (DTW Protocol Filter) complete** (preferred)
  ```bash
  ls 4a_model_data/cohort_name={cohort}/age_band={age_band}/model_events_no_protocols.parquet
  ```

- [ ] **Path structure matches expected format**
  - Age band: `0-12` → `0_12` in paths
  - Split type: `target` (not `combined`)
  - Event year: `train` (not `2019`)

- [ ] **Test with one cohort first**
  ```bash
  python 5d_dtw_analysis/create_dtw_features.py --cohort opioid_ed --age_band 0-12
  ```

---

## Testing Strategy

### 1. Test Locally First

```bash
# Test Step 1
python 5d_dtw_analysis/create_dtw_features.py \
  --cohort opioid_ed \
  --age_band 0-12 \
  --n_prototypes 3

# Verify output
ls 5d_dtw_analysis/outputs/feature_engineering/dtw_features_opioid_ed_0_12.csv

# Test Step 2
python 5d_dtw_analysis/add_dtw_features_to_model_data.py \
  --cohort-name opioid_ed \
  --age-band 0-12

# Verify final output
ls 5d_dtw_analysis/outputs/feature_engineering/dtw_added_features_opioid_ed_0_12.csv
```

### 2. Check Logs

```bash
# Look for errors in script output
python 5d_dtw_analysis/create_dtw_features.py --cohort opioid_ed --age_band 0-12 2>&1 | tee dtw_test.log

# Check for:
# - "Model data not found"
# - "Itemsets file not found"
# - "No patients found"
# - "No cutoff dates found"
# - "No trajectories extracted"
```

### 3. Verify Output Format

```python
import pandas as pd

# Load DTW features
df = pd.read_csv('5d_dtw_analysis/outputs/feature_engineering/dtw_added_features_opioid_ed_0_12.csv')

# Check:
# - Has 'mi_person_key' column
# - Has DTW feature columns (distance, length, diversity)
# - Row count matches expected patient count
# - No NaN values in critical columns
print(df.head())
print(df.info())
print(df.isnull().sum())
```

---

## Recommended Fixes

### 1. Add Better Error Handling

**Current:** Scripts return empty DataFrames on failure.

**Recommended:** Raise exceptions with clear error messages:
```python
if not model_data_path.exists():
    raise FileNotFoundError(
        f"Model data not found: {model_data_path}\n"
        f"Expected locations:\n"
        f"  - {model_data_filtered}\n"
        f"  - {model_data_dir / 'model_events.parquet'}\n"
        f"Run Step 4a (create_model_data.py) first."
    )
```

### 2. Add Dependency Checking

**Recommended:** Check prerequisites at script start:
```python
def check_prerequisites(cohort_name, age_band):
    """Check all prerequisites before running DTW analysis."""
    issues = []
    
    # Check model data
    model_data_path = get_model_data_path(cohort_name, age_band)
    if not model_data_path.exists():
        issues.append(f"Model data not found: {model_data_path}")
    
    # Check FP-Growth itemsets
    itemsets_paths = get_itemsets_paths(cohort_name, age_band)
    for path in itemsets_paths:
        if not path.exists():
            issues.append(f"Itemsets file not found: {path}")
    
    if issues:
        raise RuntimeError(
            "Prerequisites not met:\n" + "\n".join(f"  - {issue}" for issue in issues)
        )
```

### 3. Add Path Validation

**Recommended:** Validate constructed paths match actual file structure:
```python
def validate_fpgrowth_paths(fpgrowth_output_dir, age_band_fname):
    """Validate FP-Growth output directory structure."""
    expected_files = [
        "drug_name_itemsets_target_only.json",
        "icd_code_itemsets_target_only.json",
        "cpt_code_itemsets_target_only.json",
        "medical_code_itemsets_target_only.json"
    ]
    
    missing = []
    for filename in expected_files:
        path = fpgrowth_output_dir / filename
        if not path.exists():
            missing.append(str(path))
    
    if missing:
        logger.warning(f"Missing FP-Growth itemsets files:\n" + "\n".join(f"  - {p}" for p in missing))
        logger.info(f"Expected directory: {fpgrowth_output_dir}")
        logger.info(f"Age band format: {age_band_fname}")
```

### 4. Add Progress Logging

**Recommended:** Add detailed progress logging:
```python
logger.info(f"Starting DTW feature creation for {cohort_name} / {age_band}")
logger.info(f"Model data: {model_data_path}")
logger.info(f"FP-Growth directory: {fpgrowth_output_dir}")
logger.info(f"Allowed codes: {len(allowed_codes)}")
logger.info(f"Patients: {len(base_df)} (target: {target_count}, control: {control_count})")
```

---

## Next Steps

1. **Test locally** with one cohort (`opioid_ed / 0-12`)
2. **Verify prerequisites** are met
3. **Check logs** for any warnings or errors
4. **Validate output** format and content
5. **Fix any path resolution issues**
6. **Add better error handling** if needed
7. **Run on EC2** once local test passes

---

## Related Documentation

- `status/WORKFLOW_ACTION_PLAN.md` - Known DTW issues
- `docs/CrossStep_Development/README_target_leakage.md` - Target leakage concerns
- `4b_dtw_filter/DTW_ROLE.md` - DTW's role in pipeline
- `docs/Step5d_DTW/README_dtw_feature_extraction.md` - Feature extraction details
