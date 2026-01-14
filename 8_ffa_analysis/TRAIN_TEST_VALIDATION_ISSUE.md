# FFA Analysis: Train/Test Validation Issue

## Critical Finding

**The FFA analysis uses training data (2016-2018) for both:**
1. **Model/Rule extraction** (from trained model JSON)
2. **Rule matching and explanation generation** (on training data)

**The test data (2019) is NOT used in FFA analysis.**

## Current Implementation

### Model Source
- **Location**: `6_final_model/outputs/{cohort}/{age_band}/final_model_json/`
- **Source**: Model trained on 2016-2018 data
- **Process**: Model is trained, then exported as JSON for FFA

### Data Source
- **Location**: `6_final_model/outputs/{cohort}/{age_band}/inputs/model_train/final_features.parquet`
- **OR**: `{cohort}_{age_band}_train_final_features_no_leakage.csv`
- **Key indicator**: Filename contains `_train_` - this is **training data (2016-2018)**
- **NOT test data (2019)**

### Code Evidence
```python
# From run_full_ffa_analysis.py line 97-106
DATA_PATH_PARQUET = (
    PROJECT_ROOT
    / "6_final_model"
    / "outputs"
    / COHORT_NAME
    / AGE_BAND_FNAME
    / "inputs"
    / "model_train"  # ← "model_train" directory
    / "final_features.parquet"
)
DATA_PATH_CSV = (
    PROJECT_ROOT
    / "6_final_model"
    / "outputs"
    / COHORT_NAME
    / AGE_BAND_FNAME
    / f"{COHORT_NAME}_{AGE_BAND_FNAME}_train_final_features_no_leakage.csv"  # ← "_train_" in filename
)
```

## Implications

### ✅ **What This Means:**

1. **Rules are learned from training data (2016-2018)**
   - Model is trained on 2016-2018
   - Rules are extracted from this trained model
   - Rules reflect patterns in training data

2. **Explanations are generated on training data (2016-2018)**
   - `explain_dataset()` is called on training data
   - Feature importance is calculated from training data
   - Causal analysis is performed on training data
   - Interaction analysis is performed on training data

3. **No validation on test data (2019)**
   - We don't know if explanations generalize to 2019
   - We don't know if interactions hold on unseen data
   - We don't validate that rules match test instances

### ⚠️ **Limitations:**

1. **No External Validation**
   - Explanations/interactions may not generalize to 2019
   - Rules may be overfit to training data patterns
   - Findings may not hold on future data

2. **Potential Overfitting**
   - If model overfits, rules may reflect noise
   - Explanations may not be generalizable
   - Interactions may be spurious

3. **Temporal Validity Unknown**
   - We don't know if 2016-2018 patterns hold in 2019
   - Drug interactions may change over time
   - Population characteristics may shift

## Why This Might Be Acceptable

### 1. **Model Performance Validation**
- Model is validated on 2019 test set (in Step 6)
- If model generalizes well (high AUC on 2019), rules likely generalize
- **However**: Model performance ≠ explanation validity

### 2. **Explanations Reflect Model Behavior**
- FFA explains what the model learned
- If model generalizes, explanations should generalize
- **However**: Explanations may reflect training-specific patterns

### 3. **Large Dataset**
- State of Virginia population-level data
- Large training set reduces overfitting risk
- **However**: Still no direct validation of explanations

## What Should Be Done

### Option 1: **Validate Explanations on Test Data** (Recommended)

**Add test data validation to FFA analysis:**

```python
# Load test data (2019)
test_data_path = PROJECT_ROOT / "6_final_model" / "outputs" / cohort / age_band / "inputs" / "model_test" / "final_features.parquet"

# Generate explanations on test data
test_explanations = explainer.explain_dataset(X_test, y_test, ...)

# Compare training vs test explanations
# - Rule matching rates
# - Feature importance correlation
# - Interaction consistency
```

**Benefits:**
- Validates that explanations generalize
- Confirms interactions hold on unseen data
- Provides external validation

**Challenges:**
- Requires test data to be available
- May need to modify data loading logic
- Adds computational overhead

### Option 2: **Use Test Data for Explanations** (Alternative)

**Use test data (2019) for explanation generation:**

```python
# Use test data instead of training data
DATA_PATH = test_data_path  # 2019 data

# Model still from training (2016-2018)
# But explanations generated on test (2019)
```

**Benefits:**
- Explanations validated on unseen data
- More robust findings
- Aligns with temporal validation

**Challenges:**
- Test data may not be available in same format
- May need to ensure feature alignment
- Different sample sizes/distributions

### Option 3: **Hybrid Approach** (Best Practice)

**Use both training and test data:**

```python
# Training data: Rule extraction, feature importance baseline
train_explanations = explainer.explain_dataset(X_train, y_train, ...)

# Test data: Validation of explanations
test_explanations = explainer.explain_dataset(X_test, y_test, ...)

# Compare and report:
# - Rule matching consistency
# - Feature importance correlation
# - Interaction stability
```

**Benefits:**
- Best of both worlds
- Validates generalizability
- Provides confidence intervals

## Current Status

**The FFA analysis currently:**
- ✅ Uses training data (2016-2018) for explanations
- ✅ Model is trained on 2016-2018 (correct)
- ⚠️ Does NOT validate explanations on test data (2019)
- ⚠️ Does NOT use test data for explanation generation

**This is a limitation** but may be acceptable if:
1. Model generalizes well (validated in Step 6)
2. Large dataset reduces overfitting risk
3. Explanations are used for understanding, not prediction

## Recommendation

**For maximum robustness:**
1. **Add test data validation** to FFA analysis
2. **Compare training vs test explanations** to assess generalizability
3. **Report both training and test results** in outputs

**For current analysis:**
- **Acknowledge limitation** in documentation
- **Note that model performance validation** (Step 6) provides some assurance
- **Consider explanations as "model behavior on training data"** rather than "universal truths"

## Summary

**Question**: Are rules getting matched over test data?

**Answer**: **NO** - Rules are extracted from training model (2016-2018) and matched against training data (2016-2018). Test data (2019) is NOT used in FFA analysis.

**Implication**: Explanations/interactions are not validated on unseen data, which is a limitation for generalizability claims.

**Mitigation**: Model performance validation (Step 6) provides some assurance, but direct explanation validation would be stronger.
