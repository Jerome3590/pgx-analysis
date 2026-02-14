# FP-Growth Target Leakage Analysis

**Date:** 2026-01-03  
**Status:** Confirmed - FP-Growth features cause target leakage

---

## Confirmation: Your Concern is Accurate

**Yes, your concern is 100% accurate.** FP-Growth features introduce target leakage through multiple mechanisms.

---

## Mechanism 1: Direct Target Code Leakage (Most Critical)

### The Problem

1. **Target codes are included in transactions**: When `item_type == 'icd_code'`, FP-Growth includes ALL ICD codes from diagnosis columns, including target codes like `F1120`.

2. **Association rules can have target codes as consequents**: Rules can be:
   - `{DRUG_A, DRUG_B} → {F1120}` (drugs predict target code)
   - `{ICD_X, ICD_Y} → {F1120}` (other ICDs predict target code)

3. **Feature creation checks for BOTH antecedents AND consequents**:
   ```python
   def match_rule(patient_items: Set[str], antecedents: List[str], consequents: List[str]) -> bool:
       return antecedents_set.issubset(patient_items) and consequents_set.issubset(patient_items)
   ```

4. **Result**: Features like `icd_code_rule_0_match` will be `1` only if the patient has:
   - The antecedents (e.g., `DRUG_A, DRUG_B`)
   - **AND the consequent (e.g., `F1120` - the target code itself!)**

### Example of Leakage

**Rule discovered from combined data:**
- `{HYDROCODONE, TRAMADOL} → {F1120}` (confidence: 0.85)

**Feature created:**
- `icd_code_rule_0_match`: Binary indicator if patient has `HYDROCODONE`, `TRAMADOL`, **AND `F1120`**

**Problem**: This feature directly encodes the target! A patient with `F1120` will have this feature = 1, which is perfect target leakage.

---

## Mechanism 2: Target-Only Pattern Mining

### The Problem

The code runs **target-only FP-Growth** (lines 756-838 in `cohort_fpgrowth.py`):

```python
# Target-only FP-Growth: within-target patterns (target == 1)
df_target = df[df["target"] == 1].copy()
# ... mine patterns from target patients only ...
```

**If these target-only patterns are used as features**, this is clear leakage because:
- Patterns are discovered ONLY from target patients
- These patterns are then applied to all patients (target + control)
- The patterns implicitly encode target-specific information

---

## Mechanism 3: Indirect Target Information Encoding

### The Problem

Even if rules don't explicitly include target codes, patterns mined from combined data can encode target information:

1. **Patterns more common in targets**: Itemsets that are frequent in target patients but rare in controls
2. **Rule confidence reflects target distribution**: High-confidence rules may reflect target class imbalance
3. **Lift metrics encode target relationships**: High lift indicates strong association with target class

### Example

**Itemset discovered from combined data:**
- `{DRUG_A, DRUG_B, ICD_X}` appears in 60% of target patients but only 5% of controls

**Feature created:**
- `drug_name_itemset_0_match`: Binary indicator if patient has `{DRUG_A, DRUG_B, ICD_X}`

**Problem**: While not direct leakage, this feature encodes target information because:
- The itemset was selected based on its frequency in the combined dataset
- The combined dataset includes target labels, so the selection process "saw" the target
- The feature implicitly captures target-specific patterns

---

## Why This Matters

### Impact on Model Performance

1. **Artificially inflated performance**: Models with FP-Growth features may show high accuracy, but this is due to leakage, not genuine predictive power.

2. **Poor generalization**: Models trained with leaked features won't generalize to new data where target labels aren't available.

3. **Misleading feature importance**: FP-Growth features may rank highly in importance, but this reflects leakage, not true predictive signal.

### Impact on Clinical Validity

1. **False causal claims**: Features that encode target information can't be used to make causal claims about what predicts the target.

2. **Circular reasoning**: "Patients with F1120 are more likely to have F1120" is not a useful finding.

3. **Invalid risk assessment**: Risk predictions based on leaked features are not valid for clinical decision-making.

---

## Solution: Visualization Only

FP-Growth should be used **only for visualization and exploratory analysis**, not as model features:

### ✅ Safe Uses

1. **Network visualizations**: Show co-occurrence patterns for clinical review
2. **Exploratory analysis**: Understand which items frequently appear together
3. **Hypothesis generation**: Identify patterns of interest for further investigation
4. **Risk dashboard integration**: Visualize patterns alongside causal analysis (FFA/SHAP)

### ❌ Unsafe Uses

1. **Model features**: Do NOT use FP-Growth features in predictive models
2. **Feature engineering**: Do NOT create features from FP-Growth patterns
3. **Direct prediction**: Do NOT use FP-Growth patterns to predict target

---

## Code Evidence

### Direct Target Code Leakage

**File**: `9_dashboard_visuals/fpgrowth/create_fpgrowth_features.py`

**Line 158-162**: Rule matching includes consequents (which can be target codes)
```python
def match_rule(patient_items: Set[str], antecedents: List[str], consequents: List[str]) -> bool:
    """Check if patient matches an association rule (has antecedents AND consequents)."""
    antecedents_set = set(antecedents)
    consequents_set = set(consequents)
    return antecedents_set.issubset(patient_items) and consequents_set.issubset(patient_items)
```

**Line 269-273**: Features check for both antecedents and consequents
```python
features_df[f'{item_type}_rule_{idx}_match'] = (
    patient_transactions['items_set'].apply(
        lambda x, antecedents=antecedents, consequents=consequents: match_rule(x, antecedents, consequents)
    ).astype(int)
)
```

### Target Codes in Transactions

**File**: `9_dashboard_visuals/fpgrowth/create_fpgrowth_features.py`

**Line 86-101**: ICD code extraction includes ALL ICD codes (including target codes)
```python
elif item_type == 'icd_code':
    item_query = """
    WITH all_icds AS (
        SELECT mi_person_key, primary_icd_diagnosis_code as icd FROM read_parquet('{path}')
        UNION ALL
        SELECT mi_person_key, two_icd_diagnosis_code as icd FROM read_parquet('{path}')
        # ... all ICD positions ...
    )
    SELECT mi_person_key, icd as item FROM all_icds
    WHERE icd IS NOT NULL AND icd != ''
"""
```

**Result**: Target codes like `F1120` are included in transactions and can appear in rules as consequents.

---

## Conclusion

**Your concern is completely accurate.** FP-Growth features introduce target leakage through:

1. ✅ **Direct leakage**: Rule features check for target codes as consequents
2. ✅ **Target-only patterns**: Patterns mined from target-only data
3. ✅ **Indirect encoding**: Patterns that encode target information from combined data

**Solution**: Exclude FP-Growth features from the final model. Use FP-Growth only for visualization and exploratory analysis.

---

**Last Updated:** 2026-01-03
