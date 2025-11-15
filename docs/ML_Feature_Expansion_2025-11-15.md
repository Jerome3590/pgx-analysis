# ML Feature Expansion - All Diagnosis & Procedure Codes - 2025-11-15

## Summary

Expanded cohort data to include **ALL ICD diagnosis codes (10 columns)** and **ALL ICD procedure codes (9 columns)** as features for machine learning analysis. This enables FPGrowth and CatBoost to discover predictive patterns without pre-defining which codes are "opioid-related."

## Problem Statement

### Before This Update
- Only `primary_icd_diagnosis_code` was included as a feature
- Secondary/tertiary diagnosis codes (positions 2-10) were **missing**
- ALL ICD procedure codes were **missing**
- Only pre-defined "opioid-related" codes could be analyzed for prediction

### Impact
- FPGrowth could not discover patterns in secondary diagnosis codes
- CatBoost could not use procedure codes as features
- Potentially missing important predictive signals in non-primary diagnosis positions
- Required manual curation of "important" procedure codes

## Solution: Data-Driven Approach

Instead of pre-defining which procedure codes are "opioid-related," we now include **ALL** diagnosis and procedure codes as features and let the ML algorithms discover which ones are predictive.

### Added Features

**ICD Diagnosis Codes (10 columns):**
- `primary_icd_diagnosis_code` ✅ (already included)
- `two_icd_diagnosis_code` ✨ NEW
- `three_icd_diagnosis_code` ✨ NEW
- `four_icd_diagnosis_code` ✨ NEW
- `five_icd_diagnosis_code` ✨ NEW
- `six_icd_diagnosis_code` ✨ NEW
- `seven_icd_diagnosis_code` ✨ NEW
- `eight_icd_diagnosis_code` ✨ NEW
- `nine_icd_diagnosis_code` ✨ NEW
- `ten_icd_diagnosis_code` ✨ NEW

**ICD Procedure Codes (9 columns):**
- `two_icd_procedure_code` ✨ NEW
- `three_icd_procedure_code` ✨ NEW
- `four_icd_procedure_code` ✨ NEW
- `five_icd_procedure_code` ✨ NEW
- `six_icd_procedure_code` ✨ NEW
- `seven_icd_procedure_code` ✨ NEW
- `eight_icd_procedure_code` ✨ NEW
- `nine_icd_procedure_code` ✨ NEW
- `ten_icd_procedure_code` ✨ NEW

**Note:** There is no `primary_icd_procedure_code` in the source data - ICD procedure codes start with "two".

### Already Included (unchanged)
- `procedure_code` (CPT code)
- `cpt_mod_1_code`, `cpt_mod_2_code`
- `drug_name`, `therapeutic_class_1`
- Demographic fields
- HCG fields

## Files Updated

### 1. `2_create_cohort/phases/phase1_data_preparation.py`
- **Lines 52-72**: Load all 10 ICD diagnosis codes and 9 ICD procedure codes from gold medical data
- Adds clear comments: "ALL ICD diagnosis codes (for ML feature discovery)"

### 2. `2_create_cohort/phases/phase2_event_processing.py`
- **Lines 127-147**: Include all diagnosis and procedure codes in medical events
- **Lines 178-198**: Set all diagnosis and procedure codes to NULL for pharmacy events (as expected)
- Maintains proper column alignment between UNION ALL sections

### 3. `2_create_cohort/phases/common.py`
- **Lines 150-170**: Include all codes in checkpoint recovery view creation (medical_base)
- **Lines 322-342**: Include all codes in unified_event_fact_table medical section
- **Lines 371-391**: Set all codes to NULL in unified_event_fact_table pharmacy section
- Ensures consistent schema for checkpoint recovery

## Benefits

### 1. **ML Pattern Discovery**
- FPGrowth can now discover frequent patterns across ALL diagnosis and procedure codes
- CatBoost can use secondary diagnoses and procedures as features for prediction
- No need to pre-define which codes are "important"

### 2. **Comprehensive Analysis**
- Capture subtle patterns (e.g., specific procedure + secondary diagnosis combinations)
- Identify unexpected predictors that manual curation might miss
- Better feature space for model training

### 3. **Research Flexibility**
- Researchers can analyze any diagnosis/procedure code post-hoc
- Support for questions like "What procedures are associated with opioid events?"
- No need to re-run cohort creation to add specific codes

### 4. **Completeness**
- No missed diagnosis codes due to position
- All procedure codes available for analysis
- Prevents data loss from incomplete feature sets

## Example Use Cases

### FPGrowth Pattern Mining
```python
# Now can discover patterns like:
# {primary_icd: 'J441', three_icd_diagnosis_code: 'F1120', 
#  five_icd_procedure_code: '3E033GC'} → opioid_ed (support: 0.15)
```

### CatBoost Feature Importance
```python
# Can identify important features like:
# - two_icd_diagnosis_code: importance 0.23
# - seven_icd_procedure_code: importance 0.18
# - drug_name: importance 0.31
```

### Research Questions Enabled
- "What are the most common secondary diagnoses for opioid ED patients?"
- "Which procedures (inpatient) are associated with opioid use disorder?"
- "Do certain procedure+diagnosis combinations predict opioid events?"

## Implementation Notes

### Data Schema
- Medical events: All 19 code columns populated from source data
- Pharmacy events: All code columns set to NULL (pharmacy has no diagnosis/procedure codes)
- Column order maintained across UNION ALL for proper data alignment

### Performance Considerations
- Additional columns increase parquet file size slightly (~15-20%)
- DuckDB efficiently handles sparse data (many NULLs in procedure codes)
- No significant query performance impact (columns only used when needed)

### Backward Compatibility
- Existing analyses using `primary_icd_diagnosis_code` continue to work
- New columns are optional - can be ignored if not needed
- No breaking changes to existing ML pipelines

## Migration Notes

### Reprocessing Required

To benefit from this update, regenerate cohorts:

```bash
# Delete old cohorts and checkpoints (already done)
# Regenerate with new features
python 2_create_cohort/create_cohort.py \
  --all-age-bands \
  --all-event-years \
  --cohorts both
```

### Expected Changes After Regeneration
- Parquet files will be larger (~15-20% increase)
- New columns will be available in cohort data
- FPGrowth will discover more patterns
- CatBoost will have more features to evaluate

### FPGrowth/CatBoost Updates
No code changes needed! The algorithms will automatically:
- FPGrowth: Include new diagnosis/procedure codes in pattern mining
- CatBoost: Consider new columns as potential features
- Feature importance: Evaluate predictive power of all codes

## Related Updates

This update complements the earlier fix:
- **ICD Diagnosis Column Update (2025-11-15)**: Fixed opioid identification to check ALL 10 ICD diagnosis columns
- **This Update (ML Feature Expansion)**: Made all diagnosis and procedure codes available as ML features

Together, these ensure:
1. ✅ Complete opioid identification (all 10 diagnosis columns checked)
2. ✅ Complete feature set for ML (all diagnosis + procedure codes available)

## Technical Details

### Column Naming Convention
- ICD diagnosis: `primary_icd_diagnosis_code`, `two_icd_diagnosis_code`, ..., `ten_icd_diagnosis_code`
- ICD procedure: `two_icd_procedure_code`, ..., `ten_icd_procedure_code` (no "primary")
- CPT: `procedure_code` (single column)

### Sparse Data Handling
- Most events have 1-3 diagnosis codes (rest are NULL)
- ICD procedure codes are even sparser (mostly NULL)
- DuckDB/Parquet efficiently compress NULL-heavy columns

### Column Order in Parquet
```
mi_person_key, event_date, event_type, ...,
primary_icd_diagnosis_code,
two_icd_diagnosis_code,
three_icd_diagnosis_code,
...,
ten_icd_diagnosis_code,
two_icd_procedure_code,
...,
ten_icd_procedure_code,
drug_name, therapeutic_class_1,
procedure_code, cpt_mod_1_code, cpt_mod_2_code,
...
```

## References

- **Original Issue**: User question: "Are there procedures we could be missing due to not attached to an ICD Diagnosis code?"
- **Solution**: Data-driven approach - include ALL codes, let ML discover patterns
- **Files Changed**: 3 files (phase1, phase2, common)
- **Commit**: `cca2c8b` - "feat: Add ALL ICD diagnosis and procedure codes as ML features"

## Related Documentation

- `docs/ICD_Diagnosis_Column_Update_2025-11-15.md` - Opioid identification across all 10 columns
- `docs/Cohort_Creation_SQL.md` - SQL reference for cohort creation
- `docs/Create_Cohort_README.md` - Pipeline guide

---

**Last Updated**: 2025-11-15  
**Status**: Complete  
**Author**: PGx Analytics Engineering Team

