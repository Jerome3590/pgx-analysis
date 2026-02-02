# ICD and CPT Code Validation: Informative vs Administrative

## Overview

This folder contains tools and processes for validating which ICD and CPT codes in feature importance data are **informative** (should be kept) vs **administrative/non-informative** (should be filtered in Step 4b).

## Purpose

Before filtering codes in Step 4b, we need to:
1. **Analyze codes by groups** (ICD by letter/chapter, CPT by range)
2. **Identify administrative codes** that don't add predictive value
3. **Document findings** for reference and future filtering decisions
4. **Validate filtering decisions** against feature importance data

**Note**: This step only validates and identifies administrative codes. The actual filtering happens in Step 4b (`4b_dtw_filter/filter_protocol_events.py`) at the event level.

## Process

### Step 1: Run Code Group Analysis

Run the analysis script to group codes and identify administrative vs informative codes:

```bash
python analyze_code_groups.py <cohort> <age_band>
```

**Example:**
```bash
python analyze_code_groups.py opioid_ed 13-24
```

**Outputs:**
- Console output showing breakdown by ICD chapter and CPT range
- JSON file: `outputs/{cohort}/{age_band}/code_group_analysis.json`

### Step 2: Review Analysis Results

The analysis provides:
- **ICD codes by letter** (A-Z chapters) with counts of administrative vs informative codes
- **CPT codes by range** (00000-99999) with counts of administrative vs informative codes
- **Classification** for each group: Informative, Mixed, or Administrative
- **Importance statistics** (average and max) for each group

### Step 3: Validate Against Administrative Codes Lookup

Compare analysis results with the administrative codes lookup table:
- **Local Copy:** `administrative_codes_lookup.json` (in this folder)
- **Original Location:** `4b_dtw_filter/administrative_codes_lookup.json`
- **Contains:** Pre-identified administrative ICD and CPT codes

**Note:** A copy of the administrative codes lookup JSON is maintained in this folder for easy reference during validation.

### Step 4: Document Findings

Document findings in:
- **Feature Importance README:** `3a_feature_importance/README.md`
- **Administrative Codes Lookup:** `4b_dtw_filter/administrative_codes_lookup.json`

## Key Findings

### ICD Codes

**All ICD chapters A-Y are 100% informative** - No administrative codes identified in these chapters.

**Only Z chapter contains administrative codes:**
- **Total Z codes:** 353 (for opioid_ed 13-24)
- **Administrative:** 4 codes (Z00.00, Z00.01, Z00.121, Z00.129)
- **Informative:** 349 codes
- **Classification:** Mixed

**Administrative Z Codes:**
- Z00.00 - Encounter for general adult medical examination without abnormal findings
- Z00.01 - Encounter for general adult medical examination with abnormal findings
- Z00.121 - Encounter for routine child health examination with abnormal findings
- Z00.129 - Encounter for routine child health examination without abnormal findings

### CPT Codes

**All CPT ranges 00000-89999 are 100% informative** - No administrative codes identified in these ranges.

**Only 90000-99999 range contains administrative codes:**
- **Total codes in range:** 561 (for opioid_ed 13-24)
- **Administrative:** 1 code (99024-99027 series)
- **Informative:** 560 codes
- **Classification:** Mixed

**Administrative CPT Codes:**
- 99024 - Post-operative follow-up visit
- 99025 - Initial office visit for new patient
- 99026 - Initial office visit for established patient
- 99027 - Office visit for established patient

## Usage

### Interactive Workflow

Use `validate_icd_cpt_codes.py` for interactive validation:

```bash
python validate_icd_cpt_codes.py
```

This will:
1. Load feature importance data for specified cohort/age band
2. Run code group analysis
3. Display results by ICD chapter and CPT range
4. Compare with administrative codes lookup
5. Generate summary report

### Command Line

Run analysis directly:

```bash
python analyze_code_groups.py opioid_ed 13-24
```

## Output Files

- **Analysis JSON:** `outputs/{cohort}/{age_band}/code_group_analysis.json`
  - Contains detailed breakdown by ICD chapter and CPT range
  - Includes counts, importance statistics, and classifications

## Integration with Feature Importance EDA and Step 4b

The results from this validation process inform filtering:
1. **Administrative codes** identified here are added to `administrative_codes_lookup.json`
2. **Feature Importance EDA workflow** loads these codes for reference and validation (does not filter them from feature importances)
3. **BupaR analysis** identifies post-target leakage features to filter from aggregated importances
4. **Step 4b** uses the administrative codes lookup to filter events at the event level
5. **Feature Importance EDA final filtering** only filters post-target leakage features from aggregated feature importance list (not administrative codes)

## Related Files

- **Analysis Script:** `analyze_code_groups.py`
- **Workflow Script:** `validate_icd_cpt_codes.py`
- **Administrative Codes Lookup (Local Copy):** `administrative_codes_lookup.json`
- **Administrative Codes Lookup (Original):** `../../4b_dtw_filter/administrative_codes_lookup.json`
- **Feature Importance README:** `../../3a_feature_importance/README.md`

## Files in This Folder

- `analyze_code_groups.py` - Script to analyze ICD/CPT codes by letter/range groups
- `validate_icd_cpt_codes.py` - Interactive workflow for code validation
- `administrative_codes_lookup.json` - Copy of administrative codes filter (for reference)
- `README_icd_cpt_check.md` - This documentation file
