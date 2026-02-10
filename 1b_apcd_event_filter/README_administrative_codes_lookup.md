# Administrative Codes Lookup Table

## Overview

The `administrative_codes_lookup.json` file is a manually maintained lookup table that classifies codes (ICD, CPT, drug) as **administrative** vs. **medical/pharmacy**. Codes listed in this lookup table will be filtered out during event filtering (Step 4b). Codes are identified in Step 3b (`0_icd_cpt_check`).

## Purpose

The DTW filter now uses **code-based classification** instead of time intervals to identify administrative events. This approach:

1. **Filters administrative events** (billing, scheduling, post-event documentation) regardless of time intervals
2. **Keeps medical/pharmacy events** even if they occur close together (e.g., same day)
3. **Prevents filtering based solely on proximity** - two clinical events on the same day are both kept

## File Structure

```json
{
  "description": "Lookup table for classifying codes as administrative vs. medical/pharmacy",
  "version": "1.0",
  "last_updated": "2024-01-01",
  "administrative_codes": {
    "icd": ["Z00.00", "Z00.01", ...],
    "cpt": ["99024", "99025", ...],
    "drug": []
  }
}
```

## How to Build the Lookup Table

### Step 1: Run Research Analysis

First, run the event filter with research outputs enabled:

```bash
python 4b_event_filter/filter_protocol_events.py \
    --cohort-name opioid_ed \
    --age-band 0-12 \
    --min-interval-days 1
```

This generates research outputs in `4b_event_filter/outputs/for_review/{cohort}/{age_band}/`:
- `code_analysis_protocol_vs_clinical_{cohort}_{age_band}.csv` - Code-level analysis

### Step 2: Review Code Analysis

Open the `code_analysis_protocol_vs_clinical_*.csv` file and review codes with high `protocol_pct` (>80%). These codes frequently appear in protocol-like sequences (events < 1 day apart).

**Key Questions:**
- Is this code **administrative** (billing, scheduling, documentation)?
- Is this code **clinical** (diagnosis, procedure, medication)?
- Does this code represent **post-event documentation** (leakage)?

### Step 3: Manually Classify Codes

For each code with high `protocol_pct`, determine if it's administrative:

**Administrative Codes:**
- Billing codes (e.g., CPT 99024 - Post-operative follow-up)
- Scheduling codes (e.g., appointment scheduling procedures)
- Administrative documentation (e.g., ICD Z00.00 - General health check)
- Post-event documentation (events after target event date are automatically filtered)

**Clinical Codes (Keep):**
- Diagnoses (ICD codes for medical conditions)
- Procedures (CPT codes for medical procedures)
- Medications (drug names for prescriptions)

### Step 4: Update Lookup Table

Add confirmed administrative codes to `administrative_codes_lookup.json`:

```json
{
  "administrative_codes": {
    "icd": ["Z00.00", "Z00.01", "Z00.121"],
    "cpt": ["99024", "99025", "99026"],
    "drug": []
  }
}
```

**Note:** Most drugs are clinical; the `drug` array is used for values that are not drugs or are excluded from model training (see below).

### Drug name exclusions (model training)

The following values are excluded from the **drug name column** for model training. They are listed in `administrative_codes.drug` and in `py_helpers.constants.DRUG_NAMES_EXCLUDED_MODEL_TRAINING`:

| Value    | Reason |
|----------|--------|
| **Narcan**   | Excluded per model-training requirements. |
| **Unknown**  | Placeholder, not a drug. |
| **Fentanyl** | Excluded per model-training requirements. |
| **1036F**    | Not a drug. CPT Category II tracking code used to document that a patient (18+) is a current tobacco non-user, typically during preventive screenings; part of quality measures for tobacco use assessment and preventive care. |
| **T401XA1**  | Not a drug. ICD-10-CM diagnosis code for *Poisoning by 4-aminophenol derivatives, accidental (unintentional), initial encounter* — in practice usually unintentional overdose or poisoning with acetaminophen (paracetamol) or closely related compounds, at the patient’s initial encounter. |

These are filtered in Step 1b (event filter), Step 3a (aggregated feature importance), Step 3b (`filter_and_refine_features.py`), Step 4 (`get_important_items` in `create_model_data.py`), and Step 6 (`build_final_cohort_model_features.py`).

### Step 5: Re-run Filtering

After updating the lookup table, re-run the filter:

```bash
python 4b_event_filter/filter_protocol_events.py \
    --cohort-name opioid_ed \
    --age-band 0-12 \
    --use-lookup-table
```

## Automatic Filtering

The filter automatically classifies the following as administrative (no lookup table needed):

1. **Post-event leakage**: Events occurring on or after the target event date
   - For `opioid_ed` cohort: events >= `first_opioid_ed_date`
   - For `non_opioid_ed` cohort: events >= `first_ed_non_opioid_date`

## Using Research Outputs Instead

If you prefer to use research outputs instead of a manual lookup table:

```bash
python 4b_event_filter/filter_protocol_events.py \
    --cohort-name opioid_ed \
    --age-band 0-12 \
    --use-lookup-table false \
    --admin-code-threshold-pct 80.0
```

This will automatically classify codes with `protocol_pct >= 80.0` as administrative based on the research outputs.

## Best Practices

1. **Start with research**: Always run research analysis first to identify candidate codes
2. **Manual review**: Don't rely solely on `protocol_pct` - manually review each code
3. **Conservative approach**: When in doubt, keep the code (don't filter it)
4. **Version control**: Track changes to the lookup table with version numbers
5. **Document decisions**: Add notes to the lookup table explaining why codes are classified as administrative

## Example Workflow

```bash
# 1. Run research analysis
python 4b_event_filter/filter_protocol_events.py \
    --cohort-name opioid_ed \
    --age-band 0-12 \
    --min-interval-days 1

# 2. Review research outputs
# Open: 4b_event_filter/outputs/for_review/opioid_ed/0_12/code_analysis_protocol_vs_clinical_*.csv

# 3. Update lookup table
# Edit: 4b_event_filter/administrative_codes_lookup.json

# 4. Re-run with lookup table
python 4b_event_filter/filter_protocol_events.py \
    --cohort-name opioid_ed \
    --age-band 0-12 \
    --use-lookup-table
```

## Notes

- **Time intervals**: The `--min-interval-days` parameter (default: 1 day) is now only used for **research analysis**, not for filtering. Filtering is based on code classification.
- **BupaR alignment**: The 1-day interval matches BupaR's time window analysis for consistency.
- **Post-event leakage**: Events after the target event date are automatically filtered regardless of the lookup table.
