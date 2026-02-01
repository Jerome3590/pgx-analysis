# Step 1b: Event Filtering (ICD / Administrative Codes)

## Overview

Step 4b filters events at the event level to remove administrative codes and post-event leakage, creating `model_events_no_protocols.parquet` for downstream feature engineering.

## Purpose

Filter events in two passes to reduce feature count and improve Step 3a feature importance on a second pass:

1. **Aggregated feature importance (first pass)**: Keep only events whose codes (ICD, CPT, drug) appear in the aggregated feature-importance CSV from Step 3/3a. This reduces the number of features before final cohorts.
2. **Administrative codes and post-event leakage (second pass)**: Remove administrative codes and events occurring on or after target event date.

## Workflow

### Input
- `model_events.parquet` from Step 4a (created using refined features from Feature Importance EDA)
- Aggregated feature importance CSV from Step 3 or 3a: `{cohort}_{age_band}_aggregated_feature_importance.csv` (must exist; script validates and cleans it before use)

### Filtering Logic
1. **Aggregated feature-importance filter (first)**: Keep only events where at least one of (drug_name, ICD diagnosis columns, procedure_code) is in the allowed set from the aggregated FI CSV. Events that do not match any important feature are dropped. Step 3a will run feature importance again on this reduced set for greater accuracy.

2. **Administrative code filtering**: Remove events with codes listed in research outputs and `administrative_codes_lookup.json`
   - Codes are identified in Step 3b `0_icd_cpt_check` through code research and validation
   - Lookup table: `1b_apcd_event_filter/administrative_codes_lookup.json`

3. **Target leakage**: Not applied here. Events on or after target date are removed in **Step 4** (model data) after 3b identifies leakage (linear flow: 3b → 4).

4. **Code classification**: Events are classified as administrative vs. medical/pharmacy
   - Administrative: Billing, scheduling, documentation codes
   - Medical/Pharmacy: Clinical diagnoses, procedures, medications

### Output
- `model_events_no_protocols.parquet` - Filtered event data used by downstream steps (Step 5: PGx Feature Engineering)

## Usage

```bash
python 1b_apcd_event_filter/filter_protocol_events.py \
    --cohort-name opioid_ed \
    --age-band 13-24
```

## Integration with Workflow

### Step 3b: Feature Importance EDA
- **0_icd_cpt_check**: Identifies administrative codes → `administrative_codes_lookup.json`
- **1_bupaR**: Identifies post-target leakage → **Step 4** removes those events when building model data

### Step 4: Model Data Creation
- Creates `model_events.parquet` using refined features from Feature Importance EDA

### Step 1b: Event Filtering (This Step – runs before cohorts for efficiency)
- Filters `model_events.parquet` → `model_events_no_protocols.parquet`
- Uses codes from Step 3b

### Step 5: PGx Feature Engineering
- Uses `model_events_no_protocols.parquet` as input (preferred over `model_events.parquet`)

## Files

- `filter_protocol_events.py` - Main filtering script
- `administrative_codes_lookup.json` - Lookup table for administrative codes (from Step 3b)
- `README_administrative_codes_lookup.md` - Documentation for administrative codes lookup
- `README_code_classification.md` - Methodology for code classification

## Related Documentation

- Step 3b: `3b_feature_importance_eda/` - Feature refinement and code identification
- Step 4: `4_model_data/` - Model data creation
- Step 5: `5_pgx_analysis/` - PGx feature engineering
