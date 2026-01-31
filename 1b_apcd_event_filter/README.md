# Step 1b: Event Filtering (ICD / Administrative Codes)

## Overview

Step 4b filters events at the event level to remove administrative codes and post-event leakage, creating `model_events_no_protocols.parquet` for downstream feature engineering.

## Purpose

Filter events to remove:
1. **Administrative codes** - Identified in Step 3b (`0_icd_cpt_check`) and stored in `administrative_codes_lookup.json`
2. **Post-event leakage** - Events occurring after target event date, identified in Step 3b (`1_bupaR` post-target analysis)

## Workflow

### Input
- `model_events.parquet` from Step 4a (created using refined features from Feature Importance EDA)

### Filtering Logic
1. **Administrative Code Filtering**: Remove events with codes listed in `administrative_codes_lookup.json`
   - Codes are identified in Step 3b `0_icd_cpt_check` through code research and validation
   - Lookup table: `1b_apcd_event_filter/administrative_codes_lookup.json`

2. **Post-Event Leakage Filtering**: Remove events occurring on or after target event date
   - Target event date identified in Step 3b `1_bupaR` post-target analysis
   - Prevents target leakage by removing events that occur after the outcome

3. **Code Classification**: Events are classified as administrative vs. medical/pharmacy
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
- **1_bupaR**: Identifies post-target leakage features → Used to filter events after target date

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
