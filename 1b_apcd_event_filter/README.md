# Step 1b: Event Filtering (ICD / Administrative Codes)

## Overview

The event filter runs **after Step 1a (APCD input) but before Step 2 (create cohort)**. It removes administrative codes from gold medical/pharmacy so cohort creation uses filtered event data.

**Two feature importances:** (1) **Baseline FI** — precomputed from an initial run; does not need to be recomputed. (2) **Updated FI** — second pass after event filtering for greater accuracy.

- **Before cohorts (recommended):** `--before-cohorts` — reads gold medical/pharmacy, removes admin codes, and optionally keeps only events whose codes appear in **baseline** aggregated FI (precomputed) to reduce cohort processing. Writes `gold/medical_filtered/` and `gold/pharmacy_filtered/`. Step 2 uses filtered gold when present.
- **After cohorts (optional):** Default mode reads cohort parquets (Step 2), applies **baseline** aggregated FI + administrative codes, writes `model_events_no_protocols.parquet`. Step 3a can then run **updated** FI (second pass) on this.

## Purpose

1. **Before cohorts:** Remove administrative codes from gold medical/pharmacy so Step 2 (create cohort) is based on filtered event data. No feature importance required.
2. **After cohorts (optional):** Filter cohort events by aggregated feature importance and administrative codes for downstream model data.

## Workflow

### Before cohorts (run after Step 1a, before Step 2)

- **Input:** Gold medical/pharmacy from Step 1a: `gold/{medical|pharmacy}/age_band=.../event_year=.../`
- **Output:** `gold/medical_filtered/` and `gold/pharmacy_filtered/` (same layout). Step 2 prefers filtered when present.
- **Filter:** Administrative codes only (`administrative_codes_lookup.json`). No aggregated FI.

### After cohorts (optional)

- **Input:** Cohort parquets (Step 2); aggregated feature importance CSV (Step 3/3a)

### Filtering Logic
1. **Baseline aggregated feature-importance filter (first)**: Keep only events where at least one of (drug_name, ICD diagnosis columns, procedure_code) is in the allowed set from the **baseline** aggregated FI CSV (precomputed; does not need to be recomputed). Events that do not match any important feature are dropped. Step 3a then runs **updated** FI (second pass) on this reduced set for greater accuracy.

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

### Before cohorts (after Step 1a, before Step 2)

Run for each (age_band, event_year) after APCD input (Step 1a) produces gold medical/pharmacy:

```bash
# Admin only
python 1b_apcd_event_filter/filter_protocol_events.py --before-cohorts --age-band 13-24 --event-year 2016

# With baseline FI (precomputed) to reduce cohort processing — auto-resolve from cohort/age_band
python 1b_apcd_event_filter/filter_protocol_events.py --before-cohorts --age-band 13-24 --event-year 2016 --cohort-name opioid_ed

# Or pass baseline FI CSV path explicitly
python 1b_apcd_event_filter/filter_protocol_events.py --before-cohorts --age-band 13-24 --event-year 2016 --aggregated-fi-csv path/to/opioid_ed_13_24_aggregated_feature_importance.csv
```

Then run Step 2 (create cohort). Step 2 will use filtered gold when present (`gold/medical_filtered/`, `gold/pharmacy_filtered/`).

To cover all partitions, loop over age bands and event years (e.g. 2016–2019) and call the script for each.

### After cohorts (optional)

If you also run the event filter on cohort data (FI + admin), use:

```bash
python 1b_apcd_event_filter/filter_protocol_events.py --cohort-name opioid_ed --age-band 13-24
```

Requires **baseline** aggregated feature importance (precomputed from Step 3a with `--baseline` once; does not need to be recomputed) and cohort parquets (Step 2).

## Integration with Workflow

### Step 3b: Feature Importance EDA
- **0_icd_cpt_check**: Identifies administrative codes → `administrative_codes_lookup.json`
- **1_bupaR**: Identifies post-target leakage → **Step 4** removes those events when building model data

### Step 1b: Event Filtering (This Step)
- Reads **cohort parquets** from Step 2 (create cohort); multiple event years are unioned.
- Filters cohort events → `model_events_no_protocols.parquet` (written to same layout as Step 4 output).
- Uses codes from Step 3b. No dependency on Step 4 (model data).

### Step 4: Model Data Creation
- Creates `model_events.parquet` using refined features from Feature Importance EDA (downstream of 1b).

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
