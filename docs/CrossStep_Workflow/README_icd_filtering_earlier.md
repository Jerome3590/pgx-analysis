# Event Filtering (Step 1b)

## Summary

ICD/administrative code filtering runs in **Step 1b** (`1b_apcd_event_filter`) before cohort creation (Step 2). Step 1b applies aggregated feature importance filtering and administrative code filtering; Step 4 removes target leakage when building model data.

## Pipeline order

1. **Step 1a** – APCD input data (bronze → silver → gold).
2. **Step 1b** – Event filter (aggregated FI + ICD/administrative codes) → filtered events.
3. **Step 2** – Cohort creation (5:1 target:control) using filtered events.
4. **Step 3a** – Feature importance (MC-CV) on cohort data (from filtered events).
5. **Step 3b** – Feature Importance EDA (BupaR, code research) → refined `cohort_feature_importance.csv`.
6. **Step 4** – Model data (`model_events.parquet`) using refined features; removes target leakage for case events.
7. Steps 5–9 – PGx, final model, SHAP, FFA, risk dashboard.

## Why Step 1b runs before cohorts

- **Less data downstream** – Filtering at the event level before cohort assembly reduces the volume of events that flow into cohort creation and feature importance.
- **Single filtered event set** – Cohorts and feature importance (Steps 3a/3b) are built on the same filtered event set.
- **Feature importances** – Feature importance (Step 3a/3b) is computed on events that already exclude administrative codes, so importances reflect predictive signal rather than administrative noise.

## References

- Event filter implementation: `1b_apcd_event_filter/filter_protocol_events.py`
- Lookup: `1b_apcd_event_filter/administrative_codes_lookup.json`
- Workflow notebooks: `1_cohort_workflow.ipynb` (Steps 1–2), `2_feature_importance.ipynb` (Steps 3a–3b)
