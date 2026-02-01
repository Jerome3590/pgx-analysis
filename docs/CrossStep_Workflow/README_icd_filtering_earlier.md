# ICD Filtering Moved Earlier (Step 1b)

## Summary

ICD/administrative code filtering has been **moved earlier** in the pipeline: it now runs in **Step 1b** (`1b_apcd_event_filter`) **before** cohort creation (Step 2), instead of as a separate post–model-data step.

## Why This Is More Efficient

1. **Less data downstream** – Filtering at the event level before cohort assembly reduces the volume of events that flow into cohort creation and feature importance. Fewer administrative/scheduling codes are carried through the pipeline.
2. **Single filtered event set** – Cohorts and feature importance (Steps 3a/3b) are built on the **same** filtered event set. There is no mismatch between “raw” events in cohorts and “filtered” events in later steps.
3. **True feature importances** – Feature importance (Step 3a/3b) is computed on events that already exclude administrative codes. Importances therefore reflect **predictive** signal (e.g. clinical ICD/CPT and drugs) rather than administrative noise, giving a more accurate ranking of features for model data (Step 4) and final model (Step 6).

## Verification

- **Efficiency:** Filtering once in 1b avoids duplicate filtering logic and reduces data volume before expensive cohort and MC-CV steps.
- **Correctness:** Cohorts (Step 2) and feature importance (3a/3b) both consume the filtered event set produced by 1b; model data (Step 4) applies **target leakage removal** (events on/after target date) and final model (Step 6) use the same refined feature set (linear flow: 3b → 4).
- **Rerun requirement:** After moving ICD filtering earlier, **cohorts must be rebuilt** (Step 2) and **feature importances must be rerun** (Steps 3a and 3b) so that all downstream steps use the new, filtered event set and consistent feature lists.

## Pipeline Order (Current)

1. **Step 1a** – APCD input data (bronze → silver → gold).
2. **Step 1b** – Event filter (ICD/administrative codes) → filtered events.
3. **Step 2** – Cohort creation (5:1 target:control) using filtered events.
4. **Step 3a** – Feature importance (MC-CV) on cohort data (from filtered events).
5. **Step 3b** – Feature Importance EDA (BupaR, code research) → refined `cohort_feature_importance.csv`.
6. **Step 4** – Model data (`model_events.parquet`) using refined features.
7. Steps 5–9 – PGx, final model, SHAP, FFA, risk dashboard.

## References

- Event filter implementation: `1b_apcd_event_filter/filter_protocol_events.py`
- Lookup: `1b_apcd_event_filter/administrative_codes_lookup.json`
- Workflow notebooks: `1_cohort_workflow.ipynb` (Steps 1–2), `2_feature_importance.ipynb` (Steps 3a–3b)
