# Step 4: Model Data

This folder documents the Step 4 pipeline that prepares cohort event data for downstream feature engineering and final modeling.

## Step 4: Model Data (`4_model_data/`)

- **Model Data Extraction** – Creates compact, model-ready `model_events.parquet` datasets for each `(cohort, age_band)` using refined feature importance from Step 3c (final update to features in `2_feature_importance.ipynb`).
- **Target leakage removal (Step 4)**: For case events, keeps only events **strictly before** the target date (`event_date < first_opioid_ed_date` or `first_ed_non_opioid_date`). Events on/after target date are dropped here (linear flow: 3b/3c identify leakage → Step 4 removes it).
- Writes target and control event data under:
  - `4_model_data/cohort_name={cohort}/age_band={band}/model_events.parquet`

Event-level ICD/administrative code filtering runs in **Step 1b** (`1b_apcd_event_filter`). Step 4 removes target leakage for case events when building model data.

## Purpose in the Workflow

Step 4 is the bridge between refined feature importance (Step 3c) and final model training (Step 6):

- Produces a consistent, model-ready unit of analysis (`model_events.parquet`) per (cohort, age_band).
- Uses refined `cohort_feature_importance.csv` from Step 3c (required; no fallback to aggregated importances).
- Output is read by Step 5 (PGx feature engineering) and Step 6 (final model); train/test from Step 6 are uploaded to S3 (required for SHAP and FFA).

## Related Documentation

- `docs/README_analysis_workflow.md` – Full analysis workflow (Steps 1a–9).
- `docs/README_overview.md` – High-level repository and workflow overview.
- `4_model_data/README_model_data.md` – Model data extraction and target vs control logic.
