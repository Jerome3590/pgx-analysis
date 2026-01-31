# Step 4: Model Data

This folder documents the Step 4 pipeline that prepares cohort event data for downstream feature engineering and final modeling.

## Step 4: Model Data (`4_model_data/`)

- **Model Data Extraction** – Creates compact, model-ready `model_events.parquet` datasets for each `(cohort, age_band)` using refined feature importance from Step 3b.
- Writes target and control event data under:
  - `4_model_data/cohort_name={cohort}/age_band={band}/model_events.parquet`

Event-level ICD/administrative code filtering runs earlier in **Step 1b** (`1b_apcd_event_filter`), so Step 4 consumes already-filtered cohort and gold medical/pharmacy data.

## Purpose in the Workflow

Step 4 is the bridge between refined feature importance (Step 3b) and final model training (Step 6):

- Produces a consistent, model-ready unit of analysis (`model_events.parquet`) per (cohort, age_band).
- Uses refined `cohort_feature_importance.csv` from Step 3b (required; no fallback to aggregated importances).
- Output is read by Step 5 (PGx feature engineering) and Step 6 (final model); train/test from Step 6 are uploaded to S3 (required for SHAP and FFA).

## Related Documentation

- `docs/README_analysis_workflow.md` – Full analysis workflow (Steps 1a–9).
- `docs/README_overview.md` – High-level repository and workflow overview.
- `4_model_data/README_model_data.md` – Model data extraction and target vs control logic.
