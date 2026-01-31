## Model Data (Step 4) – Event-Level Inputs for Feature Engineering and Final Models

This directory contains event-level `model_events.parquet` files used as inputs to:

- Step 5a–5d feature engineering (FP-Growth, BupaR, DTW, PGx)
- Step 6 final models (RandomForest, XGBoost, CatBoost)

Each file corresponds to a specific `(cohort_name, age_band)` cell:

- `4a_model_data/cohort_name={cohort}/age_band={age_band}/model_events.parquet`

### Label Definition and Case/Control Construction

- **Positive class (`target = 1`) – cases**
  - Source: `gold_cohorts/cohort_name={cohort}/event_year={year}/age_band={age_band}/cohort.parquet`
  - Per cohort and age band, all **case patients** are identified by `is_target_case = 1`.
  - For these patients, **only events whose item-bearing fields appear in the aggregated feature-importance list** are kept:
    - `drug_name`
    - all ICD diagnosis columns (`primary_icd_diagnosis_code` … `ten_icd_diagnosis_code`)
    - `procedure_code`
  - Event-level label: `target = 1` (derived from `is_target_case = 1`).

- **Negative class (`target = 0`) – controls**
  - Controls are defined **within the same cohort and age band**, but their events are drawn from the **gold medical / pharmacy event tables**, not from an “opposite” cohort:
    - `gold/medical/age_band={age_band}/event_year={year}/medical_data*.parquet`
    - `gold/pharmacy/age_band={age_band}/event_year={year}/pharmacy_data.parquet`
  - Control patients (`target = 0`) must satisfy:
    - **No target ICD codes** (e.g., `F1120` and other configured target codes) in *any* diagnosis column across their medical events.
    - Their `mi_person_key` **does not appear in the case set** for the same `(cohort, age_band, year)` (i.e., they are not a target patient in the cohort tables).
  - For selected control patients, **all available events** (medical + pharmacy) are retained:
    - controls are *not* filtered by feature importance.
  - Event-level label: `target = 0`.

This asymmetry is intentional and standard for classification problems:

- **Cases**: tightly defined and optionally event-filtered by feature importance.
- **Controls**: broad, label-clean background population, with full event histories preserved.

### Feature-Importance Filtering

Refined feature-importance CSVs (from Step 3b) drive the case-side event filtering:

- Inputs (REQUIRED - Step 3b must run before Step 4a):
  - `3b_feature_importance_eda/outputs/{cohort}/{age_band}/{cohort}_{age_band}_cohort_feature_importance.csv`
  - or downloaded from S3: `s3://pgxdatalake/gold/feature_importance/{cohort}/{age_band}/{cohort}_{age_band}_cohort_feature_importance.csv`
- For each `(cohort_name, age_band)`:
  - The `feature` column is parsed; `item_` prefixes are stripped to get raw item codes.
  - The resulting item list is used in the DuckDB query that filters **case events**:
    - keep only events where `drug_name`, any ICD diagnosis column, or `procedure_code` matches one of the important items.
  - Control events are never filtered by this list; they remain a neutral reference.
- **Note**: Step 4a requires `cohort_feature_importance` files from Step 3b. There is no fallback to `aggregated_feature_importance` files from Step 3. Step 3b must run before Step 4a.

### Idempotency and Rebuilds

- `4a_model_data/create_model_data.py` is designed to be **idempotent**:
  - It can be re-run safely after updating feature importances or cohort/gold inputs.
  - Existing `model_events.parquet` files may be skipped to avoid Windows file-in-use errors; deleting a partition’s file and re-running will force a fresh rebuild for that cell.
- Model data can be mirrored between environments via:
  - `aws s3 sync 4a_model_data s3://pgxdatalake/gold/cohorts_model_data --exclude "*" --include "cohort_name=*/age_band=*/model_events.parquet" --profile <profile>`


