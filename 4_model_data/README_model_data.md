## Model Data (Step 4) – Event-Level Inputs for Feature Engineering and Final Models

**Single canonical location:** All model data uses one root: `py_helpers.env_utils.get_model_data_root()` (local: `4_model_data` under data root or project root; S3: `gold/cohorts_model_data/`). Do not use `gold/model_training_data` or other paths—use one location for efficiency.

This directory contains event-level `model_events.parquet` files used as inputs to:

- Step 5 PGx features, Step 6 final models, and dashboard mining visualizations (BupaR, FP-Growth, DTW under `9_dashboard_visuals/`) that consume cohort / model data
- Step 6 final models (RandomForest, XGBoost, CatBoost)

Each file corresponds to a specific `(cohort_name, age_band)` cell:

- `4_model_data/cohort_name={cohort}/age_band={age_band}/model_events.parquet`

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

### Target leakage removal (Step 4)

Step 4 removes target leakage when building model data: for **case events**, only events **strictly before** the target date are kept. Model_events uses explicit target-date column names: **`first_f1120_date`** (opioid_ed; F11.20 = opioid use disorder) and **`first_o11_p_date`** (non_opioid_ed; O11_P = canonical ED identifier, includes P51b/O11/P33, 21-day drug window). See 2_create_cohort README § HCG-Based ED Visit Targets. Events on or after the target date are dropped.

**Target-date column naming (`non_opioid_ed`):** The target date is taken from one cohort column (`first_ed_non_opioid_date` or `first_opioid_ed_date` depending on export). Semantics align; canonical model_events output uses **`first_o11_p_date`**. If an older parquet still has the alternate names, **rename** to `first_o11_p_date` rather than recomputing events.

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
- **Drug name exclusions (model training)**: The following values are removed from the drug-name feature set when building model data and final features. They are defined in `py_helpers.constants.DRUG_NAMES_EXCLUDED_MODEL_TRAINING` and applied in Step 3b, Step 4 (`get_important_items`), and Step 6 (`build_final_cohort_model_features`). See `1b_apcd_event_filter/README_administrative_codes_lookup.md` for the lookup table.

  | Value     | Reason |
  |-----------|--------|
  | **Narcan**   | Excluded per model-training requirements. |
  | **Unknown**  | Placeholder, not a drug. |
  | **Fentanyl** | Excluded per model-training requirements. |
  | **1036F**    | Not a drug. CPT Category II tracking code used to document that a patient (18+) is a current tobacco non-user, usually during preventive screenings; part of quality measures for tobacco use assessment and preventive care. |
  | **T401XA1**  | Not a drug. ICD-10-CM diagnosis code for *Poisoning by 4-aminophenol derivatives, accidental (unintentional), initial encounter* — in practice usually unintentional overdose or poisoning with acetaminophen (paracetamol) or closely related compounds, at the patient's initial encounter. |
- **Note**: Step 4a requires `cohort_feature_importance` files from Step 3b. There is no fallback to `aggregated_feature_importance` files from Step 3. Step 3b must run before Step 4a.

### Idempotency and Rebuilds

- `4_model_data/create_model_data.py` is designed to be **idempotent**:
  - It can be re-run safely after updating feature importances or cohort/gold inputs.
  - Existing `model_events.parquet` files may be skipped to avoid Windows file-in-use errors; deleting a partition’s file and re-running will force a fresh rebuild for that cell.
- Model data can be mirrored between environments via:
  - `aws s3 sync 4_model_data s3://pgxdatalake/gold/cohorts_model_data --exclude "*" --include "cohort_name=*/age_band=*/model_events.parquet" --profile <profile>`

### Checking gold/pharmacy (and gold/medical) completeness

Before or after syncing from S3, you can verify that `gold/pharmacy` (and optionally `gold/medical`) on NVMe has all expected cells. Step 4 expects:

- **Age bands:** 13-24, 25-44, 45-54, 55-64, 65-74, 75-84, 85-114 (full set for both cohorts)
- **Event years:** 2016, 2017, 2018, 2019  
- **Layout:** `gold/pharmacy/age_band={band}/event_year={year}/*.parquet`

From project root:

```bash
python 4_model_data/check_gold_pharmacy_completeness.py           # pharmacy only
python 4_model_data/check_gold_pharmacy_completeness.py --medical  # pharmacy + medical
python 4_model_data/check_gold_pharmacy_completeness.py --s3       # compare to S3 object count/size
```

The script reports present/missing cells, file count, and total size. If pharmacy looks low (e.g. &lt; ~4 GB total), re-sync from S3:  
`aws s3 sync s3://pgxdatalake/gold/pharmacy /mnt/nvme/gold/pharmacy`


