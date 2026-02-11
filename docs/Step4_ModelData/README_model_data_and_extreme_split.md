## Step 4: Model Data

This step produces **model-ready event data** for each `(cohort, age_band)` using refined feature importance from Step 3c. Step 4 removes target leakage for case events (events before target date only). ICD/administrative code filtering runs in **Step 1b** (`1b_apcd_event_filter`).

### Step 4 – Model Data (`4_model_data/`)

**Goal**: Build compact, analysis-ready `model_events.parquet` files for target and control cohorts.

- **Target and control** (per cohort/age_band):
  - Reads refined `cohort_feature_importance.csv` from Step 3c (required).
  - Reads Step 2 cohort parquets and gold medical/pharmacy from `DATA_ROOT/gold/cohorts`, `gold/medical`, `gold/pharmacy` (or project data paths on Windows).
  - Builds cases (target=1) and controls (target=0), samples controls to approximate 5:1 ratio.
  - Writes: `4_model_data/cohort_name={cohort}/age_band={band}/model_events.parquet`

These `model_events.parquet` files are the **canonical inputs** for Step 5 (PGx), Step 6 (final model), and dashboard visualizations.

### Optional: Extreme-Density Cohort Extraction (Dashboard Visualizations)

**Goal**: For exploratory dashboard analysis only, extreme-density patients can be split out so they do not dominate main models.

- **Scripts** (if present): `9_risk_dashboard/visualizations/dtw/extract_extreme_density_cohort.py`, `summarize_extreme_density_cohort.py`
- **Input**: `4_model_data/cohort_name={cohort}/age_band={band}/model_events.parquet`
- **Outputs**: Optional extreme-density cohort parquets and summaries under `9_risk_dashboard/visualizations/` or project data paths. **Not required** for the main pipeline (Steps 5–9).

### How Step 4 Connects to Later Steps

- **Main output** (`model_events.parquet`) flows into:
  - PGx feature engineering (Step 5: `5_pgx_analysis/`)
  - Final model training (Step 6: `6_final_model/`) – train/test from Step 6 are uploaded to S3 (required for SHAP/FFA).
- **Extreme cohorts** (if generated) are for **exploratory visualization** only in `9_risk_dashboard/visualizations/`, not for the risk calculator models.
