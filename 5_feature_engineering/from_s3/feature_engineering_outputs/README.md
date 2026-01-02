## Central Feature Engineering Outputs

This directory is a **cohort- and age-band–organized mirror** of the main
feature engineering steps. It exists to make it easy to grab all inputs
for a given `(cohort, age_band)` without chasing paths across multiple
step-specific folders.

### Layout

Top-level structure:

- `4_fpgrowth/{cohort}/{age_band}/`
- `5_bupar/{cohort}/{age_band}/`
- `6_dtw/{cohort}/{age_band}/`
- `7_pgx/{cohort}/{age_band}/`

Where:

- **`{cohort}`**: e.g. `opioid_ed`, `non_opioid_ed`  
- **`{age_band}`**: canonical age-band label, e.g. `0-12`, `13-24`, `65-74`

Each leaf directory contains **CSV artifacts and, where applicable, plots**
for that `(cohort, age_band)`.

### Step 5b – FP-Growth (`4_fpgrowth/`)

Mirror of outputs from `5b_fpgrowth_analysis/`:

- `4_fpgrowth/{cohort}/{age_band}/fpgrowth_features_{cohort}_{age_band_fname}.csv`  
  - Intermediate patient-level FP-Growth features.
- `4_fpgrowth/{cohort}/{age_band}/fpgrowth_added_features_{cohort}_{age_band_fname}.csv`  
  - Final FP-Growth feature block ready to join to model data.
- `4_fpgrowth/{cohort}/{age_band}/plots/`  
  - PNG/HTML visualizations for FP-Growth itemsets and rules.

Source scripts:

- `5b_fpgrowth_analysis/create_fpgrowth_features.py`
- `5b_fpgrowth_analysis/add_fpgrowth_features_to_model_data.py`
- `5b_fpgrowth_analysis/create_plots.py` (called via `run_analysis.py`)

### Step 5a – BupaR (`5_bupar/`)

Mirror of outputs from `5a_bupaR_analysis/`:

- `5_bupar/{cohort}/{age_band}/bupaR_added_features_{cohort}_{age_band_fname}.csv`  
  - Final BupaR feature block ready to join to model data.
- `5_bupar/{cohort}/{age_band}/sequence_features_{cohort}_{age_band_fname}.csv` (if present)  
  - Optional sequence-level features derived from trace tables.
- `5_bupar/{cohort}/{age_band}/plots/` (for `opioid_ed` currently)  
  - Gantt charts, activity frequency plots, and related BupaR visuals.

Source scripts:

- `5a_bupaR_analysis/create_bupar_outputs_opioid_ed.R`
- `5a_bupaR_analysis/create_bupar_outputs_non_opioid_ed.R`
- `5a_bupaR_analysis/add_bupar_features_to_model_data.R`

### Step 4b / 6 – DTW (`6_dtw/`)

Mirror of DTW feature outputs from `5d_dtw_analysis/` (DTW feature-add step, currently labeled as step 6_dtw in the feature tree):

- `6_dtw/{cohort}/{age_band}/dtw_features_{cohort}_{age_band_fname}.csv`  
  - Patient-level DTW trajectory features.
- `6_dtw/{cohort}/{age_band}/dtw_added_features_{cohort}_{age_band_fname}.csv`  
  - Final DTW feature block ready to join to model data.

Source scripts:

- `5d_dtw_analysis/create_dtw_features.py`
- `5d_dtw_analysis/add_dtw_features_to_model_data.py`

### Step 5c – PGx (`7_pgx/`)

Mirror of outputs from `5c_pgx_analysis/` and `7_pgx_analysis/`:

- `7_pgx/{cohort}/{age_band}/pgx_features_{cohort}_{age_band_fname}.csv`  
  - Patient-level PGx features.
- `7_pgx/{cohort}/{age_band}/pgx_added_features_{cohort}_{age_band_fname}.csv`  
  - Final PGx feature block ready to join to model data.

Source scripts:

- `5c_pgx_analysis/create_pgx_features.py`
- `5c_pgx_analysis/add_pgx_features_to_model_data.py`

### Usage

For a given `(cohort, age_band)` (for example, `opioid_ed / 0-12`), you can:

- Inspect all feature blocks and plots under:
  - `feature_engineering_outputs/4_fpgrowth/opioid_ed/0-12/`
  - `feature_engineering_outputs/5_bupar/opioid_ed/0-12/`
  - `feature_engineering_outputs/6_dtw/opioid_ed/0-12/`
  - `feature_engineering_outputs/7_pgx/opioid_ed/0-12/`
- Join any `*_added_features_*.csv` to the final modeling table by `mi_person_key`.

This mirror is **read-only** from the perspective of the main pipeline; all
authoritative writes still happen in the step-specific directories
(`5b_fpgrowth_analysis/outputs/...`, `5a_bupaR_analysis/outputs/...`, etc.).

