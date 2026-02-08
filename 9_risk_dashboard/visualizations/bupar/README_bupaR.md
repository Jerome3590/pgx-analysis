## BupaR Process Mining Analysis

This directory contains scripts to run process-mining analysis on cohort trajectories using the BupaR ecosystem (R), and to turn those outputs into patient-level features for modeling.

### Key Scripts

- `build_bupar_eventlogs.R`  
  - Builds BupaR-compatible event logs from `4a_model_data` and FP-Growth itemsets.  
  - Creates:
    - `target_eventlog`: target-only sequence of DRUG/ICD/CPT activities.
    - `sankey_eventlog`: combined target + control eventlog for visualization.  
  - Uses:
    - `4a_model_data/cohort_name={cohort}/age_band={age_band}/model_events.parquet`
    - `4_fpgrowth_analysis/outputs/{cohort}/target/{age_band}/train/*_itemsets_target_only.json`

- `create_bupar_outputs_opioid_ed.R` / `create_bupar_outputs_non_opioid_ed.R`  
  - Run full pre-/post-target sequence analyses for each cohort and age band.  
  - **Allowed codes:** If `10c_bupaR_dashboard_visual/outputs/allowed_codes_shap_ffa_{cohort}_{age_band_fname}.json` exists (written by `run_analysis.py` from SHAP/FFA), the event log is restricted to those model-important items; otherwise FP-Growth target-only itemsets are used.  
  - Generate:
    - Trace tables (overall, pre-target, post-target; top and rare sequences).
    - Process matrices and Gantt-style visualizations.
    - Per-patient pre-target features, time-to-target features, and (optionally) post-target features.
  - Save per-cohort outputs under `10c_bupaR_dashboard_visual/outputs/{cohort}/{age_band_fname}/...` and upload CSVs to S3 `gold/bupar/...`.

- `create_sequence_features.R`  
  - Consumes BupaR trace outputs and builds sequence-based patient features.  
  - Reads from `10c_bupaR_dashboard_visual/outputs/{cohort}/{age_band_fname}/features/*traces*_bupar.csv`.  
  - Produces:
    - `sequence_features_{cohort}_{age_band}.csv` under `10c_bupaR_dashboard_visual/outputs/feature_engineering/`.  
  - Features include:
    - Binary indicators for top vs rare sequences (overall / pre-/post-target).
    - Sequence frequency and match counts.
    - Simple sequence category fields (`top`, `rare`, `other`).

- `create_top_sequences.py`  
  - Backfills "top sequences" tables when only raw trace tables exist.  
  - Reads `*_traces_bupar.csv` and writes corresponding `*_traces_top_bupar.csv` using simple frequency thresholds.

- `add_bupar_features_to_model_data.R`  
  - Final aggregation step for BupaR-derived features.  
  - Inputs:
    - Pre-target per-patient features.
    - Post-target per-patient features (descriptive, optional).
    - Time-to-target features.
    - Optional `sequence_features_{cohort}_{age_band}.csv`.  
  - Outputs:
    - `10c_bupaR_dashboard_visual/outputs/feature_engineering/bupaR_added_features_{cohort}_{age_band}.csv`, ready to join to model data via `mi_person_key`.

### Inputs and Outputs

- **Inputs** (local):
  - `4a_model_data/cohort_name={cohort}/age_band={age_band}/model_events.parquet`
  - FP-Growth target-only itemsets under `4_fpgrowth_analysis/outputs/{cohort}/target/{age_band_fname}/train/`.
  - BupaR trace and feature CSVs under `10c_bupaR_dashboard_visual/outputs/{cohort}/{age_band_fname}/features/`.

- **Outputs** (local + S3):
  - Eventlogs, trace tables, process matrices, and plots under `10c_bupaR_dashboard_visual/outputs/{cohort}/{age_band_fname}/`.  
  - Final merged feature file:
    - `10c_bupaR_dashboard_visual/outputs/feature_engineering/bupaR_added_features_{cohort}_{age_band}.csv`  
    - Uploaded to `s3://pgxdatalake/gold/feature_engineering/5_bupar/{cohort}/{age_band}/bupaR_added_features_{cohort}_{age_band}.csv` by `add_bupar_features_to_model_data.R`.

### Typical Workflow

For a given `(cohort, age_band)`:

1. Ensure `4a_model_data` and FP-Growth itemsets exist locally.
2. In R:
   - Run `build_bupar_eventlogs.R` (if you need standalone eventlogs).  
   - Run `create_bupar_outputs_opioid_ed.R` or `create_bupar_outputs_non_opioid_ed.R` to generate traces and per-patient BupaR feature CSVs.
3. In R:
   - Run `create_sequence_features.R --cohort-name {cohort} --age-band {age_band}` to create `sequence_features_{cohort}_{age_band}.csv`.
4. In R:
   - Run `add_bupar_features_to_model_data.R --cohort-name {cohort} --age-band {age_band}` to produce `bupaR_added_features_{cohort}_{age_band}.csv`.

All BupaR feature outputs are designed to be joined to model-level tables on `mi_person_key` just before final model training.

