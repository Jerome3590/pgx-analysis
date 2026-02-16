## BupaR Process Mining Analysis

This directory contains scripts to run process-mining analysis on cohort trajectories using the BupaR ecosystem (R), and to turn those outputs into patient-level features for dashboard visualization. **We do not add BupaR (or DTW or FP-Growth) features to model data**; they are for dashboard visuals only.

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
  - **Allowed codes:** **SHAP/FFA combined only.** The file `10_risk_dashboard/visualizations/bupar/outputs/allowed_codes_shap_ffa_{cohort}_{age_band_fname}.json` is written by `create_bupar_visuals.py` from merged SHAP + FFA causal importance (Step 7/8). The R scripts use this file only; there are no FP-Growth inputs. If the file is missing, the event log is empty.  
  - Generate:
    - Trace tables (overall, pre-target, post-target; top and rare sequences).
    - Process matrices and Gantt-style visualizations.
    - Per-patient pre-target features, time-to-target features, and (optionally) post-target features.
  - Save per-cohort outputs under `10_risk_dashboard/visualizations/bupar/outputs/{cohort}/{age_band_fname}/...` and upload CSVs to S3 `gold/bupar/...`.

- `create_sequence_features.R`  
  - Consumes BupaR trace outputs and builds sequence-based patient features.  
  - Reads from `10_risk_dashboard/visualizations/bupar/outputs/{cohort}/{age_band_fname}/features/*traces*_bupar.csv`.  
  - Produces:
    - `sequence_features_{cohort}_{age_band}.csv` under `10_risk_dashboard/visualizations/bupar/outputs/feature_engineering/`.  
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
    - `10_risk_dashboard/visualizations/bupar/outputs/feature_engineering/bupaR_added_features_{cohort}_{age_band}.csv` (dashboard visualization only; not added to model data).

### Inputs and Outputs

- **Inputs** (local):
  - `4_model_data/cohort_name={cohort}/age_band={age_band}/model_events.parquet` (or 3b path if present).
  - **Allowed codes:** `10_risk_dashboard/visualizations/bupar/outputs/allowed_codes_shap_ffa_{cohort}_{age_band_fname}.json` (SHAP/FFA combined; written by Python before R runs). No FP-Growth inputs.
  - BupaR trace and feature CSVs under `10_risk_dashboard/visualizations/bupar/outputs/{cohort}/{age_band_fname}/features/` (for merge step).

- **Outputs** (local + S3):
  - Eventlogs, trace tables, process matrices, and plots under `10_risk_dashboard/visualizations/bupar/outputs/{cohort}/{age_band_fname}/`.
  - Interactive HTML (trace explorer, process matrix, activity frequency) are saved as **single self-contained files** (`saveWidget(..., selfcontained = TRUE, libdir = NULL)`) so they work from any path (S3, dashboard iframe)—no folder or `lib/` dependency.  
  - Final merged feature file:
    - `10_risk_dashboard/visualizations/bupar/outputs/feature_engineering/bupaR_added_features_{cohort}_{age_band}.csv`  
    - Uploaded to `s3://pgxdatalake/gold/feature_engineering/5_bupar/{cohort}/{age_band}/bupaR_added_features_{cohort}_{age_band}.csv` by `add_bupar_features_to_model_data.R`.

### Typical Workflow

For a given `(cohort, age_band)`:

1. Ensure `4_model_data` (or 3b) and the SHAP/FFA allowed-codes JSON exist (the Python step writes the JSON from SHAP/FFA combined).
2. In R:
   - Run `build_bupar_eventlogs.R` (if you need standalone eventlogs).  
   - Run `create_bupar_outputs_opioid_ed.R` or `create_bupar_outputs_non_opioid_ed.R` to generate traces and per-patient BupaR feature CSVs.
3. In R:
   - Run `create_sequence_features.R --cohort-name {cohort} --age-band {age_band}` to create `sequence_features_{cohort}_{age_band}.csv`.
4. In R:
   - Run `add_bupar_features_to_model_data.R --cohort-name {cohort} --age-band {age_band}` to produce `bupaR_added_features_{cohort}_{age_band}.csv`.

BupaR feature outputs are for dashboard visualization only; we do not add them to model data (same as DTW and FP-Growth).

**Feature importance source:** BupaR uses **SHAP/FFA combined** only (see [9_dashboard_visuals/README.md](../README.md#feature-importance-sources-for-visuals)). DTW uses the same; FP-Growth uses final feature importances instead.

### Testing BupaR locally (sample data from S3)

You can test BupaR visuals locally by syncing a **sample** of model data (and optionally feature importance) from S3, then running the R script or the Python BupaR step for one cohort/age_band.

1. **Download a sample** (from repo root):
   ```bash
   # One cohort/age_band (recommended for a quick test)
   python 9_dashboard_visuals/sync_visualization_data_from_s3.py --cohort non_opioid_ed --age-band 65-74

   # Or two age bands
   python 9_dashboard_visuals/sync_visualization_data_from_s3.py --cohort opioid_ed --age-band 25-44 --age-band 55-64
   ```
   This writes to `4_model_data/cohort_name={cohort}/age_band={age_band}/model_events.parquet` (and, unless you pass `--model-data-only`, also syncs feature importance to `data_root/gold/feature_importance/`).

2. **Run BupaR for that combination**  
   **Option A – Python runner** (from repo root; writes SHAP/FFA allowed codes then runs R, then merge):
   ```bash
   python 9_dashboard_visuals/run_dashboard_visuals.py --no-sync --cohort non_opioid_ed --age-band 65-74
   ```
   This runs only the BupaR (and DTW/FP-Growth) steps for that one cohort/age_band. Use `--no-sync` because you already synced the sample in step 1.

   **Option B – R only** (from repo root; uses FP-Growth itemsets if SHAP/FFA allowed codes file is missing):
   ```bash
   Rscript 9_dashboard_visuals/bupar/create_bupar_outputs_non_opioid_ed.R 65-74
   # or
   Rscript 9_dashboard_visuals/bupar/create_bupar_outputs_opioid_ed.R 25-44
   ```
   R looks for model data under `4_model_data/cohort_name=.../age_band=.../model_events.parquet` (or 3b path if present).

3. **Plots and outputs**  
   - Plots: `10_risk_dashboard/visualizations/bupar/outputs/{cohort}/{age_band_fname}/plots/*.png`  
   - Features/CSVs: `10_risk_dashboard/visualizations/bupar/outputs/{cohort}/{age_band_fname}/features/` and `.../feature_engineering/`.

If you only sync model data (no feature importance), the R script will use FP-Growth itemsets when present under `4_fpgrowth_analysis/outputs/...`; if those are missing too, allowed_codes may be empty and the event log may be built from all codes (depending on script logic). For a minimal test, syncing one cohort/age_band + feature importance is enough.

### TODO: Patient-level BupaR visuals

Patient-level BupaR visuals (trace explorer, process matrix, frequency map filtered by cohort/age_band/patient subset) are not yet implemented. They require on-demand R execution; we will need to install R in Lambda (or use a separate R runtime/service) and add an endpoint (e.g. POST /visualizations/bupar/patient-level) plus dashboard filter UI to submit selections for Lambda to process.

