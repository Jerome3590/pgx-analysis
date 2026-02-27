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
  - **Allowed codes:** **SHAP/FFA combined only.** The file `10_risk_dashboard/visualizations/bupar/outputs/allowed_codes_shap_ffa_{cohort}_{age_band_fname}.json` is written by `create_bupar_visuals.py` from merged SHAP + FFA (Step 7/8). **Allowed codes are always created on EC2** and uploaded to S3; locally run `python 9_dashboard_visuals/sync_visualization_data_from_s3.py --allowed-codes-only` to download them. The R scripts use this file only; if missing, the event log is empty.  
  - Generate:
    - Trace tables (overall, pre-target, post-target; top and rare sequences).
    - Process matrices. (Gantt charts are not produced for the dashboard; see ARCHIVE_GANTT_REMOVAL.md.)
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
  - Interactive HTML (trace explorer, activity frequency) are saved with **external dependencies** (`saveWidget(..., selfcontained = FALSE, libdir = "lib")`) so the plot renders correctly; the `plots/lib/` folder must be deployed alongside the HTML (e.g. sync or upload the whole `plots/` directory).  
  - Final merged feature file:
    - `10_risk_dashboard/visualizations/bupar/outputs/feature_engineering/bupaR_added_features_{cohort}_{age_band}.csv`  
    - Uploaded to `s3://pgxdatalake/gold/feature_engineering/5_bupar/{cohort}/{age_band}/bupaR_added_features_{cohort}_{age_band}.csv` by `add_bupar_features_to_model_data.R`.

### Dashboard aggregated visuals vs patient-level outputs

**Dashboard aggregated visuals** are the cohort/age-band-level plots and interactive HTMLs that the risk dashboard displays. The API (`GET /visualizations/bupar`) returns URLs to these prebuilt assets only. All are under `outputs/{cohort}/{age_band_fname}/plots/` and uploaded to `{S3_DASHBOARD_PREFIX}/bupar/{cohort}/{age_band}/plots/`.

| Artifact | Filename pattern | Description |
|----------|------------------|-------------|
| Activity frequency (image) | `{cohort}_{age_band_fname}_overall_activity_frequency.png` | Top activities by frequency (static) |
| Activity frequency (interactive) | `{cohort}_{age_band_fname}_activity_frequency_interactive.html` | Same with year dropdown (Plotly); requires `plots/lib/` |
| Trace explorer (pre-target only, image) | `{cohort}_{age_band_fname}_trace_explorer_pre_f1120.png` (opioid_ed) or `_trace_explorer_pre_hcg.png` (non_opioid_ed) | Pre-target trace patterns |
| Trace explorer (Plotly from JSON) | `{cohort}_{age_band_fname}_trace_explorer_plot.json` | Pre-target only; dashboard renders Plotly from JSON with filters; PNG fallback `*_trace_explorer_pre_*.png` |
| Pre-target activity frequency | `{cohort}_{age_band_fname}_pre_f1120_activity_frequency.png` (opioid_ed only) | Pre-target activity frequency |
| Process matrix | `{cohort}_{age_band_fname}_process_matrix.png` | Flows between activities ([bupaR Process Matrix](https://bupaverse.github.io/docs/process_matrix.html)) |
| Process matrix (type-pair) | `{cohort}_{age_band_fname}_process_matrix_drug_drug.png` | Drug × Drug only (production) |
| Frequency map | `{cohort}_{age_band_fname}_frequency_map.png` | Process map frequency view (optional; requires processmapR::export_map) |

**Generating all outputs:** Run the full dashboard visuals pipeline for all cohort/age_band combinations (e.g. `python 9_dashboard_visuals/run_dashboard_visuals.py` or the BupaR step in `4_dashboard_visuals.ipynb`) to produce every visual, including type-pair process matrices. The dashboard lets users choose which visuals to show per research question (cohort).

**Produced by this pipeline:** process_matrix.png, process_matrix_drug_drug.png, frequency_map.png (when `processmapR::export_map` is available), trace_explorer_pre_*.png, trace_explorer_plot.json, and for opioid_ed: trace_explorer_post_f1120.png, trace_explorer_post_f1120_interactive.html, post_f1120_activity_frequency.png (non_opioid_ed: post_hcg variants). **Not produced:** overall trace_explorer.png (single PNG for “top 20” without pre/post), process_matrix_interactive.html. **Produced for N2:** activity_sequence_top.png (Sequences to Target). Dashboard uses full JSON + Plotly/Chart.js with filters; no HTML artifacts. PNG fallback only.

**Activity frequency (implemented):** Pipeline exports **overall**, **pre-target**, and **post-target** activity frequency as JSON (`*_activity_frequency.json`, `*_pre_target_activity_frequency.json`, `*_post_target_activity_frequency.json`) to the plots dir; uploaded to S3. Lambda `GET /visualizations/bupar/activity_frequency?cohort=&age_band=` returns all three; frontend renders three bar charts (Chart.js) with year filter. No HTML/iframe for these.

### JSON vs PNG/JPEG (prefer JSON for dashboard visuals)

| Data | JSON available | PNG/JPEG | Dashboard preference |
|------|----------------|----------|----------------------|
| Activity frequency (overall, pre-target, post-target) | Yes: `*_activity_frequency.json`, `*_pre_target_activity_frequency.json`, `*_post_target_activity_frequency.json` | Yes: `*_overall_activity_frequency.png`, `*_pre_f1120_activity_frequency.png`, etc. | **JSON** — frontend fetches JSON and renders Chart.js bar charts (year filter, interactive). |
| Traces (top/rare, pre-target) | Yes (optional): with `--export-csv-to-json`, `*_traces_top.json`, `*_traces_rare.json`, `*_pre_target_traces_top.json`, `*_pre_target_traces_rare.json` in `plots/` | No | **JSON** when exported; API can serve these for future visualizations. |
| Process matrix, trace explorer | Yes: `*_process_matrix_drug_drug.json`, `*_trace_explorer_plot.json` | Yes: `*_process_matrix_drug_drug.png`, `*_trace_explorer_pre_*.png` | **JSON** — frontend renders Plotly from JSON; PNG fallback. |
| Sequence-to-target image | Yes: `*_activity_sequence_top.png` (N2) | Yes | PNG: top 20 pre-target sequences. |

**Rule:** Prefer JSON over static images where JSON exists so the dashboard can render flexible, interactive visuals (e.g. Chart.js, Plotly) instead of fixed PNGs.

**Patient-level metrics** are not displayed in the current dashboard. They are produced for a **follow-on project** (e.g. on-demand patient-level trace/process-matrix views or API). Outputs include:

| Output | Location / pattern | Use |
|--------|---------------------|-----|
| Per-patient pre-target features | `*_train_target_pre_f1120_patient_features_bupar.csv` / `*_pre_hcg_*` | Follow-on patient-level API |
| Per-patient post-target features | `*_train_target_post_f1120_patient_features_bupar.csv` (opioid_ed) | Follow-on / EDA |
| Time-to-target features | `*_train_target_time_to_f1120_features_bupar.csv` / `*_time_to_hcg_*` | Follow-on |
| Trace tables (per case) | `*_traces_bupar.csv`, `*_traces_top_bupar.csv`, `*_traces_rare_bupar.csv` | Follow-on / analytics |
| Process matrix (tabular) | `*_train_target_process_matrix_bupar.csv` | Follow-on / analytics |
| Merged BupaR features | `bupaR_added_features_{cohort}_{age_band}.csv` (feature_engineering/) | Dashboard visualization only; not model input; patient-level drill-down in follow-on |

Pipeline logging uses the tag **`[DASHBOARD_AGGREGATED]`** for the artifacts in the first table and **patient-level** for the second so logs clearly separate what is served to the dashboard vs what is for the follow-on project.

### Why is SHAP/FFA missing for only one cohort/age_band?

BupaR needs the **SHAP/FFA allowed-codes** file for each `(cohort, age_band)`. That file is built from Step 7 (SHAP) and Step 8 (FFA) outputs. If BupaR fails for one combination (e.g. `non_opioid_ed` / `75-84`) while others (e.g. `non_opioid_ed` / `65-74`) succeed, common causes are:

1. **Step 7 or Step 8 never ran for that combination**  
   Phase 3 runs Step 7 → Step 8 → Combine in a single loop over all `REQUIRED_COHORTS`. If the run was **interrupted** (timeout, crash, manual stop) before that combination’s Step 7 or Step 8 finished, only later combinations will be missing. Order is: `opioid_ed` 0-12 … 85-114, then `non_opioid_ed` 0-12 … 65-74, **75-84**, 85-114. So 75-84 is near the end; an early stop can leave 75-84 (and 85-114) without SHAP/FFA.

2. **Step 7 or Step 8 failed for that combination**  
   If Step 8 (FFA) fails for 75-84, the notebook cell exits and Combine never runs for 75-84. Re-run Phase 3 from Step 7 (or Step 8) for the full loop so 75-84 gets SHAP and FFA outputs.

3. **Only a subset was run**  
   If Step 7 / Step 8 / Combine were run manually for a subset (e.g. only 65-74), then 75-84 will have no outputs. Run the full loop in Phase 3 for all combinations.

4. **Where BupaR looks for SHAP/FFA**  
   Allowed codes are read from:  
   - `7_shap_analysis/outputs/{cohort}/{age_band_fname}/*_shap_global_importance_xgboost.csv`  
   - `8_ffa_analysis/outputs/{cohort}/{age_band_fname}/xgboost/causal_importance.parquet` or `feature_importance_axp.parquet`  
   - Fallback: `10_risk_dashboard/outputs/{cohort}/{age_band_fname}/combined_importance.csv` (Combine step).  
   If your pipeline writes SHAP/FFA under a data root (e.g. NVMe), ensure BupaR is run with that same layout or that Combine has been run so the fallback file exists.

**Fix:** Run Phase 3 (Model train + SHAP/FFA) so that **Step 7**, **Step 8**, and **Combine** complete for the missing combination (e.g. `non_opioid_ed` / `75-84`), then re-run dashboard visuals (Phase 4).

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

BupaR feature outputs are for dashboard visualization only; we do not add them to model data (same as DTW and FP-Growth) due to concern about target leakage.

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

