# 9. Dashboard visuals (pipeline step 9)

**Pipeline step:** `9_dashboard_visuals`  
**Run by:** [4_dashboard_visuals.ipynb](../4_dashboard_visuals.ipynb)

This is **phase 9** of the PGx analysis pipeline. The notebook prebuilds all dashboard visualization artifacts (BupaR, DTW, FP-Growth) and uploads them to S3 for the risk calculator dashboard (step 10).

## Folder layout (matches 10_risk_dashboard/visualizations)

Each dashboard visualization tab has a folder here, aligned with `10_risk_dashboard/visualizations/`. **Each folder’s README starts with “Research questions this visual answers”**—what the visual is for and how feature importance drives BupaR, DTW, and FP-Growth to reduce noise and focus on what drives the target cohorts.

| Folder | Dashboard tab | Produced here? | Output destination |
|--------|----------------|-----------------|---------------------|
| **bupar/** | BupaR Process Mining | Yes | `10_risk_dashboard/visualizations/bupar/` |
| **causal/** | Causal Analysis | No (see [causal/README.md](causal/README.md)) | `10_risk_dashboard/visualizations/causal/` (from combine_shap_ffa_results) |
| **cohort_pgx/** | PGx Cohort | Yes | `10_risk_dashboard/visualizations/cohort_pgx/` |
| **dtw/** | DTW Trajectories | Yes | `10_risk_dashboard/visualizations/dtw/` |
| **fpgrowth/** | FP-Growth Patterns | Yes | `10_risk_dashboard/visualizations/fpgrowth/` |
| **feature_importance/** | Feature Importance | No (see [feature_importance/README.md](feature_importance/README.md)) | `10_risk_dashboard/visualizations/feature_importance/` (copied from 3a by notebook 4) |

Feature importance heatmaps are produced in **3a_feature_importance**; notebook 4 **copies** them to `10_risk_dashboard/visualizations/feature_importance/` so deploy (notebook 5) syncs from the same location as other visuals.

## What runs here

- **BupaR** – Process mining sequences and plots → `10_risk_dashboard/visualizations/bupar/`
- **DTW** – Trajectory features and plots → `10_risk_dashboard/visualizations/dtw/`. Trajectories are binned by **event density** (events per month: low/medium/high/extreme); chart_data includes density-stratified series for the dashboard Event density filter.
- **FP-Growth** – Risk-predictive co-occurrence (SHAP/FFA-gated, target-only): itemsets, network HTML, PNGs → `10_risk_dashboard/visualizations/fpgrowth/`
- **Cohort PGx** – PharmGKB VIP reports and network topology → `10_risk_dashboard/visualizations/cohort_pgx/`
- **Feature importance** – Heatmaps built in 3a; notebook 4 copies to `10_risk_dashboard/visualizations/feature_importance/` (per cohort + combined).

Outputs: `10_risk_dashboard/visualizations/{bupar,dtw,fpgrowth,cohort_pgx}/` (canonical paths). Scripts live in `9_dashboard_visuals/{bupar,dtw,fpgrowth,cohort_pgx}/`. **Outputs are not committed**—they are generated on EC2 (or locally when running step 9) and uploaded to S3; `*/outputs/` is in `.gitignore`.

**Logs** are written under **`pgx-analysis/logs/{step_name}/`** at the repo root (e.g. `logs/9_dtw/`, `logs/9_bupar/`, `logs/9_fpgrowth/`, `logs/9_cohort_pgx/`). Paths are resolved from the repo root (`REPO_ROOT`). Run the pipeline from **inside the repo** (e.g. `cd pgx-analysis && python 9_dashboard_visuals/run_dashboard_visuals.py`) so that logs end up under `pgx-analysis/logs/` and not in a sibling directory. When S3 mirroring is enabled, DTW (and other pipeline) logs are also uploaded to **`s3://pgx-repository/{step_name}_log/{cohort}/{age_band}/{script}_{cohort}_{age_band}_{timestamp}.log`** (e.g. `s3://pgx-repository/9_dtw_log/opioid_ed/25-44/create_dtw_visuals_opioid_ed_25_44_20250301_120000.log`). To see why N3 (Times Between / Time to Target) did not generate: check those DTW log files for lines like `N3 times_between_sequences: not built — ...` and `N3 time_to_target_sequences: not built — ...`, or inspect `chart_data.json` → `metrics.charts_not_built` (e.g. `times_between_sequences`, `time_to_target_sequences`) for the exact reason.

**Not used for feature engineering:** BupaR, DTW, and FP-Growth outputs under `outputs/feature_engineering/` (sequence features, trajectory/predictive-time CSVs, DTW alignment distances, itemsets/rules) are computed for dashboard visualization and analysis. Results are not added to model data due to concern about target leakage. DTW alignment IS computed using dtaidistance library; FP-Growth and BupaR also perform full analyses for dashboard insights.

**FP-Growth:** This step uses **fpgrowth** (`9_dashboard_visuals/fpgrowth/`) for itemsets and visuals. **4_fpgrowth_analysis** is the template for that code and is **gitignored** (see `.gitignore`: `9_dashboard_visuals/4_fpgrowth_analysis/`); the committed pipeline is `fpgrowth` only.

## How features are filtered by feature importance and used downstream

1. **Source of importance**  
   For each (cohort, age_band), **Step 3b** produces `cohort_feature_importance` (final feature importances). That is the same input used for model training (Step 4) and for SHAP/FFA (Steps 7–8). No other source is used for filtering; there are no fallbacks.

2. **Allowed-codes list**  
   The pipeline builds a single **allowed-codes** set per (cohort, age_band):
   - **BupaR and DTW:** `get_shap_ffa_allowed_codes_combined()` in `py_helpers/shap_ffa_fpgrowth_utils.py` reads Step 3b cohort feature importance and returns the top-N codes by type (drug, ICD, CPT). These are written to `allowed_codes_shap_ffa_{cohort}_{age_band}.json` (under `10_risk_dashboard/visualizations/bupar/outputs/` or the path used by the driver). R (BupaR) and Python (DTW) use this list to **keep only events** whose drug, ICD, or CPT is in the set; all other events are dropped before process mining or trajectory building.
   - **FP-Growth:** `get_final_feature_importance_codes()` loads the same Step 3b CSV and returns allowed items by type (`drug_name`, `icd_code`, `cpt_code`, `medical_code`). FP-Growth **mines itemsets and rules only over these items**; transactions are restricted to items in the allowed set so the patterns reflect what drives the model.

3. **Downstream use**  
   - **BupaR:** Event logs are filtered so each activity (drug/ICD/CPT) is in the allowed-codes JSON. Process matrices, activity frequency, and trace explorer then show only important-feature pathways.
   - **DTW:** Trajectories are built from model_events after filtering: only events with at least one code (drug or ICD or CPT) in the allowed set are kept. DTW alignment and archetypes therefore reflect only important-feature sequences.
   - **FP-Growth:** The transaction list passed to FP-Growth contains only allowed items; frequent itemsets and association rules are computed on this filtered set, so the dashboard shows co-occurrence among model-important codes only.

**Implementation:** `py_helpers/shap_ffa_fpgrowth_utils.py` — `get_shap_ffa_allowed_codes_combined`, `write_shap_ffa_allowed_codes_for_bupar`, `get_final_feature_importance_codes`. Before any BupaR/DTW/FP-Growth run, `run_dashboard_visuals.py` (and notebook 4) ensure the allowed-codes file exists for every (cohort, age_band); if any is missing, the pipeline exits and does not run visuals.

## Feature importance sources for visuals

Dashboard visuals use **two different** feature-importance sources so the right inputs drive each workflow:

| Visual | Feature importance source | Description |
|--------|---------------------------|-------------|
| **BupaR** | **SHAP/FFA combined** | Allowed codes (drug/ICD/CPT) come from merged SHAP + FFA causal importance (Step 7 + 8). Written to `allowed_codes_shap_ffa_{cohort}_{age_band}.json`; R scripts use this file only (no FP-Growth inputs). |
| **DTW** | **SHAP/FFA combined** | Same as BupaR: trajectories and plots are restricted to codes from the combined SHAP/FFA list. |
| **FP-Growth** | **Final feature importances** | Allowed items come from **cohort feature importance** (Step 3b: `cohort_feature_importance` CSV). FP-Growth does not use SHAP/FFA combined. |

Allowed codes for **BupaR, DTW, and FP-Growth** are mandatory from a single source only: **Step 3b cohort_feature_importance** (final feature importances). No fallbacks. Implemented in `py_helpers/shap_ffa_fpgrowth_utils.py` (`get_shap_ffa_allowed_codes_combined`, `write_shap_ffa_allowed_codes_for_bupar`).

## S3 checkpoint

- **Step name:** `9_dashboard_visuals`
- **Path:** `s3://pgx-repository/pipeline_checkpoints/9_dashboard_visuals/{cohort}/{age_band}/checkpoint.json`
- Per–cohort/age_band checkpoints are written by `create_dtw_visuals.py`; BupaR and FP-Growth use local-output idempotency.

## Test notebook

**[test_dashboard_visuals.ipynb](test_dashboard_visuals.ipynb)** – (1) Validates that BupaR, DTW, and FP-Growth outputs exist locally for each cohort/age_band. (2) **Tests actual creation** by running the BupaR, DTW, and FP-Growth creation scripts for one cohort/age_band (smoke test) and verifying outputs. Run from repo root; creation test requires 4_model_data and (for BupaR) R + bupaR.

## Running locally (S3 sync + Python workflow)

To run dashboard visuals **locally** without the full pipeline or Jupyter:

1. **Sync model data and feature importance from S3** (so BupaR/DTW/FP-Growth have inputs):
   ```bash
   python 9_dashboard_visuals/sync_visualization_data_from_s3.py
   ```
   Optionally: `--profile NAME`, `--model-data-only`, or `--feature-importance-only`.

2. **Run the same workflow as the notebook** (BupaR → DTW trajectories → DTW visuals → FP-Growth):
   ```bash
   python 9_dashboard_visuals/run_dashboard_visuals.py
   ```
   - `--no-sync` – skip sync (data already local)
   - `--sync-only` – only run sync, then exit
   - `--cohort X --age-band Y` – restrict to specific combination(s)
   - `--force` – re-run even if outputs exist
   - `--workers N` – parallel workers for BupaR/DTW (default: min(CPU count, combo count))
   - `--fpgrowth-workers N` – parallel FP-Growth combos (default: all combos in parallel for max EC2 capacity; use N to cap)
   DTW runs in three steps: **create_dtw_trajectories.py** (trajectory CSV with N3 time-between metrics and **event density** bins: temporal_span_days, events_per_month, event_density_bin), **create_dtw_features.py** (DTW alignment: distances to prototype trajectories + common_sequences.json), then **create_dtw_visuals.py** (plots and chart_data.json, including density-stratified chart data for the dashboard filter). Requires `dtaidistance` for alignment.

**Why might all age bands and both cohorts not get processed?**
- **Allowed-codes prerequisite:** Before any BupaR/DTW/FP-Growth run, `run_dashboard_visuals.py` checks that every (cohort, age_band) has a non-empty `allowed_codes_shap_ffa_{cohort}_{age_band}.json`. Those files are built only from **Step 3b cohort_feature_importance**. If any combination is missing that CSV (e.g. 0-12 or some bands never ran Step 3b), the allowed-codes file is missing or empty and the script **exits immediately** without running any visuals.
- **Fail-fast (default):** If any BupaR, DTW, or FP-Growth run fails for one combination, the script exits and does not run the remaining combinations. Fix the failing combo (or run with `--cohort X --age-band Y` to retry that combo only), then re-run.
- **FP-Growth batch mode:** Running `cohort_fpgrowth.py` directly (batch) used to limit to 5 combinations when `DRY_RUN = True`; that is now `False` so batch runs process all. The dashboard path uses `run_single_cohort_fpgrowth.py` per (cohort, age_band), so it always submits all combinations unless the prerequisite or fail-fast stops earlier.

3. **Quick DTW test (one age band, both cohorts):**
   ```bash
   python 9_dashboard_visuals/run_dtw_test_one_age_band.py --age-band 25-44
   ```
   Requires allowed_codes and model_events for that age band and both opioid_ed and non_opioid_ed. Use `--force` to re-run.

This mirrors [4_dashboard_visuals.ipynb](../4_dashboard_visuals.ipynb) so you can run from VS Code or the terminal.

**See also:** [archived/dtw_restoration_plan.md](../archived/dtw_restoration_plan.md) for DTW restoration/optimization (reference only); [bupar/README_bupaR.md](bupar/README_bupaR.md) for BupaR scripts and outputs.

## Pipeline order

Run after [3_model_train_shap_ffa.ipynb](3_model_train_shap_ffa.ipynb). Then run [5_build_and_deploy.ipynb](5_build_and_deploy.ipynb) to build and deploy.

## 10-step pipeline (notebooks)

| Step | Notebook | Description |
|------|----------|-------------|
| 0 | 0_config_and_pipeline.ipynb | Config and pipeline setup |
| 1 | 1_cohort_workflow.ipynb | Cohort workflow |
| 2 | 2_feature_importance.ipynb | Feature importance |
| 3 | 3_model_train_shap_ffa.ipynb | Model train, SHAP, FFA |
| 4 | 4_dashboard_visuals.ipynb | **Dashboard visuals (phase 9)** |
| 5 | 5_build_and_deploy.ipynb | Build and deploy |

## Ensuring no outputs are pushed (one-time)

If visualization outputs were ever committed, remove them from tracking so only code is pushed (outputs are generated on EC2). From repo root:

```bash
git rm -r --cached 10_risk_dashboard/visualizations/bupar/outputs/   # if any were tracked
git commit -m "Stop tracking dashboard visualization outputs; generate on EC2"
```

After that, `.gitignore` keeps `10_risk_dashboard/visualizations/*/outputs/` from being re-added.
