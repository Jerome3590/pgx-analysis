# 9. Dashboard visuals (pipeline step 9)

**Pipeline step:** `9_dashboard_visuals`  
**Run by:** [4_dashboard_visuals.ipynb](../4_dashboard_visuals.ipynb)

This is **phase 9** of the PGx analysis pipeline. The notebook prebuilds all dashboard visualization artifacts (BupaR, DTW, FP-Growth) and uploads them to S3 for the risk calculator dashboard (step 10).

## What runs here

- **BupaR** – Process mining sequences and plots → `10_risk_dashboard/visualizations/bupar/`
- **DTW** – Trajectory features and plots → `10_risk_dashboard/visualizations/dtw/`
- **FP-Growth** – Itemsets, Plotly network HTML, PNGs → `10_risk_dashboard/visualizations/fpgrowth/`

Outputs: `10_risk_dashboard/visualizations/{bupar,dtw,fpgrowth}/` (canonical paths). Scripts live in `9_dashboard_visuals/{bupar,dtw,fpgrowth}/`. **Outputs are not committed**—they are generated on EC2 (or locally when running step 9) and uploaded to S3; `*/outputs/` is in `.gitignore`.

**Not used in the model:** BupaR and DTW outputs under `outputs/feature_engineering/` (sequence features, trajectory/predictive-time CSVs) are for dashboard visualization only. We do not add them to model data due to concern about target leakage.

**FP-Growth:** This step uses **fpgrowth** (`9_dashboard_visuals/fpgrowth/`) for itemsets and visuals. **4_fpgrowth_analysis** is the template for that code and is **gitignored** (see `.gitignore`: `9_dashboard_visuals/4_fpgrowth_analysis/`); the committed pipeline is `fpgrowth` only.

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
   - `--workers N` – parallel workers (default 4)
   DTW runs in three steps: **create_dtw_trajectories.py** (trajectory CSV with N3 time-between metrics), **create_dtw_features.py** (DTW alignment: distances to prototype trajectories + common_sequences.json), then **create_dtw_visuals.py** (plots and chart_data.json). Requires `dtaidistance` for alignment.

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
