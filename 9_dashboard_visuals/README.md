# 9. Dashboard visuals (pipeline step 9)

**Pipeline step:** `9_dashboard_visuals`  
**Run by:** [4_dashboard_visuals.ipynb](../4_dashboard_visuals.ipynb)

This is **phase 9** of the PGx analysis pipeline. The notebook prebuilds all dashboard visualization artifacts (BupaR, DTW, FP-Growth) and uploads them to S3 for the risk calculator dashboard (step 10).

## What runs here

- **BupaR** – Process mining sequences and plots → `10_risk_dashboard/visualizations/bupar/`
- **DTW** – Trajectory features and plots → `10_risk_dashboard/visualizations/dtw/`
- **FP-Growth** – Itemsets, Plotly network HTML, PNGs → `10_risk_dashboard/visualizations/fpgrowth/`

Outputs: `10_risk_dashboard/visualizations/{bupar,dtw,fpgrowth}/` (canonical paths). Scripts live in `9_dashboard_visuals/{bupar,dtw,fpgrowth}/`.

## Feature importance sources for visuals

Dashboard visuals use **two different** feature-importance sources so the right inputs drive each workflow:

| Visual | Feature importance source | Description |
|--------|---------------------------|-------------|
| **BupaR** | **SHAP/FFA combined** | Allowed codes (drug/ICD/CPT) come from merged SHAP + FFA causal importance (Step 7 + 8). Written to `allowed_codes_shap_ffa_{cohort}_{age_band}.json`; R scripts use this file only (no FP-Growth inputs). |
| **DTW** | **SHAP/FFA combined** | Same as BupaR: trajectories and plots are restricted to codes from the combined SHAP/FFA list. |
| **FP-Growth** | **Final feature importances** | Allowed items come from **cohort feature importance** (Step 3b: `cohort_feature_importance` CSV). FP-Growth does not use SHAP/FFA combined. |

So: **BupaR and DTW** always use the **SHAP/FFA combined** output; **FP-Growth** uses the **final (cohort) feature importance** list. Implemented in `py_helpers/shap_ffa_fpgrowth_utils.py` (`get_shap_ffa_allowed_codes_combined` for BupaR/DTW, `get_final_feature_importance_codes` for FP-Growth).

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

2. **Run the same workflow as the notebook** (BupaR → DTW → FP-Growth):
   ```bash
   python 9_dashboard_visuals/run_dashboard_visuals.py
   ```
   - `--no-sync` – skip sync (data already local)
   - `--sync-only` – only run sync, then exit
   - `--cohort X --age-band Y` – restrict to specific combination(s)
   - `--force` – re-run even if outputs exist
   - `--workers N` – parallel workers (default 4)

This mirrors [4_dashboard_visuals.ipynb](../4_dashboard_visuals.ipynb) so you can run from VS Code or the terminal.

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
