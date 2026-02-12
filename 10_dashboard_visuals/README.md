# 10. Dashboard visuals (pipeline step 10)

**Pipeline step:** `10_dashboard_visuals`  
**Run by:** [4_dashboard_visuals.ipynb](../4_dashboard_visuals.ipynb)

This is **phase 10** of the PGx analysis pipeline. The notebook prebuilds all dashboard visualization artifacts (BupaR, DTW, FP-Growth) and uploads them to S3 for the risk calculator dashboard.

## What runs here

- **BupaR** – Process mining sequences and plots → `9_risk_dashboard/visualizations/bupar/`
- **DTW** – Trajectory features and plots → `9_risk_dashboard/visualizations/dtw/`
- **FP-Growth** – Itemsets, Plotly network HTML, PNGs → `9_risk_dashboard/visualizations/fpgrowth/`

Symlinks at repo root: `10b_fpgrowth_dashboard_visual`, `10c_bupaR_dashboard_visual`, `10d_dtw_dashboard_visual` point into `9_risk_dashboard/visualizations/`.

## S3 checkpoint

- **Step name:** `10_dashboard_visuals`
- **Path:** `s3://pgx-repository/pipeline_checkpoints/10_dashboard_visuals/{cohort}/{age_band}/checkpoint.json`
- Per–cohort/age_band checkpoints are written by `create_dtw_visuals.py`; BupaR and FP-Growth use local-output idempotency.

## Test notebook

**[test_dashboard_visuals.ipynb](test_dashboard_visuals.ipynb)** – (1) Validates that BupaR, DTW, and FP-Growth outputs exist locally for each cohort/age_band. (2) **Tests actual creation** by running the BupaR, DTW, and FP-Growth creation scripts for one cohort/age_band (smoke test) and verifying outputs. Run from repo root; creation test requires 4_model_data and (for BupaR) R + bupaR.

## Running locally (S3 sync + Python workflow)

To run dashboard visuals **locally** without the full pipeline or Jupyter:

1. **Sync model data and feature importance from S3** (so BupaR/DTW/FP-Growth have inputs):
   ```bash
   python 10_dashboard_visuals/sync_visualization_data_from_s3.py
   ```
   Optionally: `--profile NAME`, `--model-data-only`, or `--feature-importance-only`.

2. **Run the same workflow as the notebook** (BupaR → DTW → FP-Growth):
   ```bash
   python 10_dashboard_visuals/run_dashboard_visuals.py
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
| 4 | 4_dashboard_visuals.ipynb | **Dashboard visuals (phase 10)** |
| 5 | 5_build_and_deploy.ipynb | Build and deploy |
