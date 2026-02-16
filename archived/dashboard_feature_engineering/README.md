# Archived: Dashboard feature-engineering scripts

These scripts are **no longer run by the dashboard visuals pipeline**. They were moved here so the pipeline only produces itemsets/outputs and visualizations (no feature creation or merge steps).

| Script | Original location | Purpose |
|--------|-------------------|---------|
| `fpgrowth/create_fpgrowth_features.py` | 9_dashboard_visuals/fpgrowth | Built patient-level FP-Growth features from itemsets/rules → CSV |
| `fpgrowth/add_fpgrowth_features_to_model_data.py` | 9_dashboard_visuals/fpgrowth | Merged FP-Growth features into standalone CSV for dashboard |
| `bupar/add_bupar_features_to_model_data.R` | 9_dashboard_visuals/bupar | Merged BupaR features into standalone CSV for dashboard |
| `dtw/create_dtw_features.py` | 9_dashboard_visuals/dtw | Built DTW trajectory features CSV from model_events |

**Current pipeline (no feature engineering):**

- **FP-Growth:** ensure_itemsets → create_visualizations
- **BupaR:** create_bupar_outputs → upload_bupar_plots_to_dashboard_s3
- **DTW:** create_dtw_visuals only (loads existing DTW features CSV if present)

To run any of these scripts manually (e.g. to regenerate feature CSVs), use the paths under `archived/dashboard_feature_engineering/` and set working directory or project-root as needed.
