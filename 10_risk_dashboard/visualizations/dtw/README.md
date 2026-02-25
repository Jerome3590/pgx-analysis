# DTW visualization outputs

This directory contains **outputs only** for DTW dashboard visualizations (features, plots, chart_data).  
**Creation code** lives in `9_dashboard_visuals/dtw/` (pipeline step 9). Do not add scripts here; follow the same pattern as `6_final_model`, `7_shap_analysis`, and `8_ffa_analysis` (code in step folder, outputs in designated output location).

## Pipeline outputs and dashboard consumption

| Location | Contents | Dashboard use |
|----------|----------|----------------|
| `outputs/feature_engineering/` | `dtw_features_*.csv`, `common_sequences_*.json`, `trajectory_status_*.json` | **Inputs** to `create_dtw_visuals`; not served directly. |
| `outputs/{cohort}/{age_band_fname}/plots/` | `dtw_trajectory_analysis_*.png`, `dtw_sample_trajectories_*.png`, interactive HTML | Uploaded to S3 `dtw/{cohort}/{age_band}/plots/`; overview and sample images. |
| S3 `dtw/{cohort}/{age_band}/chart_data.json` | routine_comparison, high_risk_trajectories, times_between_sequences, time_to_target_sequences, target_pathway_patterns | **JSON preferred.** Lambda may return inline; frontend uses for Routine vs No Routine, N3, Target Pathway, Trajectory Metrics. |
| S3 `dtw/{cohort}/{age_band}/sequence_heatmap.json` | icd/cpt/drug codes × position counts | **JSON preferred.** Common Sequences Heatmap. |

**Run order:** `create_dtw_features` → writes CSV + common_sequences in feature_engineering. Then **`create_dtw_visuals`** reads that CSV, builds chart_data and sequence_heatmap, creates plots, and uploads chart_data.json, sequence_heatmap.json, and plots to the dashboard S3 prefix. If create_dtw_visuals is not run, the DTW tab will show “no data” for metrics/routine/N3/heatmap and “not available” for overview/sample images.

**JSON vs PNG:** Prefer JSON where available. Lambda fetches chart_data.json and sequence_heatmap.json from S3 and returns them inline when present; frontend uses inline data first, then falls back to fetching chart_data_url / sequence_heatmap_url. Overview and sample visuals are PNG/HTML only; image URLs are returned only when the objects exist on S3.
