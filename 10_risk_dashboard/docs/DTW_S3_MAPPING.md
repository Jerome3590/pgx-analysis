# DTW tab: S3 layout vs manifest / API

## Frontend pattern

**JSON where able for frontend to render with Plotly; PNG only as fallback.**  
The frontend fetches `trajectory_overview_plot.json` from the static plots path when available and renders it with Plotly. Overview/sample PNG URLs are used only when the API returns them (i.e. when those objects exist on S3). The frontend does not synthesize PNG URLs, so missing PNGs do not cause 404s.

## Manifest (dashboard_visual_objects.json)

- **Tab:** DTW Trajectories  
- **s3_path:** `vcu/pgx-risk-calculator/visualizations/dtw/{cohort}/{age_band}/`  
- **static_files:** `["chart_data.json", "sequence_heatmap.json"]`  

So the **base path** for DTW is `.../visualizations/dtw/{cohort}/{age_band}/` (e.g. `opioid_ed/25-44` with hyphen).  
All URLs for chart_data, sequence_heatmap, and `plots/` are built from this base.

## Expected S3 layout (per cohort/age_band)

| Path | Source | Notes |
|------|--------|--------|
| `chart_data.json` | Manifest static_files[0] | At base path (not under plots/) |
| `sequence_heatmap.json` | Manifest static_files[1] | At base path |
| `plots/trajectory_overview_plot.json` | Pipeline (create_dtw_plots) | Plotly JSON; Lambda inlines when &lt; 2MB |
| `plots/dtw_trajectory_analysis_{cohort}_{age_band}.png` | Pipeline copy | Fallback when no Plotly JSON |
| `plots/dtw_sample_trajectories_{cohort}_{age_band}.png` | Pipeline copy | Fallback when no Plotly JSON |
| `plots/dtw_trajectory_cluster_*.html` | Pipeline (create_dtw_plots) | Actual names: `dtw_trajectory_cluster_1d_*` or `dtw_trajectory_cluster_3d_*` |

**Note:** The pipeline writes `dtw_trajectory_cluster_{1d|3d}_{cohort}_{age_band_fname}.html`, not `dtw_trajectory_cluster_interactive_*`. The API accepts both the legacy `interactive` name and the actual `1d`/`3d` names so existing S3 content works.

## S3 check (Feb 2026)

- **chart_data.json**, **sequence_heatmap.json**: Present at each cohort/age_band; mapping correct.
- **plots/trajectory_overview_plot.json**: Present where cluster plots exist; mapping correct.
- **plots/*.html**: Present as `dtw_trajectory_cluster_1d_*` or `dtw_trajectory_cluster_3d_*`; API updated to use these for `overview_interactive` when `dtw_trajectory_cluster_interactive_*` is missing.
- **plots/dtw_trajectory_analysis_*.png**, **plots/dtw_sample_trajectories_*.png**: Optional; created by pipeline only when `dtw_trajectory_cluster_*.png` exists (kaleido). If missing, dashboard uses `trajectory_overview_plot.json` (Plotly) or shows empty.

## Summary

- Manifest and API use the same base path: `.../visualizations/dtw/{cohort}/{age_band}/`.  
- Root-level files (chart_data, sequence_heatmap) match manifest.  
- Plots subfolder and HTML naming are aligned with the pipeline; API accepts actual 1d/3d HTML names.
