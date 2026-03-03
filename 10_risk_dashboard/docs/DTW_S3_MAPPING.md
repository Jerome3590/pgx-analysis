# DTW tab: S3 layout vs manifest / API

## No empty artifacts

When a DTW plot or chart doesn’t produce data, the pipeline **always** writes a JSON artifact with `message`, `empty: true`, `cohort`, `age_band`, and `metrics` (e.g. `reason`, `dtw_rows`) so the dashboard can show why there is no output. Never leave a missing file or plain `{}`. Applies to `chart_data.json`, `sequence_heatmap.json`, and `plots/trajectory_overview_plot.json`.

## Where the file lives on EC2 (local before sync)

All DTW outputs are written under the **dashboard visualizations** tree, not under `9_dashboard_visuals`:

| What | Path on EC2 (relative to project root) |
|------|----------------------------------------|
| DTW root | `10_risk_dashboard/visualizations/dtw/` |
| Per cohort/age (use **underscore** in age_band) | `10_risk_dashboard/visualizations/dtw/{cohort}/{age_band_fname}/` e.g. `.../dtw/opioid_ed/25_44/` |
| **trajectory_overview_plot.json** | `10_risk_dashboard/visualizations/dtw/{cohort}/{age_band_fname}/plots/trajectory_overview_plot.json` |
| chart_data.json | `.../dtw/{cohort}/{age_band_fname}/chart_data.json` |
| sequence_heatmap.json | `.../dtw/{cohort}/{age_band_fname}/sequence_heatmap.json` |

## All DTW objects: EC2 path and S3 path

`{cohort}` = e.g. `opioid_ed`, `non_opioid_ed`. EC2 uses **underscore** in age band (e.g. `25_44`); S3 uses **hyphen** (e.g. `25-44`). S3 keys are under the dashboard prefix (e.g. `vcu/pgx-risk-calculator/`).

| DTW artifact | EC2 path (relative to repo root) | S3 object key (under prefix) |
|--------------|-----------------------------------|-------------------------------|
| chart_data.json | `10_risk_dashboard/visualizations/dtw/{cohort}/{age_band_fname}/chart_data.json` | `visualizations/dtw/{cohort}/{age_band}/chart_data.json` |
| sequence_heatmap.json | `10_risk_dashboard/visualizations/dtw/{cohort}/{age_band_fname}/sequence_heatmap.json` | `visualizations/dtw/{cohort}/{age_band}/sequence_heatmap.json` |
| trajectory_overview_plot.json | `10_risk_dashboard/visualizations/dtw/{cohort}/{age_band_fname}/plots/trajectory_overview_plot.json` | `visualizations/dtw/{cohort}/{age_band}/plots/trajectory_overview_plot.json` |
| dtw_trajectory_analysis (PNG) | `.../dtw/{cohort}/{age_band_fname}/plots/dtw_trajectory_analysis_{cohort}_{age_band_fname}.png` | `visualizations/dtw/{cohort}/{age_band}/plots/dtw_trajectory_analysis_{cohort}_{age_band_fname}.png` |
| dtw_sample_trajectories (PNG) | `.../dtw/{cohort}/{age_band_fname}/plots/dtw_sample_trajectories_{cohort}_{age_band_fname}.png` | `visualizations/dtw/{cohort}/{age_band}/plots/dtw_sample_trajectories_{cohort}_{age_band_fname}.png` |
| dtw_trajectory_cluster (HTML) | `.../dtw/{cohort}/{age_band_fname}/plots/dtw_trajectory_cluster_1d_{cohort}_{age_band_fname}.html` (or `3d_`) | `visualizations/dtw/{cohort}/{age_band}/plots/dtw_trajectory_cluster_1d_{cohort}_{age_band_fname}.html` (or `3d_`) |
| dtw_trajectory_cluster (PNG, optional) | `.../dtw/{cohort}/{age_band_fname}/plots/dtw_trajectory_cluster_1d_{cohort}_{age_band_fname}.png` (or `3d_`) | `visualizations/dtw/{cohort}/{age_band}/plots/dtw_trajectory_cluster_1d_{cohort}_{age_band_fname}.png` (or `3d_`) |

**Example (opioid_ed, 25–44):**

- Full path: `{PROJECT_ROOT}/10_risk_dashboard/visualizations/dtw/opioid_ed/25_44/plots/trajectory_overview_plot.json`
- On EC2, `PROJECT_ROOT` is usually the repo root (e.g. `/home/pgx3874/pgx-analysis`).

**When is it created?** Only when the DTW **plots** step runs: `create_dtw_plots` (called from `create_dtw_visuals.py` or notebook 4). That step requires the DTW features CSV at `10_risk_dashboard/visualizations/dtw/feature_engineering/dtw_features_{cohort}_{age_band_fname}.csv`, plus Plotly and sklearn. If the pipeline was skipped for that cohort/age_band (or failed before the plot step), the file will not exist. Step 6 then syncs this directory to S3 so the dashboard can load it.

## Frontend pattern

**JSON where able for frontend to render with Plotly; PNG only as fallback.**  
The frontend fetches `trajectory_overview_plot.json` from the static plots path when available and renders it with Plotly. Overview/sample PNG URLs are used only when the API returns them (i.e. when those objects exist on S3). The frontend does not synthesize PNG URLs, so missing PNGs do not cause 404s.

## Manifest (dashboard_visual_objects.json)

- **File:** `10_risk_dashboard/visualizations/dashboard_visual_objects.json`
- **Tab:** DTW Trajectories  
- **s3_path:** `vcu/pgx-risk-calculator/visualizations/dtw/{cohort}/{age_band}/`  
- **static_files:** `["chart_data.json", "sequence_heatmap.json", "plots/trajectory_overview_plot.json", ...]`  

So the **base path** for DTW is `.../visualizations/dtw/{cohort}/{age_band}/` (e.g. `opioid_ed/25-44` with hyphen).  
All URLs for chart_data, sequence_heatmap, and `plots/` are built from this base.

**chart_data.json** is expected to contain (when the pipeline produced data): `routine_comparison`, `routine_comparison_counts`, `routine_by_medical_utilization`, `high_risk_trajectories`, `target_pathway_patterns`, `times_between_sequences`, `time_to_target_sequences`, and optionally `*_by_density` and `event_density_bins`. If the file is missing on S3 or contains only `empty: true` and a `message`, the Routine vs Utilization and related panels will show the placeholder.

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
- **plots/trajectory_overview_plot.json**: Optional. Present where cluster plots exist (create_dtw_plots writes it; Step 6 syncs). If missing, the frontend may see a 404 for the static URL; it then requests the DTW API and uses `trajectory_overview_plot` from the response when Lambda has it (inline from S3 when &lt; 2MB).
- **plots/*.html**: Present as `dtw_trajectory_cluster_1d_*` or `dtw_trajectory_cluster_3d_*`; API updated to use these for `overview_interactive` when `dtw_trajectory_cluster_interactive_*` is missing.
- **plots/dtw_trajectory_analysis_*.png**, **plots/dtw_sample_trajectories_*.png**: Optional; created by pipeline only when `dtw_trajectory_cluster_*.png` exists (kaleido). If missing, dashboard uses `trajectory_overview_plot.json` (Plotly) or shows empty.

## Why "Routine vs Utilization" or "routine_comparison_counts" don't render

The **Routine vs Utilization (Outcomes)** and **Medical and prescription event counts** panels need `routine_comparison` and `routine_comparison_counts` inside `chart_data.json`. If you see placeholders:

1. **Check that chart_data.json exists on S3** (dashboard bucket, under prefix):
   ```bash
   # Set your bucket and prefix, e.g.:
   BUCKET=jerome-dixon.io
   PREFIX=vcu/pgx-risk-calculator
   COHORT=opioid_ed
   AGE=65-74

   aws s3 ls "s3://${BUCKET}/${PREFIX}/visualizations/dtw/${COHORT}/${AGE}/"
   aws s3 cp "s3://${BUCKET}/${PREFIX}/visualizations/dtw/${COHORT}/${AGE}/chart_data.json" - | head -c 500
   ```
   - If the object does not exist: run the DTW pipeline for that cohort/age_band (`create_dtw_trajectories` → `create_dtw_features` → `create_dtw_visuals`) and sync to S3 (notebook 5 Step 6 or your sync script).
   - If the object exists but contains `"empty": true` and a `message`: the pipeline wrote an empty-state JSON (e.g. no DTW features CSV, or CSV had too few rows / missing `admin_icd_event_count`). Re-run the pipeline from trajectories through visuals so a full `chart_data.json` is produced.

2. **Manifest**: `dashboard_visual_objects.json` lists DTW under tab "DTW Trajectories" with `static_files`: `["chart_data.json", "sequence_heatmap.json", "plots/trajectory_overview_plot.json", ...]`. The frontend and API both use `visualizations/dtw/{cohort}/{age_band}/chart_data.json` (age_band with **hyphen**, e.g. `65-74`).

3. **Pipeline conditions for routine_comparison**: `create_dtw_visuals` builds `routine_comparison` only when the DTW features CSV has `target` and `admin_icd_event_count` and at least 10 rows. It builds `routine_comparison_counts` when the CSV has `admin_icd_event_count`, `trajectory_length`, and `seq_pattern_str`. Ensure `create_dtw_trajectories` has run so the CSV includes `admin_icd_event_count` (from the administrative ICD lookup).

## Summary

- Manifest and API use the same base path: `.../visualizations/dtw/{cohort}/{age_band}/`.  
- Root-level files (chart_data, sequence_heatmap) match manifest.  
- Plots subfolder and HTML naming are aligned with the pipeline; API accepts actual 1d/3d HTML names.
