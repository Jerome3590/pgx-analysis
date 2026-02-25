# DTW visualization outputs

This directory contains **outputs only** for DTW dashboard visualizations (features, plots, chart_data).  
**Creation code** lives in `9_dashboard_visuals/dtw/` (pipeline step 9). Do not add scripts here; follow the same pattern as `6_final_model`, `7_shap_analysis`, and `8_ffa_analysis` (code in step folder, outputs in designated output location).

## Pipeline outputs and dashboard consumption

| Location | Contents | Dashboard use |
|----------|----------|----------------|
| `outputs/feature_engineering/` | `dtw_features_*.csv`, `common_sequences_*.json`, `trajectory_status_*.json` | **Inputs** to `create_dtw_visuals`; not served directly. |
| `outputs/{cohort}/{age_band_fname}/plots/` | `dtw_trajectory_analysis_*.png`, `dtw_sample_trajectories_*.png`, interactive HTML | Uploaded to S3 `dtw/{cohort}/{age_band}/plots/`; overview and sample images. |
| S3 `dtw/{cohort}/{age_band}/chart_data.json` | routine_comparison, high_risk_trajectories, times_between_sequences, time_to_target_sequences, target_pathway_patterns | **JSON preferred.** Lambda may return inline. **routine_comparison** is the core analysis: outcome rate by routine vs no routine (admin ICD), highlighting how routine screenings may reduce extreme outcomes. Frontend uses for Routine vs No Routine, N3, Target Pathway, Trajectory Metrics. |
| S3 `dtw/{cohort}/{age_band}/sequence_heatmap.json` | drug codes × position counts | **JSON preferred.** Common Sequences Heatmap (drug slice only). |

**Event density:** When the trajectory CSV has `event_density_bin` (from `create_dtw_trajectories.py`), chart_data.json also includes `event_density_bins` and per-bin series: `routine_comparison_by_density`, `routine_comparison_counts_by_density`, `high_risk_trajectories_by_density`. The DTW tab provides an **Event density** dropdown (All | Low | Medium | High | Extreme) to filter those three charts by trajectory events-per-month bin; "All" uses the aggregate series. Trajectory CSV columns: temporal_span_days, events_per_month, event_density_bin.

**Run order:** `create_dtw_features` → writes CSV + common_sequences in feature_engineering. Then **`create_dtw_visuals`** reads that CSV, builds chart_data and sequence_heatmap, creates plots, and uploads chart_data.json, sequence_heatmap.json, and plots to the dashboard S3 prefix. If create_dtw_visuals is not run, the DTW tab will show “no data” for metrics/routine/N3/heatmap and “not available” for overview/sample images.

**JSON vs PNG:** Prefer JSON where available. Lambda fetches chart_data.json and sequence_heatmap.json from S3 and returns them inline when present; frontend uses inline data first, then falls back to fetching chart_data_url / sequence_heatmap_url. Overview and sample visuals are PNG/HTML only; image URLs are returned only when the objects exist on S3.

## Checking DTW logs when artifacts are missing

If the artifact path check reports missing `chart_data.json` or `sequence_heatmap.json` for some cohort/age_band:

1. **Find DTW visuals logs** for that combo, e.g. step logs under `9_dashboard_visuals` or the output of `run_dashboard_visuals.py` / `create_dtw_visuals.py`. Search for the cohort and age_band (e.g. `opioid_ed` and `13-24`).

2. **Common log messages:**
   - `DTW features not found: ...; skipping` → **create_dtw_features** did not produce the CSV for that combo. Run `create_dtw_features` for that cohort/age_band first.
   - `DTW artifacts exist at ...; skipping` → Artifacts were present when the run started. Use `--force` to re-run and overwrite.
   - `DTW chart_data not produced for X/Y: empty dataframe` → CSV exists but is empty; check feature_engineering outputs.
   - `DTW chart_data empty for X/Y: no routine_comparison, high_risk, or N3 data` → CSV has rows but required columns or row counts failed (e.g. need `admin_icd_event_count`, `target`, `seq_pattern_str`; or too few rows).
   - `DTW sequence_heatmap not produced for X/Y: empty dataframe or missing seq_pattern_str` → Need non-empty `seq_pattern_str` in the CSV.

3. **Re-run missing combos:** Run `create_dtw_visuals.py --project-root <repo_root> --cohort-name <cohort> --age-band <age_band>` for each missing combo (optionally with `--force` if you want to regenerate existing files). The script now writes `chart_data.json` and `sequence_heatmap.json` even when chart_data is empty (`{}`), so the artifact path check can pass; the dashboard will show an empty state for that combo.
