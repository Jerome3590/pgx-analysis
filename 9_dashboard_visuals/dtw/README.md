# DTW Trajectories (dashboard tab)

## Research questions this visual answers

- **What are the typical care trajectories leading to the target event?** DTW clusters time-ordered sequences of diagnoses, procedures, and drugs in the lookback window before the anchor (e.g. first opioid-ED or first non-opioid ED encounter), so we see the main “on-ramp” pathways.
- **How do trajectory archetypes differ by event density and outcome?** Chart data and plots support filters by event density (low/medium/high/extreme) and by outcome (e.g. routine vs utilization, high-risk vs low-risk), so we can compare pathways across utilization and risk.
- **Which trajectories are most predictive of the target?** By building trajectories only from **feature-important** codes, we reduce noise and focus on what the model uses—so the visual reflects what is actually driving our target cohorts.

**Feature importance and drug-only.** We use only **SHAP/FFA important drug** codes when constructing trajectories. **DTW is drug-only for both cohorts** (opioid_ed and non_opioid_ed): only prescription (drug) events in the allowed set are included; ICD and CPT are excluded. That keeps DTW focused on drug pathways and aligns with drug-sequence research questions.

### How features are filtered and used downstream

- The pipeline builds an **allowed-codes** set per (cohort, age_band) from Step 3b cohort feature importance (see [../README.md#how-features-are-filtered-by-feature-importance-and-used-downstream](../README.md#how-features-are-filtered-by-feature-importance-and-used-downstream)).
- DTW **trajectory construction** (Python) filters model_events to **drug transactions only**: only events whose drug is in the SHAP/FFA allowed drug set are kept; ICD and CPT events are excluded for both cohorts. **Routine vs utilization** (N1) uses **administrative ICD codes to identify routine appointments** and **medical utilization** (medical events per patient) to bin patients: the lookup `1b_apcd_event_filter/administrative_codes_lookup.json` lists ICD codes for routine care (e.g. well visits, screenings). We count how many such events each patient has in the same time window as trajectories; 1+ = routine appointments, 0 = no routine. We also compute `medical_event_count_full` and bin it as `medical_utilization_bin` (low/medium/high). That drives the dashboard’s “Routine vs Utilization” sub-tab: outcome rate by routine vs utilization, mean prescription and medical events per patient, and a **Routine × medical utilization** chart (`routine_by_medical_utilization`) showing outcome rate by routine and utilization bin. N3 breakdowns (times between, time to target) are by routine bucket.
- Clustering and archetypes are then computed on these filtered trajectories, so the dashboard shows pathway types that reflect only model-important features.

---

## Pipeline and technical details

- **Pipeline:** Filter by feature importance → build cohort trajectories (target-anchored, lookback) → DTW (distances, clustering, archetypes) → dashboard chart_data and plots. For large cohorts (e.g. 65-74, 75-84), the trajectory cluster plot is built from a **subsample** (cap 25k rows) to avoid OOM; chart_data and dashboard tables use the full data. See [README_DTW_COHORT_ANALYSIS.md](README_DTW_COHORT_ANALYSIS.md) and [README_DTW_S3_CHECKPOINTS.md](README_DTW_S3_CHECKPOINTS.md).
- **Data format:** Trajectories and DTW features are written as **Parquet** (primary) and **CSV** (backward compatibility). We use **Parquet and DuckDB for I/O and transformations whenever possible**: downstream steps (create_dtw_features, create_dtw_visuals, create_dtw_plots) prefer Parquet and use DuckDB to read parquet; create_dtw_visuals runs N3 aggregations (times between / time to target) in DuckDB when available; they fall back to pandas/CSV otherwise. Model events are read from parquet via DuckDB in create_dtw_trajectories. Extreme-density and predictive-time outputs also write Parquet alongside CSV.
- **Outputs:** `10_risk_dashboard/visualizations/dtw/{cohort}/{age_band_fname}/` (chart_data.json, sequence_heatmap, plots/). Notebook 5 Step 6 syncs to S3.

**No empty artifacts.** When a plot or chart doesn’t produce data, the pipeline **always** writes a JSON artifact with `message`, `empty: true`, `cohort`, `age_band`, and `metrics` (e.g. `reason`, `dtw_rows`) so the dashboard can show why there is no output. Never leave a missing file or plain `{}`. Applies to `chart_data.json`, `sequence_heatmap.json`, and `plots/trajectory_overview_plot.json`. See [10_risk_dashboard/docs/README_dashboard_visual_artifact_paths.md](../../10_risk_dashboard/docs/README_dashboard_visual_artifact_paths.md#dtw-trajectories) for the full DTW EC2 + S3 path table.

### chart_data.json: parameters and what they tell us

The pipeline writes a **robust** `chart_data.json` so multiple visuals (dashboard, reports, API consumers) can use the same structure.

| Section / key | Type | What it tells us |
|---------------|------|------------------|
| **summary** | object | Cohort-level counts and stats (always present). |
| summary.total_trajectories | number | Number of drug-only trajectories (one per patient with ≥1 drug event). |
| summary.trajectories_with_time_between | number | Trajectories with ≥2 drug events, so mean_days_between_events is defined (used for N3 “times between”). |
| summary.trajectories_target1_with_time_to_target | number | Target=1 trajectories with valid days_first_event_to_target (used for N3 “time to target”). |
| summary.trajectory_length | { min, max, mean, median } | Drug events per trajectory: spread and center. |
| summary.has_dtw_distances | boolean | Whether DTW alignment was run (dtw_min_distance present). |
| summary.target_counts | { target_1, target_0 } | Case/control split. |
| **routine_comparison** | chart | Outcome rate by routine vs utilization (admin ICD). **n**: sample size per bucket. |
| **routine_comparison_counts** | chart | Mean medical and prescription events per patient by routine vs utilization. **n**: per bucket. |
| **routine_by_medical_utilization** | chart | Outcome rate by routine and medical utilization bin (Routine × utilization). **n**: per bucket. |
| **high_risk_trajectories** | chart | Outcome rate by trajectory archetype (Q1–Q4 by DTW distance or length). **n**: per quartile. |
| **times_between_sequences** | chart | N3: mean days between consecutive drug events, by routine vs utilization. **n**: trajectories per bucket. |
| **time_to_target_sequences** | chart | N3: mean days from first drug event to target (target=1 only), by routine vs utilization. **n**: per bucket. |
| **target_pathway_patterns** | chart | Common codes in target=1 trajectories; metadata has total_target_patients, total_control_patients. |
| **metrics** | object | dtw_rows (same as summary.total_trajectories), charts_built, charts_not_built, success. |

Each bar chart object includes **x**, **y**, **type**, **name**, **x_label**, **y_label**, and when applicable **n** (array of sample sizes, same order as **x**) so visuals can show “n = …” or gate on reliability.

---

### Files the DTW visual code creates (and when they’re missing)

| File | Created by | When it exists | When it’s missing |
|------|------------|----------------|-------------------|
| `chart_data.json` | `create_dtw_visuals` | Full chart data (routine_comparison, routine_comparison_counts, routine_by_medical_utilization, high_risk, N3, etc.) when DTW dataframe has data. | **Always written.** If no data, empty-state JSON: `message`, `empty: true`, `cohort`, `age_band`, `metrics` (e.g. `reason`, `dtw_rows`, `has_admin_icd`, `has_target`). |
| `sequence_heatmap.json` | `create_dtw_visuals` | Full heatmap (drug/icd/cpt codes, positions, counts) when sequences exist. | **Always written.** If no data or no code counts, empty-state JSON: `message`, `empty: true`, `cohort`, `age_band`, `metrics` (e.g. `reason`, `dtw_rows`). |
| `plots/trajectory_overview_plot.json` | `create_dtw_plots.create_trajectory_cluster_plots()` | **Always written.** Full Plotly payload when features exist and Plotly/sklearn available. | When visual is skipped, same function writes empty-state JSON: `message` (includes cohort/age and reason), `empty: true`, `cohort`, `age_band`, and `metrics` (e.g. `reason`, `dtw_rows`, `count_df_rows`, `n_axes_required`, `csv_path`) so the dashboard shows why no output. |
| `plots/dtw_trajectory_cluster_*.html` | Same | Same conditions as above. | Same as above. |
| `plots/dtw_trajectory_cluster_*.png` | Same (optional) | Same as above **and** `fig.write_image()` succeeds (requires **kaleido**). | Kaleido not installed or write_image fails (often the case). |
| `plots/dtw_trajectory_analysis_{cohort}_{age}.png` | `create_dtw_visuals` (copy) | Only if `plots/dtw_trajectory_cluster_*.png` exists. | No cluster PNG was written (kaleido usually not available). |
| `plots/dtw_sample_trajectories_{cohort}_{age}.png` | Same (copy) | Same as above. | Same as above. |

The pipeline **always** writes `chart_data.json`, `sequence_heatmap.json`, and `trajectory_overview_plot.json` (full payload or empty-state JSON with message + metrics). Step 6 syncs them so the dashboard never gets 404. HTML and PNGs are only written when the visual is produced (kaleido required for PNGs). Full EC2 + S3 path table: [README_dashboard_visual_artifact_paths.md § DTW Trajectories](../../10_risk_dashboard/docs/README_dashboard_visual_artifact_paths.md#dtw-trajectories).

---

### Troubleshooting: "No DTW chart data for {cohort}/{age_band}"

1. **Check create_dtw_visuals logs**  
   Local: `logs/dtw_s3/{cohort}/{age_band}/create_dtw_visuals_*.log` or `9_dashboard_visuals/logs/5_dtw/create_dtw_visuals_{cohort}_{age_band_fname}_*.log`.  
   Look for:
   - **"Loaded 0 patients"** → The DTW features CSV has **no data rows** (header only). Cause is upstream in `create_dtw_features` or `create_dtw_trajectories`.
   - **"DTW features not found"** → CSV missing; run `create_dtw_trajectories` then `create_dtw_features` for that cohort/age_band.
   - **"empty dataframe"** / **"charts_not_built"** → Same as 0 patients or rows dropped (e.g. all `seq_pattern_str` empty).

2. **If "Loaded 0 patients"**  
   The CSV at `10_risk_dashboard/visualizations/dtw/feature_engineering/dtw_features_{cohort}_{age_band_fname}.csv` exists but has zero data rows. Check:
   - **create_dtw_features** logs for that cohort/age_band: did it read any trajectories? Did DTW alignment produce no rows?
   - **create_dtw_trajectories** logs: did it extract any trajectories? If you see **"No trajectories with SHAP/FFA drug filter"** then the allowed drug codes (from `allowed_codes_shap_ffa_{cohort}_{age_band}.json`) did not match any `drug_name` in model_events (e.g. normalization or naming mismatch). The script **automatically retries using all drug events** (no SHAP/FFA filter) so you still get trajectories when patients have drug events; look for **"Fallback succeeded: N trajectories using all drug events"** in the same run. If fallback also yields 0, check model_events path and target/lookback (e.g. target date column, max lookback months).
   - **create_dtw_trajectories** output: does the trajectories CSV (or the input to create_dtw_features) have any rows for this cohort/age_band?

3. **Plots missing for some age bands (e.g. 65-74, 75-84) but chart_data exists**  
   Large cohorts can cause the trajectory cluster plot step to run out of memory (OOM). The pipeline **subsamples to 25,000 rows** when building the cluster plot (chart_data and dashboard still use the full CSV). If you see **"DTW trajectory cluster plots failed (MemoryError)"** in create_dtw_visuals logs, either re-run (subsampling should now avoid OOM) or reduce `MAX_PLOT_ROWS` in `create_dtw_plots.py` or increase process memory.

4. **S3 logs**  
   If the pipeline runs on EC2 and uploads logs to S3, see [README_DTW_S3_CHECKPOINTS.md](README_DTW_S3_CHECKPOINTS.md) for log locations (e.g. `s3://pgx-repository/5_dtw_log/{cohort}/{age_band}/`).
