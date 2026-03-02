# DTW Trajectories (dashboard tab)

## Research questions this visual answers

- **What are the typical care trajectories leading to the target event?** DTW clusters time-ordered sequences of diagnoses, procedures, and drugs in the lookback window before the anchor (e.g. first opioid-ED or first non-opioid ED encounter), so we see the main “on-ramp” pathways.
- **How do trajectory archetypes differ by event density and outcome?** Chart data and plots support filters by event density (low/medium/high/extreme) and by outcome (e.g. routine vs no routine, high-risk vs low-risk), so we can compare pathways across utilization and risk.
- **Which trajectories are most predictive of the target?** By building trajectories only from **feature-important** codes, we reduce noise and focus on what the model uses—so the visual reflects what is actually driving our target cohorts.

**Feature importance and drug-only.** We use only **SHAP/FFA important drug** codes when constructing trajectories. **DTW is drug-only for both cohorts** (opioid_ed and non_opioid_ed): only prescription (drug) events in the allowed set are included; ICD and CPT are excluded. That keeps DTW focused on drug pathways and aligns with drug-sequence research questions.

### How features are filtered and used downstream

- The pipeline builds an **allowed-codes** set per (cohort, age_band) from Step 3b cohort feature importance (see [../README.md#how-features-are-filtered-by-feature-importance-and-used-downstream](../README.md#how-features-are-filtered-by-feature-importance-and-used-downstream)).
- DTW **trajectory construction** (Python) filters model_events to **drug transactions only**: only events whose drug is in the SHAP/FFA allowed drug set are kept; ICD and CPT events are excluded for both cohorts. **Routine vs no routine** (N1) uses **administrative ICD codes to identify routine appointments**: the lookup `1b_apcd_event_filter/administrative_codes_lookup.json` lists ICD codes for routine care (e.g. well visits, screenings). We count how many such events each patient has in the same time window as trajectories; 1+ = routine appointments, 0 = no routine. That `admin_icd_event_count` drives the dashboard’s “Routine vs no routine (admin ICD filter)” charts and N3 breakdowns. The routine_comparison_counts chart (mean prescription and medical events per patient by routine vs no routine) shows whether routine care is associated with lower drug counts—i.e. routine care driving down prescription utilization.
- Clustering and archetypes are then computed on these filtered trajectories, so the dashboard shows pathway types that reflect only model-important features.

---

## Pipeline and technical details

- **Pipeline:** Filter by feature importance → build cohort trajectories (target-anchored, lookback) → DTW (distances, clustering, archetypes) → dashboard chart_data and plots. See [README_DTW_COHORT_ANALYSIS.md](README_DTW_COHORT_ANALYSIS.md) for full cohort DTW analysis and [README_DTW_S3_CHECKPOINTS.md](README_DTW_S3_CHECKPOINTS.md) for S3 checkpoint behavior.
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
| **routine_comparison** | chart | Outcome rate by routine vs no routine (admin ICD). **n**: sample size per bucket. |
| **routine_comparison_counts** | chart | Mean medical and prescription events per patient by routine vs no routine. **n**: per bucket. |
| **high_risk_trajectories** | chart | Outcome rate by trajectory archetype (Q1–Q4 by DTW distance or length). **n**: per quartile. |
| **times_between_sequences** | chart | N3: mean days between consecutive drug events, by routine vs no routine. **n**: trajectories per bucket. |
| **time_to_target_sequences** | chart | N3: mean days from first drug event to target (target=1 only), by routine vs no routine. **n**: per bucket. |
| **target_pathway_patterns** | chart | Common codes in target=1 trajectories; metadata has total_target_patients, total_control_patients. |
| **metrics** | object | dtw_rows (same as summary.total_trajectories), charts_built, charts_not_built, success. |

Each bar chart object includes **x**, **y**, **type**, **name**, **x_label**, **y_label**, and when applicable **n** (array of sample sizes, same order as **x**) so visuals can show “n = …” or gate on reliability.

---

### Files the DTW visual code creates (and when they’re missing)

| File | Created by | When it exists | When it’s missing |
|------|------------|----------------|-------------------|
| `chart_data.json` | `create_dtw_visuals` | Full chart data (routine_comparison, high_risk, N3, etc.) when DTW dataframe has data. | **Always written.** If no data, empty-state JSON: `message`, `empty: true`, `cohort`, `age_band`, `metrics` (e.g. `reason`, `dtw_rows`, `has_admin_icd`, `has_target`). |
| `sequence_heatmap.json` | `create_dtw_visuals` | Full heatmap (drug/icd/cpt codes, positions, counts) when sequences exist. | **Always written.** If no data or no code counts, empty-state JSON: `message`, `empty: true`, `cohort`, `age_band`, `metrics` (e.g. `reason`, `dtw_rows`). |
| `plots/trajectory_overview_plot.json` | `create_dtw_plots.create_trajectory_cluster_plots()` | **Always written.** Full Plotly payload when features exist and Plotly/sklearn available. | When visual is skipped, same function writes empty-state JSON: `message` (includes cohort/age and reason), `empty: true`, `cohort`, `age_band`, and `metrics` (e.g. `reason`, `dtw_rows`, `count_df_rows`, `n_axes_required`, `csv_path`) so the dashboard shows why no output. |
| `plots/dtw_trajectory_cluster_*.html` | Same | Same conditions as above. | Same as above. |
| `plots/dtw_trajectory_cluster_*.png` | Same (optional) | Same as above **and** `fig.write_image()` succeeds (requires **kaleido**). | Kaleido not installed or write_image fails (often the case). |
| `plots/dtw_trajectory_analysis_{cohort}_{age}.png` | `create_dtw_visuals` (copy) | Only if `plots/dtw_trajectory_cluster_*.png` exists. | No cluster PNG was written (kaleido usually not available). |
| `plots/dtw_sample_trajectories_{cohort}_{age}.png` | Same (copy) | Same as above. | Same as above. |

The pipeline **always** writes `chart_data.json`, `sequence_heatmap.json`, and `trajectory_overview_plot.json` (full payload or empty-state JSON with message + metrics). Step 6 syncs them so the dashboard never gets 404. HTML and PNGs are only written when the visual is produced (kaleido required for PNGs). Full EC2 + S3 path table: [README_dashboard_visual_artifact_paths.md § DTW Trajectories](../../10_risk_dashboard/docs/README_dashboard_visual_artifact_paths.md#dtw-trajectories).
