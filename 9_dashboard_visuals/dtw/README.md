# DTW Trajectories (dashboard tab)

## Research questions this visual answers

- **What are the typical care trajectories leading to the target event?** DTW clusters time-ordered sequences of diagnoses, procedures, and drugs in the lookback window before the anchor (e.g. first opioid-ED or first non-opioid ED encounter), so we see the main “on-ramp” pathways.
- **How do trajectory archetypes differ by event density and outcome?** Chart data and plots support filters by event density (low/medium/high/extreme) and by outcome (e.g. routine vs no routine, high-risk vs low-risk), so we can compare pathways across utilization and risk.
- **Which trajectories are most predictive of the target?** By building trajectories only from **feature-important** codes, we reduce noise and focus on what the model uses—so the visual reflects what is actually driving our target cohorts.

**Feature importance drives this visual.** We use only **SHAP/FFA important** codes (drug, ICD, CPT) when constructing trajectories. Events whose codes are not in that set are excluded. That keeps DTW aligned with the risk model and makes pathway archetypes interpretable.

### How features are filtered and used downstream

- The pipeline builds an **allowed-codes** set per (cohort, age_band) from Step 3b cohort feature importance (see [../README.md#how-features-are-filtered-by-feature-importance-and-used-downstream](../README.md#how-features-are-filtered-by-feature-importance-and-used-downstream)).
- DTW **trajectory construction** (Python) filters model_events before building sequences: only events whose drug, ICD, or CPT is in the allowed set are kept; all other events are dropped.
- Clustering and archetypes are then computed on these filtered trajectories, so the dashboard shows pathway types that reflect only model-important features.

---

## Pipeline and technical details

- **Pipeline:** Filter by feature importance → build cohort trajectories (target-anchored, lookback) → DTW (distances, clustering, archetypes) → dashboard chart_data and plots. See [README_DTW_COHORT_ANALYSIS.md](README_DTW_COHORT_ANALYSIS.md) for full cohort DTW analysis and [README_DTW_S3_CHECKPOINTS.md](README_DTW_S3_CHECKPOINTS.md) for S3 checkpoint behavior.
- **Outputs:** `10_risk_dashboard/visualizations/dtw/{cohort}/{age_band_fname}/` (chart_data.json, sequence_heatmap, plots/). Notebook 5 Step 6 syncs to S3.

### Files the DTW visual code creates (and when they’re missing)

| File | Created by | When it exists | When it’s missing |
|------|------------|----------------|-------------------|
| `plots/trajectory_overview_plot.json` | `create_dtw_plots.create_trajectory_cluster_plots()` | DTW features CSV exists, has `seq_pattern_str`, and code counts from sequences are non-empty; Plotly/sklearn available. | No DTW features CSV for that cohort/age; CSV missing `seq_pattern_str`; `count_df` empty (no valid code counts); fewer than required axes (e.g. &lt; 3 codes for 3D); Plotly or sklearn not installed. |
| `plots/dtw_trajectory_cluster_*.html` | Same | Same conditions as above. | Same as above. |
| `plots/dtw_trajectory_cluster_*.png` | Same (optional) | Same as above **and** `fig.write_image()` succeeds (requires **kaleido**). | Kaleido not installed or write_image fails (often the case). |
| `plots/dtw_trajectory_analysis_{cohort}_{age}.png` | `create_dtw_visuals` (copy) | Only if `plots/dtw_trajectory_cluster_*.png` exists. | No cluster PNG was written (kaleido usually not available). |
| `plots/dtw_sample_trajectories_{cohort}_{age}.png` | Same (copy) | Same as above. | Same as above. |

So **the code does create** `trajectory_overview_plot.json` and the two PNGs when the pipeline runs successfully and (for the PNGs) kaleido works. For cohort/age bands where you see 404s (e.g. opioid_ed/25-44), either the trajectory cluster step was skipped (no or invalid DTW features CSV, or empty code counts), or the plots were never synced. Re-run the DTW step for that cohort/age after ensuring `create_dtw_trajectories` and `create_dtw_features` have produced the features CSV under `10_risk_dashboard/visualizations/dtw/feature_engineering/`.
