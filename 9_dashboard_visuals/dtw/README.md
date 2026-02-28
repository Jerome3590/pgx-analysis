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
