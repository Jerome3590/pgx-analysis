# Feature Importance (dashboard tab)

## Research questions this visual answers

- **Which features (drugs, diagnoses, procedures) does the model use most to predict the target?** Aggregated feature-importance heatmaps show ranked importance across age bands and cohorts, so we see what is driving risk in our target cohorts.
- **How does importance vary by age band within a cohort?** Per-cohort heatmaps (feature × age band) reveal which codes matter in pediatric vs older age bands, so we can tailor interpretation and downstream visuals (BupaR, DTW, FP-Growth) to the right signals.
- **Which features matter across both cohorts (opioid-ED vs non–opioid-ED)?** The combined-cohort heatmap highlights shared vs cohort-specific drivers, reducing noise and clarifying what is common versus cohort-specific.

**Role in the pipeline:** This visual is the **canonical “what drives the model”** view. The same feature-importance outputs (and filtered code sets) drive **BupaR**, **DTW**, and **FP-Growth**: we use only important features in those visuals to reduce noise and understand what is driving our target cohorts.

### How features are filtered by feature importance and used downstream

- **Source:** Step 3b `cohort_feature_importance` (per cohort/age_band) is the single source. The pipeline builds an **allowed-codes** set (top-N drug, ICD, CPT) from that CSV.
- **BupaR:** Allowed codes are written to `allowed_codes_shap_ffa_{cohort}_{age_band}.json`. R scripts filter event logs so only activities in that set are kept; process matrices and activity frequency then show only important-feature pathways.
- **DTW:** The same allowed-codes list is used to filter model_events before building trajectories; only events whose drug/ICD/CPT is in the set are kept, so DTW archetypes reflect important-feature sequences only.
- **FP-Growth:** Allowed items from Step 3b (same or separate helper) restrict the transaction set; itemsets and rules are mined only over those items, so the dashboard shows co-occurrence among model-important codes only.

See **[../README.md#how-features-are-filtered-by-feature-importance-and-used-downstream](../README.md#how-features-are-filtered-by-feature-importance-and-used-downstream)** in the parent README for implementation details and file paths.

---

## Where this visual is produced

This folder exists so **9_dashboard_visuals** has one folder per dashboard visualization type, matching **10_risk_dashboard/visualizations**.

The **Feature Importance** tab data is **not produced here**. It is produced by:

- **Step 3a / heatmap code:** `py_helpers/feature_importance_heatmap.py` (e.g. `create_aggregated_fi_heatmap`, `create_combined_cohorts_fi_heatmap`), using Step 3b cohort feature importance CSVs.
- **Output (source):** `3a_feature_importance/{cohort}/plots/` (per-cohort heatmaps + JSON) and `3a_feature_importance/plots/` and `3a_feature_importance/combined/` (combined heatmap + JSON).
- **Copy:** Notebook **4_dashboard_visuals** copies these files to `10_risk_dashboard/visualizations/feature_importance/` (per cohort and combined) so deploy (notebook 5 Step 6) syncs from the same location as other visuals.

The dashboard frontend loads heatmap PNGs and JSON from `visualizations/feature_importance/` (static-first or Lambda API).

**See also:** [README_dashboard_visual_artifact_paths.md](../../10_risk_dashboard/docs/README_dashboard_visual_artifact_paths.md) (Feature Importance row).
