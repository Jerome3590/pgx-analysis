# Causal Analysis (dashboard tab)

## Research questions this visual answers

- **Which features (drugs, diagnoses, procedures) does the model treat as causally important for the target?** The Causal tab surfaces the combined **SHAP + FFA (Functional Feature Attribution)** importance: both local (SHAP) and global/causal (FFA) contributions, merged per feature.
- **How does importance vary by cohort and age band?** Side-by-side or filtered views let you compare which codes matter for opioid-ED vs non–opioid-ED, and across age bands, so we can see what is driving each target cohort.
- **Where should clinicians and researchers look first?** Ranked importance (and optional visual encoding) highlights the top drivers of risk, reducing noise and focusing interpretation on model-relevant signals.

**Data source:** Combined SHAP + FFA importance from Steps 7 and 8; the visual does not use raw feature importance alone—it uses the causal/functional view for interpretability.

---

## Where this visual is produced

This folder exists so **9_dashboard_visuals** has one folder per dashboard visualization type, matching **10_risk_dashboard/visualizations**.

The **Causal Analysis** tab data is **not produced here**. It is produced by:

- **Script:** `10_risk_dashboard/data_preparation/combine_shap_ffa_results.py` (SHAP + FFA combined importance)
- **Output:** `10_risk_dashboard/visualizations/causal/{cohort}/{age_band_fname}/dashboard_data.json`
- **Upload:** `10_risk_dashboard/data_preparation/upload_causal_outputs_to_s3.py` (or notebook 5 Step 6) uploads as `visualizations/causal/{cohort}/{age_band}/causal_data.json`

Notebook **4_dashboard_visuals** runs the combine step and upload; the dashboard frontend loads `causal_data.json` (static-first) or the Lambda API.

**See also:** [README_dashboard_visual_artifact_paths.md](../../10_risk_dashboard/docs/README_dashboard_visual_artifact_paths.md) (Causal Analysis row).
