---
description: Prefer JSON over PNG/JPEG for dashboard visuals where able (BupaR, DTW, feature importance)
globs: 10_risk_dashboard/**/*.*, 9_dashboard_visuals/**/*.py
alwaysApply: false
---

# Dashboard visuals: prefer JSON over static images

When building or changing dashboard visualizations (BupaR, DTW, feature importance, FP-Growth, etc.):

- **Prefer JSON over PNG/JPEG** where the pipeline can provide data as JSON. The frontend can then render with Chart.js, Plotly, or custom components (filters, tooltips, responsive layout) instead of a fixed image.
- **Where JSON is available:** Use it. Examples: BupaR activity frequency (`*_activity_frequency.json`, `*_pre_target_activity_frequency.json`, `*_post_target_activity_frequency.json`) → Chart.js bar charts; DTW `chart_data.json` and `sequence_heatmap.json` → Plotly; feature importance `aggregated_fi_heatmap.json` → Plotly heatmap.
- **Where only PNG/HTML exist:** Use PNG or interactive HTML; add JSON export in the pipeline when feasible so future work can switch to JSON.
- **API/Lambda:** Return JSON or URLs to JSON when available; document “prefer JSON” in docstrings. Frontend should consume JSON first and fall back to image URL only when JSON is missing.

See: `9_dashboard_visuals/bupar/README_bupaR.md` (§ JSON vs PNG/JPEG), `9_dashboard_visuals/dtw/README_DTW_COHORT_ANALYSIS.md` (§ JSON vs PNG/JPEG).
