# Dashboard tabs ↔ data sources

Mapping of each dashboard tab to the **EC2 folder path** (where artifacts are deployed from), the API endpoint, and the S3 path. All paths below are relative to **repo root on EC2** (e.g. `/home/pgx3874/pgx-analysis`). S3 prefix is `vcu/pgx-risk-calculator/` (or `S3_DASHBOARD_PREFIX`); bucket is `jerome-dixon.io` (or `S3_DASHBOARD_BUCKET`).

**We only save and use artifacts tied to research questions.** See **[RESEARCH_QUESTIONS_ARTIFACTS.md](RESEARCH_QUESTIONS_ARTIFACTS.md)** for the canonical RQ → tab → artifact list. Unused artifacts are in [ARCHIVED_ARTIFACTS_NO_LONGER_USED.md](ARCHIVED_ARTIFACTS_NO_LONGER_USED.md). **Visual → artifact → EC2/S3 paths:** [README_dashboard_visual_artifact_paths.md](README_dashboard_visual_artifact_paths.md).

**Data pattern:** We use **JSON as much as possible** for visuals: pipeline exports JSON → Lambda returns inline when present → frontend renders from JSON (Plotly/chart) with fallback to image/iframe. **Exception:** FP-Growth network and PGx Cohort network are processed on EC2 and served as **HTML only** (no JSON). See [VISUALIZATION_DATA_PATTERN.md](VISUALIZATION_DATA_PATTERN.md).

## S3 URL format for assets (required)

Dashboard **HTML** (iframes) and **images** must use **path-style** S3 URLs. Do not use virtual-hosted style (`bucket.s3.region.amazonaws.com`).

**Template:**

```
https://s3.{region}.amazonaws.com/{bucket}/{prefix}/{object_key}
```

**Example (PGx Cohort network):**

- `https://s3.us-east-1.amazonaws.com/jerome-dixon.io/vcu/pgx-risk-calculator/cohort_pgx/networks/non_opioid_ed/55_64/network_topology.html`

**Example (image):**

- `https://s3.us-east-1.amazonaws.com/jerome-dixon.io/vcu/pgx-risk-calculator/feature_importance/opioid_ed/aggregated_fi_heatmap.png`

**Why:** The Lambda returns URLs in this form (`_dashboard_s3_url(key)` in `lambda_function.py`). The frontend uses these URLs in `iframe.src` and `<img src>`. Path-style is required for correct resolution and CORS behavior with the dashboard bucket.

| Tab | EC2 folder (deployed from) | API / data source | S3 path (dashboard bucket) | When uploaded |
|-----|----------------------------|-------------------|----------------------------|----------------|
| **Risk Assessment** | 10_risk_dashboard/outputs/models/, outputs/metadata/ (→ Lambda image) | POST /risk, GET /metadata | — (Lambda + container) | Deploy Lambda |
| **Drugs** | 10_risk_dashboard/outputs/metadata/ | GET /metadata?cohort= | metadata/opioid_ed.json, metadata/non_opioid_ed.json | Step 6 |
| **ICD Codes** | 10_risk_dashboard/outputs/metadata/ | (same) | (same) | Step 6 |
| **CPT Codes** | 10_risk_dashboard/outputs/metadata/ | (same) | (same) | Step 6 |
| **PGx Card** | 10_risk_dashboard/outputs/cpic/ (→ Lambda image) | POST /pgx/card | — (Lambda + CPIC in container) | Deploy Lambda |
| **Documentation** | 10_risk_dashboard/outputs/metadata/model_performance_metrics.json | Same-origin JSON | metadata/model_performance_metrics.json | Step 6 |
| **Feature Importance** | 3a_feature_importance/outputs/{cohort}/plots/, 3a_feature_importance/outputs/plots/ | GET /visualizations/feature_importance?cohort= | feature_importance/{cohort}/aggregated_fi_heatmap.png\|.json, feature_importance/combined_cohorts_*.png | Step 6 |
| **Causal Analysis** | 10_risk_dashboard/outputs/{cohort}/{age_band_fname}/dashboard_data.json | GET /visualizations/causal?cohort=&age_band= | causal/{cohort}/{age_band_fname}/causal_data.json | Step 6 (upload_causal_outputs_to_s3.py) |
| **BupaR Process Mining** | 10_risk_dashboard/visualizations/bupar/outputs/{cohort}/{age_band_fname}/plots/ | GET /visualizations/bupar, /bupar/activity_frequency, /bupar/html | bupar/{cohort}/{age_band}/plots/ | 4_dashboard_visuals (BupaR) |
| **DTW Trajectories** | 10_risk_dashboard/visualizations/dtw/outputs/{cohort}/{age_band_fname}/ (chart_data.json, sequence_heatmap.json, plots/) | GET /visualizations/dtw?cohort=&age_band= | dtw/{cohort}/{age_band}/chart_data.json, sequence_heatmap.json, plots/ | 4_dashboard_visuals (DTW) |
| **FP-Growth Patterns** | 10_risk_dashboard/visualizations/fpgrowth/outputs/{cohort}/{age_band_fname}/plots/, .../outputs/.../ (itemsets JSON) | GET /visualizations/fpgrowth, /fpgrowth/network_html | fpgrowth/{cohort}/{age_band}/plots/, .../data/*.json | 4_dashboard_visuals (FP-Growth) |
| **PGx Cohort** | 10_risk_dashboard/visualizations/cohort_pgx/networks/{cohort}/{age_band_fname}/ | GET /visualizations/cohort_pgx?cohort=&age_band= | cohort_pgx/networks/{cohort}/{age_band_fname}/network_topology.html | Step 6 (Cohort PGx sync) |

**Frontend (HTML/JS):** EC2 path `10_risk_dashboard/frontend/` → S3 prefix root (Step 6 sync).

**Step 6** = 5_build_and_deploy.ipynb "Step 6: Sync Dashboard Frontend to S3" (frontend sync + metadata + metrics + feature importance heatmaps + Cohort PGx sync + causal upload).

**4_dashboard_visuals** = BupaR, DTW, and FP-Growth pipelines write to the EC2 paths above and upload to the dashboard bucket when run; Step 6 does not re-upload those.

**Research focus (BupaR, DTW, FP-Growth):** Final production pipeline produces **drug-only** artifacts for BupaR/FP-Growth and drug sequence heatmap for DTW. **Routine vs no routine (admin codes)** remains a core analysis on the DTW tab: outcome rate by routine screenings (admin ICD) to highlight how routine care may reduce extreme outcomes; data comes from `chart_data.json` (`routine_comparison`).
