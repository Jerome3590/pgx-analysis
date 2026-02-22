# Test Plan: Final Dashboard

This document defines the test plan for the PGx Risk Calculator dashboard: **required artifacts**, **Lambda/API Gateway setup**, and **per-component automated checks**.

---

## 1. Required Artifacts

### 1.1 Container / Lambda bundle (deployed with Docker image)

| Artifact | Path | Purpose |
|----------|------|---------|
| Metadata (cohort dropdowns) | `/var/task/metadata/metadata_opioid_ed.json`, `metadata_non_opioid_ed.json` | GET /metadata |
| Model performance metrics | `/var/task/metadata/model_performance_metrics.json` | GET /metrics (fallback) |
| CPIC gene-drug pairs | `/var/task/data/cpic_gene-drug_pairs.xlsx` | POST /pgx/card |
| Ensemble models (per cohort/age_band) | `/var/task/models/{cohort}/{age_band_fname}/` | POST /risk, /risk/comparison |
| | `catboost.joblib`, `xgboost.joblib`, `xgboost.json`, `feature_schema.json` | |
| PGx interaction matrix (optional) | `/var/task/data/pgx_interaction_matrix.csv` | PGx card drug-gene lookup |

Cohorts: `opioid_ed`, `non_opioid_ed`. Age bands: `0-12`, `13-24`, `25-44`, `45-54`, `55-64`, `65-74`, `75-84`, `85-114` (age_band_fname: hyphen → underscore).

### 1.2 S3 – Results bucket (e.g. pgxdatalake)

| Prefix | Contents | Used by |
|--------|----------|---------|
| `gold/dashboard/metadata/` | `metadata_opioid_ed.json`, `metadata_non_opioid_ed.json`, `model_performance_metrics.json`, `cpic_gene-drug_pairs.xlsx` | Lambda fallback when not in container |
| `gold/dashboard/models/{cohort}/{age_band_fname}/` | Same model files as container | Lambda model load fallback |
| `gold/dashboard/data/` | CPIC, PGx data (if not in container) | PGx card |

### 1.3 S3 – Dashboard bucket (e.g. jerome-dixon.io)

Frontend and visualization assets. Prefix: `S3_DASHBOARD_PREFIX` (e.g. `vcu/pgx-risk-calculator`).

| Path pattern | Contents | Used by |
|--------------|----------|---------|
| `{prefix}/index.html` | Dashboard app | Static site |
| `{prefix}/metadata/opioid_ed.json`, `non_opioid_ed.json` | Cohort metadata (same-origin) | Frontend dropdowns |
| `{prefix}/metadata/model_performance_metrics.json` | Documentation tab | Frontend |
| `{prefix}/bupar/{cohort}/{age_band}/plots/` | BupaR PNGs + activity_frequency JSONs | GET /visualizations/bupar, /bupar/activity_frequency |
| `{prefix}/dtw/{cohort}/{age_band}/plots/`, `chart_data.json`, `sequence_heatmap.json` | DTW assets | GET /visualizations/dtw |
| `{prefix}/fpgrowth/{cohort}/{age_band}/plots/` | FP-Growth itemsets, network HTML/PNG | GET /visualizations/fpgrowth |
| `{prefix}/feature_importance/{cohort}/aggregated_fi_heatmap.png` | Feature importance heatmap | GET /visualizations/feature_importance |
| `{prefix}/feature_importance/combined_cohorts_feature_importance_heatmap.png` | Combined heatmap | GET /visualizations/feature_importance |
| `{prefix}/cohort_pgx/networks/{cohort}/{age_band_fname}/network_topology.html` | PGx Cohort network | GET /visualizations/cohort_pgx |

---

## 2. API Endpoints (Lambda + API Gateway)

All paths below are relative to the deployed stage (e.g. `/prod/...`). API Gateway should have either **{proxy+}** (any path → Lambda) or **explicit resources** (e.g. `/metadata`, `/risk`, `/visualizations/dtw`) with GET/POST/OPTIONS and Lambda proxy integration.

| Method | Path | Query / Body | Expected behavior |
|--------|------|--------------|-------------------|
| OPTIONS | (any) | — | 200, CORS headers |
| GET | `/metadata` | `cohort` (opioid_ed \| non_opioid_ed) | 200 + JSON (age_bands, codes) or 404 |
| GET | `/metrics` | — | 200 + model performance JSON or 404 |
| POST | `/risk` | Body: cohort, age_band (or age), drugs[], icds[], cpts[] | 200 + risk score/band or 4xx/5xx |
| POST | `/risk/comparison` | Body: base, scenarios[] | 200 + comparison or 4xx/5xx |
| POST | `/pgx/card` | Body: patient_id?, variants[] | 200 + PGx card or 4xx/5xx |
| POST | `/causal/importance` | Body: cohort, age_band, ... | 200 + importance or 4xx/5xx |
| POST | `/causal/interactions` | Body: cohort, age_band, ... | 200 + interactions or 4xx/5xx |
| GET | `/visualizations/causal` | cohort, age_band | 200 + causal/SHAP URLs or data |
| GET | `/visualizations/dtw` | cohort, age_band | 200 + DTW URLs (overview_image, chart_data_url, …) |
| GET | `/visualizations/fpgrowth` | cohort, age_band, item_type? | 200 + FP-Growth URLs or empty_state |
| GET | `/visualizations/bupar` | cohort, age_band | 200 + BupaR plot URLs |
| GET | `/visualizations/bupar/activity_frequency` | cohort, age_band | 200 + { overall, pre_target, post_target } |
| GET | `/visualizations/feature_importance` | cohort | 200 + heatmap_url, combined_url |
| GET | `/visualizations/cohort_pgx` | cohort, age_band | 200 + network_topology_url; 400 if params missing |

---

## 3. Dashboard Components (tabs) – What to check

| Tab | Data source | Automated check |
|-----|-------------|-----------------|
| Risk Assessment | GET /metadata, POST /risk, POST /risk/comparison | Metadata returns age_bands and codes; risk returns numeric score and band |
| PGx Card | POST /pgx/card, CPIC Excel in Lambda | PGx card returns genes/drugs; CPIC present in container or S3 |
| Documentation | GET /metrics or same-origin metadata/model_performance_metrics.json | 200 and valid JSON with cohort metrics |
| Feature Importance | GET /visualizations/feature_importance | 200 and URLs; optional: HEAD request to heatmap URL returns 200 |
| BupaR Process Mining | GET /visualizations/bupar, GET /visualizations/bupar/activity_frequency | 200 and URLs; activity_frequency has overall/pre_target/post_target |
| DTW Trajectories | GET /visualizations/dtw | 200 and overview_image, chart_data_url, etc. |
| FP-Growth Patterns | GET /visualizations/fpgrowth | 200 and URLs or empty_state.json |
| PGx Cohort | GET /visualizations/cohort_pgx | 200 and network_topology_url; 400 when cohort/age_band missing |

---

## 4. Automated Test Suite

Run the automated tests from repo root:

```bash
# Unit-style: required artifact paths (local or S3) and Lambda handler with mock events (no network)
pytest 10_risk_dashboard/tests/test_final_dashboard.py -v

# With live API (set base URL; tests will hit API Gateway)
BASE_URL=https://YOUR_API_ID.execute-api.us-east-1.amazonaws.com/prod pytest 10_risk_dashboard/tests/test_final_dashboard.py -v

# Only artifact checks (skip API if no BASE_URL)
pytest 10_risk_dashboard/tests/test_final_dashboard.py -v -k "artifacts or lambda_handler"
```

The suite:

1. **Artifacts** – Asserts presence of required files (locally under `10_risk_dashboard/outputs/` or, if configured, in S3).
2. **Lambda handler** – Builds mock API Gateway events for each route and calls the Lambda handler; asserts status code and minimal response shape.
3. **Live API** – If `BASE_URL` is set, sends GET/POST to each endpoint and asserts status and response structure (optional, for post-deploy verification).

---

## 5. Manual / CI Checklist

- [ ] Run `prepare_lambda_dir.py` (or equivalent) and confirm container bundle has metadata, models, CPIC.
- [ ] Build and push Docker image; deploy Lambda with new image.
- [ ] Run `create_api_gateway_pgx_risk_calculator.sh` (and optionally `add_api_gateway_resources_phts_style.sh`).
- [ ] Upload frontend to dashboard bucket; upload metadata and model_performance_metrics.json.
- [ ] Run visualization pipeline (4_dashboard_visuals / run_dashboard_visuals) so BupaR, DTW, FP-Growth, Cohort PGx assets exist in dashboard bucket.
- [ ] Run `pytest 11_testing/tests/ -v` with `BASE_URL` set; fix any failing endpoint or missing artifact.
- [ ] **Dashboard errors review:** Open each tab referenced in `status/dashboard_errors/` (Causal Analysis, Feature Importance, BupaR, DTW, FP-Growth); run checks in Section 6 and fix any regressions.

---

## 6. Dashboard errors (status/dashboard_errors) – checks and fixes

**Reference:** The folder `status/dashboard_errors/` contains evidence of past dashboard errors (screenshots/PDFs) for **Causal Analysis**, **Feature Importance**, **BupaR**, **DTW**, and **FP-Growth** tabs. Use this section to prevent regressions and to verify fixes.

### 6.1 Checks to run (per tab)

| Tab | Evidence in status/dashboard_errors | Automated check | Manual / fix check |
|-----|-------------------------------------|-----------------|---------------------|
| **Causal Analysis** | `causal_analysis_tab.png` | GET /visualizations/causal returns 200 and dict; 400 when cohort/age_band missing. | Load tab with cohort/age_band; confirm either causal viz or clear status message (no uncaught error or blank panel). |
| **Feature Importance** | `feature_importance_tab.png` | GET /visualizations/feature_importance returns 200 and heatmap_url/combined_url. Optional: HEAD to returned URL returns 200. | Load tab; if image fails, status must show "Cohort heatmap image not found. Upload aggregated_fi_heatmap.png to the dashboard bucket." (frontend `onerror`). |
| **BupaR Process Mining** | `PGx Risk Assessment Dashboard_bupaR_tab.pdf` | GET /visualizations/bupar and /bupar/activity_frequency return 200 and expected shape; activity_frequency may have null overall/pre_target/post_target. | Load tab; confirm images or friendly message; activity frequency charts show data or "not found" message. |
| **DTW Trajectories** | `PGx Risk Assessment Dashboard_dtw_tab.pdf` | GET /visualizations/dtw returns 200 and overview_image or chart_data_url or metrics. | Load tab; optional chart_data.json fetch failure must not break tab; status shows error message on API failure. |
| **FP-Growth Patterns** | `PGx Risk Assessment Dashboard_fpgrowth_tab.pdf` | GET /visualizations/fpgrowth returns 200 with URLs or empty_state. | Load tab; if itemsets image 404s, panel must show "Plot not found. Run 4_dashboard_visuals.ipynb to build visuals." (frontend `onerror`). |
| **PGx Cohort** | (none yet) | GET /visualizations/cohort_pgx returns 200 and network_topology_url; 400 when params missing. | Load tab; iframe loads network HTML or status shows "No network URL returned" / error. |

### 6.2 Fixes applied / to verify

- **Lambda:** All visualization handlers return **200** with a consistent JSON shape (URLs or empty_state) when cohort/age_band are valid; return **400** when required query params are missing; avoid **500** for missing S3 objects where possible (return URLs and let frontend/image handle 404).
- **Frontend:** Every "Load" button for visualization tabs uses `try/catch` and sets the tab's `status-message` to `error` with a short message on fetch failure. Images that load from returned URLs have `onerror` where appropriate (Feature Importance heatmap, FP-Growth itemsets).
- **S3:** After running 4_dashboard_visuals and 5_build_and_deploy, confirm dashboard bucket has: `bupar/`, `dtw/`, `fpgrowth/`, `feature_importance/`, `cohort_pgx/` under the dashboard prefix. Missing prefixes cause 404 on image/iframe load; frontend should show a clear message, not a blank or uncaught exception.
- **Tests:** Suite under `11_testing/tests/` asserts status in (200, 400, 404, 500) and response body shape for each visualization endpoint. Run `pytest 11_testing/tests/ -v` and fix any failing tab test before release.

### 6.3 Manual regression pass (after deploy)

1. Open the deployed dashboard URL.
2. For each of **Causal Analysis**, **Feature Importance**, **BupaR**, **DTW**, **FP-Growth**, **PGx Cohort**: select a valid cohort and age band, click Load.
3. Confirm either (a) visualizations load, or (b) a clear status message is shown (e.g. "Plot not found", "Upload ... to the dashboard bucket", "Error: ..."). No blank panels or uncaught console errors.
4. Compare with any screenshots in `status/dashboard_errors/` to ensure the same error no longer appears.
