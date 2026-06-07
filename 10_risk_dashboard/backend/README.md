# Backend API (Lambda Function)

## Overview

Lambda receives **user input** (cohort, age_band, model/feature selections) and **filters** only—it does not process or generate visualization data. All visuals are prebuilt on EC2 and saved to S3. **Visualization pattern:** We prefer **JSON** where possible: Lambda loads JSON from S3 and returns it inline so the frontend can render with Plotly; when JSON is missing, Lambda returns prebuilt S3 URLs (image/HTML). Network plots (FP-Growth, PGx Cohort) are HTML only. See `10_risk_dashboard/docs/VISUALIZATION_DATA_PATTERN.md`. Risk inference uses the ensemble with user-provided features; no analytics or chart building runs in Lambda.

## Files

- **`lambda_function.py`** - Main Lambda handler with all API endpoints
- **`requirements.txt`** - Python dependencies
- **`Dockerfile`** - Docker container configuration for Lambda (ECR)
- **`lambda_api_template.py`** - Template for API Gateway integration

## API Endpoints

### Core Endpoints

- **`GET /metadata`** - Get valid codes for dropdowns (filter by cohort). **Fallback only:** the frontend loads from same-origin `metadata/{cohort}.json` (deployed with the dashboard). If missing, the frontend calls this endpoint.
  - Query params: `cohort` (opioid_ed | non_opioid_ed)
  - Returns: Age bands and code lists (drugs, ICDs, CPTs)

- **`POST /risk`** - Risk score from best model per cohort/age_band (or 2019 baseline when no codes)
  - Body: `{cohort, age_band, drugs[], icds[], cpts[]}` (optional: `age`)
  - **Optional body fields:** `n_events`, `n_drugs` — used for **risk bucket** (low/medium/high). `pgx_num_drugs`, `pgx_num_cpic_drugs` are **separate inputs** (for model/display only, not used for risk bucket). When PGx counts are provided or auto-derived, Lambda also derives PGx burden features (`pgx_non_cpic_drugs`, `pgx_has_any_drug`, `pgx_has_cpic_drug`, `pgx_cpic_fraction`, `pgx_num_drugs_log1p`, `pgx_num_cpic_drugs_log1p`) so training and inference stay aligned. Other omitted non-item features use schema defaults.
  - When no Drug/ICD/CPT codes: returns **baseline_risk** (actual 2019 outcome rate). When any code is provided: returns the **best model’s** predicted probability (MC-CV best per cohort/age_band).
  - Returns: `risk_score`, `risk_band`, `is_baseline`, `patient_bucket`, `patient_bucket_detail` (n_events_bucket, n_drugs_bucket), `n_pgx_drugs`, `pgx_num_cpic_drugs`, `model_breakdown`, `dist` (2019 histogram when available)

**Risk band cutoffs (Low / Medium / High)**  
The displayed `risk_band` uses **absolute probability thresholds** (not cohort-relative percentiles) so labels match user intuition (e.g. 7.7% is Low):

| Band    | Condition (score = probability in [0, 1]) |
|---------|--------------------------------------------|
| **Low** | score &lt; 0.20 (&lt; 20%)                 |
| **Medium** | 0.20 ≤ score &lt; 0.50 (20–50%)         |
| **High**   | score ≥ 0.50 (≥ 50%)                   |

Thresholds are fixed in the backend (`DEFAULT_RISK_BAND_THRESHOLDS`: `low_medium` = 0.2, `medium_high` = 0.5). The 2019 distribution file’s `risk_band_thresholds` (33rd/67th percentiles) are **not** used for the band label; they remain in `dist` for the histogram and reference only.

### Model input features mapped to dashboard labels

The trained Step 6 model uses numeric features, while the dashboard should present interpretable labels. These labels are display summaries of model inputs; they are not separate model columns unless listed in `feature_schema.json`.

| Dashboard label family | Model input feature(s) | User-facing interpretation |
|------------------------|------------------------|----------------------------|
| **Event density** | `n_event_bin_ordinal` (`low=0`, `medium=1`, `high=2`, `extreme=3`) | Overall event/claim density stratum used for model routing and as a retained density feature. |
| **Event velocity** | `event_rate_per30`, `early_event_rate_per30`, `late_event_rate_per30` | How quickly events accumulate over the observed pre-target history. |
| **Event acceleration / trajectory** | `event_rate_delta_per30`, `event_rate_ratio_late_vs_early` | Whether utilization is decreasing, stable, accelerating, or rapidly accelerating. |
| **Event timing regularity** | `mean_inter_event_days`, `median_inter_event_days`, `std_inter_event_days`, `event_burstiness` | Whether events are evenly spaced, variable, bursty, or highly bursty. |
| **Recent activity** | `recent30_event_count`, `recent90_event_count`, `recent30_event_fraction`, `recent90_event_fraction` | Whether recent pre-index activity is low, moderate, high, or spiking. |
| **Medication / PGx burden** | `pgx_num_drugs`, `pgx_num_cpic_drugs`, `pgx_non_cpic_drugs`, `pgx_cpic_fraction`, `pgx_has_any_drug`, `pgx_has_cpic_drug`, log-count variants | Medication burden and CPIC-relevant exposure context for model adjustment and display. |
| **Specific clinical codes** | `item_*` drug, ICD, and CPT indicators | The user-selected medications, diagnoses, and procedures that directly affect the score when present in the model schema. |

Recommended dashboard text examples:

- **Event density:** Low / Moderate / High / Extreme activity
- **Event velocity:** Slow / Moderate / Fast / Very fast accumulation
- **Trajectory:** Decelerating / Stable / Accelerating / Rapidly accelerating
- **Event timing:** Regular / Variable / Bursty / Highly bursty
- **Recent activity:** Low recent activity / Moderate recent activity / High recent activity / Recent spike

Temporal dynamics are engineered during Step 6 from pre-target `event_date` values. For manual `/risk` requests without dated events, these temporal features are taken from `feature_schema.json` defaults; patient-specific temporal labels require a dated event history or precomputed patient-history context.

- **`POST /risk/comparison`** - Compare risk for user-provided scenarios (filter by selection)
  - Body: `{base: {...}, scenarios: [...]}`
  - Returns: Risk scores for base and scenarios

- **`POST /pgx/card`** - Generate PGx patient card
  - Body: `{patient_id?, variants: [{gene, variants[]}]}`
  - Returns: PGx card data with drug-gene interactions

- **`GET /metrics`** - Return prebuilt model performance metrics (Documentation tab). **Fallback only:** the frontend loads metrics from the same-origin static asset `metadata/model_performance_metrics.json` (deployed with the dashboard to the dashboard bucket). If that file is missing (e.g. local dev), the frontend calls this endpoint. Lambda reads from S3 (`gold/dashboard/metadata/model_performance_metrics.json`) or container bundle; no recomputation.

### Visualization Endpoints (filter only; return prebuilt S3 URLs)

- **`GET /visualizations/scenario`** - Return scenario/SHAP data (same pattern as Feature Importance: load JSON, optional Lambda processing)
  - Query params: `cohort`, `age_band`; optional `drugs`, `icds`, `cpts`, `whatif` (comma-separated codes) to filter
  - Returns: `scenario_data` (raw from S3), `chart_data` (Lambda-built: interaction_factors, shap_importance, interaction_factors_whatif, shap_importance_whatif, feature_interactions when present) for bar charts, radar, and drug/feature interactions

- **`GET /visualizations/dtw`** - Return DTW assets (prefer inline JSON when present)
  - Query params: `cohort`, `age_band`
  - Returns: `chart_data`, `sequence_heatmap`, `trajectory_overview_plot` (inline JSON when in S3); else `chart_data_url`, `sequence_heatmap_url`, image URLs

- **`GET /visualizations/fpgrowth`** - Return FP-Growth assets (itemsets JSON when present; network = HTML only, EC2)
  - Query params: `cohort`, `age_band` (item_type is fixed to `drug_name`)
  - Returns: `itemsets_data` (inline JSON when in S3); S3 URLs for network HTML and itemsets PNG

- **`GET /visualizations/bupar`** - Return BupaR assets (prefer inline JSON when present)
  - Query params: `cohort`, `age_band`
  - Returns: `trace_explorer_plot`, `process_matrix_drug_drug` (inline JSON when in S3); S3 URLs for other plot images/HTML. Process matrix type-pair is **Drug × Drug** only.

- **`GET /visualizations/bupar/activity_frequency`** - Return activity frequency JSON for bar charts
  - Query params: `cohort`, `age_band`
  - Returns: `{ overall, pre_target, post_target }` (each with `year_labels` and `data`); frontend builds Plotly bar charts with year filter

- **`GET /visualizations/feature_importance`** - Return feature importance heatmap (prefer inline JSON)
  - Query params: `cohort` (opioid_ed | non_opioid_ed | combined)
  - Returns: `heatmap_data` (inline JSON when in S3); else `heatmap_url` (PNG)

- **`GET /visualizations/cohort_pgx`** - Return PGx Cohort network (HTML only, EC2-built)
  - Query params: `cohort`, `age_band`
  - Returns: `network_topology_url` when HTML exists on S3

## Model Loading

Models are loaded from:
1. **Container filesystem** (`/var/task/models/`) - Primary source (ECR)
2. **S3** (`s3://pgxdatalake/gold/dashboard/models/`) - Fallback

## Environment Variables

- `PGX_RESULTS_BUCKET` - S3 bucket for data/models (default: `pgxdatalake`)
- `S3_DASHBOARD_BUCKET` - Bucket where the dashboard frontend is deployed; FP-Growth assets are uploaded here (default: `jerome-dixon.io`)
- `S3_DASHBOARD_PREFIX` - Key prefix for the dashboard app in that bucket (default: `vcu/pgx-risk-calculator`). BupaR and other visualization APIs use `{prefix}/visualizations/...` (the key must include `visualizations`). **Target path for BupaR plots:** `{prefix}/visualizations/bupar/{cohort}/{age_band}/plots/` — e.g. `s3://jerome-dixon.io/vcu/pgx-risk-calculator/visualizations/bupar/opioid_ed/45-54/plots/`.
- `MODEL_BASE_PATH` - Path to models in container (default: `/var/task/models`)
- `MODEL_CACHE_TTL` - Model cache TTL in seconds (default: `3600`)

**Empty visualization responses:** Notebook 4 (**4_dashboard_visuals.ipynb**) and **run_dashboard_visuals.py** both **build** the artifacts (BupaR, DTW, FP-Growth) and **upload** them to the dashboard bucket using the same `S3_DASHBOARD_BUCKET` and `S3_DASHBOARD_PREFIX`. Lambda reads from that same bucket/prefix. So in the normal flow, notebook 4 builds and uploads; the API then returns URLs to those objects. If you still see empty responses: (1) Lambda may not have permission to read the bucket (403) — we return 200 with empty payload instead of 500 so the frontend shows "not available"; (2) Lambda env `S3_DASHBOARD_BUCKET` / `S3_DASHBOARD_PREFIX` must match the bucket/prefix used when running the notebook on EC2; (3) **PGx Cohort network visuals**: Built in notebook 4 (fetch_vip_reports + build_network_topology); each build uploads to the dashboard bucket (same as BupaR/DTW/FP-Growth). Notebook 5 (Step 6: Sync Dashboard Frontend) syncs `10_risk_dashboard/visualizations/cohort_pgx/` to S3 when you deploy. Lambda returns `network_topology_url` when `{prefix}/cohort_pgx/networks/{cohort}/{age_band_fname}/network_topology.html` exists.

## Generating visualization artifacts

All visualization **artifacts** (BupaR plots, DTW images and chart data, FP-Growth itemsets/plots) are **built and uploaded to S3** by notebook 4 (or run_dashboard_visuals.py) on EC2. The scripts use `S3_DASHBOARD_BUCKET` and `S3_DASHBOARD_PREFIX`; Lambda uses the same env vars to return URLs. The API returns only URLs to these prebuilt assets (no computation at request time). To (re)generate from repo root:

- **Notebook:** `4_dashboard_visuals.ipynb` (run from repo root)
- **Script (VS Code Jupyter format):** `pgx_dashboard_visuals.py` (run as script or by cell with `# %%`)

Both run BupaR, DTW, and FP-Growth for configured cohorts/age bands. **FP-Growth**, **BupaR**, and **DTW** assets are uploaded to the **dashboard bucket** (e.g. `jerome-dixon.io`) under `{S3_DASHBOARD_PREFIX}/fpgrowth/`, `{S3_DASHBOARD_PREFIX}/bupar/`, and `{S3_DASHBOARD_PREFIX}/dtw/`; the dashboard loads them directly from S3 (or via API URL responses). **Redeploy the Lambda image** only when backend code changes.

## Deployment

See `../deployment/README.md` for deployment instructions. To (re)create the API Gateway REST API and wire it to Lambda: `utility_scripts/create_api_gateway_pgx_risk_calculator.sh` (or `.ps1` on Windows).

## Troubleshooting 502 (Bad Gateway) on /visualizations/dtw

A **502** from API Gateway usually means the Lambda **timed out** or **ran out of memory** before returning. Visualization endpoints load JSON from S3 (chart_data, sequence_heatmap, trajectory_overview_plot); large objects can cause this.

- **Lambda timeout:** In AWS Console → Lambda → your function → Configuration → General → Timeout: set to **15–30 seconds** (default 3 s is often too low for several S3 GETs).
- **Lambda memory:** Use **512 MB** (logs show ~235 MB peak; 30 s timeout helps cold starts). Use CLI to set both (replace `YOUR_FUNCTION_NAME` with your Lambda name):
  ```bash
  aws lambda update-function-configuration \
    --function-name YOUR_FUNCTION_NAME \
    --timeout 30 \
    --memory-size 512
  ```
- **Large trajectory_overview_plot.json:** The handler skips loading `trajectory_overview_plot.json` when it is larger than 2 MB (frontend falls back to overview/sample image URLs). If you still see 502, ensure DTW chart_data and sequence_heatmap in S3 are not unusually large.
- **CORS:** If the browser reports "blocked by CORS" instead of 502, enable CORS on the API Gateway API (OPTIONS method returning `Access-Control-Allow-Origin`) so requests from `https://jerome-dixon.io` are allowed. See `../deployment/README.md` (test OPTIONS with curl).

## Probability Calibration

Tree-based models (XGBoost, CatBoost) often over-predict — they rank patients correctly but the raw probability is higher than the actual observed event rate. A **Platt scaling** second-stage calibrator is applied at inference time to correct this.

### How it works

1. **Training (`run_final_model.py`, notebook 3):** During Monte-Carlo CV, each split's test-fold predictions are **out-of-fold** (never used for training that split). After all `n_runs` splits, a `LogisticRegression(C=1)` is fitted on the concatenated OOF probabilities vs actual outcomes — one calibrator per model type (`xgboost`, `xgboost_rf`, `catboost`). Saved to `6_final_model/outputs/{cohort}/{age_band_fname}/models/calibration_{model_type}.joblib`.

2. **Deploy (`prepare_models.py` → `prepare_lambda_dir.py`):** Calibration joblib files are copied alongside regular model files into the Lambda container image.

3. **Inference (`lambda_function.py`):**
   - `load_calibration_model(cohort, age_band, model_type)` — loads from container → S3, in-memory cached.
   - After each raw model probability, `apply_calibration(raw_prob, calibrator)` maps it to a calibrated probability.
   - Ensemble is computed from **calibrated** probabilities.
   - **Graceful degradation:** if a calibration file is absent (model trained before this feature was added), raw probability is used unchanged — functional but uncalibrated.

### New `/risk` response fields

| Field | Description |
|-------|-------------|
| `calibrated` | `true` when at least one model had a calibrator applied |
| `raw_risk_score` | Weighted ensemble of raw (uncalibrated) probabilities; only present when `calibrated=true` |
| `risk_score` | Calibrated ensemble probability (or raw when no calibrator is available) |

### ⚠️ Calibration files require a model training run

If `models/calibration_*.joblib` does not exist for a cohort/age_band, **re-run notebook 3 (`run_final_model.py`)** for that cohort/age_band. The Lambda logs a warning and falls back to raw probabilities.

## Risk score: recommendations

- **Best model only:** We run only the model(s) with non-zero weight (best per cohort/age_band), so cold start and failure surface are reduced.
- **Baseline when no codes:** No Drug/ICD/CPT → return 2019 outcome rate; with codes → model probability. Keeps risk calibrated to the 2019 population.
- **Implemented:** Risk band uses **absolute cutoffs** (low <20%, medium 20–50%, high ≥50%); comparison uses baseline when base or scenario has no codes; `codes_used`/`codes_unknown` and `model_used` in response; `interpretation` in API and frontend; Platt calibration applied when calibration files are present.
