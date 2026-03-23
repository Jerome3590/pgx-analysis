# Backend API (Lambda Function)

## Overview

Lambda receives user input (cohort, age_band, model/feature selections) and filters only—it does not process or generate visualization data. All visuals are prebuilt on EC2 and saved to S3. Visualization pattern: We prefer JSON where possible: Lambda loads JSON from S3 and returns it inline so the frontend can render with Plotly; when JSON is missing, Lambda returns prebuilt S3 URLs (image/HTML). Network plots (FP-Growth, PGx Cohort) are HTML only. See `10_risk_dashboard/docs/VISUALIZATION_DATA_PATTERN.md`. Risk inference uses the ensemble with user-provided features; no analytics or chart building runs in Lambda.

## Files

- `lambda_function.py` - Main Lambda handler with all API endpoints
- `requirements.txt` - Python dependencies
- `Dockerfile` - Docker container configuration for Lambda (ECR)
- `lambda_api_template.py` - Template for API Gateway integration

## API Endpoints

### Core Endpoints

- `GET /metadata` - Get valid codes for dropdowns (filter by cohort). Fallback only: the frontend loads from same-origin `metadata/{cohort}.json` (deployed with the dashboard). If missing, the frontend calls this endpoint.
  - Query params: `cohort` (opioid_ed | non_opioid_ed)
  - Returns: Age bands and code lists (drugs, ICDs, CPTs)

- `POST /risk` - Risk score from best model per cohort/age_band (or 2019 baseline when no codes)
  - Body: `{cohort, age_band, drugs[], icds[], cpts[]}` (optional: `age`)
  - Optional body fields: `n_events`, `n_drugs` — used for risk bucket (low/medium/high). `pgx_num_drugs`, `pgx_num_cpic_drugs` are separate inputs (for model/display only, not used for risk bucket). When not provided, schema defaults are used for the model.
  - When no Drug/ICD/CPT codes: returns baseline_risk (actual 2019 outcome rate). When any code is provided: returns the best model’s predicted probability (MC-CV best per cohort/age_band).
  - Returns: `risk_score`, `risk_band`, `is_baseline`, `patient_bucket`, `patient_bucket_detail` (n_events_bucket, n_drugs_bucket), `n_pgx_drugs`, `pgx_num_cpic_drugs`, `model_breakdown`, `dist` (2019 histogram when available)

**Risk band cutoffs (Low / Medium / High)**
The displayed `risk_band` uses absolute probability thresholds (not cohort-relative percentiles) so labels match user intuition (e.g. 7.7% is Low):

| Band    | Condition (score = probability in [0, 1]) |
|---------|--------------------------------------------|
| Low     | score < 0.20 (< 20%)                      |
| Medium  | 0.20 ≤ score < 0.50 (20–50%)              |
| High    | score ≥ 0.50 (≥ 50%)                      |

Thresholds are fixed in the backend (`DEFAULT_RISK_BAND_THRESHOLDS`: `low_medium` = 0.2, `medium_high` = 0.5). The 2019 distribution file’s `risk_band_thresholds` (33rd/67th percentiles) are not used for the band label; they remain in `dist` for the histogram and reference only.

- `POST /risk/comparison` - Compare risk for user-provided scenarios (filter by selection)
  - Body: `{base: {...}, scenarios: [...]}`
  - Returns: Risk scores for base and scenarios

- `POST /pgx/card` - Generate PGx patient card
  - Body: `{patient_id?, variants: [{gene, variants[]}]}`
  - Returns: PGx card data with drug-gene interactions

- `GET /metrics` - Return prebuilt model performance metrics (Documentation tab). Fallback only: the frontend loads metrics from the same-origin static asset `metadata/model_performance_metrics.json` (deployed with the dashboard to the dashboard bucket). If that file is missing (e.g. local dev), the frontend calls this endpoint. Lambda reads from S3 (`gold/dashboard/metadata/model_performance_metrics.json`) or container bundle; no recomputation.

### Visualization Endpoints (filter only; return prebuilt S3 URLs)

- `GET /visualizations/causal` - Return causal/SHAP data (same pattern as Feature Importance: load JSON, optional Lambda processing)
  - Query params: `cohort`, `age_band`; optional `drugs`, `icds`, `cpts`, `whatif` (comma-separated codes) to filter
  - Returns: `causal_data` (raw from S3), `chart_data` (Lambda-built: causal_factors, shap_importance, causal_factors_whatif, shap_importance_whatif, feature_interactions when present) for bar charts, radar, and drug/feature interactions

- `GET /visualizations/dtw` - Return DTW assets (prefer inline JSON when present)
  - Query params: `cohort`, `age_band`
  - Returns: `chart_data`, `sequence_heatmap`, `trajectory_overview_plot` (inline JSON when in S3); else `chart_data_url`, `sequence_heatmap_url`, image URLs

- `GET /visualizations/fpgrowth` - Return FP-Growth assets (itemsets JSON when present; network = HTML only, EC2)
  - Query params: `cohort`, `age_band` (item_type is fixed to `drug_name`)
