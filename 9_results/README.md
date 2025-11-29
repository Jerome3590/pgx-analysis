## 9_results – Final Results & Dashboard

This directory contains the **results and reporting layer** for the PGx analysis pipeline, focused on:

- Exposing the **final selected model ensemble** (CatBoost, XGBoost, XGBoost RF) and **FFA outputs**.
- Providing a **Plotly HTML/JavaScript dashboard** hosted on S3.
- Backing the dashboard with a **Lambda + API Gateway** service that queries artifacts in S3 on demand.

### Components

- `dashboard_index_template.html`
  - Template HTML/JS for the Plotly dashboard.
  - Loaded directly from S3 (static website hosting) and runs entirely client‑side.
  - Talks to a Lambda/API Gateway endpoint for metadata, risk estimates, and causal “what‑if” summaries.

- `lambda_api_template.py`
  - Python Lambda handler template for an API Gateway **proxy integration**.
  - Implements three logical endpoints:
    - `GET /metadata` – valid age bands and code lists (Drugs, ICDs, CPTs) per cohort/age band.
    - `POST /risk` – risk estimate from the final ensemble for a given configuration.
    - `POST /causal` – causal “what‑if” summary derived from FFA artifacts or model counterfactuals.
  - Reads artifacts (final features, models, FFA outputs) from S3; the exact bucket/prefixes should be aligned with the rest of the project (e.g. `pgxdatalake/gold/...` and `ffa_analysis/...`).

### High‑Level Flow

1. **User opens dashboard** at the S3 static website URL:
   - Browser loads `dashboard_index_template.html` (deployed as `index.html`) and Plotly.

2. **Dashboard initialization**:
   - Calls `GET /metadata?cohort=...` on the API Gateway URL to populate:
     - Allowed age bands.
     - Valid Drugs / ICDs / CPTs for that cohort and age band.

3. **Risk estimation**:
   - When the user clicks “Update Risk”, the dashboard sends `POST /risk` with:
     - `cohort`, `age_band`, and selected Drug/ICD/CPT codes.
   - Lambda:
     - Parses the request, loads the final model + any necessary preprocessing metadata from S3.
     - Constructs the feature vector for the requested configuration.
     - Returns the predicted probability of F1120/ADE plus optional per‑model probabilities and a simple risk distribution summary for plotting.

4. **Causal “what‑if” view**:
   - When the user clicks “Causal ‘What‑if’”, the dashboard sends `POST /causal` with the same payload.
   - Lambda:
     - Uses FFA artifacts (e.g., stored causal summaries) or model‑based counterfactuals to compute:
       - `Δ risk` if each selected Drug/ICD/CPT were removed or changed.
     - Returns a list of `{code, delta_risk_remove}` records for Plotly.

### Deployment Notes

- **Static dashboard**:
  - Upload a copy of `dashboard_index_template.html` as `index.html` under a prefix such as:
    - `s3://pgxdatalake/ffa_dashboard/index.html`
  - Enable S3 static website hosting and point a friendly URL (e.g., CloudFront) at the bucket.

- **Lambda/API Gateway**:
  - Deploy `lambda_api_template.py` as a Lambda function (Python 3.10+).
  - Create an HTTP API or REST API in API Gateway with proxy integration to the Lambda.
  - Update `API_BASE` in the dashboard template to match your API Gateway invoke URL.
  - Add IAM permissions for Lambda to read from:
    - `s3://pgxdatalake/gold/...` (final features/models)
    - `s3://pgxdatalake/ffa_analysis/...` (FFA outputs)

This directory is intentionally templated: fill in the S3 paths and model/FFA wiring as the final ensemble and FFA artifacts are finalized.


