# Static-first JSON pattern (dashboard data)

## Why

- **Fastest:** Same-origin JSON is served from CloudFront edge; no Lambda cold start or execution.
- **Cheapest:** S3 GET + CloudFront transfer only; no Lambda invocations or API Gateway request cost for that data.
- **Lambda** is used only when static JSON is missing (e.g. dev, or a cohort not yet deployed) or for **dynamic** endpoints (e.g. `/risk`, `/risk/comparison`).

## Pattern

1. **Frontend** requests a **same-origin path** first (e.g. `metadata/opioid_ed.json`, `feature_importance/opioid_ed.json`).
2. If the response is **200**, use that JSON (from S3/CloudFront).
3. If **404** (or other error), **fall back** to the Lambda API (e.g. `GET ${API_BASE}/metadata?cohort=opioid_ed`).
4. Dynamic endpoints (risk score, comparison) **always** use the API; no static file.

## When static files are not deployed (404)

If you haven’t uploaded `metadata/*.json` (or other static JSON) to S3/CloudFront, the first request will **404**; the app then falls back to the Lambda API and metadata still loads. To avoid the 404 in the network tab and go straight to the API, open the dashboard with:

- **`?metadata=api`** — skip the static metadata request and use the API only.

Example: `https://jerome-dixon.io/vcu/pgx-risk-calculator/?metadata=api`

## Same-origin paths

Paths are relative to the dashboard root (e.g. `/vcu/pgx-risk-calculator/`). The frontend uses `staticJsonPath(relativePath)` so that:

- From `https://example.com/vcu/pgx-risk-calculator/index.html`
- `staticJsonPath("metadata/opioid_ed.json")` → `/vcu/pgx-risk-calculator/metadata/opioid_ed.json`

## S3 layout (dashboard bucket prefix)

Under the dashboard prefix (e.g. `vcu/pgx-risk-calculator/`), deploy these for static-first behavior:

| Same-origin path | S3 key (under prefix) | Shape / content |
|------------------|------------------------|-----------------|
| `metadata/{cohort}.json` | `metadata/opioid_ed.json`, `metadata/non_opioid_ed.json` | Same as Lambda `GET /metadata?cohort=...`: `{ codes: { "25-44": { drugs, icds, cpts }, ... }, ... }` |
| `feature_importance/{cohort}.json` | `feature_importance/opioid_ed.json`, etc. | Same as Lambda `GET /visualizations/feature_importance?cohort=...`: `{ heatmap_data?, heatmap_url, combined_url }` |
| `metadata/model_performance_metrics.json` | `metadata/model_performance_metrics.json` | Same as Lambda metrics payload (optional; doc tab can use API) |

Other assets (BupaR, DTW, FP-Growth, etc.) continue to be loaded via URLs returned by the API or as direct S3/CloudFront URLs; they are not part of this static-first JSON pattern unless we add more static paths later.

## Deployment (EC2 / pipeline)

When syncing the full deployment package to S3:

1. Upload frontend: `aws s3 sync frontend/ s3://bucket/vcu/pgx-risk-calculator/`
2. Upload metadata:  
   `aws s3 cp outputs/metadata/metadata_opioid_ed.json s3://bucket/vcu/pgx-risk-calculator/metadata/opioid_ed.json --content-type application/json`  
   (and `non_opioid_ed.json` similarly)
3. Upload feature importance JSON per cohort (same shape as API):  
   e.g. copy `3a_feature_importance/outputs/opioid_ed/plots/opioid_ed_aggregated_fi_heatmap.json` to a single file or build a wrapper `{ heatmap_data: <that JSON>, heatmap_url: "...", combined_url: "..." }` and upload as `feature_importance/opioid_ed.json`.

After upload, invalidate CloudFront for the changed paths so the edge serves the new files.

## References

- Deployment: `10_risk_dashboard/deployment/README.md`
- S3 URL format for assets: `.cursor/rules/dashboard-s3-url-format.mdc`
- Cursor rule for this pattern: `.cursor/rules/dashboard-static-first-json.mdc`
