---
description: Dashboard data — prefer same-origin static JSON (S3/CloudFront), fallback to Lambda API
globs: 10_risk_dashboard/**/*.*, 9_dashboard_visuals/**/*.py
alwaysApply: false
---

# Dashboard: static-first JSON pattern

When adding or changing how the dashboard loads **pre-built** data (metadata, feature importance, or other JSON that can be deployed with the frontend):

- **Prefer same-origin static JSON** (S3/CloudFront). The frontend should request a path like `metadata/{cohort}.json` or `feature_importance/{cohort}.json` first (using `staticJsonPath(relativePath)` so it resolves under the dashboard root).
- **Fallback to Lambda API** only when the static request returns 404 or fails. This keeps latency and cost low when the full deployment package is on S3.
- **Dynamic data** (e.g. risk score, comparison) must always go to the API; no static file.
- **Static file shape** must match the API response shape for that resource so the same rendering code works for both (e.g. `metadata/{cohort}.json` same as `GET /metadata?cohort=...`; `feature_importance/{cohort}.json` same as `GET /visualizations/feature_importance?cohort=...`).
- **Document** any new static path and its S3 key in `10_risk_dashboard/docs/STATIC_FIRST_JSON.md` (S3 layout table and deployment steps).

See: `10_risk_dashboard/docs/STATIC_FIRST_JSON.md`, `10_risk_dashboard/deployment/README.md`.
