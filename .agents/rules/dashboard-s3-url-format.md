---
description: Use path-style S3 URLs for dashboard assets (HTML, images); not virtual-hosted style
globs: 10_risk_dashboard/**/*.*
alwaysApply: false
---

# Dashboard S3 URLs: path-style required

When adding or changing URLs to dashboard assets in S3 (HTML for iframes, images):

- **Use path-style S3 URLs only.** Do not use virtual-hosted style (`bucket.s3.region.amazonaws.com`).
- **Template:** `https://s3.{region}.amazonaws.com/{bucket}/{prefix}/{object_key}`
- **Example:** `https://s3.us-east-1.amazonaws.com/jerome-dixon.io/vcu/pgx-risk-calculator/cohort_pgx/networks/non_opioid_ed/55_64/network_topology.html`

Lambda builds these via `_dashboard_s3_url(key)` in `10_risk_dashboard/backend/lambda_function.py`. Any code that constructs dashboard asset URLs (backend or frontend) must use this same format for correct resolution and CORS.

See: `10_risk_dashboard/docs/DASHBOARD_TABS.md` (§ S3 URL format for assets), `10_risk_dashboard/docs/S3_DASHBOARD_OBJECTS_SCAN.md` (§ URL format).
