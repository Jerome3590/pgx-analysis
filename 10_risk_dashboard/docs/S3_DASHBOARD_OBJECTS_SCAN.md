# S3 Dashboard Objects Scan (HTML & Images)

Scan of `s3://jerome-dixon.io/vcu/pgx-risk-calculator/` (recursive).  
Full listing: see agent-tools output file used for this summary.

---

## Why HTML and image objects don’t render in the dashboard

**Cause:** The dashboard uses **direct S3 object URLs** in path-style form (e.g. `https://s3.us-east-1.amazonaws.com/jerome-dixon.io/vcu/pgx-risk-calculator/.../network_topology.html`). The frontend loads them in iframes (`iframe.src`) and images (`<img src>`). When the bucket or those objects are **not public**, S3 responds with **403 Forbidden** to the browser. The browser then shows “might be temporarily down or it may have moved permanently” instead of the HTML or image.

**So:** Objects exist in S3 and URLs are correct; they don’t render because **unauthenticated (public) read is not allowed** on the bucket/objects.

**Ways to fix:**

1. **Public read on S3**  
   - Bucket policy or object ACL that allows `s3:GetObject` for public (`*` or a public principal).  
   - Direct URLs then work in iframes and `<img>` with no code change.  
   - Easiest operationally; bucket/prefix becomes publicly readable.

2. **Presigned URLs**  
   - Lambda generates time-limited URLs with `s3_client.generate_presigned_url('get_object', ...)` and returns those instead of direct S3 URLs.  
   - Bucket stays private; URLs expire (e.g. 1 hour).  
   - Requires changing Lambda to return presigned URLs for each visualization URL (cohort_pgx, fpgrowth, bupar, etc.).

3. **Lambda proxy**  
   - New API route (e.g. `GET /visualizations/proxy?key=...`) where Lambda reads the object from S3 (using IAM) and returns the body with the right `Content-Type`.  
   - Frontend calls the API URL instead of the S3 URL.  
   - Bucket stays private; no URL expiry; more moving parts and Lambda bandwidth.

Once one of these is in place, the same HTML and image URLs (or their proxy/presigned equivalents) will render in the dashboard.

## Summary

- **HTML + image objects (by extension):** 257 keys ending in `.html` or `.png` (includes checkpoints).
- **Dashboard-relevant HTML:** `network_topology.html` (PGx Cohort), `combined_rules_network.html` (FP-Growth), plus BupaR interactive `.html` in `bupar/…/plots/`.
- **Dashboard-relevant images:** FP-Growth `*_combined_top_itemsets.png`, feature importance `aggregated_fi_heatmap.png`, BupaR `*.png` in `bupar/…/plots/`.

## HTML objects (dashboard-relevant)

### PGx Cohort – network topology

- **Path pattern:** `vcu/pgx-risk-calculator/cohort_pgx/networks/{cohort}/{age_band_fname}/network_topology.html`
- **age_band_fname:** underscore (e.g. `25_44`), not hyphen.
- **Count:** 18 (2 cohorts × 9 age bands).
- **Example:** `cohort_pgx/networks/opioid_ed/25_44/network_topology.html`

### FP-Growth – combined rules network

- **Path pattern:** `vcu/pgx-risk-calculator/fpgrowth/{cohort}/{age_band}/plots/{cohort}_{age_band_fname}_combined_rules_network.html`
- **age_band in path:** hyphen (e.g. `25-44`); **age_band_fname in filename:** underscore (`25_44`).
- **Count:** 18 (2 cohorts × 9 age bands).
- **Note:** Some non_opioid_ed files are ~480 bytes (likely placeholders).

### Other HTML

- **BupaR:** `bupar/{cohort}/{age_band}/plots/*_activity_frequency_interactive.html`, `*_trace_explorer_interactive.html`, etc.
- **Root:** `dashboard_index_template.html`, `index.html`.

---

## Image objects (dashboard-relevant)

### FP-Growth – Top Itemsets

- **Path pattern:** `vcu/pgx-risk-calculator/fpgrowth/{cohort}/{age_band}/plots/{cohort}_{age_band_fname}_{drug_name|cpt_code|icd_code}_combined_top_itemsets.png`
- **Count:** 28 (non_opioid_ed: 8 drug_name; opioid_ed: 9 age bands × up to 3 types).
- **Example:** `fpgrowth/opioid_ed/25-44/plots/opioid_ed_25_44_drug_name_combined_top_itemsets.png`

### Feature importance

- **Paths:**
  - `feature_importance/combined_cohorts_feature_importance_heatmap.png`
  - `feature_importance/non_opioid_ed/aggregated_fi_heatmap.png`
  - `feature_importance/opioid_ed/aggregated_fi_heatmap.png`

### BupaR

- **Path pattern:** `bupar/{cohort}/{age_band}/plots/*.png` and `*.html`.
- **Dashboard only shows URLs for objects that exist:** Lambda does a HEAD check per key and returns only existing S3 URLs. Missing assets show as “Visual not available” in the UI.
- **Produced by 9_dashboard_visuals/bupar R scripts:** `overall_activity_frequency.png`, `pre_f1120_activity_frequency.png` / `pre_hcg_activity_frequency.png`, `activity_frequency_interactive.html`, `process_matrix.png`, `process_matrix_*.png`, `trace_explorer_interactive.html`, `trace_explorer_pre_f1120.png` / `trace_explorer_pre_hcg.png`, `frequency_map.png` (when `processmapR::export_map` exists). For **opioid_ed** also: `trace_explorer_post_f1120.png`, `trace_explorer_post_f1120_interactive.html`, `post_f1120_activity_frequency.png`; for **non_opioid_ed**: `trace_explorer_post_hcg.png`, etc.
- **Not produced by this pipeline:** `activity_sequence_top.png`, standalone `trace_explorer.png`, `process_matrix_interactive.html`. If those keys are missing on S3, Lambda omits them and the dashboard shows “Visual not available.”

---

## URL format (required: path-style)

We **must** use **path-style** S3 URLs for dashboard assets (HTML and images), not virtual-hosted style.

- **Template:** `https://s3.{region}.amazonaws.com/{bucket}/{prefix}/{object_key}`
- **Example:** `https://s3.us-east-1.amazonaws.com/jerome-dixon.io/vcu/pgx-risk-calculator/cohort_pgx/networks/non_opioid_ed/55_64/network_topology.html`

Lambda builds these via `_dashboard_s3_url(key)` in `lambda_function.py`. Use this same format for any manual links or frontend URLs to S3 objects in the dashboard bucket.

For browser access, bucket or object ACL/policy must allow public read, or use presigned URLs / Lambda proxy.
