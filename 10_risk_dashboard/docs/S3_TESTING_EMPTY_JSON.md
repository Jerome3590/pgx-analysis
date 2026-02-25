# Testing folder on S3 – why the JSON files are empty

## What’s in the testing folder (synced locally)

After syncing `s3://jerome-dixon.io/vcu/pgx-risk-calculator/testing/` to `10_risk_dashboard/visualizations/s3_sync_testing/`, the JSON files **parse correctly** but their **content is empty**:

| File | Content |
|------|--------|
| `reports/*_vip_reports_summary.json` | `reports_fetched: 0`, `genes_with_vip_text: 0`, `genes: []` |
| `reports/*_vip_reports.json` | `[]` (empty array) |
| `networks/*/gene_metadata.json` | `{"gene_tiers":{},"cpic_genes":[]}` |
| `networks/*/network_stats.json` | All counts 0 (`nodes_total: 0`, `edges_total: 0`, etc.) |
| `networks/*/key_phrases.json` | `{}` |

So this is **not a parsing issue** when reading the JSON. The pipeline **wrote** these files with empty/minimal data.

## Root cause

1. **fetch_vip_reports** (Cohort PGx) loads cohort genes from feature importance (e.g. 29 genes for opioid_ed/25-44). For each gene it calls the PharmGKB API `GET /data/gene?symbol=...`.
2. On EC2, **every** call to the API failed or returned something that doesn’t pass the current checks: `_get()` returns `{}` on any `RequestException` (e.g. 403, 429, timeout, connection error), and `get_gene_report()` returns `{}` if the response has no `"data"` or empty `data` list.
3. So no reports were appended → `reports_fetched: 0`, and the saved `*_vip_reports.json` is `[]`.
4. **build_network_topology** then ran with an empty reports list → 0 nodes/edges → empty `network_stats.json`, `gene_metadata.json`, `key_phrases.json`.

So the empty JSON is a consequence of **PharmGKB API calls failing or returning unexpected data on EC2**, not of mis-parsing the JSON later.

## What to do

- **On EC2:** Ensure the instance can reach `https://api.pharmgkb.org` (no firewall/proxy blocking, no TLS issues). Run `python 9_dashboard_visuals/cohort_pgx/test_pharmgkb_throttle.py --inspect CYP2D6` on EC2 to see the actual response.
- **Logging:** Use the improved error logging in `fetch_vip_reports.py` (status code and response snippet on failure) to see why each request fails on EC2.
- **Retries:** If the issue is transient (e.g. throttling), add retries with backoff in `_get()`.
