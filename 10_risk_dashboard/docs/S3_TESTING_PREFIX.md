# S3 testing prefix – sync for local analysis

**Production path:** The API (Lambda) looks for Cohort PGx at  
`{S3_DASHBOARD_PREFIX}/cohort_pgx/networks/{cohort}/{age_band}/network_topology.html` (S3 uses hyphen, e.g. `25-44`).  
Upload via notebook 5 Step 6 (runs `sync_cohort_pgx_to_s3.py`) so EC2 dirs `25_44` map to S3 keys `25-44`.

The **testing** prefix (`vcu/pgx-risk-calculator/testing/`) was used only to sync a copy of assets to the local project for analysis. It is not the path the API uses.

## Syncing testing prefix to local

To pull a copy from the **testing** prefix into the repo for inspection/analysis:

```bash
# From repo root
python 9_dashboard_visuals/sync_testing_from_s3.py
# Optional: list only (no download)
python 9_dashboard_visuals/sync_testing_from_s3.py --list-only
# With AWS profile
python 9_dashboard_visuals/sync_testing_from_s3.py --profile YOUR_PROFILE
```

**Source:** `s3://jerome-dixon.io/vcu/pgx-risk-calculator/testing/`  
**Local destination:** `10_risk_dashboard/visualizations/s3_sync_testing/` (gitignored)

Contents may include a copy of Cohort PGx network topology and other visuals. For the dashboard to show PGx Cohort, ensure production assets are under `.../cohort_pgx/networks/...` as above.
