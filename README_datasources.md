# Data sources (APCD, Athena, S3)

Mushin account `535362115856`, region `us-east-1`, CLI profile `mushin` or `pgx`.
Inventory verified 26 Sep 2026.

## Strategy

Keep **two durable APCD layers** on `pgxdatalake`. Do not store intermediates,
vendor text, FDA mirrors, or lake snapshots.

| Keep | Why |
|------|-----|
| **Bronze parquet** | Full vendor column set. Rebuild source. Query this when you need every APCD field. |
| **Gold** | Filtered working set for cohorts, models, SHAP/FFA, and the dashboard. |

| Do not keep | Why |
|-------------|-----|
| Vendor `.txt` (`Medical/`, `Pharmacy/`, `bronze/_staging/`) | Bronze parquet already has those quarters. |
| **Silver** (imputed and `*_raw`) | Gold is the working set; bronze has all columns. Rebuild silver from bronze only if gold must be remade. |
| **FAERS copies** | Public extracts on the FDA site. Re-download if a notebook needs them. |
| **`pgxdatalake-backups`** | Stale gold/silver/FAERS snapshot. Not a second source of truth. |

**Query rule:** full APCD columns → bronze. Pipeline / cohorts → gold. Do not use
gold pharmacy (8 columns) or old silver paths when you need NPI, paid, ICD
descriptions, NDC/GPI, or days supply.

**Rebuild gold** (only if current gold is missing or invalid):

1. `1a_apcd_input_data/2_global_imputation.py` from `bronze/medical` and `bronze/pharmacy`
2. `1a_apcd_input_data/3_apcd_clean.py` (script default still says `silver/imputed/`; that prefix is empty until step 1 writes it)

**Athena:** workgroup `APCD` for the lake. Results bucket
`aws-athena-query-results-us-east-1-535362115856` has **no 30-day expire** until
the external AIM3 / HIV analysis is confirmed complete. Then put expire back.

**Dashboard:** `s3://jerome-dixon.io/pgx/` only (`pgx.jerome-dixon.io`). Do not
recreate `/vcu/pgx-risk-calculator/`.

## Buckets this repo uses

| Bucket | Role |
|--------|------|
| `pgxdatalake` | Bronze + gold APCD, cohorts, models, dashboard artifacts |
| `pgx-repository` | Pipeline logs and checkpoints (not APCD extracts) |
| `jerome-dixon.io` | Live dashboard under `pgx/` |
| `mushin-solutions-project-metadata` | Notebook output pointers |
| `aws-athena-query-results-us-east-1-535362115856` | Athena `APCD` / `primary` results and CTAS |

`pgx-repository` and `jerome-dixon.io` expire **noncurrent versions** after 30 days
(current objects stay).

## Bronze (full columns)

Vendor Genomic_Screening extracts, Q1 2016–Q4 2020 (20 medical + 20 RX files).
Headers match `1a_apcd_input_data/0_txt_to_parquet.py` (113 medical, 51 pharmacy).
Original column names (spaces).

| Layer | S3 prefix | Columns | Athena table |
|-------|-----------|---------|--------------|
| Bronze medical | `s3://pgxdatalake/bronze/medical/` | 113 | `medical_raw.medical` |
| Bronze pharmacy | `s3://pgxdatalake/bronze/pharmacy/` | 51 | `bronze_pharmacy.pharmacy` |

```sql
SELECT * FROM medical_raw.medical LIMIT 10;
SELECT * FROM bronze_pharmacy.pharmacy LIMIT 10;
```

## Gold (filtered working set)

| Layer | S3 prefix | Columns | Athena table |
|-------|-----------|---------|--------------|
| Gold medical | `s3://pgxdatalake/gold/medical/` | 69 | `medical.medical` |
| Gold pharmacy | `s3://pgxdatalake/gold/pharmacy/` | 8 | `pharmacy.pharmacy` / `gold_pharmacy.pharmacy` |

Gold pharmacy is person + drug + date (`mi_person_key`, `drug_name`,
`standardized_drug_name`, `incurred_date`, plus partition / source flags).

Downstream pipeline gold:

- `s3://pgxdatalake/gold/cohorts/`
- `s3://pgxdatalake/gold/cohorts_model_data/`
- `s3://pgxdatalake/gold/dashboard/`
- `s3://pgxdatalake/gold/dtw_filter/`
- `s3://pgxdatalake/gold/feature_importance/`

```sql
SELECT * FROM medical.medical LIMIT 10;
SELECT * FROM pharmacy.pharmacy LIMIT 10;
```

## Removed 26 Sep 2026

Do not point new jobs at these.

- `s3://pgxdatalake/Medical/` and `Pharmacy/` — vendor `.txt`
- `s3://pgxdatalake/bronze/_staging/` — duplicate TXT
- `s3://pgxdatalake/bronze/FAERS/` — re-download from FDA if needed
- `s3://pgxdatalake/silver/` — rebuild from bronze only to remake gold
- `s3://pgxdatalake/pgx_pipeline/` — old intermediates
- Bucket `pgxdatalake-backups`
- `jerome-dixon.io/vcu/pgx-risk-calculator/` — use `pgx/`
- Unused buckets: `fda-ade-pgx`, `plotly-demo`, `athena-output-testing-pgx`,
  `aws-athena-query-results-535362115856-us-east-1`, empty drone-simulation
  and inbox buckets

`0_txt_to_parquet.py` still defaults to `s3://pgxdatalake/Medical/`. That path
is empty. Start from bronze parquet.

`py_helpers/faers_time_to_onset.py` still defaults to the deleted backups FAERS
prefix. Stage FDA extracts again if that QA is rerun.

Stale Glue name `pgxdatalake.pharmacy_partitioned` may still exist (Lake
Formation blocked drop). Files behind it are gone.

## Related docs

- Preprocessing flow: `1a_apcd_input_data/README.md`
- Pipeline architecture: `docs/Step1-2_DataPipeline/README_data_pipeline.md`
- EC2 sessions: `.cursor/rules/ec2.mdc`
