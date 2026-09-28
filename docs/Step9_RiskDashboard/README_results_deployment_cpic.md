# CPIC Data Deployment Guide

## Overview

The PGx Patient Card uses official CPIC gene–drug pairs. Monthly refresh writes Parquet to S3; Lambda honors `PREFER_S3` and an in-process cache so `/pgx/card` does not reload S3 on every request.

Official sources only — do not invent phenotype or recommendation rows.

## Canonical S3 layout

| Object | Purpose |
|---|---|
| `s3://pgxdatalake/gold/reference/cpic/manifest.json` | Source URL, `retrieved_at`, ETag, `row_count`, `content_sha256`, `cpicKnowledgeVersion` |
| `s3://pgxdatalake/gold/reference/cpic/*.parquet` | Official CPIC tables (pairs, diplotype/phenotype, recommendations, alleles, …) |
| `s3://pgxdatalake/gold/reference/cpic/versions/{stamp}/` | Immutable snapshot of the same files |
| `s3://pgxdatalake/gold/reference/pharmgkb/` | Official ClinPGx/PharmGKB tables when the ingest retrieved them |
| `s3://pgxdatalake/gold/dashboard/data/cpic_gene-drug_pairs.parquet` | Live PGx Card mirror (also `.xlsx`) |

Phenotype: if `gold/reference/cpic/cpic_diplotype_phenotype.parquet` exists, Lambda loads official gene + diplotype → phenotype columns only. Otherwise it keeps the hardcoded 10-gene table (`pgx-phenotype-v1`).

## Monthly refresh (preferred)

Not a manual wget + Docker rebuild. EventBridge invokes `pgx-cpic-reference-refresh` on `cron(0 12 1 * ? *)` (1st of the month, 12:00 UTC). The job HEADs official files, compares the manifest, skips (optional SES) when unchanged, otherwise downloads, writes Parquet, uploads current + versioned objects, and emails via `py_helpers.aws_utils.send_status_email_ses`.

```bash
# Laptop / any machine with AWS creds (first run or catch-up)
python utility_scripts/refresh_cpic_reference.py
python utility_scripts/refresh_cpic_reference.py --pairs-only
python utility_scripts/refresh_cpic_reference.py --check-only

# After the dedicated Lambda is deployed
aws lambda invoke --function-name pgx-cpic-reference-refresh --cli-binary-format raw-in-base64-out --payload "{\"pairs_only\":false}" cpic-refresh-out.json
```

IaC and deploy: `aws-pgx-setup/lambda/cpic_refresh/README.md`.

Do **not** use GitHub Actions as the datalake writer, and do **not** schedule this on EC2 Spot.

## Container packaging (still used as PREFER_S3 fallback)

From the **repository root**:

```bash
python 10_risk_dashboard/data_preparation/prepare_cpic_data.py
```

This stages `10_risk_dashboard/outputs/cpic/cpic_gene-drug_pairs.xlsx` (and `.parquet` when possible) for the Docker image at `/var/task/data/`.

The one-shot official-table ingest (same prefixes as the monthly job) is:

```bash
python 10_risk_dashboard/data_preparation/ingest_cpic_pharmgkb_reference.py
```

## Loading priority

`load_cpic_data()` honors `PREFER_S3` (same as metadata/models):

1. When `PREFER_S3=true` (required for monthly S3 refreshes to reach the live card): S3 mirror / canonical Parquet, then container
2. When `PREFER_S3=false`: container first, then S3
3. `cpicKnowledgeVersion` is read from `gold/reference/cpic/manifest.json` when present

Set `PREFER_S3=true` on `pgx-risk-calculator`. After Python-only Lambda edits, use the existing code-only deploy (no dashboard image rebuild):

```bash
aws s3 cp 10_risk_dashboard/backend/lambda_function.py s3://pgxdatalake/gold/dashboard/code/lambda_function.py
aws s3 cp 10_risk_dashboard/backend/cpic_allele_resolver.py s3://pgxdatalake/gold/dashboard/code/cpic_allele_resolver.py
```

Then bump `DEPLOY_TS` on the function **without** dropping other environment variables:

```bash
aws lambda update-function-configuration --function-name pgx-risk-calculator --environment "Variables={S3_BUCKET=pgxdatalake,CODE_S3_KEY=gold/dashboard/code/lambda_function.py,DEPLOY_TS=YYYYMMDDHHMMSS,CODE_OVERRIDE_VERSION=YYYYMMDDHHMMSS,PREFER_S3=true,S3_DASHBOARD_BUCKET=jerome-dixon.io,PGX_RESULTS_BUCKET=pgxdatalake,S3_DASHBOARD_PREFIX=pgx,CPIC_RESOLVER_S3_KEY=gold/dashboard/code/cpic_allele_resolver.py}"
```

`POST /pgx/card` accepts `genotypes: [{rsid, genotype}]` and resolves star alleles from official `cpic_allele_rsid.csv` (preferred on the live image, which has no pyarrow) or `.parquet`. Manual `variants: [{gene, variants}]` still skips rsid calling.

The live image cannot write `/var/task`. `entrypoint.sh` now downloads overrides to `/tmp/pgx_code` and prepends `PYTHONPATH`. That entrypoint change needs an ECR rebuild to take effect; until then, `lambda_function.py` also loads `cpic_allele_resolver.py` from S3 into `/tmp`.

## Verification

```bash
curl -X POST https://cmv0qislq3.execute-api.us-east-1.amazonaws.com/prod/pgx/card \
  -H "Content-Type: application/json" \
  -d '{
    "patient_id": "TEST001",
    "variants": [
      {"gene": "CYP2D6", "variants": ["*1", "*2"]},
      {"gene": "CYP2C19", "variants": ["*1", "*17"]}
    ]
  }'
```

Expected response includes genes, matched drugs, CPIC guideline URLs, and `versions.cpicKnowledgeVersion`.

## Troubleshooting

### Excel / Parquet not found
- Confirm `python utility_scripts/refresh_cpic_reference.py` uploaded the dashboard mirror
- Confirm `pgx-lambda-role` has `PgxCpicReferenceRead` (`gold/reference/*` and `gold/dashboard/data/*`)
- Confirm `PREFER_S3=true` if you expect S3 to win over the baked-in container file

### pandas/openpyxl import errors
- Container fallback still needs `openpyxl>=3.1.0` in `10_risk_dashboard/backend/requirements.txt`

### Column detection issues
- Gene/drug columns are matched case-insensitively
- Official phenotype parquet is used only when gene + diplotype + phenotype/generesult columns are present
