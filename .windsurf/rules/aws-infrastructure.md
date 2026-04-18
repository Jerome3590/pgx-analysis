---
description: AWS infrastructure metadata for the PGx Risk Dashboard — account resources, IDs, and deployment commands
---

## S3 Buckets

| Bucket | Purpose |
|---|---|
| `jerome-dixon.io` | Dashboard frontend static hosting + visualization assets |
| `pgxdatalake` | Model artifacts, analysis outputs, Lambda code |

**Dashboard frontend prefix:** `s3://jerome-dixon.io/vcu/pgx-risk-calculator/`

## CloudFront Distribution

| Distribution ID | Alias | Origin |
|---|---|---|
| `E3MZK5HYTJ14P3` | `jerome-dixon.io` | `jerome-dixon.io.s3-website-us-east-1.amazonaws.com` |

## AWS Account

| Resource | Value |
|---|---|
| Account ID | `535362115856` |
| Region | `us-east-1` |

## API Gateway

| Resource | Value |
|---|---|
| API ID | `cmv0qislq3` |
| Base URL | `https://cmv0qislq3.execute-api.us-east-1.amazonaws.com/prod` |
| Stage | `prod` |

## Lambda

| Resource | Value |
|---|---|
| Function name | `pgx-risk-calculator` |
| Runtime | Container image (ECR) |
| Code-only S3 key | `s3://pgxdatalake/gold/dashboard/code/lambda_function.py` |
| Environment vars | `S3_BUCKET=pgxdatalake`, `PGX_RESULTS_BUCKET=pgxdatalake`, `S3_DASHBOARD_BUCKET=jerome-dixon.io` |

## ECR

| Resource | Value |
|---|---|
| Repository | `pgx-risk-calculator` |
| Registry | `535362115856.dkr.ecr.us-east-1.amazonaws.com` |
| Image URI | `535362115856.dkr.ecr.us-east-1.amazonaws.com/pgx-risk-calculator:latest` |

## EC2

Both instances are **spot**, type `x2iedn.8xlarge`, region `us-east-1`.

| Instance ID | Name | Purpose | Lifecycle |
|---|---|---|---|
| `i-0e7d1bd469620c0bb` | `pgx-analysis-1a` | Model training + artifact generation (notebooks 3–4) | spot |
| `i-0c968462d413a1028` | `pgx-dashboard-1b` | Dashboard Docker builds + deployment | spot |

| Resource | Value |
|---|---|
| User / home | `pgx3874` · `/home/pgx3874/pgx-analysis` |
| Python env | `/home/pgx3874/jupyter-env/bin/python3.11` |
| Last known IP (`pgx-dashboard-1b`) | `54.235.245.123` (dynamic — changes on start) |

**Spot price history — `x2iedn.8xlarge` Linux/UNIX (as of 2026-04-18):**

| AZ | Price/hr | Cheapest? |
|---|---|---|
| `us-east-1b` | $2.6125 | ✅ lowest |
| `us-east-1c` | $2.8278 | |
| `us-east-1a` | $2.8784 | |
| `us-east-1d` | $3.0061 | |
| `us-east-1f` | $3.2598 | highest |

> Refresh: `aws ec2 describe-spot-price-history --region us-east-1 --instance-types x2iedn.8xlarge --product-descriptions "Linux/UNIX" --max-items 10 --query "SpotPriceHistory[*].{AZ:AvailabilityZone,Price:SpotPrice,Time:Timestamp}" --output table`

## Dashboard URL

`https://jerome-dixon.io/vcu/pgx-risk-calculator/index.html`

## Standard Deployment Commands

```bash
# Frontend — sync + invalidate index.html (use after index.html or tab HTML changes)
python3 10_risk_dashboard/deployment/sync_frontend_to_s3.py
aws cloudfront create-invalidation --distribution-id E3MZK5HYTJ14P3 --paths "/vcu/pgx-risk-calculator/index.html"

# Frontend — full invalidation (use after tab HTML or asset changes)
aws cloudfront create-invalidation --distribution-id E3MZK5HYTJ14P3 --paths "/vcu/pgx-risk-calculator/*"

# Lambda — code-only update (no Docker rebuild, fastest)
aws s3 cp 10_risk_dashboard/backend/lambda_function.py s3://pgxdatalake/gold/dashboard/code/lambda_function.py
aws lambda update-function-configuration --function-name pgx-risk-calculator --environment "Variables={S3_BUCKET=pgxdatalake,CODE_S3_KEY=gold/dashboard/code/lambda_function.py,PREFER_S3=false,PGX_RESULTS_BUCKET=pgxdatalake}"

# Lambda — full rebuild + ECR push (Windows PS7, requires Docker Desktop)
pwsh.exe -ExecutionPolicy Bypass -File "/mnt/c/Projects/pgx-analysis/10_risk_dashboard/deployment/scripts/build_and_push.ps1"

# Lambda — full rebuild + ECR push (EC2)
bash 10_risk_dashboard/deployment/docker_build.sh
```
