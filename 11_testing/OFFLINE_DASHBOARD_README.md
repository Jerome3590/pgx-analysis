---
description: Offline testing guide for PGx Risk Dashboard
---

# Offline Dashboard Testing (Airplane Mode)

This guide explains how to download all required artifacts and run the PGx Risk Dashboard **fully offline**, including a local API that mimics **API Gateway + Lambda**.

## 1) One-time prep (while online)

### 1.1 Create / activate the offline virtual environment

```powershell
python -m venv .venv-offline
.\.venv-offline\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install boto3 botocore joblib numpy pandas catboost xgboost openpyxl duckdb
```

### 1.2 Download artifacts + models (S3 sync)

This pulls:

- Dashboard static files from `s3://jerome-dixon.io/vcu/pgx-risk-calculator/` into `11_testing/offline_dashboard/s3/`
- Lambda model artifacts from `s3://pgxdatalake/gold/dashboard/models/` into `11_testing/offline_dashboard/models/`
- Code snapshots into `11_testing/offline_dashboard/code/`

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\11_testing\offline_dashboard_download.ps1
```

Optional overrides:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\11_testing\offline_dashboard_download.ps1 `
  -AwsProfile mushin `
  -Region us-east-1 `
  -BucketPrefix "s3://jerome-dixon.io/vcu/pgx-risk-calculator/" `
  -ModelBucket "pgxdatalake" `
  -ModelPrefix "gold/dashboard/models" `
  -OutDir "11_testing/offline_dashboard"
```

### 1.3 Verify bundle contents

Expected key files:

- `11_testing/offline_dashboard/s3/index.html`
- `11_testing/offline_dashboard/s3/metadata/opioid_ed.json`
- `11_testing/offline_dashboard/s3/metadata/non_opioid_ed.json`
- `11_testing/offline_dashboard/s3/visualizations/dashboard_visual_objects.json`
- `11_testing/offline_dashboard/models/<cohort>/<age_band>/feature_schema.json`
- `11_testing/offline_dashboard/models/<cohort>/<age_band>/*.joblib` and/or `catboost.json`

## 2) Run everything offline (static site + local API)

Start the offline server (serves static dashboard + `/prod/*` API routes):

```powershell
.\.venv-offline\Scripts\Activate.ps1
python .\11_testing\offline_dashboard_server.py
```

Open in browser:

```text
http://127.0.0.1:8000/index.html?apiBase=http://127.0.0.1:8000/prod&staticBase=http://127.0.0.1:8000
```

Notes:

- `staticBase` forces the dashboard to load JSON/PNG/HTML assets from the local mirror under `11_testing/offline_dashboard/s3/`.
- `apiBase` forces the dashboard to call your local API Gateway/Lambda emulator under `/prod`.

## 3) API quick tests (curl)

Use these to confirm the local API is working.

### 3.1 Metadata

```powershell
curl "http://127.0.0.1:8000/prod/metadata?cohort=opioid_ed"
```

### 3.2 Risk (example)

```powershell
$body = @{
  age = 55
  cohort = "non_opioid_ed"
  age_band = "55-64"
  drugs = @("METFORMIN")
  icds = @()
  cpts = @()
  n_drugs = 7
  pgx_num_cpic_drugs = 2
} | ConvertTo-Json

curl -Method POST "http://127.0.0.1:8000/prod/risk" -ContentType "application/json" -Body $body
```

Expected in response:

- `risk_score`
- `risk_band`
- `model_breakdown` (per-model probabilities)
- `model_inputs` (echoes `n_drugs`, `pgx_num_cpic_drugs`)

## 4) Tab-by-tab offline test checklist

Open the dashboard (URL above), then run these checks.

### 4.1 Risk Assessment

- **Inputs**
  - Enter age (13–114)
  - Optionally set `# drugs` and `# CPIC drugs` (per mi_person_key)
- **Expected**
  - Risk score renders
  - "Risk Score by Model" shows per-model bars **with percentage labels**
  - Selected codes are echoed in the output

### 4.2 Drugs / ICD Codes / CPT Codes

- **Expected**
  - Dropdowns show real values (not `undefined`)
  - Search filters options
  - Selected chips update

### 4.3 Feature Importance

- Click **Load** for a cohort + age band.
- **Expected**
  - PNG or JSON-driven Plotly renders (depending on the selection logic)
  - If a file is missing, you get a clear message (not a blank panel)

### 4.4 Causal Analysis

- Load the tab with a cohort + age band.
- **Expected**
  - If static JSON exists locally, it renders
  - Otherwise the panel shows an explicit message (no infinite spinners)

### 4.5 BupaR Process Mining

- Load cohort + age band.
- **Expected**
  - Images/Plotly render from local `visualizations/bupar/...`

### 4.6 DTW Trajectories

- Load cohort + age band.
- **Expected**
  - Images/JSON render from local `visualizations/dtw/...`

### 4.7 FP-Growth Patterns

- Load cohort + age band.
- **Expected**
  - Support distribution chart renders
  - Itemset labels:
    - strip `DRUG:` prefix
    - normalize names
    - `,` replaced with ` : ` for multi-drug itemsets
  - Network renders in iframe via local API route (`/prod/visualizations/fpgrowth/network_html?...`)

### 4.8 PGx Cohort

- Load cohort + age band.
- **Expected**
  - Network topology loads from the local mirror under `visualizations/cohort_pgx/...`

## 5) Common offline failure modes

- **Model inference fails**
  - Ensure models were synced into `11_testing/offline_dashboard/models/`
  - Ensure packages are installed in `.venv-offline`
- **Blank charts**
  - Open DevTools Console and check for missing JSON/PNG paths
  - Confirm you opened the dashboard with both query params: `apiBase` and `staticBase`

## 6) Files involved

- **Downloader**: `11_testing/offline_dashboard_download.ps1`
- **Offline server**: `11_testing/offline_dashboard_server.py`
- **Static mirror**: `11_testing/offline_dashboard/s3/`
- **Models mirror**: `11_testing/offline_dashboard/models/`
- **Code snapshot**: `11_testing/offline_dashboard/code/`
