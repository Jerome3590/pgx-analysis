param(
  [string]$BucketPrefix = "s3://jerome-dixon.io/vcu/pgx-risk-calculator/",
  [string]$AwsProfile = "mushin",
  [string]$Region = "us-east-1",
  [string]$OutDir = "11_testing/offline_dashboard",
  [string]$ModelBucket = "pgxdatalake",
  [string]$ModelPrefix = "gold/dashboard/models",
  [string]$PgxMetadataPrefix = "gold/dashboard/metadata",
  [string]$PgxDataPrefix = "gold/dashboard/data"
)

$ErrorActionPreference = "Stop"

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $OutDir "s3") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $OutDir "code") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $OutDir "models") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $OutDir "pgx_metadata") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $OutDir "pgx_data") | Out-Null

aws s3 sync $BucketPrefix (Join-Path $OutDir "s3") --profile $AwsProfile --region $Region

# Models + feature schemas used by Lambda risk inference
aws s3 sync ("s3://" + $ModelBucket + "/" + $ModelPrefix + "/") (Join-Path $OutDir "models") --profile $AwsProfile --region $Region

# Lambda-side metadata + data (for full offline API testing: /metrics, /pgx/card)
aws s3 sync ("s3://" + $ModelBucket + "/" + $PgxMetadataPrefix + "/") (Join-Path $OutDir "pgx_metadata") --profile $AwsProfile --region $Region
aws s3 sync ("s3://" + $ModelBucket + "/" + $PgxDataPrefix + "/") (Join-Path $OutDir "pgx_data") --profile $AwsProfile --region $Region

Copy-Item -Force "10_risk_dashboard/frontend/index.html" (Join-Path $OutDir "code/index.html")
Copy-Item -Force "10_risk_dashboard/backend/lambda_function.py" (Join-Path $OutDir "code/lambda_function.py")
Copy-Item -Recurse -Force "10_risk_dashboard/backend" (Join-Path $OutDir "code/backend")

Write-Host "Offline bundle created at: $OutDir"
