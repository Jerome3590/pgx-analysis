# Lambda Container Deployment Guide (ECR)

This guide covers deploying the PGx Risk Dashboard using AWS Lambda with container images (ECR), which supports up to **10GB container images** for bundling models directly.

## Prerequisites

- AWS CLI configured with appropriate credentials
- Docker installed and running
- AWS account with permissions to:
  - Create ECR repositories
  - Create/update Lambda functions
  - Create API Gateway endpoints
  - Access S3 bucket (`pgxdatalake`)

## Quick Start

```bash
# 1. Prepare models (from repository root)
python 10_risk_dashboard/data_preparation/prepare_models.py --all

# 2. Build and push container image (from 10_risk_dashboard/deployment or as documented there)
./docker_build.sh

# 3. Create Lambda function (see below)
```

## Step-by-Step Deployment

### Step 1: Prepare Models

```bash
# From repository root
python 10_risk_dashboard/data_preparation/prepare_models.py --all
```

This creates the packaged model layout under `10_risk_dashboard/outputs/models/`:
```
10_risk_dashboard/outputs/models/
├── opioid_ed/
│   ├── 13_24/
│   │   ├── catboost.joblib
│   │   ├── xgboost.joblib
│   │   ├── xgboost_rf.joblib
│   │   └── feature_schema.json
│   └── ...
└── non_opioid_ed/
    └── ...
```

### Step 2: Build Container Image

**Option A: Using the build script**
```bash
# Set environment variables
export AWS_REGION=us-east-1
export AWS_ACCOUNT_ID=123456789012
export ECR_REPOSITORY=pgx-risk-dashboard
export IMAGE_TAG=latest

# Make script executable
chmod +x docker_build.sh

# Build and push
./docker_build.sh
```

**Option B: Manual build**
```bash
# Build image
docker build -t pgx-risk-dashboard:latest .

# Login to ECR
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin \
  123456789012.dkr.ecr.us-east-1.amazonaws.com

# Create repository (if needed)
aws ecr create-repository \
  --repository-name pgx-risk-dashboard \
  --region us-east-1

# Tag image
docker tag pgx-risk-dashboard:latest \
  123456789012.dkr.ecr.us-east-1.amazonaws.com/pgx-risk-dashboard:latest

# Push image
docker push \
  123456789012.dkr.ecr.us-east-1.amazonaws.com/pgx-risk-dashboard:latest
```

### Step 3: Create Lambda Function

```bash
# Create execution role (if not exists)
aws iam create-role \
  --role-name pgx-lambda-execution-role \
  --assume-role-policy-document '{
    "Version": "2012-10-17",
    "Statement": [{
      "Effect": "Allow",
      "Principal": {"Service": "lambda.amazonaws.com"},
      "Action": "sts:AssumeRole"
    }]
  }'

# Attach policies
aws iam attach-role-policy \
  --role-name pgx-lambda-execution-role \
  --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole

# S3: Lambda must read models/metadata from pgxdatalake (gold/dashboard/*, gold/ffa_analysis/*, etc.)
# Option A: Broad read (simple)
aws iam attach-role-policy \
  --role-name pgx-lambda-execution-role \
  --policy-arn arn:aws:iam::aws:policy/AmazonS3ReadOnlyAccess

# Option B: If using a custom role (e.g. pgx-lambda-role) without S3 access, add an inline policy.
# See 10_risk_dashboard/docs/LAMBDA_IAM_POLICY_S3.md for the exact policy document.

# Create Lambda function
aws lambda create-function \
  --function-name pgx-risk-dashboard-api \
  --package-type Image \
  --code ImageUri=123456789012.dkr.ecr.us-east-1.amazonaws.com/pgx-risk-dashboard:latest \
  --role arn:aws:iam::123456789012:role/pgx-lambda-execution-role \
  --timeout 60 \
  --memory-size 3008 \
  --environment Variables="{PGX_RESULTS_BUCKET=pgxdatalake,MODEL_CACHE_TTL=3600}" \
  --region us-east-1
```

### Step 4: Create API Gateway

```bash
# Create HTTP API
aws apigatewayv2 create-api \
  --name pgx-risk-dashboard-api \
  --protocol-type HTTP \
  --cors-configuration AllowOrigins="*",AllowMethods="GET,POST,OPTIONS",AllowHeaders="Content-Type"

# Create integration
aws apigatewayv2 create-integration \
  --api-id <API_ID> \
  --integration-type AWS_PROXY \
  --integration-uri arn:aws:lambda:us-east-1:123456789012:function:pgx-risk-dashboard-api \
  --payload-format-version "2.0"

# Create routes
aws apigatewayv2 create-route \
  --api-id <API_ID> \
  --route-key "GET /metadata" \
  --target integrations/<INTEGRATION_ID>

aws apigatewayv2 create-route \
  --api-id <API_ID> \
  --route-key "POST /risk" \
  --target integrations/<INTEGRATION_ID>

aws apigatewayv2 create-route \
  --api-id <API_ID> \
  --route-key "POST /risk/comparison" \
  --target integrations/<INTEGRATION_ID>

# Deploy
aws apigatewayv2 create-stage \
  --api-id <API_ID> \
  --stage-name prod \
  --auto-deploy
```

### Step 5: Update Dashboard

Update `API_BASE` in `index.html`:
```javascript
const API_BASE = "https://<API_ID>.execute-api.us-east-1.amazonaws.com/prod";
```

### Step 6: Deploy Dashboard to S3

```bash
# Upload dashboard
aws s3 cp index.html s3://pgxdatalake/dashboard/index.html \
  --content-type text/html

# Enable static website hosting
aws s3 website s3://pgxdatalake/dashboard/ \
  --index-document index.html

# Set bucket policy (for public read access)
aws s3api put-bucket-policy --bucket pgxdatalake \
  --policy file://bucket-policy.json
```

## Container Image Size

With all models bundled, the container image will be approximately:
- **Base image**: ~500MB (Python 3.10 Lambda base)
- **Dependencies**: ~2-3GB (CatBoost, XGBoost, pandas, numpy, etc.)
- **Models**: ~1-5GB (depending on number of cohorts/age_bands)
- **Total**: ~4-8GB (well within 10GB limit)

## Model Loading Strategy

The Lambda function uses a two-tier loading strategy:

1. **Primary**: Load from container filesystem (`/var/task/models/`)
   - Fastest (no network latency)
   - Models bundled in image
   - Available immediately on cold start

2. **Fallback**: Load from S3 (`s3://pgxdatalake/gold/dashboard/models/`)
   - Used if container models not available
   - Useful for development/testing
   - Slower but more flexible

## Environment Variables

- `PGX_RESULTS_BUCKET`: S3 bucket name (default: `pgxdatalake`)
- `MODEL_CACHE_TTL`: Model cache TTL in seconds (default: `3600`)
- `MODEL_BASE_PATH`: Path to models in container (default: `/var/task/models`)

## Updating the Function

```bash
# After making code changes, rebuild and push
./docker_build.sh

# Update Lambda function
aws lambda update-function-code \
  --function-name pgx-risk-dashboard-api \
  --image-uri 123456789012.dkr.ecr.us-east-1.amazonaws.com/pgx-risk-dashboard:latest
```

## Monitoring

```bash
# View logs
aws logs tail /aws/lambda/pgx-risk-dashboard-api --follow

# Check function metrics
aws cloudwatch get-metric-statistics \
  --namespace AWS/Lambda \
  --metric-name Duration \
  --dimensions Name=FunctionName,Value=pgx-risk-dashboard-api \
  --start-time 2024-01-01T00:00:00Z \
  --end-time 2024-01-02T00:00:00Z \
  --period 3600 \
  --statistics Average,Maximum
```

## Troubleshooting

### Container too large
- Check model sizes: `du -sh models/*`
- Consider model quantization
- Remove unused dependencies

### Cold start too slow
- Increase Lambda memory (faster CPU)
- Use provisioned concurrency
- Optimize model loading (already cached after first load)

### Model loading fails
- Check container filesystem: `ls -la /var/task/models/`
- Verify model paths match age_band format (13_24 vs 13-24)
- Check Lambda logs for specific errors

## Cost Optimization

- **Provisioned Concurrency**: Only if needed for low latency requirements
- **Memory**: Start with 3008MB, adjust based on performance
- **Timeout**: 60 seconds is sufficient for most requests
- **Container Reuse**: Models cached in memory across invocations (free)

## Security

- Use IAM roles for Lambda execution (not access keys)
- Restrict S3 bucket access to specific prefixes
- Enable API Gateway authentication if needed
- Use VPC endpoints for S3 access (optional, reduces internet egress)

## References

- [Lambda Container Images](https://docs.aws.amazon.com/lambda/latest/dg/images-create.html)
- [ECR Best Practices](https://docs.aws.amazon.com/AmazonECR/latest/userguide/best-practices.html)
- [Lambda Performance](https://docs.aws.amazon.com/lambda/latest/dg/performance.html)

