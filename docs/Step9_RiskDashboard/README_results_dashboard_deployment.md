# Step 9: Dashboard Deployment Guide

## Overview

The dashboard deployment process supports incremental builds, allowing deployment with partial data. The system gracefully handles missing cohorts and can be updated as more cohorts complete.

## Deployment Architecture

```
┌─────────────┐
│  User Browser│
└──────┬──────┘
       │ HTTPS
       ↓
┌─────────────────┐
│  S3 Static Site │  (index.html, assets)
└──────┬──────────┘
       │ API Calls
       ↓
┌─────────────────┐
│  API Gateway    │
└──────┬──────────┘
       │
       ↓
┌─────────────────┐
│  Lambda (ECR)   │  (lambda_function.py)
└──────┬──────────┘
       │
       ├──→ Models (S3: gold/dashboard/models/)
       ├──→ Metadata (S3: gold/dashboard/metadata/)
       └──→ Visualization Data (S3: gold/{bupar,fpgrowth,dtw_trajectories}/)
```

## Deployment Steps

### 1. Prepare Dashboard Artifacts

#### Option A: Incremental Build (Recommended)

```bash
# Build with available cohorts
See 10_risk_dashboard/deployment/ or archived/utility_scripts/build_dashboard.sh
```

**What it does**:
- Checks which cohorts have completed Step 6 (models exist)
- Prepares models for available cohorts only
- Generates metadata for available cohorts only
- Skips missing cohorts gracefully

#### Option B: Full Build

```bash
cd 10_risk_dashboard

# Prepare all models
python prepare_models.py --all

# Generate all metadata
python generate_metadata.py --all
```

### 2. Build Docker Container

```bash
cd 10_risk_dashboard
./docker_build.sh
```

**What it does**:
- Checks if models directory exists
- If empty, runs `prepare_models.py --all` (skips missing cohorts)
- Builds Docker image with available models
- Tags and pushes to ECR

**Configuration**:
- `AWS_REGION`: AWS region (default: us-east-1)
- `AWS_ACCOUNT_ID`: Your AWS account ID
- `ECR_REPOSITORY`: ECR repository name (default: pgx-risk-dashboard)
- `IMAGE_TAG`: Image tag (default: latest)

### 3. Deploy to AWS

#### Create/Update Lambda Function

```bash
# Create Lambda function from container image
aws lambda create-function \
  --function-name pgx-risk-dashboard \
  --package-type Image \
  --code ImageUri=<ECR_URI> \
  --role <LAMBDA_ROLE_ARN> \
  --timeout 60 \
  --memory-size 3008 \
  --environment Variables='{
    "PGX_RESULTS_BUCKET":"pgxdatalake",
    "MODEL_CACHE_TTL":"3600"
  }'
```

#### Update Existing Function

```bash
aws lambda update-function-code \
  --function-name pgx-risk-dashboard \
  --image-uri <ECR_URI>
```

#### Configure API Gateway

- Create REST API or use existing
- Create resources: `/metadata`, `/risk`, `/risk/comparison`, `/visualizations/{type}`, `/pgx/card`
- Configure CORS
- Deploy to stage (e.g., `prod`)

#### Deploy Static Site to S3

```bash
# Upload dashboard HTML
aws s3 cp 10_risk_dashboard/index.html s3://<BUCKET>/index.html

# Update API_BASE in index.html to point to your API Gateway URL
```

## Incremental Deployment

### How It Works

1. **Build Phase**: Only processes available cohorts
2. **Deployment Phase**: Deploys with whatever models are available
3. **Runtime Phase**: Dashboard handles missing cohorts gracefully

### Benefits

- **No Waiting**: Deploy as cohorts complete
- **Graceful Degradation**: Missing cohorts show helpful errors
- **Incremental Updates**: Rebuild and redeploy as more cohorts finish
- **No Failures**: Dashboard works with partial data

### Example Scenario

**Initial Deployment**:
- Completed: e.g. `opioid_ed/13-24`, `opioid_ed/25-44`
- Not completed: Other cohort/age_band combinations
- **Result**: Dashboard works for completed cohort/age_band combinations; user selects cohort via Opioid ED or Polypharmacy tab

**After More Cohorts Complete**:
- Rebuild: See `10_risk_dashboard/deployment/` or `archived/utility_scripts/build_dashboard.sh`
- Redeploy: `./docker_build.sh` and update Lambda
- **Result**: Dashboard now includes new cohorts

## Environment Variables

### Lambda Function

- `PGX_RESULTS_BUCKET`: S3 bucket name (default: `pgxdatalake`)
- `MODEL_CACHE_TTL`: Model cache TTL in seconds (default: `3600`)
- `MODEL_BASE_PATH`: Path to models in container (default: `/var/task/models`)

### Frontend

- `API_BASE`: API Gateway invoke URL (set in `index.html`)

## Monitoring

### Lambda Metrics

- **Duration**: Should be < 5 seconds for most requests
- **Errors**: Monitor for 4xx/5xx responses
- **Throttles**: Watch for concurrent execution limits

### Dashboard Metrics

- **Page Load Time**: Static site from S3 should be < 1 second
- **API Response Time**: Risk calculation should be < 3 seconds
- **Visualization Load Time**: Depends on data size, typically < 5 seconds

## Troubleshooting

### Models Not Found

**Symptom**: 404 errors for model endpoints

**Solution**:
- Verify models exist in S3: `aws s3 ls s3://pgxdatalake/gold/dashboard/models/`
- Check Lambda has S3 read permissions
- Verify `PGX_RESULTS_BUCKET` environment variable

### Visualizations Not Loading

**Symptom**: Empty visualization panels

**Solution**:
- Check S3 paths for visualization data
- Verify visualization endpoints are configured in API Gateway
- Check browser console for API errors

### CORS Errors

**Symptom**: CORS errors in browser console

**Solution**:
- Verify API Gateway CORS configuration
- Check Lambda response headers include CORS headers
- Ensure `Access-Control-Allow-Origin` is set correctly

## Related Documentation

- **[README_results_dashboard_tabs.md](README_results_dashboard_tabs.md)** - Dashboard tab organization and API endpoints
- **[README_results_dashboard.md](README_results_dashboard.md)** - Complete dashboard system overview
- **[../DASHBOARD_INCREMENTAL_BUILD.md](../DASHBOARD_INCREMENTAL_BUILD.md)** - Incremental build details
- **[README_results_deployment.md](README_results_deployment.md)** - Complete deployment guide

