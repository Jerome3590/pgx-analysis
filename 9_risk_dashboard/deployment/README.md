# Deployment

## Overview

Scripts and configurations for deploying the dashboard to AWS.

## Files

- **`docker_build.sh`** - Build and push Docker image to ECR
- **`prepare_lambda_dir.py`** - Prepare Lambda deployment package
- **`scripts/`** - Additional deployment helper scripts

## Deployment Steps

1. **Prepare Models and Metadata**:
   ```bash
   cd ../data_preparation
   python prepare_models.py --all
   python generate_metadata.py --all
   python prepare_cpic_data.py
   ```

2. **Build Docker Image**:
   ```bash
   ./docker_build.sh
   ```

3. **Create Lambda Function**:
   - Use ECR image created in step 2
   - Configure API Gateway integration
   - Set environment variables

4. **Deploy Frontend**:
   - **S3 location:** `s3://jerome-dixon.io/vcu/pgx-risk-calculator/`
   - Upload `../frontend/index.html` and assets to that prefix (e.g. `aws s3 sync ../frontend/ s3://jerome-dixon.io/vcu/pgx-risk-calculator/`)
   - Configure the bucket for static website hosting (or use CloudFront with that origin)
   - (Optional) Set up CloudFront distribution

## Architecture

```
User Browser
    ↓
S3 Static Website (frontend/index.html)
    ↓
API Gateway
    ↓
Lambda Function (ECR Container)
    ├── Models (from /var/task/models/)
    ├── Metadata (from /var/task/metadata/)
    └── CPIC Data (from /var/task/cpic/)
```

## See Also

- **Backend README**: `../backend/README.md`
- **Frontend README**: `../frontend/README.md`
- **Main Dashboard README**: `../README.md`
