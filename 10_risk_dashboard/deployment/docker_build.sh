#!/bin/bash
# Build and push Lambda container image to ECR

set -e

# Configuration
AWS_REGION=${AWS_REGION:-us-east-1}
AWS_ACCOUNT_ID=${AWS_ACCOUNT_ID:-YOUR_ACCOUNT_ID}
ECR_REPOSITORY=${ECR_REPOSITORY:-pgx-risk-calculator}
IMAGE_TAG=${IMAGE_TAG:-latest}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Building Lambda container image...${NC}"

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DASHBOARD_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$DASHBOARD_ROOT"

# Step 1: Prepare lambda_dir.
# Auto-detect environment:
#   EC2  (/mnt/nvme present OR 6_final_model/outputs exists) → local paths only (--no-s3)
#   Windows / CI (no local training outputs)                  → S3 (default, no flag)
PREPARE_FLAGS=""
FINAL_MODEL_OUTPUTS="${DASHBOARD_ROOT}/../6_final_model/outputs"
if [ -d "/mnt/nvme" ] || [ -d "${FINAL_MODEL_OUTPUTS}" ]; then
    echo -e "${GREEN}EC2 detected (local training outputs found) — using local paths${NC}"
    PREPARE_FLAGS="--no-s3"
else
    echo -e "${YELLOW}Windows/CI detected — pulling models from S3${NC}"
fi

python deployment/prepare_lambda_dir.py ${PREPARE_FLAGS}
if [ $? -ne 0 ]; then
    echo -e "${RED}ERROR: Model preparation failed.${NC}"
    if [ -n "${PREPARE_FLAGS}" ]; then
        echo -e "${RED}Local EC2 paths not found. Run training pipeline (notebook 3) first.${NC}"
    else
        echo -e "${RED}S3 models not found. Run notebook 5 (prepare_models.py --upload-s3) on EC2 first.${NC}"
    fi
    exit 1
fi

# Step 2: Build Docker image (Dockerfile is in backend/, build from dashboard root)
echo -e "${GREEN}Building Docker image...${NC}"
docker build -t ${ECR_REPOSITORY}:${IMAGE_TAG} -f backend/Dockerfile .

# Step 3: Get ECR login token
echo -e "${GREEN}Logging in to ECR...${NC}"
aws ecr get-login-password --region ${AWS_REGION} | docker login --username AWS --password-stdin ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com

# Step 4: Create ECR repository if it doesn't exist
echo -e "${GREEN}Checking ECR repository...${NC}"
aws ecr describe-repositories --repository-names ${ECR_REPOSITORY} --region ${AWS_REGION} || \
    aws ecr create-repository --repository-name ${ECR_REPOSITORY} --region ${AWS_REGION}

# Step 5: Tag image for ECR
ECR_URI=${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPOSITORY}:${IMAGE_TAG}
docker tag ${ECR_REPOSITORY}:${IMAGE_TAG} ${ECR_URI}

# Step 6: Push to ECR
echo -e "${GREEN}Pushing image to ECR...${NC}"
docker push ${ECR_URI}

echo -e "${GREEN}✓ Image pushed successfully: ${ECR_URI}${NC}"
echo -e "${YELLOW}Next steps:${NC}"
echo "  1. Create Lambda function from container image"
echo "  2. Set environment variables:"
echo "     - PGX_RESULTS_BUCKET=pgxdatalake"
echo "     - S3_DASHBOARD_BUCKET=jerome-dixon.io   (bucket where frontend is deployed; FP-Growth URLs)"
echo "     - S3_DASHBOARD_PREFIX=vcu/pgx-risk-calculator"
echo "     - MODEL_CACHE_TTL=3600"
echo "  3. Configure API Gateway to use this Lambda"

