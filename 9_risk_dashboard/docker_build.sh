#!/bin/bash
# Build and push Lambda container image to ECR

set -e

# Configuration
AWS_REGION=${AWS_REGION:-us-east-1}
AWS_ACCOUNT_ID=${AWS_ACCOUNT_ID:-YOUR_ACCOUNT_ID}
ECR_REPOSITORY=${ECR_REPOSITORY:-pgx-risk-dashboard}
IMAGE_TAG=${IMAGE_TAG:-latest}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Building Lambda container image...${NC}"

# Step 1: Prepare models (if not already done)
if [ ! -d "models" ]; then
    echo -e "${YELLOW}Models directory not found. Preparing models...${NC}"
    python prepare_models.py --all
fi

# Step 2: Build Docker image
echo -e "${GREEN}Building Docker image...${NC}"
docker build -t ${ECR_REPOSITORY}:${IMAGE_TAG} .

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
echo "     - MODEL_CACHE_TTL=3600"
echo "  3. Configure API Gateway to use this Lambda"

