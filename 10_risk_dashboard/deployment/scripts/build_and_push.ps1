$ACCOUNT = "535362115856"
$REPO    = "pgx-risk-calculator"
$REGION  = "us-east-1"
$ECR     = "$ACCOUNT.dkr.ecr.$REGION.amazonaws.com"

Write-Host "Logging in to ECR..."
aws ecr get-login-password --region $REGION | docker login --username AWS --password-stdin $ECR

$DashboardRoot = (Resolve-Path "$PSScriptRoot\..\..")   # deployment/scripts/ -> deployment/ -> 10_risk_dashboard/
Write-Host "Building Docker image (context: $DashboardRoot)..."
docker build -t "${REPO}:latest" -f "$DashboardRoot\backend\Dockerfile" "$DashboardRoot"

Write-Host "Tagging..."
docker tag "${REPO}:latest" "${ECR}/${REPO}:latest"

Write-Host "Pushing..."
docker push "${ECR}/${REPO}:latest"

Write-Host "Updating Lambda function..."
aws lambda update-function-code --function-name pgx-risk-calculator --image-uri "${ECR}/${REPO}:latest" --region $REGION

Write-Host "Done."
