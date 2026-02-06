#!/bin/bash
#
# Create ECR repository and API Gateway for PGx Risk Calculator.
# Run where AWS CLI is configured (profile or env).
#
# Usage: ./create_ecr_and_apigateway.sh [--profile PROFILE]
#   On EC2: no profile or credentials (uses instance role).
#   Off EC2: default profile mushin, credentials at C:\Projects\credentials if present.
# Region: us-east-1 (or AWS_REGION)
#

set -e
REGION="${AWS_REGION:-us-east-1}"

# Shared credentials at C:\Projects\credentials (parent of pgx-analysis); not used on EC2 (instance role)
PROFILE=""
if [[ -z "${AWS_SHARED_CREDENTIALS_FILE:-}" ]]; then
  SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
  PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
  CREDENTIALS_FILE="$PROJECT_ROOT/credentials"
  if [[ -f "$CREDENTIALS_FILE" ]]; then
    export AWS_SHARED_CREDENTIALS_FILE="$CREDENTIALS_FILE"
    PROFILE="${AWS_PROFILE:-mushin}"
    echo "Using credentials: $CREDENTIALS_FILE"
  fi
  # On EC2 no credentials file → leave PROFILE empty so AWS CLI uses instance role
fi
[[ -z "$PROFILE" ]] && PROFILE="${AWS_PROFILE:-}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --profile)
      PROFILE="$2"
      shift 2
      ;;
    *)
      echo "Usage: $0 [--profile PROFILE]"
      exit 1
      ;;
  esac
done

run_aws() {
  if [[ -n "$PROFILE" ]]; then
    aws --profile "$PROFILE" "$@"
  else
    aws "$@"
  fi
}

if [[ -n "$PROFILE" ]]; then
  echo "Using profile: $PROFILE"
else
  echo "Using default credentials (e.g. instance role on EC2)"
fi

echo "=== ECR repository ==="
ECR_REPO="pgx-risk-dashboard"
if run_aws ecr describe-repositories --repository-names "$ECR_REPO" --region "$REGION" 2>/dev/null; then
  echo "Already exists: $ECR_REPO"
else
  run_aws ecr create-repository --repository-name "$ECR_REPO" --region "$REGION"
  echo "Created: $ECR_REPO"
fi
echo ""

echo "=== API Gateway (template) ==="
API_NAME="pgx-calculator-api"
EXISTING_ID=$(run_aws apigateway get-rest-apis --region "$REGION" --query "items[?name=='$API_NAME'].id" --output text 2>/dev/null | tr -d '\r')
if [[ -n "$EXISTING_ID" ]]; then
  echo "Already exists: $API_NAME (id: $EXISTING_ID)"
else
  run_aws apigateway create-rest-api --name "$API_NAME" \
    --description "PGx Risk Calculator API" \
    --endpoint-configuration types=EDGE \
    --region "$REGION"
  echo "Created: $API_NAME"
fi
echo ""
echo "Next (API Gateway console): add resource (e.g. /predict), add POST method, integrate with Lambda (container image from ECR)."
