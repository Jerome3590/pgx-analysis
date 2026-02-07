#!/bin/bash
#
# Create API Gateway REST API "pgx-risk-calculator" and wire it to Lambda "pgx-risk-calculator"
# (proxy integration: all paths go to Lambda). Run after Lambda exists.
#
# Usage:
#   ./create_api_gateway_pgx_risk_calculator.sh [--profile PROFILE]
#   Optional: AWS_REGION=us-east-1 (default)
#
# Prerequisites: Lambda function "pgx-risk-calculator" must exist in the same account/region.
#
set -e

REGION="${AWS_REGION:-us-east-1}"
API_NAME="pgx-risk-calculator"
LAMBDA_NAME="pgx-risk-calculator"
PROFILE="${AWS_PROFILE:-}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --profile) PROFILE="$2"; shift 2 ;;
    *) echo "Usage: $0 [--profile PROFILE]"; exit 1 ;;
  esac
done

run_aws() {
  if [[ -n "$PROFILE" ]]; then
    aws --profile "$PROFILE" --region "$REGION" "$@"
  else
    aws --region "$REGION" "$@"
  fi
}

echo "Region: $REGION  API: $API_NAME  Lambda: $LAMBDA_NAME"
echo ""

# Account ID (for ARNs)
ACCOUNT_ID=$(run_aws sts get-caller-identity --query Account --output text)
echo "Account: $ACCOUNT_ID"
echo ""

# 1. Create REST API (skip if already exists)
EXISTING_ID=$(run_aws apigateway get-rest-apis --query "items[?name=='$API_NAME'].id" --output text 2>/dev/null | tr -d '\r')
if [[ -n "$EXISTING_ID" ]]; then
  echo "API already exists: $API_NAME (id: $EXISTING_ID)"
  API_ID="$EXISTING_ID"
else
  API_ID=$(run_aws apigateway create-rest-api \
    --name "$API_NAME" \
    --description "PGx Risk Calculator API" \
    --endpoint-configuration types=EDGE \
    --query id --output text)
  echo "Created API: $API_ID"
fi
echo ""

# 2. Root resource ID
ROOT_ID=$(run_aws apigateway get-resources --rest-api-id "$API_ID" --query "items[?path=='/'].id" --output text)
echo "Root resource id: $ROOT_ID"
echo ""

# 3. Create {proxy+} resource (catches all paths under /)
PROXY_ID=$(run_aws apigateway create-resource \
  --rest-api-id "$API_ID" \
  --parent-id "$ROOT_ID" \
  --path-part "{proxy+}" \
  --query id --output text 2>/dev/null || true)
if [[ -z "$PROXY_ID" ]]; then
  PROXY_ID=$(run_aws apigateway get-resources --rest-api-id "$API_ID" --query "items[?pathPart=='{proxy+}'].id" --output text)
fi
echo "Proxy resource id: $PROXY_ID"
echo ""

# 4. ANY method on {proxy+} with Lambda proxy integration
LAMBDA_ARN="arn:aws:lambda:${REGION}:${ACCOUNT_ID}:function:${LAMBDA_NAME}"
INTEGRATION_URI="arn:aws:apigateway:${REGION}:lambda:path/2015-03-31/functions/${LAMBDA_ARN}/invocations"

run_aws apigateway put-method \
  --rest-api-id "$API_ID" \
  --resource-id "$PROXY_ID" \
  --http-method ANY \
  --authorization-type NONE \
  --request-parameters "method.request.path.proxy=true" \
  --output text >/dev/null
echo "Put method ANY on {proxy+}"

run_aws apigateway put-integration \
  --rest-api-id "$API_ID" \
  --resource-id "$PROXY_ID" \
  --http-method ANY \
  --type AWS_PROXY \
  --integration-http-method POST \
  --uri "$INTEGRATION_URI" \
  --output text >/dev/null
echo "Put Lambda proxy integration on {proxy+}"
echo ""

# 5. Method on root (/) so that GET / and OPTIONS / go to Lambda
run_aws apigateway put-method \
  --rest-api-id "$API_ID" \
  --resource-id "$ROOT_ID" \
  --http-method ANY \
  --authorization-type NONE \
  --output text >/dev/null 2>/dev/null || true
run_aws apigateway put-integration \
  --rest-api-id "$API_ID" \
  --resource-id "$ROOT_ID" \
  --http-method ANY \
  --type AWS_PROXY \
  --integration-http-method POST \
  --uri "$INTEGRATION_URI" \
  --output text >/dev/null 2>/dev/null || true
echo "Put ANY on root (/)"
echo ""

# 6. Grant API Gateway permission to invoke Lambda
SOURCE_ARN="arn:aws:execute-api:${REGION}:${ACCOUNT_ID}:${API_ID}/*"
run_aws lambda add-permission \
  --function-name "$LAMBDA_NAME" \
  --statement-id "apigateway-invoke-${API_ID}" \
  --action lambda:InvokeFunction \
  --principal apigateway.amazonaws.com \
  --source-arn "$SOURCE_ARN" \
  --output text >/dev/null 2>/dev/null || echo "(Lambda permission may already exist)"
echo "Lambda invoke permission set"
echo ""

# 7. Deploy to prod stage
run_aws apigateway create-deployment \
  --rest-api-id "$API_ID" \
  --stage-name prod \
  --description "Initial deployment" \
  --output text >/dev/null 2>/dev/null || run_aws apigateway create-deployment --rest-api-id "$API_ID" --stage-name prod --output text >/dev/null
echo "Deployed to stage: prod"
echo ""

# 8. Output invoke URL
INVOKE_URL="https://${API_ID}.execute-api.${REGION}.amazonaws.com/prod"
echo "=============================================="
echo "API Gateway: $API_NAME (id: $API_ID)"
echo "Invoke URL:  $INVOKE_URL"
echo "=============================================="
echo "Update your frontend API_BASE to: $INVOKE_URL"
