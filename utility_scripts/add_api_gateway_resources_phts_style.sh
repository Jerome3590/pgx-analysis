#!/bin/bash
#
# Add PHTS-style explicit resources to the pgx-risk-calculator API Gateway:
#   /metadata (GET, OPTIONS), /risk (POST, OPTIONS), /risk/comparison (POST, OPTIONS),
#   /causal/importance (POST, OPTIONS), /causal/interactions (POST, OPTIONS).
# Same Lambda proxy integration; these paths take precedence over {proxy+}.
#
# Usage:
#   ./add_api_gateway_resources_phts_style.sh [--profile PROFILE]
#
# Prerequisites: API "pgx-risk-calculator" and Lambda "pgx-risk-calculator" must exist.
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

# Get resource ID by path (from get-resources output)
get_resource_id() {
  local api_id="$1"
  local path="$2"
  run_aws apigateway get-resources --rest-api-id "$api_id" \
    --query "items[?path=='$path'].id" --output text 2>/dev/null | tr -d '\r' | head -1
}

# Create resource under parent if not present; output resource id
ensure_resource() {
  local api_id="$1"
  local parent_id="$2"
  local path_part="$3"
  local expected_path="$4"
  local id
  id=$(get_resource_id "$api_id" "$expected_path")
  if [[ -z "$id" || "$id" == "None" ]]; then
    id=$(run_aws apigateway create-resource \
      --rest-api-id "$api_id" \
      --parent-id "$parent_id" \
      --path-part "$path_part" \
      --query id --output text)
    echo "Created resource $expected_path ($id)" >&2
  else
    echo "Resource $expected_path exists ($id)" >&2
  fi
  echo "$id"
}

# Put method and Lambda proxy integration on a resource
put_method_integration() {
  local api_id="$1"
  local resource_id="$2"
  local http_method="$3"
  run_aws apigateway put-method \
    --rest-api-id "$api_id" \
    --resource-id "$resource_id" \
    --http-method "$http_method" \
    --authorization-type NONE \
    --output text >/dev/null
  run_aws apigateway put-integration \
    --rest-api-id "$api_id" \
    --resource-id "$resource_id" \
    --http-method "$http_method" \
    --type AWS_PROXY \
    --integration-http-method POST \
    --uri "$INTEGRATION_URI" \
    --output text >/dev/null
  echo "  $http_method on resource $resource_id" >&2
}

echo "Region: $REGION  API: $API_NAME  Lambda: $LAMBDA_NAME"
echo ""

ACCOUNT_ID=$(run_aws sts get-caller-identity --query Account --output text)
API_ID=$(run_aws apigateway get-rest-apis --query "items[?name=='$API_NAME'].id" --output text 2>/dev/null | tr -d '\r')
if [[ -z "$API_ID" || "$API_ID" == "None" ]]; then
  echo "Error: API $API_NAME not found. Create it first with create_api_gateway_pgx_risk_calculator.sh"
  exit 1
fi
echo "API id: $API_ID"

ROOT_ID=$(run_aws apigateway get-resources --rest-api-id "$API_ID" --query "items[?path=='/'].id" --output text | tr -d '\r')
echo "Root resource id: $ROOT_ID"
echo ""

LAMBDA_ARN="arn:aws:lambda:${REGION}:${ACCOUNT_ID}:function:${LAMBDA_NAME}"
INTEGRATION_URI="arn:aws:apigateway:${REGION}:lambda:path/2015-03-31/functions/${LAMBDA_ARN}/invocations"

# Create resources (order: top-level first, then children)
METADATA_ID=$(ensure_resource "$API_ID" "$ROOT_ID" "metadata" "/metadata")
RISK_ID=$(ensure_resource "$API_ID" "$ROOT_ID" "risk" "/risk")
COMPARISON_ID=$(ensure_resource "$API_ID" "$RISK_ID" "comparison" "/risk/comparison")
CAUSAL_ID=$(ensure_resource "$API_ID" "$ROOT_ID" "causal" "/causal")
IMPORTANCE_ID=$(ensure_resource "$API_ID" "$CAUSAL_ID" "importance" "/causal/importance")
INTERACTIONS_ID=$(ensure_resource "$API_ID" "$CAUSAL_ID" "interactions" "/causal/interactions")
echo ""

# Methods: /metadata GET + OPTIONS; /risk and /risk/comparison POST + OPTIONS; causal/importance and causal/interactions POST + OPTIONS
echo "Putting methods and Lambda proxy integration..."
put_method_integration "$API_ID" "$METADATA_ID" "GET"
put_method_integration "$API_ID" "$METADATA_ID" "OPTIONS"
put_method_integration "$API_ID" "$RISK_ID" "POST"
put_method_integration "$API_ID" "$RISK_ID" "OPTIONS"
put_method_integration "$API_ID" "$COMPARISON_ID" "POST"
put_method_integration "$API_ID" "$COMPARISON_ID" "OPTIONS"
put_method_integration "$API_ID" "$IMPORTANCE_ID" "POST"
put_method_integration "$API_ID" "$IMPORTANCE_ID" "OPTIONS"
put_method_integration "$API_ID" "$INTERACTIONS_ID" "POST"
put_method_integration "$API_ID" "$INTERACTIONS_ID" "OPTIONS"
echo ""

# Deploy
run_aws apigateway create-deployment --rest-api-id "$API_ID" --stage-name prod --description "Add PHTS-style resources" --output text >/dev/null
echo "Deployed to stage: prod"
echo ""
echo "Resources now match PHTS-style: /metadata, /risk, /risk/comparison, /causal/importance, /causal/interactions (plus existing / and {proxy+})."
