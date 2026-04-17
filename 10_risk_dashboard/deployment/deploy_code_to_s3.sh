#!/bin/bash
# Deploy lambda_function.py to S3 so the Lambda container picks it up on next cold start.
# No Docker build or EC2 required — runs from any machine with AWS credentials.
#
# Prerequisites:
#   - AWS_SHARED_CREDENTIALS_FILE and AWS_PROFILE set (or default profile configured)
#   - CODE_S3_KEY env var set on the Lambda function:
#       gold/dashboard/code/lambda_function.py
#
# Usage:
#   cd <repo_root>
#   AWS_SHARED_CREDENTIALS_FILE=/path/to/credentials AWS_PROFILE=pgx \
#     bash 10_risk_dashboard/deployment/deploy_code_to_s3.sh
#
# To force a cold start after upload, touch an env var on the Lambda:
#   aws lambda update-function-configuration \
#     --function-name pgx-risk-calculator \
#     --environment "Variables={DEPLOY_TS=$(date +%s),...}"

set -e

BUCKET="${PGX_RESULTS_BUCKET:-pgxdatalake}"
S3_KEY="${CODE_S3_KEY:-gold/dashboard/code/lambda_function.py}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="${SCRIPT_DIR}/../backend/lambda_function.py"

echo "Uploading lambda_function.py -> s3://${BUCKET}/${S3_KEY}"
aws s3 cp "${SRC}" "s3://${BUCKET}/${S3_KEY}"
echo "Done. Lambda will use the new code on next cold start."
echo ""
echo "To force a cold start now, run:"
echo "  aws lambda update-function-configuration \\"
echo "    --function-name pgx-risk-calculator \\"
echo "    --environment 'Variables={DEPLOY_TS=$(date +%s)}'"
