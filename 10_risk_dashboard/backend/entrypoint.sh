#!/bin/bash
# Lambda container entrypoint.
# If CODE_S3_KEY is set, downloads the latest lambda_function.py from S3 before
# starting the runtime — allowing code-only updates without a container rebuild.
#
# Usage: set env var CODE_S3_KEY=gold/dashboard/code/lambda_function.py on the Lambda function.
# Deploy code: aws s3 cp lambda_function.py s3://${PGX_RESULTS_BUCKET}/${CODE_S3_KEY}
#              then trigger a cold start (redeploy / update env var touch).

set -e

if [ -n "${CODE_S3_KEY}" ] && [ -n "${PGX_RESULTS_BUCKET}" ]; then
    echo "[entrypoint] Downloading code override from s3://${PGX_RESULTS_BUCKET}/${CODE_S3_KEY}"
    python3 -c "
import boto3, os, sys
try:
    boto3.client('s3').download_file(
        os.environ['PGX_RESULTS_BUCKET'],
        os.environ['CODE_S3_KEY'],
        os.path.join(os.environ.get('LAMBDA_TASK_ROOT', '/var/task'), 'lambda_function.py')
    )
    print('[entrypoint] Code override loaded successfully.')
except Exception as e:
    print(f'[entrypoint] WARNING: S3 code download failed ({e}) — using baked-in lambda_function.py.')
" 2>&1
fi

exec /lambda-entrypoint.sh "$@"
