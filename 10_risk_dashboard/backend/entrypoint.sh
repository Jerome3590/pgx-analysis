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
bucket = os.environ['PGX_RESULTS_BUCKET']
task_root = os.path.join(os.environ.get('LAMBDA_TASK_ROOT', '/var/task'))
client = boto3.client('s3')
code_key = os.environ['CODE_S3_KEY']
code_dir = os.path.dirname(code_key)
try:
    client.download_file(bucket, code_key, os.path.join(task_root, 'lambda_function.py'))
    print('[entrypoint] lambda_function.py override loaded.')
    scenario_key = os.environ.get('CODE_SCENARIO_PATHS_KEY') or f\"{code_dir}/scenario_paths.py\"
    try:
        client.download_file(bucket, scenario_key, os.path.join(task_root, 'scenario_paths.py'))
        print('[entrypoint] scenario_paths.py override loaded.')
    except Exception as e2:
        print(f'[entrypoint] scenario_paths.py not loaded ({e2}) — using baked-in module if present.')
except Exception as e:
    print(f'[entrypoint] WARNING: S3 code download failed ({e}) — using baked-in lambda_function.py.')
" 2>&1
fi

exec /lambda-entrypoint.sh "$@"
