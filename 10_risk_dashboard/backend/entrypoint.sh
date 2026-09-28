#!/bin/bash
# Lambda container entrypoint.
# The image is the default. Set CODE_S3_OVERRIDE=true to download Python from S3
# before the runtime starts (CODE_S3_KEY, plus the resolver and pipeline keys).
#
# /var/task is read-only on the live image, so overrides go to /tmp/pgx_code
# and PYTHONPATH is prepended so the runtime imports that copy first.

set -e

OVERRIDE_DIR="/tmp/pgx_code"
mkdir -p "${OVERRIDE_DIR}"

OVERRIDE_FLAG="$(printf '%s' "${CODE_S3_OVERRIDE:-}" | tr '[:upper:]' '[:lower:]')"
if [ "${OVERRIDE_FLAG}" = "true" ] || [ "${OVERRIDE_FLAG}" = "1" ] || [ "${OVERRIDE_FLAG}" = "yes" ]; then
if [ -n "${CODE_S3_KEY}" ] && [ -n "${PGX_RESULTS_BUCKET}" ]; then
    echo "[entrypoint] Downloading code override from s3://${PGX_RESULTS_BUCKET}/${CODE_S3_KEY}"
    python3 -c "
import boto3, os
bucket = os.environ['PGX_RESULTS_BUCKET']
override = os.environ.get('PGX_CODE_OVERRIDE_DIR', '/tmp/pgx_code')
os.makedirs(override, exist_ok=True)
client = boto3.client('s3')
code_key = os.environ['CODE_S3_KEY']
code_dir = os.path.dirname(code_key)
try:
    client.download_file(bucket, code_key, os.path.join(override, 'lambda_function.py'))
    print('[entrypoint] lambda_function.py override loaded to', override)
    scenario_key = os.environ.get('CODE_SCENARIO_PATHS_KEY') or f\"{code_dir}/scenario_paths.py\"
    try:
        client.download_file(bucket, scenario_key, os.path.join(override, 'scenario_paths.py'))
        print('[entrypoint] scenario_paths.py override loaded.')
    except Exception as e2:
        print(f'[entrypoint] scenario_paths.py not loaded ({e2}) — using baked-in module if present.')
    resolver_key = os.environ.get('CPIC_RESOLVER_S3_KEY') or f\"{code_dir}/cpic_allele_resolver.py\"
    try:
        client.download_file(bucket, resolver_key, os.path.join(override, 'cpic_allele_resolver.py'))
        print('[entrypoint] cpic_allele_resolver.py override loaded.')
    except Exception as e3:
        print(f'[entrypoint] cpic_allele_resolver.py not loaded ({e3}) — lambda_function will fetch from S3 if needed.')
    pipeline_key = os.environ.get('PGX_PIPELINE_S3_KEY') or f\"{code_dir}/pgx_exploratory_pipeline.py\"
    try:
        client.download_file(bucket, pipeline_key, os.path.join(override, 'pgx_exploratory_pipeline.py'))
        print('[entrypoint] pgx_exploratory_pipeline.py override loaded.')
    except Exception as e4:
        print(f'[entrypoint] pgx_exploratory_pipeline.py not loaded ({e4}) — lambda_function will fetch from S3 if needed.')
except Exception as e:
    print(f'[entrypoint] WARNING: S3 code download failed ({e}) — using baked-in lambda_function.py.')
" 2>&1
    export PYTHONPATH="${OVERRIDE_DIR}${PYTHONPATH:+:$PYTHONPATH}"
fi
else
    echo "[entrypoint] Using baked image code (set CODE_S3_OVERRIDE=true to pull Python from S3)."
fi

exec /lambda-entrypoint.sh "$@"
