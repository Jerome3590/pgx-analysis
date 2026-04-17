# Dockerfile for Lambda Container Image
# Template: 9_risk_dashboard/deployment/Dockerfile.phts (Python 3.11, yum, prefer-binary).
# Supports up to 10GB container size for model storage.

FROM public.ecr.aws/lambda/python:3.11

# Install system dependencies (match Dockerfile.phts; Amazon Linux 2 has yum)
RUN yum install -y gcc gcc-c++ cmake make && \
    yum clean all && \
    rm -rf /var/cache/yum

# Copy requirements and install Python dependencies (context is 9_risk_dashboard)
COPY backend/requirements.txt ${LAMBDA_TASK_ROOT}/
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir --prefer-binary -r ${LAMBDA_TASK_ROOT}/requirements.txt -t ${LAMBDA_TASK_ROOT}

# Copy Lambda function code
COPY backend/lambda_function.py ${LAMBDA_TASK_ROOT}/

# Copy prepared models, metadata, and data (lambda_dir/ populated by prepare_lambda_dir.py)
COPY lambda_dir/models/ ${LAMBDA_TASK_ROOT}/models/
COPY lambda_dir/metadata/ ${LAMBDA_TASK_ROOT}/metadata/
COPY lambda_dir/data/ ${LAMBDA_TASK_ROOT}/data/

# Entrypoint: optionally downloads latest lambda_function.py from S3 before runtime starts.
# Set CODE_S3_KEY env var on the Lambda function to enable code-only updates from S3.
COPY backend/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Set working directory
WORKDIR ${LAMBDA_TASK_ROOT}

ENTRYPOINT ["/bin/bash", "/entrypoint.sh"]
CMD [ "lambda_function.lambda_handler" ]

