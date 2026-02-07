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

# Copy prepared models, metadata, and data (outputs/ from dashboard root)
COPY outputs/models/ ${LAMBDA_TASK_ROOT}/models/
COPY outputs/metadata/ ${LAMBDA_TASK_ROOT}/metadata/
COPY outputs/cpic/ ${LAMBDA_TASK_ROOT}/data/

# Set working directory
WORKDIR ${LAMBDA_TASK_ROOT}

# Set Lambda handler
CMD [ "lambda_function.lambda_handler" ]

