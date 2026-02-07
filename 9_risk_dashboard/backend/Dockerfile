# Dockerfile for Lambda Container Image (Python 3.12)
# Supports up to 10GB container size for model storage

FROM public.ecr.aws/lambda/python:3.12

# Install system dependencies
RUN yum install -y gcc g++ && \
    yum clean all && \
    rm -rf /var/cache/yum

# Copy requirements and install Python dependencies (context is 9_risk_dashboard)
COPY backend/requirements.txt ${LAMBDA_TASK_ROOT}/
RUN pip install --no-cache-dir -r requirements.txt -t ${LAMBDA_TASK_ROOT}

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

