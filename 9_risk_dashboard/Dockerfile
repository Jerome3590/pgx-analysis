# Dockerfile for Lambda Container Image
# Supports up to 10GB container size for model storage

FROM public.ecr.aws/lambda/python:3.10

# Install system dependencies
RUN yum install -y gcc g++ && \
    yum clean all && \
    rm -rf /var/cache/yum

# Copy requirements and install Python dependencies
COPY requirements.txt ${LAMBDA_TASK_ROOT}
RUN pip install --no-cache-dir -r requirements.txt -t ${LAMBDA_TASK_ROOT}

# Copy Lambda function code
COPY lambda_function.py ${LAMBDA_TASK_ROOT}

# Copy lambda_dir contents (prepared by prepare_lambda_dir.py)
# This includes models, metadata, and data directories
COPY lambda_dir/models/ ${LAMBDA_TASK_ROOT}/models/
COPY lambda_dir/metadata/ ${LAMBDA_TASK_ROOT}/metadata/
COPY lambda_dir/data/ ${LAMBDA_TASK_ROOT}/data/

# Set working directory
WORKDIR ${LAMBDA_TASK_ROOT}

# Set Lambda handler
CMD [ "lambda_function.lambda_handler" ]

