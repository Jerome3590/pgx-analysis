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

# Copy models directory (prepared by prepare_models.py)
# Models will be copied into /var/task/models/ during build
COPY models/ ${LAMBDA_TASK_ROOT}/models/

# Copy data directory (CPIC master Excel file)
# CPIC data will be copied into /var/task/data/ during build
COPY data/ ${LAMBDA_TASK_ROOT}/data/

# Set working directory
WORKDIR ${LAMBDA_TASK_ROOT}

# Set Lambda handler
CMD [ "lambda_function.lambda_handler" ]

