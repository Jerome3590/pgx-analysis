import sys
import os

# Set root of project (e.g., /home/pgx3874/pgx-analysis)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if project_root not in sys.path:
    sys.path.append(project_root)


# Standard library imports
import argparse
import concurrent.futures
import io
import json
import certifi
import logging
import os
import re
import socket
import sys
import time
import traceback
import warnings
from datetime import datetime
from typing import List, Any, Optional, Dict

# Third-party imports
import boto3
from boto3.s3.transfer import TransferConfig
from botocore.config import Config
from botocore.exceptions import ClientError
import duckdb
import hashlib
from jinja2 import Template
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from mlxtend.frequent_patterns import fpgrowth, association_rules
from mlxtend.preprocessing import TransactionEncoder
import networkx as nx
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests
from tenacity import retry, stop_after_attempt, wait_exponential
import urllib3


# AWS Configuration
S3_BUCKET = "pgxdatalake"
AWS_REGION = "us-east-1"

# Configure boto3 with retries
boto3_config = Config(
    retries = dict(
        max_attempts = 3,
        mode = 'adaptive'
    ),
    connect_timeout = 5,
    read_timeout = 10,
    region_name = AWS_REGION
)

# Initialize boto3 clients with retry configuration
s3_client = boto3.client("s3", config=boto3_config, verify=certifi.where())

# Suppress SSL verification warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning) 

# Note: Logging utilities moved to helpers_1997_13.logging_utils to avoid circular imports


