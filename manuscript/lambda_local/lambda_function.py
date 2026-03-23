"""
AWS Lambda function for PGx Risk Dashboard API.

Lambda receives user input (cohort, model/feature selections) and filters only—it does not
process or generate visualization data. All visuals are prebuilt on EC2 and saved to S3; Lambda
returns URLs to those prebuilt assets. Risk inference uses the ensemble (CatBoost, XGBoost,
XGBoost RF) with user-provided features; causal/visualization endpoints return prebuilt or
pre-indexed data filtered by cohort/age_band/features.

Handles:
- GET /metadata - Returns valid codes for cohorts/age_bands (filter by cohort)
- POST /risk - Risk score from ensemble, filtered by user-selected cohort and features
- POST /risk/comparison - Compares risk scores for user-provided scenarios
- POST /causal/importance - Returns causal importance filtered by selected drugs/features (prebuilt data)
- POST /causal/interactions - Returns interaction results filtered by selection (prebuilt data)
- GET /visualizations/* - Returns URLs to prebuilt S3 assets only (no processing)
- GET /visualizations/cohort_pgx - Returns network_topology_url for PGx Cohort tab

Environment Variables:
- PGX_RESULTS_BUCKET: S3 bucket name (default: pgxdatalake)
- MODEL_CACHE_TTL: Model cache TTL in seconds (default: 3600)
- MODEL_BASE_PATH: Path to models in container (default: /var/task/models)
"""

import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from io import BytesIO

import boto3
from botocore.exceptions import ClientError

# Try to import model libraries (may not be available in Lambda)
try:
    import joblib
    import numpy as np
    import pandas as pd
    from catboost import CatBoostClassifier
    import xgboost as xgb
    MODEL_LIBS_AVAILABLE = True
except ImportError:
    MODEL_LIBS_AVAILABLE = False
    print("Warning: Model libraries not available. Model inference will fail.")

# Try to import openpyxl for Excel reading (needed for CPIC master file)
try:
    import openpyxl
    EXCEL_AVAILABLE = True
except ImportError:
    EXCEL_AVAILABLE = False
    print("Warning: openpyxl not available. Excel reading may fail.")

# DuckDB for efficient CPIC Parquet reads (preferred over Excel when parquet file is present)
try:
    import duckdb
    DUCKDB_AVAILABLE = True
except ImportError:
    DUCKDB_AVAILABLE = False

# Configuration
S3_BUCKET = os.environ.get("PGX_RESULTS_BUCKET", "pgxdatalake")
# Dashboard frontend bucket/prefix (where FP-Growth assets are uploaded so they render with the app)
S3_DASHBOARD_BUCKET = os.environ.get("S3_DASHBOARD_BUCKET", "jerome-dixon.io")
S3_DASHBOARD_PREFIX = os.environ.get("S3_DASHBOARD_PREFIX", "vcu/pgx-risk-calculator")
S3_DASHBOARD_REGION = os.environ.get("S3_DASHBOARD_REGION", "us-east-1")
MODEL_CACHE_TTL = int(os.environ.get("MODEL_CACHE_TTL", "3600"))

# Optional: fully offline/local testing paths (bypass S3)
OFFLINE_METADATA_PATH = os.environ.get("PGX_OFFLINE_METADATA_PATH")
OFFLINE_DATA_PATH = os.environ.get("PGX_OFFLINE_DATA_PATH")


def _dashboard_s3_url(key: str) -> str:
    """Build path-style S3 URL for dashboard assets: https://s3.{region}.amazonaws.com/{bucket}/{key}"""
    return f"https://s3.{S3_DASHBOARD_REGION}.amazonaws.com/{S3_DASHBOARD_BUCKET}/{key}"
METADATA_PREFIX = "gold/dashboard/metadata"
MODEL_PREFIX = "gold/dashboard/models"
