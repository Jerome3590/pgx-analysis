"""
Template Lambda/API Gateway handler for the PGx FFA & Final Model dashboard.

This function is designed for proxy integration with API Gateway and handles
three logical endpoints:

- GET  /metadata  -> handle_metadata
- POST /risk      -> handle_risk
- POST /causal    -> handle_causal

The dashboard HTML/JS (served from S3) calls these endpoints to:
- Discover valid age bands and value sets for Drug/ICD/CPT controls.
- Compute risk of F1120 / ADE from the final model ensemble.
- Provide causal “what-if” results based on FFA outputs or model counterfactuals.
"""

import json
import os
from typing import Any, Dict, List

import boto3
from botocore.exceptions import ClientError

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

S3_BUCKET = os.environ.get("PGX_RESULTS_BUCKET", "pgxdatalake")

# Prefixes should be aligned with where your models and FFA outputs live.
# These are placeholders; update to match your project layout.
FINAL_MODEL_PREFIX = os.environ.get("PGX_FINAL_MODEL_PREFIX", "final_models")
FFA_PREFIX = os.environ.get("PGX_FFA_PREFIX", "ffa_analysis")
METADATA_PREFIX = os.environ.get("PGX_METADATA_PREFIX", "ffa_dashboard/metadata")

s3_client = boto3.client("s3")


def _response(status_code: int, body: Dict[str, Any]) -> Dict[str, Any]:
    """Standard API Gateway proxy response."""
    return {
        "statusCode": status_code,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET,POST,OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type",
        },
        "body": json.dumps(body),
    }


def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    Entry point for API Gateway proxy events.

    Routes based on HTTP method + path:
    - GET /metadata
    - POST /risk
    - POST /causal
    """
    try:
        method = event.get("httpMethod", "GET")
        raw_path = event.get("path", "/")

        # Normalize path (strip stage prefix if any)
        path = raw_path.rstrip("/")

        if method == "OPTIONS":
            # CORS preflight
            return _response(200, {"message": "OK"})

        if method == "GET" and path.endswith("/metadata"):
            return handle_metadata(event)
        if method == "POST" and path.endswith("/risk"):
            return handle_risk(event)
        if method == "POST" and path.endswith("/causal"):
            return handle_causal(event)

        return _response(404, {"error": f"Unsupported route: {method} {raw_path}"})
