"""
Template Lambda/API Gateway handler for the PGx FFA & Final Model dashboard.

This function is designed for **proxy integration** with API Gateway and handles
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

    except Exception as exc:  # pragma: no cover - defensive catch-all
        return _response(
            500,
            {
                "error": "Internal server error",
                "details": str(exc),
            },
        )


# ---------------------------------------------------------------------------
# /metadata – Age bands + valid code lists (Drug/ICD/CPT)
# ---------------------------------------------------------------------------

def handle_metadata(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    GET /metadata?cohort=...

    Returns:
    {
      "age_bands": ["0-12","13-24",...],
      "codes": {
        "0-12": { "drugs": [...], "icds": [...], "cpts": [...] },
        "13-24": { ... },
        ...
      }
    }

    This template assumes you precompute a metadata JSON per cohort, e.g.:
      s3://<bucket>/<METADATA_PREFIX>/metadata_{cohort}.json
    """
    params = event.get("queryStringParameters") or {}
    cohort = (params.get("cohort") or "").strip() or "opioid_ed"

    key = f"{METADATA_PREFIX}/metadata_{cohort}.json"

    try:
        obj = s3_client.get_object(Bucket=S3_BUCKET, Key=key)
        data = json.loads(obj["Body"].read().decode("utf-8"))
        return _response(200, data)
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code")
        if code in ("NoSuchKey", "404", "NotFound"):
            return _response(
                404,
                {"error": "metadata_not_found", "bucket": S3_BUCKET, "key": key},
            )
        raise


# ---------------------------------------------------------------------------
# /risk – risk estimate from final ensemble
# ---------------------------------------------------------------------------

def handle_risk(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    POST /risk

    Expected request body:
      {
        "cohort": "opioid_ed",
        "age_band": "25-44",
        "drugs": ["DRUG:XYZ", ...],
        "icds": ["ICD:F1120", ...],
        "cpts": ["CPT:12345", ...]
      }

    Response (example):
      {
        "p_outcome": 0.42,
        "risk_band": "high",
        "p_models": {"catboost":0.43,"xgboost":0.41,"xgboost_rf":0.42},
        "dist": { "bins": [...], "counts": [...] }
      }

    Template implementation:
      - Parses request body.
      - TODOs where you should:
        * Load final model artifacts from S3.
        * Build feature vector for the (cohort, age_band, selected codes).
        * Run model(s) to obtain risk estimate.
    """
    body = json.loads(event.get("body") or "{}")
    cohort = body.get("cohort", "opioid_ed")
    age_band = body.get("age_band", "25-44")
    drugs: List[str] = body.get("drugs", []) or []
    icds: List[str] = body.get("icds", []) or []
    cpts: List[str] = body.get("cpts", []) or []

    # TODO: Implement real model loading + inference logic.
    # Example placeholder: constant risk based on simple heuristic.
    # Replace this with:
    #  - Load final ensemble models from S3 (CatBoost/XGBoost/XGBoost RF).
    #  - Build feature vector matching final_feature_schema.json.
    #  - Run inference and aggregate probabilities.
    base_risk = 0.05
    risk = base_risk + 0.01 * len(drugs) + 0.02 * len(icds) + 0.015 * len(cpts)
    risk = max(0.0, min(risk, 0.99))

    if risk < 0.2:
        band = "low"
    elif risk < 0.5:
        band = "medium"
    else:
        band = "high"

    response_body = {
        "cohort": cohort,
        "age_band": age_band,
        "codes": {"drugs": drugs, "icds": icds, "cpts": cpts},
        "p_outcome": risk,
        "risk_band": band,
        "p_models": {
            # Placeholder per‑model probabilities; update with real model outputs.
            "catboost": risk,
            "xgboost": risk,
            "xgboost_rf": risk,
        },
        # Optional distribution for Plotly histogram; you can fetch this from S3 instead.
        "dist": {
            "bins": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            "counts": [100, 200, 300, 250, 200, 150, 100, 50, 25, 10],
        },
    }

    return _response(200, response_body)


# ---------------------------------------------------------------------------
# /causal – causal “what-if” from FFA
# ---------------------------------------------------------------------------

def handle_causal(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    POST /causal

    Expected request body:
      {
        "cohort": "opioid_ed",
        "age_band": "25-44",
        "drugs": [...],
        "icds": [...],
        "cpts": [...]
      }

    Response (example):
      {
        "effects": [
          {"code": "DRUG:XYZ", "delta_risk_remove": -0.12},
          {"code": "ICD:F1120", "delta_risk_remove": -0.25}
        ]
      }

    Template implementation:
      - In production, this should read precomputed FFA causal summaries from S3
        (e.g., s3://<bucket>/<FFA_PREFIX>/causal_summary_{cohort}_{age_band}.json)
        OR perform fast on‑the‑fly counterfactual scoring using the final model.
    """
    body = json.loads(event.get("body") or "{}")
    cohort = body.get("cohort", "opioid_ed")
    age_band = body.get("age_band", "25-44")
    drugs: List[str] = body.get("drugs", []) or []
    icds: List[str] = body.get("icds", []) or []
    cpts: List[str] = body.get("cpts", []) or []

    # TODO: Replace this with real FFA causal effects loaded from S3.
    # For now, create simple placeholders that show negative deltas (risk reduction)
    # when removing codes that look like “high‑risk” markers.
    effects: List[Dict[str, Any]] = []

    for code in drugs + icds + cpts:
        if "F1120" in code:
            delta = -0.25
        elif code.startswith("DRUG:"):
            delta = -0.10
        elif code.startswith("ICD:"):
            delta = -0.08
        elif code.startswith("CPT:"):
            delta = -0.05
        else:
            delta = -0.02
        effects.append({"code": code, "delta_risk_remove": float(delta)})

    response_body = {
        "cohort": cohort,
        "age_band": age_band,
        "effects": effects,
    }

    return _response(200, response_body)


if __name__ == "__main__":
    # Simple local test harness (for manual debugging).
    test_event = {
      "httpMethod": "GET",
      "path": "/metadata",
      "queryStringParameters": {"cohort": "opioid_ed"},
    }
    print(lambda_handler(test_event, None))


