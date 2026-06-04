"""
AWS Lambda function for PGx Risk Dashboard API.

Lambda receives user input (cohort, model/feature selections) and **filters** only—it does not
process or generate visualization data. All visuals are prebuilt on EC2 and saved to S3; Lambda
returns URLs to those prebuilt assets. Risk inference uses the ensemble (CatBoost, XGBoost,
XGBoost RF) with user-provided features; scenario visualization endpoints return prebuilt or
pre-indexed data filtered by cohort/age_band/features.

Handles:
- GET /metadata - Returns valid codes for cohorts/age_bands (filter by cohort)
- POST /risk - Risk score from ensemble, filtered by user-selected cohort and features
- POST /risk/comparison - Compares risk scores for user-provided scenarios
- POST /risk/drug_contributions - Leave-one-out per-drug Δp̂ for What-If / deprescribing tab
- POST /scenario/importance - Interaction importance filtered by selected drugs/features (prebuilt data)
- POST /scenario/interactions - Multi-feature interaction results filtered by selection (prebuilt data)
- GET /visualizations/scenario - Scenario / FFA+SHAP dashboard JSON (canonical)
- GET /visualizations/* - Returns URLs to other prebuilt S3 assets (no processing)
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

from scenario_paths import (
    API_SCENARIO_IMPORTANCE,
    API_SCENARIO_INTERACTIONS,
    API_VISUALIZATIONS_SCENARIO,
    s3_scenario_data_key,
)

# Try to import model libraries (may not be available in Lambda)
try:
    import joblib
    import numpy as np
    import pandas as pd
    from catboost import CatBoostClassifier, Pool
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

# Model storage paths (ECR container has models in /var/task/models/)
MODEL_BASE_PATH = os.environ.get("MODEL_BASE_PATH", "/var/task/models")
USE_CONTAINER_MODELS = os.path.exists(MODEL_BASE_PATH)
# When PREFER_S3=true (default): S3 is checked first; container is fallback.
# This lets model and code updates on S3 take effect on next cold start without
# rebuilding the container.  Set PREFER_S3=false to restore container-first order.
PREFER_S3 = os.environ.get("PREFER_S3", "true").lower() not in ("0", "false", "no")

# In-memory model cache
_model_cache: Dict[str, Dict[str, Any]] = {}
_cache_timestamps: Dict[str, float] = {}

# Dashboard manifest cache (single source of truth for viz paths; same JSON the frontend loads)
_dashboard_manifest: Optional[Dict[str, Any]] = None

s3_client = boto3.client("s3")


def _s3_object_exists(bucket: str, key: str) -> bool:
    """Return True if the S3 object exists (HEAD). Treat 403/AccessDenied as not found so
    we return empty payloads instead of 500 when the dashboard bucket is not accessible (e.g. from EC2)."""
    try:
        s3_client.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code")
        if code in ("404", "NoSuchKey", "403", "AccessDenied"):
            return False
        raise


def _get_dashboard_manifest() -> Optional[Dict[str, Any]]:
    """Load dashboard_visual_objects.json from S3 (same manifest the frontend uses). Cached in memory."""
    global _dashboard_manifest
    if _dashboard_manifest is not None:
        return _dashboard_manifest
    key = f"{S3_DASHBOARD_PREFIX.strip('/')}/visualizations/dashboard_visual_objects.json"
    try:
        obj = s3_client.get_object(Bucket=S3_DASHBOARD_BUCKET, Key=key)
        _dashboard_manifest = json.loads(obj["Body"].read().decode("utf-8"))
        return _dashboard_manifest
    except ClientError as e:
        if e.response.get("Error", {}).get("Code") in ("404", "NoSuchKey", "403", "AccessDenied"):
            return None
        raise


def _paths_checked_from_error(e: FileNotFoundError) -> List[str]:
    """Extract paths from 'Checked: a; b; c' in FileNotFoundError message for frontend display."""
    msg = str(e)
    if "Checked:" in msg:
        rest = msg.split("Checked:", 1)[1].strip()
        return [p.strip() for p in rest.split(";") if p.strip()]
    return [msg] if msg else []


def _cors_headers() -> Dict[str, str]:
    """CORS headers so browser allows fetch from S3/dashboard origin."""
    return {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "GET,POST,OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type,Accept",
        "Access-Control-Max-Age": "86400",
    }


def _response(status_code: int, body: Dict[str, Any], headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """Standard API Gateway proxy response with CORS."""
    default_headers = {
        "Content-Type": "application/json",
        **_cors_headers(),
    }
    if headers:
        default_headers.update(headers)
    
    return {
        "statusCode": status_code,
        "headers": default_headers,
        "body": json.dumps(body),
    }


def _response_html(status_code: int, html_body: str) -> Dict[str, Any]:
    """Return HTML for iframe embedding (Content-Type: text/html, no Content-Disposition)."""
    return {
        "statusCode": status_code,
        "headers": {
            "Content-Type": "text/html; charset=utf-8",
            **_cors_headers(),
        },
        "body": html_body,
    }


def determine_cohort_and_age_band(age: int) -> Tuple[str, str]:
    """
    Determine cohort and age band from age (when cohort/age_band not provided).
    
    Both cohorts support all 8 age bands (0-12 through 85-114). This helper maps
    a single age value to a default (cohort, age_band) for backward compatibility:
    - Ages 0-64: opioid_ed cohort
    - Ages 65-114: non_opioid_ed (polypharmacy) cohort
    """
    if age < 0:
        raise ValueError("Age must be 0 or older.")
    elif age <= 64:
        cohort = "opioid_ed"
        if age <= 12:
            age_band = "0-12"
        elif age <= 24:
            age_band = "13-24"
        elif age <= 44:
            age_band = "25-44"
        elif age <= 54:
            age_band = "45-54"
        else:  # 55 <= age <= 64
            age_band = "55-64"
    elif 65 <= age <= 114:
        cohort = "non_opioid_ed"
        if age <= 74:
            age_band = "65-74"
        elif age <= 84:
            age_band = "75-84"
        else:  # 85 <= age <= 114
            age_band = "85-114"
    else:  # age > 114
        raise ValueError("Age must be 114 or younger.")
    
    return cohort, age_band


def load_metadata(cohort: str) -> Dict[str, Any]:
    """
    Load metadata JSON from container filesystem or S3.
    
    Priority:
    1. Container filesystem (/var/task/metadata/) - fastest, bundled in image
    2. S3 fallback - for development or if container metadata not available
    """
    metadata_file = f"metadata_{cohort}.json"
    container_path = Path(f"/var/task/metadata/{metadata_file}")

    # Offline/local override (prefer explicit local path)
    if OFFLINE_METADATA_PATH:
        try:
            p = Path(OFFLINE_METADATA_PATH) / metadata_file
            if p.exists():
                with open(p, "r", encoding="utf-8") as f:
                    return json.load(f)
        except Exception as e:
            print(f"Warning: Failed to load metadata from OFFLINE_METADATA_PATH: {e}")
    
    key = f"{METADATA_PREFIX}/{metadata_file}"

    # S3 first (PREFER_S3 default) so updated metadata is picked up on cold start
    # without a container rebuild.  Fall through to container on any S3 error.
    if PREFER_S3:
        try:
            obj = s3_client.get_object(Bucket=S3_BUCKET, Key=key)
            return json.loads(obj["Body"].read().decode("utf-8"))
        except ClientError:
            pass  # fall through to container

    # Container filesystem
    if container_path.exists():
        try:
            with open(container_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Failed to load metadata from container: {e}")

    # S3 fallback (when PREFER_S3=false and container missed)
    if not PREFER_S3:
        try:
            obj = s3_client.get_object(Bucket=S3_BUCKET, Key=key)
            return json.loads(obj["Body"].read().decode("utf-8"))
        except ClientError as e:
            code = e.response.get("Error", {}).get("Code")
            if code in ("NoSuchKey", "404", "NotFound"):
                raise FileNotFoundError(f"Metadata not found. Checked: {container_path}; s3://{S3_BUCKET}/{key}")
            raise

    raise FileNotFoundError(f"Metadata not found. Checked: s3://{S3_BUCKET}/{key}; {container_path}")


def handle_performance(_event: Dict[str, Any]) -> Dict[str, Any]:
    """GET /performance — return MC-CV model scores per cohort/age_band from feature schemas."""
    COHORTS = ["opioid_ed", "non_opioid_ed"]
    AGE_BANDS = ["0-12", "13-24", "25-44", "45-54", "55-64", "65-74", "75-84", "85-114"]
    result: Dict[str, Any] = {}
    for cohort in COHORTS:
        result[cohort] = {}
        for age_band in AGE_BANDS:
            schema = load_feature_schema(cohort, age_band)
            scores = schema.get("model_scores")
            weights = schema.get("model_weights", {})
            if scores:
                result[cohort][age_band] = {
                    "model_scores": {
                        m: {
                            "mean_pr_auc": round(s["mean_pr_auc"], 4),
                            "mean_logloss": round(s["mean_logloss"], 4),
                            "composite_score": round(s["composite_score"], 4),
                            "selected": s.get("selected", weights.get(m, 0.0) == 1.0),
                        }
                        for m, s in scores.items()
                    }
                }
    return _response(200, {"by_cohort": result, "source": "feature_schema"})


def handle_available(_event: Dict[str, Any]) -> Dict[str, Any]:
    """GET /available — return structured artifact availability map (available.json) from S3.
    Generated by check_dashboard_artifact_paths.py --json-out during notebook 5 deployment."""
    key = f"{METADATA_PREFIX}/available.json"
    try:
        obj = s3_client.get_object(Bucket=S3_BUCKET, Key=key)
        data = json.loads(obj["Body"].read().decode("utf-8"))
        return _response(200, data)
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code")
        if code in ("NoSuchKey", "404", "NotFound", "AccessDenied", "403"):
            return _response(404, {"error": "available.json not found — run notebook 5 to generate it"})
        raise


def handle_metrics(_event: Dict[str, Any]) -> Dict[str, Any]:
    """GET /metrics — return prebuilt model performance metrics from S3 (no recomputation). Fallback to container bundle."""
    key = f"{METADATA_PREFIX}/model_performance_metrics.json"

    # Offline/local override
    if OFFLINE_METADATA_PATH:
        try:
            p = Path(OFFLINE_METADATA_PATH) / "model_performance_metrics.json"
            if p.exists():
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f)
                out = data if isinstance(data, dict) else {"by_cohort": {}, "payload": data}
                if "source" not in out:
                    out["source"] = "offline"
                return _response(200, out)
        except Exception as e:
            print(f"Metrics offline load failed: {e}")
    try:
        obj = s3_client.get_object(Bucket=S3_BUCKET, Key=key)
        data = json.loads(obj["Body"].read().decode("utf-8"))
        if isinstance(data, dict):
            if "source" not in data:
                data["source"] = "s3"
            return _response(200, data)
        return _response(200, {"by_cohort": {}, "payload": data, "source": "s3"})
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code")
        if code in ("NoSuchKey", "404", "NotFound"):
            pass
        else:
            print(f"Metrics S3 load failed: {e}")
    except Exception as e:
        print(f"Metrics S3 load failed: {e}")
    # Fallback: container bundle (optional at build time)
    container_path = Path("/var/task/metadata/model_performance_metrics.json")
    if container_path.exists():
        try:
            with open(container_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            out = data if isinstance(data, dict) else {"by_cohort": {}, "payload": data}
            if "source" not in out:
                out["source"] = "container"
            return _response(200, out)
        except Exception as e:
            print(f"Metrics container load failed: {e}")
    return _response(200, {"by_cohort": {}, "source": "none"})


def load_model(cohort: str, age_band: str, model_type: str, bin_name: Optional[str] = None) -> Any:
    """
    Load model from container filesystem (ECR) or S3 with caching.

    When bin_name is supplied the per-bin model is loaded exclusively — no
    full-cohort fallback.  Raises FileNotFoundError if the bin model is absent.
    When bin_name is None the full-cohort model is loaded (legacy/baseline path).

    model_type: 'catboost', 'xgboost', or 'xgboost_rf'
    bin_name:   one of 'low', 'medium', 'high', 'extreme'
    """
    cache_key = f"{cohort}/{age_band}/{model_type}" if not bin_name else f"{cohort}/{age_band}/bin/{bin_name}/{model_type}"
    
    # Check cache
    if cache_key in _model_cache:
        timestamp = _cache_timestamps.get(cache_key, 0)
        if time.time() - timestamp < MODEL_CACHE_TTL:
            return _model_cache[cache_key]['model']
    
    age_band_fname = age_band.replace("-", "_")

    def _try_load_from_path(path: str) -> Optional[Any]:
        """Load a single model file (joblib or CatBoost json/cbm). Returns None on any failure."""
        if not os.path.exists(path):
            return None
        try:
            if model_type == 'catboost':
                m = CatBoostClassifier()
                m.load_model(path)
            else:
                m = joblib.load(path)
            print(f"Loaded {model_type} from {path}")
            return m
        except Exception as e:
            print(f"Warning: Failed to load {path}: {e}")
            return None

    if model_type not in ('catboost', 'xgboost', 'xgboost_rf'):
        raise ValueError(f"Unknown model type: {model_type}")

    def _load_from_s3_bytes(s3_key: str) -> Optional[Any]:
        try:
            obj = s3_client.get_object(Bucket=S3_BUCKET, Key=s3_key)
            model_bytes = obj["Body"].read()
            if model_type == 'catboost':
                m = CatBoostClassifier()
                m.load_model(BytesIO(model_bytes))
            else:
                m = joblib.load(BytesIO(model_bytes))
            print(f"Loaded {model_type} from S3: s3://{S3_BUCKET}/{s3_key}")
            return m
        except ClientError as e:
            code = e.response.get("Error", {}).get("Code", "")
            if code in ("NoSuchKey", "404", "NotFound"):
                return None
            if code in ("AccessDenied", "403"):
                # AWS returns 403 instead of 404 when s3:ListBucket is not granted
                # and the requested key does not exist — treat as key-not-found.
                print(f"[load_model] S3 AccessDenied on {s3_key!r} — treating as key not found (no ListBucket permission)")
                return None
            raise

    if bin_name:
        # ── Per-bin path ONLY — no full-cohort fallback ──────────────────────
        s3_key = f"{MODEL_PREFIX}/{cohort}/{age_band_fname}/bin_models/{bin_name}/{model_type}.joblib"
        bin_dir = os.path.join(MODEL_BASE_PATH, cohort, age_band_fname, "bin_models", bin_name) if USE_CONTAINER_MODELS else None

        if PREFER_S3:
            model = _load_from_s3_bytes(s3_key)
            if model is None and bin_dir:
                for fname in (f"{model_type}.joblib", f"{model_type}.json"):
                    model = _try_load_from_path(os.path.join(bin_dir, fname))
                    if model is not None:
                        break
        else:
            model = None
            if bin_dir:
                for fname in (f"{model_type}.joblib", f"{model_type}.json"):
                    model = _try_load_from_path(os.path.join(bin_dir, fname))
                    if model is not None:
                        break
            if model is None:
                model = _load_from_s3_bytes(s3_key)

        if model is None:
            raise FileNotFoundError(
                f"Per-bin model not found for bin='{bin_name}', model='{model_type}'. "
                f"Run train_per_bin() (notebook 3) then prepare_models.py to generate it."
            )
        _model_cache[cache_key] = {'model': model}
        _cache_timestamps[cache_key] = time.time()
        return model
    else:
        # ── Full-cohort path (no bin) ─────────────────────────────────────────
        s3_key = f"{MODEL_PREFIX}/{cohort}/{age_band_fname}/{model_type}.joblib"
        full_dir = os.path.join(MODEL_BASE_PATH, cohort, age_band_fname) if USE_CONTAINER_MODELS else None

        if PREFER_S3:
            model = _load_from_s3_bytes(s3_key)
            if model is None and full_dir:
                for fname in (f"{model_type}.joblib", f"{model_type}.json"):
                    model = _try_load_from_path(os.path.join(full_dir, fname))
                    if model is not None:
                        break
        else:
            model = None
            if full_dir:
                for fname in (f"{model_type}.joblib", f"{model_type}.json"):
                    model = _try_load_from_path(os.path.join(full_dir, fname))
                    if model is not None:
                        break
            if model is None:
                model = _load_from_s3_bytes(s3_key)

        if model is None:
            raise FileNotFoundError(f"Full-cohort model not found: s3://{S3_BUCKET}/{s3_key}")
        _model_cache[cache_key] = {'model': model}
        _cache_timestamps[cache_key] = time.time()
        return model
    raise FileNotFoundError(f"Model not found for {cohort}/{age_band}/{model_type}" + (f" (bin={bin_name})" if bin_name else ""))


def load_risk_distribution_2019(cohort: str, age_band: str) -> Optional[Dict[str, Any]]:
    """Load 2019 holdout risk distribution (bins/counts, optional baseline_risk) from container or S3. Returns None if not found."""
    age_band_fname = age_band.replace("-", "_")

    def _from_data(data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        bins = data.get("bins")
        counts = data.get("counts")
        if bins is None or counts is None or len(bins) != len(counts):
            return None
        out: Dict[str, Any] = {"bins": bins, "counts": counts}
        if "baseline_risk" in data and data["baseline_risk"] is not None:
            out["baseline_risk"] = float(data["baseline_risk"])
        if "risk_band_thresholds" in data and isinstance(data["risk_band_thresholds"], dict):
            out["risk_band_thresholds"] = {k: float(v) for k, v in data["risk_band_thresholds"].items()}
        return out

    key = f"{MODEL_PREFIX}/{cohort}/{age_band_fname}/risk_distribution_2019.json"
    container_p = os.path.join(MODEL_BASE_PATH, cohort, age_band_fname, "risk_distribution_2019.json") if USE_CONTAINER_MODELS else None

    if PREFER_S3:
        try:
            obj = s3_client.get_object(Bucket=S3_BUCKET, Key=key)
            return _from_data(json.loads(obj["Body"].read().decode("utf-8")))
        except ClientError:
            pass
    if container_p and os.path.exists(container_p):
        try:
            with open(container_p, "r") as f:
                return _from_data(json.load(f))
        except Exception:
            pass
    if not PREFER_S3:
        try:
            obj = s3_client.get_object(Bucket=S3_BUCKET, Key=key)
            return _from_data(json.loads(obj["Body"].read().decode("utf-8")))
        except ClientError as e:
            code = e.response.get("Error", {}).get("Code")
            if code not in ("NoSuchKey", "404", "NotFound", "AccessDenied", "403"):
                raise
    return None


def _normalize_feature_schema_for_training(schema: Dict[str, Any]) -> Dict[str, Any]:
    """
    Align schema with run_final_model.py _EXCLUDE_FROM_FEATURES:
    - n_event_bin (string label for per-bin routing) is not a model input.
    - n_events (continuous claim count) is excluded post-refactor: the per-bin
      routing already stratifies by density; keeping n_events as a continuous
      feature dominates gradient-boosted splits and makes per-drug
      what-if deltas flat (Δp̂ ≈ 0).  n_event_bin_ordinal (0–3) is retained
      as the density signal.
    Older feature_schema.json files may still list both → drop them here so
    the feature vector matches the retrained model's expected shape.
    """
    if not schema or not isinstance(schema, dict):
        return schema if isinstance(schema, dict) else {"features": [], "defaults": {}}
    out = dict(schema)
    feats = out.get("features")
    _to_drop = {"n_event_bin", "n_events"}
    if isinstance(feats, list) and _to_drop.intersection(feats):
        dropped = [f for f in feats if f in _to_drop]
        out["features"] = [f for f in feats if f not in _to_drop]
        defaults = out.get("defaults")
        if isinstance(defaults, dict):
            defaults = {k: v for k, v in defaults.items() if k not in _to_drop}
            out["defaults"] = defaults
        if "n_features" in out:
            out["n_features"] = len(out["features"])
        print(f"feature_schema: dropped non-model-input column(s): {dropped}")
    return out


def load_feature_schema(cohort: str, age_band: str) -> Dict[str, Any]:
    """Load feature schema JSON from container filesystem or S3."""
    age_band_fname = age_band.replace("-", "_")
    data: Optional[Dict[str, Any]] = None

    key = f"{MODEL_PREFIX}/{cohort}/{age_band_fname}/feature_schema.json"
    container_schema_path = os.path.join(MODEL_BASE_PATH, cohort, age_band_fname, "feature_schema.json") if USE_CONTAINER_MODELS else None

    if PREFER_S3:
        try:
            obj = s3_client.get_object(Bucket=S3_BUCKET, Key=key)
            data = json.loads(obj["Body"].read().decode("utf-8"))
        except ClientError:
            pass  # fall through to container

    if data is None and container_schema_path and os.path.exists(container_schema_path):
        try:
            with open(container_schema_path, 'r') as f:
                data = json.load(f)
        except Exception as e:
            print(f"Warning: Failed to load schema from container: {e}")

    if data is None and not PREFER_S3:
        try:
            obj = s3_client.get_object(Bucket=S3_BUCKET, Key=key)
            data = json.loads(obj["Body"].read().decode("utf-8"))
        except ClientError as e:
            code = e.response.get("Error", {}).get("Code")
            if code in ("NoSuchKey", "404", "NotFound"):
                return _normalize_feature_schema_for_training({"features": [], "defaults": {}})
            raise

    return _normalize_feature_schema_for_training(data or {"features": [], "defaults": {}})


# ---- n_event_bin: utilization density bin (low/medium/high/extreme) ----
# Thresholds are written by run_final_model.py (Step 6); shared with DTW and FP-Growth.
_DEFAULT_NEVENT_THRESHOLDS: Dict[str, float] = {"p25": 5.0, "p50": 15.0, "p95": 50.0}
_nevent_threshold_cache: Dict[str, Dict] = {}


def load_n_event_bin_thresholds(cohort: str, age_band: str) -> Dict[str, float]:
    """Load n_event_bin P25/P50/P95 thresholds from container or S3; fall back to defaults."""
    cache_key = f"{cohort}/{age_band}"
    if cache_key in _nevent_threshold_cache:
        return _nevent_threshold_cache[cache_key]
    age_band_fname = age_band.replace("-", "_")
    key = f"{MODEL_PREFIX}/{cohort}/{age_band_fname}/n_event_bin_thresholds.json"
    container_path = os.path.join(MODEL_BASE_PATH, cohort, age_band_fname, "n_event_bin_thresholds.json") if USE_CONTAINER_MODELS else None

    def _valid(d): return isinstance(d, dict) and "p25" in d and "p50" in d and "p95" in d

    # 1) S3 first (PREFER_S3 default) — retrained thresholds picked up without container rebuild
    if PREFER_S3:
        try:
            obj = s3_client.get_object(Bucket=S3_BUCKET, Key=key)
            data = json.loads(obj["Body"].read().decode("utf-8"))
            if _valid(data):
                _nevent_threshold_cache[cache_key] = data
                return data
        except Exception:
            pass

    # 2) Container filesystem fallback
    if container_path and os.path.exists(container_path):
        try:
            with open(container_path, "r") as fh:
                data = json.load(fh)
            if _valid(data):
                _nevent_threshold_cache[cache_key] = data
                return data
        except Exception as e:
            print(f"Warning: could not load n_event_bin thresholds from container: {e}")

    # 3) S3 fallback when PREFER_S3=false
    if not PREFER_S3:
        try:
            obj = s3_client.get_object(Bucket=S3_BUCKET, Key=key)
            data = json.loads(obj["Body"].read().decode("utf-8"))
            if _valid(data):
                _nevent_threshold_cache[cache_key] = data
                return data
        except Exception:
            pass

    return dict(_DEFAULT_NEVENT_THRESHOLDS)


def n_event_bin_from_n_events(n_events: int, thresholds: Dict[str, float]) -> str:
    """Assign low/medium/high/extreme based on P25/P50/P95 cut-points."""
    p25 = thresholds.get("p25", _DEFAULT_NEVENT_THRESHOLDS["p25"])
    p50 = thresholds.get("p50", _DEFAULT_NEVENT_THRESHOLDS["p50"])
    p95 = thresholds.get("p95", _DEFAULT_NEVENT_THRESHOLDS["p95"])
    if n_events <= p25:
        return "low"
    if n_events <= p50:
        return "medium"
    if n_events <= p95:
        return "high"
    return "extreme"


_DENSITY_BIN_ORDER: List[str] = ["low", "medium", "high", "extreme"]


def _nearest_available_bin_s3(bucket: str, requested_bin: str, probe_key_fn) -> Optional[str]:
    """Return the nearest density bin (by position in _DENSITY_BIN_ORDER) for which
    probe_key_fn(bin_name) exists in S3.  Searches outward from the requested bin,
    alternating below/above: idx-1, idx+1, idx-2, idx+2, ...
    Returns None when no adjacent bin has data (caller falls back to full-cohort)."""
    try:
        idx = _DENSITY_BIN_ORDER.index(requested_bin)
    except ValueError:
        return None
    n = len(_DENSITY_BIN_ORDER)
    for delta in range(1, n):
        for sign in (-1, 1):
            candidate_idx = idx + sign * delta
            if 0 <= candidate_idx < n:
                candidate = _DENSITY_BIN_ORDER[candidate_idx]
                if _s3_object_exists(bucket, probe_key_fn(candidate)):
                    return candidate
    return None


def _nearest_available_bin_dict(by_density: Dict[str, Any], requested_bin: str) -> Optional[str]:
    """Return the nearest density bin available as a key in by_density dict.
    Searches outward from requested_bin position in _DENSITY_BIN_ORDER."""
    try:
        idx = _DENSITY_BIN_ORDER.index(requested_bin)
    except ValueError:
        return None
    n = len(_DENSITY_BIN_ORDER)
    for delta in range(1, n):
        for sign in (-1, 1):
            candidate_idx = idx + sign * delta
            if 0 <= candidate_idx < n:
                candidate = _DENSITY_BIN_ORDER[candidate_idx]
                if by_density.get(candidate) is not None:
                    return candidate
    return None


# Ordinal encoding used in feature schema: low=0, medium=1, high=2, extreme=3
_BIN_ORDINAL_MAP: Dict[str, float] = {"low": 0.0, "medium": 1.0, "high": 2.0, "extreme": 3.0}


# ---- Platt calibration models ----
# Each calibrator is a LogisticRegression fitted on OOF predictions during MC-CV.
# It maps raw model probability → calibrated probability that matches observed event rates.
_calibration_cache: Dict[str, Any] = {}


def load_calibration_model(cohort: str, age_band: str, model_type: str, bin_name: Optional[str] = None) -> Optional[Any]:
    """Load Platt calibrator (LogisticRegression) from container or S3.

    When bin_name is supplied, loads exclusively from bin_models/{bin_name}/ —
    no full-cohort fallback.  Returns None if the per-bin calibrator is absent
    (calibration is optional; raw probability is used in that case).
    When bin_name is None, loads the full-cohort calibrator.
    """
    cache_key = f"{cohort}/{age_band}/{model_type}" if not bin_name else f"{cohort}/{age_band}/bin/{bin_name}/{model_type}"
    if cache_key in _calibration_cache:
        return _calibration_cache[cache_key]
    age_band_fname = age_band.replace("-", "_")
    filename = f"calibration_{model_type}.joblib"
    import io

    if bin_name:
        # ── Per-bin calibrator ONLY — no full-cohort fallback ─────────────────
        if USE_CONTAINER_MODELS:
            path = os.path.join(MODEL_BASE_PATH, cohort, age_band_fname, "bin_models", bin_name, filename)
            if os.path.exists(path):
                try:
                    cal = joblib.load(path)
                    _calibration_cache[cache_key] = cal
                    print(f"Loaded per-bin calibrator ({model_type}, {bin_name}) from container: {path}")
                    return cal
                except Exception as e:
                    print(f"Warning: could not load per-bin calibrator from {path}: {e}")
        s3_key = f"{MODEL_PREFIX}/{cohort}/{age_band_fname}/bin_models/{bin_name}/{filename}"
        try:
            obj = s3_client.get_object(Bucket=S3_BUCKET, Key=s3_key)
            cal = joblib.load(io.BytesIO(obj["Body"].read()))
            _calibration_cache[cache_key] = cal
            print(f"Loaded per-bin calibrator ({model_type}, {bin_name}) from S3: {s3_key}")
            return cal
        except Exception:
            print(f"Per-bin calibrator not found for bin='{bin_name}', model='{model_type}'; using raw probability.")
            return None
    else:
        # ── Full-cohort calibrator ─────────────────────────────────────────────
        if USE_CONTAINER_MODELS:
            path = os.path.join(MODEL_BASE_PATH, cohort, age_band_fname, filename)
            if os.path.exists(path):
                try:
                    cal = joblib.load(path)
                    _calibration_cache[cache_key] = cal
                    print(f"Loaded calibrator ({model_type}) from container: {path}")
                    return cal
                except Exception as e:
                    print(f"Warning: could not load calibrator from {path}: {e}")
        s3_key = f"{MODEL_PREFIX}/{cohort}/{age_band_fname}/{filename}"
        try:
            obj = s3_client.get_object(Bucket=S3_BUCKET, Key=s3_key)
            cal = joblib.load(io.BytesIO(obj["Body"].read()))
            _calibration_cache[cache_key] = cal
            print(f"Loaded calibrator ({model_type}) from S3: {s3_key}")
            return cal
        except Exception:
            return None


def apply_calibration(raw_prob: float, calibrator: Any) -> float:
    """Apply Platt calibrator to a raw probability. Returns calibrated probability in [0, 1]."""
    try:
        import numpy as _np
        cal_prob = float(calibrator.predict_proba(_np.array([[raw_prob]]))[0][1])
        return max(0.0, min(1.0, cal_prob))
    except Exception:
        return raw_prob  # Fall back to raw if calibration fails


# Risk band uses absolute thresholds (not cohort-relative percentiles) so labels match intuition (e.g. 7.7% = Low).
# low: < 20%, medium: 20–50%, high: >= 50%
DEFAULT_RISK_BAND_THRESHOLDS = {"low_medium": 0.2, "medium_high": 0.5}


def risk_band_from_score(score: float, thresholds: Optional[Dict[str, float]] = None) -> str:
    """Return low / medium / high from score using absolute thresholds (fixed cutoffs)."""
    t = thresholds if thresholds is not None else DEFAULT_RISK_BAND_THRESHOLDS
    low_med = t.get("low_medium", 0.2)
    med_high = t.get("medium_high", 0.5)
    if score < low_med:
        return "low"
    if score < med_high:
        return "medium"
    return "high"


def _bucket_one(value: float, thresholds: Dict[str, float]) -> str:
    """Return low / medium / high for a single variable using low_medium and medium_high thresholds."""
    low_med = thresholds.get("low_medium", 0)
    med_high = thresholds.get("medium_high", float("inf"))
    if value < low_med:
        return "low"
    if value < med_high:
        return "medium"
    return "high"


def patient_bucket_from_inputs(
    n_drugs: Optional[float],
    feature_schema: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Compute risk bucket (low/medium/high) from n_drugs only.
    n_pgx_drugs is a separate input and is not used for this bucket.
    Uses schema defaults for any missing value.
    """
    thresholds = feature_schema.get("patient_bucket_thresholds") or {}
    if not thresholds:
        return {"patient_bucket": None, "n_drugs_bucket": None}
    defaults = feature_schema.get("defaults", {})
    n_drugs_val = n_drugs if n_drugs is not None else defaults.get("n_drugs", 0)
    n_drugs_bucket = _bucket_one(float(n_drugs_val), thresholds.get("n_drugs", {})) if "n_drugs" in thresholds else None
    buckets = [b for b in [n_drugs_bucket] if b is not None]
    if not buckets:
        patient_bucket = None
    elif "high" in buckets:
        patient_bucket = "high"
    elif "medium" in buckets:
        patient_bucket = "medium"
    else:
        patient_bucket = "low"
    return {
        "patient_bucket": patient_bucket,
        "n_drugs_bucket": n_drugs_bucket,
    }


def get_codes_used_unknown(
    drugs: List[str],
    icds: List[str],
    cpts: List[str],
    feature_schema: Dict[str, Any],
) -> Dict[str, Dict[str, List[str]]]:
    """Validate codes against feature schema; return codes_used and codes_unknown per type."""
    feature_set = set(feature_schema.get("features", []))

    def _candidate_features(code: str, code_type: str) -> List[str]:
        """Return candidate item_* feature names for a user-provided code.

        Important: drug codes in metadata are often like 'drug_AMOXICILLIN' and the
        model feature is 'item_drug_AMOXICILLIN' (note: do NOT uppercase the 'drug_' prefix).
        """
        if code is None:
            return []
        raw = str(code).strip()
        if not raw:
            return []

        # If caller already passed a full feature name
        if raw.startswith("item_"):
            return [raw]

        if code_type == "drug":
            # Preserve casing of prefixes like drug_ / cpic_ if present
            base = raw
            # Also tolerate labels like "DRUG: XYZ" from some visualizations
            if base.upper().startswith("DRUG:"):
                base = base.split(":", 1)[1].strip().replace(" ", "_")
            return [
                f"item_{base}",
                f"item_{base.upper()}",
            ]

        # ICD/CPT are stored as uppercase codes in the schema (item_F1120, item_99213)
        return [
            f"item_{raw.upper()}",
            f"item_{raw}",
        ]

    used = {"drugs": [], "icds": [], "cpts": []}
    unknown = {"drugs": [], "icds": [], "cpts": []}
    for drug in drugs or []:
        if any(c in feature_set for c in _candidate_features(drug, "drug")):
            used["drugs"].append(drug)
        else:
            unknown["drugs"].append(drug)
    for icd in icds or []:
        if icd.upper() == 'F1120':
            continue
        if any(c in feature_set for c in _candidate_features(icd, "icd")):
            used["icds"].append(icd)
        else:
            unknown["icds"].append(icd)
    for cpt in cpts or []:
        if any(c in feature_set for c in _candidate_features(cpt, "cpt")):
            used["cpts"].append(cpt)
        else:
            unknown["cpts"].append(cpt)
    return {"codes_used": used, "codes_unknown": unknown}


def build_feature_vector(
    age: int,
    drugs: List[str],
    icds: List[str],
    cpts: List[str],
    feature_schema: Dict[str, Any],
    n_drugs: Optional[float] = None,
    pgx_num_drugs: Optional[float] = None,
    pgx_num_cpic_drugs: Optional[float] = None,
    auto_pgx_num_drugs: bool = False,
    n_event_bin: Optional[str] = None,
) -> np.ndarray:
    """
    Build feature vector matching model's expected schema.

    When auto_pgx_num_drugs=True and pgx_num_drugs/pgx_num_cpic_drugs are not
    supplied, both aggregate counts are derived from len(drugs) so that
    leave-one-out what-if deltas produce meaningful Δp̂ values instead of
    collapsing to zero (previously the schema training-median default was used
    regardless of submitted drug count, making the dominant model features
    constant across all what-if scenarios).

    When n_event_bin is provided (e.g. 'low', 'medium', 'high', 'extreme'),
    n_event_bin_ordinal is set from _BIN_ORDINAL_MAP instead of falling back
    to the schema default (training median = 1.0/medium). This ensures the
    model sees the density stratum that corresponds to the actual submitted
    code count rather than the population median.
    """
    features = {}
    
    # Initialize all features to 0
    for feature in feature_schema.get('features', []):
        features[feature] = 0.0
    
    # Set age
    if 'age' in features:
        features['age'] = float(age)

    def _set_item_feature(code: str, code_type: str) -> None:
        """Set an item_* feature to 1.0 if it exists in the schema."""
        if code is None:
            return
        raw = str(code).strip()
        if not raw:
            return

        candidates: List[str]
        if raw.startswith("item_"):
            candidates = [raw]
        elif code_type == "drug":
            base = raw
            if base.upper().startswith("DRUG:"):
                base = base.split(":", 1)[1].strip().replace(" ", "_")
            candidates = [f"item_{base}", f"item_{base.upper()}"]
        else:
            candidates = [f"item_{raw.upper()}", f"item_{raw}"]

        for f in candidates:
            if f in features:
                features[f] = 1.0
                return

    # Set drug features
    for drug in drugs:
        _set_item_feature(drug, "drug")

    # Set ICD features
    for icd in icds:
        # Exclude F1120 from ICD codes (it's the target, not an input)
        if icd.upper() == 'F1120':
            continue
        _set_item_feature(icd, "icd")
    
    for cpt in cpts:
        _set_item_feature(cpt, "cpt")

    # Optional numeric inputs: only apply when the feature exists in the schema.
    # These override schema defaults so the user can see how predictions change.
    if n_drugs is not None and "n_drugs" in features:
        features["n_drugs"] = float(n_drugs)

    # Auto-derive pgx_num_drugs / pgx_num_cpic_drugs from submitted drug count
    # when auto_pgx_num_drugs=True and no explicit values were provided.
    # This ensures what-if scenarios (drug removed → n_drug-1) produce
    # a real Δp̂ instead of staying stuck at the training-median default.
    if auto_pgx_num_drugs:
        if pgx_num_drugs is None:
            pgx_num_drugs = float(len([d for d in (drugs or []) if d]))
        if pgx_num_cpic_drugs is None:
            # Conservative proxy: CPIC ≈ pgx_num_drugs (overestimate is safe;
            # callers that know the true count should pass it explicitly).
            pgx_num_cpic_drugs = pgx_num_drugs

    if pgx_num_drugs is not None and "pgx_num_drugs" in features:
        features["pgx_num_drugs"] = float(pgx_num_drugs)
    if pgx_num_cpic_drugs is not None and "pgx_num_cpic_drugs" in features:
        features["pgx_num_cpic_drugs"] = float(pgx_num_cpic_drugs)

    # Apply schema defaults for any feature not yet set by the request.
    # Uses == 0.0 as "unset" sentinel — must run BEFORE n_event_bin_ordinal
    # override so that the ordinal value of 0.0 (low bin) is not clobbered
    # by the training-median default (1.0 / medium).
    defaults = feature_schema.get('defaults', {})
    for feature in features:
        if features[feature] == 0.0 and feature in defaults:
            features[feature] = defaults[feature]

    # Set n_event_bin_ordinal AFTER defaults so low-bin (0.0) is not
    # overwritten by the default (1.0 = medium).  Overrides the schema
    # default so the model sees the density stratum that corresponds to
    # the actual submitted code count rather than the population median.
    if n_event_bin is not None and "n_event_bin_ordinal" in features:
        features["n_event_bin_ordinal"] = _BIN_ORDINAL_MAP.get(n_event_bin, 1.0)
    
    # Convert to array in correct order
    feature_list = [features.get(f, 0.0) for f in feature_schema.get('features', [])]
    return np.array(feature_list).reshape(1, -1)


def _catboost_predict_proba(model: Any, feature_vector: np.ndarray, feature_names: List[str]) -> float:
    """
    CatBoost was trained with Pool(cat_features=item_*), not a raw float ndarray.
    Matches py_helpers.feature_importance_model_utils.predict_proba_catboost.
    Subsets to model.feature_names_ when training dropped constant columns.
    """
    row = np.asarray(feature_vector, dtype=np.float64).reshape(1, -1)
    if len(feature_names) != row.shape[1]:
        raise ValueError(
            f"Feature count mismatch: schema has {len(feature_names)} names, vector has {row.shape[1]} columns"
        )
    df = pd.DataFrame(row, columns=feature_names)
    if hasattr(model, "feature_names_") and model.feature_names_ is not None:
        mnames = list(model.feature_names_)
        missing = [c for c in mnames if c not in df.columns]
        if missing:
            raise ValueError(
                f"CatBoost model expects {len(missing)} column(s) not in schema (e.g. {missing[:3]})"
            )
        df = df[mnames].copy()
    cat_cols = [c for c in df.columns if str(c).startswith("item_")]
    if cat_cols:
        df[cat_cols] = df[cat_cols].astype(int)
    cat_indices = [df.columns.get_loc(c) for c in cat_cols] if cat_cols else None
    pool = Pool(data=df, cat_features=cat_indices)
    proba = model.predict_proba(pool)
    return float(proba[0, 1])


def predict_risk(
    cohort: str,
    age_band: str,
    feature_vector: np.ndarray,
    model_types: List[str] = ['catboost', 'xgboost', 'xgboost_rf'],
    require_all_models: bool = True,
    n_event_bin: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run prediction using the best model for this cohort/age band.

    When n_event_bin is supplied, per-bin models are loaded when present; if a
    per-bin file is missing (FileNotFoundError), the full-cohort model for that
    type is loaded and Platt calibration uses the full-cohort calibrator.

    Returns:
        {
            'predictions': {model_type: probability, ...},
            'ensemble_score': float,  # Weighted average
            'ensemble_method': str,
            'models_used': int,
            'models_failed': List[str],
            'bin_model_used': bool,   # True when a per-bin model was loaded
        }
    """
    if not MODEL_LIBS_AVAILABLE:
        raise RuntimeError("Model libraries not available")
    
    predictions = {}
    raw_predictions = {}  # Store raw (uncalibrated) probabilities for diagnostics
    calibration_applied = {}
    errors = {}
    
    # Load model weights from feature schema (performance-based from MC-CV)
    feature_schema = load_feature_schema(cohort, age_band)
    model_weights = feature_schema.get('model_weights', {
        'catboost': 1.0,
        'xgboost': 1.0,
        'xgboost_rf': 1.0
    })
    
    # Normalize weights to ensure they sum to 1.0 (if not already normalized)
    total_weight = sum(model_weights.values())
    if total_weight > 0:
        model_weights = {k: v / total_weight for k, v in model_weights.items()}
    else:
        # Fallback to equal weights if all weights are zero
        model_weights = {
            'catboost': 1.0 / 3,
            'xgboost': 1.0 / 3,
            'xgboost_rf': 1.0 / 3
        }
    
    # When using best-model-only (one weight 1.0, others 0), run only models with weight > 0 (faster, fewer failures)
    models_to_run = [m for m in model_types if model_weights.get(m, 0.0) > 0]
    if not models_to_run:
        models_to_run = model_types
    print(f"Using model weights: {model_weights} (running: {models_to_run})")
    
    bin_model_used = False  # True if at least one model was loaded from bin_models/{bin}/
    fallback_model_types: List[str] = []  # Models where per-bin artifact was missing → full-cohort used
    models_absent: List[str] = []  # Models whose artifact file was not deployed (not an error when weight < 1.0)
    for model_type in models_to_run:
        try:
            this_model_from_bin = False
            try:
                model = load_model(cohort, age_band, model_type, bin_name=n_event_bin)
                if n_event_bin:
                    _ck = f"{cohort}/{age_band}/bin/{n_event_bin}/{model_type}"
                    if _ck in _model_cache:
                        this_model_from_bin = True
                        bin_model_used = True
            except FileNotFoundError:
                # Training may copy full-cohort artifacts into each bin; if still missing, use aggregate models.
                if n_event_bin:
                    print(
                        f"Per-bin model not found for {model_type} (bin={n_event_bin}); "
                        f"loading full-cohort model."
                    )
                    model = load_model(cohort, age_band, model_type, bin_name=None)
                    this_model_from_bin = False
                    fallback_model_types.append(model_type)
                else:
                    raise

            if model_type == 'catboost':
                prob = _catboost_predict_proba(
                    model,
                    feature_vector,
                    feature_schema.get("features", []),
                )
            elif model_type in ['xgboost', 'xgboost_rf']:
                if isinstance(model, xgb.Booster):
                    dmatrix = xgb.DMatrix(feature_vector)
                    prob = model.predict(dmatrix)[0]
                    prob = max(0.0, min(1.0, prob))
                else:
                    prob = model.predict_proba(feature_vector)[0][1]
            else:
                raise ValueError(f"Unknown model type: {model_type}")

            raw_prob = float(prob)
            raw_predictions[model_type] = raw_prob
            # Platt calibration: per-bin calibrator only when this model was loaded from per-bin path
            cal_bin = n_event_bin if this_model_from_bin else None
            calibrator = load_calibration_model(cohort, age_band, model_type, bin_name=cal_bin)
            if calibrator is not None:
                calibrated = apply_calibration(raw_prob, calibrator)
                predictions[model_type] = calibrated
                calibration_applied[model_type] = True
                print(f"Calibrated {model_type}: {raw_prob:.4f} → {calibrated:.4f}")
            else:
                predictions[model_type] = raw_prob
                calibration_applied[model_type] = False

        except FileNotFoundError as fnfe:
            weight = model_weights.get(model_type, 0.0)
            if weight < 1.0:
                # Ensemble component not deployed — skip silently.
                # When this model outperforms in the future and weight becomes 1.0,
                # the missing artifact will surface as a real error below.
                print(f"[absent] {model_type} artifact not found (weight={weight:.3f}); skipping.")
                models_absent.append(model_type)
            else:
                # Sole best model is missing — this is a deployment error.
                errors[model_type] = str(fnfe)
                print(f"Error: best model {model_type} (weight=1.0) not found: {fnfe}")
        except Exception as e:
            error_msg = str(e)
            errors[model_type] = error_msg
            print(f"Error predicting with {model_type}: {error_msg}")
            # Don't add failed models to predictions
    
    # Validate that we have at least one successful prediction
    if not predictions:
        raise RuntimeError(f"All models failed. Errors: {errors}")
    
    models_used = len(predictions)
    models_failed = list(errors.keys())
    # Absent models were skipped silently (missing artifact, weight < 1.0) — not counted as failures
    required_count = len(models_to_run) - len(models_absent)
    if require_all_models and models_used < required_count:
        print(f"Warning: Only {models_used}/{required_count} required models succeeded. "
              f"Failed: {models_failed}")
        # Still proceed but log warning
    
    # Calculate ensemble score using performance-based weighted average
    # Only use weights for models that succeeded
    available_weights = {m: model_weights.get(m, 0.0) for m in predictions.keys()}
    total_weight = sum(available_weights.values())
    
    if total_weight > 0:
        # Weighted average using performance-based weights
        ensemble_score = sum(
            predictions[m] * available_weights[m]
            for m in predictions.keys()
        ) / total_weight
        ensemble_method = 'performance_weighted_average'
        weights_source = 'mc_cv_performance'
    else:
        # Fallback to simple average if weights are zero
        ensemble_score = sum(predictions.values()) / len(predictions)
        ensemble_method = 'simple_average'
        weights_source = 'equal_fallback'
        available_weights = {m: 1.0/len(predictions) for m in predictions.keys()}
        print("Warning: All model weights are zero, using simple average")
    
    # Primary model used (best for this cohort/age when weights are best-model-only)
    model_used = None
    if available_weights:
        by_weight = [(m, available_weights[m]) for m in predictions.keys() if available_weights.get(m, 0) > 0]
        if by_weight:
            model_used = max(by_weight, key=lambda x: x[1])[0]
        elif predictions:
            model_used = next(iter(predictions.keys()))

    used_full_cohort_fallback = bool(n_event_bin and fallback_model_types)
    inference_note: Optional[str] = None
    if used_full_cohort_fallback and n_event_bin:
        pretty = ", ".join(fallback_model_types)
        inference_note = (
            f"Note: Your event-density bin is «{n_event_bin}», but the per-bin trained model(s) "
            f"({pretty}) were not available on the server; the full-cohort model(s) were used for those "
            f"components instead. Risk is still valid; retrain or redeploy per-bin artifacts if you need "
            f"bin-specific models."
        )

    return {
        'predictions': predictions,
        'raw_predictions': raw_predictions,
        'calibration_applied': calibration_applied,
        'ensemble_score': float(ensemble_score),
        'ensemble_method': ensemble_method,
        'models_used': models_used,
        'models_failed': models_failed,
        'models_absent': models_absent,
        'weights_used': available_weights,
        'weights_source': weights_source,
        'model_used': model_used,
        'bin_model_used': bin_model_used,
        'n_event_bin': n_event_bin,
        'used_full_cohort_fallback': used_full_cohort_fallback,
        'fallback_model_types': fallback_model_types,
        'inference_note': inference_note,
    }


def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    Main Lambda handler for API Gateway proxy integration.
    """
    try:
        method = event.get("httpMethod", "GET")
        path = (event.get("path") or "/").rstrip("/")
        # Strip stage prefix if present (e.g. /prod/metadata -> /metadata)
        if path.startswith("/prod/"):
            path = path[5:]
        if not path.startswith("/"):
            path = "/" + path

        if method == "OPTIONS":
            return _response(200, {"message": "OK"})

        if method == "GET" and path.endswith("/metadata"):
            return handle_metadata(event)
        elif method == "GET" and path.endswith("/metrics"):
            return handle_metrics(event)
        elif method == "GET" and path.endswith("/available"):
            return handle_available(event)
        elif method == "GET" and path.endswith("/performance"):
            return handle_performance(event)
        elif method == "POST" and path.endswith("/pgx/card"):
            return handle_pgx_card(event)
        elif method == "POST" and path.endswith("/risk/drug_contributions"):
            return handle_drug_contributions(event)
        elif method == "POST" and path.endswith("/risk/comparison"):
            return handle_risk_comparison(event)
        elif method == "POST" and path.endswith("/risk"):
            return handle_risk(event)
        elif method == "POST" and path.endswith(API_SCENARIO_INTERACTIONS):
            return handle_scenario_interactions(event)
        elif method == "POST" and path.endswith(API_SCENARIO_IMPORTANCE):
            return handle_scenario_importance(event)
        elif method == "POST" and path.endswith("/scenario"):
            return _response(
                404,
                {
                    "error": "Unknown endpoint — use /scenario/importance or /scenario/interactions"
                },
            )
        elif method == "GET" and path.startswith("/visualizations/"):
            if path.endswith(API_VISUALIZATIONS_SCENARIO):
                return handle_visualizations_scenario(event)
            elif path.endswith("/dtw"):
                return handle_visualizations_dtw(event)
            elif path.endswith("/fpgrowth/network_html"):
                return handle_fpgrowth_network_html_proxy(event)
            elif path.endswith("/fpgrowth"):
                return handle_visualizations_fpgrowth(event)
            elif path.endswith("/activity_frequency") and "bupar" in path:
                return handle_visualizations_bupar_activity_frequency(event)
            elif path.endswith("/bupar"):
                return handle_visualizations_bupar(event)
            elif path.endswith("/feature_importance") or path.endswith("/feature_importance/"):
                return handle_visualizations_feature_importance(event)
            elif path.endswith("/cohort_pgx"):
                return handle_visualizations_cohort_pgx(event)
            else:
                return _response(404, {"error": "Unknown visualization endpoint"})
        
        return _response(404, {"error": f"Unsupported route: {method} {path}"})
    
    except Exception as exc:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error: {error_details}")
        return _response(
            500,
            {
                "error": "Internal server error",
                "message": str(exc),
            }
        )


def handle_metadata(event: Dict[str, Any]) -> Dict[str, Any]:
    """GET /metadata?cohort=..."""
    params = event.get("queryStringParameters") or {}
    cohort = (params.get("cohort") or "").strip() or "opioid_ed"
    
    try:
        metadata = load_metadata(cohort)
        return _response(200, metadata)
    except FileNotFoundError as e:
        return _response(404, {"error": str(e), "paths_checked": _paths_checked_from_error(e)})
    except Exception as e:
        return _response(500, {"error": str(e)})


def handle_risk(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    POST /risk
    
    Supported request body shapes:
      1) Age-driven (backward compatible):
           {
             "age": 52,
             "cohort": "opioid_ed",        # optional, inferred from age if omitted
             "drugs": ["DRUG_A", ...],
             "icds": ["F1120", ...],
             "cpts": ["80305", ...]
           }
      
      2) Dashboard-driven (explicit cohort + age_band, no raw age required):
           {
             "cohort": "opioid_ed",
             "age_band": "25-44",
             "drugs": ["DRUG_A", ...],
             "icds": ["F1120", ...],
             "cpts": ["80305", ...]
           }
      
    For the dashboard risk calculator design:
      - Both cohorts support all 8 age bands: 0-12, 13-24, 25-44, 45-54, 55-64, 65-74, 75-84, 85-114.
      - The front end populates Drugs / CPT / ICD grids from aggregated feature
        importances and sends the selected codes as the drugs/icds/cpts lists.
    """
    try:
        body = json.loads(event.get("body") or "{}")
    except json.JSONDecodeError as e:
        return _response(400, {"error": "Invalid JSON body", "detail": str(e)})
    
    raw_age = body.get("age")
    age_band_override = body.get("age_band")
    cohort = body.get("cohort")
    drugs = body.get("drugs") if body.get("drugs") is not None else []
    icds = body.get("icds") if body.get("icds") is not None else []
    cpts = body.get("cpts") if body.get("cpts") is not None else []
    if not isinstance(drugs, list):
        drugs = list(drugs) if drugs else []
    if not isinstance(icds, list):
        icds = list(icds) if icds else []
    if not isinstance(cpts, list):
        cpts = list(cpts) if cpts else []
    # Optional: n_drugs for risk bucket (low/medium/high); n_pgx_drugs, pgx_num_cpic_drugs as separate inputs (for model/display)
    n_drugs = body.get("n_drugs")
    pgx_num_drugs = body.get("pgx_num_drugs")  # separate input, not used for risk bucket
    pgx_num_cpic_drugs = body.get("pgx_num_cpic_drugs")
    # Optional: caller may supply n_events (total claim rows per patient, matching the
    # training definition COUNT(*) per mi_person_key) for accurate bin routing.
    # When absent, falls back to len(drugs)+len(icds)+len(cpts) which typically
    # routes to "low" (submitted unique-code count << training event-row count).
    n_events_override = body.get("n_events")
    if n_events_override is not None:
        try:
            n_events_override = int(n_events_override)
        except (ValueError, TypeError):
            n_events_override = None
    if n_drugs is not None:
        try:
            n_drugs = float(n_drugs)
        except (TypeError, ValueError):
            n_drugs = None
    if pgx_num_drugs is not None:
        try:
            pgx_num_drugs = float(pgx_num_drugs)
        except (TypeError, ValueError):
            pgx_num_drugs = None
    if pgx_num_cpic_drugs is not None:
        try:
            pgx_num_cpic_drugs = float(pgx_num_cpic_drugs)
        except (TypeError, ValueError):
            pgx_num_cpic_drugs = None
    
    # Resolve cohort, age_band, and effective numeric age for the model.
    if age_band_override and cohort:
        # Dashboard-style request: use explicit cohort and age_band.
        age_band = str(age_band_override)
        
        # If age is not provided, approximate with the midpoint of the band
        # so that the "age" feature gets a reasonable value.
        if raw_age is None:
            try:
                parts = age_band.split("-")
                if len(parts) == 2:
                    low = int(parts[0])
                    high = int(parts[1])
                    age = int((low + high) / 2)
                else:
                    age = 50
            except Exception:
                age = 50
        else:
            age = int(raw_age)
    else:
        # Backward compatible path: require numeric age and infer cohort/age_band.
        age = int(raw_age or 0)
        if age <= 0:
            raise ValueError("Age must be provided when cohort/age_band are not specified.")
        if not cohort:
            cohort, age_band = determine_cohort_and_age_band(age)
        else:
            # Validate / normalize age_band based on numeric age
            _, age_band = determine_cohort_and_age_band(age)
    
    try:
        # Load 2019 distribution first (needed for baseline_risk when no codes, and for chart)
        dist_2019 = load_risk_distribution_2019(cohort, age_band)

        # No Drug/ICD/CPT codes => use 2019 baseline risk (actual outcome rate) so risk is calibrated to population
        no_codes = not (drugs or icds or cpts)
        baseline_risk = dist_2019.get("baseline_risk") if dist_2019 else None
        if no_codes and baseline_risk is not None:
            risk_score = float(baseline_risk)
            risk_band = risk_band_from_score(risk_score, None)  # absolute thresholds: low <20%, medium 20-50%, high >=50%
            age_mapped = age >= 95 and age <= 114 and not age_band_override
            interpretation = "Estimated probability of target outcome (2019 holdout population) in this cohort and age band."
            feature_schema_baseline = load_feature_schema(cohort, age_band)
            bucket_info = patient_bucket_from_inputs(n_drugs, feature_schema_baseline)
            body = {
                "risk_score": risk_score,
                "risk_band": risk_band,
                "is_baseline": True,
                "patient_bucket": bucket_info.get("patient_bucket"),
                "patient_bucket_detail": {k: v for k, v in bucket_info.items() if k != "patient_bucket" and v is not None},
                "n_pgx_drugs": pgx_num_drugs,
                "pgx_num_cpic_drugs": pgx_num_cpic_drugs,
                "model_breakdown": {},
                "ensemble_info": {
                    "method": "baseline",
                    "models_used": 0,
                    "models_failed": [],
                    "weights": {},
                    "weights_source": "2019_outcome_rate",
                },
                "age_band_used": age_band,
                "cohort_used": cohort,
                "age": age,
                "age_mapped": age_mapped,
                "age_mapping_note": f"Age {age} in age band 85-114" if age_mapped else None,
                "codes_used": {"drugs": [], "icds": [], "cpts": []},
                "codes_unknown": {"drugs": [], "icds": [], "cpts": []},
                "interpretation": interpretation,
            }
            if dist_2019 is not None:
                body["dist"] = dist_2019
            body["risk_band_thresholds"] = DEFAULT_RISK_BAND_THRESHOLDS
            return _response(200, body)

        # Load feature schema and run ensemble when user has entered codes
        feature_schema = load_feature_schema(cohort, age_band)
        codes_validation = get_codes_used_unknown(drugs, icds, cpts, feature_schema)
        # n_event_bin reflects PATIENT UTILIZATION DENSITY (total claim-event rows,
        # COUNT(*) per mi_person_key) — NOT the number of codes the clinician selected.
        # Codes selected in the UI identify which conditions/drugs the patient has;
        # n_events captures how heavily the patient uses the healthcare system.
        # When the caller supplies n_events, map it to low/medium/high/extreme via
        # cohort-/age-band-specific P25/P50/P95 thresholds.  Without n_events, default
        # to "low" (most conservative; do not use code count as a proxy).
        n_events_submitted = len(drugs or []) + len(icds or [])
        nevent_thresholds = load_n_event_bin_thresholds(cohort, age_band)
        if n_events_override is not None:
            n_events_for_bin = n_events_override
            n_event_bin_value = n_event_bin_from_n_events(n_events_for_bin, nevent_thresholds)
        else:
            n_events_for_bin = None
            n_event_bin_value = "low"  # default when patient utilization density is unknown

        # When pgx_num_drugs is not explicitly provided by the caller, auto-derive
        # it from the submitted drug count so that the model's dominant aggregate
        # features respond to what the user actually submitted.
        _auto_pgx = pgx_num_drugs is None and pgx_num_cpic_drugs is None
        feature_vector = build_feature_vector(
            age,
            drugs,
            icds,
            cpts,
            feature_schema,
            n_drugs=n_drugs,
            pgx_num_drugs=pgx_num_drugs,
            pgx_num_cpic_drugs=pgx_num_cpic_drugs,
            auto_pgx_num_drugs=_auto_pgx,
            n_event_bin=n_event_bin_value,
        )

        try:
            ensemble_result = predict_risk(
                cohort, age_band, feature_vector,
                require_all_models=True,
                n_event_bin=n_event_bin_value,
            )
        except FileNotFoundError as model_err:
            return _response(404, {
                "error": f"Per-bin model not available for density bin '{n_event_bin_value}'",
                "detail": str(model_err),
                "cohort": cohort,
                "age_band": age_band,
                "n_event_bin": n_event_bin_value,
                "nevent_thresholds": nevent_thresholds,
                "suggestion": "Select a different event density bin, or leave # total events blank to use the low-density model.",
            })

        risk_score = ensemble_result['ensemble_score']
        model_predictions = ensemble_result['predictions']
        risk_band = risk_band_from_score(risk_score, None)  # absolute thresholds: low <20%, medium 20-50%, high >=50%

        age_mapped = age >= 95 and age <= 114 and not age_band_override
        age_mapping_note = f"Age {age} in age band 85-114" if age_mapped else None

        model_used = ensemble_result.get("model_used")
        bin_model_used = ensemble_result.get("bin_model_used", False)
        used_full_cohort_fallback = ensemble_result.get("used_full_cohort_fallback", False)
        fallback_model_types = ensemble_result.get("fallback_model_types") or []
        inference_note = ensemble_result.get("inference_note")
        interpretation = "Estimated probability of target outcome (2019 holdout context) for the selected codes in this cohort and age band."
        if model_used:
            bin_note = f" [per-bin model: {n_event_bin_value}]" if bin_model_used else ""
            interpretation = f"Risk from {model_used}{bin_note} (best model for this cohort/age). " + interpretation

        bucket_info = patient_bucket_from_inputs(n_drugs, feature_schema)
        calibration_applied = ensemble_result.get('calibration_applied', {})
        is_calibrated = any(calibration_applied.values()) if calibration_applied else False
        _raw_preds = ensemble_result.get('raw_predictions', {})
        _weights = ensemble_result.get('weights_used', {})
        _w_sum = sum(_weights.get(m, 0.0) for m in _raw_preds) or 1.0
        raw_risk_score = float(sum(_raw_preds[m] * _weights.get(m, 0.0) for m in _raw_preds) / _w_sum) if _raw_preds else float(risk_score)
        body = {
            "risk_score": float(risk_score),
            "risk_band": risk_band,
            "n_event_bin": n_event_bin_value,
            "n_events": n_events_submitted,
            "n_events_for_bin": n_events_for_bin,
            "nevent_thresholds": nevent_thresholds,
            "is_baseline": False,
            "calibrated": is_calibrated,
            "raw_risk_score": raw_risk_score if is_calibrated else None,
            "model_inputs": {
                "n_drugs": n_drugs,
                "pgx_num_drugs": pgx_num_drugs,
                "pgx_num_cpic_drugs": pgx_num_cpic_drugs,
                "n_events": n_events_submitted,
            },
            "patient_bucket": bucket_info.get("patient_bucket"),
            "patient_bucket_detail": {k: v for k, v in bucket_info.items() if k != "patient_bucket" and v is not None},
            "n_pgx_drugs": pgx_num_drugs,
            "pgx_num_cpic_drugs": pgx_num_cpic_drugs,
            "model_used": model_used,
            "bin_model_used": bin_model_used,
            "used_full_cohort_fallback": used_full_cohort_fallback,
            "fallback_model_types": fallback_model_types,
            "inference_note": inference_note,
            "model_breakdown": model_predictions,
            "ensemble_info": {
                "method": ensemble_result['ensemble_method'],
                "models_used": ensemble_result['models_used'],
                "models_failed": ensemble_result['models_failed'],
                "models_absent": ensemble_result.get('models_absent', []),
                "weights": ensemble_result['weights_used'],
                "weights_source": ensemble_result.get('weights_source', 'unknown')
            },
            "age_band_used": age_band,
            "cohort_used": cohort,
            "age": age,
            "age_mapped": age_mapped,
            "age_mapping_note": age_mapping_note,
            "codes_used": codes_validation["codes_used"],
            "codes_unknown": codes_validation["codes_unknown"],
            "interpretation": interpretation,
        }
        if dist_2019 is not None:
            body["dist"] = dist_2019
        return _response(200, body)
    
    except Exception as e:
        import traceback
        return _response(500, {
            "error": str(e),
            "traceback": traceback.format_exc()
        })


def handle_risk_comparison(event: Dict[str, Any]) -> Dict[str, Any]:
    """POST /risk/comparison. Base may include cohort and age_band (from cohort tab + age); else derived from age.
    When base or a scenario has no Drug/ICD/CPT codes, uses baseline_risk (2019 outcome rate) for consistency with POST /risk."""
    try:
        body = json.loads(event.get("body") or "{}")
    except json.JSONDecodeError as e:
        return _response(400, {"error": "Invalid JSON body", "detail": str(e)})
    
    base = body.get("base") or {}
    scenarios = body.get("scenarios")
    if not isinstance(scenarios, list):
        scenarios = []
    
    base_age = int(base.get("age", 0))
    cohort = base.get("cohort")
    age_band = base.get("age_band")
    if cohort and age_band:
        cohort, age_band = str(cohort), str(age_band)
    else:
        cohort, age_band = determine_cohort_and_age_band(base_age)
    
    try:
        dist_2019 = load_risk_distribution_2019(cohort, age_band)
        baseline_risk = dist_2019.get("baseline_risk") if dist_2019 else None
        feature_schema = load_feature_schema(cohort, age_band)
        
        base_drugs = base.get("drugs") or []
        base_icds = base.get("icds") or []
        base_cpts = base.get("cpts") or []
        if not isinstance(base_drugs, list):
            base_drugs = list(base_drugs) if base_drugs else []
        if not isinstance(base_icds, list):
            base_icds = list(base_icds) if base_icds else []
        if not isinstance(base_cpts, list):
            base_cpts = list(base_cpts) if base_cpts else []
        base_no_codes = not (base_drugs or base_icds or base_cpts)
        
        nevent_thresholds = load_n_event_bin_thresholds(cohort, age_band)
        if base_no_codes and baseline_risk is not None:
            base_risk = float(baseline_risk)
        else:
            base_n_events = len(base_drugs) + len(base_icds) + len(base_cpts)
            base_bin = n_event_bin_from_n_events(base_n_events, nevent_thresholds)
            base_feature_vector = build_feature_vector(
                base_age, base_drugs, base_icds, base_cpts, feature_schema,
                auto_pgx_num_drugs=True,
                n_event_bin=base_bin,
            )
            base_ensemble = predict_risk(cohort, age_band, base_feature_vector,
                                         require_all_models=True, n_event_bin=base_bin)
            base_risk = base_ensemble['ensemble_score']
        
        scenario_results = []
        for scenario in scenarios:
            s_drugs = scenario.get("drugs") or []
            s_icds = scenario.get("icds") or []
            s_cpts = scenario.get("cpts") or []
            if not isinstance(s_drugs, list):
                s_drugs = list(s_drugs) if s_drugs else []
            if not isinstance(s_icds, list):
                s_icds = list(s_icds) if s_icds else []
            if not isinstance(s_cpts, list):
                s_cpts = list(s_cpts) if s_cpts else []
            s_no_codes = not (s_drugs or s_icds or s_cpts)
            if s_no_codes and baseline_risk is not None:
                scenario_risk = float(baseline_risk)
                scenario_ensemble_preds = {}
            else:
                s_n_events = len(s_drugs) + len(s_icds) + len(s_cpts)
                s_bin = n_event_bin_from_n_events(s_n_events, nevent_thresholds)
                scenario_feature_vector = build_feature_vector(
                    base_age, s_drugs, s_icds, s_cpts, feature_schema,
                    auto_pgx_num_drugs=True,
                    n_event_bin=s_bin,
                )
                scenario_ensemble = predict_risk(cohort, age_band, scenario_feature_vector,
                                                 require_all_models=True, n_event_bin=s_bin)
                scenario_risk = scenario_ensemble['ensemble_score']
                scenario_ensemble_preds = scenario_ensemble['predictions']
            scenario_results.append({
                "name": scenario.get("name", "Scenario"),
                "risk_score": float(scenario_risk),
                "delta": float(scenario_risk - base_risk),
                "model_breakdown": scenario_ensemble_preds
            })
        
        return _response(200, {
            "base_risk": float(base_risk),
            "scenarios": scenario_results
        })
    
    except Exception as e:
        import traceback
        return _response(500, {
            "error": str(e),
            "traceback": traceback.format_exc()
        })


def handle_drug_contributions(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    POST /risk/drug_contributions

    Computes per-drug leave-one-out what-if Δp̂ for every drug in the
    submitted list, powering the What-If / deprescribing tab.

    For each drug D:
      - Build feature vector with D removed (item_drug_D=0) AND pgx_num_drugs -= 1
      - Δp̂(D) = p̂_base − p̂_without_D
      - Positive Δp̂ means removing D reduces risk → deprescribing candidate

    Request body (same shape as POST /risk):
    {
        "cohort": "non_opioid_ed",
        "age_band": "65-74",
        "age": 70,                 // optional
        "drugs": ["LEVOFLOXACIN", "LORAZEPAM", "CARVEDILOL"],
        "icds": [...],             // optional
        "cpts": [...],             // optional
        "pgx_num_cpic_drugs": 3    // optional — passed through if known
    }

    Response:
    {
        "base_risk": 0.412,
        "n_event_bin": "medium",
        "drug_contributions": [
            {"drug": "LEVOFLOXACIN", "risk_without": 0.318, "delta_risk": 0.094,
             "pct_contribution": 22.8, "rank": 1},
            {"drug": "LORAZEPAM",    "risk_without": 0.355, "delta_risk": 0.057,
             "pct_contribution": 13.8, "rank": 2},
            ...
        ],
        "cohort": "non_opioid_ed",
        "age_band": "65-74"
    }
    """
    try:
        body = json.loads(event.get("body") or "{}")
    except json.JSONDecodeError as e:
        return _response(400, {"error": "Invalid JSON body", "detail": str(e)})

    cohort         = body.get("cohort")
    age_band_input = body.get("age_band")
    raw_age        = body.get("age")
    drugs          = body.get("drugs") or []
    icds           = body.get("icds") or []
    cpts           = body.get("cpts") or []
    pgx_num_cpic_drugs_input = body.get("pgx_num_cpic_drugs")

    if not isinstance(drugs, list): drugs = list(drugs) if drugs else []
    if not isinstance(icds,  list): icds  = list(icds)  if icds  else []
    if not isinstance(cpts,  list): cpts  = list(cpts)  if cpts  else []

    try:
        # Resolve cohort / age_band / age
        if age_band_input and cohort:
            age_band = str(age_band_input)
            if raw_age is None:
                try:
                    parts = age_band.split("-")
                    age = int((int(parts[0]) + int(parts[1])) / 2) if len(parts) == 2 else 50
                except Exception:
                    age = 50
            else:
                age = int(raw_age)
        else:
            age = int(raw_age or 0)
            if age <= 0:
                return _response(400, {"error": "age or (cohort + age_band) required"})
            cohort, age_band = determine_cohort_and_age_band(age)

        if not drugs:
            return _response(400, {"error": "At least one drug is required"})

        feature_schema = load_feature_schema(cohort, age_band)
        n_events_submitted = len(drugs) + len(icds) + len(cpts)
        nevent_thresholds  = load_n_event_bin_thresholds(cohort, age_band)
        n_event_bin_value  = n_event_bin_from_n_events(n_events_submitted, nevent_thresholds)

        # ── Base prediction (all drugs present) ──────────────────────────────
        n_drugs_base = float(len(drugs))
        cpic_base    = float(pgx_num_cpic_drugs_input) if pgx_num_cpic_drugs_input is not None else n_drugs_base
        base_fv = build_feature_vector(
            age, drugs, icds, cpts, feature_schema,
            pgx_num_drugs=n_drugs_base,
            pgx_num_cpic_drugs=cpic_base,
            n_event_bin=n_event_bin_value,
        )
        base_result = predict_risk(
            cohort, age_band, base_fv,
            require_all_models=False,
            n_event_bin=n_event_bin_value,
        )
        base_risk = base_result["ensemble_score"]

        # ── Leave-one-out per drug ────────────────────────────────────────────
        contributions = []
        n_drugs_loo = max(n_drugs_base - 1.0, 0.0)
        cpic_loo    = max(cpic_base - 1.0, 0.0)

        # n_event_bin for LOO (one fewer event)
        n_events_loo = max(n_events_submitted - 1, 0)
        n_event_bin_loo = n_event_bin_from_n_events(n_events_loo, nevent_thresholds)

        for drug in drugs:
            drugs_loo = [d for d in drugs if d != drug]
            try:
                loo_fv = build_feature_vector(
                    age, drugs_loo, icds, cpts, feature_schema,
                    pgx_num_drugs=n_drugs_loo,
                    pgx_num_cpic_drugs=cpic_loo,
                    n_event_bin=n_event_bin_loo,
                )
                loo_result = predict_risk(
                    cohort, age_band, loo_fv,
                    require_all_models=False,
                    n_event_bin=n_event_bin_loo,
                )
                risk_without = loo_result["ensemble_score"]
            except Exception as loo_err:
                print(f"LOO failed for {drug}: {loo_err}")
                risk_without = base_risk  # fallback: no contribution

            delta      = float(base_risk - risk_without)
            pct_contrib = (delta / base_risk * 100.0) if base_risk > 0 else 0.0
            contributions.append({
                "drug":          drug,
                "risk_without":  round(float(risk_without), 6),
                "delta_risk":    round(delta, 6),
                "pct_contribution": round(pct_contrib, 2),
            })

        # Sort by delta_risk descending (highest contributor first)
        contributions.sort(key=lambda x: x["delta_risk"], reverse=True)
        for i, c in enumerate(contributions, 1):
            c["rank"] = i

        return _response(200, {
            "base_risk":          round(base_risk, 6),
            "n_event_bin":        n_event_bin_value,
            "n_drugs":            int(n_drugs_base),
            "drug_contributions": contributions,
            "cohort":             cohort,
            "age_band":           age_band,
        })

    except Exception as e:
        import traceback
        return _response(500, {"error": str(e), "traceback": traceback.format_exc()})


def handle_pgx_card(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handle PGx card generation request.
    Generates anonymous, generic card with timestamp and IP address.
    Patient ID is optional and not required for privacy.
    """
    try:
        body = json.loads(event.get("body", "{}"))
        variants = body.get("variants", [])
        patient_id = body.get("patient_id")  # Optional, can be None
        
        if not variants:
            return _response(400, {"error": "No variants provided"})
        
        # Extract IP address from event (API Gateway provides this)
        ip_address = (
            event.get("requestContext", {}).get("identity", {}).get("sourceIp") or
            event.get("headers", {}).get("X-Forwarded-For", "").split(",")[0].strip() or
            event.get("headers", {}).get("x-forwarded-for", "").split(",")[0].strip() or
            "Unknown"
        )
        
        # Generate timestamp
        from datetime import datetime
        timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
        
        card_data = generate_pgx_card(variants, timestamp, ip_address, patient_id)
        return _response(200, card_data)
    
    except Exception as e:
        import traceback
        return _response(500, {
            "error": str(e),
            "traceback": traceback.format_exc()
        })


def generate_pgx_card(variants: List[Dict[str, Any]], timestamp: str, ip_address: str, patient_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Generate anonymous, generic PGx Patient Card from SNP variants.
    
    Args:
        variants: List of dicts with 'gene' and 'variants' keys
        timestamp: Timestamp when card was generated
        ip_address: IP address of requester (for tracking, not identification)
        patient_id: Optional patient identifier (not required, for privacy)
        
    Returns:
        Dict with timestamp, ip_address, optional patient_id, genes, and drugs requiring modifications
    """
    import csv
    
    # Load CPIC data (from container or S3)
    cpic_data = load_cpic_data()
    
    # Process variants
    genes_processed = []
    drugs_found = []
    
    for variant in variants:
        gene = variant.get("gene", "").upper()
        variant_list = variant.get("variants", [])
        
        if not gene or not variant_list:
            continue
        
        # Store gene info
        genes_processed.append({
            "gene": gene,
            "variants": variant_list,
            "allele_count": len([v for v in variant_list if v and v != "0"])
        })
        
        # Find drugs associated with this gene
        gene_drugs = cpic_data.get(gene, [])
        for drug_info in gene_drugs:
            # Avoid duplicates
            if not any(d["drug"] == drug_info["drug"] and d["gene"] == gene for d in drugs_found):
                drugs_found.append({
                    "gene": gene,
                    "drug": drug_info["drug"],
                    "guideline_url": drug_info.get("guideline", ""),
                    "cpic_level": drug_info.get("cpic_level", ""),
                    "fda_label": drug_info.get("pgx_on_fda_label", "")
                })
    
    result = {
        "timestamp": timestamp,
        "ip_address": ip_address,
        "genes": genes_processed,
        "drugs": drugs_found
    }
    
    # Only include patient_id if provided (optional)
    if patient_id:
        result["patient_id"] = patient_id
    
    return result


def _dataframe_to_cpic_dict(df: Any) -> Dict[str, List[Dict[str, Any]]]:
    """Build gene -> list of drug-info dicts from CPIC DataFrame (Excel or Parquet)."""
    cpic_data = {}
    gene_col = None
    drug_col = None
    guideline_col = None
    cpic_level_col = None
    fda_label_col = None
    for col in df.columns:
        col_lower = str(col).lower()
        if "gene" in col_lower and gene_col is None:
            gene_col = col
        elif "drug" in col_lower and drug_col is None:
            drug_col = col
        elif "guideline" in col_lower and guideline_col is None:
            guideline_col = col
        elif "cpic" in col_lower and "level" in col_lower and cpic_level_col is None:
            cpic_level_col = col
        elif ("fda" in col_lower or "label" in col_lower) and fda_label_col is None:
            fda_label_col = col
    if not gene_col or not drug_col:
        return cpic_data
    for _, row in df.iterrows():
        gene = str(row.get(gene_col, "")).upper().strip()
        drug = str(row.get(drug_col, "")).strip()
        if not gene or not drug or gene == "NAN" or drug == "NAN":
            continue
        if gene not in cpic_data:
            cpic_data[gene] = []
        if not any(d["drug"] == drug for d in cpic_data[gene]):
            cpic_data[gene].append({
                "drug": drug,
                "guideline": str(row.get(guideline_col, "")) if guideline_col else "",
                "cpic_level": str(row.get(cpic_level_col, "")) if cpic_level_col else "",
                "pgx_on_fda_label": str(row.get(fda_label_col, "")) if fda_label_col else "",
            })
    return cpic_data


def load_cpic_data() -> Dict[str, List[Dict[str, Any]]]:
    """
    Load CPIC gene-drug pairs. Prefers Parquet via DuckDB (faster), then Excel.
    Tries: container parquet -> container Excel -> S3 parquet -> S3 Excel.
    """
    import pandas as pd

    base_data_dir = OFFLINE_DATA_PATH or "/var/task/data"
    container_parquet = os.path.join(base_data_dir, "cpic_gene-drug_pairs.parquet")
    container_excel = os.path.join(base_data_dir, "cpic_gene-drug_pairs.xlsx")

    # CPIC data is stored under gold/dashboard/data (not metadata)
    s3_parquet_key = "gold/dashboard/data/cpic_gene-drug_pairs.parquet"
    s3_excel_key = "gold/dashboard/data/cpic_gene-drug_pairs.xlsx"

    # 1) Container Parquet (DuckDB)
    if DUCKDB_AVAILABLE and os.path.exists(container_parquet):
        try:
            con = duckdb.connect(":memory:")
            df = con.execute("SELECT * FROM read_parquet(?)", [container_parquet]).fetchdf()
            con.close()
            cpic_data = _dataframe_to_cpic_dict(df)
            if cpic_data:
                n = sum(len(drugs) for drugs in cpic_data.values())
                print(f"Loaded {n} gene-drug pairs from Parquet (DuckDB)")
                return cpic_data
        except Exception as e:
            print(f"CPIC Parquet (container) failed: {e}, trying Excel...")

    # 2) Container Excel
    if EXCEL_AVAILABLE and os.path.exists(container_excel):
        try:
            df = pd.read_excel(container_excel, engine="openpyxl")
            cpic_data = _dataframe_to_cpic_dict(df)
            if cpic_data:
                n = sum(len(drugs) for drugs in cpic_data.values())
                print(f"Loaded {n} gene-drug pairs from Excel (container)")
                return cpic_data
        except Exception as e:
            print(f"CPIC Excel (container) failed: {e}")

    # 3) S3 Parquet (DuckDB)
    if DUCKDB_AVAILABLE:
        try:
            s3 = boto3.client("s3")
            obj = s3.get_object(Bucket=S3_BUCKET, Key=s3_parquet_key)
            body = obj["Body"].read()
            with tempfile.NamedTemporaryFile(delete=False, suffix=".parquet") as f:
                f.write(body)
                tmp = f.name
            try:
                con = duckdb.connect(":memory:")
                df = con.execute("SELECT * FROM read_parquet(?)", [tmp]).fetchdf()
                con.close()
                cpic_data = _dataframe_to_cpic_dict(df)
                if cpic_data:
                    n = sum(len(drugs) for drugs in cpic_data.values())
                    print(f"Loaded {n} gene-drug pairs from Parquet (S3)")
                    return cpic_data
            finally:
                try:
                    os.remove(tmp)
                except OSError:
                    pass
        except Exception as e:
            print(f"CPIC Parquet (S3) failed: {e}, trying S3 Excel...")

    # 4) S3 Excel
    try:
        s3 = boto3.client("s3")
        obj = s3.get_object(Bucket=S3_BUCKET, Key=s3_excel_key)
        df = pd.read_excel(BytesIO(obj["Body"].read()), engine="openpyxl")
        cpic_data = _dataframe_to_cpic_dict(df)
        if cpic_data:
            n = sum(len(drugs) for drugs in cpic_data.values())
            print(f"Loaded {n} gene-drug pairs from Excel (S3)")
            return cpic_data
    except Exception as e:
        print(f"CPIC Excel (S3) failed: {e}")

    raise FileNotFoundError(
        "CPIC data not found. Ensure cpic_gene-drug_pairs.parquet or cpic_gene-drug_pairs.xlsx "
        "is in container data/ or S3 under gold/dashboard/data/."
    )


def load_interaction_analysis(cohort: str, age_band: str, model_type: str = "xgboost") -> pd.DataFrame:
    """
    Load interaction analysis results from S3.
    
    Args:
        cohort: Cohort name
        age_band: Age band
        model_type: Model type ('xgboost', 'catboost', 'xgboost_rf')
    
    Returns:
        DataFrame with interaction analysis results, or empty DataFrame if not found
    """
    age_band_fname = age_band.replace("-", "_")
    
    # Try container filesystem first
    if USE_CONTAINER_MODELS:
        container_interaction_path = os.path.join(
            MODEL_BASE_PATH,
            "..",  # Go up from models to outputs
            "..",
            "8_ffa_analysis",
            "outputs",
            cohort,
            age_band_fname,
            model_type,
            "interaction_analysis.csv"
        )
        if os.path.exists(container_interaction_path):
            try:
                return pd.read_csv(container_interaction_path)
            except Exception as e:
                print(f"Warning: Failed to load interaction analysis from container: {e}")
    
    # Fallback to S3
    s3_key = f"gold/ffa_analysis/{cohort}/{age_band}/{model_type}/interaction_analysis.csv"
    
    try:
        obj = s3_client.get_object(Bucket=S3_BUCKET, Key=s3_key)
        df = pd.read_csv(BytesIO(obj["Body"].read()))
        print(f"Loaded interaction analysis from S3: s3://{S3_BUCKET}/{s3_key}")
        return df
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code")
        if code in ("NoSuchKey", "404", "NotFound"):
            print(f"Interaction analysis not found: s3://{S3_BUCKET}/{s3_key}")
            return pd.DataFrame()
        raise


def filter_interactions_by_features(interaction_df: pd.DataFrame, selected_features: List[str]) -> pd.DataFrame:
    """
    Filter interaction DataFrame to only include interactions involving selected features.
    
    Args:
        interaction_df: DataFrame with interaction results
        selected_features: List of feature names to filter by
    
    Returns:
        Filtered DataFrame
    """
    if interaction_df.empty:
        return interaction_df
    
    # Create set of selected features for fast lookup
    selected_set = set(selected_features)
    
    # Filter rows where any feature in the combination is in selected_features
    def has_selected_feature(combo_str: str) -> bool:
        features = combo_str.split("|")
        return any(f in selected_set for f in features)
    
    mask = interaction_df['feature_combination'].apply(has_selected_feature)
    return interaction_df[mask].copy()


def handle_scenario_interactions(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    POST /scenario/interactions
    
    Returns multi-feature interaction analysis results.
    
    Request Body:
    {
        "cohort": "opioid_ed",
        "age_band": "25-44",
        "selected_features": ["item_drug_A", "item_drug_B"],  // optional
        "max_interaction_size": 2,  // optional, default 2
        "model_type": "xgboost"  // optional, default "xgboost"
    }
    
    Response:
    {
        "interactions": [...],
        "top_interactions": [...],
        "summary": {
            "total_interactions_tested": 45,
            "positive_synergies": 12,
            "negative_synergies": 8,
            "neutral": 25
        }
    }
    """
    try:
        body = json.loads(event.get("body") or "{}")
        cohort = body.get("cohort")
        age_band = body.get("age_band")
        selected_features = body.get("selected_features", [])
        max_interaction_size = body.get("max_interaction_size", 2)
        model_type = body.get("model_type", "xgboost")
        
        if not cohort or not age_band:
            return _response(400, {"error": "cohort and age_band are required"})
        
        # Load interaction analysis results from S3
        interaction_df = load_interaction_analysis(cohort, age_band, model_type)
        
        if interaction_df.empty:
            return _response(200, {
                "interactions": [],
                "top_interactions": [],
                "summary": {
                    "total_interactions_tested": 0,
                    "positive_synergies": 0,
                    "negative_synergies": 0,
                    "neutral": 0
                },
                "message": "No interaction analysis results found. Run FFA analysis with enable_interaction_analysis=True."
            })
        
        # Filter to selected features if provided
        if selected_features:
            interaction_df = filter_interactions_by_features(interaction_df, selected_features)
        
        # Filter by max_interaction_size
        interaction_df = interaction_df[interaction_df['interaction_size'] <= max_interaction_size]
        
        # Ensure synergy_type column exists (for backward compatibility)
        if 'synergy_type' not in interaction_df.columns:
            interaction_df['synergy_type'] = interaction_df['interaction_effect'].apply(
                lambda x: 'positive' if x > 0.01 else ('negative' if x < -0.01 else 'neutral')
            )
        
        # Format response
        interactions = interaction_df.to_dict('records')
        top_interactions = interaction_df.nlargest(10, 'interaction_effect', keep='all').to_dict('records')
        
        summary = {
            'total_interactions_tested': len(interaction_df),
            'positive_synergies': len(interaction_df[interaction_df['synergy_type'] == 'positive']),
            'negative_synergies': len(interaction_df[interaction_df['synergy_type'] == 'negative']),
            'neutral': len(interaction_df[interaction_df['synergy_type'] == 'neutral'])
        }
        
        return _response(200, {
            'interactions': interactions,
            'top_interactions': top_interactions,
            'summary': summary
        })
    
    except Exception as e:
        import traceback
        return _response(500, {
            "error": str(e),
            "traceback": traceback.format_exc()
        })


def load_interaction_importance(cohort: str, age_band: str, model_type: str = "xgboost", bin_name: Optional[str] = None) -> pd.DataFrame:
    """
    Load feature importance results.

    When bin_name is supplied the per-bin XGBoost/CatBoost feature importance
    CSV is loaded exclusively from bin_models/{bin_name}/ (container or S3).
    Returns an empty DataFrame if the per-bin file is absent — no full-cohort
    fallback, so the caller can surface a meaningful "not available" message.

    When bin_name is None the full-cohort FFA interaction importance parquet is
    loaded (legacy/baseline path).

    The per-bin CSV has columns 'feature' and 'importance'; the latter is
    normalized to 'interaction_importance' for the scenario response schema.
    """
    if not MODEL_LIBS_AVAILABLE:
        print("ERROR: pandas not available. Cannot load interaction importance.")
        return pd.DataFrame()

    age_band_fname = age_band.replace("-", "_")

    if bin_name:
        # ── Per-bin feature importance CSV only ───────────────────────────────
        fi_fname = f"{cohort}_{age_band_fname}_{model_type}_feature_importance.csv"
        if USE_CONTAINER_MODELS:
            p = Path(MODEL_BASE_PATH) / cohort / age_band_fname / "bin_models" / bin_name / fi_fname
            if p.exists():
                try:
                    df = pd.read_csv(p)
                    if "importance" in df.columns and "interaction_importance" not in df.columns:
                        df = df.rename(columns={"importance": "interaction_importance"})
                    print(f"Loaded per-bin FI ({bin_name}/{model_type}) from container")
                    return df
                except Exception as e:
                    print(f"Warning: could not load per-bin FI from container: {e}")
        s3_key = f"gold/final_model/{cohort}/{age_band}/bin_models/{bin_name}/{fi_fname}"
        try:
            obj = s3_client.get_object(Bucket=S3_BUCKET, Key=s3_key)
            import io
            df = pd.read_csv(io.BytesIO(obj["Body"].read()))
            if "importance" in df.columns and "interaction_importance" not in df.columns:
                df = df.rename(columns={"importance": "interaction_importance"})
            print(f"Loaded per-bin FI ({bin_name}/{model_type}) from S3: {s3_key}")
            return df
        except Exception:
            print(f"Per-bin feature importance not found for bin='{bin_name}', model='{model_type}'. Run notebook 3 first.")
            return pd.DataFrame()
    else:
        # ── Full-cohort FFA interaction importance (parquet) ──────────────────
        if USE_CONTAINER_MODELS:
            container_path = Path(MODEL_BASE_PATH).parent.parent / "8_ffa_analysis" / "outputs" / cohort / age_band_fname / model_type / "interaction_importance.parquet"
            if container_path.exists():
                try:
                    return pd.read_parquet(container_path)
                except Exception as e:
                    print(f"Warning: Failed to load interaction importance from container: {e}")
        s3_key = f"gold/ffa_analysis/{cohort}/{age_band}/{model_type}/interaction_importance.parquet"
        try:
            obj = s3_client.get_object(Bucket=S3_BUCKET, Key=s3_key)
            df = pd.read_parquet(BytesIO(obj["Body"].read()))
            print(f"Loaded interaction importance from S3: s3://{S3_BUCKET}/{s3_key}")
            return df
        except ClientError as e:
            code = e.response.get("Error", {}).get("Code")
            if code in ("NoSuchKey", "404", "NotFound"):
                print(f"Interaction importance not found: s3://{S3_BUCKET}/{s3_key}")
                return pd.DataFrame()
            raise


def handle_scenario_importance(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    POST /scenario/importance
    
    Returns single-feature interaction importance results, optionally filtered by selected drugs.
    
    Request Body:
    {
        "cohort": "opioid_ed",
        "age_band": "25-44",
        "selected_drugs": ["DRUG_A", "DRUG_B"],  // optional - filter to these drugs
        "top_n": 10,  // optional, default 10 - return top N features
        "model_type": "xgboost"  // optional, default "xgboost"
    }
    
    Response:
    {
        "interaction_importance": [
            {
                "feature": "item_DRUG_A",
                "interaction_importance": 0.123456,
                "rank": 1
            },
            ...
        ],
        "summary": {
            "total_features": 50,
            "filtered_features": 5,
            "selected_drugs": ["DRUG_A", "DRUG_B"]
        }
    }
    """
    try:
        body = json.loads(event.get("body") or "{}")
        cohort = body.get("cohort")
        age_band = body.get("age_band")
        selected_drugs = body.get("selected_drugs", [])
        top_n = body.get("top_n", 10)
        model_type = body.get("model_type", "xgboost")
        n_event_bin = body.get("n_event_bin") or None  # 'low'/'medium'/'high'/'extreme' or None

        if not cohort or not age_band:
            return _response(400, {"error": "cohort and age_band are required"})

        # Load per-bin or full-cohort feature importance
        interaction_df = load_interaction_importance(cohort, age_band, model_type, bin_name=n_event_bin)
        
        if interaction_df.empty:
            msg = (
                f"No feature importance found for bin='{n_event_bin}'. Run train_per_bin() (notebook 3) first."
                if n_event_bin else
                "No interaction importance results found. Run Step 8 (FFA Analysis) first."
            )
            return _response(200, {
                "interaction_importance": [],
                "summary": {
                    "total_features": 0,
                    "filtered_features": 0,
                    "selected_drugs": selected_drugs,
                    "n_event_bin": n_event_bin,
                    "message": msg,
                }
            })
        
        # Filter to selected drugs if provided
        filtered_df = interaction_df.copy()
        if selected_drugs:
            # Convert drug codes to feature names (item_DRUG_CODE format)
            selected_features = [f"item_{drug.upper()}" for drug in selected_drugs]
            # Filter to features that match selected drugs
            filtered_df = filtered_df[filtered_df['feature'].isin(selected_features)]
        
        # Sort by interaction importance and get top N
        importance_col = "interaction_importance" if "interaction_importance" in filtered_df.columns else "importance"
        filtered_df = filtered_df.sort_values(importance_col, ascending=False)
        top_df = filtered_df.head(top_n).copy()
        top_df['rank'] = range(1, len(top_df) + 1)
        if importance_col != "interaction_importance":
            top_df = top_df.rename(columns={importance_col: "interaction_importance"})
        
        # Format response
        interaction_importance = top_df[['feature', 'interaction_importance', 'rank']].to_dict('records')
        
        summary = {
            "total_features": len(interaction_df),
            "filtered_features": len(filtered_df),
            "selected_drugs": selected_drugs,
            "top_n_returned": len(interaction_importance),
            "n_event_bin": n_event_bin,
            "importance_source": "per_bin" if n_event_bin else "full_cohort_ffa",
        }

        return _response(200, {
            "interaction_importance": interaction_importance,
            "summary": summary
        })
    
    except Exception as e:
        import traceback
        return _response(500, {
            "error": str(e),
            "traceback": traceback.format_exc()
        })


def _interaction_feature_set_from_codes(drugs: List[str], icds: List[str], cpts: List[str]) -> set:
    """Build set of feature names for scenario data (item_X, item_icd_X, item_cpt_X, item_drug_X)."""
    out = set()
    for code in drugs:
        c = str(code).strip().upper()
        if c:
            out.add(f"item_{c}")
            out.add(f"item_drug_{c}")
    for code in icds:
        c = str(code).strip().upper()
        if c:
            out.add(f"item_{c}")
            out.add(f"item_icd_{c}")
    for code in cpts:
        c = str(code).strip().upper()
        if c:
            out.add(f"item_{c}")
            out.add(f"item_cpt_{c}")
    return out


def handle_visualizations_scenario(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    GET /visualizations/scenario?cohort=...&age_band=...[&drugs=...&icds=...&cpts=...&whatif=...]

    Same pattern as Feature Importance: load scenario_data.json from S3 and return inline.
    Lambda applies optional filters (drugs, icds, cpts, whatif) and returns chart_data
    (interaction_factors, shap_importance, whatif variants, feature_interactions) so the
    frontend can render without re-filtering. Radar plot uses top N of interaction_factors.
    S3 key: {S3_DASHBOARD_PREFIX}/visualizations/scenario/{cohort}/{age_band}/scenario_data.json (age_band with hyphen, e.g. 25-44).
    """
    try:
        params = event.get("queryStringParameters") or {}
        cohort = params.get("cohort")
        age_band = params.get("age_band")
        n_event_bin = params.get("n_event_bin") or None  # 'low'/'medium'/'high'/'extreme' or None
        model_type = params.get("model_type", "xgboost")

        if not cohort or not age_band:
            return _response(400, {"error": "cohort and age_band parameters required"})

        drugs = [x.strip() for x in (params.get("drugs") or "").split(",") if x.strip()]
        icds = [x.strip() for x in (params.get("icds") or "").split(",") if x.strip()]
        cpts = [x.strip() for x in (params.get("cpts") or "").split(",") if x.strip()]
        whatif_codes = [x.strip() for x in (params.get("whatif") or "").split(",") if x.strip()]
        selected_set = _interaction_feature_set_from_codes(drugs, icds, cpts)
        whatif_set = _interaction_feature_set_from_codes(whatif_codes, whatif_codes, whatif_codes) if whatif_codes else set()

        # S3 paths use hyphen (25-44); EC2/file paths use underscore (25_44)
        scenario_key = s3_scenario_data_key(S3_DASHBOARD_PREFIX, cohort, age_band)
        payload: Dict[str, Any] = {"scenario_data_url": _dashboard_s3_url(scenario_key), "n_event_bin": n_event_bin}

        # When n_event_bin is supplied, load per-bin FI CSV exclusively (no scenario_data.json fallback)
        if n_event_bin:
            bin_df = load_interaction_importance(cohort, age_band, model_type, bin_name=n_event_bin)
            if not bin_df.empty:
                fi_col = "interaction_importance" if "interaction_importance" in bin_df.columns else "importance"
                top_rows = bin_df.sort_values(fi_col, ascending=False)
                if selected_set:
                    top_rows = top_rows[top_rows["feature"].isin(selected_set)]
                    filtered_by_codes = True
                else:
                    filtered_by_codes = False
                interaction_factors = [
                    {"feature": r["feature"], "importance": float(r[fi_col])}
                    for _, r in top_rows.iterrows()
                ]
                chart_data: Dict[str, Any] = {
                    "interaction_factors": interaction_factors,
                    "shap_importance": interaction_factors,  # same source; SHAP not available per-bin
                    "filtered_by_codes": filtered_by_codes,
                    "importance_source": "per_bin",
                }
                if whatif_set:
                    wif_rows = bin_df[bin_df["feature"].isin(whatif_set)].sort_values(fi_col, ascending=False)
                    chart_data["interaction_factors_whatif"] = [
                        {"feature": r["feature"], "importance": float(r[fi_col])}
                        for _, r in wif_rows.iterrows()
                    ]
                    chart_data["shap_importance_whatif"] = chart_data["interaction_factors_whatif"]
                payload["chart_data"] = chart_data
            else:
                payload["message"] = (
                    f"Per-bin feature importance not available for bin='{n_event_bin}'. "
                    "Run train_per_bin() (notebook 3) and prepare_models.py first."
                )
            return _response(200, payload)

        # Full-cohort path: load scenario_data.json from S3
        try:
            obj = s3_client.get_object(Bucket=S3_DASHBOARD_BUCKET, Key=scenario_key)
            data = json.loads(obj["Body"].read().decode("utf-8"))
            payload["scenario_data"] = data
        except ClientError as e:
            if e.response.get("Error", {}).get("Code") not in ("NoSuchKey", "404", "403", "AccessDenied"):
                raise
        except (json.JSONDecodeError, TypeError):
            pass

        # Build chart_data from full-cohort scenario_data.json
        if payload.get("scenario_data"):
            raw = payload["scenario_data"]
            top = raw.get("top_interaction_factors") or raw.get("feature_importance") or []

            def row_to_factor(r: Dict[str, Any]) -> Dict[str, Any]:
                feat = r.get("feature") or ""
                return {
                    "feature": feat,
                    "importance": float(
                        r.get("interaction_score")
                        or r.get("interaction_responsibility")
                        or r.get("importance")
                        or r.get("combined_importance_norm")
                        or r.get("combined_importance")
                        or 0
                    ),
                }

            def row_to_shap(r: Dict[str, Any]) -> Dict[str, Any]:
                feat = r.get("feature") or ""
                return {
                    "feature": feat,
                    "importance": float(
                        r.get("shap_importance")
                        or r.get("shap_norm")
                        or r.get("importance")
                        or r.get("combined_importance_norm")
                        or 0
                    ),
                }

            if selected_set:
                interaction_factors = [row_to_factor(r) for r in top if (r.get("feature") or "") in selected_set]
                shap_importance = [row_to_shap(r) for r in top if (r.get("feature") or "") in selected_set]
                # Backward-compat fallback: older artifacts may use naming variants that don't
                # exactly match selected codes; show top model features rather than empty charts.
                if not interaction_factors and top:
                    interaction_factors = [row_to_factor(r) for r in top]
                    shap_importance = [row_to_shap(r) for r in top]
                    filtered_by_codes = False
                    filter_fallback = "top_model_features"
                else:
                    filtered_by_codes = True
                    filter_fallback = None
            else:
                interaction_factors = [row_to_factor(r) for r in top]
                shap_importance = [row_to_shap(r) for r in top]
                filtered_by_codes = False
                filter_fallback = None

            interaction_factors_whatif = [row_to_factor(r) for r in top if (r.get("feature") or "") in whatif_set] if whatif_set else []
            shap_importance_whatif = [row_to_shap(r) for r in top if (r.get("feature") or "") in whatif_set] if whatif_set else []

            chart_data = {
                "interaction_factors": interaction_factors,
                "shap_importance": shap_importance,
                "filtered_by_codes": filtered_by_codes,
                "importance_source": "full_cohort_ffa",
            }
            if filter_fallback:
                chart_data["filter_fallback"] = filter_fallback
            if interaction_factors_whatif:
                chart_data["interaction_factors_whatif"] = interaction_factors_whatif
            if shap_importance_whatif:
                chart_data["shap_importance_whatif"] = shap_importance_whatif
            if raw.get("feature_interactions"):
                chart_data["feature_interactions"] = raw["feature_interactions"]
            payload["chart_data"] = chart_data

        return _response(200, payload)
    except Exception as e:
        return _response(500, {"error": str(e)})


# Max size (bytes) for inline trajectory_overview_plot.json to avoid Lambda timeout/OOM (Plotly JSON can be large)
DTW_TRAJECTORY_PLOT_MAX_INLINE_BYTES = 2 * 1024 * 1024  # 2 MB


def handle_visualizations_dtw(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    GET /visualizations/dtw?cohort=...&age_band=...[&n_event_bin=low|medium|high|extreme]

    When n_event_bin is supplied, chart fields with *_by_density variants are replaced
    with the per-bin slice.  Falls back to full-cohort data (labelled accordingly) when
    the bin-specific slice is absent.
    """
    try:
        params = event.get("queryStringParameters") or {}
        cohort = params.get("cohort")
        age_band = params.get("age_band")
        n_event_bin = params.get("n_event_bin") or None

        if not cohort or not age_band:
            return _response(400, {"error": "cohort and age_band parameters required"})

        age_band_fname = age_band.replace("-", "_")
        prefix: Optional[str] = None
        chart_data_key: Optional[str] = None
        sequence_heatmap_key: Optional[str] = None

        # Prefer manifest: use full path from s3_path + static_files (single source of truth)
        manifest = _get_dashboard_manifest()
        if manifest:
            visual_objects = manifest.get("visual_objects") or []
            dtw_entry = next(
                (o for o in visual_objects if (o.get("dashboard_tab") or "") == "DTW Trajectories"),
                None,
            )
            if dtw_entry and dtw_entry.get("s3_path") and dtw_entry.get("static_files"):
                s3_path = (dtw_entry["s3_path"] or "").rstrip("/")
                prefix = s3_path.replace("{cohort}", cohort).replace("{age_band}", age_band)
                static_files = dtw_entry["static_files"]
                if len(static_files) >= 2:
                    chart_data_key = f"{prefix}/{static_files[0]}"
                    sequence_heatmap_key = f"{prefix}/{static_files[1]}"

        if prefix is None:
            prefix = f"{S3_DASHBOARD_PREFIX.strip('/')}/visualizations/dtw/{cohort}/{age_band}"
        if chart_data_key is None:
            chart_data_key = f"{prefix}/chart_data.json"
        if sequence_heatmap_key is None:
            sequence_heatmap_key = f"{prefix}/sequence_heatmap.json"

        plots_key = f"{prefix}/plots"
        bucket = S3_DASHBOARD_BUCKET

        payload = {
            "chart_data_url": _dashboard_s3_url(chart_data_key),
            "sequence_heatmap_url": _dashboard_s3_url(sequence_heatmap_key),
            "metrics": {},
            "n_event_bin": n_event_bin,
        }

        # Prefer JSON: load chart_data and sequence_heatmap from S3 (full path from manifest when available)
        for key, s3_key in [
            ("chart_data", chart_data_key),
            ("sequence_heatmap", sequence_heatmap_key),
        ]:
            try:
                obj = s3_client.get_object(Bucket=bucket, Key=s3_key)
                data = json.loads(obj["Body"].read().decode("utf-8"))
                payload[key] = data
            except ClientError as e:
                if e.response.get("Error", {}).get("Code") not in ("NoSuchKey", "404", "403", "AccessDenied"):
                    raise
            except (json.JSONDecodeError, TypeError):
                pass

        # Always return consumable JSON so frontend workflow does not break (missing or empty S3 = payload with message)
        _empty_chart_msg = "No DTW chart data for this cohort/age band. Run create_dtw_visuals (notebook 4) and promote to S3 (notebook 5 Step 6)."
        if payload.get("chart_data") is None:
            payload["chart_data"] = {"message": _empty_chart_msg, "empty": True}
        elif isinstance(payload["chart_data"], dict) and not any(payload["chart_data"].get(k) for k in ("routine_comparison", "high_risk_trajectories", "times_between_sequences", "target_pathway_patterns")):
            payload["chart_data"].setdefault("message", "DTW chart data is empty for this cohort/age band.")
            payload["chart_data"].setdefault("empty", True)
        if payload.get("sequence_heatmap") is None:
            payload["sequence_heatmap"] = {
                "drug": {"codes": [], "positions": [], "counts": []},
                "icd": {"codes": [], "positions": [], "counts": []},
                "cpt": {"codes": [], "positions": [], "counts": []},
                "message": "No sequence heatmap for this cohort/age band. Run create_dtw_visuals and promote to S3.",
                "empty": True,
            }
        elif isinstance(payload.get("sequence_heatmap"), dict):
            for slice_key in ("drug", "icd", "cpt"):
                if not payload["sequence_heatmap"].get(slice_key):
                    payload["sequence_heatmap"][slice_key] = {"codes": [], "positions": [], "counts": []}
            if not (payload["sequence_heatmap"].get("drug", {}).get("codes") or payload["sequence_heatmap"].get("empty")):
                payload["sequence_heatmap"].setdefault("message", "Sequence heatmap is empty for this cohort/age band.")
                payload["sequence_heatmap"].setdefault("empty", True)

        # When n_event_bin requested: overlay per-bin slices from *_by_density fields
        if n_event_bin and payload.get("chart_data") and isinstance(payload["chart_data"], dict):
            cd = payload["chart_data"]
            _BY_DENSITY_FIELDS = (
                "routine_comparison",
                "routine_comparison_counts",
                "high_risk_trajectories",
            )
            bin_found = any(
                cd.get(f"{field}_by_density", {}).get(n_event_bin)
                for field in _BY_DENSITY_FIELDS
            )
            bin_cd: Dict[str, Any] = {}
            _resolved_bin = n_event_bin  # tracks which bin was actually used per-field
            for field in _BY_DENSITY_FIELDS:
                by_density = cd.get(f"{field}_by_density") or {}
                if n_event_bin in by_density:
                    bin_cd[field] = by_density[n_event_bin]
                    bin_cd[f"{field}_data_scope"] = "per_bin"
                else:
                    # Try nearest available bin key in this field's by_density dict
                    nearest_for_field = _nearest_available_bin_dict(by_density, n_event_bin)
                    if nearest_for_field:
                        bin_cd[field] = by_density[nearest_for_field]
                        bin_cd[f"{field}_data_scope"] = "nearest_bin_fallback"
                        bin_cd[f"{field}_nearest_bin"] = nearest_for_field
                    elif cd.get(field):
                        bin_cd[field] = cd[field]
                        bin_cd[f"{field}_data_scope"] = "full_cohort_fallback"
            # Fields without per-bin variants — carry over from full cohort
            for passthru in ("times_between_sequences", "target_pathway_patterns",
                             "time_to_target_sequences", "event_density_bins"):
                if cd.get(passthru):
                    bin_cd[passthru] = cd[passthru]
                    bin_cd[f"{passthru}_data_scope"] = "full_cohort"
            # Determine overall data_scope for summary
            scopes = {bin_cd.get(f"{f}_data_scope") for f in _BY_DENSITY_FIELDS if f in bin_cd}
            if "per_bin" in scopes:
                overall_scope = "per_bin"
            elif "nearest_bin_fallback" in scopes:
                overall_scope = "nearest_bin_fallback"
            else:
                overall_scope = "full_cohort_fallback"
            bin_cd["data_scope"] = overall_scope
            bin_cd["requested_bin"] = n_event_bin
            if overall_scope != "per_bin":
                bin_cd["message"] = (
                    f"No per-bin DTW data for density='{n_event_bin}'. "
                    + (f"Showing nearest available bin data where available."
                       if overall_scope == "nearest_bin_fallback"
                       else "Showing full-cohort data. Run create_dtw_visuals (notebook 4) to generate per-bin visuals.")
                )
            payload["chart_data"] = bin_cd

        # Optional: simple metrics from chart_data for Trajectory Metrics panel
        if payload.get("chart_data"):
            cd = payload["chart_data"]
            metrics = {}
            if cd.get("routine_comparison") and cd["routine_comparison"].get("y"):
                metrics["routine_comparison_series"] = len(cd["routine_comparison"]["y"])
            if cd.get("times_between_sequences") and cd["times_between_sequences"].get("y"):
                metrics["times_between_categories"] = len(cd["times_between_sequences"]["y"])
            if cd.get("target_pathway_patterns") and cd["target_pathway_patterns"].get("y"):
                metrics["target_pathway_codes"] = len(cd["target_pathway_patterns"]["y"])
            if metrics:
                payload["metrics"] = metrics

        # Trajectory overview: prefer Plotly figure JSON (skip if too large to avoid Lambda timeout/OOM/502)
        trajectory_plot_key = f"{plots_key}/trajectory_overview_plot.json"
        try:
            head = s3_client.head_object(Bucket=bucket, Key=trajectory_plot_key)
            size = head.get("ContentLength", 0) or 0
            if size <= DTW_TRAJECTORY_PLOT_MAX_INLINE_BYTES:
                obj = s3_client.get_object(Bucket=bucket, Key=trajectory_plot_key)
                payload["trajectory_overview_plot"] = json.loads(obj["Body"].read().decode("utf-8"))
        except ClientError as e:
            if e.response.get("Error", {}).get("Code") not in ("NoSuchKey", "404", "403", "AccessDenied"):
                raise
        except (json.JSONDecodeError, TypeError):
            pass

        # Fallback: image/HTML URLs when Plotly JSON not present — must check BEFORE setting empty-message
        # sentinel, otherwise the guard condition below can never be True.
        overview_key = f"{plots_key}/dtw_trajectory_analysis_{cohort}_{age_band_fname}.png"
        sample_key = f"{plots_key}/dtw_sample_trajectories_{cohort}_{age_band_fname}.png"
        # Interactive HTML: pipeline writes 1d/3d; try interactive, then 1d, then 3d
        overview_html_key = None
        for suffix in ("interactive", "1d", "3d"):
            candidate = f"{plots_key}/dtw_trajectory_cluster_{suffix}_{cohort}_{age_band_fname}.html"
            if _s3_object_exists(bucket, candidate):
                overview_html_key = candidate
                break
        if not payload.get("trajectory_overview_plot"):
            if _s3_object_exists(bucket, overview_key):
                payload["overview_image"] = _dashboard_s3_url(overview_key)
            if _s3_object_exists(bucket, sample_key):
                payload["sample_trajectories_image"] = _dashboard_s3_url(sample_key)
        if overview_html_key:
            payload["overview_interactive"] = _dashboard_s3_url(overview_html_key)

        # Always return a consumable trajectory_overview_plot value so the frontend never gets undefined
        if not payload.get("trajectory_overview_plot"):
            payload["trajectory_overview_plot"] = {
                "message": "No trajectory overview for this cohort/age band. Run create_dtw_visuals (notebook 4) and sync to S3 (notebook 5 Step 6).",
                "empty": True,
            }

        return _response(200, payload)
    except Exception as e:
        return _response(500, {"error": str(e)})


def handle_fpgrowth_network_html_proxy(event: Dict[str, Any]) -> Dict[str, Any]:
    """GET /visualizations/fpgrowth/network_html?cohort=...&age_band=...[&n_event_bin=low|medium|high|extreme]
    Fetches combined_rules_network.html from S3 and returns it with Content-Type: text/html
    so the dashboard iframe renders it instead of triggering a download.
    When n_event_bin is supplied, the per-bin density/{bin}/plots/ path is tried first.
    """
    try:
        params = event.get("queryStringParameters") or {}
        cohort = params.get("cohort")
        age_band = params.get("age_band")
        n_event_bin = params.get("n_event_bin") or None
        if not cohort or not age_band:
            return _response(400, {"error": "cohort and age_band parameters required"})
        age_band_fname = age_band.replace("-", "_")
        prefix = f"{S3_DASHBOARD_PREFIX.strip('/')}/visualizations/fpgrowth"
        fname = f"{cohort}_{age_band_fname}_combined_rules_network.html"

        # Per-bin path first when n_event_bin supplied
        if n_event_bin:
            bin_key = f"{prefix}/{cohort}/{age_band}/density/{n_event_bin}/plots/{fname}"
            if _s3_object_exists(S3_DASHBOARD_BUCKET, bin_key):
                obj = s3_client.get_object(Bucket=S3_DASHBOARD_BUCKET, Key=bin_key)
                return _response_html(200, obj["Body"].read().decode("utf-8"))
            # Fall through to full-cohort with a visible notice in the HTML
            full_key = f"{prefix}/{cohort}/{age_band}/plots/{fname}"
            try:
                obj = s3_client.get_object(Bucket=S3_DASHBOARD_BUCKET, Key=full_key)
                html = obj["Body"].read().decode("utf-8")
                notice = (
                    f'<div style="background:#fff3cd;padding:8px;font-family:sans-serif;font-size:13px">'
                    f'Showing full-cohort network (no per-bin data for density=\'{n_event_bin}\'). '
                    f'Run cohort_fpgrowth.py with per-bin output enabled.</div>'
                )
                html = html.replace("<body>", f"<body>{notice}", 1)
                return _response_html(200, html)
            except ClientError as e:
                if e.response.get("Error", {}).get("Code") in ("NoSuchKey", "404", "AccessDenied"):
                    return _response_html(404, f"<!DOCTYPE html><html><body><p>Network visualization not found for bin='{n_event_bin}' or full-cohort. Run the FP-Growth pipeline.</p></body></html>")
                raise

        full_key = f"{prefix}/{cohort}/{age_band}/plots/{fname}"
        obj = s3_client.get_object(Bucket=S3_DASHBOARD_BUCKET, Key=full_key)
        return _response_html(200, obj["Body"].read().decode("utf-8"))
    except ClientError as e:
        if e.response.get("Error", {}).get("Code") in ("NoSuchKey", "404", "AccessDenied"):
            checked = f"s3://{S3_DASHBOARD_BUCKET}/{full_key}"
            return _response_html(404, f"<!DOCTYPE html><html><body><p>Network visualization not found. Run the FP-Growth pipeline to build it.</p><p>Checked: {checked}</p></body></html>")
        raise
    except Exception as e:
        return _response(500, {"error": str(e)})


def handle_visualizations_fpgrowth(event: Dict[str, Any]) -> Dict[str, Any]:
    """GET /visualizations/fpgrowth?cohort=...&age_band=...[&n_event_bin=low|medium|high|extreme]

    When n_event_bin is supplied, per-bin output files under
    visualizations/fpgrowth/{cohort}/{age_band}/density/{bin_name}/plots/ are tried first.
    Falls back to full-cohort combined outputs with data_scope label when absent.
    """
    try:
        params = event.get("queryStringParameters") or {}
        cohort = params.get("cohort")
        age_band = params.get("age_band")
        n_event_bin = params.get("n_event_bin") or None
        item_type = "drug_name"

        if not cohort or not age_band:
            return _response(400, {"error": "cohort and age_band parameters required"})

        prefix = f"{S3_DASHBOARD_PREFIX.strip('/')}/visualizations/fpgrowth"
        age_band_fname = age_band.replace("-", "_")

        def _build_fpgrowth_payload(base_key: str, data_scope: str) -> Dict[str, Any]:
            network_combined_key = f"{base_key}/{cohort}_{age_band_fname}_combined_rules_network.html"
            itemsets_key = f"{base_key}/{cohort}_{age_band_fname}_{item_type}_combined_top_itemsets.png"
            network_html_key = f"{base_key}/{cohort}_{age_band_fname}_{item_type}_target_rules_network.html"
            network_png_key = f"{base_key}/{cohort}_{age_band_fname}_{item_type}_target_rules_network.png"
            itemsets_interactive_key = f"{base_key}/{cohort}_{age_band_fname}_{item_type}_itemsets_interactive.html"
            network_interactive_key = f"{base_key}/{cohort}_{age_band_fname}_{item_type}_network_interactive.html"
            p: Dict[str, Any] = {
                "network_combined_html": _dashboard_s3_url(network_combined_key),
                "itemsets_image": _dashboard_s3_url(itemsets_key),
                "support_image": _dashboard_s3_url(itemsets_key),
                "network_html": _dashboard_s3_url(network_html_key),
                "network_png": _dashboard_s3_url(network_png_key),
                "itemsets_interactive": _dashboard_s3_url(itemsets_interactive_key),
                "network_interactive": _dashboard_s3_url(network_interactive_key),
                "data_scope": data_scope,
                "n_event_bin": n_event_bin,
            }
            if data_scope == "full_cohort_fallback":
                p["message"] = (
                    f"No per-bin FP-Growth data for density='{n_event_bin}'. "
                    "Showing full-cohort combined output. Run cohort_fpgrowth.py with per-bin output enabled."
                )
            # Per-bin: read itemsets from the bin-specific path; full-cohort: use age_band root
            if data_scope == "per_bin":
                data_key = f"{base_key}/{item_type}_itemsets.json"
            else:
                data_key = f"{prefix}/{cohort}/{age_band}/{item_type}_itemsets.json"
            try:
                obj = s3_client.get_object(Bucket=S3_DASHBOARD_BUCKET, Key=data_key)
                p["itemsets_data"] = json.loads(obj["Body"].read().decode("utf-8"))
            except (ClientError, json.JSONDecodeError, TypeError):
                pass
            return p

        # When n_event_bin specified: try per-bin path first
        if n_event_bin:
            bin_base_key = f"{prefix}/{cohort}/{age_band}/density/{n_event_bin}/plots"
            bin_empty_key = f"{bin_base_key}/empty_state.json"
            # Probe for per-bin itemsets JSON (pipeline writes {item_type}_itemsets.json here)
            bin_itemsets_key = f"{bin_base_key}/{item_type}_itemsets.json"
            if _s3_object_exists(S3_DASHBOARD_BUCKET, bin_itemsets_key):
                return _response(200, _build_fpgrowth_payload(bin_base_key, "per_bin"))
            # Backward-compat: pre-fix pipeline saved JSON directly under density/{bin}/ (no plots/)
            legacy_itemsets_key = f"{prefix}/{cohort}/{age_band}/density/{n_event_bin}/{item_type}_itemsets.json"
            if _s3_object_exists(S3_DASHBOARD_BUCKET, legacy_itemsets_key):
                return _response(200, _build_fpgrowth_payload(bin_base_key, "per_bin"))
            # Requested bin has no itemsets (missing or empty_state). Try nearest bin with
            # real data before falling back — empty_state means insufficient transactions,
            # but a nearby bin may have usable patterns.
            full_base_key = f"{prefix}/{cohort}/{age_band}/plots"
            nearest = _nearest_available_bin_s3(
                S3_DASHBOARD_BUCKET, n_event_bin,
                lambda b: (
                    f"{prefix}/{cohort}/{age_band}/density/{b}/plots/{item_type}_itemsets.json"
                    if _s3_object_exists(S3_DASHBOARD_BUCKET, f"{prefix}/{cohort}/{age_band}/density/{b}/plots/{item_type}_itemsets.json")
                    else f"{prefix}/{cohort}/{age_band}/density/{b}/{item_type}_itemsets.json"
                ),
            )
            if nearest:
                nearest_base_key = f"{prefix}/{cohort}/{age_band}/density/{nearest}/plots"
                p = _build_fpgrowth_payload(nearest_base_key, "nearest_bin_fallback")
                p["nearest_bin"] = nearest
                p["message"] = (
                    f"No per-bin FP-Growth data for density='{n_event_bin}'. "
                    f"Showing nearest available bin '{nearest}'."
                )
                return _response(200, p)
            # No nearest bin — return empty_state for requested bin if it exists
            if _s3_object_exists(S3_DASHBOARD_BUCKET, bin_empty_key):
                try:
                    obj = s3_client.get_object(Bucket=S3_DASHBOARD_BUCKET, Key=bin_empty_key)
                    return _response(200, json.loads(obj["Body"].read().decode("utf-8")))
                except (ClientError, json.JSONDecodeError):
                    pass
            return _response(200, _build_fpgrowth_payload(full_base_key, "full_cohort_fallback"))

        # No bin requested: serve combined full-cohort output
        full_base_key = f"{prefix}/{cohort}/{age_band}/plots"
        empty_state_key = f"{full_base_key}/empty_state.json"
        try:
            obj = s3_client.get_object(Bucket=S3_DASHBOARD_BUCKET, Key=empty_state_key)
            return _response(200, json.loads(obj["Body"].read().decode("utf-8")))
        except ClientError as e:
            if e.response.get("Error", {}).get("Code") not in ("NoSuchKey", "404", "403", "AccessDenied"):
                raise
        except (json.JSONDecodeError, KeyError):
            pass
        return _response(200, _build_fpgrowth_payload(full_base_key, "full_cohort"))
    except Exception as e:
        return _response(500, {"error": str(e)})


def handle_visualizations_feature_importance(event: Dict[str, Any]) -> Dict[str, Any]:
    """GET /visualizations/feature_importance?cohort=...
    Cohort: opioid_ed, non_opioid_ed, or combined. Always returns aggregated heatmap (no model filter).
    Prefer JSON over PNG: returns heatmap_data when available; frontend falls back to heatmap_url (PNG).
    S3: {prefix}/feature_importance/{cohort}/aggregated_fi_heatmap.json|png; combined uses combined/ or combined_cohorts_*.
    """
    try:
        params = event.get("queryStringParameters") or {}
        cohort = (params.get("cohort") or "").strip()
        if not cohort:
            return _response(400, {"error": "cohort parameter required"})
        if cohort not in ("opioid_ed", "non_opioid_ed", "combined"):
            cohort = "opioid_ed"
        prefix = f"{S3_DASHBOARD_PREFIX.strip('/')}/visualizations/feature_importance"
        combined_key = f"{prefix}/combined_cohorts_feature_importance_heatmap.png"
        if cohort == "combined":
            heatmap_key = combined_key
            json_key = f"{prefix}/combined/aggregated_fi_heatmap.json"
        else:
            heatmap_key = f"{prefix}/{cohort}/aggregated_fi_heatmap.png"
            json_key = f"{prefix}/{cohort}/aggregated_fi_heatmap.json"

        payload = {
            "heatmap_url": _dashboard_s3_url(heatmap_key),
            "combined_url": _dashboard_s3_url(combined_key),
        }

        try:
            obj = s3_client.get_object(Bucket=S3_DASHBOARD_BUCKET, Key=json_key)
            body = obj.get("Body").read().decode("utf-8")
            data = json.loads(body)
            if isinstance(data, dict) and "cohort" not in data:
                data["cohort"] = cohort
            payload["heatmap_data"] = data
        except (ClientError, json.JSONDecodeError, TypeError):
            pass

        return _response(200, payload)
    except Exception as e:
        return _response(500, {"error": str(e)})


# TODO: Patient-level BupaR visuals (trace explorer, process matrix, frequency map filtered by
# cohort/age_band/patient subset) require on-demand R execution. Implement when R is available
# in Lambda (e.g. custom runtime/layer or separate R service) and add POST /visualizations/bupar/patient-level.


# Allowed BupaR HTML visual names for proxy (must match S3 object suffix: base_<visual>.html)
def handle_visualizations_bupar_activity_frequency(event: Dict[str, Any]) -> Dict[str, Any]:
    """GET /visualizations/bupar/activity_frequency?cohort=...&age_band=...[&n_event_bin=...]

    When n_event_bin is supplied, per-bin JSON under
    visualizations/bupar/{cohort}/{age_band}/density/{bin_name}/plots/ is tried first.
    Falls back to full-cohort data with data_scope label when absent.
    """
    try:
        params = event.get("queryStringParameters") or {}
        cohort = params.get("cohort")
        age_band = params.get("age_band")
        n_event_bin = params.get("n_event_bin") or None
        if not cohort or not age_band:
            return _response(400, {"error": "cohort and age_band parameters required"})
        age_band_fname = age_band.replace("-", "_")
        base = f"{cohort}_{age_band_fname}"
        bucket = S3_DASHBOARD_BUCKET
        vis_prefix = f"{S3_DASHBOARD_PREFIX.strip('/')}/visualizations/bupar"

        def _load_activity_freq_from_prefix(plots_prefix: str, scope: str) -> Dict[str, Any]:
            keys = {
                "overall": f"{plots_prefix}/{base}_activity_frequency.json",
                "pre_target": f"{plots_prefix}/{base}_pre_target_activity_frequency.json",
                "post_target": f"{plots_prefix}/{base}_post_target_activity_frequency.json",
            }
            result: Dict[str, Any] = {"data_scope": scope, "n_event_bin": n_event_bin}
            for name, key in keys.items():
                try:
                    obj = s3_client.get_object(Bucket=bucket, Key=key)
                    result[name] = json.loads(obj["Body"].read().decode("utf-8"))
                except ClientError as e:
                    code = e.response.get("Error", {}).get("Code")
                    result[name] = None if code in ("NoSuchKey", "404", "403", "AccessDenied") else (_ for _ in ()).throw(e)
            return result

        full_prefix = f"{vis_prefix}/{cohort}/{age_band}/plots"
        if n_event_bin:
            bin_prefix = f"{vis_prefix}/{cohort}/{age_band}/density/{n_event_bin}/plots"
            # Probe for per-bin overall activity frequency
            probe_key = f"{bin_prefix}/{base}_activity_frequency.json"
            if _s3_object_exists(bucket, probe_key):
                return _response(200, _load_activity_freq_from_prefix(bin_prefix, "per_bin"))
            # Try nearest available bin before falling back to full-cohort
            nearest = _nearest_available_bin_s3(
                bucket, n_event_bin,
                lambda b: f"{vis_prefix}/{cohort}/{age_band}/density/{b}/plots/{base}_activity_frequency.json",
            )
            if nearest:
                nearest_prefix = f"{vis_prefix}/{cohort}/{age_band}/density/{nearest}/plots"
                result = _load_activity_freq_from_prefix(nearest_prefix, "nearest_bin_fallback")
                result["nearest_bin"] = nearest
                result["message"] = (
                    f"No per-bin BupaR activity data for density='{n_event_bin}'. "
                    f"Showing nearest available bin '{nearest}'."
                )
                return _response(200, result)
            result = _load_activity_freq_from_prefix(full_prefix, "full_cohort_fallback")
            result["message"] = (
                f"No per-bin BupaR activity data for density='{n_event_bin}'. "
                "Showing full-cohort data. Run create_bupar_visuals.py with per-bin output enabled."
            )
            return _response(200, result)

        return _response(200, _load_activity_freq_from_prefix(full_prefix, "full_cohort"))
    except Exception as e:
        return _response(500, {"error": str(e)})


def handle_visualizations_bupar(event: Dict[str, Any]) -> Dict[str, Any]:
    """GET /visualizations/bupar?cohort=...&age_band=...[&n_event_bin=low|medium|high|extreme]

    When n_event_bin is supplied, per-bin plot objects under
    visualizations/bupar/{cohort}/{age_band}/density/{bin_name}/plots/ are tried first.
    Falls back to full-cohort objects with data_scope label when absent.
    """
    try:
        params = event.get("queryStringParameters") or {}
        cohort = params.get("cohort")
        age_band = params.get("age_band")
        n_event_bin = params.get("n_event_bin") or None

        if not cohort or not age_band:
            return _response(400, {"error": "cohort and age_band parameters required"})

        age_band_fname = age_band.replace("-", "_")
        base = f"{cohort}_{age_band_fname}"
        pre_suffix = "pre_f1120" if cohort == "opioid_ed" else "pre_hcg"

        payload: Dict[str, Any] = {}
        base_key: Optional[str] = None

        # Prefer manifest: use full path from s3_path + static_files (single source of truth)
        manifest = _get_dashboard_manifest()
        if manifest:
            visual_objects = manifest.get("visual_objects") or []
            bupar_entry = next(
                (o for o in visual_objects if (o.get("dashboard_tab") or "") == "BupaR Process Mining"),
                None,
            )
            if bupar_entry and bupar_entry.get("s3_path") and bupar_entry.get("static_files"):
                s3_path = (bupar_entry["s3_path"] or "").rstrip("/")
                base_key = s3_path.replace("{cohort}", cohort).replace("{age_band}", age_band)
                static_files = [f.replace("{base}", base) for f in bupar_entry["static_files"]]
                # Payload key by manifest static_files index (order must match manifest)
                # 6=overall_activity_frequency, 7=activity_sequence_top, 8=process_matrix_drug_drug,
                # 9=pre_f1120_activity_frequency, 10=pre_hcg_activity_frequency,
                # 11=trace_explorer_pre_f1120, 12=trace_explorer_pre_hcg
                image_indices = [
                    ("activity_frequency_image", 6),
                    ("sequence_image", 7),
                    ("process_matrix_drug_drug", 8),
                    ("pre_target_frequency_image", 9 if cohort == "opioid_ed" else 10),
                    ("trace_explorer_pre_image", 11 if cohort == "opioid_ed" else 12),
                ]
                for payload_key, idx in image_indices:
                    if idx < len(static_files):
                        s3_key = f"{base_key}/{static_files[idx]}"
                        if _s3_object_exists(S3_DASHBOARD_BUCKET, s3_key):
                            payload[payload_key] = _dashboard_s3_url(s3_key)
                # JSON from manifest static_files indices 3, 4, 5
                for payload_key, idx in [("trace_explorer_plot", 3), ("process_matrix_drug_drug", 4), ("activity_sequence_top", 5)]:
                    if idx < len(static_files):
                        s3_key = f"{base_key}/{static_files[idx]}"
                        try:
                            obj = s3_client.get_object(Bucket=S3_DASHBOARD_BUCKET, Key=s3_key)
                            payload[payload_key] = json.loads(obj["Body"].read().decode("utf-8"))
                        except ClientError as e:
                            if e.response.get("Error", {}).get("Code") not in ("NoSuchKey", "404", "403", "AccessDenied"):
                                raise
                        except (json.JSONDecodeError, TypeError):
                            pass

        vis_prefix_bupar = f"{S3_DASHBOARD_PREFIX.strip('/')}/visualizations/bupar"

        # Fallback when manifest missing or no BupaR entry: hardcoded paths (must match manifest layout)
        if base_key is None:
            base_key = f"{vis_prefix_bupar}/{cohort}/{age_band}/plots"
            candidates: List[Tuple[str, str]] = [
                ("activity_frequency_image", f"{base_key}/{base}_overall_activity_frequency.png"),
                ("pre_target_frequency_image", f"{base_key}/{base}_{pre_suffix}_activity_frequency.png"),
                ("sequence_image", f"{base_key}/{base}_activity_sequence_top.png"),
                ("trace_explorer_pre_image", f"{base_key}/{base}_trace_explorer_{pre_suffix}.png"),
                ("process_matrix_drug_drug", f"{base_key}/{base}_process_matrix_drug_drug.png"),
            ]
            for payload_key, s3_key in candidates:
                if _s3_object_exists(S3_DASHBOARD_BUCKET, s3_key):
                    payload[payload_key] = _dashboard_s3_url(s3_key)
            for key, json_file in [
                ("trace_explorer_plot", f"{base}_trace_explorer_plot.json"),
                ("process_matrix_drug_drug", f"{base}_process_matrix_drug_drug.json"),
                ("activity_sequence_top", f"{base}_activity_sequence_top.json"),
            ]:
                s3_key = f"{base_key}/{json_file}"
                try:
                    obj = s3_client.get_object(Bucket=S3_DASHBOARD_BUCKET, Key=s3_key)
                    payload[key] = json.loads(obj["Body"].read().decode("utf-8"))
                except ClientError as e:
                    if e.response.get("Error", {}).get("Code") not in ("NoSuchKey", "404", "403", "AccessDenied"):
                        raise
                except (json.JSONDecodeError, TypeError):
                    pass

        # Per-bin override: when n_event_bin supplied, try density/{bin}/plots/ first
        payload["n_event_bin"] = n_event_bin
        if n_event_bin:
            bin_base_key = f"{vis_prefix_bupar}/{cohort}/{age_band}/density/{n_event_bin}/plots"
            bin_probe = f"{bin_base_key}/{base}_overall_activity_frequency.png"
            if _s3_object_exists(S3_DASHBOARD_BUCKET, bin_probe):
                # Reload payload from per-bin prefix
                bin_payload: Dict[str, Any] = {"n_event_bin": n_event_bin, "data_scope": "per_bin"}
                bin_candidates: List[Tuple[str, str]] = [
                    ("activity_frequency_image", f"{bin_base_key}/{base}_overall_activity_frequency.png"),
                    ("pre_target_frequency_image", f"{bin_base_key}/{base}_{pre_suffix}_activity_frequency.png"),
                    ("sequence_image", f"{bin_base_key}/{base}_activity_sequence_top.png"),
                    ("trace_explorer_pre_image", f"{bin_base_key}/{base}_trace_explorer_{pre_suffix}.png"),
                    ("process_matrix_drug_drug", f"{bin_base_key}/{base}_process_matrix_drug_drug.png"),
                ]
                for pk, sk in bin_candidates:
                    if _s3_object_exists(S3_DASHBOARD_BUCKET, sk):
                        bin_payload[pk] = _dashboard_s3_url(sk)
                for key, json_file in [
                    ("trace_explorer_plot", f"{base}_trace_explorer_plot.json"),
                    ("process_matrix_drug_drug", f"{base}_process_matrix_drug_drug.json"),
                    ("activity_sequence_top", f"{base}_activity_sequence_top.json"),
                ]:
                    try:
                        obj = s3_client.get_object(Bucket=S3_DASHBOARD_BUCKET, Key=f"{bin_base_key}/{json_file}")
                        bin_payload[key] = json.loads(obj["Body"].read().decode("utf-8"))
                    except (ClientError, json.JSONDecodeError, TypeError):
                        pass
                return _response(200, bin_payload)
            # Try nearest available bin before falling back to full-cohort
            nearest = _nearest_available_bin_s3(
                S3_DASHBOARD_BUCKET, n_event_bin,
                lambda b: f"{vis_prefix_bupar}/{cohort}/{age_band}/density/{b}/plots/{base}_overall_activity_frequency.png",
            )
            if nearest:
                nearest_base_key = f"{vis_prefix_bupar}/{cohort}/{age_band}/density/{nearest}/plots"
                nearest_probe = f"{nearest_base_key}/{base}_overall_activity_frequency.png"
                if _s3_object_exists(S3_DASHBOARD_BUCKET, nearest_probe):
                    nb_payload: Dict[str, Any] = {"n_event_bin": n_event_bin, "data_scope": "nearest_bin_fallback", "nearest_bin": nearest}
                    nb_payload["message"] = (
                        f"No per-bin BupaR visuals for density='{n_event_bin}'. "
                        f"Showing nearest available bin '{nearest}'."
                    )
                    nb_candidates: List[Tuple[str, str]] = [
                        ("activity_frequency_image", f"{nearest_base_key}/{base}_overall_activity_frequency.png"),
                        ("pre_target_frequency_image", f"{nearest_base_key}/{base}_{pre_suffix}_activity_frequency.png"),
                        ("sequence_image", f"{nearest_base_key}/{base}_activity_sequence_top.png"),
                        ("trace_explorer_pre_image", f"{nearest_base_key}/{base}_trace_explorer_{pre_suffix}.png"),
                        ("process_matrix_drug_drug", f"{nearest_base_key}/{base}_process_matrix_drug_drug.png"),
                    ]
                    for pk, sk in nb_candidates:
                        if _s3_object_exists(S3_DASHBOARD_BUCKET, sk):
                            nb_payload[pk] = _dashboard_s3_url(sk)
                    for key, json_file in [
                        ("trace_explorer_plot", f"{base}_trace_explorer_plot.json"),
                        ("process_matrix_drug_drug", f"{base}_process_matrix_drug_drug.json"),
                        ("activity_sequence_top", f"{base}_activity_sequence_top.json"),
                    ]:
                        try:
                            obj = s3_client.get_object(Bucket=S3_DASHBOARD_BUCKET, Key=f"{nearest_base_key}/{json_file}")
                            nb_payload[key] = json.loads(obj["Body"].read().decode("utf-8"))
                        except (ClientError, json.JSONDecodeError, TypeError):
                            pass
                    return _response(200, nb_payload)
            # Per-bin not found — keep full-cohort payload with fallback label
            payload["data_scope"] = "full_cohort_fallback"
            payload["message"] = (
                f"No per-bin BupaR visuals for density='{n_event_bin}'. "
                "Showing full-cohort data. Run create_bupar_visuals.py with per-bin output enabled."
            )
        else:
            payload["data_scope"] = "full_cohort"

        return _response(200, payload)
    except Exception as e:
        return _response(500, {"error": str(e)})


def handle_visualizations_cohort_pgx(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    GET /visualizations/cohort_pgx?cohort=...&age_band=...[&n_event_bin=low|medium|high|extreme]

    Returns network_topology_url only when the S3 object exists (HEAD check). Built by
    Cohort PGx pipeline (fetch_vip_reports + build_network_topology); expected key:
    {S3_DASHBOARD_PREFIX}/visualizations/cohort_pgx/networks/{cohort}/{age_band}/network_topology.html (age_band with hyphen).
    When n_event_bin supplied, tries density/{bin}/network_topology.html first; falls back to
    full-cohort with data_scope label when absent.
    Sync 10_risk_dashboard/visualizations/cohort_pgx/ to dashboard S3 after building (use hyphen in S3 path).
    """
    try:
        params = event.get("queryStringParameters") or {}
        cohort = params.get("cohort")
        age_band = params.get("age_band")
        n_event_bin = params.get("n_event_bin") or None

        if not cohort or not age_band:
            return _response(400, {"error": "cohort and age_band parameters required"})

        # S3 paths use hyphen (25-44); EC2 dirs use underscore (25_44)
        prefix   = f"{S3_DASHBOARD_PREFIX.strip('/')}/visualizations/cohort_pgx"
        net_base = f"{prefix}/networks/{cohort}/{age_band}"
        full_html_key = f"{net_base}/network_topology.html"
        payload: Dict[str, Any] = {"n_event_bin": n_event_bin}

        # --- Resolve network topology URL + track which dir was actually used ---
        resolved_dir = net_base  # default: full-cohort dir

        if n_event_bin:
            bin_dir      = f"{net_base}/density/{n_event_bin}"
            bin_html_key = f"{bin_dir}/network_topology.html"
            if _s3_object_exists(S3_DASHBOARD_BUCKET, bin_html_key):
                payload["network_topology_url"] = _dashboard_s3_url(bin_html_key)
                payload["data_scope"] = "per_bin"
                resolved_dir = bin_dir
            else:
                # Try nearest available bin before falling back to full-cohort
                nearest = _nearest_available_bin_s3(
                    S3_DASHBOARD_BUCKET, n_event_bin,
                    lambda b: f"{net_base}/density/{b}/network_topology.html",
                )
                if nearest:
                    nearest_dir = f"{net_base}/density/{nearest}"
                    payload["network_topology_url"] = _dashboard_s3_url(f"{nearest_dir}/network_topology.html")
                    payload["data_scope"] = "nearest_bin_fallback"
                    payload["nearest_bin"] = nearest
                    payload["message"] = (
                        f"No per-bin PGx network for density='{n_event_bin}'. "
                        f"Showing nearest available bin '{nearest}'."
                    )
                    resolved_dir = nearest_dir
                else:
                    if _s3_object_exists(S3_DASHBOARD_BUCKET, full_html_key):
                        payload["network_topology_url"] = _dashboard_s3_url(full_html_key)
                    payload["data_scope"] = "full_cohort_fallback"
                    payload["message"] = (
                        f"No per-bin PGx network for density='{n_event_bin}'. "
                        "Showing full-cohort network. Run fetch_vip_reports + build_network_topology with --bin."
                    )
        else:
            if _s3_object_exists(S3_DASHBOARD_BUCKET, full_html_key):
                payload["network_topology_url"] = _dashboard_s3_url(full_html_key)
            payload["data_scope"] = "full_cohort"

        # --- Citations + radar chart live in the same directory as network_topology.html ---
        cit_key   = f"{resolved_dir}/pubmed_citations.json"
        radar_key = f"{resolved_dir}/pgx_radar_data.json"
        if _s3_object_exists(S3_DASHBOARD_BUCKET, cit_key):
            payload["citations_url"] = _dashboard_s3_url(cit_key)
        if _s3_object_exists(S3_DASHBOARD_BUCKET, radar_key):
            payload["radar_chart_url"] = _dashboard_s3_url(radar_key)

        return _response(200, payload)
    except Exception as e:
        return _response(500, {"error": str(e)})


def load_shap_importance(cohort: str, age_band: str, model_type: str = "xgboost") -> pd.DataFrame:
    """Load SHAP importance from S3."""
    age_band_fname = age_band.replace("-", "_")
    s3_key = f"gold/shap_analysis/{cohort}/{age_band}/{cohort}_{age_band_fname}_shap_global_importance_{model_type}.csv"
    
    try:
        obj = s3_client.get_object(Bucket=S3_BUCKET, Key=s3_key)
        df = pd.read_csv(BytesIO(obj["Body"].read()))
        return df
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code")
        if code in ("NoSuchKey", "404", "NotFound"):
            return pd.DataFrame()
        raise


if __name__ == "__main__":
    # Local testing
    test_event = {
        "httpMethod": "GET",
        "path": "/metadata",
        "queryStringParameters": {"cohort": "opioid_ed"},
    }
    print(json.dumps(lambda_handler(test_event, None), indent=2))

