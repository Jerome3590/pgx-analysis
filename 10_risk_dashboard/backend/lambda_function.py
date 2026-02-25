"""
AWS Lambda function for PGx Risk Dashboard API.

Lambda receives user input (cohort, model/feature selections) and **filters** only—it does not
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
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
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

# Configuration
S3_BUCKET = os.environ.get("PGX_RESULTS_BUCKET", "pgxdatalake")
# Dashboard frontend bucket/prefix (where FP-Growth assets are uploaded so they render with the app)
S3_DASHBOARD_BUCKET = os.environ.get("S3_DASHBOARD_BUCKET", "jerome-dixon.io")
S3_DASHBOARD_PREFIX = os.environ.get("S3_DASHBOARD_PREFIX", "vcu/pgx-risk-calculator")
S3_DASHBOARD_REGION = os.environ.get("S3_DASHBOARD_REGION", "us-east-1")
MODEL_CACHE_TTL = int(os.environ.get("MODEL_CACHE_TTL", "3600"))


def _dashboard_s3_url(key: str) -> str:
    """Build path-style S3 URL for dashboard assets: https://s3.{region}.amazonaws.com/{bucket}/{key}"""
    return f"https://s3.{S3_DASHBOARD_REGION}.amazonaws.com/{S3_DASHBOARD_BUCKET}/{key}"
METADATA_PREFIX = "gold/dashboard/metadata"
MODEL_PREFIX = "gold/dashboard/models"

# Model storage paths (ECR container has models in /var/task/models/)
MODEL_BASE_PATH = os.environ.get("MODEL_BASE_PATH", "/var/task/models")
USE_CONTAINER_MODELS = os.path.exists(MODEL_BASE_PATH)

# In-memory model cache
_model_cache: Dict[str, Dict[str, Any]] = {}
_cache_timestamps: Dict[str, float] = {}

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
    
    # Try container filesystem first
    if container_path.exists():
        try:
            with open(container_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Failed to load metadata from container: {e}. Trying S3...")
    
    # Fallback to S3
    key = f"{METADATA_PREFIX}/{metadata_file}"
    try:
        obj = s3_client.get_object(Bucket=S3_BUCKET, Key=key)
        data = json.loads(obj["Body"].read().decode("utf-8"))
        return data
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code")
        if code in ("NoSuchKey", "404", "NotFound"):
            raise FileNotFoundError(f"Metadata not found: s3://{S3_BUCKET}/{key} or {container_path}")
        raise


def handle_metrics(_event: Dict[str, Any]) -> Dict[str, Any]:
    """GET /metrics — return prebuilt model performance metrics from S3 (no recomputation). Fallback to container bundle."""
    key = f"{METADATA_PREFIX}/model_performance_metrics.json"
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


def load_model(cohort: str, age_band: str, model_type: str) -> Any:
    """
    Load model from container filesystem (ECR) or S3 with caching.
    
    Priority:
    1. Container filesystem (/var/task/models/) - fastest, bundled in image
    2. S3 fallback - for development or if container models not available
    
    model_type: 'catboost', 'xgboost', or 'xgboost_rf'
    """
    cache_key = f"{cohort}/{age_band}/{model_type}"
    
    # Check cache
    if cache_key in _model_cache:
        timestamp = _cache_timestamps.get(cache_key, 0)
        if time.time() - timestamp < MODEL_CACHE_TTL:
            return _model_cache[cache_key]['model']
    
    age_band_fname = age_band.replace("-", "_")
    
    # Try loading from container filesystem first (ECR deployment)
    if USE_CONTAINER_MODELS:
        container_model_path = os.path.join(
            MODEL_BASE_PATH,
            cohort,
            age_band_fname,
            f"{model_type}.joblib"
        )
        
        # Also try JSON format for CatBoost
        if model_type == 'catboost':
            container_model_json = os.path.join(
                MODEL_BASE_PATH,
                cohort,
                age_band_fname,
                f"{model_type}.json"
            )
            if os.path.exists(container_model_json):
                try:
                    model = CatBoostClassifier()
                    model.load_model(container_model_json)
                    _model_cache[cache_key] = {'model': model}
                    _cache_timestamps[cache_key] = time.time()
                    return model
                except Exception as e:
                    print(f"Warning: Failed to load {container_model_json}: {e}")
        
        if os.path.exists(container_model_path):
            try:
                if model_type == 'catboost':
                    model = CatBoostClassifier()
                    model.load_model(container_model_path)
                else:
                    model = joblib.load(container_model_path)
                
                # Cache model
                _model_cache[cache_key] = {'model': model}
                _cache_timestamps[cache_key] = time.time()
                print(f"Loaded {model_type} from container: {container_model_path}")
                return model
            except Exception as e:
                print(f"Warning: Failed to load from container: {e}, falling back to S3")
    
    # Fallback to S3
    if model_type == 'catboost':
        key = f"{MODEL_PREFIX}/{cohort}/{age_band_fname}/catboost.joblib"
    elif model_type == 'xgboost':
        key = f"{MODEL_PREFIX}/{cohort}/{age_band_fname}/xgboost.joblib"
    elif model_type == 'xgboost_rf':
        key = f"{MODEL_PREFIX}/{cohort}/{age_band_fname}/xgboost_rf.joblib"
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    try:
        obj = s3_client.get_object(Bucket=S3_BUCKET, Key=key)
        model_bytes = obj["Body"].read()
        
        # Load model
        if model_type == 'catboost':
            model = CatBoostClassifier()
            model.load_model(BytesIO(model_bytes))
        else:
            model = joblib.load(BytesIO(model_bytes))
        
        # Cache model
        _model_cache[cache_key] = {'model': model}
        _cache_timestamps[cache_key] = time.time()
        print(f"Loaded {model_type} from S3: s3://{S3_BUCKET}/{key}")
        return model
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code")
        if code in ("NoSuchKey", "404", "NotFound"):
            raise FileNotFoundError(f"Model not found: s3://{S3_BUCKET}/{key}")
        raise


def load_feature_schema(cohort: str, age_band: str) -> Dict[str, Any]:
    """Load feature schema JSON from container filesystem or S3."""
    age_band_fname = age_band.replace("-", "_")
    
    # Try container filesystem first
    if USE_CONTAINER_MODELS:
        container_schema_path = os.path.join(
            MODEL_BASE_PATH,
            cohort,
            age_band_fname,
            "feature_schema.json"
        )
        if os.path.exists(container_schema_path):
            try:
                with open(container_schema_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Warning: Failed to load schema from container: {e}")
    
    # Fallback to S3
    key = f"{MODEL_PREFIX}/{cohort}/{age_band_fname}/feature_schema.json"
    
    try:
        obj = s3_client.get_object(Bucket=S3_BUCKET, Key=key)
        data = json.loads(obj["Body"].read().decode("utf-8"))
        return data
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code")
        if code in ("NoSuchKey", "404", "NotFound"):
            # Return default schema if not found
            return {'features': [], 'defaults': {}}
        raise


def build_feature_vector(
    age: int,
    drugs: List[str],
    icds: List[str],
    cpts: List[str],
    feature_schema: Dict[str, Any]
) -> np.ndarray:
    """
    Build feature vector matching model's expected schema.
    """
    features = {}
    
    # Initialize all features to 0
    for feature in feature_schema.get('features', []):
        features[feature] = 0.0
    
    # Set age
    if 'age' in features:
        features['age'] = float(age)
    
    # Set item features (drugs, ICDs, CPTs)
    for drug in drugs:
        feature_name = f"item_{drug.upper()}"
        if feature_name in features:
            features[feature_name] = 1.0
    
    for icd in icds:
        # Exclude F1120 (target variable, not an input)
        if icd.upper() == 'F1120':
            continue
        feature_name = f"item_{icd.upper()}"
        if feature_name in features:
            features[feature_name] = 1.0
    
    for cpt in cpts:
        feature_name = f"item_{cpt.upper()}"
        if feature_name in features:
            features[feature_name] = 1.0
    
    # Set default values for trajectory/sequence features
    defaults = feature_schema.get('defaults', {})
    for feature in features:
        if feature.startswith('trajectory_') or feature.startswith('pre_') or feature.startswith('itemset_'):
            if features[feature] == 0.0 and feature in defaults:
                features[feature] = defaults[feature]
    
    # Convert to array in correct order
    feature_list = [features.get(f, 0.0) for f in feature_schema.get('features', [])]
    return np.array(feature_list).reshape(1, -1)


def predict_risk(
    cohort: str,
    age_band: str,
    feature_vector: np.ndarray,
    model_types: List[str] = ['catboost', 'xgboost', 'xgboost_rf'],
    require_all_models: bool = True
) -> Dict[str, Any]:
    """
    Run ensemble prediction using all three models (CatBoost, XGBoost, XGBoost RF).
    
    Returns:
        {
            'predictions': {model_type: probability, ...},
            'ensemble_score': float,  # Weighted average
            'ensemble_method': str,   # 'weighted_average' or 'simple_average'
            'models_used': int,       # Number of models that succeeded
            'models_failed': List[str] # List of failed model types
        }
    """
    if not MODEL_LIBS_AVAILABLE:
        raise RuntimeError("Model libraries not available")
    
    predictions = {}
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
    
    print(f"Using model weights: {model_weights}")
    
    # Run predictions for all three models
    for model_type in model_types:
        try:
            model = load_model(cohort, age_band, model_type)
            
            if model_type == 'catboost':
                prob = model.predict_proba(feature_vector)[0][1]
            elif model_type in ['xgboost', 'xgboost_rf']:
                if isinstance(model, xgb.Booster):
                    dmatrix = xgb.DMatrix(feature_vector)
                    prob = model.predict(dmatrix)[0]
                    # Ensure probability is in [0, 1] range
                    prob = max(0.0, min(1.0, prob))
                else:
                    prob = model.predict_proba(feature_vector)[0][1]
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            predictions[model_type] = float(prob)
            
        except Exception as e:
            error_msg = str(e)
            errors[model_type] = error_msg
            print(f"Error predicting with {model_type}: {error_msg}")
            # Don't add failed models to predictions
    
    # Validate that we have at least one successful prediction
    if not predictions:
        raise RuntimeError(f"All models failed. Errors: {errors}")
    
    # Check if we have all three models (for robustness)
    models_used = len(predictions)
    models_failed = list(errors.keys())
    
    if require_all_models and models_used < len(model_types):
        print(f"Warning: Only {models_used}/{len(model_types)} models succeeded. "
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
    
    return {
        'predictions': predictions,
        'ensemble_score': float(ensemble_score),
        'ensemble_method': ensemble_method,
        'models_used': models_used,
        'models_failed': models_failed,
        'weights_used': available_weights,
        'weights_source': weights_source
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
        elif method == "POST" and path.endswith("/pgx/card"):
            return handle_pgx_card(event)
        elif method == "POST" and path.endswith("/risk"):
            if path.endswith("/risk/comparison"):
                return handle_risk_comparison(event)
            else:
                return handle_risk(event)
        elif method == "POST" and path.endswith("/causal"):
            if path.endswith("/causal/interactions"):
                return handle_causal_interactions(event)
            elif path.endswith("/causal/importance"):
                return handle_causal_importance(event)
            else:
                return _response(404, {"error": "Unknown causal endpoint"})
        elif method == "GET" and path.startswith("/visualizations/"):
            if path.endswith("/causal"):
                return handle_visualizations_causal(event)
            elif path.endswith("/dtw"):
                return handle_visualizations_dtw(event)
            elif path.endswith("/fpgrowth/network_html"):
                return handle_fpgrowth_network_html_proxy(event)
            elif path.endswith("/fpgrowth"):
                return handle_visualizations_fpgrowth(event)
            elif path.endswith("/activity_frequency") and "bupar" in path:
                return handle_visualizations_bupar_activity_frequency(event)
            elif path.endswith("/bupar/html"):
                return handle_visualizations_bupar_html_proxy(event)
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
        return _response(404, {"error": str(e)})
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
    body = json.loads(event.get("body") or "{}")
    
    raw_age = body.get("age")
    age_band_override = body.get("age_band")
    cohort = body.get("cohort")
    drugs = body.get("drugs", [])
    icds = body.get("icds", [])
    cpts = body.get("cpts", [])
    
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
        # Load feature schema
        feature_schema = load_feature_schema(cohort, age_band)
        
        # Build feature vector
        feature_vector = build_feature_vector(age, drugs, icds, cpts, feature_schema)
        
        # Predict using ensemble of all three models
        ensemble_result = predict_risk(cohort, age_band, feature_vector, require_all_models=True)
        
        risk_score = ensemble_result['ensemble_score']
        model_predictions = ensemble_result['predictions']
        
        # Determine risk band
        if risk_score < 0.2:
            risk_band = "low"
        elif risk_score < 0.5:
            risk_band = "medium"
        else:
            risk_band = "high"
        
        # Age-driven path uses 85-114 for ages 85+
        age_mapped = age >= 95 and age <= 114 and not age_band_override
        age_mapping_note = None
        if age_mapped:
            age_mapping_note = f"Age {age} in age band 85-114"
        
        return _response(200, {
            "risk_score": float(risk_score),
            "risk_band": risk_band,
            "model_breakdown": model_predictions,
            "ensemble_info": {
                "method": ensemble_result['ensemble_method'],
                "models_used": ensemble_result['models_used'],
                "models_failed": ensemble_result['models_failed'],
                "weights": ensemble_result['weights_used'],
                "weights_source": ensemble_result.get('weights_source', 'unknown')
            },
            "age_band_used": age_band,
            "cohort_used": cohort,
            "age": age,
            "age_mapped": age_mapped,
            "age_mapping_note": age_mapping_note
        })
    
    except Exception as e:
        import traceback
        return _response(500, {
            "error": str(e),
            "traceback": traceback.format_exc()
        })


def handle_risk_comparison(event: Dict[str, Any]) -> Dict[str, Any]:
    """POST /risk/comparison. Base may include cohort and age_band (from cohort tab + age); else derived from age."""
    body = json.loads(event.get("body") or "{}")
    
    base = body.get("base", {})
    scenarios = body.get("scenarios", [])
    
    base_age = int(base.get("age", 0))
    cohort = base.get("cohort")
    age_band = base.get("age_band")
    if cohort and age_band:
        cohort, age_band = str(cohort), str(age_band)
    else:
        cohort, age_band = determine_cohort_and_age_band(base_age)
    
    try:
        feature_schema = load_feature_schema(cohort, age_band)
        
        # Calculate base risk using ensemble
        base_feature_vector = build_feature_vector(
            base_age,
            base.get("drugs", []),
            base.get("icds", []),
            base.get("cpts", []),
            feature_schema
        )
        base_ensemble = predict_risk(cohort, age_band, base_feature_vector, require_all_models=True)
        base_risk = base_ensemble['ensemble_score']
        
        # Calculate scenario risks using ensemble
        scenario_results = []
        for scenario in scenarios:
            scenario_feature_vector = build_feature_vector(
                base_age,
                scenario.get("drugs", []),
                scenario.get("icds", []),
                scenario.get("cpts", []),
                feature_schema
            )
            scenario_ensemble = predict_risk(cohort, age_band, scenario_feature_vector, require_all_models=True)
            scenario_risk = scenario_ensemble['ensemble_score']
            
            scenario_results.append({
                "name": scenario.get("name", "Scenario"),
                "risk_score": float(scenario_risk),
                "delta": float(scenario_risk - base_risk),
                "model_breakdown": scenario_ensemble['predictions']
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


def load_cpic_data() -> Dict[str, List[Dict[str, Any]]]:
    """
    Load CPIC gene-drug pairs data from the master Excel file.
    Uses the official CPIC Excel file: cpic_gene-drug_pairs.xlsx
    Tries container path first, then S3.
    """
    # Try container path first (master Excel file)
    container_excel_path = "/var/task/data/cpic_gene-drug_pairs.xlsx"
    s3_excel_path = f"{METADATA_PREFIX}/cpic_gene-drug_pairs.xlsx"
    
    cpic_data = {}
    
    # Try loading Excel file from container
    try:
        if os.path.exists(container_excel_path):
            import pandas as pd
            df = pd.read_excel(container_excel_path)
            
            # Standardize column names (handle variations)
            gene_col = None
            drug_col = None
            guideline_col = None
            cpic_level_col = None
            fda_label_col = None
            
            for col in df.columns:
                col_lower = col.lower()
                if 'gene' in col_lower and gene_col is None:
                    gene_col = col
                elif 'drug' in col_lower and drug_col is None:
                    drug_col = col
                elif 'guideline' in col_lower and guideline_col is None:
                    guideline_col = col
                elif 'cpic' in col_lower and 'level' in col_lower and cpic_level_col is None:
                    cpic_level_col = col
                elif ('fda' in col_lower or 'label' in col_lower) and fda_label_col is None:
                    fda_label_col = col
            
            if gene_col and drug_col:
                for _, row in df.iterrows():
                    gene = str(row.get(gene_col, "")).upper().strip()
                    drug = str(row.get(drug_col, "")).strip()
                    
                    if gene and drug and gene != "NAN" and drug != "NAN":
                        if gene not in cpic_data:
                            cpic_data[gene] = []
                        
                        # Check for duplicates
                        if not any(d["drug"] == drug for d in cpic_data[gene]):
                            cpic_data[gene].append({
                                "drug": drug,
                                "guideline": str(row.get(guideline_col, "")) if guideline_col else "",
                                "cpic_level": str(row.get(cpic_level_col, "")) if cpic_level_col else "",
                                "pgx_on_fda_label": str(row.get(fda_label_col, "")) if fda_label_col else ""
                            })
            
            if cpic_data:
                print(f"Loaded {sum(len(drugs) for drugs in cpic_data.values())} gene-drug pairs from Excel file")
                return cpic_data
    except ImportError:
        print("ERROR: pandas not available. Cannot load CPIC Excel file.")
        raise
    except Exception as e:
        print(f"Error loading CPIC Excel from container: {e}")
    
    # Try S3 Excel file
    try:
        import pandas as pd
        s3 = boto3.client('s3')
        obj = s3.get_object(Bucket=S3_BUCKET, Key=s3_excel_path)
        
        # Read Excel from S3
        excel_data = obj['Body'].read()
        df = pd.read_excel(BytesIO(excel_data))
        
        # Same column detection logic as above
        gene_col = None
        drug_col = None
        guideline_col = None
        cpic_level_col = None
        fda_label_col = None
        
        for col in df.columns:
            col_lower = col.lower()
            if 'gene' in col_lower and gene_col is None:
                gene_col = col
            elif 'drug' in col_lower and drug_col is None:
                drug_col = col
            elif 'guideline' in col_lower and guideline_col is None:
                guideline_col = col
            elif 'cpic' in col_lower and 'level' in col_lower and cpic_level_col is None:
                cpic_level_col = col
            elif ('fda' in col_lower or 'label' in col_lower) and fda_label_col is None:
                fda_label_col = col
        
        if gene_col and drug_col:
            for _, row in df.iterrows():
                gene = str(row.get(gene_col, "")).upper().strip()
                drug = str(row.get(drug_col, "")).strip()
                
                if gene and drug and gene != "NAN" and drug != "NAN":
                    if gene not in cpic_data:
                        cpic_data[gene] = []
                    
                    if not any(d["drug"] == drug for d in cpic_data[gene]):
                        cpic_data[gene].append({
                            "drug": drug,
                            "guideline": str(row.get(guideline_col, "")) if guideline_col else "",
                            "cpic_level": str(row.get(cpic_level_col, "")) if cpic_level_col else "",
                            "pgx_on_fda_label": str(row.get(fda_label_col, "")) if fda_label_col else ""
                        })
        
        if cpic_data:
            print(f"Loaded {sum(len(drugs) for drugs in cpic_data.values())} gene-drug pairs from S3 Excel")
            return cpic_data
    except ImportError:
        print("ERROR: pandas not available. Cannot load CPIC Excel file from S3.")
        raise
    except Exception as e:
        print(f"Error loading CPIC Excel from S3: {e}")
        raise
    
    # If we get here, CPIC data loading failed
    raise FileNotFoundError("CPIC Excel file not found in container or S3. Please ensure cpic_gene-drug_pairs.xlsx is available.")


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


def handle_causal_interactions(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    POST /causal/interactions
    
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


def load_causal_importance(cohort: str, age_band: str, model_type: str = "xgboost") -> pd.DataFrame:
    """
    Load causal importance results from S3.
    
    Args:
        cohort: Cohort name
        age_band: Age band
        model_type: Model type ('xgboost', 'catboost', 'xgboost_rf')
    
    Returns:
        DataFrame with causal importance results, or empty DataFrame if not found
    """
    if not MODEL_LIBS_AVAILABLE:
        print("ERROR: pandas not available. Cannot load causal importance.")
        return pd.DataFrame()
    
    age_band_fname = age_band.replace("-", "_")
    
    # Try container filesystem first
    if USE_CONTAINER_MODELS:
        container_causal_path = Path(MODEL_BASE_PATH).parent.parent / "8_ffa_analysis" / "outputs" / cohort / age_band_fname / model_type / "causal_importance.parquet"
        if container_causal_path.exists():
            try:
                return pd.read_parquet(container_causal_path)
            except Exception as e:
                print(f"Warning: Failed to load causal importance from container: {e}")
    
    # Fallback to S3
    s3_key = f"gold/ffa_analysis/{cohort}/{age_band}/{model_type}/causal_importance.parquet"
    
    try:
        obj = s3_client.get_object(Bucket=S3_BUCKET, Key=s3_key)
        df = pd.read_parquet(BytesIO(obj["Body"].read()))
        print(f"Loaded causal importance from S3: s3://{S3_BUCKET}/{s3_key}")
        return df
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code")
        if code in ("NoSuchKey", "404", "NotFound"):
            print(f"Causal importance not found: s3://{S3_BUCKET}/{s3_key}")
            return pd.DataFrame()
        raise


def handle_causal_importance(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    POST /causal/importance
    
    Returns single-feature causal importance results, optionally filtered by selected drugs.
    
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
        "causal_importance": [
            {
                "feature": "item_DRUG_A",
                "causal_importance": 0.123456,
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
        
        if not cohort or not age_band:
            return _response(400, {"error": "cohort and age_band are required"})
        
        # Load causal importance results from S3 (pre-computed)
        causal_df = load_causal_importance(cohort, age_band, model_type)
        
        if causal_df.empty:
            return _response(200, {
                "causal_importance": [],
                "summary": {
                    "total_features": 0,
                    "filtered_features": 0,
                    "selected_drugs": selected_drugs,
                    "message": "No causal importance results found. Run Step 8 (FFA Analysis) first."
                }
            })
        
        # Filter to selected drugs if provided
        filtered_df = causal_df.copy()
        if selected_drugs:
            # Convert drug codes to feature names (item_DRUG_CODE format)
            selected_features = [f"item_{drug.upper()}" for drug in selected_drugs]
            # Filter to features that match selected drugs
            filtered_df = filtered_df[filtered_df['feature'].isin(selected_features)]
        
        # Sort by causal importance and get top N
        filtered_df = filtered_df.sort_values('causal_importance', ascending=False)
        top_df = filtered_df.head(top_n).copy()
        top_df['rank'] = range(1, len(top_df) + 1)
        
        # Format response
        causal_importance = top_df[['feature', 'causal_importance', 'rank']].to_dict('records')
        
        summary = {
            "total_features": len(causal_df),
            "filtered_features": len(filtered_df),
            "selected_drugs": selected_drugs,
            "top_n_returned": len(causal_importance)
        }
        
        return _response(200, {
            "causal_importance": causal_importance,
            "summary": summary
        })
    
    except Exception as e:
        import traceback
        return _response(500, {
            "error": str(e),
            "traceback": traceback.format_exc()
        })


def _causal_feature_set_from_codes(drugs: List[str], icds: List[str], cpts: List[str]) -> set:
    """Build set of feature names that match causal_data (item_X, item_icd_X, item_cpt_X, item_drug_X)."""
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


def handle_visualizations_causal(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    GET /visualizations/causal?cohort=...&age_band=...[&drugs=...&icds=...&cpts=...&whatif=...]

    Same pattern as Feature Importance: load causal_data.json from S3 and return inline.
    Lambda applies optional filters (drugs, icds, cpts, whatif) and returns chart_data
    (causal_factors, shap_importance, whatif variants, feature_interactions) so the
    frontend can render without re-filtering. Radar plot uses top N of causal_factors.
    S3 key: {S3_DASHBOARD_PREFIX}/causal/{cohort}/{age_band_fname}/causal_data.json.
    """
    try:
        params = event.get("queryStringParameters") or {}
        cohort = params.get("cohort")
        age_band = params.get("age_band")

        if not cohort or not age_band:
            return _response(400, {"error": "cohort and age_band parameters required"})

        age_band_fname = age_band.replace("-", "_")
        prefix = f"{S3_DASHBOARD_PREFIX.strip('/')}/causal/{cohort}/{age_band_fname}"
        causal_key = f"{prefix}/causal_data.json"
        payload: Dict[str, Any] = {"causal_data_url": _dashboard_s3_url(causal_key)}

        try:
            obj = s3_client.get_object(Bucket=S3_DASHBOARD_BUCKET, Key=causal_key)
            data = json.loads(obj["Body"].read().decode("utf-8"))
            payload["causal_data"] = data
        except ClientError as e:
            if e.response.get("Error", {}).get("Code") not in ("NoSuchKey", "404", "403", "AccessDenied"):
                raise
        except (json.JSONDecodeError, TypeError):
            pass

        # Build chart_data (ready-to-render) with optional filter and whatif, same pattern as FI
        if payload.get("causal_data"):
            raw = payload["causal_data"]
            top = raw.get("top_causal_factors") or []
            drugs = [x.strip() for x in (params.get("drugs") or "").split(",") if x.strip()]
            icds = [x.strip() for x in (params.get("icds") or "").split(",") if x.strip()]
            cpts = [x.strip() for x in (params.get("cpts") or "").split(",") if x.strip()]
            whatif_codes = [x.strip() for x in (params.get("whatif") or "").split(",") if x.strip()]

            selected_set = _causal_feature_set_from_codes(drugs, icds, cpts)
            whatif_set = _causal_feature_set_from_codes(
                whatif_codes, whatif_codes, whatif_codes
            ) if whatif_codes else set()

            def row_to_factor(r: Dict[str, Any]) -> Dict[str, Any]:
                feat = r.get("feature") or ""
                return {
                    "feature": feat,
                    "importance": float(r.get("causal_responsibility") or r.get("importance") or 0),
                }

            def row_to_shap(r: Dict[str, Any]) -> Dict[str, Any]:
                feat = r.get("feature") or ""
                return {
                    "feature": feat,
                    "importance": float(r.get("shap_importance") or r.get("importance") or 0),
                }

            if selected_set:
                causal_factors = [row_to_factor(r) for r in top if (r.get("feature") or "") in selected_set]
                shap_importance = [row_to_shap(r) for r in top if (r.get("feature") or "") in selected_set]
                filtered_by_codes = True
            else:
                causal_factors = [row_to_factor(r) for r in top]
                shap_importance = [row_to_shap(r) for r in top]
                filtered_by_codes = False

            causal_factors_whatif = [row_to_factor(r) for r in top if (r.get("feature") or "") in whatif_set] if whatif_set else []
            shap_importance_whatif = [row_to_shap(r) for r in top if (r.get("feature") or "") in whatif_set] if whatif_set else []

            chart_data: Dict[str, Any] = {
                "causal_factors": causal_factors,
                "shap_importance": shap_importance,
                "filtered_by_codes": filtered_by_codes,
            }
            if causal_factors_whatif:
                chart_data["causal_factors_whatif"] = causal_factors_whatif
            if shap_importance_whatif:
                chart_data["shap_importance_whatif"] = shap_importance_whatif
            if raw.get("feature_interactions"):
                chart_data["feature_interactions"] = raw["feature_interactions"]
            payload["chart_data"] = chart_data

        return _response(200, payload)
    except Exception as e:
        return _response(500, {"error": str(e)})


def handle_visualizations_dtw(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    GET /visualizations/dtw?cohort=...&age_band=...

    Prefer JSON: fetches chart_data.json and sequence_heatmap.json from S3 when present and returns
    them inline (chart_data, sequence_heatmap) so the frontend can render without a second request
    and works even when bucket is not public. Also returns chart_data_url and sequence_heatmap_url
    for fallback. Overview/sample images: only included if objects exist (HEAD check).
    """
    try:
        params = event.get("queryStringParameters") or {}
        cohort = params.get("cohort")
        age_band = params.get("age_band")

        if not cohort or not age_band:
            return _response(400, {"error": "cohort and age_band parameters required"})

        age_band_fname = age_band.replace("-", "_")
        prefix = f"{S3_DASHBOARD_PREFIX.strip('/')}/dtw/{cohort}/{age_band}"
        plots_key = f"{prefix}/plots"
        bucket = S3_DASHBOARD_BUCKET

        payload = {
            "chart_data_url": _dashboard_s3_url(f"{prefix}/chart_data.json"),
            "sequence_heatmap_url": _dashboard_s3_url(f"{prefix}/sequence_heatmap.json"),
            "metrics": {},
        }

        # Prefer JSON: load chart_data and sequence_heatmap from S3 when present (one round-trip, works without public read)
        for key, s3_key in [
            ("chart_data", f"{prefix}/chart_data.json"),
            ("sequence_heatmap", f"{prefix}/sequence_heatmap.json"),
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

        # Trajectory overview: prefer Plotly figure JSON so frontend builds Plotly (no iframe/HTML)
        trajectory_plot_key = f"{plots_key}/trajectory_overview_plot.json"
        try:
            obj = s3_client.get_object(Bucket=bucket, Key=trajectory_plot_key)
            payload["trajectory_overview_plot"] = json.loads(obj["Body"].read().decode("utf-8"))
        except ClientError as e:
            if e.response.get("Error", {}).get("Code") not in ("NoSuchKey", "404", "403", "AccessDenied"):
                raise
        except (json.JSONDecodeError, TypeError):
            pass

        # Fallback: image URLs when Plotly JSON not present (avoid broken image when pipeline not run)
        overview_key = f"{plots_key}/dtw_trajectory_analysis_{cohort}_{age_band_fname}.png"
        sample_key = f"{plots_key}/dtw_sample_trajectories_{cohort}_{age_band_fname}.png"
        overview_html_key = f"{plots_key}/dtw_trajectory_cluster_interactive_{cohort}_{age_band_fname}.html"
        if not payload.get("trajectory_overview_plot") and _s3_object_exists(bucket, overview_key):
            payload["overview_image"] = _dashboard_s3_url(overview_key)
        if not payload.get("trajectory_overview_plot") and _s3_object_exists(bucket, sample_key):
            payload["sample_trajectories_image"] = _dashboard_s3_url(sample_key)
        if _s3_object_exists(bucket, overview_html_key):
            payload["overview_interactive"] = _dashboard_s3_url(overview_html_key)

        return _response(200, payload)
    except Exception as e:
        return _response(500, {"error": str(e)})


def handle_fpgrowth_network_html_proxy(event: Dict[str, Any]) -> Dict[str, Any]:
    """GET /visualizations/fpgrowth/network_html?cohort=...&age_band=...
    Fetches combined_rules_network.html from S3 and returns it with Content-Type: text/html
    so the dashboard iframe renders it instead of triggering a download.
    """
    try:
        params = event.get("queryStringParameters") or {}
        cohort = params.get("cohort")
        age_band = params.get("age_band")
        if not cohort or not age_band:
            return _response(400, {"error": "cohort and age_band parameters required"})
        age_band_fname = age_band.replace("-", "_")
        prefix = f"{S3_DASHBOARD_PREFIX.strip('/')}/fpgrowth"
        base_key = f"{prefix}/{cohort}/{age_band}/plots"
        key = f"{base_key}/{cohort}_{age_band_fname}_combined_rules_network.html"
        obj = s3_client.get_object(Bucket=S3_DASHBOARD_BUCKET, Key=key)
        html_body = obj["Body"].read().decode("utf-8")
        return _response_html(200, html_body)
    except ClientError as e:
        if e.response.get("Error", {}).get("Code") in ("NoSuchKey", "404", "AccessDenied"):
            return _response_html(404, "<!DOCTYPE html><html><body><p>Network visualization not found. Run the FP-Growth pipeline to build it.</p></body></html>")
        raise
    except Exception as e:
        return _response(500, {"error": str(e)})


def handle_visualizations_fpgrowth(event: Dict[str, Any]) -> Dict[str, Any]:
    """GET /visualizations/fpgrowth?cohort=...&age_band=...&item_type=...
    Returns HTTPS URLs for dashboard: itemsets PNG, network Plotly HTML (iframe by cohort).
    Prefer network_html_proxy_url for the iframe so HTML is served with Content-Type: text/html.
    When itemsets JSON is present in S3, returns itemsets_data for client-side Plotly rendering.
    If itemsets/rules were empty, returns JSON with empty=True and message for dashboard to show.
    Files are built by 4_dashboard_visuals / create_plots and uploaded to the dashboard bucket
    (S3_DASHBOARD_BUCKET) under {S3_DASHBOARD_PREFIX}/fpgrowth/{cohort}/{age_band}/plots/.
    """
    try:
        params = event.get("queryStringParameters") or {}
        cohort = params.get("cohort")
        age_band = params.get("age_band")
        # Research focus: drugs only; ignore other item_type for cleaner drug-sequence visuals.
        item_type = "drug_name"
        
        if not cohort or not age_band:
            return _response(400, {"error": "cohort and age_band parameters required"})
        
        prefix = f"{S3_DASHBOARD_PREFIX.strip('/')}/fpgrowth"
        base_key = f"{prefix}/{cohort}/{age_band}/plots"
        empty_state_key = f"{base_key}/empty_state.json"

        # When itemsets/rules were empty, pipeline uploads empty_state.json; return it for dashboard
        try:
            obj = s3_client.get_object(Bucket=S3_DASHBOARD_BUCKET, Key=empty_state_key)
            body = obj["Body"].read().decode("utf-8")
            payload = json.loads(body)
            return _response(200, payload)
        except ClientError as e:
            if e.response.get("Error", {}).get("Code") not in ("NoSuchKey", "404", "403", "AccessDenied"):
                raise
        except (json.JSONDecodeError, KeyError):
            pass

        age_band_fname = age_band.replace("-", "_")
        # Combined network: drug association rules only (no type filter)
        network_combined_key = f"{base_key}/{cohort}_{age_band_fname}_combined_rules_network.html"
        # Legacy per-item-type filenames (itemsets + optional legacy network)
        itemsets_key = f"{base_key}/{cohort}_{age_band_fname}_{item_type}_combined_top_itemsets.png"
        network_html_key = f"{base_key}/{cohort}_{age_band_fname}_{item_type}_target_rules_network.html"
        network_png_key = f"{base_key}/{cohort}_{age_band_fname}_{item_type}_target_rules_network.png"
        itemsets_interactive_key = f"{base_key}/{cohort}_{age_band_fname}_{item_type}_itemsets_interactive.html"
        network_interactive_key = f"{base_key}/{cohort}_{age_band_fname}_{item_type}_network_interactive.html"

        payload = {
            "network_combined_html": _dashboard_s3_url(network_combined_key),
            "itemsets_image": _dashboard_s3_url(itemsets_key),
            "support_image": _dashboard_s3_url(itemsets_key),
            "network_html": _dashboard_s3_url(network_html_key),
            "network_png": _dashboard_s3_url(network_png_key),
            "itemsets_interactive": _dashboard_s3_url(itemsets_interactive_key),
            "network_interactive": _dashboard_s3_url(network_interactive_key),
        }
        # Itemsets JSON for client-side Plotly rendering (when uploaded by pipeline)
        data_prefix = f"{prefix}/{cohort}/{age_band}/data"
        itemsets_json_key = f"{data_prefix}/{item_type}_itemsets.json"
        try:
            obj = s3_client.get_object(Bucket=S3_DASHBOARD_BUCKET, Key=itemsets_json_key)
            raw = obj["Body"].read().decode("utf-8")
            payload["itemsets_data"] = json.loads(raw)
        except (ClientError, json.JSONDecodeError, TypeError):
            pass
        return _response(200, payload)
    except Exception as e:
        return _response(500, {"error": str(e)})


def handle_visualizations_feature_importance(event: Dict[str, Any]) -> Dict[str, Any]:
    """GET /visualizations/feature_importance?cohort=...
    Cohort: opioid_ed, non_opioid_ed, or combined. Always returns aggregated heatmap (no model filter).
    Prefer JSON over PNG: returns heatmap_data when available; frontend falls back to heatmap_url (PNG).
    S3: {prefix}/feature_importance/{cohort}/aggregated_fi_heatmap.json|png; combined uses combined_cohorts_*.
    """
    try:
        params = event.get("queryStringParameters") or {}
        cohort = (params.get("cohort") or "").strip()
        if not cohort:
            return _response(400, {"error": "cohort parameter required"})
        if cohort not in ("opioid_ed", "non_opioid_ed", "combined"):
            cohort = "opioid_ed"
        prefix = f"{S3_DASHBOARD_PREFIX.strip('/')}/feature_importance"
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
BUPAR_HTML_VISUALS = frozenset({
    "activity_frequency_interactive",
    "trace_explorer_interactive",
    "process_matrix_interactive",
})


def handle_visualizations_bupar_html_proxy(event: Dict[str, Any]) -> Dict[str, Any]:
    """GET /visualizations/bupar/html?cohort=...&age_band=...&visual=...
    Fetches BupaR interactive HTML from S3 and returns it with Content-Type: text/html
    so the dashboard iframe renders it instead of triggering a download.
    visual: one of activity_frequency_interactive, trace_explorer_interactive, process_matrix_interactive.
    """
    try:
        params = event.get("queryStringParameters") or {}
        cohort = params.get("cohort")
        age_band = params.get("age_band")
        visual = (params.get("visual") or "").strip()
        if not cohort or not age_band:
            return _response(400, {"error": "cohort and age_band parameters required"})
        if visual not in BUPAR_HTML_VISUALS:
            return _response(400, {"error": f"visual must be one of: {sorted(BUPAR_HTML_VISUALS)}"})
        age_band_fname = age_band.replace("-", "_")
        prefix = f"{S3_DASHBOARD_PREFIX.strip('/')}/bupar"
        base_key = f"{prefix}/{cohort}/{age_band}/plots"
        base = f"{cohort}_{age_band_fname}"
        key = f"{base_key}/{base}_{visual}.html"
        obj = s3_client.get_object(Bucket=S3_DASHBOARD_BUCKET, Key=key)
        html_body = obj["Body"].read().decode("utf-8")
        return _response_html(200, html_body)
    except ClientError as e:
        if e.response.get("Error", {}).get("Code") in ("NoSuchKey", "404", "AccessDenied"):
            return _response_html(404, "<!DOCTYPE html><html><body><p>BupaR visualization not found. Run the BupaR pipeline to build it.</p></body></html>")
        raise
    except Exception as e:
        return _response(500, {"error": str(e)})


def handle_visualizations_bupar_activity_frequency(event: Dict[str, Any]) -> Dict[str, Any]:
    """GET /visualizations/bupar/activity_frequency?cohort=...&age_band=...
    Returns JSON: { overall, pre_target, post_target } each { year_labels, data } for dashboard bar charts.
    Pipeline writes activity_frequency.json, pre_target_activity_frequency.json, post_target_activity_frequency.json to S3.
    Prefer JSON over PNG for these visuals: frontend renders Chart.js from this endpoint instead of static images.
    """
    try:
        params = event.get("queryStringParameters") or {}
        cohort = params.get("cohort")
        age_band = params.get("age_band")
        if not cohort or not age_band:
            return _response(400, {"error": "cohort and age_band parameters required"})
        age_band_fname = age_band.replace("-", "_")
        prefix = f"{S3_DASHBOARD_PREFIX.strip('/')}/bupar/{cohort}/{age_band}/plots"
        base = f"{cohort}_{age_band_fname}"
        keys = {
            "overall": f"{prefix}/{base}_activity_frequency.json",
            "pre_target": f"{prefix}/{base}_pre_target_activity_frequency.json",
            "post_target": f"{prefix}/{base}_post_target_activity_frequency.json",
        }
        payload = {}
        for name, key in keys.items():
            try:
                obj = s3_client.get_object(Bucket=S3_DASHBOARD_BUCKET, Key=key)
                data = json.loads(obj["Body"].read().decode("utf-8"))
                payload[name] = data
            except ClientError as e:
                if e.response.get("Error", {}).get("Code") in ("NoSuchKey", "404", "403", "AccessDenied"):
                    payload[name] = None
                else:
                    raise
        return _response(200, payload)
    except Exception as e:
        return _response(500, {"error": str(e)})


def handle_visualizations_bupar(event: Dict[str, Any]) -> Dict[str, Any]:
    """GET /visualizations/bupar?cohort=...&age_band=...
    Returns HTTPS URLs only for BupaR plot objects that exist in S3 (HEAD check).
    Dashboard bucket: S3_DASHBOARD_BUCKET under {S3_DASHBOARD_PREFIX}/bupar/{cohort}/{age_band}/plots/.
    """
    try:
        params = event.get("queryStringParameters") or {}
        cohort = params.get("cohort")
        age_band = params.get("age_band")

        if not cohort or not age_band:
            return _response(400, {"error": "cohort and age_band parameters required"})

        age_band_fname = age_band.replace("-", "_")
        prefix = f"{S3_DASHBOARD_PREFIX.strip('/')}/bupar"
        base_key = f"{prefix}/{cohort}/{age_band}/plots"
        pre_suffix = "pre_f1120" if cohort == "opioid_ed" else "pre_hcg"
        base = f"{cohort}_{age_band_fname}"

        # RQ-only artifact keys (RESEARCH_QUESTIONS_ARTIFACTS.md). Do not request archived artifacts.
        candidates: List[Tuple[str, str]] = [
            ("activity_frequency_image", f"{base_key}/{base}_overall_activity_frequency.png"),
            ("activity_frequency_interactive", f"{base_key}/{base}_activity_frequency_interactive.html"),
            ("pre_target_frequency_image", f"{base_key}/{base}_{pre_suffix}_activity_frequency.png"),
            ("sequence_image", f"{base_key}/{base}_activity_sequence_top.png"),
            ("trace_explorer_interactive", f"{base_key}/{base}_trace_explorer_interactive.html"),
            ("trace_explorer_pre_image", f"{base_key}/{base}_trace_explorer_{pre_suffix}.png"),
            ("process_matrix_drug_drug", f"{base_key}/{base}_process_matrix_drug_drug.png"),
        ]

        payload: Dict[str, Any] = {}
        for payload_key, s3_key in candidates:
            if _s3_object_exists(S3_DASHBOARD_BUCKET, s3_key):
                payload[payload_key] = _dashboard_s3_url(s3_key)

        # Prefer JSON for frontend Plotly: load trace_explorer_plot and process_matrix_drug_drug when present
        for key, s3_key in [
            ("trace_explorer_plot", f"{base_key}/{base}_trace_explorer_plot.json"),
            ("process_matrix_drug_drug", f"{base_key}/{base}_process_matrix_drug_drug.json"),
        ]:
            try:
                obj = s3_client.get_object(Bucket=S3_DASHBOARD_BUCKET, Key=s3_key)
                payload[key] = json.loads(obj["Body"].read().decode("utf-8"))
            except ClientError as e:
                if e.response.get("Error", {}).get("Code") not in ("NoSuchKey", "404", "403", "AccessDenied"):
                    raise
            except (json.JSONDecodeError, TypeError):
                pass

        return _response(200, payload)
    except Exception as e:
        return _response(500, {"error": str(e)})


def handle_visualizations_cohort_pgx(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    GET /visualizations/cohort_pgx?cohort=...&age_band=...

    Returns network_topology_url only when the S3 object exists (HEAD check). Built by
    Cohort PGx pipeline (fetch_vip_reports + build_network_topology); expected key:
    {S3_DASHBOARD_PREFIX}/cohort_pgx/networks/{cohort}/{age_band_fname}/network_topology.html.
    Sync 10_risk_dashboard/visualizations/cohort_pgx/ to dashboard S3 after building.
    """
    try:
        params = event.get("queryStringParameters") or {}
        cohort = params.get("cohort")
        age_band = params.get("age_band")

        if not cohort or not age_band:
            return _response(400, {"error": "cohort and age_band parameters required"})

        age_band_fname = age_band.replace("-", "_")
        prefix = f"{S3_DASHBOARD_PREFIX.strip('/')}/cohort_pgx"
        base_key = f"{prefix}/networks/{cohort}/{age_band_fname}"
        html_key = f"{base_key}/network_topology.html"
        payload = {}
        if _s3_object_exists(S3_DASHBOARD_BUCKET, html_key):
            payload["network_topology_url"] = _dashboard_s3_url(html_key)
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

