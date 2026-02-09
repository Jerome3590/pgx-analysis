"""
AWS Lambda function for PGx Risk Dashboard API.

This function implements a robust ensemble approach using all three models:
1. CatBoost
2. XGBoost
3. XGBoost RF (Random Forest mode)

The ensemble combines predictions from all three models using weighted averaging
for improved robustness and reliability.

Handles:
- GET /metadata - Returns valid codes for cohorts/age_bands
- POST /risk - Calculates risk score using ensemble of all three models
- POST /risk/comparison - Compares risk scores for different scenarios
- POST /causal/importance - Returns causal importance for features (optionally filtered by selected drugs)
- POST /causal/interactions - Returns multi-feature interaction analysis results

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
MODEL_CACHE_TTL = int(os.environ.get("MODEL_CACHE_TTL", "3600"))
METADATA_PREFIX = "gold/dashboard/metadata"
MODEL_PREFIX = "gold/dashboard/models"

# Model storage paths (ECR container has models in /var/task/models/)
MODEL_BASE_PATH = os.environ.get("MODEL_BASE_PATH", "/var/task/models")
USE_CONTAINER_MODELS = os.path.exists(MODEL_BASE_PATH)

# In-memory model cache
_model_cache: Dict[str, Dict[str, Any]] = {}
_cache_timestamps: Dict[str, float] = {}

s3_client = boto3.client("s3")


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


def determine_cohort_and_age_band(age: int) -> Tuple[str, str]:
    """
    Determine cohort and age band from age.
    
    Rules:
    - Ages 13-64: opioid_ed cohort (age bands: 13-24, 25-44, 45-54, 55-64)
    - Ages 65-94: non_opioid_ed (polypharmacy) cohort (age bands: 65-74, 75-84, 85-94)
    - Ages 95-114: mapped to 85-94 age band (small cohort size, uses 85-94 model)
    
    Note: Age band 0-12 is excluded due to small cohort size.
    """
    if age < 13:
        raise ValueError("Age must be 13 or older. Age band 0-12 is not supported due to small cohort size.")
    elif 13 <= age <= 64:
        cohort = "opioid_ed"
        if age <= 24:
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
        else:  # 85 <= age <= 114 (mapped to 85-94)
            age_band = "85-94"
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
    """GET /metrics — return model performance metrics from container (bundled in ECR at build time)."""
    container_path = Path("/var/task/metadata/model_performance_metrics.json")
    if not container_path.exists():
        return _response(200, {"by_cohort": {}, "source": "none"})
    try:
        with open(container_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return _response(200, data if isinstance(data, dict) else {"by_cohort": {}, "payload": data})
    except Exception as e:
        print(f"Metrics load failed: {e}")
        return _response(200, {"by_cohort": {}, "source": "none", "error": str(e)})


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
            elif path.endswith("/fpgrowth"):
                return handle_visualizations_fpgrowth(event)
            elif path.endswith("/bupar"):
                return handle_visualizations_bupar(event)
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
      - Cohort/age_band combinations correspond to the modeled age bands:
          opioid_ed:    13-24, 25-44, 45-54
          non_opioid_ed: 65-74, 75-84, 85-94
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
        
        # Check if age was mapped (95-114 mapped to 85-94) in the age-driven path
        age_mapped = age >= 95 and age <= 114 and not age_band_override
        age_mapping_note = None
        if age_mapped:
            age_mapping_note = f"Age {age} mapped to age band 85-94 due to small cohort size"
        
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
    """POST /risk/comparison"""
    body = json.loads(event.get("body") or "{}")
    
    base = body.get("base", {})
    scenarios = body.get("scenarios", [])
    
    base_age = int(base.get("age", 0))
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


def handle_visualizations_causal(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    GET /visualizations/causal?cohort=...&age_band=...[&drugs=...&icds=...&cpts=...]

    Optional filter by user-selected codes: drugs, icds, cpts (each comma-separated).
    When provided, causal and SHAP results are restricted to features matching those codes (item_<CODE>).
    """
    try:
        params = event.get("queryStringParameters") or {}
        cohort = params.get("cohort")
        age_band = params.get("age_band")
        drugs_param = params.get("drugs", "")
        icds_param = params.get("icds", "")
        cpts_param = params.get("cpts", "")

        if not cohort or not age_band:
            return _response(400, {"error": "cohort and age_band parameters required"})

        # Build optional feature filter from user-selected codes (same naming as risk model: item_<CODE>)
        selected_features: Optional[Set[str]] = None
        if drugs_param or icds_param or cpts_param:
            selected_features = set()
            for code in (c.strip() for c in drugs_param.split(",") if c.strip()):
                selected_features.add(f"item_{code.upper()}")
            for code in (c.strip() for c in icds_param.split(",") if c.strip()):
                selected_features.add(f"item_{code.upper()}")
            for code in (c.strip() for c in cpts_param.split(",") if c.strip()):
                selected_features.add(f"item_{code.upper()}")
            if not selected_features:
                selected_features = None  # no valid codes; don't filter

        # Load causal importance and SHAP data
        causal_df = load_causal_importance(cohort, age_band)
        shap_df = load_shap_importance(cohort, age_band)

        # When no user selection, restrict to SHAP/FFA important features (top 500 by combined importance)
        if selected_features is None and (not causal_df.empty or not shap_df.empty):
            causal_col = "causal_importance" if "causal_importance" in (causal_df.columns if not causal_df.empty else []) else causal_df.columns[1] if not causal_df.empty and len(causal_df.columns) > 1 else None
            shap_col = "shap_importance" if not shap_df.empty and "shap_importance" in shap_df.columns else (shap_df.columns[1] if not shap_df.empty and len(shap_df.columns) > 1 else None)
            merged = []
            if not causal_df.empty and causal_col:
                merged.append(causal_df[["feature", causal_col]].rename(columns={causal_col: "importance"}))
            if not shap_df.empty and shap_col:
                merged.append(shap_df[["feature", shap_col]].rename(columns={shap_col: "importance"}))
            if merged:
                combined = pd.concat(merged, ignore_index=True)
                combined = combined.groupby("feature", as_index=False)["importance"].max()
                combined = combined.sort_values("importance", ascending=False).head(500)
                selected_features = set(combined["feature"].astype(str).tolist())

        if selected_features and not causal_df.empty:
            causal_df = causal_df[causal_df["feature"].isin(selected_features)].copy()
        if selected_features and not shap_df.empty:
            shap_df = shap_df[shap_df["feature"].isin(selected_features)].copy()

        # Format for frontend (top 20 each)
        causal_factors = []
        if not causal_df.empty:
            causal_df = causal_df.sort_values("causal_importance", ascending=False).head(20)
            causal_factors = causal_df.apply(
                lambda row: {"feature": row["feature"], "importance": float(row["causal_importance"])},
                axis=1
            ).tolist()

        shap_importance = []
        if not shap_df.empty:
            shap_df = shap_df.sort_values("shap_importance", ascending=False).head(20)
            shap_importance = shap_df.apply(
                lambda row: {"feature": row["feature"], "importance": float(row["shap_importance"])},
                axis=1
            ).tolist()

        return _response(200, {
            "causal_factors": causal_factors,
            "shap_importance": shap_importance,
            "interactions": [],
            "filtered_by_codes": bool(selected_features),
        })
    except Exception as e:
        return _response(500, {"error": str(e)})


def handle_visualizations_dtw(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    GET /visualizations/dtw?cohort=...&age_band=...

    Returns DTW image paths and, when DTW feature data exists in S3, chart data for:
    - routine_comparison: outcome rate by trajectory intensity (proxy for routine vs non-routine)
    - high_risk_trajectories: outcome rate by trajectory archetype (quartiles)
    """
    try:
        params = event.get("queryStringParameters") or {}
        cohort = params.get("cohort")
        age_band = params.get("age_band")

        if not cohort or not age_band:
            return _response(400, {"error": "cohort and age_band parameters required"})

        age_band_fname = age_band.replace("-", "_")

        # Construct S3 paths for DTW visualization images
        base_s3_path = f"gold/feature_importance/{cohort}/{age_band}/plots"
        payload = {
            "overview_image": f"s3://{S3_BUCKET}/{base_s3_path}/dtw_trajectory_analysis_{cohort}_{age_band_fname}.png",
            "sample_trajectories_image": f"s3://{S3_BUCKET}/{base_s3_path}/dtw_sample_trajectories_{cohort}_{age_band_fname}.png",
            "metrics": {},
        }

        # Load DTW features from S3 to prepopulate routine vs non-routine and high-risk trajectory visuals
        dtw_df = load_dtw_features(cohort, age_band)
        if not dtw_df.empty:
            routine = _compute_dtw_routine_comparison(dtw_df)
            if routine:
                payload["routine_comparison"] = routine
            high_risk = _compute_dtw_high_risk_trajectories(dtw_df)
            if high_risk:
                payload["high_risk_trajectories"] = high_risk

        return _response(200, payload)
    except Exception as e:
        return _response(500, {"error": str(e)})


def handle_visualizations_fpgrowth(event: Dict[str, Any]) -> Dict[str, Any]:
    """GET /visualizations/fpgrowth?cohort=...&age_band=...&item_type=..."""
    try:
        params = event.get("queryStringParameters") or {}
        cohort = params.get("cohort")
        age_band = params.get("age_band")
        item_type = params.get("item_type", "drug_name")
        
        if not cohort or not age_band:
            return _response(400, {"error": "cohort and age_band parameters required"})
        
        age_band_fname = age_band.replace("-", "_")
        
        # Construct S3 paths for FP-Growth visualization images
        base_s3_path = f"gold/fpgrowth/{cohort}/{age_band}/plots"
        
        return _response(200, {
            "itemsets_image": f"s3://{S3_BUCKET}/{base_s3_path}/{cohort}_{age_band_fname}_train_{item_type}_top20_itemsets.png",
            "support_image": f"s3://{S3_BUCKET}/{base_s3_path}/{cohort}_{age_band_fname}_train_{item_type}_itemset_support.png",
            "network_html": f"s3://{S3_BUCKET}/{base_s3_path}/{cohort}_{age_band_fname}_train_{item_type}_network.html"
        })
    except Exception as e:
        return _response(500, {"error": str(e)})


def handle_visualizations_bupar(event: Dict[str, Any]) -> Dict[str, Any]:
    """GET /visualizations/bupar?cohort=...&age_band=..."""
    try:
        params = event.get("queryStringParameters") or {}
        cohort = params.get("cohort")
        age_band = params.get("age_band")
        
        if not cohort or not age_band:
            return _response(400, {"error": "cohort and age_band parameters required"})
        
        age_band_fname = age_band.replace("-", "_")
        
        # Construct S3 paths for BupaR visualization images
        base_s3_path = f"gold/feature_importance/{cohort}/{age_band}/plots"
        
        return _response(200, {
            "activity_frequency_image": f"s3://{S3_BUCKET}/{base_s3_path}/{cohort}_{age_band_fname}_overall_activity_frequency.png",
            "pre_target_frequency_image": f"s3://{S3_BUCKET}/{base_s3_path}/{cohort}_{age_band_fname}_pre_f1120_activity_frequency.png",
            "post_target_frequency_image": f"s3://{S3_BUCKET}/{base_s3_path}/{cohort}_{age_band_fname}_post_f1120_activity_frequency.png",
            "gantt_image": f"s3://{S3_BUCKET}/{base_s3_path}/{cohort}_{age_band_fname}_gantt.png",
            "pre_target_gantt_image": f"s3://{S3_BUCKET}/{base_s3_path}/{cohort}_{age_band_fname}_pre_f1120_gantt.png",
            "post_target_gantt_image": f"s3://{S3_BUCKET}/{base_s3_path}/{cohort}_{age_band_fname}_post_f1120_gantt.png",
            "sequence_image": f"s3://{S3_BUCKET}/{base_s3_path}/{cohort}_{age_band_fname}_activity_sequence_top.png",
            "milestones_image": f"s3://{S3_BUCKET}/{base_s3_path}/{cohort}_{age_band_fname}_activity_milestones_gantt.png"
        })
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


def load_dtw_features(cohort: str, age_band: str) -> pd.DataFrame:
    """Load DTW feature CSV from S3 (gold/feature_engineering/6_dtw)."""
    age_band_fname = age_band.replace("-", "_")
    s3_key = f"gold/feature_engineering/6_dtw/{cohort}/{age_band}/dtw_features_{cohort}_{age_band_fname}.csv"
    try:
        obj = s3_client.get_object(Bucket=S3_BUCKET, Key=s3_key)
        df = pd.read_csv(BytesIO(obj["Body"].read()))
        return df
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code")
        if code in ("NoSuchKey", "404", "NotFound"):
            return pd.DataFrame()
        raise


def _compute_dtw_routine_comparison(df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """
    Outcome rate by trajectory intensity (event count) as proxy for routine vs non-routine.
    Buckets: Low / Medium / High by trajectory_length tertiles.
    """
    if df.empty or "target" not in df.columns or "trajectory_length" not in df.columns:
        return None
    col = "trajectory_length"
    df = df[[col, "target"]].dropna()
    if len(df) < 10:
        return None
    q1, q2 = df[col].quantile(0.33), df[col].quantile(0.67)

    def bucket(x):
        if x <= q1:
            return "Low (fewer events)"
        if x <= q2:
            return "Medium"
        return "High (more events)"

    df = df.copy()
    df["bucket"] = df[col].apply(bucket)
    agg = df.groupby("bucket", as_index=False).agg(target_rate=("target", "mean"), n=("target", "count"))
    order = ["Low (fewer events)", "Medium", "High (more events)"]
    agg = agg.set_index("bucket").reindex([b for b in order if b in agg.index]).reset_index()
    agg = agg.dropna(subset=["target_rate"])
    if agg.empty or agg["n"].sum() == 0:
        return None
    return {
        "x": agg["bucket"].astype(str).tolist(),
        "y": [float(round(v, 4)) for v in agg["target_rate"]],
        "type": "bar",
        "x_label": "Trajectory intensity (event count)",
        "y_label": "Target outcome rate",
    }


def _compute_dtw_high_risk_trajectories(df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """
    Target outcome rate by trajectory archetype (quartiles of DTW min distance or trajectory length).
    """
    if df.empty or "target" not in df.columns:
        return None
    col = "dtw_min_distance" if "dtw_min_distance" in df.columns else "trajectory_length"
    if col not in df.columns:
        return None
    df = df[["target", col]].dropna()
    if len(df) < 10:
        return None
    df = df.copy()
    try:
        df["q"] = pd.qcut(df[col], q=4, labels=["Q1 (closest)", "Q2", "Q3", "Q4 (furthest)"], duplicates="drop")
    except (ValueError, TypeError):
        return None
    agg = df.groupby("q", as_index=False).agg(target_rate=("target", "mean"), n=("target", "count"))
    if agg.empty or agg["n"].sum() == 0:
        return None
    return {
        "x": [str(v) for v in agg["q"]],
        "y": [float(round(v, 4)) for v in agg["target_rate"]],
        "type": "bar",
        "x_label": "Trajectory archetype (by DTW distance)" if col == "dtw_min_distance" else "Trajectory archetype (by length)",
        "y_label": "Target outcome rate",
    }


if __name__ == "__main__":
    # Local testing
    test_event = {
        "httpMethod": "GET",
        "path": "/metadata",
        "queryStringParameters": {"cohort": "opioid_ed"},
    }
    print(json.dumps(lambda_handler(test_event, None), indent=2))

