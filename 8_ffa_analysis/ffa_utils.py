#!/usr/bin/env python3
"""
Utility functions for FFA analysis (from PHTS risk calculator).

Used by SHAP+FFA workflow for model loading and feature extraction.
"""

import json
import logging
import time
from pathlib import Path
from typing import Dict, Any

logger = logging.getLogger(__name__)


def load_model_json(model_json_path: Path) -> Dict[str, Any]:
    """
    Load model JSON file.

    Args:
        model_json_path: Path to model JSON file

    Returns:
        Dictionary containing model JSON data with normalized model_type
    """
    logger.info(f"Loading model JSON from: {model_json_path}")
    start_time = time.time()

    if not model_json_path.exists():
        logger.error(f"Model file not found: {model_json_path}")
        raise FileNotFoundError(f"Model file not found: {model_json_path}")

    logger.info(f"Reading JSON file (size: {model_json_path.stat().st_size / 1024 / 1024:.2f} MB)...")
    with open(model_json_path, "r") as f:
        model_json = json.load(f)
    logger.info(f"JSON file loaded successfully in {time.time() - start_time:.2f} seconds")

    model_type = model_json.get("model_type", "unknown")
    if model_type == "unknown":
        if "oblivious_trees" in model_json or "non_oblivious_trees" in model_json:
            model_type = "catboost"
        elif "learner" in model_json or "gradient_booster" in model_json:
            model_type = "xgboost"

    if model_type == "xgb":
        model_type = "xgboost"
    elif model_type == "xgb_rf":
        model_type = "xgboost_rf"

    model_json["model_type"] = model_type
    logger.info(f"Model type detected: {model_type}")

    if "trees" in model_json:
        logger.info(f"Found {len(model_json['trees'])} trees")
    if "feature_names" in model_json:
        logger.info(f"Found {len(model_json['feature_names'])} features")

    logger.info(f"Model loading completed in {time.time() - start_time:.2f} seconds")
    return model_json


def extract_feature_mappings(model_json: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract feature name mappings from model JSON.

    Args:
        model_json: Model JSON dictionary

    Returns:
        Dictionary with 'model_type' and 'feature_names' mapping
    """
    logger.info("Extracting feature mappings from model JSON...")
    start_time = time.time()

    model_type = model_json.get("model_type", "unknown")
    if model_type == "unknown" and (
        "oblivious_trees" in model_json or "non_oblivious_trees" in model_json
    ):
        model_type = "catboost"

    if model_type == "xgb":
        model_type = "xgboost"
    elif model_type == "xgb_rf":
        model_type = "xgboost_rf"

    model_json["model_type"] = model_type

    if model_type in ["catboost", "CatBoost"]:
        logger.info("Processing CatBoost feature mappings...")
        features_info = model_json.get("features_info", {})
        float_features = features_info.get("float_features", [])
        logger.info(f"Found {len(float_features)} float features")
        feature_names = {f["flat_feature_index"]: f["feature_id"] for f in float_features}
    else:
        logger.info("Processing XGBoost feature mappings...")
        if "feature_names" in model_json:
            feature_names = {
                i: name for i, name in enumerate(model_json["feature_names"])
            }
            logger.info(f"Found {len(feature_names)} feature names")
        else:
            logger.warning("No feature_names found in model JSON")
            feature_names = {}

    logger.info(f"Feature mapping extraction completed in {time.time() - start_time:.2f} seconds")
    return {"model_type": model_type, "feature_names": feature_names}
