#!/usr/bin/env python3
"""
Combined Causal Analysis Using All Three Models - Dual Approach

This script performs comprehensive causal analysis using TWO approaches:
1. EXPLAINER-BASED (FFA): Measures how explanations change with interventions
2. PROBABILITY-BASED: Measures how prediction probabilities change with interventions

Both approaches use all three models (CatBoost, XGBoost, XGBoost RF) and aggregate results.
"""

import sys
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "7_ffa_analysis"))
sys.path.insert(0, str(PROJECT_ROOT / "py_helpers"))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
COHORT_NAME = "opioid_ed"
AGE_BAND = "13-24"
AGE_BAND_FNAME = AGE_BAND.replace("-", "_")

# Paths (updated to use 6_final_model outputs)
MODEL_BASE = PROJECT_ROOT / '6_final_model' / 'outputs' / COHORT_NAME / AGE_BAND_FNAME
DATA_PATH = MODEL_BASE / f'{COHORT_NAME}_{AGE_BAND_FNAME}_train_final_features_no_leakage.csv'
OUTPUT_DIR = (
    PROJECT_ROOT
    / "7_ffa_analysis"
    / "outputs"
    / COHORT_NAME
    / AGE_BAND_FNAME
    / "causal_analysis"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Analysis configuration
CAUSAL_CONFIG = {
    'target_class': 1,
    'sample_size': 100,  # Number of instances to analyze (reduced for speed)
    'top_k_features': 15,  # Number of top features to analyze
    'intervention_types': ['remove', 'median', 'zero', 'increase'],  # Types of interventions
    'random_seed': 1997,
    'use_explainer_method': False,  # Use FFA explainer-based approach (SLOW - disabled by default)
    'use_probability_method': True,  # Use probability-based approach
    'train_models': False,  # Set to True to train models fresh (slower but more accurate)
    'explainer_sample_size': 50,  # Smaller sample for explainer method (very slow)
}

# Model parameters (matching final model training)
MODEL_PARAMS = {
    'catboost': {
        'iterations': 500,
        'learning_rate': 0.1,
        'depth': 6,
        'verbose': False,
        'random_seed': 42,
    },
    'xgboost': {
        'max_depth': 6,
        'learning_rate': 0.1,
        'n_estimators': 500,
        'subsample': 1.0,
        'colsample_bytree': 1.0,
        'random_seed': 42,
    },
    'xgboost_rf': {
        'max_depth': 6,
        'learning_rate': 0.1,
        'n_estimators': 500,
        'subsample': 0.8,
        'random_seed': 42,
    },
}


def load_data() -> Tuple[pd.DataFrame, pd.Series]:
    """Load the training data."""
    logger.info(f"Loading data from: {DATA_PATH}")
    
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Data file not found: {DATA_PATH}")
    
    # Load full dataset
    data = pd.read_csv(DATA_PATH)
    
    # Separate features and target
    target_cols = ['target', 'is_target_case']
    target_col = None
    for col in target_cols:
        if col in data.columns:
            target_col = col
            break
    
    if not target_col:
        raise ValueError("No target column found in data")
    
    y = data[target_col]
    X = data.drop(target_col, axis=1)
    
    # Drop non-feature columns
    drop_cols = ['mi_person_key', 'target_time', 'first_time']
    X = X.drop(columns=[c for c in drop_cols if c in X.columns])
    
    # Handle missing values
    X = X.replace([float('inf'), float('-inf')], 0)
    X = X.fillna(0)
    
    logger.info(f"Loaded {len(X)} samples, {len(X.columns)} features")
    return X, y


def get_model_weights() -> Dict[str, float]:
    """Get model weights based on FFA analysis coverage rates."""
    model_weights = {}
    
    for model_type in ['catboost', 'xgboost', 'xgboost_rf']:
        summary_path = (
            PROJECT_ROOT
            / "8_ffa_analysis"
            / "outputs"
            / COHORT_NAME
            / AGE_BAND_FNAME
            / model_type
            / "analysis_summary.json"
        )
        if summary_path.exists():
            with open(summary_path, 'r') as f:
                summary = json.load(f)
            coverage_rate = summary.get('explanations_with_conditions', 0) / max(summary.get('total_explanations', 1), 1)
            model_weights[model_type] = coverage_rate
        else:
            model_weights[model_type] = 1.0
    
    # Normalize weights
    total_weight = sum(model_weights.values())
    if total_weight > 0:
        model_weights = {k: v / total_weight for k, v in model_weights.items()}
    else:
        model_weights = {k: 1.0 / len(model_weights) for k in model_weights.keys()}
    
    logger.info(f"Model weights: {model_weights}")
    return model_weights


def get_feature_importance_features() -> List[str]:
    """Get top features from combined weighted importance."""
    importance_path = (
        PROJECT_ROOT
        / "8_ffa_analysis"
        / "outputs"
        / COHORT_NAME
        / AGE_BAND_FNAME
        / "visualizations"
        / "combined_weighted_feature_importance.csv"
    )
    
    if importance_path.exists():
        df = pd.read_csv(importance_path)
        return df.head(CAUSAL_CONFIG['top_k_features'])['feature'].tolist()
    else:
        # Fallback: get from individual model results
        features = set()
        for model_type in ['catboost', 'xgboost', 'xgboost_rf']:
            importance_path = (
                PROJECT_ROOT
                / "8_ffa_analysis"
                / "outputs"
                / COHORT_NAME
                / AGE_BAND_FNAME
                / model_type
                / "feature_importance_axp.parquet"
            )
            if importance_path.exists():
                df = pd.read_parquet(importance_path)
                features.update(df.head(CAUSAL_CONFIG['top_k_features'])['feature'].tolist())
        
        return list(features)[:CAUSAL_CONFIG['top_k_features']]


def create_intervention(X: pd.DataFrame, feature: str, intervention_type: str) -> pd.DataFrame:
    """Create a counterfactual dataset with an intervention on a feature."""
    X_intervened = X.copy()
    
    if feature not in X.columns:
        return X_intervened
    
    feature_values = X[feature]
    
    if intervention_type == 'remove' or intervention_type == 'median':
        # Set to median (neutral value)
        X_intervened[feature] = feature_values.median()
    elif intervention_type == 'zero':
        # Set to zero
        X_intervened[feature] = 0
    elif intervention_type == 'increase':
        # Increase by one standard deviation
        std_val = feature_values.std()
        if std_val > 0:
            X_intervened[feature] = feature_values + std_val
        else:
            X_intervened[feature] = feature_values + 1
    elif intervention_type == 'decrease':
        # Decrease by one standard deviation
        std_val = feature_values.std()
        if std_val > 0:
            X_intervened[feature] = feature_values - std_val
        else:
            X_intervened[feature] = feature_values - 1
    
    return X_intervened


def load_or_train_models(X: pd.DataFrame, y: pd.Series) -> Dict[str, any]:
    """Load trained models or train them fresh."""
    models = {}
    
    if CAUSAL_CONFIG['train_models']:
        logger.info("Training models fresh...")
        from feature_importance_model_utils import train_catboost, train_xgboost, train_xgboost_rf
        
        # Train all three models
        logger.info("Training CatBoost...")
        models['catboost'] = train_catboost(X, y, MODEL_PARAMS['catboost'])
        
        logger.info("Training XGBoost...")
        models['xgboost'] = train_xgboost(X, y, MODEL_PARAMS['xgboost'])
        
        logger.info("Training XGBoost RF...")
        models['xgboost_rf'] = train_xgboost_rf(X, y, MODEL_PARAMS['xgboost_rf'])
        
        logger.info("All models trained successfully")
    else:
        # Try to load from joblib (only best model is saved)
        import joblib
        model_path = MODEL_BASE / 'models' / f'{COHORT_NAME}_{AGE_BAND_FNAME}_final_model.joblib'
        
        if model_path.exists():
            logger.info(f"Loading model from: {model_path}")
            best_model = joblib.load(model_path)
            # Note: Only one model is saved, so we'll use explainers for the others
            models['best_model'] = best_model
            logger.info("Model loaded (will use explainers for other models)")
        else:
            logger.warning("No saved model found - will use explainer-based analysis only")
    
    return models


def get_probability_predictions(models: Dict[str, any], X: pd.DataFrame, model_type: str) -> Optional[np.ndarray]:
    """Get probability predictions from a model."""
    try:
        if model_type == 'catboost' and 'catboost' in models:
            from feature_importance_model_utils import predict_proba_catboost
            probs = predict_proba_catboost(models['catboost'], X)
            if len(probs.shape) == 2 and probs.shape[1] > CAUSAL_CONFIG['target_class']:
                return probs[:, CAUSAL_CONFIG['target_class']]
            elif len(probs.shape) == 2:
                return probs[:, 1] if probs.shape[1] == 2 else probs[:, 0]
            else:
                return probs
        elif model_type in ['xgboost', 'xgboost_rf'] and model_type in models:
            from feature_importance_model_utils import predict_proba_xgboost
            probs = predict_proba_xgboost(models[model_type], X)
            if len(probs.shape) == 2 and probs.shape[1] > CAUSAL_CONFIG['target_class']:
                return probs[:, CAUSAL_CONFIG['target_class']]
            elif len(probs.shape) == 2:
                return probs[:, 1] if probs.shape[1] == 2 else probs[:, 0]
            else:
                return probs
        elif 'best_model' in models:
            # Use best model for all if only one is available
            model = models['best_model']
            if hasattr(model, 'predict_proba'):
                probs = model.predict_proba(X)
                if len(probs.shape) == 2:
                    if probs.shape[1] > CAUSAL_CONFIG['target_class']:
                        return probs[:, CAUSAL_CONFIG['target_class']]
                    else:
                        return probs[:, 1] if probs.shape[1] == 2 else probs[:, 0]
                else:
                    return probs
    except Exception as e:
        logger.warning(f"Error getting predictions for {model_type}: {e}")
    
    return None


def causal_analysis_explainer_method(X: pd.DataFrame, y: pd.Series, 
                                     feature_importance_features: List[str],
                                     model_weights: Dict[str, float]) -> pd.DataFrame:
    """Causal analysis using explainer-based (FFA) approach."""
    logger.info("=" * 80)
    logger.info("CAUSAL ANALYSIS: Explainer-Based (FFA) Method")
    logger.info("=" * 80)
    
    # Filter to target class
    mask = (y == CAUSAL_CONFIG['target_class'])
    X_class = X[mask].reset_index(drop=True)
    y_class = y[mask]
    
    # Use smaller sample for explainer method (it's very slow)
    explainer_sample_size = CAUSAL_CONFIG.get('explainer_sample_size', 50)
    if len(X_class) > explainer_sample_size:
        X_class = X_class.sample(n=explainer_sample_size, 
                                 random_state=CAUSAL_CONFIG['random_seed']).reset_index(drop=True)
        y_class = y_class[X_class.index]
    
    logger.info(f"Analyzing {len(X_class)} instances (explainer method uses smaller sample)")
    
    # Load explainers
    explainers = {}
    for model_type in ['catboost', 'xgboost', 'xgboost_rf']:
        try:
            model_json_path = MODEL_BASE / 'final_model_json' / f'{COHORT_NAME}_{AGE_BAND_FNAME}_final_model_{model_type}.json'
            if not model_json_path.exists():
                continue
            
            if model_type == 'catboost':
                from catboost_axp_explainer import CatBoostSymbolicExplainer, PathConfig
                path_config = PathConfig(
                    model_path=str(model_json_path),
                    data_dir=str(DATA_PATH.parent),
                    output_dir=str(OUTPUT_DIR),
                    tree_rules_path=None,
                    age_band=AGE_BAND
                )
                explainer = CatBoostSymbolicExplainer(path_config)
            else:
                from xgboost_axp_explainer import XGBoostSymbolicExplainer, PathConfig
                path_config = PathConfig(
                    model_path=str(model_json_path),
                    data_dir=str(DATA_PATH.parent),
                    output_dir=str(OUTPUT_DIR),
                    tree_rules_path=None,
                    age_band=AGE_BAND
                )
                explainer = XGBoostSymbolicExplainer(path_config)
            
            with open(model_json_path, 'r') as f:
                model_json = json.load(f)
            explainer.model_json = model_json
            explainer.fit_from_model_json(model_json)
            explainers[model_type] = explainer
            logger.info(f"Loaded {model_type} explainer")
            
        except Exception as e:
            logger.warning(f"Could not load {model_type} explainer: {e}")
    
    if not explainers:
        logger.warning("No explainers available - skipping explainer-based analysis")
        return pd.DataFrame()
    
    # Filter features
    available_features = [f for f in feature_importance_features if f in X_class.columns]
    logger.info(f"Analyzing {len(available_features)} features")
    
    causal_results = []
    
    for feat_idx, feat_name in enumerate(available_features):
        logger.info(f"Feature {feat_idx+1}/{len(available_features)}: {feat_name}")
        
        # Get baseline explanations (simplified - just count rule matches)
        baseline_counts = {}
        for model_type, explainer in explainers.items():
            try:
                # Use a faster approach: check rule matches directly instead of full explanations
                # Sample a few instances for speed
                sample_size = min(10, len(X_class))
                X_sample = X_class.head(sample_size)
                y_sample = y_class.head(sample_size)
                
                baseline_explanations = explainer.explain_dataset(
                    X_sample,
                    predictions=y_sample.values,
                    return_df=True,
                    show_progress=False,
                    n_jobs=1
                )
                baseline_count = sum(
                    1 for axp in baseline_explanations['axp'] 
                    if axp and feat_name in str(axp)
                )
                baseline_counts[model_type] = baseline_count / len(baseline_explanations) if len(baseline_explanations) > 0 else 0.0
            except Exception as e:
                logger.warning(f"Error getting baseline for {model_type}: {e}")
                baseline_counts[model_type] = 0.0
        
        # Create interventions and measure changes
        intervention_results = {}
        
        for intervention_type in CAUSAL_CONFIG['intervention_types']:
            try:
                X_intervened = create_intervention(X_class, feat_name, intervention_type)
                
                # Use same sample as baseline
                sample_size = min(10, len(X_class))
                X_intervened_sample = X_intervened.head(sample_size)
                y_sample = y_class.head(sample_size)
                
                intervened_counts = {}
                for model_type, explainer in explainers.items():
                    try:
                        intervened_explanations = explainer.explain_dataset(
                            X_intervened_sample,
                            predictions=y_sample.values,
                            return_df=True,
                            show_progress=False,
                            n_jobs=1
                        )
                        intervened_count = sum(
                            1 for axp in intervened_explanations['axp'] 
                            if axp and feat_name in str(axp)
                        )
                        intervened_counts[model_type] = intervened_count / len(intervened_explanations) if len(intervened_explanations) > 0 else 0.0
                    except Exception as e:
                        logger.warning(f"Error getting intervened prediction for {model_type}: {e}")
                        intervened_counts[model_type] = baseline_counts.get(model_type, 0.0)
                
                # Calculate weighted average change
                weighted_change = 0.0
                for model_type in explainers.keys():
                    baseline = baseline_counts.get(model_type, 0.0)
                    intervened = intervened_counts.get(model_type, 0.0)
                    change = abs(intervened - baseline)
                    weighted_change += change * model_weights.get(model_type, 1.0 / len(explainers))
                
                intervention_results[intervention_type] = weighted_change
                
            except Exception as e:
                logger.warning(f"Error with intervention {intervention_type} for {feat_name}: {e}")
                intervention_results[intervention_type] = 0.0
        
        # Calculate overall causal importance
        overall_importance = np.mean(list(intervention_results.values()))
        
        causal_results.append({
            'feature': feat_name,
            'causal_importance': overall_importance,
            'remove_effect': intervention_results.get('remove', 0.0),
            'median_effect': intervention_results.get('median', 0.0),
            'zero_effect': intervention_results.get('zero', 0.0),
            'increase_effect': intervention_results.get('increase', 0.0),
        })
    
    causal_df = pd.DataFrame(causal_results)
    causal_df = causal_df.sort_values('causal_importance', ascending=False)
    
    logger.info(f"Explainer-based analysis completed for {len(causal_df)} features")
    return causal_df


def causal_analysis_probability_method(X: pd.DataFrame, y: pd.Series,
                                       feature_importance_features: List[str],
                                       model_weights: Dict[str, float],
                                       models: Dict[str, any]) -> pd.DataFrame:
    """Causal analysis using probability-based approach."""
    logger.info("=" * 80)
    logger.info("CAUSAL ANALYSIS: Probability-Based Method")
    logger.info("=" * 80)
    
    # Filter to target class
    mask = (y == CAUSAL_CONFIG['target_class'])
    X_class = X[mask].reset_index(drop=True)
    y_class = y[mask].reset_index(drop=True)
    
    # Sample if needed
    if len(X_class) > CAUSAL_CONFIG['sample_size']:
        sample_indices = X_class.sample(n=CAUSAL_CONFIG['sample_size'], 
                                       random_state=CAUSAL_CONFIG['random_seed']).index
        X_class = X_class.loc[sample_indices].reset_index(drop=True)
        y_class = y_class.loc[sample_indices].reset_index(drop=True)
    
    logger.info(f"Analyzing {len(X_class)} instances")
    
    # Get baseline predictions from all available models
    baseline_probs = {}
    model_types_to_use = []
    
    # Try to get predictions from loaded models
    for model_type in ['catboost', 'xgboost', 'xgboost_rf']:
        probs = get_probability_predictions(models, X_class, model_type)
        if probs is not None:
            baseline_probs[model_type] = probs
            model_types_to_use.append(model_type)
    
    # If we have best_model but no specific models, use it for all types
    if not baseline_probs and 'best_model' in models:
        logger.info("Using best_model for all model types")
        probs = get_probability_predictions(models, X_class, 'best_model')
        if probs is not None:
            # Use same predictions for all model types (weighted equally)
            for model_type in ['catboost', 'xgboost', 'xgboost_rf']:
                baseline_probs[model_type] = probs
                model_types_to_use.append(model_type)
    
    # If no models available, return empty
    if not baseline_probs:
        logger.warning("No models available for probability predictions")
        return pd.DataFrame()
    
    logger.info(f"Using {len(model_types_to_use)} models for probability-based analysis")
    
    # Filter features
    available_features = [f for f in feature_importance_features if f in X_class.columns]
    logger.info(f"Analyzing {len(available_features)} features")
    
    causal_results = []
    
    for feat_idx, feat_name in enumerate(available_features):
        logger.info(f"Feature {feat_idx+1}/{len(available_features)}: {feat_name}")
        
        # Create interventions and measure probability changes
        intervention_results = {}
        
        for intervention_type in CAUSAL_CONFIG['intervention_types']:
            try:
                X_intervened = create_intervention(X_class, feat_name, intervention_type)
                
                # Get predictions after intervention
                intervened_probs = {}
                for model_type in model_types_to_use:
                    probs = get_probability_predictions(models, X_intervened, model_type)
                    if probs is not None:
                        intervened_probs[model_type] = probs
                    else:
                        # Fallback to baseline if prediction fails
                        intervened_probs[model_type] = baseline_probs.get(model_type, np.zeros(len(X_intervened)))
                
                # Calculate weighted average probability change
                weighted_change = 0.0
                for model_type in model_types_to_use:
                    baseline = baseline_probs[model_type]
                    intervened = intervened_probs.get(model_type, baseline)
                    change = np.mean(np.abs(intervened - baseline))
                    weight = model_weights.get(model_type, 1.0 / len(model_types_to_use))
                    weighted_change += change * weight
                
                intervention_results[intervention_type] = weighted_change
                
            except Exception as e:
                logger.warning(f"Error with intervention {intervention_type} for {feat_name}: {e}")
                intervention_results[intervention_type] = 0.0
        
        # Calculate overall causal importance
        overall_importance = np.mean(list(intervention_results.values()))
        
        causal_results.append({
            'feature': feat_name,
            'causal_importance': overall_importance,
            'remove_effect': intervention_results.get('remove', 0.0),
            'median_effect': intervention_results.get('median', 0.0),
            'zero_effect': intervention_results.get('zero', 0.0),
            'increase_effect': intervention_results.get('increase', 0.0),
        })
    
    causal_df = pd.DataFrame(causal_results)
    causal_df = causal_df.sort_values('causal_importance', ascending=False)
    
    logger.info(f"Probability-based analysis completed for {len(causal_df)} features")
    return causal_df


def combine_results(explainer_df: pd.DataFrame, probability_df: pd.DataFrame) -> pd.DataFrame:
    """Combine results from both methods."""
    logger.info("Combining results from both methods...")
    
    if explainer_df.empty and probability_df.empty:
        return pd.DataFrame()
    
    if explainer_df.empty:
        # Only probability method - use it as combined
        probability_df = probability_df.copy()
        probability_df['combined_importance'] = probability_df['causal_importance']
        probability_df['method'] = 'probability'
        return probability_df
    
    if probability_df.empty:
        # Only explainer method - use it as combined
        explainer_df = explainer_df.copy()
        explainer_df['combined_importance'] = explainer_df['causal_importance']
        explainer_df['method'] = 'explainer'
        return explainer_df
    
    # Combine both methods
    explainer_df = explainer_df.rename(columns={
        'causal_importance': 'explainer_importance',
        'remove_effect': 'explainer_remove',
        'median_effect': 'explainer_median',
        'zero_effect': 'explainer_zero',
        'increase_effect': 'explainer_increase',
    })
    
    probability_df = probability_df.rename(columns={
        'causal_importance': 'probability_importance',
        'remove_effect': 'probability_remove',
        'median_effect': 'probability_median',
        'zero_effect': 'probability_zero',
        'increase_effect': 'probability_increase',
    })
    
    # Merge on feature
    combined_df = explainer_df[['feature']].merge(
        probability_df[['feature']], 
        on='feature', 
        how='outer'
    )
    
    # Merge data
    combined_df = combined_df.merge(explainer_df, on='feature', how='left')
    combined_df = combined_df.merge(probability_df, on='feature', how='left')
    
    # Calculate combined importance (average of both methods, normalized)
    explainer_norm = combined_df['explainer_importance'] / (combined_df['explainer_importance'].max() + 1e-10)
    prob_norm = combined_df['probability_importance'] / (combined_df['probability_importance'].max() + 1e-10)
    combined_df['combined_importance'] = (explainer_norm.fillna(0) + prob_norm.fillna(0)) / 2
    
    combined_df = combined_df.sort_values('combined_importance', ascending=False)
    
    return combined_df


def create_radar_chart(probability_df: pd.DataFrame, explainer_df: pd.DataFrame, combined_df: pd.DataFrame):
    """Create Plotly radar chart for intervention effects."""
    logger.info("Creating radar chart for intervention effects...")
    
    # Use probability_df if available, otherwise explainer_df
    method_df = probability_df if not probability_df.empty else explainer_df
    
    if method_df.empty:
        logger.warning("No data available for radar chart")
        return
    
    # Get top features (limit to 8-10 for readability)
    top_features = method_df.head(8)
    
    if len(top_features) == 0:
        logger.warning("No features available for radar chart")
        return
    
    # Prepare data for radar chart
    feature_names = top_features['feature'].tolist()
    
    # Get intervention effects
    remove_effects = top_features['remove_effect'].tolist()
    median_effects = top_features['median_effect'].tolist()
    zero_effects = top_features['zero_effect'].tolist()
    increase_effects = top_features['increase_effect'].tolist()
    
    # Close the radar chart by repeating first value at the end
    feature_names_closed = feature_names + [feature_names[0]]
    remove_effects_closed = remove_effects + [remove_effects[0]]
    median_effects_closed = median_effects + [median_effects[0]]
    zero_effects_closed = zero_effects + [zero_effects[0]]
    increase_effects_closed = increase_effects + [increase_effects[0]]
    
    # Create radar chart
    fig = go.Figure()
    
    # Add traces for each intervention type
    fig.add_trace(go.Scatterpolar(
        r=remove_effects_closed,
        theta=feature_names_closed,
        fill='toself',
        name='Remove Effect',
        line_color='#FF6B6B',
        fillcolor='rgba(255, 107, 107, 0.3)',
        line_width=2
    ))
    
    fig.add_trace(go.Scatterpolar(
        r=median_effects_closed,
        theta=feature_names_closed,
        fill='toself',
        name='Median Effect',
        line_color='#4ECDC4',
        fillcolor='rgba(78, 205, 196, 0.3)',
        line_width=2
    ))
    
    fig.add_trace(go.Scatterpolar(
        r=zero_effects_closed,
        theta=feature_names_closed,
        fill='toself',
        name='Zero Effect',
        line_color='#45B7D1',
        fillcolor='rgba(69, 183, 209, 0.3)',
        line_width=2
    ))
    
    fig.add_trace(go.Scatterpolar(
        r=increase_effects_closed,
        theta=feature_names_closed,
        fill='toself',
        name='Increase Effect',
        line_color='#96CEB4',
        fillcolor='rgba(150, 206, 180, 0.3)',
        line_width=2
    ))
    
    # Update layout
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, max(max(remove_effects), max(median_effects), 
                            max(zero_effects), max(increase_effects)) * 1.1],
                tickfont=dict(size=10),
                title=dict(text='Effect Size', font=dict(size=12, color='black'))
            ),
            angularaxis=dict(
                tickfont=dict(size=9),
                rotation=90,
                direction='counterclockwise'
            )
        ),
        title=dict(
            text='Intervention Effects Radar Chart: Top Features',
            x=0.5,
            xanchor='center',
            font=dict(size=18, color='darkblue')
        ),
        showlegend=True,
        legend=dict(
            orientation='v',
            yanchor='top',
            y=1,
            xanchor='left',
            x=1.1,
            font=dict(size=11)
        ),
        width=1000,
        height=800,
        margin=dict(l=50, r=150, t=80, b=50)
    )
    
    # Save as HTML
    html_path = OUTPUT_DIR / 'intervention_effects_radar_chart.html'
    fig.write_html(str(html_path))
    logger.info(f"Saved radar chart to: {html_path}")
    
    # Also create a version with individual features as separate radars
    create_individual_radar_charts(method_df)


def create_individual_radar_charts(method_df: pd.DataFrame):
    """Create individual radar charts for top features."""
    logger.info("Creating individual radar charts for top features...")
    
    top_features = method_df.head(6)  # Limit to 6 for readability
    
    if len(top_features) == 0:
        return
    
    intervention_types = ['remove_effect', 'median_effect', 'zero_effect', 'increase_effect']
    intervention_labels = ['Remove', 'Median', 'Zero', 'Increase']
    
    # Create a single figure with multiple polar subplots
    fig = go.Figure()
    
    # Add each feature as a separate trace (overlayed, can toggle visibility)
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#F7DC6F', '#BB8FCE']
    
    for idx, (_, row) in enumerate(top_features.iterrows()):
        # Get effect values for this feature
        effect_values = [row[intervention] for intervention in intervention_types]
        effect_values_closed = effect_values + [effect_values[0]]
        theta_closed = intervention_labels + [intervention_labels[0]]
        
        fig.add_trace(go.Scatterpolar(
            r=effect_values_closed,
            theta=theta_closed,
            fill='toself',
            name=row['feature'][:40],  # Truncate long names
            line_color=colors[idx % len(colors)],
            fillcolor=colors[idx % len(colors)],
            line_width=2,
            opacity=0.6
        ))
    
    # Find max effect for scaling
    all_effects = []
    for _, row in top_features.iterrows():
        all_effects.extend([row[intervention] for intervention in intervention_types])
    max_effect = max(all_effects) * 1.2 if max(all_effects) > 0 else 0.01
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                range=[0, max_effect],
                title=dict(text='Effect Size', font=dict(size=12)),
                tickfont=dict(size=10)
            ),
            angularaxis=dict(
                tickfont=dict(size=10),
                rotation=90,
                direction='counterclockwise'
            )
        ),
        title=dict(
            text='Individual Feature Intervention Effects (Overlay)',
            x=0.5,
            xanchor='center',
            font=dict(size=18, color='darkblue')
        ),
        showlegend=True,
        legend=dict(
            orientation='v',
            yanchor='top',
            y=1,
            xanchor='left',
            x=1.1,
            font=dict(size=10)
        ),
        width=1000,
        height=800,
        margin=dict(l=50, r=150, t=80, b=50)
    )
    
    html_path = OUTPUT_DIR / 'individual_intervention_effects_radar_charts.html'
    fig.write_html(str(html_path))
    logger.info(f"Saved individual radar charts to: {html_path}")


def create_visualizations(explainer_df: pd.DataFrame, probability_df: pd.DataFrame, combined_df: pd.DataFrame):
    """Create comprehensive visualizations as separate charts."""
    logger.info("Creating visualizations...")
    
    # Chart 1a: Probability-Based Method - Top Features
    if not probability_df.empty:
        fig1a = plt.figure(figsize=(14, 10))
        top_prob = probability_df.head(15)
        y_pos = np.arange(len(top_prob))
        bars = plt.barh(y_pos, top_prob['causal_importance'].values, color='coral', alpha=0.7, edgecolor='black')
        plt.yticks(y_pos, top_prob['feature'].values, fontsize=10)
        plt.xlabel('Causal Importance', fontsize=12, fontweight='bold')
        plt.title('Probability-Based Method: Top 15 Features by Causal Importance', fontsize=16, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.grid(axis='x', linestyle='--', alpha=0.3)
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, top_prob['causal_importance'].values)):
            plt.text(val, i, f' {val:.4f}', va='center', fontsize=9)
        
        plt.tight_layout()
        save_path1a = OUTPUT_DIR / 'causal_analysis_probability_method_top_features.png'
        plt.savefig(save_path1a, bbox_inches='tight', facecolor='white', dpi=300)
        plt.close()
        logger.info(f"Saved: {save_path1a}")
    
    # Chart 1b: Probability-Based Method - Intervention Effects
    if not probability_df.empty:
        fig1b = plt.figure(figsize=(14, 8))
        top_features = probability_df.head(10)
        x = np.arange(len(top_features))
        width = 0.2
        
        plt.bar(x - 1.5*width, top_features['remove_effect'], width, label='Remove', alpha=0.7, color='#FF6B6B')
        plt.bar(x - 0.5*width, top_features['median_effect'], width, label='Median', alpha=0.7, color='#4ECDC4')
        plt.bar(x + 0.5*width, top_features['zero_effect'], width, label='Zero', alpha=0.7, color='#45B7D1')
        plt.bar(x + 1.5*width, top_features['increase_effect'], width, label='Increase', alpha=0.7, color='#96CEB4')
        
        plt.xticks(x, top_features['feature'].values, rotation=45, ha='right', fontsize=9)
        plt.ylabel('Effect Size', fontsize=12, fontweight='bold')
        plt.title('Probability-Based Method: Intervention Effects for Top 10 Features', fontsize=16, fontweight='bold')
        plt.legend(fontsize=10, loc='upper right')
        plt.grid(axis='y', linestyle='--', alpha=0.3)
        
        plt.tight_layout()
        save_path1b = OUTPUT_DIR / 'causal_analysis_probability_method_intervention_effects.png'
        plt.savefig(save_path1b, bbox_inches='tight', facecolor='white', dpi=300)
        plt.close()
        logger.info(f"Saved: {save_path1b}")
    
    # Chart 2: Explainer-Based Method (if available)
    if not explainer_df.empty:
        fig2, axes2 = plt.subplots(2, 1, figsize=(14, 12))
        fig2.suptitle('Causal Analysis: Explainer-Based (FFA) Method', fontsize=16, fontweight='bold')
        
        # Top features bar chart
        ax1 = axes2[0]
        top_explainer = explainer_df.head(15)
        y_pos = np.arange(len(top_explainer))
        bars = ax1.barh(y_pos, top_explainer['causal_importance'].values, color='steelblue', alpha=0.7, edgecolor='black')
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(top_explainer['feature'].values, fontsize=10)
        ax1.set_xlabel('Causal Importance', fontsize=12, fontweight='bold')
        ax1.set_title('Top 15 Features by Causal Importance', fontsize=14, fontweight='bold')
        ax1.invert_yaxis()
        ax1.grid(axis='x', linestyle='--', alpha=0.3)
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, top_explainer['causal_importance'].values)):
            ax1.text(val, i, f' {val:.4f}', va='center', fontsize=9)
        
        # Intervention effects
        ax2 = axes2[1]
        top_features = explainer_df.head(10)
        x = np.arange(len(top_features))
        width = 0.2
        
        ax2.bar(x - 1.5*width, top_features['remove_effect'], width, label='Remove', alpha=0.7, color='#FF6B6B')
        ax2.bar(x - 0.5*width, top_features['median_effect'], width, label='Median', alpha=0.7, color='#4ECDC4')
        ax2.bar(x + 0.5*width, top_features['zero_effect'], width, label='Zero', alpha=0.7, color='#45B7D1')
        ax2.bar(x + 1.5*width, top_features['increase_effect'], width, label='Increase', alpha=0.7, color='#96CEB4')
        
        ax2.set_xticks(x)
        ax2.set_xticklabels(top_features['feature'].values, rotation=45, ha='right', fontsize=9)
        ax2.set_ylabel('Effect Size', fontsize=12, fontweight='bold')
        ax2.set_title('Intervention Effects: Top 10 Features', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=10, loc='upper right')
        ax2.grid(axis='y', linestyle='--', alpha=0.3)
        
        plt.tight_layout()
        save_path2 = OUTPUT_DIR / 'causal_analysis_explainer_method.png'
        plt.savefig(save_path2, bbox_inches='tight', facecolor='white', dpi=300)
        plt.close()
        logger.info(f"Saved: {save_path2}")
    
    # Chart 3a: Combined Results - Top Features Bar Chart
    if not combined_df.empty:
        fig3a = plt.figure(figsize=(14, 10))
        top_combined = combined_df.head(15)
        y_pos = np.arange(len(top_combined))
        bars = plt.barh(y_pos, top_combined['combined_importance'].values, color='green', alpha=0.7, edgecolor='black')
        plt.yticks(y_pos, top_combined['feature'].values, fontsize=10)
        plt.xlabel('Combined Causal Importance', fontsize=12, fontweight='bold')
        plt.title('Top 15 Features by Combined Causal Importance', fontsize=16, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.grid(axis='x', linestyle='--', alpha=0.3)
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, top_combined['combined_importance'].values)):
            plt.text(val, i, f' {val:.4f}', va='center', fontsize=9)
        
        plt.tight_layout()
        save_path3a = OUTPUT_DIR / 'causal_analysis_combined_top_features.png'
        plt.savefig(save_path3a, bbox_inches='tight', facecolor='white', dpi=300)
        plt.close()
        logger.info(f"Saved: {save_path3a}")
    
    # Chart 3b: Combined Results - Method Comparison Scatter Plot
    if not explainer_df.empty and not probability_df.empty:
        fig3b = plt.figure(figsize=(12, 10))
        merged = explainer_df[['feature', 'causal_importance']].merge(
            probability_df[['feature', 'causal_importance']], 
            on='feature', 
            suffixes=('_explainer', '_prob')
        )
        
        scatter = plt.scatter(merged['causal_importance_explainer'], 
                     merged['causal_importance_prob'],
                     alpha=0.7, s=150, edgecolors='black', linewidth=1.5,
                     c=merged.index, cmap='viridis')
        
        # Add feature labels
        for idx, row in merged.iterrows():
            plt.annotate(row['feature'], 
                       (row['causal_importance_explainer'], row['causal_importance_prob']),
                       fontsize=8, alpha=0.7,
                       xytext=(5, 5), textcoords='offset points')
        
        plt.xlabel('Explainer-Based Importance', fontsize=12, fontweight='bold')
        plt.ylabel('Probability-Based Importance', fontsize=12, fontweight='bold')
        plt.title('Method Comparison: Explainer vs Probability', fontsize=16, fontweight='bold')
        plt.grid(True, linestyle='--', alpha=0.3)
        
        # Add diagonal line
        max_val = max(merged['causal_importance_explainer'].max(), 
                     merged['causal_importance_prob'].max())
        plt.plot([0, max_val], [0, max_val], 'r--', alpha=0.5, linewidth=2, label='y=x')
        plt.legend(fontsize=10)
        plt.colorbar(scatter, label='Feature Index')
        
        plt.tight_layout()
        save_path3b = OUTPUT_DIR / 'causal_analysis_combined_method_comparison.png'
        plt.savefig(save_path3b, bbox_inches='tight', facecolor='white', dpi=300)
        plt.close()
        logger.info(f"Saved: {save_path3b}")
    
    
    # Create Plotly Radar Chart for Intervention Effects
    create_radar_chart(probability_df, explainer_df, combined_df)
    
    logger.info("All visualizations created successfully!")


def main():
    """Main function to run combined causal analysis."""
    logger.info("=" * 80)
    logger.info("Combined Causal Analysis: Dual Approach")
    logger.info(f"Cohort: {COHORT_NAME}, Age Band: {AGE_BAND}")
    logger.info("=" * 80)
    
    try:
        # Load data
        X, y = load_data()
        
        # Get model weights
        model_weights = get_model_weights()
        
        # Get top features
        feature_importance_features = get_feature_importance_features()
        logger.info(f"Selected {len(feature_importance_features)} features for analysis")
        
        # Load or train models
        models = load_or_train_models(X, y)
        
        # Run explainer-based analysis
        explainer_df = pd.DataFrame()
        if CAUSAL_CONFIG['use_explainer_method']:
            try:
                logger.info("WARNING: Explainer method is very slow. This may take a long time...")
                explainer_df = causal_analysis_explainer_method(
                    X, y, feature_importance_features, model_weights
                )
                if not explainer_df.empty:
                    explainer_df.to_csv(OUTPUT_DIR / 'causal_importance_explainer_method.csv', index=False)
                    logger.info(f"Saved explainer-based results: {len(explainer_df)} features")
            except Exception as e:
                logger.error(f"Explainer-based analysis failed: {e}")
                logger.info("Continuing with probability-based method only...")
                explainer_df = pd.DataFrame()
        
        # Run probability-based analysis
        probability_df = pd.DataFrame()
        if CAUSAL_CONFIG['use_probability_method']:
            try:
                # If we don't have all models, try to train them quickly
                if len(models) < 3 and not CAUSAL_CONFIG['train_models']:
                    logger.info("Training models for probability-based analysis...")
                    from feature_importance_model_utils import train_catboost, train_xgboost, train_xgboost_rf
                    
                    # Use smaller iterations for faster training
                    quick_params = {
                        'catboost': {**MODEL_PARAMS['catboost'], 'iterations': 100},
                        'xgboost': {**MODEL_PARAMS['xgboost'], 'n_estimators': 100},
                        'xgboost_rf': {**MODEL_PARAMS['xgboost_rf'], 'n_estimators': 100},
                    }
                    
                    if 'catboost' not in models:
                        logger.info("Training CatBoost (quick mode)...")
                        models['catboost'] = train_catboost(X, y, quick_params['catboost'])
                    if 'xgboost' not in models:
                        logger.info("Training XGBoost (quick mode)...")
                        models['xgboost'] = train_xgboost(X, y, quick_params['xgboost'])
                    if 'xgboost_rf' not in models:
                        logger.info("Training XGBoost RF (quick mode)...")
                        models['xgboost_rf'] = train_xgboost_rf(X, y, quick_params['xgboost_rf'])
                
                probability_df = causal_analysis_probability_method(
                    X, y, feature_importance_features, model_weights, models
                )
                if not probability_df.empty:
                    probability_df.to_csv(OUTPUT_DIR / 'causal_importance_probability_method.csv', index=False)
                    logger.info(f"Saved probability-based results: {len(probability_df)} features")
            except Exception as e:
                logger.error(f"Probability-based analysis failed: {e}", exc_info=True)
                logger.info("Skipping probability-based method...")
                probability_df = pd.DataFrame()
        
        # Combine results
        combined_df = combine_results(explainer_df, probability_df)
        if not combined_df.empty:
            combined_df.to_csv(OUTPUT_DIR / 'causal_importance_combined.csv', index=False)
            logger.info(f"Saved combined results: {len(combined_df)} features")
        
        # Create visualizations
        create_visualizations(explainer_df, probability_df, combined_df)
        
        # Print summary
        print("\n" + "=" * 80)
        print("Causal Analysis Summary")
        print("=" * 80)
        
        if not explainer_df.empty:
            print(f"\nTop 10 Features (Explainer-Based):")
            print(explainer_df.head(10)[['feature', 'causal_importance']].to_string(index=False))
        
        if not probability_df.empty:
            print(f"\nTop 10 Features (Probability-Based):")
            print(probability_df.head(10)[['feature', 'causal_importance']].to_string(index=False))
        
        if not combined_df.empty:
            print(f"\nTop 10 Features (Combined):")
            print(combined_df.head(10)[['feature', 'combined_importance']].to_string(index=False))
        
        logger.info("=" * 80)
        logger.info("Combined causal analysis completed successfully!")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"Error in causal analysis: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
