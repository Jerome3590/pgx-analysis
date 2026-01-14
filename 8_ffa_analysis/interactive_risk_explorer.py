#!/usr/bin/env python3
"""
Interactive Risk Explorer with Plotly
Creates an interactive dashboard with sliders to explore how feature changes affect risk predictions.
"""

import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import joblib
import warnings
warnings.filterwarnings('ignore')

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
AGE_BAND = "0-12"
AGE_BAND_FNAME = AGE_BAND.replace("-", "_")

# Paths
MODEL_BASE = PROJECT_ROOT / '8_final_model' / 'outputs' / COHORT_NAME / AGE_BAND_FNAME
DATA_PATH = MODEL_BASE / f'{COHORT_NAME}_{AGE_BAND_FNAME}_train_final_features_no_leakage.csv'
CAUSAL_ANALYSIS_DIR = (
    PROJECT_ROOT
    / "7_ffa_analysis"
    / "outputs"
    / COHORT_NAME
    / AGE_BAND_FNAME
    / "causal_analysis"
)
OUTPUT_DIR = (
    PROJECT_ROOT
    / "7_ffa_analysis"
    / "outputs"
    / COHORT_NAME
    / AGE_BAND_FNAME
    / "interactive"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Configuration
CONFIG = {
    'top_k_features': 10,  # Number of features to include in sliders
    'baseline_sample_size': 50,  # Number of baseline instances to use
    'target_class': 1,
    'random_seed': 1997,
}


def load_data() -> Tuple[pd.DataFrame, pd.Series]:
    """Load the training data."""
    logger.info(f"Loading data from: {DATA_PATH}")
    
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Data file not found: {DATA_PATH}")
    
    data = pd.read_csv(DATA_PATH)
    
    # Separate features and target
    target_col = 'target'
    if target_col not in data.columns:
        raise ValueError(f"Target column '{target_col}' not found in data")
    
    y = data[target_col]
    X = data.drop(columns=[target_col])
    
    logger.info(f"Loaded {len(X)} samples, {len(X.columns)} features")
    return X, y


def load_models() -> Dict[str, any]:
    """Load all available models."""
    models = {}
    
    # Try to load the best model
    model_path = MODEL_BASE / 'models' / f'{COHORT_NAME}_{AGE_BAND_FNAME}_final_model.joblib'
    
    if model_path.exists():
        logger.info(f"Loading model from: {model_path}")
        best_model = joblib.load(model_path)
        models['best_model'] = best_model
        logger.info("Model loaded successfully")
    else:
        logger.warning("No saved model found")
    
    # Try to load individual models if they exist
    for model_type in ['catboost', 'xgboost', 'xgboost_rf']:
        model_path = MODEL_BASE / 'models' / f'{COHORT_NAME}_{AGE_BAND_FNAME}_{model_type}_model.joblib'
        if model_path.exists():
            logger.info(f"Loading {model_type} model...")
            models[model_type] = joblib.load(model_path)
    
    if not models:
        raise FileNotFoundError("No models found. Please train models first.")
    
    logger.info(f"Loaded {len(models)} model(s)")
    return models


def get_top_features() -> List[str]:
    """Load top features from causal analysis results."""
    causal_csv = CAUSAL_ANALYSIS_DIR / 'causal_importance_probability_method.csv'
    
    if causal_csv.exists():
        df = pd.read_csv(causal_csv)
        top_features = df.head(CONFIG['top_k_features'])['feature'].tolist()
        logger.info(f"Loaded {len(top_features)} top features from causal analysis")
        return top_features
    else:
        logger.warning("Causal analysis results not found. Will use all features.")
        return []


def get_feature_ranges(X: pd.DataFrame, features: List[str]) -> Dict[str, Dict]:
    """Get min, max, median, and current values for each feature."""
    ranges = {}
    
    for feat in features:
        if feat not in X.columns:
            continue
        
        feat_values = X[feat].dropna()
        if len(feat_values) == 0:
            continue
        
        ranges[feat] = {
            'min': float(feat_values.min()),
            'max': float(feat_values.max()),
            'median': float(feat_values.median()),
            'mean': float(feat_values.mean()),
            'std': float(feat_values.std()) if len(feat_values) > 1 else 0.0,
        }
    
    return ranges


def predict_proba(model, X: pd.DataFrame, model_type: Optional[str] = None) -> np.ndarray:
    """Get probability predictions from a model."""
    try:
        # Detect model type if not provided
        if model_type is None:
            model_class_name = type(model).__name__.lower()
            if 'catboost' in model_class_name:
                model_type = 'catboost'
            elif 'xgboost' in model_class_name or 'xgb' in model_class_name:
                model_type = 'xgboost'
        
        # Try to use model-specific prediction functions
        if model_type == 'catboost':
            try:
                from feature_importance_model_utils import predict_proba_catboost
                probs = predict_proba_catboost(model, X)
                if len(probs.shape) == 2:
                    return probs[:, CONFIG['target_class']] if probs.shape[1] > CONFIG['target_class'] else probs[:, 1]
                return probs
            except (ImportError, AttributeError):
                # Fallback to direct predict_proba
                if hasattr(model, 'predict_proba'):
                    probs = model.predict_proba(X)
                    if len(probs.shape) == 2:
                        return probs[:, CONFIG['target_class']] if probs.shape[1] > CONFIG['target_class'] else probs[:, 1]
                    return probs
        
        if model_type in ['xgboost', 'xgboost_rf']:
            try:
                from feature_importance_model_utils import predict_proba_xgboost
                probs = predict_proba_xgboost(model, X)
                if len(probs.shape) == 2:
                    return probs[:, CONFIG['target_class']] if probs.shape[1] > CONFIG['target_class'] else probs[:, 1]
                return probs
            except (ImportError, AttributeError, Exception):
                # Fallback to standard predict_proba
                if hasattr(model, 'predict_proba'):
                    probs = model.predict_proba(X)
                    if len(probs.shape) == 2:
                        return probs[:, CONFIG['target_class']] if probs.shape[1] > CONFIG['target_class'] else probs[:, 1]
                    return probs
        
        # Fallback to standard predict_proba
        if hasattr(model, 'predict_proba'):
            probs = model.predict_proba(X)
            if len(probs.shape) == 2:
                return probs[:, CONFIG['target_class']] if probs.shape[1] > CONFIG['target_class'] else probs[:, 1]
            return probs
        elif hasattr(model, 'predict'):
            # For models without predict_proba, use predict
            preds = model.predict(X)
            return preds.astype(float)
    except Exception as e:
        logger.error(f"Error in prediction: {e}")
        return np.zeros(len(X))
    
    return np.zeros(len(X))


def get_weighted_prediction(models: Dict[str, any], X: pd.DataFrame, model_weights: Optional[Dict[str, float]] = None) -> np.ndarray:
    """Get weighted average prediction across all models."""
    if model_weights is None:
        # Equal weights
        model_weights = {k: 1.0 / len(models) for k in models.keys()}
    
    all_predictions = []
    total_weight = 0.0
    
    # Determine model type from model name or try to infer
    for model_name, model in models.items():
        weight = model_weights.get(model_name, 1.0 / len(models))
        
        # Try to determine model type
        model_type = None
        if 'catboost' in model_name.lower():
            model_type = 'catboost'
        elif 'xgboost' in model_name.lower():
            model_type = 'xgboost_rf' if 'rf' in model_name.lower() else 'xgboost'
        elif 'best_model' in model_name.lower():
            # Try to detect from model class
            model_class_name = type(model).__name__.lower()
            if 'catboost' in model_class_name:
                model_type = 'catboost'
            elif 'xgboost' in model_class_name:
                model_type = 'xgboost'
        
        pred = predict_proba(model, X, model_type=model_type)
        all_predictions.append(pred * weight)
        total_weight += weight
    
    if total_weight > 0:
        return np.sum(all_predictions, axis=0) / total_weight
    else:
        return np.mean(all_predictions, axis=0) if all_predictions else np.zeros(len(X))


def get_model_weights() -> Dict[str, float]:
    """Get model weights from analysis summary."""
    weights = {}
    total_coverage = 0.0
    
    for model_type in ['catboost', 'xgboost', 'xgboost_rf']:
        summary_path = (
            PROJECT_ROOT
            / "7_ffa_analysis"
            / "outputs"
            / COHORT_NAME
            / AGE_BAND_FNAME
            / model_type
            / "analysis_summary.json"
        )
        if summary_path.exists():
            try:
                with open(summary_path, 'r') as f:
                    summary = json.load(f)
                    coverage = summary.get('coverage_rate', 0.0)
                    weights[model_type] = coverage
                    total_coverage += coverage
            except Exception as e:
                logger.warning(f"Could not load weights for {model_type}: {e}")
    
    # Normalize weights
    if total_coverage > 0:
        weights = {k: v / total_coverage for k, v in weights.items()}
    else:
        # Equal weights if no coverage data
        weights = {k: 1.0 / len(weights) for k in weights.keys()} if weights else {}
    
    # Add best_model weight if it exists
    if 'best_model' not in weights and weights:
        # Distribute best_model weight equally
        avg_weight = np.mean(list(weights.values())) if weights else 1.0 / 3
        weights['best_model'] = avg_weight
    
    logger.info(f"Model weights: {weights}")
    return weights


def create_interactive_dashboard(X: pd.DataFrame, y: pd.Series, models: Dict[str, any], 
                                 top_features: List[str], feature_ranges: Dict[str, Dict],
                                 model_weights: Dict[str, float]):
    """Create interactive Plotly dashboard with sliders."""
    logger.info("Creating interactive dashboard...")
    
    # Get baseline sample
    mask = (y == CONFIG['target_class'])
    X_baseline = X[mask].sample(n=min(CONFIG['baseline_sample_size'], len(X[mask])), 
                                random_state=CONFIG['random_seed']).reset_index(drop=True)
    
    # Get baseline predictions
    baseline_preds = get_weighted_prediction(models, X_baseline, model_weights)
    
    # Create figure with subplots
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            'Risk Distribution: Baseline vs Modified',
            'Risk Change by Feature',
            'Individual Risk Trajectories',
            'Feature Value vs Risk',
            'Combined Feature Effects',
            'Risk Summary Statistics'
        ),
        specs=[[{"type": "histogram"}, {"type": "bar"}],
               [{"type": "scatter"}, {"type": "scatter"}],
               [{"type": "bar"}, {"type": "box"}]],
        vertical_spacing=0.12,
        horizontal_spacing=0.15
    )
    
    # Initialize modified data (start with baseline)
    X_modified = X_baseline.copy()
    
    # Create sliders
    sliders = []
    steps = []
    
    # Create initial traces
    # 1. Risk distribution
    fig.add_trace(
        go.Histogram(
            x=baseline_preds,
            name='Baseline Risk',
            marker_color='lightblue',
            opacity=0.7,
            nbinsx=20
        ),
        row=1, col=1
    )
    
    # 2. Risk change by feature (will be updated)
    fig.add_trace(
        go.Bar(
            x=top_features[:5],
            y=[0] * min(5, len(top_features)),
            name='Risk Change',
            marker_color='coral',
            showlegend=False
        ),
        row=1, col=2
    )
    
    # 3. Individual trajectories (sample of 10)
    sample_indices = list(range(min(10, len(X_baseline))))
    for idx in sample_indices:
        fig.add_trace(
            go.Scatter(
                x=[0],
                y=[baseline_preds[idx]],
                mode='lines+markers',
                name=f'Instance {idx}',
                showlegend=False,
                line=dict(width=1, color='gray'),
                marker=dict(size=4)
            ),
            row=2, col=1
        )
    
    # 4. Feature value vs risk (placeholder)
    fig.add_trace(
        go.Scatter(
            x=X_baseline[top_features[0]] if top_features else [0],
            y=baseline_preds[:len(X_baseline)],
            mode='markers',
            name='Risk',
            marker=dict(color=baseline_preds[:len(X_baseline)], 
                       colorscale='RdYlGn_r', 
                       size=5,
                       showscale=True,
                       colorbar=dict(title="Risk", x=1.15)),
            showlegend=False
        ),
        row=2, col=2
    )
    
    # 5. Combined effects (placeholder)
    fig.add_trace(
        go.Bar(
            x=['Current'],
            y=[np.mean(baseline_preds)],
            name='Mean Risk',
            marker_color='green',
            showlegend=False
        ),
        row=3, col=1
    )
    
    # 6. Summary statistics (placeholder)
    fig.add_trace(
        go.Box(
            y=baseline_preds,
            name='Baseline',
            marker_color='lightblue',
            showlegend=False
        ),
        row=3, col=2
    )
    
    # Create sliders for each feature
    for i, feat in enumerate(top_features):
        if feat not in feature_ranges:
            continue
        
        feat_range = feature_ranges[feat]
        min_val = feat_range['min']
        max_val = feat_range['max']
        median_val = feat_range['median']
        
        # Create step for this feature
        step = {
            'args': [
                [feat],  # Feature name
                [median_val],  # Default value
            ],
            'label': feat,
            'method': 'restyle'
        }
        steps.append(step)
        
        # Create slider
        slider = {
            'active': 0,
            'currentvalue': {
                'prefix': f'{feat}: ',
                'xanchor': 'right'
            },
            'pad': {'t': 50},
            'steps': [{
                'args': [
                    {'visible': [True] * len(fig.data)},
                    {'title': f'Feature: {feat}'}
                ],
                'label': 'Reset',
                'method': 'update'
            }],
            'min': min_val,
            'max': max_val,
            'step': (max_val - min_val) / 100,
            'value': median_val,
            'visible': True
        }
        sliders.append(slider)
    
    # Update layout
    fig.update_layout(
        title={
            'text': 'Interactive Risk Explorer: How Features Affect Risk Predictions',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20, 'color': 'darkblue'}
        },
        height=1200,
        showlegend=True,
        sliders=sliders[:10] if len(sliders) > 10 else sliders,  # Limit to 10 sliders
    )
    
    # Update axes labels
    fig.update_xaxes(title_text="Risk Score", row=1, col=1)
    fig.update_yaxes(title_text="Frequency", row=1, col=1)
    
    fig.update_xaxes(title_text="Feature", row=1, col=2)
    fig.update_yaxes(title_text="Risk Change", row=1, col=2)
    
    fig.update_xaxes(title_text="Step", row=2, col=1)
    fig.update_yaxes(title_text="Risk Score", row=2, col=1)
    
    fig.update_xaxes(title_text=top_features[0] if top_features else "Feature Value", row=2, col=2)
    fig.update_yaxes(title_text="Risk Score", row=2, col=2)
    
    fig.update_xaxes(title_text="Configuration", row=3, col=1)
    fig.update_yaxes(title_text="Mean Risk", row=3, col=1)
    
    fig.update_xaxes(title_text="", row=3, col=2)
    fig.update_yaxes(title_text="Risk Score", row=3, col=2)
    
    # Save as HTML
    html_path = OUTPUT_DIR / 'interactive_risk_explorer.html'
    fig.write_html(str(html_path))
    logger.info(f"Saved interactive dashboard to: {html_path}")
    
    # Also create a simpler version with Plotly Express for easier interaction
    create_simplified_dashboard(X_baseline, baseline_preds, models, top_features, 
                               feature_ranges, model_weights)


def create_dropdown_dashboard(X_baseline: pd.DataFrame, baseline_preds: np.ndarray,
                              models: Dict[str, any], top_features: List[str],
                              risk_changes: Dict, model_weights: Dict[str, float]):
    """Create a dashboard with dropdown menus for feature selection."""
    logger.info("Creating dropdown dashboard...")
    
    # Create figure with dropdowns
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Risk Distribution Comparison',
            'Feature Value vs Risk Change',
            'Intervention Effects',
            'Risk Change Summary'
        ),
        specs=[[{"type": "histogram"}, {"type": "scatter"}],
               [{"type": "bar"}, {"type": "box"}]],
        vertical_spacing=0.15,
        horizontal_spacing=0.15
    )
    
    # Add baseline distribution
    fig.add_trace(
        go.Histogram(
            x=baseline_preds,
            name='Baseline',
            marker_color='lightblue',
            opacity=0.7,
            nbinsx=25,
            showlegend=True
        ),
        row=1, col=1
    )
    
    # Create dropdown menu for feature selection
    dropdown_buttons = []
    
    for feat in top_features[:10]:
        if feat not in risk_changes:
            continue
        
        # Create traces for this feature's interventions
        interventions = risk_changes[feat]['interventions']
        
        # Risk change scatter
        intervention_names = list(interventions.keys())
        intervention_values = [interventions[i]['value'] for i in intervention_names]
        mean_changes = [interventions[i]['mean_change'] for i in intervention_names]
        
        # Add traces (initially hidden except first)
        visible_list = [feat == top_features[0]] * (len(fig.data) + 2)
        
        # Histogram for modified risk
        fig.add_trace(
            go.Histogram(
                x=baseline_preds,  # Placeholder, will be updated
                name=f'{feat} Modified',
                marker_color='coral',
                opacity=0.7,
                nbinsx=25,
                visible=(feat == top_features[0])
            ),
            row=1, col=1
        )
        
        # Scatter for risk change
        fig.add_trace(
            go.Scatter(
                x=intervention_values,
                y=mean_changes,
                mode='lines+markers',
                name=f'{feat} Risk Change',
                marker=dict(size=10, color='red'),
                line=dict(width=2),
                visible=(feat == top_features[0])
            ),
            row=1, col=2
        )
        
        # Button for this feature
        dropdown_buttons.append({
            'label': feat,
            'method': 'update',
            'args': [
                {'visible': [True] + [feat == f for f in top_features[:10] for _ in range(2)]},
                {'title': f'Feature: {feat}'}
            ]
        })
    
    # Update layout with dropdown
    fig.update_layout(
        title={
            'text': 'Interactive Risk Explorer: Select Feature to Explore',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18}
        },
        height=900,
        showlegend=True,
        updatemenus=[{
            'buttons': dropdown_buttons,
            'direction': 'down',
            'showactive': True,
            'x': 0.1,
            'xanchor': 'left',
            'y': 1.15,
            'yanchor': 'top'
        }]
    )
    
    # Update axes
    fig.update_xaxes(title_text="Risk Score", row=1, col=1)
    fig.update_yaxes(title_text="Frequency", row=1, col=1)
    
    fig.update_xaxes(title_text="Feature Value", row=1, col=2)
    fig.update_yaxes(title_text="Mean Risk Change", row=1, col=2)
    
    # Save
    html_path = OUTPUT_DIR / 'dropdown_dashboard.html'
    fig.write_html(str(html_path))
    logger.info(f"Saved dropdown dashboard to: {html_path}")


def compute_risk_changes(X_baseline: pd.DataFrame, baseline_preds: np.ndarray,
                         models: Dict[str, any], top_features: List[str],
                         feature_ranges: Dict[str, Dict], model_weights: Dict[str, float]) -> Dict:
    """Pre-compute risk changes for different feature values."""
    logger.info("Computing risk changes for different feature values...")
    
    results = {}
    
    for feat in top_features[:10]:  # Limit to top 10
        if feat not in feature_ranges or feat not in X_baseline.columns:
            continue
        
        feat_range = feature_ranges[feat]
        min_val = feat_range['min']
        max_val = feat_range['max']
        median_val = feat_range['median']
        mean_val = feat_range['mean']
        
        # Test different intervention values
        test_values = {
            'min': min_val,
            'q25': np.percentile(X_baseline[feat].dropna(), 25),
            'median': median_val,
            'mean': mean_val,
            'q75': np.percentile(X_baseline[feat].dropna(), 75),
            'max': max_val,
            'zero': 0.0 if min_val <= 0 <= max_val else median_val,
            'increase': min(max_val, mean_val + 2 * feat_range['std']) if feat_range['std'] > 0 else max_val
        }
        
        risk_changes = {}
        for val_name, val in test_values.items():
            X_modified = X_baseline.copy()
            X_modified[feat] = val
            modified_preds = get_weighted_prediction(models, X_modified, model_weights)
            
            # Calculate change statistics
            risk_changes[val_name] = {
                'value': float(val),
                'mean_risk': float(np.mean(modified_preds)),
                'mean_change': float(np.mean(modified_preds - baseline_preds)),
                'median_change': float(np.median(modified_preds - baseline_preds)),
                'std_change': float(np.std(modified_preds - baseline_preds)),
            }
        
        results[feat] = {
            'baseline_mean': float(np.mean(baseline_preds)),
            'interventions': risk_changes,
            'range': {'min': min_val, 'max': max_val, 'median': median_val}
        }
    
    return results


def create_feature_slider_dashboard(X: pd.DataFrame, y: pd.Series, models: Dict[str, any],
                                   top_features: List[str], feature_ranges: Dict[str, Dict],
                                   model_weights: Dict[str, float]):
    """Create a dashboard focused on individual and combined feature effects."""
    logger.info("Creating feature slider dashboard...")
    
    # Get baseline sample
    mask = (y == CONFIG['target_class'])
    X_baseline = X[mask].sample(n=min(CONFIG['baseline_sample_size'], len(X[mask])), 
                                random_state=CONFIG['random_seed']).reset_index(drop=True)
    
    baseline_preds = get_weighted_prediction(models, X_baseline, model_weights)
    
    # Pre-compute risk changes
    risk_changes = compute_risk_changes(X_baseline, baseline_preds, models, top_features, 
                                       feature_ranges, model_weights)
    
    # Create figure with multiple views
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=(
            'Risk Distribution: Baseline vs Modified',
            'Feature Impact on Risk (Mean Change)',
            'Individual Feature Effects',
            'Risk Change Distribution',
            'Combined Feature Effects',
            'Risk Summary Statistics'
        ),
        specs=[[{"type": "histogram"}, {"type": "bar"}],
               [{"type": "scatter"}, {"type": "box"}],
               [{"type": "bar"}, {"type": "box"}]],
        vertical_spacing=0.12,
        horizontal_spacing=0.15
    )
    
    # 1. Risk distribution - baseline
    fig.add_trace(
        go.Histogram(
            x=baseline_preds,
            name='Baseline Risk',
            marker_color='rgba(100, 200, 255, 0.7)',
            nbinsx=25,
            showlegend=True
        ),
        row=1, col=1
    )
    
    # 2. Feature impact bar chart
    if risk_changes:
        features_list = list(risk_changes.keys())[:10]
        mean_changes = [risk_changes[f]['interventions']['increase']['mean_change'] 
                       for f in features_list]
        
        fig.add_trace(
            go.Bar(
                x=features_list,
                y=mean_changes,
                name='Risk Change (Increase)',
                marker=dict(
                    color=mean_changes,
                    colorscale='RdYlGn',
                    showscale=True,
                    colorbar=dict(title="Risk Change", x=1.02)
                ),
                showlegend=False
            ),
            row=1, col=2
        )
    
    # 3. Individual feature effects - show risk change for different interventions
    if risk_changes and top_features:
        feat = top_features[0]
        if feat in risk_changes:
            interventions = risk_changes[feat]['interventions']
            intervention_names = list(interventions.keys())
            mean_risks = [interventions[i]['mean_risk'] for i in intervention_names]
            
            fig.add_trace(
                go.Scatter(
                    x=intervention_names,
                    y=mean_risks,
                    mode='lines+markers',
                    name=f'{feat} Effects',
                    marker=dict(size=10, color='coral'),
                    line=dict(width=2),
                    showlegend=True
                ),
                row=2, col=1
            )
    
    # 4. Risk change distribution box plot
    if risk_changes:
        all_changes = []
        feature_labels = []
        for feat, data in list(risk_changes.items())[:5]:
            changes = []
            for interv_name, interv_data in data['interventions'].items():
                if interv_name != 'median':  # Compare to baseline
                    X_modified = X_baseline.copy()
                    X_modified[feat] = interv_data['value']
                    modified_preds = get_weighted_prediction(models, X_modified, model_weights)
                    changes.extend(modified_preds - baseline_preds)
            
            if changes:
                all_changes.extend(changes)
                feature_labels.extend([feat] * len(changes))
        
        if all_changes:
            fig.add_trace(
                go.Box(
                    y=all_changes,
                    name='Risk Changes',
                    marker_color='lightgreen',
                    showlegend=False
                ),
                row=2, col=2
            )
    
    # 5. Combined effects - show what happens when we modify multiple features
    if risk_changes and len(top_features) >= 3:
        # Test combinations: modify top 3 features together
        X_combined_low = X_baseline.copy()
        X_combined_high = X_baseline.copy()
        
        for feat in top_features[:3]:
            if feat in risk_changes:
                X_combined_low[feat] = risk_changes[feat]['range']['min']
                X_combined_high[feat] = risk_changes[feat]['range']['max']
        
        preds_low = get_weighted_prediction(models, X_combined_low, model_weights)
        preds_high = get_weighted_prediction(models, X_combined_high, model_weights)
        
        fig.add_trace(
            go.Bar(
                x=['Baseline', 'All Low', 'All High'],
                y=[np.mean(baseline_preds), np.mean(preds_low), np.mean(preds_high)],
                name='Combined Effects',
                marker_color=['lightblue', 'green', 'red'],
                showlegend=False
            ),
            row=3, col=1
        )
    
    # 6. Risk summary statistics
    fig.add_trace(
        go.Box(
            y=baseline_preds,
            name='Baseline Risk',
            marker_color='lightblue',
            showlegend=False
        ),
        row=3, col=2
    )
    
    # Create animation frames for different feature values
    frames = []
    slider_steps = []
    
    for i, feat in enumerate(top_features[:8]):
        if feat not in risk_changes:
            continue
        
        interventions = risk_changes[feat]['interventions']
        intervention_names = list(interventions.keys())
        
        # Create frame for each intervention
        for interv_name in intervention_names:
            frame_data = []
            
            # Update histogram with modified predictions
            X_modified = X_baseline.copy()
            X_modified[feat] = interventions[interv_name]['value']
            modified_preds = get_weighted_prediction(models, X_modified, model_weights)
            
            frame_data.append(go.Histogram(x=modified_preds, name='Modified Risk'))
            
            frames.append(go.Frame(
                data=frame_data,
                name=f"{feat}_{interv_name}"
            ))
            
            slider_steps.append({
                'args': [
                    [f"{feat}_{interv_name}"],
                    {'frame': {'duration': 300, 'redraw': True},
                     'mode': 'immediate',
                     'transition': {'duration': 300}}
                ],
                'label': f'{feat}: {interv_name}',
                'method': 'animate'
            })
    
    # Create slider
    slider = dict(
        active=0,
        currentvalue={'prefix': 'Feature: '},
        pad={'t': 50},
        steps=slider_steps
    )
    
    # Update layout
    fig.update_layout(
        title={
            'text': 'Interactive Feature Risk Explorer: How Features Affect Risk',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 18}
        },
        height=1200,
        sliders=[slider],
        showlegend=True,
        updatemenus=[{
            'type': 'buttons',
            'direction': 'right',
            'x': 0.5,
            'xanchor': 'center',
            'y': -0.1,
            'yanchor': 'top',
            'buttons': [{
                'label': 'Play',
                'method': 'animate',
                'args': [None, {
                    'frame': {'duration': 500, 'redraw': True},
                    'fromcurrent': True,
                    'transition': {'duration': 300}
                }]
            }, {
                'label': 'Reset',
                'method': 'animate',
                'args': [[None], {
                    'frame': {'duration': 0, 'redraw': True},
                    'mode': 'immediate',
                    'transition': {'duration': 0}
                }]
            }]
        }]
    )
    
    # Add frames
    if frames:
        fig.frames = frames
    
    # Update axes
    fig.update_xaxes(title_text="Risk Score", row=1, col=1)
    fig.update_yaxes(title_text="Frequency", row=1, col=1)
    
    fig.update_xaxes(title_text="Feature", row=1, col=2, tickangle=45)
    fig.update_yaxes(title_text="Mean Risk Change", row=1, col=2)
    
    fig.update_xaxes(title_text="Intervention Type", row=2, col=1)
    fig.update_yaxes(title_text="Mean Risk", row=2, col=1)
    
    fig.update_xaxes(title_text="", row=2, col=2)
    fig.update_yaxes(title_text="Risk Change", row=2, col=2)
    
    fig.update_xaxes(title_text="Configuration", row=3, col=1)
    fig.update_yaxes(title_text="Mean Risk", row=3, col=1)
    
    fig.update_xaxes(title_text="", row=3, col=2)
    fig.update_yaxes(title_text="Risk Score", row=3, col=2)
    
    # Save
    html_path = OUTPUT_DIR / 'feature_slider_dashboard.html'
    fig.write_html(str(html_path))
    logger.info(f"Saved feature slider dashboard to: {html_path}")
    
    # Also create a simpler interactive version with dropdowns
    create_dropdown_dashboard(X_baseline, baseline_preds, models, top_features, 
                             risk_changes, model_weights)


def main():
    """Main function to create interactive dashboards."""
    logger.info("=" * 80)
    logger.info("Interactive Risk Explorer")
    logger.info(f"Cohort: {COHORT_NAME}, Age Band: {AGE_BAND}")
    logger.info("=" * 80)
    
    try:
        # Load data
        X, y = load_data()
        
        # Load models
        models = load_models()
        
        # Get model weights
        model_weights = get_model_weights()
        
        # Get top features
        top_features = get_top_features()
        if not top_features:
            # Fallback: use features with highest variance
            feature_vars = X.var().sort_values(ascending=False)
            top_features = feature_vars.head(CONFIG['top_k_features']).index.tolist()
            logger.info(f"Using top {len(top_features)} features by variance")
        
        logger.info(f"Top features: {top_features}")
        
        # Get feature ranges
        feature_ranges = get_feature_ranges(X, top_features)
        logger.info(f"Feature ranges calculated for {len(feature_ranges)} features")
        
        # Create dashboards
        create_feature_slider_dashboard(X, y, models, top_features, feature_ranges, model_weights)
        
        logger.info("=" * 80)
        logger.info("Interactive dashboards created successfully!")
        logger.info(f"Output directory: {OUTPUT_DIR}")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"Error creating interactive dashboard: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()

