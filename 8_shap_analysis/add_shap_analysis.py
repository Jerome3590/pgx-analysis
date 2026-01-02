#!/usr/bin/env python3
"""
Add SHAP (SHapley Additive exPlanations) analysis to feature importance results.

This provides row-level feature importance, showing which specific drug combinations
drive outcomes for individual patients.

Usage:
    python 3_feature_importance/add_shap_analysis.py \
        --cohort non_opioid_ed \
        --age-band 65-74 \
        --method xgboost \
        --n-samples 1000  # Optional: sample patients for faster computation
"""

import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from py_helpers.feature_importance_utils import load_cohort_data
from py_helpers.feature_importance_model_utils import train_xgboost, train_catboost
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logger.warning("SHAP not installed. Install with: pip install shap")


def compute_shap_values(
    model,
    X_test: pd.DataFrame,
    method: str = 'xgboost',
    n_samples: int = None,
    max_display: int = 20
):
    """
    Compute SHAP values for row-level feature importance.
    
    Args:
        model: Trained model
        X_test: Test features DataFrame
        method: Model type ('xgboost', 'catboost')
        n_samples: Number of samples to use (None = use all)
        max_display: Max features to display in summary
        
    Returns:
        shap_values: SHAP values array
        shap_explainer: SHAP explainer object
    """
    if not SHAP_AVAILABLE:
        raise ImportError("SHAP library not available. Install with: pip install shap")
    
    # Sample if requested (for faster computation on large datasets)
    if n_samples and len(X_test) > n_samples:
        logger.info(f"Sampling {n_samples} patients from {len(X_test)} for SHAP computation")
        sample_idx = np.random.choice(len(X_test), size=n_samples, replace=False)
        X_test_sample = X_test.iloc[sample_idx].copy()
    else:
        X_test_sample = X_test.copy()
    
    logger.info(f"Computing SHAP values for {len(X_test_sample)} patients...")
    
    # Create explainer based on model type
    if method == 'xgboost':
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test_sample)
        
        # For binary classification, get positive class SHAP values
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Positive class
            
    elif method == 'catboost':
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test_sample)
        
        # For binary classification, get positive class SHAP values
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
    else:
        raise ValueError(f"SHAP not supported for method: {method}")
    
    logger.info(f"SHAP values computed: shape {shap_values.shape}")
    
    return shap_values, explainer, X_test_sample


def analyze_drug_combinations_shap(
    shap_values: np.ndarray,
    X_test: pd.DataFrame,
    feature_names: list,
    patient_ids: pd.Series = None,
    top_k: int = 10
):
    """
    Analyze which drug combinations drive outcomes for specific patients.
    
    Args:
        shap_values: SHAP values array (n_patients, n_features)
        X_test: Test features DataFrame
        feature_names: List of feature names
        patient_ids: Patient IDs (optional)
        top_k: Number of top patients to analyze
        
    Returns:
        DataFrame with patient-level drug combination analysis
    """
    # Get drug features (assuming they start with 'item_' or are drug names)
    drug_features = [f for f in feature_names if 'item_' in f.lower() or any(
        drug_term in f.lower() for drug_term in ['drug', 'medication', 'prescription']
    )]
    
    if not drug_features:
        logger.warning("No drug features found. Using all features.")
        drug_features = feature_names
    
    drug_feature_indices = [feature_names.index(f) for f in drug_features if f in feature_names]
    
    # Calculate total SHAP contribution from drugs for each patient
    drug_shap_sums = shap_values[:, drug_feature_indices].sum(axis=1)
    
    # Get top patients by drug SHAP contribution
    top_patient_indices = np.argsort(drug_shap_sums)[-top_k:][::-1]
    
    results = []
    for idx in top_patient_indices:
        patient_shap = shap_values[idx, :]
        patient_features = X_test.iloc[idx]
        
        # Get top contributing drugs for this patient
        drug_contributions = []
        for feat_idx in drug_feature_indices:
            feat_name = feature_names[feat_idx]
            shap_val = patient_shap[feat_idx]
            feat_value = patient_features[feat_name]
            
            if feat_value > 0:  # Drug is present
                drug_contributions.append({
                    'feature': feat_name,
                    'shap_value': shap_val,
                    'feature_value': feat_value
                })
        
        # Sort by SHAP value
        drug_contributions.sort(key=lambda x: abs(x['shap_value']), reverse=True)
        
        # Extract drug names (remove 'item_' prefix if present)
        drug_names = [
            d['feature'].replace('item_', '').replace('DRUG:', '').strip()
            for d in drug_contributions[:10]  # Top 10 drugs
        ]
        
        results.append({
            'patient_index': idx,
            'patient_id': patient_ids.iloc[idx] if patient_ids is not None else idx,
            'total_drug_shap': drug_shap_sums[idx],
            'top_drugs': ', '.join(drug_names),
            'drug_contributions': drug_contributions[:5]  # Top 5
        })
    
    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(description="Add SHAP analysis to feature importance")
    parser.add_argument("--cohort", required=True, help="Cohort name (e.g., non_opioid_ed)")
    parser.add_argument("--age-band", required=True, help="Age band (e.g., 65-74)")
    parser.add_argument("--method", default="xgboost", choices=["xgboost", "catboost"],
                       help="Model method")
    parser.add_argument("--n-samples", type=int, default=None,
                       help="Number of patients to sample for SHAP (None = use all)")
    parser.add_argument("--output-dir", default="3_feature_importance/outputs",
                       help="Output directory")
    
    args = parser.parse_args()
    
    if not SHAP_AVAILABLE:
        logger.error("SHAP library not available. Install with: pip install shap")
        sys.exit(1)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data (you'll need to adapt this to your data loading function)
    logger.info(f"Loading data for {args.cohort} / {args.age_band}...")
    # TODO: Adapt this to your actual data loading
    # data = load_cohort_data(args.cohort, args.age_band)
    
    logger.info("SHAP analysis script ready. Adapt data loading to your pipeline.")
    logger.info("This script provides the framework for computing SHAP values.")
    logger.info("You'll need to:")
    logger.info("1. Load your trained model")
    logger.info("2. Load test data")
    logger.info("3. Call compute_shap_values()")
    logger.info("4. Call analyze_drug_combinations_shap()")
    
    # Example usage (commented out - adapt to your pipeline):
    """
    # Load model and data
    model = load_trained_model(...)
    X_test, y_test = load_test_data(...)
    
    # Compute SHAP values
    shap_values, explainer, X_test_sample = compute_shap_values(
        model, X_test, method=args.method, n_samples=args.n_samples
    )
    
    # Analyze drug combinations
    drug_analysis = analyze_drug_combinations_shap(
        shap_values, X_test_sample, feature_names=X_test.columns.tolist()
    )
    
    # Save results
    output_path = output_dir / f"{args.cohort}_{args.age_band}_shap_analysis.csv"
    drug_analysis.to_csv(output_path, index=False)
    logger.info(f"Saved SHAP analysis to {output_path}")
    """


if __name__ == "__main__":
    main()

