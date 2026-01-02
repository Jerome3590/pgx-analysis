#!/usr/bin/env python3
"""
Extract, aggregate, and scale feature importances from the final trained model.

This script:
1. Loads the trained final model
2. Extracts feature importances
3. Aggregates and scales them (similar to Step 3 feature importance)
4. Saves results to CSV

Usage:
    python extract_final_feature_importance.py --cohort-name opioid_ed --age-band 0-12
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import warnings

warnings.filterwarnings("ignore")

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def extract_feature_importance(
    project_root: Path,
    cohort_name: str,
    age_band: str,
) -> None:
    """Extract, aggregate, and scale feature importances from final model."""
    
    age_band_fname = age_band.replace("-", "_")
    
    # Load trained model
    model_path = (
        project_root
        / "8_final_model"
        / "outputs"
        / cohort_name
        / age_band_fname
        / "models"
        / f"{cohort_name}_{age_band_fname}_final_model.joblib"
    )
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    print(f"[INFO] Loading model from {model_path}")
    model = joblib.load(model_path)
    
    # Check if model is XGBoost Booster and get the underlying classifier if needed
    if isinstance(model, xgb.core.Booster):
        print("[INFO] Model is XGBoost Booster - extracting feature importances directly")
    elif hasattr(model, 'get_booster'):
        # XGBClassifier - get the booster
        print("[INFO] Model is XGBClassifier - using get_booster()")
        model = model.get_booster()
    
    # Load feature table to get feature names (use no_leakage version)
    feature_table_path = (
        project_root
        / "8_final_model"
        / "outputs"
        / cohort_name
        / age_band_fname
        / f"{cohort_name}_{age_band_fname}_train_final_features_no_leakage.csv"
    )
    
    # Fallback to regular version if no_leakage doesn't exist
    if not feature_table_path.exists():
        feature_table_path = (
            project_root
            / "8_final_model"
            / "outputs"
            / cohort_name
            / age_band_fname
            / f"{cohort_name}_{age_band_fname}_train_final_features.csv"
        )
    
    if not feature_table_path.exists():
        raise FileNotFoundError(f"Feature table not found: {feature_table_path}")
    
    print(f"[INFO] Loading feature table from {feature_table_path}")
    df = pd.read_csv(feature_table_path)
    
    # Prepare X (same preprocessing as training)
    X = df.drop(columns=["mi_person_key", "target"], errors="ignore")
    
    # Drop datetime columns
    datetime_cols = ["target_time", "first_time"]
    cols_to_drop = [c for c in datetime_cols if c in X.columns]
    if cols_to_drop:
        X = X.drop(columns=cols_to_drop, errors='ignore')
    
    # Handle infinite and NaN values
    X = X.replace([float('inf'), float('-inf')], 0)
    X = X.fillna(0)
    
    # Ensure feature order matches model
    if hasattr(model, 'feature_names_in_'):
        # XGBoost models have feature_names_in_
        feature_names = model.feature_names_in_
        X = X[feature_names]
    elif hasattr(model, 'feature_names_'):
        # CatBoost models have feature_names_
        feature_names = model.feature_names_
        X = X[feature_names]
    else:
        # Fallback: use column names
        feature_names = X.columns.tolist()
    
    print(f"[INFO] Extracting feature importances for {len(feature_names)} features")
    
    # Extract feature importances based on model type
    if isinstance(model, xgb.core.Booster):
        # XGBoost Booster object (from xgb.train())
        importances_dict = model.get_score(importance_type='gain')
        print(f"[INFO] Found {len(importances_dict)} features with importance scores")
        
        # Convert to array in feature order
        # XGBoost Booster uses feature names as keys, not f0, f1, etc.
        feature_names_list = feature_names.tolist() if hasattr(feature_names, 'tolist') else list(feature_names)
        importances = np.array([importances_dict.get(fname, 0.0) for fname in feature_names_list])
        
        # Debug output
        if len(importances) > 0:
            print(f"[INFO] Non-zero importances: {(importances > 0).sum()}")
            print(f"[INFO] Max importance: {importances.max():.6f}")
    elif hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'get_feature_importance'):
        # CatBoost
        importances = model.get_feature_importance()
    else:
        raise ValueError(f"Unknown model type: {type(model)}")
    
    # Create DataFrame with feature importances
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    })
    
    # Sort by importance (descending)
    importance_df = importance_df.sort_values('importance', ascending=False).reset_index(drop=True)
    
    # Scale importances to 0-100 (similar to Step 3)
    max_importance = importance_df['importance'].max()
    if max_importance > 0:
        importance_df['importance_scaled'] = (importance_df['importance'] / max_importance) * 100
    else:
        importance_df['importance_scaled'] = 0
    
    # Add cumulative importance
    importance_df['cumulative_importance'] = importance_df['importance_scaled'].cumsum()
    
    # Add rank
    importance_df['rank'] = range(1, len(importance_df) + 1)
    
    # Reorder columns
    importance_df = importance_df[['rank', 'feature', 'importance', 'importance_scaled', 'cumulative_importance']]
    
    print(f"\n[INFO] Feature Importance Summary:")
    print(f"  Total features: {len(importance_df)}")
    print(f"  Max importance: {max_importance:.6f}")
    print(f"  Top 10 features:")
    print(importance_df.head(10).to_string(index=False))
    
    # Save results to main outputs folder
    output_dir = (
        project_root
        / "8_final_model"
        / "outputs"
        / cohort_name
        / age_band_fname
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / f"{cohort_name}_{age_band_fname}_final_feature_importance.csv"
    
    print(f"\n[INFO] Saving feature importances to {output_path}")
    importance_df.to_csv(output_path, index=False)
    print(f"[INFO] Saved {len(importance_df)} features")
    
    # Also save top N features (e.g., top 50)
    top_n = 50
    top_features_path = output_dir / f"{cohort_name}_{age_band_fname}_final_feature_importance_top_{top_n}.csv"
    importance_df.head(top_n).to_csv(top_features_path, index=False)
    print(f"[INFO] Saved top {top_n} features to {top_features_path}")
    
    # Also save aggregated/scaled version (similar to Step 3 format)
    aggregated_path = output_dir / f"{cohort_name}_{age_band_fname}_final_feature_importance_aggregated_scaled.csv"
    aggregated_df = importance_df.copy()
    aggregated_df['importance_normalized'] = importance_df['importance_scaled'] / 100.0  # Convert back to 0-1
    aggregated_df['importance_scaled'] = aggregated_df['importance_normalized'] * 100.0  # Keep scaled version
    aggregated_df = aggregated_df[['rank', 'feature', 'importance', 'importance_normalized', 'importance_scaled', 'cumulative_importance']]
    aggregated_df.to_csv(aggregated_path, index=False)
    print(f"[INFO] Saved aggregated/scaled version to {aggregated_path}")
    
    # Print summary statistics
    print(f"\n[INFO] Feature Importance Statistics:")
    print(f"  Mean importance: {importance_df['importance'].mean():.6f}")
    print(f"  Median importance: {importance_df['importance'].median():.6f}")
    print(f"  Std importance: {importance_df['importance'].std():.6f}")
    print(f"  Features with importance > 0: {(importance_df['importance'] > 0).sum()}")
    print(f"  Features accounting for 80% importance: {(importance_df['cumulative_importance'] <= 80).sum()}")
    print(f"  Features accounting for 90% importance: {(importance_df['cumulative_importance'] <= 90).sum()}")
    
    print("\n[INFO] Done.")


def main():
    parser = argparse.ArgumentParser(
        description="Extract feature importances from final model"
    )
    parser.add_argument(
        "--cohort-name",
        type=str,
        default="opioid_ed",
        help="Cohort name (e.g., opioid_ed)",
    )
    parser.add_argument(
        "--age-band",
        type=str,
        default="0-12",
        help="Age band (e.g., 0-12)",
    )
    parser.add_argument(
        "--project-root",
        type=str,
        default=".",
        help="Project root path (default: current directory)",
    )
    
    args = parser.parse_args()
    
    project_root = Path(args.project_root).resolve()
    extract_feature_importance(
        project_root=project_root,
        cohort_name=args.cohort_name,
        age_band=args.age_band,
    )


if __name__ == "__main__":
    main()

