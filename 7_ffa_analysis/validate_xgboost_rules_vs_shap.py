#!/usr/bin/env python3
"""
Validation: Compare XGBoost JSON-extracted rules with SHAP values

This script validates that SHAP values can be used to accurately filter and
build the rule set for causal analysis. It demonstrates that rules extracted
from XGBoost JSON align well with SHAP importance patterns, validating that
SHAP-guided rule filtering (as used in the three-set union approach) produces
meaningful results for causal analysis.

Usage:
    python validate_xgboost_rules_vs_shap.py --cohort opioid_ed --age-band 13-24
"""

import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
from scipy.stats import spearmanr, pearsonr
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from py_helpers.env_utils import get_sklearn_n_jobs

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    from xgboost_axp_explainer import XGBoostSymbolicExplainer, PathConfig
    XGBOOST_EXPLAINER_AVAILABLE = True
except ImportError:
    XGBOOST_EXPLAINER_AVAILABLE = False
    logger.error("XGBoost explainer not available")


def load_shap_importance(cohort: str, age_band: str, model_type: str = "xgboost") -> Dict[str, float]:
    """Load SHAP importance values from Step 7."""
    age_band_fname = age_band.replace("-", "_")
    
    shap_path = (
        PROJECT_ROOT
        / "8_shap_analysis"
        / "outputs"
        / cohort
        / age_band_fname
        / f"{cohort}_{age_band_fname}_shap_global_importance_{model_type}.csv"
    )
    
    if not shap_path.exists():
        raise FileNotFoundError(
            f"SHAP importance file not found: {shap_path}\n"
            f"Please run Step 7 (SHAP Analysis) first."
        )
    
    shap_df = pd.read_csv(shap_path)
    if 'feature' not in shap_df.columns or 'mean_abs_shap' not in shap_df.columns:
        raise ValueError(
            f"SHAP file missing required columns. Expected 'feature' and 'mean_abs_shap', "
            f"got: {list(shap_df.columns)}"
        )
    
    # Filter to features with importance > 0
    shap_df = shap_df[shap_df['mean_abs_shap'] > 0]
    
    # Create mapping: feature_name -> mean_abs_shap
    shap_map = dict(zip(shap_df['feature'], shap_df['mean_abs_shap'], strict=True))
    logger.info(f"Loaded SHAP importance for {len(shap_map)} features (importance > 0)")
    return shap_map


def calculate_rule_based_importance(explainer, shap_importance_map: Dict[str, float]) -> Dict[str, float]:
    """
    Calculate feature importance from rules extracted directly from JSON.
    
    For each feature, count how many rules it appears in, weighted by SHAP importance
    of features in those rules. This demonstrates how SHAP values can be used to
    filter and prioritize rules for causal analysis, showing alignment between
    JSON-extracted rules and SHAP importance patterns.
    """
    feature_rule_counts = defaultdict(int)
    feature_rule_shap_scores = defaultdict(float)
    
    # Get feature names from explainer
    feature_names = explainer.feature_names
    
    # Iterate through all rules
    for rule_id, clause in enumerate(explainer.rule_clauses):
        # Get features in this rule
        features_in_rule = set()
        for lit in clause:
            feat_idx, _, _ = explainer.id_condition_map[lit]
            feat_name = feature_names.get(feat_idx, f"f{feat_idx}")
            features_in_rule.add(feat_name)
        
        # Calculate rule's SHAP score (sum of SHAP values of features in rule)
        rule_shap_score = sum(shap_importance_map.get(feat_name, 0.0) for feat_name in features_in_rule)
        
        # For each feature in the rule, increment its count and add rule's SHAP score
        for feat_name in features_in_rule:
            feature_rule_counts[feat_name] += 1
            feature_rule_shap_scores[feat_name] += rule_shap_score
    
    # Normalize by rule count to get average SHAP score per feature
    rule_based_importance = {
        feat_name: feature_rule_shap_scores[feat_name] / max(feature_rule_counts[feat_name], 1)
        for feat_name in feature_rule_counts.keys()
    }
    
    return rule_based_importance


def compare_rule_shap_alignment(
    rule_based_importance: Dict[str, float],
    shap_importance: Dict[str, float]
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Compare rule-based importance (from JSON) with SHAP importance.
    
    Returns:
        - DataFrame with comparison metrics
        - Dictionary with correlation statistics
    """
    # Get common features
    common_features = set(rule_based_importance.keys()) & set(shap_importance.keys())
    
    if len(common_features) == 0:
        raise ValueError("No common features found between rule-based and SHAP importance")
    
    logger.info(f"Comparing {len(common_features)} common features")
    
    # Create comparison DataFrame
    comparison_data = []
    for feat in common_features:
        comparison_data.append({
            'feature': feat,
            'rule_based_importance': rule_based_importance[feat],
            'shap_importance': shap_importance[feat],
            'difference': abs(rule_based_importance[feat] - shap_importance[feat]),
            'relative_difference': abs(rule_based_importance[feat] - shap_importance[feat]) / max(shap_importance[feat], 1e-10)
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Calculate correlation statistics
    rule_values = comparison_df['rule_based_importance'].values
    shap_values = comparison_df['shap_importance'].values
    
    pearson_corr, pearson_p = pearsonr(rule_values, shap_values)
    spearman_corr, spearman_p = spearmanr(rule_values, shap_values)
    
    stats = {
        'pearson_correlation': pearson_corr,
        'pearson_p_value': pearson_p,
        'spearman_correlation': spearman_corr,
        'spearman_p_value': spearman_p,
        'mean_absolute_difference': comparison_df['difference'].mean(),
        'median_absolute_difference': comparison_df['difference'].median(),
        'mean_relative_difference': comparison_df['relative_difference'].mean(),
        'n_features': len(common_features)
    }
    
    return comparison_df, stats


def create_validation_plots(
    comparison_df: pd.DataFrame,
    stats: Dict[str, float],
    output_dir: Path
):
    """Create visualization plots comparing rule-based and SHAP importance."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Scatter plot: Rule-based vs SHAP importance
    plt.figure(figsize=(10, 8))
    plt.scatter(
        comparison_df['shap_importance'],
        comparison_df['rule_based_importance'],
        alpha=0.6,
        s=50
    )
    
    # Add diagonal line (perfect alignment)
    max_val = max(
        comparison_df['shap_importance'].max(),
        comparison_df['rule_based_importance'].max()
    )
    plt.plot([0, max_val], [0, max_val], 'r--', label='Perfect Alignment', linewidth=2)
    
    plt.xlabel('SHAP Importance (mean_abs_shap)', fontsize=12)
    plt.ylabel('Rule-Based Importance (from JSON)', fontsize=12)
    plt.title(
        f'SHAP-Guided Rule Filtering Validation\n'
        f'Pearson r={stats["pearson_correlation"]:.3f}, '
        f'Spearman ρ={stats["spearman_correlation"]:.3f}',
        fontsize=14
    )
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    scatter_path = output_dir / 'rule_shap_alignment_scatter.png'
    plt.savefig(scatter_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved scatter plot: {scatter_path}")
    plt.close()
    
    # Difference distribution
    plt.figure(figsize=(10, 6))
    plt.hist(comparison_df['relative_difference'], bins=50, alpha=0.7, edgecolor='black')
    plt.xlabel('Relative Difference (|rule - shap| / shap)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Distribution of Relative Differences\n(SHAP-Guided Rule Filtering Validation)', fontsize=14)
    plt.axvline(
        comparison_df['relative_difference'].median(),
        color='r',
        linestyle='--',
        label=f'Median: {comparison_df["relative_difference"].median():.3f}'
    )
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    diff_path = output_dir / 'rule_shap_difference_distribution.png'
    plt.savefig(diff_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved difference distribution: {diff_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Validate that SHAP values can accurately filter/build rule sets for causal analysis'
    )
    parser.add_argument('--cohort', type=str, required=True, help='Cohort name (e.g., opioid_ed)')
    parser.add_argument('--age-band', type=str, required=True, help='Age band (e.g., 13-24)')
    parser.add_argument('--model-type', type=str, default='xgboost', help='Model type (default: xgboost)')
    parser.add_argument('--output-dir', type=Path, default=None, help='Output directory (default: 7_ffa_analysis/outputs/{cohort}/{age_band}/validation)')
    
    args = parser.parse_args()
    
    age_band_fname = args.age_band.replace("-", "_")
    
    # Set up paths
    if args.output_dir is None:
        output_dir = (
            PROJECT_ROOT
            / "7_ffa_analysis"
            / "outputs"
            / args.cohort
            / age_band_fname
            / "validation"
        )
    else:
        output_dir = args.output_dir
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Validating XGBoost rule extraction for {args.cohort}/{args.age_band}")
    logger.info(f"Output directory: {output_dir}")
    
    # Load SHAP importance
    logger.info("Loading SHAP importance values...")
    shap_importance = load_shap_importance(args.cohort, args.age_band, args.model_type)
    
    # Load XGBoost model JSON
    model_json_path = (
        PROJECT_ROOT
        / "6_final_model"
        / "outputs"
        / args.cohort
        / age_band_fname
        / "final_model_json"
        / f"{args.cohort}_{age_band_fname}_final_model_{args.model_type}.json"
    )
    
    if not model_json_path.exists():
        raise FileNotFoundError(f"Model JSON not found: {model_json_path}")
    
    logger.info(f"Loading model JSON: {model_json_path}")
    with open(model_json_path, 'r') as f:
        model_json = json.load(f)
    
    # Load feature names from training data
    data_path = (
        PROJECT_ROOT
        / "6_final_model"
        / "outputs"
        / args.cohort
        / age_band_fname
        / f"{args.cohort}_{age_band_fname}_train_final_features_no_leakage.csv"
    )
    
    if not data_path.exists():
        raise FileNotFoundError(f"Training data not found: {data_path}")
    
    logger.info(f"Loading feature names from: {data_path}")
    train_df = pd.read_csv(data_path, nrows=1)  # Just read header
    feature_names = [col for col in train_df.columns if col != 'target']
    
    # Initialize explainer and extract rules
    logger.info("Initializing XGBoost explainer and extracting rules from JSON...")
    path_config = PathConfig(
        model_path=str(model_json_path),
        data_dir=str(data_path.parent),
        output_dir=str(output_dir),
        age_band=args.age_band
    )
    
    explainer = XGBoostSymbolicExplainer(path_config, shap_importance_map=shap_importance)
    explainer.feature_names = {i: name for i, name in enumerate(feature_names)}
    explainer.model_json = model_json
    explainer.fit_from_model_json(model_json)
    
    logger.info(f"Extracted {len(explainer.rule_clauses)} rules from JSON")
    
    # Calculate rule-based importance
    logger.info("Calculating rule-based feature importance...")
    rule_based_importance = calculate_rule_based_importance(explainer, shap_importance)
    
    # Compare with SHAP
    logger.info("Comparing rule-based importance with SHAP importance...")
    comparison_df, stats = compare_rule_shap_alignment(rule_based_importance, shap_importance)
    
    # Save comparison results
    comparison_path = output_dir / 'rule_shap_comparison.csv'
    comparison_df.to_csv(comparison_path, index=False)
    logger.info(f"Saved comparison results: {comparison_path}")
    
    # Save statistics
    stats_path = output_dir / 'validation_statistics.json'
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    logger.info(f"Saved validation statistics: {stats_path}")
    
    # Create plots
    logger.info("Creating validation plots...")
    create_validation_plots(comparison_df, stats, output_dir)
    
    # Print summary
    print("\n" + "="*80)
    print("SHAP-Guided Rule Filtering Validation Summary")
    print("="*80)
    print("Validating that SHAP values can accurately filter/build rule sets for causal analysis")
    print(f"\nFeatures compared: {stats['n_features']}")
    print(f"\nCorrelation Statistics:")
    print(f"  Pearson correlation:  {stats['pearson_correlation']:.4f} (p={stats['pearson_p_value']:.2e})")
    print(f"  Spearman correlation: {stats['spearman_correlation']:.4f} (p={stats['spearman_p_value']:.2e})")
    print(f"\nDifference Statistics:")
    print(f"  Mean absolute difference:    {stats['mean_absolute_difference']:.6f}")
    print(f"  Median absolute difference:  {stats['median_absolute_difference']:.6f}")
    print(f"  Mean relative difference:   {stats['mean_relative_difference']:.4f}")
    print("\n" + "="*80)
    
    # Interpretation
    if stats['pearson_correlation'] > 0.8:
        print("\n✅ VALIDATION PASSED: High correlation indicates JSON rule extraction")
        print("   closely matches SHAP importance patterns. XGBoost does NOT need")
        print("   SHAP as a translation layer (unlike CatBoost).")
    elif stats['pearson_correlation'] > 0.6:
        print("\n⚠️  MODERATE ALIGNMENT: Some correlation but may need investigation.")
    else:
        print("\n❌ VALIDATION FAILED: Low correlation suggests potential issues")
        print("   with rule extraction or SHAP calculation.")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    main()

