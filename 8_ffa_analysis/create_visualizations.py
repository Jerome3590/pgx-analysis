#!/usr/bin/env python3
"""
Create Visualizations for FFA Analysis Results

This script generates comprehensive visualizations from the FFA analysis outputs:
1. Feature importance comparisons across models
2. Coverage and importance metrics
3. Explanation statistics
4. Model comparison charts
"""

import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
import subprocess
import shutil

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import Counter
import ast

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration (cohort/age-band currently under analysis)
COHORT_NAME = "opioid_ed"
AGE_BAND = "0-12"
AGE_BAND_FNAME = AGE_BAND.replace("-", "_")
OUTPUT_DIR = PROJECT_ROOT / "7_ffa_analysis" / "outputs" / COHORT_NAME / AGE_BAND_FNAME
VISUALIZATION_DIR = OUTPUT_DIR / 'visualizations'
VISUALIZATION_DIR.mkdir(parents=True, exist_ok=True)
DATA_PATH = PROJECT_ROOT / '8_final_model' / 'outputs' / COHORT_NAME / AGE_BAND_FNAME / f'{COHORT_NAME}_{AGE_BAND_FNAME}_train_final_features_no_leakage.csv'

# Optional feature lookup mapping (feature index/name -> metadata) from step 6a.
FEATURE_LOOKUP_PATH = (
    PROJECT_ROOT
    / 'feature_encoding_outputs'
    / COHORT_NAME
    / AGE_BAND_FNAME
    / f'{COHORT_NAME}_{AGE_BAND_FNAME}_feature_lookup.csv'
)
if FEATURE_LOOKUP_PATH.exists():
    FEATURE_LOOKUP_DF = pd.read_csv(FEATURE_LOOKUP_PATH)
    logger.info(f"Loaded feature lookup: {FEATURE_LOOKUP_PATH}")
else:
    FEATURE_LOOKUP_DF = None

# Optional S3 base for visualizations
S3_BASE = f"s3://pgxdatalake/gold/ffa_analysis/{COHORT_NAME}/{AGE_BAND}/visualizations"

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10


def load_model_results(model_type: str) -> Optional[Dict]:
    """Load results for a specific model type."""
    model_dir = OUTPUT_DIR / model_type
    
    if not model_dir.exists():
        logger.warning(f"Model directory not found: {model_dir}")
        return None
    
    results = {}
    
    # Load feature importance (Parquet format)
    importance_path = model_dir / 'feature_importance_axp.parquet'
    if importance_path.exists():
        results['feature_importance'] = pd.read_parquet(importance_path)
        logger.info(f"Loaded feature importance for {model_type}: {len(results['feature_importance'])} features")
    
    # Load explanations (Parquet format)
    explanations_path = model_dir / 'axp_explanations.parquet'
    if explanations_path.exists():
        results['explanations'] = pd.read_parquet(explanations_path)
        logger.info(f"Loaded explanations for {model_type}: {len(results['explanations'])} explanations")
    
    # Load summary
    summary_path = model_dir / 'analysis_summary.json'
    if summary_path.exists():
        with open(summary_path, 'r') as f:
            results['summary'] = json.load(f)
        logger.info(f"Loaded summary for {model_type}")
    
    # Load causal analysis if available (Parquet format)
    causal_path = model_dir / 'causal_importance.parquet'
    if causal_path.exists():
        results['causal'] = pd.read_parquet(causal_path)
        logger.info(f"Loaded causal analysis for {model_type}")
    
    return results if results else None


def enrich_with_feature_lookup(df: pd.DataFrame) -> pd.DataFrame:
    """
    Attach human-readable labels and metadata using the feature lookup table,
    when available.
    """
    if FEATURE_LOOKUP_DF is None or 'feature' not in df.columns:
        return df

    merged = df.merge(
        FEATURE_LOOKUP_DF[
            ['feature_name', 'group', 'description', 'itemset_type', 'itemset_items']
        ],
        left_on='feature',
        right_on='feature_name',
        how='left',
    )
    # Prefer itemset contents as labels when available; fall back to feature name.
    merged['feature_label'] = np.where(
        merged['itemset_items'].notna() & (merged['itemset_items'] != ""),
        merged['itemset_items'],
        merged['feature'],
    )
    return merged


def plot_feature_importance_comparison(all_results: Dict[str, Dict], top_k: int = 20):
    """Create comparison plot of feature importance across models."""
    logger.info("Creating feature importance comparison plot...")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Feature Importance Comparison Across Models', fontsize=16, fontweight='bold')
    
    model_types = ['catboost', 'xgboost', 'xgboost_rf']
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    
    for idx, model_type in enumerate(model_types):
        ax = axes[idx]
        
        if model_type not in all_results or 'feature_importance' not in all_results[model_type]:
            ax.text(0.5, 0.5, f'No data for {model_type}', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(model_type.upper(), fontsize=12, fontweight='bold')
            continue
        
        df = all_results[model_type]['feature_importance'].head(top_k)
        df = enrich_with_feature_lookup(df)
        
        bars = ax.barh(range(len(df)), df['importance'].values, color=colors[idx], alpha=0.7)
        ax.set_yticks(range(len(df)))
        ax.set_yticklabels(df.get('feature_label', df['feature']).values, fontsize=8)
        ax.set_xlabel('Importance Score', fontsize=10)
        ax.set_title(f'{model_type.upper()} - Top {len(df)} Features', fontsize=12, fontweight='bold')
        ax.invert_yaxis()
        ax.grid(axis='x', linestyle='--', alpha=0.3)
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, df['importance'].values)):
            ax.text(val, i, f' {val:.3f}', va='center', fontsize=7)
    
    plt.tight_layout()
    save_path = VISUALIZATION_DIR / 'feature_importance_comparison.png'
    plt.savefig(save_path, bbox_inches='tight', facecolor='white')
    plt.close()
    logger.info(f"Saved: {save_path}")


def plot_top_features_consensus(all_results: Dict[str, Dict], top_k: int = 10):
    """Plot features that appear in top-k across all models."""
    logger.info("Creating top features consensus plot...")
    
    # Get top features from each model
    model_features = {}
    for model_type, results in all_results.items():
        if 'feature_importance' in results:
            df_imp = enrich_with_feature_lookup(results['feature_importance'].head(top_k).copy())
            top_features = set(df_imp.get('feature_label', df_imp['feature']).values)
            model_features[model_type] = top_features
    
    if not model_features:
        logger.warning("No feature importance data available")
        return
    
    # Find consensus features
    all_features = set()
    for features in model_features.values():
        all_features.update(features)
    
    # Count how many models include each feature
    feature_counts = Counter()
    for feature in all_features:
        count = sum(1 for features in model_features.values() if feature in features)
        feature_counts[feature] = count
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    consensus_df = pd.DataFrame([
        {'feature_label': feat, 'model_count': count}
        for feat, count in feature_counts.most_common()
    ])
    
    colors_map = {1: '#FF6B6B', 2: '#4ECDC4', 3: '#45B7D1'}
    colors = [colors_map.get(count, '#95A5A6') for count in consensus_df['model_count']]
    
    bars = ax.barh(range(len(consensus_df)), consensus_df['model_count'].values, color=colors, alpha=0.7)
    ax.set_yticks(range(len(consensus_df)))
    ax.set_yticklabels(consensus_df['feature_label'].values, fontsize=9)
    ax.set_xlabel('Number of Models', fontsize=11)
    ax.set_title(f'Feature Consensus Across Models (Top {top_k} per model)', 
                 fontsize=14, fontweight='bold')
    ax.set_xlim(0, 3.5)
    ax.invert_yaxis()
    ax.grid(axis='x', linestyle='--', alpha=0.3)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=colors_map[3], label='In all 3 models'),
        Patch(facecolor=colors_map[2], label='In 2 models'),
        Patch(facecolor=colors_map[1], label='In 1 model')
    ]
    ax.legend(handles=legend_elements, loc='lower right')
    
    plt.tight_layout()
    save_path = VISUALIZATION_DIR / 'feature_consensus.png'
    plt.savefig(save_path, bbox_inches='tight', facecolor='white')
    plt.close()
    logger.info(f"Saved: {save_path}")


def plot_coverage_vs_importance(all_results: Dict[str, Dict], top_k: int = 15):
    """Plot coverage vs importance for each model."""
    logger.info("Creating coverage vs importance plots...")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Coverage vs Importance', fontsize=16, fontweight='bold')
    
    model_types = ['catboost', 'xgboost', 'xgboost_rf']
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    
    for idx, model_type in enumerate(model_types):
        ax = axes[idx]
        
        if model_type not in all_results or 'feature_importance' not in all_results[model_type]:
            ax.text(0.5, 0.5, f'No data for {model_type}', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(model_type.upper(), fontsize=12, fontweight='bold')
            continue
        
        df = all_results[model_type]['feature_importance'].head(top_k)
        df = enrich_with_feature_lookup(df)
        
        scatter = ax.scatter(df['coverage'].values, df['importance'].values, 
                           s=100, alpha=0.6, color=colors[idx], edgecolors='black', linewidth=1)
        
        # Add feature labels
        for i, row in df.iterrows():
            label = row.get('feature_label', row['feature'])
            ax.annotate(label, 
                       (row['coverage'], row['importance']),
                       fontsize=7, alpha=0.7,
                       xytext=(5, 5), textcoords='offset points')
        
        ax.set_xlabel('Coverage', fontsize=10)
        ax.set_ylabel('Importance', fontsize=10)
        ax.set_title(f'{model_type.upper()} - Top {len(df)} Features', fontsize=12, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    save_path = VISUALIZATION_DIR / 'coverage_vs_importance.png'
    plt.savefig(save_path, bbox_inches='tight', facecolor='white')
    plt.close()
    logger.info(f"Saved: {save_path}")


def plot_explanation_statistics(all_results: Dict[str, Dict]):
    """Plot statistics about explanations."""
    logger.info("Creating explanation statistics plot...")
    
    stats_data = []
    for model_type, results in all_results.items():
        if 'summary' in results:
            summary = results['summary']
            stats_data.append({
                'model': model_type.upper(),
                'total_explanations': summary.get('total_explanations', 0),
                'explanations_with_conditions': summary.get('explanations_with_conditions', 0),
                'coverage_rate': summary.get('explanations_with_conditions', 0) / max(summary.get('total_explanations', 1), 1)
            })
    
    if not stats_data:
        logger.warning("No summary data available")
        return
    
    df_stats = pd.DataFrame(stats_data)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Explanation Statistics', fontsize=16, fontweight='bold')
    
    # Bar plot of explanation counts
    ax1 = axes[0]
    x = np.arange(len(df_stats))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, df_stats['total_explanations'], width, 
                    label='Total', color='#3498DB', alpha=0.7)
    bars2 = ax1.bar(x + width/2, df_stats['explanations_with_conditions'], width,
                    label='With Conditions', color='#2ECC71', alpha=0.7)
    
    ax1.set_xlabel('Model', fontsize=11)
    ax1.set_ylabel('Count', fontsize=11)
    ax1.set_title('Explanation Counts', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(df_stats['model'].values)
    ax1.legend()
    ax1.grid(axis='y', linestyle='--', alpha=0.3)
    
    # Coverage rate
    ax2 = axes[1]
    bars = ax2.bar(df_stats['model'].values, df_stats['coverage_rate'].values,
                   color=['#2E86AB', '#A23B72', '#F18F01'], alpha=0.7)
    ax2.set_ylabel('Coverage Rate', fontsize=11)
    ax2.set_title('Explanation Coverage Rate', fontsize=12, fontweight='bold')
    ax2.set_ylim(0, 1.1)
    ax2.grid(axis='y', linestyle='--', alpha=0.3)
    
    # Add value labels
    for bar, val in zip(bars, df_stats['coverage_rate'].values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2%}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    save_path = VISUALIZATION_DIR / 'explanation_statistics.png'
    plt.savefig(save_path, bbox_inches='tight', facecolor='white')
    plt.close()
    logger.info(f"Saved: {save_path}")


def plot_explanation_length_distribution(all_results: Dict[str, Dict]):
    """Plot distribution of explanation lengths (number of conditions)."""
    logger.info("Creating explanation length distribution plot...")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Explanation Length Distribution', fontsize=16, fontweight='bold')
    
    model_types = ['catboost', 'xgboost', 'xgboost_rf']
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    
    for idx, model_type in enumerate(model_types):
        ax = axes[idx]
        
        if model_type not in all_results or 'explanations' not in all_results[model_type]:
            ax.text(0.5, 0.5, f'No data for {model_type}', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(model_type.upper(), fontsize=12, fontweight='bold')
            continue
        
        df = all_results[model_type]['explanations']
        
        # Parse AXP strings and count conditions
        lengths = []
        for axp_str in df['axp'].dropna():
            try:
                if isinstance(axp_str, str):
                    parsed = ast.literal_eval(axp_str)
                    if isinstance(parsed, list):
                        lengths.append(len(parsed))
            except:
                continue
        
        if lengths:
            ax.hist(lengths, bins=20, color=colors[idx], alpha=0.7, edgecolor='black')
            ax.set_xlabel('Number of Conditions', fontsize=10)
            ax.set_ylabel('Frequency', fontsize=10)
            ax.set_title(f'{model_type.upper()} - Mean: {np.mean(lengths):.1f}', 
                        fontsize=12, fontweight='bold')
            ax.grid(axis='y', linestyle='--', alpha=0.3)
            
            # Add statistics text
            stats_text = f'Mean: {np.mean(lengths):.1f}\nMedian: {np.median(lengths):.1f}\nStd: {np.std(lengths):.1f}'
            ax.text(0.7, 0.95, stats_text, transform=ax.transAxes,
                   verticalalignment='top', fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        else:
            ax.text(0.5, 0.5, 'No valid explanations', 
                   ha='center', va='center', transform=ax.transAxes)
    
    plt.tight_layout()
    save_path = VISUALIZATION_DIR / 'explanation_length_distribution.png'
    plt.savefig(save_path, bbox_inches='tight', facecolor='white')
    plt.close()
    logger.info(f"Saved: {save_path}")


def plot_normalized_importance(all_results: Dict[str, Dict], top_k: int = 15):
    """Plot normalized feature importance for each model."""
    logger.info("Creating normalized importance plots...")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Normalized Feature Importance (Class 1)', fontsize=16, fontweight='bold')
    
    model_types = ['catboost', 'xgboost', 'xgboost_rf']
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    
    for idx, model_type in enumerate(model_types):
        ax = axes[idx]
        
        if model_type not in all_results or 'feature_importance' not in all_results[model_type]:
            ax.text(0.5, 0.5, f'No data for {model_type}', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(model_type.upper(), fontsize=12, fontweight='bold')
            continue
        
        df = all_results[model_type]['feature_importance'].head(top_k).copy()
        df = enrich_with_feature_lookup(df)
        # Normalize importance
        df['normalized'] = df['importance'] / df['importance'].max()
        
        bars = ax.barh(range(len(df)), df['normalized'].values, color=colors[idx], alpha=0.7)
        ax.set_yticks(range(len(df)))
        ax.set_yticklabels(df.get('feature_label', df['feature']).values, fontsize=8)
        ax.set_xlabel('Normalized Importance', fontsize=10)
        ax.set_title(f'{model_type.upper()} - Top {len(df)} Features', fontsize=12, fontweight='bold')
        ax.set_xlim(0, 1.1)
        ax.invert_yaxis()
        ax.grid(axis='x', linestyle='--', alpha=0.3)
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, df['normalized'].values)):
            ax.text(val, i, f' {val:.2f}', va='center', fontsize=7)
    
    plt.tight_layout()
    save_path = VISUALIZATION_DIR / 'normalized_importance.png'
    plt.savefig(save_path, bbox_inches='tight', facecolor='white')
    plt.close()
    logger.info(f"Saved: {save_path}")


def plot_cattail_distribution(all_results: Dict[str, Dict], X: pd.DataFrame, top_k: int = 10):
    """Plot cattail distribution of feature values."""
    logger.info("Creating cattail distribution plots...")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 8))
    fig.suptitle('Feature Value Distribution (Cattail Plots)', fontsize=16, fontweight='bold')
    
    model_types = ['catboost', 'xgboost', 'xgboost_rf']
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    
    for idx, model_type in enumerate(model_types):
        ax = axes[idx]
        
        if model_type not in all_results or 'feature_importance' not in all_results[model_type]:
            ax.text(0.5, 0.5, f'No data for {model_type}', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(model_type.upper(), fontsize=12, fontweight='bold')
            continue
        
        # Get top features
        df_importance = all_results[model_type]['feature_importance'].head(top_k)
        df_importance = enrich_with_feature_lookup(df_importance)
        top_features = df_importance.get('feature', df_importance['feature']).tolist()
        
        # Filter to features that exist in data
        available_features = [f for f in top_features if f in X.columns]
        
        if not available_features:
            ax.text(0.5, 0.5, 'No matching features in data', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(model_type.upper(), fontsize=12, fontweight='bold')
            continue
        
        # Prepare data for plotting
        X_subset = X[available_features].copy()
        
        # Melt data for plotting
        X_melted = X_subset.melt(var_name='Feature', value_name='Value')
        
        # Create box plot with stripplot overlay
        sns.boxplot(y='Feature', x='Value', data=X_melted,
                   whis=1.5, fliersize=0, color='lightgray', ax=ax)
        
        # Cattail overlay (stripplot)
        sns.stripplot(y='Feature', x='Value', data=X_melted,
                     jitter=0.25, size=2, alpha=0.6, color=colors[idx], ax=ax)
        
        ax.set_xlabel('Feature Value', fontsize=10)
        ax.set_ylabel('Feature', fontsize=10)
        ax.set_title(f'{model_type.upper()} - Top {len(available_features)} Features', 
                    fontsize=12, fontweight='bold')
        ax.grid(axis='x', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    save_path = VISUALIZATION_DIR / 'cattail_distribution.png'
    plt.savefig(save_path, bbox_inches='tight', facecolor='white')
    plt.close()
    logger.info(f"Saved: {save_path}")


def plot_combined_weighted_importance(all_results: Dict[str, Dict], top_k: int = 20):
    """Plot combined weighted feature importance across all models."""
    logger.info("Creating combined weighted feature importance plot...")
    
    # Calculate model weights based on performance metrics
    model_weights = {}
    model_scores = {}
    
    for model_type, results in all_results.items():
        if 'summary' not in results:
            continue
        
        summary = results['summary']
        coverage_rate = summary.get('explanations_with_conditions', 0) / max(summary.get('total_explanations', 1), 1)
        
        # Calculate composite score (coverage rate + normalized explanation quality)
        # Higher coverage = better model
        score = coverage_rate
        
        model_scores[model_type] = {
            'coverage_rate': coverage_rate,
            'total_explanations': summary.get('total_explanations', 0),
            'explanations_with_conditions': summary.get('explanations_with_conditions', 0),
            'score': score
        }
    
    # Normalize scores to get weights (sum to 1)
    if model_scores:
        total_score = sum(ms['score'] for ms in model_scores.values())
        for model_type, metrics in model_scores.items():
            model_weights[model_type] = metrics['score'] / total_score if total_score > 0 else 1.0 / len(model_scores)
    else:
        # Equal weights if no scores
        model_weights = {mt: 1.0 / len(all_results) for mt in all_results.keys()}
    
    logger.info(f"Model weights: {model_weights}")
    
    # Collect all features and their weighted importances
    feature_importance_dict = {}
    
    for model_type, results in all_results.items():
        if 'feature_importance' not in results:
            continue
        
        weight = model_weights.get(model_type, 0)
        df = results['feature_importance']
        
        for _, row in df.iterrows():
            feature = row['feature']
            importance = row['importance']
            coverage = row['coverage']
            
            if feature not in feature_importance_dict:
                feature_importance_dict[feature] = {
                    'weighted_importance': 0.0,
                    'weighted_coverage': 0.0,
                    'contributions': [],
                    'model_count': 0
                }
            
            # Normalize importance by max importance in this model for fair comparison
            max_importance = df['importance'].max()
            normalized_importance = importance / max_importance if max_importance > 0 else 0
            
            # Add weighted contribution
            feature_importance_dict[feature]['weighted_importance'] += normalized_importance * weight
            feature_importance_dict[feature]['weighted_coverage'] += coverage * weight
            feature_importance_dict[feature]['contributions'].append({
                'model': model_type,
                'weight': weight,
                'importance': importance,
                'normalized_importance': normalized_importance
            })
            feature_importance_dict[feature]['model_count'] += 1
    
    # Convert to DataFrame
    combined_data = []
    for feature, metrics in feature_importance_dict.items():
        combined_data.append({
            'feature': feature,
            'weighted_importance': metrics['weighted_importance'],
            'weighted_coverage': metrics['weighted_coverage'],
            'model_count': metrics['model_count'],
            'contributions': metrics['contributions']
        })
    
    combined_df = pd.DataFrame(combined_data)
    combined_df = combined_df.sort_values('weighted_importance', ascending=False).head(top_k)
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    fig.suptitle('Combined Weighted Feature Importance Across Models', fontsize=16, fontweight='bold')
    
    # Main bar chart
    y_pos = np.arange(len(combined_df))
    bars = ax1.barh(y_pos, combined_df['weighted_importance'].values, 
                    color='steelblue', alpha=0.7, edgecolor='black')
    
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(combined_df['feature'].values, fontsize=9)
    ax1.set_xlabel('Weighted Normalized Importance', fontsize=12, fontweight='bold')
    ax1.set_title(f'Top {len(combined_df)} Features (Weighted by Model Performance)', 
                 fontsize=14, fontweight='bold')
    ax1.invert_yaxis()
    ax1.grid(axis='x', linestyle='--', alpha=0.3)
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, combined_df['weighted_importance'].values)):
        ax1.text(val, i, f' {val:.3f}', va='center', fontsize=8)
    
    # Add model weight legend
    weight_text = "Model Weights:\n" + "\n".join([
        f"{mt.upper()}: {w:.2%}" for mt, w in sorted(model_weights.items(), key=lambda x: x[1], reverse=True)
    ])
    ax1.text(0.98, 0.02, weight_text, transform=ax1.transAxes,
            fontsize=9, verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Stacked bar chart showing contributions from each model
    model_types = sorted(model_weights.keys())
    model_colors = {'catboost': '#2E86AB', 'xgboost': '#A23B72', 'xgboost_rf': '#F18F01'}
    
    # Prepare stacked data
    stacked_data = {mt: [] for mt in model_types}
    feature_names = combined_df['feature'].values
    
    for feature in feature_names:
        feature_data = combined_df[combined_df['feature'] == feature].iloc[0]
        contributions = feature_data['contributions']
        
        for mt in model_types:
            # Find contribution from this model
            contrib = next((c for c in contributions if c['model'] == mt), None)
            if contrib:
                stacked_data[mt].append(contrib['normalized_importance'] * contrib['weight'])
            else:
                stacked_data[mt].append(0)
    
    # Create stacked bars
    bottom = np.zeros(len(combined_df))
    for mt in model_types:
        values = stacked_data[mt]
        ax2.barh(y_pos, values, left=bottom, 
                label=mt.upper(), color=model_colors.get(mt, 'gray'), 
                alpha=0.7, edgecolor='black')
        bottom += values
    
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(feature_names, fontsize=9)
    ax2.set_xlabel('Weighted Normalized Importance', fontsize=12, fontweight='bold')
    ax2.set_title('Model Contributions to Combined Importance', 
                 fontsize=14, fontweight='bold')
    ax2.invert_yaxis()
    ax2.legend(loc='lower right', fontsize=10)
    ax2.grid(axis='x', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    save_path = VISUALIZATION_DIR / 'combined_weighted_importance.png'
    plt.savefig(save_path, bbox_inches='tight', facecolor='white')
    plt.close()
    logger.info(f"Saved: {save_path}")


def upload_visualizations_to_s3() -> None:
    """
    Best-effort upload of visualization artifacts to S3 using the AWS CLI.
    """
    aws_cmd = shutil.which("aws")
    if not aws_cmd:
        logger.warning("AWS CLI not found; skipping S3 upload of FFA visualizations.")
        return

    logger.info("Uploading FFA visualizations to S3...")
    logger.info(f"S3 base: {S3_BASE}")

    uploaded = 0
    for path in VISUALIZATION_DIR.iterdir():
        if not path.is_file():
            continue
        s3_path = f"{S3_BASE}/{path.name}"
        try:
            result = subprocess.run(
                [aws_cmd, "s3", "cp", str(path), s3_path],
                capture_output=True,
                text=True,
                timeout=120,
            )
            if result.returncode == 0:
                logger.info(f"Uploaded: {path.name} -> {s3_path}")
                uploaded += 1
            else:
                logger.warning(
                    "Failed to upload %s: %s", path.name, (result.stderr or "").strip()
                )
        except Exception as e:
            logger.warning("Error uploading %s: %s", path.name, e)

    logger.info("Uploaded %d visualization file(s) to S3.", uploaded)


def plot_mirror_frequency_comparison(all_results: Dict[str, Dict], X: pd.DataFrame, y: pd.Series, top_k: int = 10):
    """Plot mirror chart comparing feature frequencies between classes."""
    logger.info("Creating mirror frequency comparison plots...")
    
    # Filter data by class
    X_class_0 = X[y == 0] if y is not None else None
    X_class_1 = X[y == 1] if y is not None else None
    
    if X_class_0 is None or X_class_1 is None or len(X_class_0) == 0 or len(X_class_1) == 0:
        logger.warning("Cannot create mirror plots: missing class data")
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 8))
    fig.suptitle('Feature Frequency Comparison: Class 0 vs Class 1', fontsize=16, fontweight='bold')
    
    model_types = ['catboost', 'xgboost', 'xgboost_rf']
    
    for idx, model_type in enumerate(model_types):
        ax = axes[idx]
        
        if model_type not in all_results or 'feature_importance' not in all_results[model_type]:
            ax.text(0.5, 0.5, f'No data for {model_type}', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(model_type.upper(), fontsize=12, fontweight='bold')
            continue
        
        # Get top features
        df_importance = all_results[model_type]['feature_importance'].head(top_k)
        top_features = df_importance['feature'].tolist()
        available_features = [f for f in top_features if f in X.columns]
        
        if not available_features:
            ax.text(0.5, 0.5, 'No matching features', 
                   ha='center', va='center', transform=ax.transAxes)
            continue
        
        # Calculate mean values for each class
        plot_data = []
        for feature in available_features:
            mean_0 = X_class_0[feature].mean() if len(X_class_0) > 0 else 0
            mean_1 = X_class_1[feature].mean() if len(X_class_1) > 0 else 0
            
            plot_data.append({
                'feature': feature,
                'class_0_mean': mean_0,
                'class_1_mean': mean_1
            })
        
        plot_df = pd.DataFrame(plot_data)
        
        # Create mirror bar chart
        y_pos = np.arange(len(plot_df))
        
        # Class 0 bars (left, negative)
        bars_0 = ax.barh(y_pos - 0.2, -plot_df['class_0_mean'].values, 
                         height=0.4, color='skyblue', alpha=0.7, label='Class 0', edgecolor='black')
        
        # Class 1 bars (right, positive)
        bars_1 = ax.barh(y_pos + 0.2, plot_df['class_1_mean'].values, 
                         height=0.4, color='lightcoral', alpha=0.7, label='Class 1', edgecolor='black')
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(plot_df['feature'].values, fontsize=8)
        ax.set_xlabel('Mean Feature Value', fontsize=10)
        ax.set_title(f'{model_type.upper()} - Top {len(available_features)} Features', 
                    fontsize=12, fontweight='bold')
        ax.axvline(x=0, color='black', linewidth=1)
        ax.legend(loc='lower right')
        ax.grid(axis='x', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    save_path = VISUALIZATION_DIR / 'mirror_frequency_comparison.png'
    plt.savefig(save_path, bbox_inches='tight', facecolor='white')
    plt.close()
    logger.info(f"Saved: {save_path}")


def create_summary_report(all_results: Dict[str, Dict]):
    """Create a text summary report."""
    logger.info("Creating summary report...")
    
    report_lines = [
        "=" * 80,
        "FFA Analysis Summary Report",
        f"Cohort: {COHORT_NAME}, Age Band: {AGE_BAND}",
        "=" * 80,
        ""
    ]
    
    for model_type, results in all_results.items():
        report_lines.append(f"\n{model_type.upper()} Model:")
        report_lines.append("-" * 40)
        
        if 'summary' in results:
            summary = results['summary']
            report_lines.append(f"  Total Explanations: {summary.get('total_explanations', 'N/A')}")
            report_lines.append(f"  Explanations with Conditions: {summary.get('explanations_with_conditions', 'N/A')}")
            coverage_rate = summary.get('explanations_with_conditions', 0) / max(summary.get('total_explanations', 1), 1)
            report_lines.append(f"  Coverage Rate: {coverage_rate:.2%}")
        
        if 'feature_importance' in results:
            df = results['feature_importance']
            report_lines.append(f"\n  Top 5 Features:")
            for i, row in df.head(5).iterrows():
                report_lines.append(f"    {i+1}. {row['feature']}: importance={row['importance']:.3f}, coverage={row['coverage']:.3f}")
    
    report_text = "\n".join(report_lines)
    
    report_path = VISUALIZATION_DIR / 'summary_report.txt'
    with open(report_path, 'w') as f:
        f.write(report_text)
    
    logger.info(f"Saved: {report_path}")
    print("\n" + report_text)


def load_data():
    """Load the training data for visualizations."""
    if not DATA_PATH.exists():
        logger.warning(f"Data file not found: {DATA_PATH}")
        return None, None
    
    logger.info(f"Loading data from: {DATA_PATH}")
    try:
        data = pd.read_csv(DATA_PATH, nrows=10000)  # Limit for memory efficiency
        
        # Separate features and target
        target_cols = ['target', 'is_target_case']
        target_col = None
        for col in target_cols:
            if col in data.columns:
                target_col = col
                break
        
        if target_col:
            y = data[target_col]
            X = data.drop(target_col, axis=1)
            logger.info(f"Loaded {len(X)} samples, {len(X.columns)} features, target: {target_col}")
            return X, y
        else:
            logger.warning("No target column found")
            return data, None
            
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return None, None


def main():
    """Generate all visualizations."""
    logger.info("=" * 80)
    logger.info("Creating FFA Analysis Visualizations")
    logger.info(f"Output directory: {VISUALIZATION_DIR}")
    logger.info("=" * 80)
    
    # Load results for all models
    model_types = ['catboost', 'xgboost', 'xgboost_rf']
    all_results = {}
    
    for model_type in model_types:
        results = load_model_results(model_type)
        if results:
            all_results[model_type] = results
    
    if not all_results:
        logger.error("No results found for any model!")
        return
    
    logger.info(f"Loaded results for {len(all_results)} model(s)")
    
    # Load data for cattail visualizations
    X, y = load_data()
    has_data = X is not None
    
    # Generate visualizations
    try:
        plot_feature_importance_comparison(all_results, top_k=20)
        plot_top_features_consensus(all_results, top_k=10)
        plot_coverage_vs_importance(all_results, top_k=15)
        plot_explanation_statistics(all_results)
        plot_explanation_length_distribution(all_results)
        plot_normalized_importance(all_results, top_k=15)
        plot_combined_weighted_importance(all_results, top_k=20)
        
        # Cattail visualizations (require data)
        if has_data:
            plot_cattail_distribution(all_results, X, top_k=10)
            if y is not None:
                plot_mirror_frequency_comparison(all_results, X, y, top_k=10)
        else:
            logger.warning("Skipping cattail visualizations: data not available")
        
        create_summary_report(all_results)

        # Best-effort S3 upload for all visualization artifacts
        upload_visualizations_to_s3()

        logger.info("=" * 80)
        logger.info("All visualizations created successfully!")
        logger.info(f"Visualizations saved to: {VISUALIZATION_DIR}")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"Error creating visualizations: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()

