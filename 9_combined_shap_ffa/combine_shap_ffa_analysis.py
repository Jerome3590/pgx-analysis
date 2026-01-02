#!/usr/bin/env python3
"""
Combine SHAP and FFA Analysis for Comprehensive Row-Level Patient Analysis

This script combines:
1. SHAP values (quantitative feature contributions)
2. FFA AXP explanations (rule-based logical explanations)
3. Consensus analysis (features important in both)
4. Patient-level comprehensive reports

Usage:
    python 8_final_model/combine_shap_ffa_analysis.py \
        --cohort non_opioid_ed \
        --age-band 65-74 \
        --shap-values-path path/to/shap_values.npy \
        --ffa-explanations-path 7_ffa_analysis/outputs/.../axp_explanations.csv
"""

import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Set, Optional
import json
import ast
import warnings
warnings.filterwarnings("ignore")

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_shap_results(shap_values_path: Path, shap_importance_path: Optional[Path] = None):
    """Load SHAP values and importance."""
    logger.info(f"Loading SHAP values from {shap_values_path}")
    
    if shap_values_path.suffix == '.npy':
        shap_values = np.load(shap_values_path)
    elif shap_values_path.suffix == '.csv':
        shap_values = pd.read_csv(shap_values_path).values
    else:
        raise ValueError(f"Unsupported SHAP file format: {shap_values_path.suffix}")
    
    shap_importance = None
    if shap_importance_path and shap_importance_path.exists():
        shap_importance = pd.read_csv(shap_importance_path)
        logger.info(f"Loaded SHAP importance: {len(shap_importance)} features")
    
    logger.info(f"SHAP values shape: {shap_values.shape}")
    return shap_values, shap_importance


def load_ffa_results(ffa_explanations_path: Path, ffa_importance_path: Optional[Path] = None):
    """Load FFA AXP explanations and importance."""
    logger.info(f"Loading FFA explanations from {ffa_explanations_path}")
    
    ffa_explanations = pd.read_csv(ffa_explanations_path)
    logger.info(f"Loaded {len(ffa_explanations)} FFA explanations")
    
    ffa_importance = None
    if ffa_importance_path and ffa_importance_path.exists():
        ffa_importance = pd.read_csv(ffa_importance_path)
        logger.info(f"Loaded FFA importance: {len(ffa_importance)} features")
    
    return ffa_explanations, ffa_importance


def extract_features_from_rules(rules: List) -> Set[str]:
    """Extract feature names from FFA rules."""
    features = set()
    
    if isinstance(rules, str):
        try:
            rules = ast.literal_eval(rules)
        except:
            # Try to parse as string
            pass
    
    if isinstance(rules, list):
        for rule in rules:
            if isinstance(rule, str):
                # Extract feature names from rule string
                # Format: "feature_name > threshold" or "feature_name < threshold"
                import re
                feature_matches = re.findall(r'(\w+)\s*[><=]', rule)
                features.update(feature_matches)
    
    return features


def find_consensus_features(
    shap_importance: pd.DataFrame,
    ffa_importance: pd.DataFrame,
    top_k: int = 20
) -> Set[str]:
    """Find features that appear in top K of both methods."""
    if shap_importance is None or ffa_importance is None:
        return set()
    
    shap_top = set(shap_importance.head(top_k)['feature'].values)
    ffa_top = set(ffa_importance.head(top_k)['feature'].values)
    
    consensus = shap_top.intersection(ffa_top)
    logger.info(f"Found {len(consensus)} consensus features in top {top_k}")
    
    return consensus


def analyze_patient_comprehensive(
    patient_idx: int,
    shap_values: np.ndarray,
    ffa_explanations: pd.DataFrame,
    feature_names: List[str],
    patient_id_col: str = 'instance_id'
) -> Dict:
    """
    Comprehensive patient analysis combining SHAP and FFA.
    """
    # SHAP analysis
    patient_shap = shap_values[patient_idx]
    shap_df = pd.DataFrame({
        'feature': feature_names,
        'shap_value': patient_shap
    }).sort_values('shap_value', ascending=False)
    
    top_positive = shap_df.head(5)['feature'].tolist()
    top_negative = shap_df.tail(5)['feature'].tolist()
    
    # FFA analysis
    if patient_id_col in ffa_explanations.columns:
        patient_ffa = ffa_explanations[ffa_explanations[patient_id_col] == patient_idx]
    else:
        patient_ffa = ffa_explanations.iloc[patient_idx:patient_idx+1]
    
    if len(patient_ffa) > 0:
        matched_rules = patient_ffa['axp'].values[0] if 'axp' in patient_ffa.columns else []
        conditions = patient_ffa['conditions'].values[0] if 'conditions' in patient_ffa.columns else []
        ffa_features = extract_features_from_rules(matched_rules)
    else:
        matched_rules = []
        conditions = []
        ffa_features = set()
    
    # Consensus
    shap_top_features = set(shap_df.head(10)['feature'].values)
    consensus = shap_top_features.intersection(ffa_features)
    
    return {
        'patient_idx': patient_idx,
        'shap_top_positive': top_positive,
        'shap_top_negative': top_negative,
        'shap_total': patient_shap.sum(),
        'ffa_matched_rules': matched_rules,
        'ffa_conditions': conditions,
        'ffa_features': list(ffa_features),
        'consensus_features': list(consensus),
        'shap_values': shap_df.to_dict('records')
    }


def combine_importance_scores(
    shap_importance: pd.DataFrame,
    ffa_importance: pd.DataFrame,
    weight_shap: float = 0.5,
    weight_ffa: float = 0.5
) -> pd.DataFrame:
    """
    Combine SHAP and FFA importance scores with weighted average.
    """
    if shap_importance is None or ffa_importance is None:
        return shap_importance if shap_importance is not None else ffa_importance
    
    # Normalize both to [0, 1]
    shap_norm = (shap_importance['importance'] - shap_importance['importance'].min()) / \
                (shap_importance['importance'].max() - shap_importance['importance'].min() + 1e-10)
    
    ffa_norm = (ffa_importance['importance'] - ffa_importance['importance'].min()) / \
               (ffa_importance['importance'].max() - ffa_importance['importance'].min() + 1e-10)
    
    # Merge
    combined = shap_importance[['feature']].merge(
        ffa_importance[['feature', 'importance']],
        on='feature',
        how='outer',
        suffixes=('_shap', '_ffa')
    )
    
    # Add normalized scores
    shap_dict = dict(zip(shap_importance['feature'], shap_norm))
    ffa_dict = dict(zip(ffa_importance['feature'], ffa_norm))
    
    combined['shap_norm'] = combined['feature'].map(shap_dict).fillna(0)
    combined['ffa_norm'] = combined['feature'].map(ffa_dict).fillna(0)
    
    # Weighted combination
    combined['combined_importance'] = (
        weight_shap * combined['shap_norm'] + 
        weight_ffa * combined['ffa_norm']
    )
    
    return combined.sort_values('combined_importance', ascending=False)


def main():
    parser = argparse.ArgumentParser(description="Combine SHAP and FFA analysis")
    parser.add_argument("--cohort", required=True, help="Cohort name")
    parser.add_argument("--age-band", required=True, help="Age band")
    parser.add_argument("--shap-values-path", required=True, help="Path to SHAP values")
    parser.add_argument("--ffa-explanations-path", required=True, help="Path to FFA explanations CSV")
    parser.add_argument("--shap-importance-path", help="Path to SHAP importance CSV (optional)")
    parser.add_argument("--ffa-importance-path", help="Path to FFA importance CSV (optional)")
    parser.add_argument("--feature-names-path", help="Path to feature names file (optional)")
    parser.add_argument("--output-dir", default="8_final_model/outputs", help="Output directory")
    parser.add_argument("--top-k", type=int, default=20, help="Top K features for consensus")
    parser.add_argument("--weight-shap", type=float, default=0.5, help="Weight for SHAP (0-1)")
    parser.add_argument("--weight-ffa", type=float, default=0.5, help="Weight for FFA (0-1)")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir) / args.cohort / args.age_band.replace("-", "_")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load SHAP results
    shap_values, shap_importance = load_shap_results(
        Path(args.shap_values_path),
        Path(args.shap_importance_path) if args.shap_importance_path else None
    )
    
    # Load FFA results
    ffa_explanations, ffa_importance = load_ffa_results(
        Path(args.ffa_explanations_path),
        Path(args.ffa_importance_path) if args.ffa_importance_path else None
    )
    
    # Get feature names
    if args.feature_names_path:
        with open(args.feature_names_path, 'r') as f:
            feature_names = json.load(f)
    elif shap_importance is not None:
        feature_names = shap_importance['feature'].tolist()
    elif ffa_importance is not None:
        feature_names = ffa_importance['feature'].tolist()
    else:
        feature_names = [f"feature_{i}" for i in range(shap_values.shape[1])]
    
    logger.info(f"Using {len(feature_names)} features")
    
    # Find consensus features
    consensus_features = find_consensus_features(shap_importance, ffa_importance, args.top_k)
    
    # Combine importance scores
    combined_importance = combine_importance_scores(
        shap_importance, ffa_importance, args.weight_shap, args.weight_ffa
    )
    
    # Save combined importance
    combined_path = output_dir / f"{args.cohort}_{args.age_band.replace('-', '_')}_combined_shap_ffa_importance.csv"
    combined_importance.to_csv(combined_path, index=False)
    logger.info(f"Saved combined importance to {combined_path}")
    
    # Analyze sample patients
    logger.info("Analyzing sample patients...")
    sample_patients = min(10, len(ffa_explanations))
    patient_analyses = []
    
    for i in range(sample_patients):
        analysis = analyze_patient_comprehensive(
            i, shap_values, ffa_explanations, feature_names
        )
        patient_analyses.append(analysis)
    
    # Save patient analyses
    patients_path = output_dir / f"{args.cohort}_{args.age_band.replace('-', '_')}_patient_analyses.json"
    with open(patients_path, 'w') as f:
        json.dump(patient_analyses, f, indent=2)
    logger.info(f"Saved patient analyses to {patients_path}")
    
    # Save consensus features
    consensus_path = output_dir / f"{args.cohort}_{args.age_band.replace('-', '_')}_consensus_features.json"
    with open(consensus_path, 'w') as f:
        json.dump(list(consensus_features), f, indent=2)
    logger.info(f"Saved consensus features to {consensus_path}")
    
    # Print summary
    print("\n" + "="*70)
    print("SHAP + FFA COMBINED ANALYSIS SUMMARY")
    print("="*70)
    print(f"\nConsensus Features (top {args.top_k}): {len(consensus_features)}")
    print(f"  {', '.join(list(consensus_features)[:10])}")
    print(f"\nCombined Importance (top 10):")
    print(combined_importance.head(10)[['feature', 'combined_importance', 'shap_norm', 'ffa_norm']].to_string(index=False))
    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()

