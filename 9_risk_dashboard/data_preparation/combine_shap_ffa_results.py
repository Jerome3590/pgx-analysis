#!/usr/bin/env python3
"""
Combine SHAP and FFA Results for Final Reporting

This script aggregates and combines SHAP and FFA analysis results from Steps 7 and 8
to create comprehensive patient-level explanations. Note: Consensus is already reflected
in FFA's causal importance scores, which use SHAP-prioritized rules.

Usage:
    python 10_results/combine_shap_ffa_results.py \
        --cohort non_opioid_ed \
        --age-band 65-74 \
        --output-dir 10_results/outputs
"""

import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple
import json
import ast
import warnings
from collections import defaultdict
warnings.filterwarnings("ignore")

# Add project root to path
# This script is in 9_risk_dashboard/data_preparation/
# Project root is 3 levels up
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def find_shap_results(cohort: str, age_band: str, project_root: Path) -> Optional[Path]:
    """Find SHAP results from Step 7."""
    age_band_fname = age_band.replace("-", "_")
    
    # Check common locations
    possible_paths = [
        project_root / '8_final_model' / 'outputs' / cohort / age_band_fname / 'shap_values.npy',
        project_root / '8_final_model' / 'outputs' / cohort / age_band_fname / 'shap' / 'shap_values.npy',
        project_root / '8_final_model' / 'outputs' / cohort / age_band_fname / 'shap_feature_importance.csv',
    ]
    
    for path in possible_paths:
        if path.exists():
            logger.info(f"Found SHAP results: {path}")
            return path
    
    logger.warning("SHAP results not found - will skip SHAP analysis")
    return None


def find_ffa_results(cohort: str, age_band: str, project_root: Path) -> Tuple[Optional[Path], Optional[Path]]:
    """Find FFA results from the FFA step (8_ffa_analysis)."""
    age_band_fname = age_band.replace("-", "_")
    
    explanations_path = (
        project_root
        / "8_ffa_analysis"
        / "outputs"
        / cohort
        / age_band_fname
        / "catboost"
        / "axp_explanations.csv"
    )
    importance_path = (
        project_root
        / "8_ffa_analysis"
        / "outputs"
        / cohort
        / age_band_fname
        / "catboost"
        / "feature_importance_axp.csv"
    )
    
    if explanations_path.exists():
        logger.info(f"Found FFA explanations: {explanations_path}")
    else:
        logger.warning("FFA explanations not found")
        explanations_path = None
    
    if importance_path.exists():
        logger.info(f"Found FFA importance: {importance_path}")
    else:
        logger.warning("FFA importance not found")
        importance_path = None
    
    return explanations_path, importance_path


def load_shap_data(shap_path: Path) -> Tuple[np.ndarray, Optional[pd.DataFrame]]:
    """Load SHAP values and importance."""
    if shap_path.suffix == '.npy':
        shap_values = np.load(shap_path)
        importance = None
    elif shap_path.suffix == '.csv':
        df = pd.read_csv(shap_path)
        if 'shap_value' in df.columns or 'importance' in df.columns:
            importance = df
            shap_values = None  # Will need to reconstruct from importance
        else:
            shap_values = df.values
            importance = None
    else:
        raise ValueError(f"Unsupported SHAP file format: {shap_path.suffix}")
    
    return shap_values, importance


def extract_features_from_ffa_rules(rules: List) -> Set[str]:
    """Extract feature names from FFA rule strings."""
    features = set()
    
    if isinstance(rules, str):
        try:
            rules = ast.literal_eval(rules)
        except:
            # Try regex parsing
            import re
            feature_matches = re.findall(r'(\w+)\s*[><=]', rules)
            features.update(feature_matches)
            return features
    
    if isinstance(rules, list):
        for rule in rules:
            if isinstance(rule, str):
                import re
                feature_matches = re.findall(r'(\w+)\s*[><=]', rule)
                features.update(feature_matches)
    
    return features


def calculate_consensus_features(
    shap_importance: pd.DataFrame,
    ffa_importance: pd.DataFrame,
    top_k: int = 20
) -> Dict[str, any]:
    """Calculate consensus features between SHAP and FFA."""
    if shap_importance is None or ffa_importance is None:
        return {
            'consensus_features': [],
            'shap_only': [],
            'ffa_only': [],
            'consensus_count': 0
        }
    
    # Get top K features from each
    shap_col = 'importance' if 'importance' in shap_importance.columns else shap_importance.columns[1]
    ffa_col = 'importance' if 'importance' in ffa_importance.columns else ffa_importance.columns[1]
    
    shap_top = set(shap_importance.head(top_k)['feature'].values)
    ffa_top = set(ffa_importance.head(top_k)['feature'].values)
    
    consensus = shap_top.intersection(ffa_top)
    shap_only = shap_top - ffa_top
    ffa_only = ffa_top - shap_top
    
    return {
        'consensus_features': sorted(list(consensus)),
        'shap_only': sorted(list(shap_only)),
        'ffa_only': sorted(list(ffa_only)),
        'consensus_count': len(consensus),
        'shap_count': len(shap_top),
        'ffa_count': len(ffa_top),
        'consensus_rate': len(consensus) / top_k if top_k > 0 else 0
    }


def combine_importance_scores(
    shap_importance: Optional[pd.DataFrame],
    ffa_importance: Optional[pd.DataFrame],
    weight_shap: float = 0.5,
    weight_ffa: float = 0.5
) -> pd.DataFrame:
    """Combine SHAP and FFA importance scores."""
    if shap_importance is None and ffa_importance is None:
        return pd.DataFrame()
    
    if shap_importance is None:
        return ffa_importance.copy()
    
    if ffa_importance is None:
        return shap_importance.copy()
    
    # Normalize both to [0, 1]
    shap_col = 'importance' if 'importance' in shap_importance.columns else shap_importance.columns[1]
    ffa_col = 'importance' if 'importance' in ffa_importance.columns else ffa_importance.columns[1]
    
    shap_norm = (shap_importance[shap_col] - shap_importance[shap_col].min()) / \
                (shap_importance[shap_col].max() - shap_importance[shap_col].min() + 1e-10)
    
    ffa_norm = (ffa_importance[ffa_col] - ffa_importance[ffa_col].min()) / \
               (ffa_importance[ffa_col].max() - ffa_importance[ffa_col].min() + 1e-10)
    
    # Merge
    combined = shap_importance[['feature']].merge(
        ffa_importance[['feature', ffa_col]],
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


def generate_patient_explanations(
    shap_values: Optional[np.ndarray],
    ffa_explanations: pd.DataFrame,
    feature_names: List[str],
    n_samples: int = 100
) -> pd.DataFrame:
    """Generate comprehensive patient-level explanations combining SHAP and FFA."""
    results = []
    
    # Limit to sample size
    sample_size = min(n_samples, len(ffa_explanations))
    ffa_sample = ffa_explanations.head(sample_size)
    
    for idx, row in ffa_sample.iterrows():
        patient_explanation = {
            'patient_index': idx,
            'patient_id': row.get('instance_id', idx),
        }
        
        # FFA analysis
        if 'axp' in row:
            matched_rules = row['axp']
            ffa_features = extract_features_from_ffa_rules(matched_rules)
            patient_explanation['ffa_matched_rules'] = str(matched_rules)
            patient_explanation['ffa_features'] = list(ffa_features)
            patient_explanation['ffa_rule_count'] = len(matched_rules) if isinstance(matched_rules, list) else 1
        else:
            patient_explanation['ffa_features'] = []
            patient_explanation['ffa_rule_count'] = 0
        
        # SHAP analysis
        if shap_values is not None and idx < len(shap_values):
            patient_shap = shap_values[idx]
            shap_df = pd.DataFrame({
                'feature': feature_names[:len(patient_shap)],
                'shap_value': patient_shap
            }).sort_values('shap_value', ascending=False)
            
            top_positive = shap_df.head(5)['feature'].tolist()
            top_negative = shap_df.tail(5)['feature'].tolist()
            
            patient_explanation['shap_top_positive'] = top_positive
            patient_explanation['shap_top_negative'] = top_negative
            patient_explanation['shap_total'] = float(patient_shap.sum())
            
            # Consensus
            shap_top_set = set(shap_df.head(10)['feature'].values)
            ffa_features_set = set(patient_explanation.get('ffa_features', []))
            consensus = shap_top_set.intersection(ffa_features_set)
            patient_explanation['consensus_features'] = list(consensus)
        else:
            patient_explanation['shap_top_positive'] = []
            patient_explanation['shap_top_negative'] = []
            patient_explanation['shap_total'] = None
            patient_explanation['consensus_features'] = []
        
        results.append(patient_explanation)
    
    return pd.DataFrame(results)


def generate_summary_report(
    consensus_data: Dict,
    combined_importance: pd.DataFrame,
    patient_explanations: pd.DataFrame
) -> str:
    """Generate a human-readable summary report."""
    report = []
    report.append("="*80)
    report.append("SHAP + FFA COMBINED ANALYSIS SUMMARY")
    report.append("="*80)
    report.append("")
    
    # Consensus summary
    report.append("CONSENSUS FEATURES:")
    report.append(f"  - Consensus features: {consensus_data['consensus_count']}")
    report.append(f"  - SHAP-only features: {len(consensus_data['shap_only'])}")
    report.append(f"  - FFA-only features: {len(consensus_data['ffa_only'])}")
    report.append(f"  - Consensus rate: {consensus_data['consensus_rate']:.1%}")
    report.append("")
    
    if consensus_data['consensus_features']:
        report.append("  High-confidence features (consensus):")
        for feat in consensus_data['consensus_features'][:10]:
            report.append(f"    - {feat}")
    report.append("")
    
    # Combined importance summary
    if not combined_importance.empty:
        report.append("COMBINED FEATURE IMPORTANCE (Top 10):")
        top_features = combined_importance.head(10)
        for idx, row in top_features.iterrows():
            report.append(f"  {idx+1}. {row['feature']}: {row['combined_importance']:.4f} "
                         f"(SHAP: {row['shap_norm']:.3f}, FFA: {row['ffa_norm']:.3f})")
        report.append("")
    
    # Patient explanation summary
    if not patient_explanations.empty:
        report.append("PATIENT EXPLANATIONS:")
        report.append(f"  - Total patients analyzed: {len(patient_explanations)}")
        
        if 'consensus_features' in patient_explanations.columns:
            patients_with_consensus = patient_explanations[
                patient_explanations['consensus_features'].apply(lambda x: len(x) > 0)
            ]
            report.append(f"  - Patients with consensus features: {len(patients_with_consensus)} "
                         f"({len(patients_with_consensus)/len(patient_explanations):.1%})")
        report.append("")
    
    report.append("="*80)
    
    return "\n".join(report)


def main():
    parser = argparse.ArgumentParser(description="Combine SHAP and FFA results for final reporting")
    parser.add_argument("--cohort", required=True, help="Cohort name")
    parser.add_argument("--age-band", required=True, help="Age band")
    parser.add_argument("--output-dir", default="10_results/outputs", help="Output directory")
    parser.add_argument("--top-k", type=int, default=20, help="Top K features for consensus")
    parser.add_argument("--weight-shap", type=float, default=0.5, help="Weight for SHAP (0-1)")
    parser.add_argument("--weight-ffa", type=float, default=0.5, help="Weight for FFA (0-1)")
    parser.add_argument("--n-patients", type=int, default=100, help="Number of patients to analyze")
    parser.add_argument("--all-cohorts", action="store_true", help="Process all cohorts")
    
    args = parser.parse_args()
    
    # This script is in 9_risk_dashboard/data_preparation/
# Project root is 3 levels up
project_root = Path(__file__).parent.parent.parent
    
    if args.all_cohorts:
        # Process all cohorts (implement as needed)
        logger.info("Processing all cohorts...")
        # TODO: Implement batch processing
        return
    
    output_dir = Path(args.output_dir) / args.cohort / args.age_band.replace("-", "_")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Combining SHAP and FFA results for {args.cohort} / {args.age_band}")
    
    # Find results
    shap_path = find_shap_results(args.cohort, args.age_band, project_root)
    ffa_explanations_path, ffa_importance_path = find_ffa_results(args.cohort, args.age_band, project_root)
    
    # Load data
    shap_values = None
    shap_importance = None
    if shap_path:
        shap_values, shap_importance = load_shap_data(shap_path)
    
    ffa_explanations = None
    ffa_importance = None
    if ffa_explanations_path:
        ffa_explanations = pd.read_csv(ffa_explanations_path)
    if ffa_importance_path:
        ffa_importance = pd.read_csv(ffa_importance_path)
    
    # Get feature names
    if shap_importance is not None:
        feature_names = shap_importance['feature'].tolist()
    elif ffa_importance is not None:
        feature_names = ffa_importance['feature'].tolist()
    elif shap_values is not None:
        feature_names = [f"feature_{i}" for i in range(shap_values.shape[1])]
    else:
        logger.error("Cannot determine feature names - need SHAP or FFA importance")
        return
    
    # Calculate consensus
    consensus_data = calculate_consensus_features(shap_importance, ffa_importance, args.top_k)
    
    # Combine importance scores
    combined_importance = combine_importance_scores(
        shap_importance, ffa_importance, args.weight_shap, args.weight_ffa
    )
    
    # Generate patient explanations
    patient_explanations = None
    if ffa_explanations is not None:
        patient_explanations = generate_patient_explanations(
            shap_values, ffa_explanations, feature_names, args.n_patients
        )
    
    # Save results
    consensus_path = output_dir / 'consensus_features.json'
    with open(consensus_path, 'w') as f:
        json.dump(consensus_data, f, indent=2)
    logger.info(f"Saved consensus features to {consensus_path}")
    
    if not combined_importance.empty:
        combined_path = output_dir / 'combined_importance.csv'
        combined_importance.to_csv(combined_path, index=False)
        logger.info(f"Saved combined importance to {combined_path}")
    
    if patient_explanations is not None:
        explanations_path = output_dir / 'patient_explanations.csv'
        patient_explanations.to_csv(explanations_path, index=False)
        logger.info(f"Saved patient explanations to {explanations_path}")
    
    # Generate summary report
    summary = generate_summary_report(consensus_data, combined_importance, patient_explanations)
    summary_path = output_dir / 'summary_report.txt'
    with open(summary_path, 'w') as f:
        f.write(summary)
    logger.info(f"Saved summary report to {summary_path}")
    
    # Print summary
    print("\n" + summary)


if __name__ == "__main__":
    main()

