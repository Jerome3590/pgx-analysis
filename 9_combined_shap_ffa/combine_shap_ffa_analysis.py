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


def load_ffa_results(ffa_path: Path, ffa_importance_path: Optional[Path] = None):
    """Load FFA AXP explanations and/or importance."""
    ffa_explanations = None
    ffa_importance = None
    
    # If ffa_path is actually an importance file, load as importance
    if ffa_path.name.endswith('_importance.csv') or 'importance' in ffa_path.name.lower():
        ffa_importance = pd.read_csv(ffa_path)
        logger.info(f"Loaded FFA importance from {ffa_path}: {len(ffa_importance)} features")
    else:
        # Try to load as explanations
        logger.info(f"Loading FFA explanations from {ffa_path}")
        ffa_explanations = pd.read_csv(ffa_path)
        logger.info(f"Loaded {len(ffa_explanations)} FFA explanations")
    
    # Load importance separately if provided
    if ffa_importance_path and ffa_importance_path.exists():
        ffa_importance = pd.read_csv(ffa_importance_path)
        logger.info(f"Loaded FFA importance from {ffa_importance_path}: {len(ffa_importance)} features")
    
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
    parser.add_argument("--age-band", dest="age_band", help="Age band (e.g., 13-24)")
    parser.add_argument("--age_band", help="Age band (alternative to --age-band)")
    parser.add_argument("--shap-values-path", help="Path to SHAP values (auto-detected if not provided)")
    parser.add_argument("--ffa-explanations-path", help="Path to FFA explanations CSV (auto-detected if not provided)")
    parser.add_argument("--shap-importance-path", help="Path to SHAP importance CSV (auto-detected if not provided)")
    parser.add_argument("--ffa-importance-path", help="Path to FFA importance CSV (auto-detected if not provided)")
    parser.add_argument("--feature-names-path", help="Path to feature names file (optional)")
    parser.add_argument("--output-dir", help="Output directory (default: 9_combined_shap_ffa/outputs)")
    parser.add_argument("--top-k", type=int, default=20, help="Top K features for consensus")
    parser.add_argument("--weight-shap", type=float, default=0.5, help="Weight for SHAP (0-1)")
    parser.add_argument("--weight-ffa", type=float, default=0.5, help="Weight for FFA (0-1)")

    args = parser.parse_args()
    
    # Handle both --age-band and --age_band for compatibility
    age_band = getattr(args, 'age_band', None)
    if not age_band:
        raise ValueError("Must provide --age-band or --age_band")
    
    age_band_fname = age_band.replace("-", "_")
    project_root = Path(__file__).parent.parent

    # Auto-detect paths if not provided
    if not args.shap_importance_path:
        # Try to find SHAP global importance CSV
        shap_importance_path = (
            project_root
            / "8_shap_analysis"
            / "outputs"
            / args.cohort
            / age_band_fname
            / f"{args.cohort}_{age_band_fname}_shap_global_importance_xgboost.csv"
        )
        if shap_importance_path.exists():
            args.shap_importance_path = str(shap_importance_path)
            logger.info(f"Auto-detected SHAP importance: {shap_importance_path}")
        else:
            logger.warning(f"SHAP importance not found at {shap_importance_path}")
    
    if not args.shap_values_path:
        # Try to find SHAP sample values parquet
        shap_values_path = (
            project_root
            / "8_shap_analysis"
            / "outputs"
            / args.cohort
            / age_band_fname
            / f"{args.cohort}_{age_band_fname}_shap_sample_values_xgboost.parquet"
        )
        if shap_values_path.exists():
            args.shap_values_path = str(shap_values_path)
            logger.info(f"Auto-detected SHAP values: {shap_values_path}")
        else:
            logger.warning(f"SHAP values not found at {shap_values_path}")
    
    if not args.ffa_importance_path:
        # Try combined weighted importance first, then fall back to core AXP importance
        ffa_importance_path = (
            project_root
            / "7_ffa_analysis"
            / "outputs"
            / args.cohort
            / age_band_fname
            / "visualizations"
            / "combined_weighted_feature_importance.csv"
        )
        if not ffa_importance_path.exists():
            ffa_importance_path = (
                project_root
                / "7_ffa_analysis"
                / "outputs"
                / args.cohort
                / age_band_fname
                / "xgboost"
                / "feature_importance_axp.csv"
            )
        if ffa_importance_path.exists():
            args.ffa_importance_path = str(ffa_importance_path)
            logger.info(f"Auto-detected FFA importance: {ffa_importance_path}")
        else:
            logger.warning(f"FFA importance not found at {ffa_importance_path}")
    
    if not args.ffa_explanations_path:
        # Try to find FFA AXP explanations
        ffa_explanations_path = (
            project_root
            / "7_ffa_analysis"
            / "outputs"
            / args.cohort
            / age_band_fname
            / "xgboost"
            / "axp_explanations.csv"
        )
        if ffa_explanations_path.exists():
            args.ffa_explanations_path = str(ffa_explanations_path)
            logger.info(f"Auto-detected FFA explanations: {ffa_explanations_path}")
        else:
            logger.warning(f"FFA explanations not found at {ffa_explanations_path}")
    
    # Set default output directory
    if not args.output_dir:
        args.output_dir = str(project_root / "9_combined_shap_ffa" / "outputs")

    # Check S3 for existing outputs (idempotency)
    try:
        from py_helpers.checkpoint_utils import check_step_outputs_exist, check_step_checkpoint_exists

        s3_output_paths = [
            f"s3://pgxdatalake/gold/combined_analysis/{args.cohort}/{age_band}/{args.cohort}_{age_band_fname}_combined_shap_ffa_importance.csv",
            f"s3://pgxdatalake/gold/combined_analysis/{args.cohort}/{age_band}/{args.cohort}_{age_band_fname}_consensus_features.json",
        ]

        if check_step_outputs_exist(s3_output_paths) or check_step_checkpoint_exists("9_combined_shap_ffa", args.cohort, age_band):
            logger.info(f"Step 9 outputs already exist in S3 for {args.cohort}/{age_band}; skipping.")
            print(f"[SKIP] Step 9 outputs already exist in S3 for {args.cohort}/{age_band}")
            return
    except ImportError:
        pass  # Fallback to local-only if checkpoint_utils not available

    output_dir = Path(args.output_dir) / args.cohort / age_band_fname
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load SHAP results (require at least importance, values are optional)
    if not args.shap_importance_path:
        raise FileNotFoundError(
            f"SHAP importance not found. Expected at: "
            f"{project_root / '8_shap_analysis' / 'outputs' / args.cohort / age_band_fname / f'{args.cohort}_{age_band_fname}_shap_global_importance_xgboost.csv'}"
        )
    
    # Load SHAP importance directly
    shap_importance = pd.read_csv(args.shap_importance_path)
    logger.info(f"Loaded SHAP importance: {len(shap_importance)} features")
    
    # Load SHAP values if available
    shap_values = None
    if args.shap_values_path and Path(args.shap_values_path).exists():
        shap_path = Path(args.shap_values_path)
        if shap_path.suffix == '.parquet':
            shap_df = pd.read_parquet(shap_path)
            # Remove index column if present
            if 'mi_person_key' in shap_df.columns:
                shap_values = shap_df.drop(columns=['mi_person_key']).values
            else:
                shap_values = shap_df.values
            logger.info(f"Loaded SHAP values from parquet: {shap_values.shape}")
        elif shap_path.suffix == '.npy':
            shap_values = np.load(shap_path)
            logger.info(f"Loaded SHAP values from numpy: {shap_values.shape}")
        elif shap_path.suffix == '.csv':
            shap_values = pd.read_csv(shap_path).values
            logger.info(f"Loaded SHAP values from CSV: {shap_values.shape}")
    
    # Load FFA results (require at least importance, explanations are optional)
    if not args.ffa_importance_path:
        raise FileNotFoundError(
            f"FFA importance not found. Checked:\n"
            f"  - {project_root / '7_ffa_analysis' / 'outputs' / args.cohort / age_band_fname / 'visualizations' / 'combined_weighted_feature_importance.csv'}\n"
            f"  - {project_root / '7_ffa_analysis' / 'outputs' / args.cohort / age_band_fname / 'xgboost' / 'feature_importance_axp.csv'}"
        )
    
    # Load FFA importance directly
    ffa_importance = pd.read_csv(args.ffa_importance_path)
    logger.info(f"Loaded FFA importance: {len(ffa_importance)} features")
    
    # Load FFA explanations if available
    ffa_explanations = None
    if args.ffa_explanations_path and Path(args.ffa_explanations_path).exists():
        ffa_explanations = pd.read_csv(args.ffa_explanations_path)
        logger.info(f"Loaded FFA explanations: {len(ffa_explanations)} rows")
    
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
    if consensus_features:
        print(f"\nConsensus Features (top {args.top_k}): {len(consensus_features)}")
        print(f"  {', '.join(list(consensus_features)[:10])}")
    print(f"\nCombined Importance (top 10):")
    display_cols = ['feature', 'combined_importance']
    if 'shap_norm' in combined_importance.columns:
        display_cols.append('shap_norm')
    if 'ffa_norm' in combined_importance.columns:
        display_cols.append('ffa_norm')
    display_cols = [c for c in display_cols if c in combined_importance.columns]
    print(combined_importance.head(10)[display_cols].to_string(index=False))
    print(f"\nResults saved to: {output_dir}")

    # Upload outputs to S3 and save checkpoint
    try:
        from py_helpers.checkpoint_utils import upload_file_to_s3, save_step_checkpoint

        s3_outputs = []

        if combined_path.exists():
            s3_combined = f"s3://pgxdatalake/gold/combined_analysis/{args.cohort}/{age_band}/{args.cohort}_{age_band_fname}_combined_shap_ffa_importance.csv"
            if upload_file_to_s3(combined_path, s3_combined, logger):
                s3_outputs.append(s3_combined)

        consensus_path = output_dir / f"{args.cohort}_{age_band_fname}_consensus_features.json"
        if consensus_path.exists():
            s3_consensus = f"s3://pgxdatalake/gold/combined_analysis/{args.cohort}/{age_band}/{args.cohort}_{age_band_fname}_consensus_features.json"
            if upload_file_to_s3(consensus_path, s3_consensus, logger):
                s3_outputs.append(s3_consensus)

        patients_path = output_dir / f"{args.cohort}_{age_band_fname}_patient_analyses.json"
        if patients_path.exists():
            s3_patients = f"s3://pgxdatalake/gold/combined_analysis/{args.cohort}/{age_band}/{args.cohort}_{age_band_fname}_patient_analyses.json"
            if upload_file_to_s3(patients_path, s3_patients, logger):
                s3_outputs.append(s3_patients)

        # Save checkpoint
        if s3_outputs:
            save_step_checkpoint(
                step_name="9_combined_shap_ffa",
                cohort=args.cohort,
                age_band=age_band,
                metadata={"top_k": args.top_k, "weight_shap": args.weight_shap, "weight_ffa": args.weight_ffa},
                output_paths=s3_outputs,
                logger=logger,
            )
    except ImportError:
        pass  # Checkpoint saving is optional


if __name__ == "__main__":
    main()

