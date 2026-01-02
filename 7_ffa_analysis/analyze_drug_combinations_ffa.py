#!/usr/bin/env python3
"""
Analyze specific drug combinations that drive outcomes at the row/patient level.

This script combines:
1. Feature importance results (which drugs matter overall)
2. FPGrowth patterns (frequent drug combinations)
3. Patient-level predictions (which patients are high-risk)
4. Actual drug combinations in high-risk patients

Usage:
    python 3_feature_importance/analyze_drug_combinations.py \
        --cohort non_opioid_ed \
        --age-band 65-74 \
        --feature-importance-file 3_feature_importance/outputs/non_opioid_ed_65_74_aggregated_feature_importance.csv \
        --model-data-path path/to/model_data.parquet
"""

import sys
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter
import warnings
warnings.filterwarnings("ignore")

# Add project root to path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_feature_importance(feature_file: Path, top_k: int = 50):
    """Load and filter top features from feature importance results."""
    df = pd.read_csv(feature_file)
    
    # Sort by importance and get top K
    if 'importance_mean' in df.columns:
        df = df.sort_values('importance_mean', ascending=False)
    elif 'importance' in df.columns:
        df = df.sort_values('importance', ascending=False)
    else:
        logger.warning("No importance column found. Using all features.")
        return df
    
    top_features = df.head(top_k)
    logger.info(f"Loaded top {len(top_features)} features from {feature_file}")
    
    return top_features


def extract_drug_names_from_features(feature_df: pd.DataFrame):
    """Extract drug names from feature names."""
    drug_names = []
    
    for feature_name in feature_df['feature'].values:
        # Handle different feature naming conventions
        if 'item_' in str(feature_name).lower():
            drug_name = str(feature_name).replace('item_', '').strip()
        elif 'DRUG:' in str(feature_name):
            drug_name = str(feature_name).replace('DRUG:', '').strip()
        else:
            drug_name = str(feature_name).strip()
        
        if drug_name and drug_name != 'nan':
            drug_names.append(drug_name)
    
    return drug_names


def analyze_patient_drug_combinations(
    model_data_path: Path,
    cohort_name: str,
    age_band: str,
    important_drugs: list,
    min_drugs_per_patient: int = 2,
    target_outcome: int = 1
):
    """
    Analyze which specific drug combinations appear in patients with outcomes.
    
    Args:
        model_data_path: Path to model data parquet file
        cohort_name: Cohort name
        age_band: Age band
        important_drugs: List of important drug names from feature importance
        min_drugs_per_patient: Minimum number of drugs to consider a combination
        target_outcome: Target outcome value (1 = case, 0 = control)
        
    Returns:
        DataFrame with patient-level drug combination analysis
    """
    import duckdb
    
    logger.info(f"Loading patient drug data from {model_data_path}...")
    
    # Query patient drug combinations
    query = f"""
    SELECT 
        mi_person_key,
        drug_name,
        target,
        is_target_case
    FROM read_parquet('{model_data_path}')
    WHERE drug_name IS NOT NULL 
      AND drug_name != ''
      AND age_band = '{age_band}'
    """
    
    con = duckdb.connect()
    drug_data = con.execute(query).df()
    con.close()
    
    logger.info(f"Loaded {len(drug_data):,} drug records for {drug_data['mi_person_key'].nunique():,} patients")
    
    # Filter to important drugs
    if important_drugs:
        drug_data = drug_data[drug_data['drug_name'].isin(important_drugs)]
        logger.info(f"Filtered to {len(drug_data):,} records with important drugs")
    
    # Get patients with outcomes
    patients_with_outcome = drug_data[drug_data['is_target_case'] == target_outcome]
    patients_without_outcome = drug_data[drug_data['is_target_case'] != target_outcome]
    
    logger.info(f"Patients with outcome: {patients_with_outcome['mi_person_key'].nunique():,}")
    logger.info(f"Patients without outcome: {patients_without_outcome['mi_person_key'].nunique():,}")
    
    # Analyze drug combinations per patient
    def get_patient_drugs(df):
        patient_drugs = df.groupby('mi_person_key')['drug_name'].apply(list).reset_index()
        patient_drugs['drug_count'] = patient_drugs['drug_name'].apply(len)
        patient_drugs = patient_drugs[patient_drugs['drug_count'] >= min_drugs_per_patient]
        return patient_drugs
    
    case_combinations = get_patient_drugs(patients_with_outcome)
    control_combinations = get_patient_drugs(patients_without_outcome)
    
    logger.info(f"Case patients with {min_drugs_per_patient}+ drugs: {len(case_combinations):,}")
    logger.info(f"Control patients with {min_drugs_per_patient}+ drugs: {len(control_combinations):,}")
    
    # Find frequent combinations in cases
    case_drug_sets = case_combinations['drug_name'].apply(lambda x: tuple(sorted(set(x))))
    case_combination_counts = Counter(case_drug_sets)
    
    # Find frequent combinations in controls
    control_drug_sets = control_combinations['drug_name'].apply(lambda x: tuple(sorted(set(x))))
    control_combination_counts = Counter(control_drug_sets)
    
    # Calculate combination statistics
    results = []
    for combination, case_count in case_combination_counts.most_common(100):
        control_count = control_combination_counts.get(combination, 0)
        total_count = case_count + control_count
        
        if total_count > 0:
            case_rate = case_count / total_count
            results.append({
                'drug_combination': ', '.join(combination),
                'n_drugs': len(combination),
                'case_count': case_count,
                'control_count': control_count,
                'total_count': total_count,
                'case_rate': case_rate,
                'drugs': list(combination)
            })
    
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('case_rate', ascending=False)
    
    return results_df, case_combinations, control_combinations


def main():
    parser = argparse.ArgumentParser(
        description="Analyze specific drug combinations driving outcomes"
    )
    parser.add_argument("--cohort", required=True, help="Cohort name")
    parser.add_argument("--age-band", required=True, help="Age band")
    parser.add_argument("--feature-importance-file", required=True,
                       help="Path to aggregated feature importance CSV")
    parser.add_argument("--model-data-path", required=True,
                       help="Path to model data parquet file")
    parser.add_argument("--top-k-features", type=int, default=50,
                       help="Top K features to consider")
    parser.add_argument("--min-drugs", type=int, default=2,
                       help="Minimum drugs per combination")
    parser.add_argument("--output-dir", default="3_feature_importance/outputs",
                       help="Output directory")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load feature importance
    logger.info("Loading feature importance...")
    feature_importance = load_feature_importance(
        Path(args.feature_importance_file),
        top_k=args.top_k_features
    )
    
    # Extract important drug names
    important_drugs = extract_drug_names_from_features(feature_importance)
    logger.info(f"Found {len(important_drugs)} important drugs")
    logger.info(f"Sample drugs: {important_drugs[:10]}")
    
    # Analyze patient drug combinations
    logger.info("Analyzing patient drug combinations...")
    combinations_df, case_patients, control_patients = analyze_patient_drug_combinations(
        Path(args.model_data_path),
        args.cohort,
        args.age_band,
        important_drugs,
        min_drugs_per_patient=args.min_drugs
    )
    
    # Save results
    output_file = output_dir / f"{args.cohort}_{args.age_band}_drug_combinations.csv"
    combinations_df.to_csv(output_file, index=False)
    logger.info(f"Saved drug combinations to {output_file}")
    
    # Print summary
    print("\n" + "="*70)
    print("TOP DRUG COMBINATIONS ASSOCIATED WITH OUTCOMES")
    print("="*70)
    print(f"\nTop 10 combinations by case rate:")
    for idx, row in combinations_df.head(10).iterrows():
        print(f"\n{idx+1}. {row['drug_combination']}")
        print(f"   Case rate: {row['case_rate']:.1%} ({row['case_count']} cases, {row['control_count']} controls)")
        print(f"   Total patients: {row['total_count']}")
    
    print(f"\n\nFull results saved to: {output_file}")


if __name__ == "__main__":
    main()

