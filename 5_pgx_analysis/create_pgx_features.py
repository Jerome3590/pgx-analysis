#!/usr/bin/env python3
"""
Create PGx-enriched features from drug-gene mappings and allele frequencies.

This script creates features that combine drug usage patterns with pharmacogenomic
information, including both global and population-specific allele frequencies.
The model will determine which frequency approach is most predictive.
"""

import sys
import pandas as pd
from pathlib import Path
import logging
from typing import Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))  # noqa: E402

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_pgx_features(drug_features: pd.DataFrame,
                        drug_gene_mappings: pd.DataFrame,
                        allele_frequencies: pd.DataFrame,
                        patient_demographics: Optional[pd.DataFrame] = None,
                        output_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Create PGx-enriched features combining drug importance with allele frequencies.
    
    **Approach**: Creates multiple feature variants:
    1. Global frequency features (default)
    2. Population-specific frequency features (if demographics provided)
    3. Let the model/algorithm determine which approach is most predictive
    
    Parameters:
    -----------
    drug_features : pd.DataFrame
        Drug features from feature importance (must have 'feature' and 'importance' columns)
    drug_gene_mappings : pd.DataFrame
        Drug-gene mappings from map_drugs_to_genes.py
    allele_frequencies : pd.DataFrame
        Allele frequencies from add_allele_frequencies.py
    patient_demographics : pd.DataFrame, optional
        Patient demographics with race/ethnicity (for population-specific features)
    output_path : Path, optional
        Path to save the enriched features CSV file
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with PGx-enriched features
    """
    logger.info("Creating PGx-enriched features...")
    logger.info("Creating both global and population-specific frequency features")
    logger.info("Model will determine which approach is most predictive")
    
    # Extract drug names from feature importance
    drug_features_filtered = drug_features[
        drug_features['feature'].str.startswith('DRUG:', na=False)
    ].copy()
    
    drug_features_filtered['drug_name'] = drug_features_filtered['feature'].str.replace('DRUG:', '', regex=False)
    
    # Merge with drug-gene mappings
    pgx_features = drug_features_filtered.merge(
        drug_gene_mappings[['drug_name', 'gene', 'relationship_type', 'cpic_level', 'guideline_id']],
        on='drug_name',
        how='left'
    )
    
    # Merge with allele frequencies
    pgx_features = pgx_features.merge(
        allele_frequencies[
            ['gene', 'variant_id', 'allele_name', 'allele_frequency_global',
             'allele_frequency_afr', 'allele_frequency_amr', 'allele_frequency_eas',
             'allele_frequency_eur', 'allele_frequency_sas', 'allele_frequency_assigned',
             'frequency_assignment_method']
        ],
        on='gene',
        how='left'
    )
    
    # Create feature variants for model evaluation
    
    # 1. Global frequency features (baseline)
    pgx_features['pgx_risk_global'] = (
        pgx_features['importance'] *
        pgx_features['allele_frequency_global'].fillna(0)
    )
    
    # 2. Population-specific frequency features (if available)
    # These will be evaluated by the model to see if they improve predictions
    pgx_features['pgx_risk_afr'] = (
        pgx_features['importance'] *
        pgx_features['allele_frequency_afr'].fillna(0)
    )
    pgx_features['pgx_risk_amr'] = (
        pgx_features['importance'] *
        pgx_features['allele_frequency_amr'].fillna(0)
    )
    pgx_features['pgx_risk_eas'] = (
        pgx_features['importance'] *
        pgx_features['allele_frequency_eas'].fillna(0)
    )
    pgx_features['pgx_risk_eur'] = (
        pgx_features['importance'] *
        pgx_features['allele_frequency_eur'].fillna(0)
    )
    pgx_features['pgx_risk_sas'] = (
        pgx_features['importance'] *
        pgx_features['allele_frequency_sas'].fillna(0)
    )
    
    # 3. Assigned frequency feature (uses patient demographics if available, else global)
    pgx_features['pgx_risk_assigned'] = (
        pgx_features['importance'] *
        pgx_features['allele_frequency_assigned'].fillna(
            pgx_features['allele_frequency_global']
        ).fillna(0)
    )
    
    # 4. Gene-level aggregated features
    gene_aggregated = pgx_features.groupby('gene').agg({
        'importance': 'sum',
        'allele_frequency_global': 'first',
        'allele_frequency_afr': 'first',
        'allele_frequency_amr': 'first',
        'allele_frequency_eas': 'first',
        'allele_frequency_eur': 'first',
        'allele_frequency_sas': 'first',
        'pgx_risk_global': 'sum',
        'pgx_risk_afr': 'sum',
        'pgx_risk_amr': 'sum',
        'pgx_risk_eas': 'sum',
        'pgx_risk_eur': 'sum',
        'pgx_risk_sas': 'sum',
        'pgx_risk_assigned': 'sum'
    }).reset_index()
    
    gene_aggregated.columns = ['gene'] + [f'gene_{col}' if col != 'gene' else col
                                          for col in gene_aggregated.columns[1:]]
    
    # Create summary statistics
    summary_stats = {
        'total_drugs': len(drug_features_filtered),
        'drugs_with_pgx_mapping': pgx_features['gene'].notna().sum(),
        'unique_genes': pgx_features['gene'].nunique(),
        'genes_with_frequency_data': pgx_features['allele_frequency_global'].notna().sum(),
        'features_created': len(pgx_features.columns),
        'feature_types': [
            'pgx_risk_global',
            'pgx_risk_afr', 'pgx_risk_amr', 'pgx_risk_eas', 'pgx_risk_eur', 'pgx_risk_sas',
            'pgx_risk_assigned'
        ]
    }
    
    logger.info("Created PGx features:")
    logger.info(f"  - Total drugs: {summary_stats['total_drugs']}")
    logger.info(f"  - Drugs with PGx mapping: {summary_stats['drugs_with_pgx_mapping']}")
    logger.info(f"  - Unique genes: {summary_stats['unique_genes']}")
    logger.info(f"  - Feature variants: {len(summary_stats['feature_types'])}")
    logger.info("  - Model will evaluate which frequency approach is most predictive")
    
    # Save to file if output path provided
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pgx_features.to_csv(output_path, index=False)
        logger.info(f"Saved PGx-enriched features to {output_path}")
        
        # Save summary statistics
        summary_path = output_path.parent / f"{output_path.stem}_summary_stats.csv"
        pd.DataFrame([summary_stats]).to_csv(summary_path, index=False)
        logger.info(f"Saved summary statistics to {summary_path}")
    
    return pgx_features


def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Create PGx-enriched features")
    parser.add_argument("--cohort", required=True, help="Cohort name (e.g., opioid_ed)")
    parser.add_argument("--age_band", required=True, help="Age band (e.g., 0-12)")
    parser.add_argument("--drug_features", help="Path to drug features CSV (optional)")
    parser.add_argument("--drug_gene_mappings", help="Path to drug-gene mappings CSV (optional)")
    parser.add_argument("--allele_frequencies", help="Path to allele frequencies CSV (optional)")
    parser.add_argument("--patient_demographics", help="Path to patient demographics CSV (optional)")
    parser.add_argument("--output", help="Output CSV path (optional)")
    
    args = parser.parse_args()
    
    # Load drug features
    if args.drug_features:
        drug_features_path = Path(args.drug_features)
    else:
        drug_features_path = (
            PROJECT_ROOT / "3_feature_importance" / "outputs" /
            f"{args.cohort}_{args.age_band.replace('-', '_')}_aggregated_feature_importance.csv"
        )
    
    if not drug_features_path.exists():
        logger.error(f"Drug features file not found at {drug_features_path}")
        return
    
    drug_features = pd.read_csv(drug_features_path)
    
    # Load drug-gene mappings
    if args.drug_gene_mappings:
        mappings_path = Path(args.drug_gene_mappings)
    else:
        mappings_path = (
            PROJECT_ROOT / "7_pgx_analysis" / "outputs" / args.cohort /
            args.age_band.replace("-", "_") /
            f"{args.cohort}_{args.age_band.replace('-', '_')}_drug_gene_mappings.csv"
        )
    
    if not mappings_path.exists():
        logger.error(f"Drug-gene mappings file not found at {mappings_path}")
        logger.error("Please run map_drugs_to_genes.py first")
        return
    
    drug_gene_mappings = pd.read_csv(mappings_path)
    
    # Load allele frequencies
    if args.allele_frequencies:
        freq_path = Path(args.allele_frequencies)
    else:
        freq_path = (
            PROJECT_ROOT / "7_pgx_analysis" / "outputs" / args.cohort /
            args.age_band.replace("-", "_") /
            f"{args.cohort}_{args.age_band.replace('-', '_')}_allele_frequencies.csv"
        )
    
    if not freq_path.exists():
        logger.error(f"Allele frequencies file not found at {freq_path}")
        logger.error("Please run add_allele_frequencies.py first")
        return
    
    allele_frequencies = pd.read_csv(freq_path)
    
    # Load patient demographics if provided
    patient_demographics = None
    if args.patient_demographics:
        patient_demographics = pd.read_csv(args.patient_demographics)
    
    # Set output path
    if not args.output:
        args.output = (
            PROJECT_ROOT / "7_pgx_analysis" / "outputs" / args.cohort /
            args.age_band.replace("-", "_") /
            f"{args.cohort}_{args.age_band.replace('-', '_')}_pgx_enriched_features.csv"
        )
    
    # Create PGx features
    pgx_features = create_pgx_features(
        drug_features=drug_features,
        drug_gene_mappings=drug_gene_mappings,
        allele_frequencies=allele_frequencies,
        patient_demographics=patient_demographics,
        output_path=args.output
    )
    
    print(f"\nCreated {len(pgx_features)} PGx-enriched features")
    print("Feature variants available for model evaluation:")
    print("  - pgx_risk_global (baseline)")
    print("  - pgx_risk_afr, pgx_risk_amr, pgx_risk_eas, pgx_risk_eur, pgx_risk_sas (population-specific)")
    print("  - pgx_risk_assigned (uses demographics if available)")
    print("\nModel will determine which frequency approach improves predictions")


if __name__ == "__main__":
    main()

