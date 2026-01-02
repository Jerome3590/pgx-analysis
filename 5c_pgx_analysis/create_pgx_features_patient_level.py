#!/usr/bin/env python3
"""
Create patient-level PGx features from drug-gene mappings and allele frequencies.

This script creates patient-level features that combine drug usage patterns with 
pharmacogenomic information, including both global and population-specific allele frequencies.
The model will determine which frequency approach is most predictive.

Output:
- Saves to: outputs/feature_engineering/pgx_features_{cohort}_{age_band}.csv
- This intermediate file is then merged with other features by add_pgx_features_to_model_data.py
"""

import sys
import pandas as pd
from pathlib import Path
import logging
import subprocess
import shutil
import duckdb

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_patient_pgx_features(
    project_root: Path,
    cohort_name: str,
    age_band: str,
) -> pd.DataFrame:
    """
    Create patient-level PGx features.
    
    This function:
    1. Loads drug-gene mappings and allele frequencies
    2. Loads patient drug exposure data from model_data
    3. Creates patient-level PGx risk features
    
    Returns:
    --------
    pd.DataFrame
        Patient-level PGx features with mi_person_key
    """
    age_band_fname = age_band.replace("-", "_")
    
    # Load drug-gene mappings (prefer global, then cohort-global, then legacy age-band path)
    global_out_dir = project_root / "5c_pgx_analysis" / "outputs" / "global"
    global_mappings_path = global_out_dir / "pgx_drug_gene_mappings_global.csv"
    cohort_out_dir = project_root / "5c_pgx_analysis" / "outputs" / cohort_name
    cohort_mappings_path = cohort_out_dir / f"{cohort_name}_drug_gene_mappings.csv"
    legacy_mappings_path = (
        cohort_out_dir
        / age_band_fname
        / f"{cohort_name}_{age_band_fname}_drug_gene_mappings.csv"
    )
    if global_mappings_path.exists():
        mappings_path = global_mappings_path
    elif cohort_mappings_path.exists():
        mappings_path = cohort_mappings_path
    else:
        mappings_path = legacy_mappings_path
    
    # Load allele frequencies (prefer global, then cohort-global, then legacy age-band path)
    global_freq_path = global_out_dir / "pgx_allele_frequencies_global.csv"
    cohort_freq_path = cohort_out_dir / f"{cohort_name}_allele_frequencies.csv"
    legacy_freq_path = (
        cohort_out_dir
        / age_band_fname
        / f"{cohort_name}_{age_band_fname}_allele_frequencies.csv"
    )
    if global_freq_path.exists():
        frequencies_path = global_freq_path
    elif cohort_freq_path.exists():
        frequencies_path = cohort_freq_path
    else:
        frequencies_path = legacy_freq_path
    
    # Model data path (prefer protocol-filtered version if available)
    model_data_dir = (
        project_root
        / "4a_model_data"
        / f"cohort_name={cohort_name}"
        / f"age_band={age_band}"
    )
    model_data_filtered = model_data_dir / "model_events_no_protocols.parquet"
    model_data_path = (
        model_data_filtered
        if model_data_filtered.exists()
        else model_data_dir / "model_events.parquet"
    )
    
    if not model_data_path.exists():
        logger.error(f"Model data not found: {model_data_path}")
        return pd.DataFrame()
    
    # Get base patient list (both target and control)
    con = duckdb.connect()
    base_df = con.execute(
        f"""
        SELECT DISTINCT mi_person_key
        FROM read_parquet('{model_data_path}')
        WHERE target IN (0, 1)
        """
    ).df()
    con.close()
    
    if base_df.empty:
        logger.error("No target patients found in model_data")
        return pd.DataFrame()
    
    logger.info(f"Creating PGx features for {len(base_df)} patients")
    
    # Load drug-gene mappings if available
    drug_gene_mappings = pd.DataFrame()
    if mappings_path.exists():
        drug_gene_mappings = pd.read_csv(mappings_path)
        logger.info(f"Loaded {len(drug_gene_mappings)} drug-gene mappings")
    else:
        logger.warning(f"Drug-gene mappings not found at {mappings_path}")
        logger.info("Creating empty PGx features (no drug-gene mappings available)")
    
    # Load allele frequencies if available
    allele_frequencies = pd.DataFrame()
    if frequencies_path.exists():
        allele_frequencies = pd.read_csv(frequencies_path)
        logger.info(f"Loaded {len(allele_frequencies)} allele frequency records")
    else:
        logger.warning(f"Allele frequencies not found at {frequencies_path}")
    
    # If no mappings or frequencies, return empty features dataframe
    if drug_gene_mappings.empty or allele_frequencies.empty:
        logger.info("No PGx data available - returning empty feature set")
        features_df = base_df.copy()
        # Add placeholder columns
        features_df['pgx_risk_global'] = 0.0
        features_df['pgx_risk_afr'] = 0.0
        features_df['pgx_risk_amr'] = 0.0
        features_df['pgx_risk_eas'] = 0.0
        features_df['pgx_risk_eur'] = 0.0
        features_df['pgx_risk_sas'] = 0.0
        features_df['pgx_risk_assigned'] = 0.0
        features_df['pgx_drugs_with_mappings'] = 0
        features_df['pgx_genes_covered'] = 0
        return features_df
    
    # Extract patient drug exposures from model_data
    con = duckdb.connect()
    patient_drugs_query = f"""
    SELECT DISTINCT mi_person_key, drug_name
    FROM read_parquet('{model_data_path}')
    WHERE target IN (0, 1) AND drug_name IS NOT NULL AND drug_name != ''
    """
    patient_drugs_df = con.execute(patient_drugs_query).df()
    con.close()
    
    if patient_drugs_df.empty:
        logger.warning("No patient drug exposures found")
        features_df = base_df.copy()
        features_df['pgx_risk_global'] = 0.0
        return features_df
    
    # Merge patient drugs with drug-gene mappings
    patient_pgx = patient_drugs_df.merge(
        drug_gene_mappings[['drug_name', 'gene']].drop_duplicates(),
        on='drug_name',
        how='left'
    )
    
    # Merge with allele frequencies
    patient_pgx = patient_pgx.merge(
        allele_frequencies[
            ['gene', 'allele_frequency_global', 'allele_frequency_afr',
             'allele_frequency_amr', 'allele_frequency_eas', 'allele_frequency_eur',
             'allele_frequency_sas']
        ],
        on='gene',
        how='left'
    )
    
    # Aggregate to patient level
    # Filter to only drug-gene pairs that have mappings (gene is not null)
    patient_pgx_mapped = patient_pgx[patient_pgx['gene'].notna()].copy()
    
    if len(patient_pgx_mapped) > 0:
        patient_features = patient_pgx_mapped.groupby('mi_person_key').agg({
            'allele_frequency_global': ['mean', 'max', 'sum'],
            'allele_frequency_afr': ['mean', 'max', 'sum'],
            'allele_frequency_amr': ['mean', 'max', 'sum'],
            'allele_frequency_eas': ['mean', 'max', 'sum'],
            'allele_frequency_eur': ['mean', 'max', 'sum'],
            'allele_frequency_sas': ['mean', 'max', 'sum'],
            'drug_name': 'nunique',  # Count unique drugs with PGx mappings
            'gene': 'nunique'  # Count unique genes
        }).reset_index()
    else:
        # No mappings found, create empty dataframe with correct structure
        patient_features = pd.DataFrame(columns=['mi_person_key'])
        for pop in ['global', 'afr', 'amr', 'eas', 'eur', 'sas']:
            patient_features[f'allele_frequency_{pop}_mean'] = None
            patient_features[f'allele_frequency_{pop}_max'] = None
            patient_features[f'allele_frequency_{pop}_sum'] = None
        patient_features['drug_name'] = None
        patient_features['gene'] = None
    
    # Flatten column names
    patient_features.columns = [
        'mi_person_key',
        'pgx_risk_global_mean', 'pgx_risk_global_max', 'pgx_risk_global_sum',
        'pgx_risk_afr_mean', 'pgx_risk_afr_max', 'pgx_risk_afr_sum',
        'pgx_risk_amr_mean', 'pgx_risk_amr_max', 'pgx_risk_amr_sum',
        'pgx_risk_eas_mean', 'pgx_risk_eas_max', 'pgx_risk_eas_sum',
        'pgx_risk_eur_mean', 'pgx_risk_eur_max', 'pgx_risk_eur_sum',
        'pgx_risk_sas_mean', 'pgx_risk_sas_max', 'pgx_risk_sas_sum',
        'pgx_drugs_with_mappings', 'pgx_genes_covered'
    ]
    
    # Merge with base patient list
    features_df = base_df.merge(patient_features, on='mi_person_key', how='left')
    
    # Fill NaN with 0
    for col in features_df.columns:
        if col != 'mi_person_key':
            features_df[col] = features_df[col].fillna(0.0)
    
    logger.info(f"Created {len(features_df.columns) - 1} PGx features for {len(features_df)} patients")
    
    return features_df


def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Create patient-level PGx features")
    parser.add_argument("--cohort", required=True, help="Cohort name (e.g., opioid_ed)")
    parser.add_argument("--age_band", required=True, help="Age band (e.g., 0-12)")
    parser.add_argument("--output", help="Output CSV path (optional)")
    
    args = parser.parse_args()
    
    project_root = PROJECT_ROOT
    
    # Create PGx features
    pgx_features = create_patient_pgx_features(
        project_root=project_root,
        cohort_name=args.cohort,
        age_band=args.age_band
    )
    
    if pgx_features.empty:
        logger.error("No features created. Check inputs and logs.")
        return
    
    # Set output path - intermediate file for PGx features only
    if not args.output:
        age_band_fname = args.age_band.replace("-", "_")
        feature_eng_dir = (
            project_root
            / "5c_pgx_analysis"
            / "outputs"
            / "feature_engineering"
        )
        feature_eng_dir.mkdir(parents=True, exist_ok=True)
        args.output = feature_eng_dir / f"pgx_features_{args.cohort}_{age_band_fname}.csv"
    
    # Save features
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pgx_features.to_csv(output_path, index=False)
    
    print(f"\nCreated {len(pgx_features.columns) - 1} PGx features for {len(pgx_features)} patients")
    print("Output format: Ready for merging with other features (uses mi_person_key)")
    print(f"Saved to: {output_path}")
    
    # Upload to S3 gold location (intermediate file; legacy mirror)
    age_band_fname = args.age_band.replace("-", "_")
    s3_path = f"s3://pgxdatalake/gold/feature_engineering/7_pgx/{args.cohort}/{args.age_band}/pgx_features_{args.cohort}_{age_band_fname}.csv"
    
    # Check for AWS CLI
    aws_cli = shutil.which("aws")
    if aws_cli:
        try:
            print(f"\nUploading to S3: {s3_path}")
            subprocess.run(
                [aws_cli, "s3", "cp", str(output_path), s3_path],
                capture_output=True,
                text=True,
                check=True
            )
            print(f"S3 upload successful: {s3_path}")
        except subprocess.CalledProcessError as e:
            logger.warning(f"S3 upload failed: {e.stderr}")
            print(f"Warning: Could not upload to S3: {e.stderr}")
    else:
        logger.info("AWS CLI not found, skipping S3 upload")
        print("Note: AWS CLI not found, skipping S3 upload")
    
    print(f"\nFeature columns ({len(pgx_features.columns)} total):")
    for col in pgx_features.columns[:20]:  # Show first 20
        print(f"  - {col}")
    if len(pgx_features.columns) > 20:
        print(f"  ... and {len(pgx_features.columns) - 20} more")


if __name__ == "__main__":
    main()

