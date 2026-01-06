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
    global_out_dir = project_root / "5_pgx_analysis" / "outputs" / "global"
    global_mappings_path = global_out_dir / "pgx_drug_gene_mappings_global.csv"
    cohort_out_dir = project_root / "5_pgx_analysis" / "outputs" / cohort_name
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
    
    # Model data path resolution (same logic as Step 4b)
    # Prefer protocol-filtered version, check multiple locations (NVMe, EBS, S3)
    from py_helpers.env_utils import get_data_root, is_linux
    
    data_root = get_data_root()
    is_linux_system = is_linux()
    
    # Build candidate paths - prioritize data root on Linux, project root on Windows
    if is_linux_system:
        # On Linux/EC2: prioritize /mnt/nvme
        candidates_filtered = [
            data_root / "4a_model_data" / f"cohort_name={cohort_name}" / f"age_band={age_band}" / "model_events_no_protocols.parquet",
            project_root / "4a_model_data" / f"cohort_name={cohort_name}" / f"age_band={age_band}" / "model_events_no_protocols.parquet",
        ]
        candidates_unfiltered = [
            data_root / "4a_model_data" / f"cohort_name={cohort_name}" / f"age_band={age_band}" / "model_events.parquet",
            project_root / "4a_model_data" / f"cohort_name={cohort_name}" / f"age_band={age_band}" / "model_events.parquet",
        ]
    else:
        # On Windows: prioritize project root
        candidates_filtered = [
            project_root / "4a_model_data" / f"cohort_name={cohort_name}" / f"age_band={age_band}" / "model_events_no_protocols.parquet",
            data_root / "4a_model_data" / f"cohort_name={cohort_name}" / f"age_band={age_band}" / "model_events_no_protocols.parquet",
        ]
        candidates_unfiltered = [
            project_root / "4a_model_data" / f"cohort_name={cohort_name}" / f"age_band={age_band}" / "model_events.parquet",
            data_root / "4a_model_data" / f"cohort_name={cohort_name}" / f"age_band={age_band}" / "model_events.parquet",
        ]
    
    # First try filtered version (preferred)
    model_data_path = None
    for path in candidates_filtered:
        if path.exists():
            model_data_path = path
            logger.info(f"Found filtered model data at: {model_data_path}")
            break
    
    # If filtered not found, try unfiltered version
    if model_data_path is None:
        for path in candidates_unfiltered:
            if path.exists():
                model_data_path = path
                logger.info(f"Found unfiltered model data at: {model_data_path}")
                break
    
    # If still not found, try downloading from S3
    if model_data_path is None:
        try:
            from py_helpers.common_imports import s3_client, S3_BUCKET
        except ImportError:
            import boto3
            s3_client = boto3.client("s3")
            S3_BUCKET = "pgxdatalake"
        
        # Try filtered version first from S3
        s3_key_candidates = [
            f"gold/dtw_filter/{cohort_name}/{age_band}/model_events_no_protocols.parquet",
            f"gold/cohorts_model_data/cohort_name={cohort_name}/age_band={age_band}/model_events.parquet",
        ]
        
        download_dest = candidates_filtered[0] if is_linux_system else candidates_unfiltered[0]
        download_dest.parent.mkdir(parents=True, exist_ok=True)
        
        for s3_key in s3_key_candidates:
            try:
                s3_client.head_object(Bucket=S3_BUCKET, Key=s3_key)
                logger.info(f"Downloading model data from S3: s3://{S3_BUCKET}/{s3_key}")
                logger.info(f"Downloading to: {download_dest}")
                s3_client.download_file(S3_BUCKET, s3_key, str(download_dest))
                logger.info(f"Downloaded to: {download_dest}")
                model_data_path = download_dest
                break
            except Exception as e:
                logger.debug(f"S3 key not found or error: {s3_key} - {e}")
                continue
    
    if model_data_path is None or not model_data_path.exists():
        logger.error(f"Model data not found. Checked paths:")
        for path in candidates_filtered + candidates_unfiltered:
            logger.error(f"  - {path} (exists: {path.exists()})")
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
        
        # Validate: Check for duplicate columns
        duplicate_cols = drug_gene_mappings.columns[drug_gene_mappings.columns.duplicated()].tolist()
        if duplicate_cols:
            raise ValueError(
                f"Duplicate columns detected in drug-gene mappings file for {cohort_name}/{age_band}: {duplicate_cols}. "
                f"File: {mappings_path}. "
                f"This will cause issues in downstream processing. Please regenerate the mappings file."
            )
        
        logger.info(f"Loaded {len(drug_gene_mappings)} drug-gene mappings")
    else:
        logger.warning(f"Drug-gene mappings not found at {mappings_path}")
        logger.info("Creating empty PGx features (no drug-gene mappings available)")
    
    # Load allele frequencies if available
    allele_frequencies = pd.DataFrame()
    if frequencies_path.exists():
        allele_frequencies = pd.read_csv(frequencies_path)
        
        # Validate: Check for duplicate columns
        duplicate_cols = allele_frequencies.columns[allele_frequencies.columns.duplicated()].tolist()
        if duplicate_cols:
            raise ValueError(
                f"Duplicate columns detected in allele frequencies file for {cohort_name}/{age_band}: {duplicate_cols}. "
                f"File: {frequencies_path}. "
                f"This will cause issues in downstream processing. Please regenerate the frequencies file."
            )
        
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
    # Use drug_name for joining with patient data, but keep cpic_drug_name for CPIC data joins
    mapping_cols = ['drug_name', 'cpic_drug_name', 'gene']
    if 'cpic_drug_name' not in drug_gene_mappings.columns:
        # Fallback: if cpic_drug_name doesn't exist, create it from drug_name
        drug_gene_mappings['cpic_drug_name'] = drug_gene_mappings['drug_name']
    
    patient_pgx = patient_drugs_df.merge(
        drug_gene_mappings[mapping_cols].drop_duplicates(),
        on='drug_name',
        how='left'
    )
    
    # Merge with allele frequencies
    # Allele frequencies should use cpic_drug_name for joining with CPIC data
    freq_cols = ['gene', 'allele_frequency_global', 'allele_frequency_afr',
                 'allele_frequency_amr', 'allele_frequency_eas', 'allele_frequency_eur',
                 'allele_frequency_sas']
    
    # If allele frequencies have cpic_drug_name, use it for joining
    if 'cpic_drug_name' in allele_frequencies.columns and 'cpic_drug_name' in patient_pgx.columns:
        # Join on both gene and cpic_drug_name for more accurate matching
        patient_pgx = patient_pgx.merge(
            allele_frequencies[freq_cols + ['cpic_drug_name']],
            on=['gene', 'cpic_drug_name'],
            how='left'
        )
        # Fallback: if no match on cpic_drug_name, try gene only
        missing_mask = patient_pgx['allele_frequency_global'].isna() & patient_pgx['gene'].notna()
        if missing_mask.any():
            patient_pgx.loc[missing_mask, freq_cols[1:]] = patient_pgx.loc[missing_mask].merge(
                allele_frequencies[freq_cols],
                on='gene',
                how='left',
                suffixes=('', '_fallback')
            )[freq_cols[1:]]
    else:
        # Fallback: join on gene only
        patient_pgx = patient_pgx.merge(
            allele_frequencies[freq_cols],
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
            / "5_pgx_analysis"
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
    
    # Upload to S3 gold location (primary: gold/pgx_features/, also mirror to legacy location)
    age_band_fname = args.age_band.replace("-", "_")
    s3_path_primary = f"s3://pgxdatalake/gold/pgx_features/{args.cohort}/{args.age_band}/pgx_features_{args.cohort}_{age_band_fname}.csv"
    s3_path_legacy = f"s3://pgxdatalake/gold/feature_engineering/7_pgx/{args.cohort}/{args.age_band}/pgx_features_{args.cohort}_{age_band_fname}.csv"
    
    # Check for AWS CLI
    aws_cli = shutil.which("aws")
    if aws_cli:
        # Upload to primary location (gold/pgx_features/)
        try:
            print(f"\n[INFO] Uploading to S3 (primary): {s3_path_primary}")
            subprocess.run(
                [aws_cli, "s3", "cp", str(output_path), s3_path_primary],
                capture_output=True,
                text=True,
                check=True
            )
            print(f"[INFO] Primary S3 upload successful: {s3_path_primary}")
        except subprocess.CalledProcessError as e:
            print(f"[WARNING] Primary S3 upload failed: {e.stderr if e.stderr else 'Unknown error'}")
        
        # Also upload to legacy location for backward compatibility
        try:
            print(f"[INFO] Uploading to S3 (legacy): {s3_path_legacy}")
            subprocess.run(
                [aws_cli, "s3", "cp", str(output_path), s3_path_legacy],
                capture_output=True,
                text=True,
                check=True
            )
            print(f"[INFO] Legacy S3 upload successful: {s3_path_legacy}")
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

