#!/usr/bin/env python3
"""
Create patient-level PGx features: simple drug counts.

This script creates patient-level features by counting:
1. Total number of unique drugs per patient
2. Number of CPIC drugs (drugs with CPIC pharmacogenomic guidelines) per patient

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
    Create patient-level PGx features: simple drug counts.
    
    Uses DuckDB for the full pipeline: read parquet once, then base patient list,
    total distinct drugs per patient, and CPIC drugs per patient are computed in SQL.
    Only the small CPIC mapping CSV is loaded with pandas.

    1. Loads global drug-to-CPIC mapping (pandas) to identify CPIC drugs
    2. Single DuckDB query: read_parquet + CTEs for base, total counts, CPIC counts
    3. Returns one row per patient with pgx_num_drugs, pgx_num_cpic_drugs

    Returns:
    --------
    pd.DataFrame
        Patient-level PGx features with mi_person_key, pgx_num_drugs, pgx_num_cpic_drugs
    """
    age_band_fname = age_band.replace("-", "_")
    
    # Load global drug-to-CPIC mapping to identify CPIC drugs
    global_out_dir = project_root / "5_pgx_analysis" / "outputs" / "global"
    global_drug_mapping_path = global_out_dir / "drug_cpic_mapping_global.csv"
    
    # Try loading from S3 if local file doesn't exist
    cpic_drug_set = set()
    if global_drug_mapping_path.exists():
        try:
            drug_mapping_df = pd.read_csv(global_drug_mapping_path)
            if 'drug_name' in drug_mapping_df.columns:
                cpic_drug_set = set(drug_mapping_df['drug_name'].str.upper().str.strip())
                logger.info(f"Loaded {len(cpic_drug_set)} CPIC drugs from global mapping")
        except Exception as e:
            logger.warning(f"Could not load global drug mapping: {e}")
    else:
        # Try downloading from S3
        try:
            import boto3
            from py_helpers.constants import S3_BUCKET
            
            s3_client = boto3.client('s3')
            s3_key = "gold/pgx_features/global/drug_cpic_mapping_global.csv"
            
            global_drug_mapping_path.parent.mkdir(parents=True, exist_ok=True)
            s3_client.download_file(S3_BUCKET, s3_key, str(global_drug_mapping_path))
            logger.info(f"Downloaded global drug mapping from S3")
            
            drug_mapping_df = pd.read_csv(global_drug_mapping_path)
            if 'drug_name' in drug_mapping_df.columns:
                cpic_drug_set = set(drug_mapping_df['drug_name'].str.upper().str.strip())
                logger.info(f"Loaded {len(cpic_drug_set)} CPIC drugs from S3")
        except Exception as e:
            logger.warning(f"Could not download global drug mapping from S3: {e}")
    
    if not cpic_drug_set:
        logger.warning("No CPIC drug mapping found. Will count all drugs as non-CPIC.")
    
    # Model data path resolution: single canonical location (get_model_data_root())
    from py_helpers.env_utils import get_model_data_root
    
    model_data_root = get_model_data_root()
    base_dir = model_data_root / f"cohort_name={cohort_name}" / f"age_band={age_band}"
    candidates_filtered = [base_dir / "model_events_no_protocols.parquet"]
    candidates_unfiltered = [base_dir / "model_events.parquet"]
    
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
        
        download_dest = candidates_filtered[0]  # prefer filtered path for S3 download
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
    
    # Single DuckDB query: read parquet once, aggregate in SQL (base, total drugs, CPIC drugs)
    cpic_list = list(cpic_drug_set)
    con = duckdb.connect()
    features_df = con.execute(
        """
        WITH data AS (
            SELECT mi_person_key, drug_name
            FROM read_parquet(?)
            WHERE target IN (0, 1)
        ),
        base AS (
            SELECT DISTINCT mi_person_key FROM data
        ),
        drugs AS (
            SELECT mi_person_key, UPPER(TRIM(CAST(drug_name AS VARCHAR))) AS drug_norm
            FROM data
            WHERE drug_name IS NOT NULL AND TRIM(CAST(drug_name AS VARCHAR)) != ''
        ),
        total AS (
            SELECT mi_person_key, COUNT(DISTINCT drug_norm) AS pgx_num_drugs
            FROM drugs
            GROUP BY mi_person_key
        ),
        cpic AS (
            SELECT mi_person_key, COUNT(DISTINCT drug_norm) AS pgx_num_cpic_drugs
            FROM drugs
            WHERE drug_norm IN (SELECT unnest(?))
            GROUP BY mi_person_key
        )
        SELECT base.mi_person_key,
            COALESCE(total.pgx_num_drugs, 0)::INTEGER AS pgx_num_drugs,
            COALESCE(cpic.pgx_num_cpic_drugs, 0)::INTEGER AS pgx_num_cpic_drugs
        FROM base
        LEFT JOIN total ON base.mi_person_key = total.mi_person_key
        LEFT JOIN cpic ON base.mi_person_key = cpic.mi_person_key
        """,
        [str(model_data_path), cpic_list],
    ).df()
    con.close()
    
    if features_df.empty:
        logger.error("No target patients found in model_data")
        return pd.DataFrame()
    
    logger.info(f"Created PGx features for {len(features_df)} patients (DuckDB aggregation)")
    logger.info(f"  Total drugs: {features_df['pgx_num_drugs'].sum()}")
    logger.info(f"  CPIC drugs: {features_df['pgx_num_cpic_drugs'].sum()}")
    logger.info(f"  Patients with drugs: {(features_df['pgx_num_drugs'] > 0).sum()}")
    logger.info(f"  Patients with CPIC drugs: {(features_df['pgx_num_cpic_drugs'] > 0).sum()}")
    
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

