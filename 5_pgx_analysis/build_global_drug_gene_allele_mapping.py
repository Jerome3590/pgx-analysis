#!/usr/bin/env python3
"""
Build a complete global drug-to-gene-to-allele-frequency mapping table.

This script:
1. Loads the global drug-to-CPIC mapping table
2. Maps each CPIC drug to its associated genes using CPIC data
3. Adds population-specific allele frequencies for each gene
4. Creates a complete lookup table: drug_name -> cpic_drug_name -> gene -> allele_frequencies

Output:
- 5_pgx_analysis/outputs/global/drug_gene_allele_mapping_global.csv
- Also uploads to S3: s3://pgxdatalake/gold/pgx_features/global/drug_gene_allele_mapping_global.csv

Usage:
    python 5_pgx_analysis/build_global_drug_gene_allele_mapping.py [--force]
"""

import sys
import pandas as pd
from pathlib import Path
import logging
import time
from typing import Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "5_pgx_analysis") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "5_pgx_analysis"))

# Import mapping functions
from map_drugs_to_genes import (
    load_global_drug_mapping,
    load_cpic_drug_list_from_file,
    map_drugs_to_genes,
)
from add_allele_frequencies import add_allele_frequencies

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def build_global_drug_gene_allele_mapping(
    force: bool = False,
    rate_limit_delay: float = 0.5,
) -> pd.DataFrame:
    """
    Build complete global drug-to-gene-to-allele-frequency mapping table.
    
    Parameters:
    -----------
    force : bool
        Force regeneration even if output exists
    rate_limit_delay : float
        Delay between API calls (seconds)
        
    Returns:
    --------
    pd.DataFrame
        Complete mapping table with drug_name, cpic_drug_name, gene, and allele frequencies
    """
    global_out_dir = PROJECT_ROOT / "5_pgx_analysis" / "outputs" / "global"
    global_out_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = global_out_dir / "drug_gene_allele_mapping_global.csv"
    
    # Check if output exists
    if output_path.exists() and not force:
        logger.info(f"Global drug-gene-allele mapping already exists at {output_path}")
        logger.info("Loading existing file...")
        return pd.read_csv(output_path)
    
    # Step 1: Load global drug-to-CPIC mapping
    logger.info("Step 1: Loading global drug-to-CPIC mapping...")
    drug_mapping = load_global_drug_mapping()
    
    if drug_mapping is None or drug_mapping.empty:
        logger.error("No global drug-to-CPIC mapping found. Please run build_global_drug_cpic_mapping.py first.")
        sys.exit(1)
    
    logger.info(f"Loaded {len(drug_mapping)} drug mappings")
    
    # Step 2: Load CPIC drug list from local file (contains drug-gene mappings)
    logger.info("Step 2: Loading CPIC drug list from local file...")
    cpic_drug_list = load_cpic_drug_list_from_file()
    
    if not cpic_drug_list:
        logger.error("Could not load CPIC drug list. Please ensure cpic_drug_list.json exists.")
        sys.exit(1)
    
    logger.info(f"Loaded {len(cpic_drug_list)} CPIC drugs from local file")
    
    # Step 3: Build drug-gene mappings from local CPIC data
    logger.info("Step 3: Building drug-gene mappings from local CPIC data...")
    
    # Create a lookup dictionary for CPIC drugs
    cpic_drug_dict = {drug_dict.get("name", "").upper(): drug_dict for drug_dict in cpic_drug_list if drug_dict.get("name")}
    
    # Build drug-gene mappings
    drug_gene_mappings_list = []
    cpic_drug_names = drug_mapping['cpic_drug_name'].dropna().unique()
    
    for cpic_drug_name in cpic_drug_names:
        cpic_drug_upper = cpic_drug_name.upper()
        
        # Find matching drug in CPIC list
        drug_info = cpic_drug_dict.get(cpic_drug_upper)
        
        if drug_info:
            # Get genes associated with this drug
            genes = drug_info.get("genes", [])
            if isinstance(genes, str):
                genes = [genes]
            elif not isinstance(genes, list):
                genes = []
            
            # Get CPIC levels
            cpic_levels = drug_info.get("cpic_levels", [])
            cpic_level = cpic_levels[0] if cpic_levels else ""
            
            # Create mapping for each gene
            for gene in genes:
                if gene:  # Only add non-empty genes
                    drug_gene_mappings_list.append({
                        "drug_name": cpic_drug_name,  # Will merge with original drug_name later
                        "cpic_drug_name": cpic_drug_name,
                        "gene": gene,
                        "gene_name": "",
                        "relationship_type": "metabolism",
                        "evidence_level": "CPIC",
                        "clinical_significance": "",
                        "cpic_level": cpic_level,
                        "guideline_id": "",
                        "guideline_url": "",
                        "source": "CPIC_PAIRS_LOCAL"
                    })
        else:
            logger.debug(f"No gene mapping found for CPIC drug: {cpic_drug_name}")
    
    drug_gene_mappings = pd.DataFrame(drug_gene_mappings_list)
    
    if drug_gene_mappings.empty:
        logger.warning("No drug-gene mappings found from local CPIC data.")
        return pd.DataFrame()
    
    logger.info(f"Found {len(drug_gene_mappings)} drug-gene mappings from local CPIC data")
    logger.info(f"Unique genes: {drug_gene_mappings['gene'].nunique()}")
    
    # Step 4: Add allele frequencies
    logger.info("Step 3: Adding population-specific allele frequencies...")
    enriched_mappings = add_allele_frequencies(
        drug_gene_mappings=drug_gene_mappings,
        rate_limit_delay=rate_limit_delay,
        use_patient_demographics=False,  # Use population-weighted average
    )
    
    logger.info(f"Step 4: Enriched {len(enriched_mappings)} mappings with allele frequencies")
    
    # Step 5: Merge back with original drug names to create complete mapping
    logger.info("Step 5: Creating complete drug-to-gene-to-allele mapping...")
    
    # Merge drug_gene_allele mappings with original drug names
    # Note: enriched_mappings already has 'cpic_drug_name' from the drug-gene mapping step
    # We need to add the original 'drug_name' from the global drug mapping
    complete_mapping = enriched_mappings.merge(
        drug_mapping[['drug_name', 'cpic_drug_name']].drop_duplicates(),
        on='cpic_drug_name',
        how='left'
    )
    
    # If drug_name wasn't added by merge (shouldn't happen, but handle gracefully)
    if 'drug_name' not in complete_mapping.columns:
        # Use cpic_drug_name as fallback
        complete_mapping['drug_name'] = complete_mapping['cpic_drug_name']
    
    # Reorder columns for better readability
    column_order = [
        'drug_name',  # Original drug name from feature importance
        'cpic_drug_name',  # CPIC standard name
        'gene',  # Pharmacogene
        'gene_name',
        'relationship_type',
        'evidence_level',
        'clinical_significance',
        'cpic_level',
        'variant_id',
        'allele_name',
        'allele_function',
        'allele_frequency_global',
        'allele_frequency_afr',
        'allele_frequency_amr',
        'allele_frequency_eas',
        'allele_frequency_eur',
        'allele_frequency_sas',
        'allele_frequency_assigned',
        'frequency_source',
        'frequency_assignment_method',
        'guideline_id',
        'guideline_url',
        'source',
    ]
    
    # Only include columns that exist
    available_columns = [col for col in column_order if col in complete_mapping.columns]
    complete_mapping = complete_mapping[available_columns + [col for col in complete_mapping.columns if col not in available_columns]]
    
    # Save to local file
    complete_mapping.to_csv(output_path, index=False)
    logger.info(f"Saved complete mapping to {output_path}")
    
    # Upload to S3
    try:
        import boto3
        from py_helpers.constants import S3_BUCKET
        
        s3_client = boto3.client('s3')
        s3_key = "gold/pgx_features/global/drug_gene_allele_mapping_global.csv"
        
        s3_client.upload_file(str(output_path), S3_BUCKET, s3_key)
        logger.info(f"Uploaded to s3://{S3_BUCKET}/{s3_key}")
    except Exception as e:
        logger.warning(f"Could not upload to S3: {e}")
    
    # Log summary
    logger.info("\n" + "="*60)
    logger.info("Mapping Summary:")
    logger.info(f"  Total drug-gene-allele mappings: {len(complete_mapping)}")
    if 'drug_name' in complete_mapping.columns:
        logger.info(f"  Unique original drugs: {complete_mapping['drug_name'].nunique()}")
    logger.info(f"  Unique CPIC drugs: {complete_mapping['cpic_drug_name'].nunique()}")
    logger.info(f"  Unique genes: {complete_mapping['gene'].nunique()}")
    if 'allele_frequency_global' in complete_mapping.columns:
        logger.info(f"  Mappings with allele frequencies: {complete_mapping['allele_frequency_global'].notna().sum()}")
    logger.info("="*60)
    
    return complete_mapping


def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Build complete global drug-to-gene-to-allele-frequency mapping table"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force regeneration even if output exists"
    )
    parser.add_argument(
        "--rate-limit-delay",
        type=float,
        default=0.5,
        help="Delay between API calls in seconds (default: 0.5)"
    )
    
    args = parser.parse_args()
    
    logger.info("Building global drug-to-gene-to-allele-frequency mapping table...")
    logger.info("This may take several minutes due to API rate limiting...")
    
    mapping_df = build_global_drug_gene_allele_mapping(
        force=args.force,
        rate_limit_delay=args.rate_limit_delay,
    )
    
    if not mapping_df.empty:
        logger.info("\n✅ Global drug-to-gene-to-allele mapping complete!")
        logger.info(f"  Output file: 5_pgx_analysis/outputs/global/drug_gene_allele_mapping_global.csv")
        logger.info(f"  S3 location: s3://pgxdatalake/gold/pgx_features/global/drug_gene_allele_mapping_global.csv")
    else:
        logger.error("\n❌ Failed to build mapping table")
        sys.exit(1)


if __name__ == "__main__":
    main()

