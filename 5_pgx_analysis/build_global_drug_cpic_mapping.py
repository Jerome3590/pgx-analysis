#!/usr/bin/env python3
"""
Build a global drug-to-CPIC mapping table from all aggregated feature importance files.

This script:
1. Scans all aggregated feature importance CSV files (across all cohorts/age bands)
2. Extracts unique drug names (features starting with "DRUG:")
3. Matches them to CPIC drug names using fuzzy matching (95%+ threshold)
4. For matches below 95%, searches cpic_drug_list.json for better matches
5. Creates a global lookup table saved to: 5_pgx_analysis/outputs/global/drug_cpic_mapping_global.csv
6. Generates a validation file for manual review of low-score matches

Usage:
    python 5_pgx_analysis/build_global_drug_cpic_mapping.py [--cohort <cohort>] [--age-band <age_band>]
"""

import sys
import pandas as pd
from pathlib import Path
import json
import logging
from typing import Set, Dict, List, Optional, Tuple
from collections import defaultdict

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "5_pgx_analysis") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "5_pgx_analysis"))

# Import fuzzy matching functions
from map_drugs_to_genes import (
    load_cpic_drug_list_from_file,
    fuzzy_match_drug,
    search_cpic_drug_list_json,
    suggest_google_search,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def extract_drugs_from_aggregated_fi(fi_path: Path) -> Set[str]:
    """
    Extract unique drug names from an aggregated feature importance CSV file.
    
    Parameters:
    -----------
    fi_path : Path
        Path to aggregated feature importance CSV
        
    Returns:
    --------
    Set[str]
        Set of unique drug names (features starting with "DRUG:")
    """
    try:
        df = pd.read_csv(fi_path)
        if 'feature' not in df.columns:
            logger.warning(f"No 'feature' column in {fi_path}")
            return set()
        
        # Extract drug features (starting with "DRUG:")
        drug_features = df[df['feature'].str.startswith('DRUG:', na=False)]['feature'].unique()
        # Remove "DRUG:" prefix
        drug_names = {feat.replace('DRUG:', '', 1).strip() for feat in drug_features}
        return drug_names
    except Exception as e:
        logger.warning(f"Error reading {fi_path}: {e}")
        return set()


def find_all_aggregated_fi_files(cohort: Optional[str] = None, age_band: Optional[str] = None) -> List[Path]:
    """
    Find all aggregated feature importance CSV files.
    
    Parameters:
    -----------
    cohort : str, optional
        Filter by specific cohort
    age_band : str, optional
        Filter by specific age band
        
    Returns:
    --------
    List[Path]
        List of paths to aggregated feature importance CSV files
    """
    fi_outputs_dir = PROJECT_ROOT / "3_feature_importance" / "outputs"
    fi_files = []
    
    # Search local outputs directory
    if fi_outputs_dir.exists():
        for cohort_dir in fi_outputs_dir.iterdir():
            if not cohort_dir.is_dir():
                continue
            if cohort and cohort_dir.name != cohort:
                continue
            
            # Look for aggregated feature importance files
            for fi_file in cohort_dir.glob("*_aggregated_feature_importance.csv"):
                if age_band:
                    age_band_fname = age_band.replace("-", "_")
                    if age_band_fname not in fi_file.stem:
                        continue
                fi_files.append(fi_file)
    
    # Also check S3 download location
    fi_from_s3_dir = PROJECT_ROOT / "3_feature_importance" / "from_s3" / "by_cohort"
    if fi_from_s3_dir.exists():
        for cohort_dir in fi_from_s3_dir.iterdir():
            if not cohort_dir.is_dir():
                continue
            if cohort and cohort_dir.name != cohort:
                continue
            
            for fi_file in cohort_dir.glob("*_aggregated_feature_importance.csv"):
                if age_band:
                    age_band_fname = age_band.replace("-", "_")
                    if age_band_fname not in fi_file.stem:
                        continue
                fi_files.append(fi_file)
    
    return sorted(set(fi_files))  # Remove duplicates


def build_global_drug_mapping(
    cpic_drug_list: List[Dict],
    fuzzy_threshold: int = 95,
    cohort: Optional[str] = None,
    age_band: Optional[str] = None,
) -> pd.DataFrame:
    """
    Build global drug-to-CPIC mapping table from all aggregated feature importance files.
    
    Parameters:
    -----------
    cpic_drug_list : List[Dict]
        List of CPIC drug dictionaries from JSON file
    fuzzy_threshold : int
        Minimum fuzzy match score threshold (default: 95)
    cohort : str, optional
        Filter by specific cohort
    age_band : str, optional
        Filter by specific age band
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with columns: drug_name, cpic_drug_name, fuzzy_score, match_method, needs_review, google_search_url
    """
    logger.info("Finding all aggregated feature importance files...")
    fi_files = find_all_aggregated_fi_files(cohort=cohort, age_band=age_band)
    
    if not fi_files:
        logger.warning("No aggregated feature importance files found")
        return pd.DataFrame(columns=['drug_name', 'cpic_drug_name', 'fuzzy_score', 'match_method', 'needs_review', 'google_search_url'])
    
    logger.info(f"Found {len(fi_files)} aggregated feature importance files")
    
    # Extract all unique drug names
    all_drugs: Set[str] = set()
    for fi_file in fi_files:
        drugs = extract_drugs_from_aggregated_fi(fi_file)
        all_drugs.update(drugs)
        logger.debug(f"Extracted {len(drugs)} drugs from {fi_file.name}")
    
    logger.info(f"Found {len(all_drugs)} unique drug names across all feature importance files")
    
    # Build mapping table
    mappings = []
    needs_review = []
    
    for drug_name in sorted(all_drugs):
        logger.debug(f"Processing drug: {drug_name}")
        
        matched_cpic_name = drug_name
        fuzzy_score = 100.0
        match_method = "exact"
        google_url = ""
        
        # Try fuzzy matching
        fuzzy_match = fuzzy_match_drug(drug_name, cpic_drug_list, threshold=fuzzy_threshold)
        if fuzzy_match:
            matched_cpic_name, matched_drug_info, fuzzy_score = fuzzy_match
            match_method = "fuzzy"
            
            # If score is below 95%, try searching CPIC drug list JSON
            if fuzzy_score < 95.0:
                logger.warning(
                    f"Low fuzzy match score ({fuzzy_score:.1f}) for '{drug_name}' -> '{matched_cpic_name}'. "
                    f"Searching CPIC drug list for better match..."
                )
                better_match = search_cpic_drug_list_json(drug_name, cpic_drug_list)
                if better_match:
                    matched_cpic_name, matched_drug_info, better_score = better_match
                    logger.info(
                        f"Found better match in CPIC list: '{drug_name}' -> '{matched_cpic_name}' "
                        f"(score: {better_score:.1f})"
                    )
                    fuzzy_score = better_score
                    match_method = "cpic_list_search"
                else:
                    # Suggest Google search for manual review
                    google_url = suggest_google_search(drug_name, matched_cpic_name, fuzzy_score)
                    needs_review.append({
                        'drug_name': drug_name,
                        'cpic_drug_name': matched_cpic_name,
                        'fuzzy_score': fuzzy_score,
                        'google_search_url': google_url
                    })
        
        mappings.append({
            'drug_name': drug_name,
            'cpic_drug_name': matched_cpic_name,
            'fuzzy_score': fuzzy_score,
            'match_method': match_method,
            'needs_review': fuzzy_score < 95.0,
            'google_search_url': google_url if fuzzy_score < 95.0 else ""
        })
    
    mapping_df = pd.DataFrame(mappings)
    
    # Log summary
    exact_matches = len(mapping_df[mapping_df['match_method'] == 'exact'])
    fuzzy_matches = len(mapping_df[mapping_df['match_method'] == 'fuzzy'])
    cpic_list_matches = len(mapping_df[mapping_df['match_method'] == 'cpic_list_search'])
    low_score_count = len(mapping_df[mapping_df['needs_review'] == True])
    
    logger.info(f"Mapping summary:")
    logger.info(f"  Total drugs: {len(mapping_df)}")
    logger.info(f"  Exact matches: {exact_matches}")
    logger.info(f"  Fuzzy matches (≥95%): {fuzzy_matches}")
    logger.info(f"  CPIC list matches: {cpic_list_matches}")
    logger.info(f"  Needs review (<95%): {low_score_count}")
    
    if needs_review:
        logger.warning(f"\n⚠️  {len(needs_review)} drugs need manual review:")
        for item in needs_review[:10]:  # Show first 10
            logger.warning(
                f"  '{item['drug_name']}' -> '{item['cpic_drug_name']}' "
                f"(score: {item['fuzzy_score']:.1f})"
            )
        if len(needs_review) > 10:
            logger.warning(f"  ... and {len(needs_review) - 10} more")
    
    return mapping_df


def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Build global drug-to-CPIC mapping table from aggregated feature importance files"
    )
    parser.add_argument(
        "--cohort",
        help="Filter by specific cohort (optional)"
    )
    parser.add_argument(
        "--age-band",
        help="Filter by specific age band (optional)"
    )
    parser.add_argument(
        "--fuzzy-threshold",
        type=int,
        default=95,
        help="Minimum fuzzy match score threshold (default: 95)"
    )
    parser.add_argument(
        "--output",
        help="Output path for mapping CSV (default: 5_pgx_analysis/outputs/global/drug_cpic_mapping_global.csv)"
    )
    parser.add_argument(
        "--validation-output",
        help="Output path for validation CSV (default: 5_pgx_analysis/outputs/global/drug_cpic_mapping_validation.csv)"
    )
    
    args = parser.parse_args()
    
    # Load CPIC drug list
    logger.info("Loading CPIC drug list...")
    cpic_drug_list = load_cpic_drug_list_from_file()
    if not cpic_drug_list:
        logger.error("Failed to load CPIC drug list. Please ensure cpic_drug_list.json exists.")
        sys.exit(1)
    logger.info(f"Loaded {len(cpic_drug_list)} CPIC drugs")
    
    # Build global mapping
    mapping_df = build_global_drug_mapping(
        cpic_drug_list=cpic_drug_list,
        fuzzy_threshold=args.fuzzy_threshold,
        cohort=args.cohort,
        age_band=args.age_band,
    )
    
    # Set output paths
    global_out_dir = PROJECT_ROOT / "5_pgx_analysis" / "outputs" / "global"
    global_out_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = Path(args.output) if args.output else global_out_dir / "drug_cpic_mapping_global.csv"
    validation_path = Path(args.validation_output) if args.validation_output else global_out_dir / "drug_cpic_mapping_validation.csv"
    
    # Save mapping table
    mapping_df.to_csv(output_path, index=False)
    logger.info(f"Saved global drug-to-CPIC mapping to {output_path}")
    
    # Save validation file (only drugs needing review)
    validation_df = mapping_df[mapping_df['needs_review'] == True].copy()
    if not validation_df.empty:
        validation_df.to_csv(validation_path, index=False)
        logger.info(f"Saved validation file (drugs needing review) to {validation_path}")
        logger.warning(f"\n⚠️  {len(validation_df)} drugs need manual review. Please check: {validation_path}")
        
        # Validate that no matches are below threshold
        low_scores = validation_df[validation_df['fuzzy_score'] < args.fuzzy_threshold]
        if not low_scores.empty:
            logger.error(f"\n❌ ERROR: {len(low_scores)} drugs have scores below {args.fuzzy_threshold}% threshold:")
            for _, row in low_scores.iterrows():
                logger.error(
                    f"  '{row['drug_name']}' -> '{row['cpic_drug_name']}' "
                    f"(score: {row['fuzzy_score']:.1f})"
                )
            logger.error(f"\nPlease review and fix matches in: {validation_path}")
            sys.exit(1)
    else:
        logger.info("✓ All drug matches meet threshold requirements")
    
    logger.info("\n✅ Global drug-to-CPIC mapping complete!")
    logger.info(f"  Mapping file: {output_path}")
    logger.info(f"  Validation file: {validation_path}")
    logger.info(f"  Total drugs mapped: {len(mapping_df)}")
    logger.info(f"  Drugs needing review: {len(validation_df)}")


if __name__ == "__main__":
    main()

