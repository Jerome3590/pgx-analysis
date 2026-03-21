#!/usr/bin/env python3
"""
Build a global APCD drug name -> CPIC drug name mapping from final feature importances.

This script:
1. Scans final (cohort) feature importance CSVs (cohort_feature_importance.csv) across cohorts/age bands;
   falls back to aggregated feature importance if no cohort FI found.
2. Extracts unique drug names (features with DRUG:, item_drug_*, etc. — APCD drug names from the pipeline).
3. Matches them to CPIC drug names using fuzzy matching (95%+ threshold).
4. Creates a global lookup table: 5_pgx_analysis/outputs/global/drug_cpic_mapping_global.csv
5. Idempotent: skips rebuild if output exists and is newer than all source FI files (use --force to rebuild).

Usage:
    python 5_pgx_analysis/build_global_drug_cpic_mapping.py [--cohort <cohort>] [--age-band <age_band>]
    python 5_pgx_analysis/build_global_drug_cpic_mapping.py --force   # rebuild even if output is current
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


def extract_drugs_from_aggregated_fi(fi_path: Path, cpic_drug_list: Optional[List[Dict]] = None) -> Set[str]:
    """
    Extract unique drug names from an aggregated feature importance CSV file.
    
    Parameters:
    -----------
    fi_path : Path
        Path to aggregated feature importance CSV
    cpic_drug_list : List[Dict], optional
        List of CPIC drug dictionaries to match against (for identifying drugs from item_ features)
        
    Returns:
    --------
    Set[str]
        Set of unique drug names (features starting with "DRUG:", "item_drug_", "drug_", or matching known drugs)
    """
    try:
        df = pd.read_csv(fi_path)
        if 'feature' not in df.columns:
            logger.warning(f"No 'feature' column in {fi_path}")
            return set()
        
        # Extract drug features (multiple patterns)
        feature_col = df["feature"].astype(str)
        
        drug_mask = (
            feature_col.str.startswith("DRUG:", na=False)
            | feature_col.str.startswith("item_drug_", na=False)
            | feature_col.str.startswith("drug_", na=False)
        )
        
        drug_features = feature_col[drug_mask].unique()
        
        # Remove prefixes to get drug names
        drug_names = {
            f.replace("DRUG:", "", 1)
             .replace("item_drug_", "", 1)
             .replace("drug_", "", 1)
             .strip()
            for f in drug_features
        }
        
        # Also check for drugs in item_ features (e.g., item_SUBOXONE, item_BUPRENORPHINE HCL/NALOXON)
        # These are features that start with "item_" and match known drug names
        if cpic_drug_list:
            cpic_drug_names = {drug_dict.get("name", "").upper() for drug_dict in cpic_drug_list if drug_dict.get("name")}
            
            item_features = feature_col[feature_col.str.startswith("item_", na=False)].unique()
            for item_feat in item_features:
                # Remove "item_" prefix
                item_name = item_feat.replace("item_", "", 1).strip()
                
                # Check if it matches a CPIC drug name (case-insensitive)
                item_upper = item_name.upper()
                if item_upper in cpic_drug_names:
                    drug_names.add(item_name)
                else:
                    # Also check for partial matches (e.g., "BUPRENORPHINE HCL/NALOXON" contains "buprenorphine")
                    for cpic_name in cpic_drug_names:
                        # Check if item_name contains the drug name or vice versa
                        if cpic_name in item_upper or item_upper in cpic_name:
                            # Prefer the CPIC name for consistency
                            drug_names.add(cpic_name.lower() if cpic_name.islower() else cpic_name)
                            break
        
        return drug_names
    except Exception as e:
        logger.warning(f"Error reading {fi_path}: {e}")
        return set()


def find_all_cohort_fi_files(cohort: Optional[str] = None, age_band: Optional[str] = None) -> List[Path]:
    """
    Find all final (cohort) feature importance CSV files (cohort_feature_importance.csv).
    Same sources as Cohort PGx and dashboard (Step 3b / 3a refined). APCD drug names come from these.
    """
    pattern = "*_cohort_feature_importance.csv"
    fi_files: List[Path] = []
    for base in [
        PROJECT_ROOT / "3a_feature_importance" / "outputs",
        PROJECT_ROOT / "3b_feature_importance_eda" / "outputs",
    ]:
        if not base.exists():
            continue
        for fi_file in base.rglob(pattern):
            if cohort and cohort not in str(fi_file):
                continue
            if age_band:
                ab = age_band.replace("-", "_")
                if ab not in fi_file.stem and ab not in str(fi_file):
                    continue
            fi_files.append(fi_file)
    from_s3 = PROJECT_ROOT / "3a_feature_importance" / "from_s3" / "by_cohort"
    if from_s3.exists():
        for fi_file in from_s3.rglob(pattern):
            if cohort and cohort not in str(fi_file):
                continue
            if age_band:
                if age_band.replace("-", "_") not in fi_file.stem:
                    continue
            fi_files.append(fi_file)
    return sorted(set(fi_files))


def find_all_aggregated_fi_files(cohort: Optional[str] = None, age_band: Optional[str] = None) -> List[Path]:
    """
    Find all aggregated feature importance CSV files (fallback when no cohort FI).
    """
    pattern = "*_aggregated_feature_importance.csv"
    fi_files: List[Path] = []
    for base in [
        PROJECT_ROOT / "3_feature_importance" / "outputs",
        PROJECT_ROOT / "3a_feature_importance" / "outputs",
    ]:
        if not base.exists():
            continue
        for fi_file in base.rglob(pattern):
            if cohort:
                # Check if file path contains the cohort name
                if cohort not in str(fi_file):
                    continue
            if age_band:
                age_band_fname = age_band.replace("-", "_")
                if age_band_fname not in fi_file.stem:
                    continue
            fi_files.append(fi_file)
    
    # Also check S3 download location (recursively)
    fi_from_s3_dir = PROJECT_ROOT / "3_feature_importance" / "from_s3" / "by_cohort"
    if fi_from_s3_dir.exists():
        # Search recursively for aggregated feature importance files
        pattern = "*_aggregated_feature_importance.csv"
        for fi_file in fi_from_s3_dir.rglob(pattern):
            if cohort:
                # Check if file path contains the cohort name
                if cohort not in str(fi_file):
                    continue
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
    Build global APCD drug -> CPIC drug mapping from final (cohort) or aggregated feature importance files.
    Prefers cohort_feature_importance.csv (final FI); falls back to aggregated if none found.
    """
    # Prefer final (cohort) feature importance — same source as Cohort PGx
    fi_files = find_all_cohort_fi_files(cohort=cohort, age_band=age_band)
    source_label = "cohort (final) feature importance"
    if not fi_files:
        logger.info("No cohort feature importance files found; using aggregated feature importance.")
        fi_files = find_all_aggregated_fi_files(cohort=cohort, age_band=age_band)
        source_label = "aggregated feature importance"
    if not fi_files:
        logger.warning("No feature importance files found")
        return pd.DataFrame(columns=['drug_name', 'cpic_drug_name', 'fuzzy_score', 'match_method', 'needs_review', 'google_search_url'])
    logger.info("Found %d %s files", len(fi_files), source_label)
    
    # Extract all unique drug names
    all_drugs: Set[str] = set()
    for fi_file in fi_files:
        drugs = extract_drugs_from_aggregated_fi(fi_file, cpic_drug_list=cpic_drug_list)
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
    
    if not mappings:
        logger.warning("No drug features were extracted from aggregated FI files.")
        return pd.DataFrame(
            columns=[
                "drug_name",
                "cpic_drug_name",
                "fuzzy_score",
                "match_method",
                "needs_review",
                "google_search_url",
            ]
        )
    
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
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild even if output exists and is newer than all source FI files (default: skip when up-to-date)",
    )
    args = parser.parse_args()

    global_out_dir = PROJECT_ROOT / "5_pgx_analysis" / "outputs" / "global"
    output_path = Path(args.output) if args.output else global_out_dir / "drug_cpic_mapping_global.csv"

    # Idempotent: skip only if output exists, is non-empty, and is newer than all source FI files
    if not args.force and output_path.exists():
        try:
            existing = pd.read_csv(output_path)
        except Exception as e:
            logger.info("Could not read existing mapping %s (%s); rebuilding.", output_path, e)
            existing = pd.DataFrame()
        if len(existing) == 0:
            logger.info("Existing global mapping is empty or unreadable; rebuilding.")
        else:
            fi_files = find_all_cohort_fi_files(cohort=args.cohort, age_band=args.age_band)
            if not fi_files:
                fi_files = find_all_aggregated_fi_files(cohort=args.cohort, age_band=args.age_band)
            if fi_files:
                out_mtime = output_path.stat().st_mtime
                if all(p.stat().st_mtime <= out_mtime for p in fi_files):
                    logger.info(
                        "Output %s is up-to-date (newer than all %d source FI files); skipping. Use --force to rebuild.",
                        output_path,
                        len(fi_files),
                    )
                    sys.exit(0)

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
    
    global_out_dir.mkdir(parents=True, exist_ok=True)
    validation_path = Path(args.validation_output) if args.validation_output else global_out_dir / "drug_cpic_mapping_validation.csv"
    
    # Save mapping table
    mapping_df.to_csv(output_path, index=False)
    logger.info(f"Saved global drug-to-CPIC mapping to {output_path}")

    # Upload to S3 so EC2/Lambda instances can download it without the local file
    try:
        import boto3
        from py_helpers.constants import S3_BUCKET
        s3_key = "gold/pgx_features/global/drug_cpic_mapping_global.csv"
        boto3.client("s3").upload_file(str(output_path), S3_BUCKET, s3_key)
        logger.info("Uploaded global drug mapping to s3://%s/%s", S3_BUCKET, s3_key)
    except Exception as e:
        logger.warning("Could not upload global drug mapping to S3 (local file still saved): %s", e)
    
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

