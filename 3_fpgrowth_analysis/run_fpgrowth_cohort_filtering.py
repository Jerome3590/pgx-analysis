"""
FPGrowth-based Initial Filtering for Cohort Analysis

This module provides FPGrowth pattern mining for initial filtering of:
- Drugs (ED_NON_OPIOID cohort - 30-day window)
- ICD Codes (OPIOID_ED cohort)
- CPT Codes (OPIOID_ED cohort)

Results can be used for:
1. Initial feature filtering before CatBoost analysis
2. Pattern identification for BupaR process mining
3. Association rule discovery for predictive modeling

Usage:
    python run_fpgrowth_cohort_filtering.py --cohort ED_NON_OPIOID --item-type drug
    python run_fpgrowth_cohort_filtering.py --cohort OPIOID_ED --item-type icd
    python run_fpgrowth_cohort_filtering.py --cohort OPIOID_ED --item-type cpt
"""

import os
import sys
import logging
import argparse
import pandas as pd
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed

# Project path config
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from helpers_1997_13.constants import (
    AGE_BANDS, EVENT_YEARS, TOP_K, MIN_SUPPORT_THRESHOLD, 
    MIN_CONFIDENCE_MEDIUM, TIMEOUT_SECONDS
)
from helpers_1997_13.duckdb_utils import get_duckdb_connection
from helpers_1997_13.fpgrowth_utils import (
    run_fpgrowth_drug_token_with_fallback,
    convert_frozensets,
)
from helpers_1997_13.s3_utils import (
    get_output_paths,
    save_to_s3_json,
    save_to_s3_parquet,
    s3_exists,
    get_cohort_parquet_path,
    load_from_s3_json,
    load_from_s3_parquet,
)
from helpers_1997_13.visualization_utils import create_network_visualization


def create_filtering_logger(name: str) -> logging.Logger:
    """Create logger for filtering pipeline."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def load_cohort_data(
    cohort_name: str, 
    age_band: str, 
    event_year: str, 
    item_type: str,
    logger: logging.Logger
) -> pd.DataFrame:
    """
    Load cohort data and extract items based on type.
    
    Args:
        cohort_name: Cohort name (opioid_ed, ed_non_opioid)
        age_band: Age band (e.g., '65-74')
        event_year: Event year (e.g., '2020')
        item_type: Type of items to extract ('drug', 'icd', 'cpt')
        logger: Logger instance
        
    Returns:
        DataFrame with mi_person_key and item columns
    """
    con = get_duckdb_connection(logger=logger)
    cohort_path = get_cohort_parquet_path(cohort_name, age_band, event_year)
    
    # Build query based on item type
    if item_type == 'drug':
        # For ED_NON_OPIOID: only include drugs in 30-day window
        query = f"""
        SELECT DISTINCT 
            mi_person_key,
            drug_name as item_name,
            event_date,
            days_to_target_event
        FROM read_parquet('{cohort_path}')
        WHERE drug_name IS NOT NULL
          AND event_type = 'pharmacy'
          AND (
              -- Include all drug events for targets
              (is_target_case = 1)
              OR
              -- Include drug events within 30-day window for controls
              (is_target_case = 0 AND days_to_target_event IS NOT NULL 
               AND days_to_target_event >= 0 AND days_to_target_event <= 30)
          )
        """
    elif item_type == 'icd':
        # For OPIOID_ED: include all ICD codes
        query = f"""
        SELECT DISTINCT 
            mi_person_key,
            primary_icd_diagnosis_code as item_name,
            event_date
        FROM read_parquet('{cohort_path}')
        WHERE primary_icd_diagnosis_code IS NOT NULL
          AND event_type = 'medical'
        """
    elif item_type == 'cpt':
        # For OPIOID_ED: include all CPT codes
        query = f"""
        SELECT DISTINCT 
            mi_person_key,
            procedure_code as item_name,
            event_date
        FROM read_parquet('{cohort_path}')
        WHERE procedure_code IS NOT NULL
          AND event_type = 'medical'
        """
    else:
        raise ValueError(f"Unsupported item_type: {item_type}. Must be 'drug', 'icd', or 'cpt'")
    
    df = con.execute(query).df()
    con.close()
    
    if df.empty:
        logger.warning(f"No {item_type} data found for {cohort_name}/{age_band}/{event_year}")
        return df
    
    logger.info(f"Loaded {len(df):,} {item_type} records for {cohort_name}/{age_band}/{event_year}")
    logger.info(f"  Distinct patients: {df['mi_person_key'].nunique():,}")
    logger.info(f"  Distinct {item_type}s: {df['item_name'].nunique():,}")
    
    return df


def build_transactions(
    df: pd.DataFrame, 
    item_type: str,
    logger: logging.Logger
) -> pd.DataFrame:
    """
    Build patient-level transactions from item data.
    
    Args:
        df: DataFrame with mi_person_key and item_name
        item_type: Type of items ('drug', 'icd', 'cpt')
        logger: Logger instance
        
    Returns:
        DataFrame with mi_person_key and item_tokens columns
    """
    # Create token prefix based on item type
    token_prefix = {
        'drug': 'drug_',
        'icd': 'icd_',
        'cpt': 'cpt_'
    }[item_type]
    
    # Group by patient and create sorted token sets
    grouped = (
        df.groupby("mi_person_key")["item_name"]
          .agg(lambda rows: sorted({
              f"{token_prefix}{str(d).strip().lower()}"
              for d in rows if pd.notnull(d) and str(d).strip()
          }))
          .reset_index()
          .rename(columns={"item_name": "item_tokens"})
    )
    
    logger.info(f"Built {len(grouped):,} patient transactions")
    logger.info(f"  Average items per patient: {grouped['item_tokens'].apply(len).mean():.2f}")
    
    return grouped


def run_single_cohort_filtering(job: dict) -> tuple:
    """
    Process a single cohort-age-year-item_type job end-to-end.
    
    Args:
        job: Dictionary with cohort, age_band, event_year, item_type, etc.
        
    Returns:
        Tuple of (cohort, age_band, event_year, item_type, success, message)
    """
    cohort = job['cohort']
    age_band = job['age_band']
    event_year = str(job['event_year'])
    item_type = job['item_type']
    
    logger = create_filtering_logger(
        f"fpgrowth_filtering_{cohort}_{age_band}_{event_year}_{item_type}"
    )
    
    try:
        import time
        t0 = time.time()
        
        logger.info(f"Starting FPGrowth filtering for {cohort}/{age_band}/{event_year}/{item_type}")
        
        # Load cohort data
        df = load_cohort_data(cohort, age_band, event_year, item_type, logger)
        
        if df.empty:
            logger.warning(f"No data for {cohort}/{age_band}/{event_year}/{item_type}")
            return (cohort, age_band, event_year, item_type, False, "No data")
        
        # Build transactions
        grouped = build_transactions(df, item_type, logger)
        
        if grouped.empty or grouped['item_tokens'].apply(len).sum() == 0:
            logger.warning(f"No transactions for {cohort}/{age_band}/{event_year}/{item_type}")
            return (cohort, age_band, event_year, item_type, False, "No transactions")
        
        # Get output paths
        paths = get_output_paths(cohort.lower(), age_band, event_year)
        
        # Adjust paths for item type
        base_path = paths['itemsets_json'].replace('/itemsets.json', '')
        itemsets_path = f"{base_path}/itemsets_{item_type}.json"
        rules_path = f"{base_path}/rules_{item_type}.json"
        encoding_path = f"{base_path}/encoding_{item_type}.parquet"
        
        # Run FP-Growth
        min_support = job.get('min_support_threshold', MIN_SUPPORT_THRESHOLD)
        timeout_seconds = job.get('timeout_seconds', TIMEOUT_SECONDS)
        
        logger.info(f"Running FP-Growth for {item_type} items...")
        _, itemsets, rules = run_fpgrowth_drug_token_with_fallback(
            grouped, 
            cohort.lower(), 
            age_band, 
            event_year, 
            paths, 
            logger,
            support_start=min_support, 
            TOP_K=TOP_K, 
            min_confidence=MIN_CONFIDENCE_MEDIUM, 
            timeout=timeout_seconds
        )
        
        if itemsets.empty:
            logger.warning(f"No itemsets found for {cohort}/{age_band}/{event_year}/{item_type}")
            return (cohort, age_band, event_year, item_type, False, "No itemsets")
        
        logger.info(f"FP-Growth: {len(itemsets)} itemsets, {len(rules)} rules")
        
        # Save results
        itemsets_json = convert_frozensets(itemsets.to_dict(orient="records"), logger)
        rules_json = convert_frozensets(rules.to_dict(orient="records"), logger)
        
        save_to_s3_json(itemsets_json, itemsets_path, logger)
        save_to_s3_json(rules_json, rules_path, logger)
        
        # Build and save encoding map
        try:
            unique_items = sorted(set(
                d for d in df['item_name'].dropna().astype(str)
            ))
            
            # Extract support and confidence metrics
            support_map = {item: 0.0 for item in unique_items}
            confidence_map = {item: [] for item in unique_items}
            
            for _, row in itemsets.iterrows():
                items = row.get('itemsets', [])
                sup = float(row.get('support', 0.0)) if row.get('support', None) is not None else 0.0
                for tok in items if isinstance(items, (list, set, tuple)) else []:
                    if isinstance(tok, str):
                        # Remove prefix to get original item name
                        item_name = tok.replace('drug_', '').replace('icd_', '').replace('cpt_', '')
                        if item_name in support_map:
                            support_map[item_name] = max(support_map[item_name], sup)
            
            for _, row in rules.iterrows():
                conf = float(row.get('confidence', 0.0)) if row.get('confidence', None) is not None else 0.0
                for col in ('antecedents', 'consequents'):
                    vals = row.get(col, [])
                    if isinstance(vals, (list, set, tuple)):
                        for tok in vals:
                            if isinstance(tok, str):
                                item_name = tok.replace('drug_', '').replace('icd_', '').replace('cpt_', '')
                                if item_name in confidence_map:
                                    confidence_map[item_name].append(conf)
            
            # Build encoding DataFrame
            records = []
            for i, item in enumerate(unique_items):
                rec = {
                    'item_name': item,
                    'item_type': item_type,
                    'encoded_id': i,
                    'support': support_map.get(item, 0.0),
                    'avg_confidence': (
                        sum(confidence_map.get(item, [])) / len(confidence_map.get(item, []))
                        if confidence_map.get(item, []) else 0.0
                    ),
                    'num_rules': len(confidence_map.get(item, []))
                }
                records.append(rec)
            
            encoding_df = pd.DataFrame(records)
            save_to_s3_parquet(encoding_df, encoding_path, logger)
            
        except Exception as enc_e:
            logger.warning(f"Failed to build encoding map: {enc_e}")
        
        # Create network visualization
        try:
            if not rules.empty:
                if "certainty" not in rules.columns:
                    rules["certainty"] = rules["confidence"] if "confidence" in rules.columns else 0.0
                create_network_visualization(
                    rules_df=rules,
                    title=f"{cohort} {age_band} {event_year} {item_type.upper()} Network",
                    cohort_name=cohort.lower(),
                    age_band=age_band,
                    event_year=event_year,
                    itemsets_counts=None,
                    logger=logger,
                )
                logger.info(f"Network visualization created for {item_type}")
        except Exception as viz_e:
            logger.warning(f"Network visualization failed: {viz_e}")
        
        elapsed = time.time() - t0
        logger.info(f"✓ Completed {cohort}/{age_band}/{event_year}/{item_type} in {elapsed:.2f}s")
        return (cohort, age_band, event_year, item_type, True, "Success")
        
    except Exception as e:
        logger.error(f"✗ Failed {cohort}/{age_band}/{event_year}/{item_type}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return (cohort, age_band, event_year, item_type, False, str(e))


def execute_filtering_pipeline(
    cohort: Optional[str] = None,
    item_type: Optional[str] = None,
    age_band: Optional[str] = None,
    event_year: Optional[int] = None
) -> None:
    """
    Execute FPGrowth filtering pipeline for specified cohorts and item types.
    
    Args:
        cohort: Cohort name (opioid_ed, ed_non_opioid) or None for all
        item_type: Item type (drug, icd, cpt) or None for all relevant types
        age_band: Specific age band or None for all
        event_year: Specific event year or None for all
    """
    logger = create_filtering_logger("fpgrowth_filtering_pipeline")
    
    # Determine target cohorts
    if cohort is None:
        target_cohorts = ["opioid_ed", "ed_non_opioid"]
    elif isinstance(cohort, str):
        target_cohorts = [cohort.lower()]
    else:
        target_cohorts = [c.lower() for c in cohort]
    
    # Determine item types based on cohort
    cohort_item_map = {
        'ed_non_opioid': ['drug'],  # Only drugs for ED_NON_OPIOID (30-day window)
        'opioid_ed': ['icd', 'cpt']  # ICD and CPT for OPIOID_ED
    }
    
    # Build jobs
    jobs = []
    for c in target_cohorts:
        if c not in cohort_item_map:
            logger.warning(f"Unknown cohort: {c}, skipping")
            continue
        
        item_types = cohort_item_map[c]
        if item_type:
            # Filter to requested item type if specified
            if item_type.lower() not in item_types:
                logger.warning(
                    f"Item type {item_type} not applicable for cohort {c}. "
                    f"Available: {item_types}"
                )
                continue
            item_types = [item_type.lower()]
        
        age_bands = [age_band] if age_band else AGE_BANDS
        event_years = [event_year] if event_year else EVENT_YEARS
        
        for it in item_types:
            for ab in age_bands:
                for ey in event_years:
                    # Check if already processed
                    paths = get_output_paths(c, ab, ey)
                    base_path = paths['itemsets_json'].replace('/itemsets.json', '')
                    itemsets_path = f"{base_path}/itemsets_{it}.json"
                    
                    if not s3_exists(itemsets_path):
                        job = {
                            'cohort': c,
                            'age_band': ab,
                            'event_year': ey,
                            'item_type': it
                        }
                        jobs.append(job)
    
    if not jobs:
        logger.info("All jobs already completed or no jobs to process")
        return
    
    logger.info(f"🚀 Processing {len(jobs)} FPGrowth filtering jobs")
    
    # Process jobs in parallel
    max_retries = 2
    attempt = 0
    failed = jobs
    
    while failed and attempt <= max_retries:
        logger.info(f"[RETRY] Attempt {attempt+1} for {len(failed)} jobs")
        results = []
        
        with ProcessPoolExecutor(max_workers=32) as executor:
            future_to_job = {
                executor.submit(run_single_cohort_filtering, job): job 
                for job in failed
            }
            
            for future in as_completed(future_to_job):
                job = future_to_job[future]
                logger.info(
                    f"[POOL] Processing: {job['cohort']}/{job['age_band']}/"
                    f"{job['event_year']}/{job['item_type']}"
                )
                try:
                    results.append(future.result(timeout=1800))
                except Exception as err:
                    logger.error(f"Job failed {job}: {err}")
                    results.append((
                        job["cohort"], 
                        job["age_band"], 
                        job["event_year"],
                        job["item_type"],
                        False, 
                        str(err)
                    ))
        
        succeeded = sum(1 for r in results if r[4])
        failed = [
            {
                "cohort": r[0], 
                "age_band": r[1], 
                "event_year": r[2],
                "item_type": r[3]
            }
            for r in results if not r[4]
        ]
        
        logger.info(
            f"🎉 FPGrowth filtering completed: {succeeded} / {len(results)} "
            f"succeeded on attempt {attempt+1}"
        )
        
        if failed:
            logger.info("Failed jobs:")
            for f in failed[:20]:
                logger.info(
                    f"  {f['cohort']}/{f['age_band']}/{f['event_year']}/{f['item_type']}"
                )
        
        attempt += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run FPGrowth-based initial filtering for cohort analysis"
    )
    parser.add_argument(
        "--cohort",
        type=str,
        default=None,
        help="Cohort name (opioid_ed, ed_non_opioid) or comma-separated list"
    )
    parser.add_argument(
        "--item-type",
        type=str,
        default=None,
        help="Item type (drug, icd, cpt). If not specified, processes all relevant types"
    )
    parser.add_argument(
        "--age-band",
        type=str,
        default=None,
        help="Specific age band (e.g., '65-74') or all if not specified"
    )
    parser.add_argument(
        "--event-year",
        type=int,
        default=None,
        help="Specific event year (e.g., 2020) or all if not specified"
    )
    parser.add_argument(
        "--min-support-threshold",
        type=float,
        default=None,
        help="Minimum support threshold (overrides default)"
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=None,
        help="Timeout in seconds (overrides default)"
    )
    
    args = parser.parse_args()
    
    # Parse cohort argument
    cohort_arg = None
    if args.cohort:
        if "," in args.cohort:
            cohort_arg = [c.strip() for c in args.cohort.split(",")]
        else:
            cohort_arg = args.cohort
    
    # Execute pipeline
    execute_filtering_pipeline(
        cohort=cohort_arg,
        item_type=args.item_type,
        age_band=args.age_band,
        event_year=args.event_year
    )

