#!/usr/bin/env python3
"""
Cohort-Specific FPGrowth Feature Importance Analysis

Processes each cohort separately to find cohort-specific patterns across:
- drug_name (pharmacy events)
- icd_code (medical diagnosis codes  
- cpt_code (medical procedure codes)

Outputs to: s3://pgxdatalake/gold/fpgrowth/cohort/{item_type}/cohort_name={cohort}/age_band={age}/event_year={year}/
"""

import sys
import time
import json
import logging
from pathlib import Path
from typing import Dict, List
from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd
import numpy as np
import boto3
import psutil
import duckdb
from mlxtend.frequent_patterns import fpgrowth, association_rules
from mlxtend.preprocessing import TransactionEncoder

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# =============================================================================
# CONFIGURATION
# =============================================================================

# FP-Growth parameters (higher threshold for cohort-specific patterns)
MIN_SUPPORT = 0.05       # 5% support (items must appear in 5% of patients within cohort)
MIN_CONFIDENCE = 0.5     # 50% confidence - only strong associations

# CPT-specific parameters (prevent memory exhaustion from millions of rules)
MIN_SUPPORT_CPT = 0.15   # 15% support for CPT codes (focuses on common patterns)
MIN_CONFIDENCE_CPT = 0.6 # 60% confidence for CPT (very strong associations only)

# Rule limits (focus on most important rules)
MAX_RULES_PER_COHORT = 1000  # Keep top 1000 rules by lift (practical limit)

# Target-focused rule mining
TARGET_FOCUSED = True  # Only generate rules that predict target outcomes
TARGET_ICD_CODES = ['F11.20', 'F11.21', 'F11.22', 'F11.23', 'F11.24', 'F11.25', 'F11.29']  # Opioid dependence codes
TARGET_HCG_LINES = [
    "P51 - ER Visits and Observation Care",
    "O11 - Emergency Room",
    "P33 - Urgent Care Visits"
]  # ED visits (HCG Line codes)
TARGET_PREFIXES = ['TARGET_ICD:', 'TARGET_ED:']  # Prefixes for target items in transactions

# Processing parameters
MAX_WORKERS = 1  # Sequential processing to prevent memory issues

# Transaction density bins (based on histogram/percentiles)
DENSITY_BINS = ['low', 'medium', 'high', 'extreme']  # Process in this order

# Itemset filtering (remove common/trivial itemsets)
MIN_ITEMSET_LIFT = 1.1  # Filter itemsets with lift < 1.1 (items are independent/not interesting)

# DRY RUN MODE (test with limited cohorts first)
DRY_RUN = True  # Set to False to process all cohorts
DRY_RUN_LIMIT = 5  # Number of cohort combinations to process in dry run

COHORTS_TO_PROCESS = ['opioid_ed', 'non_opioid_ed']  # Specify cohorts to process

ITEM_TYPES = ['drug_name', 'icd_code', 'cpt_code', 'medical_code']
S3_OUTPUT_BASE = "s3://pgxdatalake/gold/fpgrowth/cohort"
LOCAL_DATA_PATH = Path("/mnt/nvme/cohorts")  # Instance storage (NVMe SSD for fast I/O)

# Optional model_data root (filtered to important features + 5:1 control ratio).
# If a model_data file exists for a given (cohort, age_band), FP-Growth will
# prefer it over the raw GOLD cohorts parquet.
MODEL_DATA_ROOT = PROJECT_ROOT / "model_data"
USE_MODEL_DATA_IF_AVAILABLE = True

# Local FP-Growth outputs (mirrors feature-importance naming with cohort + age_band)
LOCAL_OUTPUT_ROOT = PROJECT_ROOT / "4_fpgrowth_analysis" / "outputs"

# Optional model_data root (filtered to important features + 5:1 control ratio).
# If a model_data file exists for a given (cohort, age_band), FP-Growth will
# prefer it over the raw GOLD cohorts parquet.
MODEL_DATA_ROOT = PROJECT_ROOT / "model_data"
USE_MODEL_DATA_IF_AVAILABLE = True

# =============================================================================
# SETUP LOGGING
# =============================================================================

def setup_logger(name: str = 'cohort_fpgrowth') -> logging.Logger:
    """Setup logger with console output."""
    logger = logging.Logger(name)
    logger.setLevel(logging.INFO)
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger

# =============================================================================
# COHORT PROCESSING
# =============================================================================

def log_memory(logger, stage=""):
    """Log current memory usage."""
    try:
        mem = psutil.virtual_memory()
        mem_used_gb = mem.used / (1024**3)
        mem_total_gb = mem.total / (1024**3)
        mem_percent = mem.percent
        mem_avail_gb = mem.available / (1024**3)
        
        logger.info(f"[MEMORY {stage}] Used: {mem_used_gb:.1f} GB / {mem_total_gb:.1f} GB ({mem_percent:.1f}%) | Available: {mem_avail_gb:.1f} GB")
        
        # Warning if memory usage is high
        if mem_percent > 85:
            logger.warning(f"⚠️  HIGH MEMORY USAGE: {mem_percent:.1f}% - May cause OOM!")
        
        return mem_percent
    except Exception as e:
        logger.error(f"Error getting memory info: {e}")
        return 0.0


def assign_transaction_density(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """
    Calculate transaction sizes per patient and assign Transaction_Density bins.
    
    Args:
        df: DataFrame with mi_person_key and item columns
        logger: Logger instance
    
    Returns:
        DataFrame with Transaction_Density column added
    """
    # Calculate transaction size per patient
    logger.info(f"Calculating transaction sizes per patient...")
    transaction_sizes = df.groupby('mi_person_key')['item'].size().reset_index(name='transaction_size')
    
    # Calculate percentiles for binning
    sizes = transaction_sizes['transaction_size'].values
    p25 = np.percentile(sizes, 25)
    p50 = np.percentile(sizes, 50)
    p75 = np.percentile(sizes, 75)
    p95 = np.percentile(sizes, 95)
    
    logger.info(f"Transaction size percentiles:")
    logger.info(f"  P25: {p25:.1f} items")
    logger.info(f"  P50 (median): {p50:.1f} items")
    logger.info(f"  P75: {p75:.1f} items")
    logger.info(f"  P95: {p95:.1f} items")
    logger.info(f"  Max: {max(sizes):,} items")
    
    # Assign density bins based on percentiles
    def assign_density(size):
        if size <= p25:
            return 'low'
        elif size <= p50:
            return 'medium'
        elif size <= p95:
            return 'high'
        else:
            return 'extreme'
    
    transaction_sizes['Transaction_Density'] = transaction_sizes['transaction_size'].apply(assign_density)
    
    # Log distribution
    density_counts = transaction_sizes['Transaction_Density'].value_counts()
    logger.info(f"Transaction density distribution:")
    for density in DENSITY_BINS:
        count = density_counts.get(density, 0)
        pct = (count / len(transaction_sizes)) * 100 if len(transaction_sizes) > 0 else 0
        logger.info(f"  {density}: {count:,} ({pct:.1f}%)")
    
    # Merge density back to original dataframe
    df_with_density = df.merge(
        transaction_sizes[['mi_person_key', 'Transaction_Density', 'transaction_size']],
        on='mi_person_key',
        how='left'
    )
    
    return df_with_density


def get_transactions_by_density(df: pd.DataFrame, density: str, logger: logging.Logger) -> List[List[str]]:
    """
    Get transactions for a specific density level.
    
    Args:
        df: DataFrame with mi_person_key, item, and Transaction_Density columns
        density: Density level ('low', 'medium', 'high', 'extreme')
        logger: Logger instance
    
    Returns:
        List of transactions (each transaction is a list of items)
    """
    df_density = df[df['Transaction_Density'] == density]
    
    if len(df_density) == 0:
        return []
    
    transactions = (
        df_density.groupby('mi_person_key')['item']
        .apply(lambda x: sorted(set(x.tolist())))
        .tolist()
    )
    
    logger.info(f"  {density}: {len(transactions):,} transactions")
    
    return transactions


def filter_itemsets_by_lift(
    itemsets: pd.DataFrame,
    df_encoded: pd.DataFrame,
    min_lift: float,
    logger: logging.Logger
) -> pd.DataFrame:
    """
    Filter itemsets by lift to remove common/trivial itemsets.
    
    Lift measures how much more likely items are to appear together than by chance.
    Lift = 1.0 means items are independent (not interesting)
    Lift > 1.0 means positive association (interesting)
    Lift < 1.0 means negative association (also interesting, but we filter these out)
    
    Args:
        itemsets: DataFrame with 'itemsets' and 'support' columns
        df_encoded: Encoded transaction DataFrame (needed to calculate individual item supports)
        min_lift: Minimum lift threshold (e.g., 1.1 = 10% more likely than chance)
        logger: Logger instance
    
    Returns:
        Filtered DataFrame with only itemsets above min_lift threshold
    """
    if len(itemsets) == 0:
        return itemsets
    
    logger.info(f"Filtering {len(itemsets):,} itemsets by lift (min_lift={min_lift})...")
    
    # Calculate individual item supports (needed for lift calculation)
    item_supports = {}
    total_transactions = len(df_encoded)
    
    for col in df_encoded.columns:
        item_supports[col] = df_encoded[col].sum() / total_transactions
    
    # Calculate lift for each itemset
    def calculate_lift(row):
        itemset = row['itemsets']
        itemset_support = row['support']
        
        # For single-item itemsets, lift is undefined (or 1.0 by convention)
        if len(itemset) == 1:
            return 1.0  # Single items don't have lift
        
        # For multi-item itemsets: lift = itemset_support / (item1_support * item2_support * ...)
        expected_support = 1.0
        for item in itemset:
            if item in item_supports:
                expected_support *= item_supports[item]
            else:
                # Item not found in transactions (shouldn't happen, but handle gracefully)
                return 0.0
        
        if expected_support == 0:
            return 0.0
        
        lift = itemset_support / expected_support
        return lift
    
    itemsets['lift'] = itemsets.apply(calculate_lift, axis=1)
    
    # Filter by lift threshold
    original_count = len(itemsets)
    itemsets_filtered = itemsets[itemsets['lift'] >= min_lift].copy()
    filtered_count = len(itemsets_filtered)
    removed_count = original_count - filtered_count
    
    logger.info(f"  Original itemsets: {original_count:,}")
    logger.info(f"  Filtered itemsets: {filtered_count:,} (lift >= {min_lift})")
    logger.info(f"  Removed common/trivial: {removed_count:,} ({removed_count/original_count*100:.1f}%)")
    
    if filtered_count > 0:
        logger.info(f"  Lift range: {itemsets_filtered['lift'].min():.3f} - {itemsets_filtered['lift'].max():.3f}")
    
    return itemsets_filtered.drop(columns=['lift'])  # Remove lift column (not needed in output)


def process_single_cohort(
    item_type: str,
    cohort_name: str,
    age_band: str,
    event_year: int,
    local_data_path: Path,
    s3_output_base: str,
    min_support: float,
    min_confidence: float
) -> Dict:
    """
    Process a single cohort for a single item type.
    
    Returns:
        Dictionary with processing metrics
    """
    # Create process-specific logger
    logger = setup_logger(f'cohort_{cohort_name}_{age_band}_{event_year}_{item_type}')
    
    cohort_id = f"{cohort_name}/{age_band}/{event_year}"
    logger.info(f"Processing {item_type} for {cohort_id}")
    log_memory(logger, "START")
    
    start_time = time.time()
    
    try:
        # Simple in-memory connection (no AWS needed for local parquet reads)
        con = duckdb.connect(':memory:')
        con.sql("SET threads = 1")

        # Prefer filtered model_data (if available) over raw GOLD cohorts parquet
        model_data_file = (
            MODEL_DATA_ROOT
            / f"cohort_name={cohort_name}"
            / f"age_band={age_band}"
            / "model_events.parquet"
        )

        if USE_MODEL_DATA_IF_AVAILABLE and model_data_file.exists():
            parquet_file = model_data_file
            logger.info(f"Using model_data file for FP-Growth: {parquet_file}")
        else:
            parquet_file = (
                local_data_path
                / f"cohort_name={cohort_name}"
                / f"event_year={event_year}"
                / f"age_band={age_band}"
                / "cohort.parquet"
            )

        if not parquet_file.exists():
            logger.warning(f"✗ Cohort file not found: {parquet_file}")
            return {
                'item_type': item_type,
                'cohort_name': cohort_name,
                'age_band': age_band,
                'event_year': event_year,
                'error': 'File not found'
            }
        
        # Build query based on item type. Always include `target` so we can run
        # a separate target-only FP-Growth pass (within-case patterns).
        if item_type == 'drug_name':
            query = f"""
            SELECT mi_person_key, drug_name as item, target
            FROM read_parquet('{parquet_file}')
            WHERE
                drug_name IS NOT NULL
                AND drug_name != ''
                AND event_type = 'pharmacy'
                AND event_year = {event_year}
            """
        elif item_type == 'icd_code':
            # Collect from ALL ICD diagnosis columns (primary through ten)
            query = f"""
            WITH all_icds AS (
                SELECT mi_person_key, primary_icd_diagnosis_code as icd, target
                FROM read_parquet('{parquet_file}') 
                WHERE primary_icd_diagnosis_code IS NOT NULL AND event_type = 'medical' AND event_year = {event_year}
                UNION ALL
                SELECT mi_person_key, two_icd_diagnosis_code as icd, target
                FROM read_parquet('{parquet_file}') 
                WHERE two_icd_diagnosis_code IS NOT NULL AND event_type = 'medical' AND event_year = {event_year}
                UNION ALL
                SELECT mi_person_key, three_icd_diagnosis_code as icd, target
                FROM read_parquet('{parquet_file}') 
                WHERE three_icd_diagnosis_code IS NOT NULL AND event_type = 'medical' AND event_year = {event_year}
                UNION ALL
                SELECT mi_person_key, four_icd_diagnosis_code as icd, target
                FROM read_parquet('{parquet_file}') 
                WHERE four_icd_diagnosis_code IS NOT NULL AND event_type = 'medical' AND event_year = {event_year}
                UNION ALL
                SELECT mi_person_key, five_icd_diagnosis_code as icd, target
                FROM read_parquet('{parquet_file}') 
                WHERE five_icd_diagnosis_code IS NOT NULL AND event_type = 'medical' AND event_year = {event_year}
                UNION ALL
                SELECT mi_person_key, six_icd_diagnosis_code as icd, target
                FROM read_parquet('{parquet_file}') 
                WHERE six_icd_diagnosis_code IS NOT NULL AND event_type = 'medical' AND event_year = {event_year}
                UNION ALL
                SELECT mi_person_key, seven_icd_diagnosis_code as icd, target
                FROM read_parquet('{parquet_file}') 
                WHERE seven_icd_diagnosis_code IS NOT NULL AND event_type = 'medical' AND event_year = {event_year}
                UNION ALL
                SELECT mi_person_key, eight_icd_diagnosis_code as icd, target
                FROM read_parquet('{parquet_file}') 
                WHERE eight_icd_diagnosis_code IS NOT NULL AND event_type = 'medical' AND event_year = {event_year}
                UNION ALL
                SELECT mi_person_key, nine_icd_diagnosis_code as icd, target
                FROM read_parquet('{parquet_file}') 
                WHERE nine_icd_diagnosis_code IS NOT NULL AND event_type = 'medical' AND event_year = {event_year}
                UNION ALL
                SELECT mi_person_key, ten_icd_diagnosis_code as icd, target
                FROM read_parquet('{parquet_file}') 
                WHERE ten_icd_diagnosis_code IS NOT NULL AND event_type = 'medical' AND event_year = {event_year}
            )
            SELECT mi_person_key, icd as item, target FROM all_icds WHERE icd != ''
            """
        elif item_type == 'cpt_code':
            query = f"""
            SELECT mi_person_key, procedure_code as item, target
            FROM read_parquet('{parquet_file}')
            WHERE
                procedure_code IS NOT NULL
                AND procedure_code != ''
                AND event_type = 'medical'
                AND event_year = {event_year}
            """
        elif item_type == 'medical_code':
            # Combined ICD (all 10 diagnosis positions) + CPT codes in a single transaction space
            query = f"""
            WITH all_med_codes AS (
                SELECT mi_person_key, primary_icd_diagnosis_code as code, target
                FROM read_parquet('{parquet_file}')
                WHERE primary_icd_diagnosis_code IS NOT NULL AND primary_icd_diagnosis_code != '' AND event_type = 'medical' AND event_year = {event_year}
                UNION ALL
                SELECT mi_person_key, two_icd_diagnosis_code as code, target
                FROM read_parquet('{parquet_file}')
                WHERE two_icd_diagnosis_code IS NOT NULL AND two_icd_diagnosis_code != '' AND event_type = 'medical' AND event_year = {event_year}
                UNION ALL
                SELECT mi_person_key, three_icd_diagnosis_code as code, target
                FROM read_parquet('{parquet_file}')
                WHERE three_icd_diagnosis_code IS NOT NULL AND three_icd_diagnosis_code != '' AND event_type = 'medical' AND event_year = {event_year}
                UNION ALL
                SELECT mi_person_key, four_icd_diagnosis_code as code, target
                FROM read_parquet('{parquet_file}')
                WHERE four_icd_diagnosis_code IS NOT NULL AND four_icd_diagnosis_code != '' AND event_type = 'medical' AND event_year = {event_year}
                UNION ALL
                SELECT mi_person_key, five_icd_diagnosis_code as code, target
                FROM read_parquet('{parquet_file}')
                WHERE five_icd_diagnosis_code IS NOT NULL AND five_icd_diagnosis_code != '' AND event_type = 'medical' AND event_year = {event_year}
                UNION ALL
                SELECT mi_person_key, six_icd_diagnosis_code as code, target
                FROM read_parquet('{parquet_file}')
                WHERE six_icd_diagnosis_code IS NOT NULL AND six_icd_diagnosis_code != '' AND event_type = 'medical' AND event_year = {event_year}
                UNION ALL
                SELECT mi_person_key, seven_icd_diagnosis_code as code, target
                FROM read_parquet('{parquet_file}')
                WHERE seven_icd_diagnosis_code IS NOT NULL AND seven_icd_diagnosis_code != '' AND event_type = 'medical' AND event_year = {event_year}
                UNION ALL
                SELECT mi_person_key, eight_icd_diagnosis_code as code, target
                FROM read_parquet('{parquet_file}')
                WHERE eight_icd_diagnosis_code IS NOT NULL AND eight_icd_diagnosis_code != '' AND event_type = 'medical' AND event_year = {event_year}
                UNION ALL
                SELECT mi_person_key, nine_icd_diagnosis_code as code, target
                FROM read_parquet('{parquet_file}')
                WHERE nine_icd_diagnosis_code IS NOT NULL AND nine_icd_diagnosis_code != '' AND event_type = 'medical' AND event_year = {event_year}
                UNION ALL
                SELECT mi_person_key, ten_icd_diagnosis_code as code, target
                FROM read_parquet('{parquet_file}')
                WHERE ten_icd_diagnosis_code IS NOT NULL AND ten_icd_diagnosis_code != '' AND event_type = 'medical' AND event_year = {event_year}
                UNION ALL
                SELECT mi_person_key, procedure_code as code, target
                FROM read_parquet('{parquet_file}')
                WHERE procedure_code IS NOT NULL AND procedure_code != '' AND event_type = 'medical' AND event_year = {event_year}
            )
            SELECT mi_person_key, code as item, target FROM all_med_codes WHERE code != ''
            """
        else:
            raise ValueError(f"Unknown item_type: {item_type}")
        
        # Load data
        df = con.execute(query).df()
        log_memory(logger, "After data extraction")
        con.close()
        
        if len(df) == 0:
            logger.warning(f"✗ No {item_type} data for {cohort_id}")
            return {
                'item_type': item_type,
                'cohort_name': cohort_name,
                'age_band': age_band,
                'event_year': event_year,
                'error': 'No data'
            }
        
        # Assign Transaction_Density based on histogram/percentiles
        logger.info(f"Assigning Transaction_Density to {len(df):,} rows...")
        df = assign_transaction_density(df, logger)
        log_memory(logger, "After density assignment")
        
        # Process transactions by density in order: low -> medium -> high -> extreme
        all_itemsets = []
        all_rules = []
        density_counts = {}
        
        logger.info(f"Processing transactions by density level...")
        for density in DENSITY_BINS:
            transactions = get_transactions_by_density(df, density, logger)
            density_counts[density] = len(transactions)
            
            if len(transactions) < 10:
                logger.warning(f"⚠️  Insufficient {density} density transactions ({len(transactions)}) - skipping")
                continue
            
            try:
                logger.info(f"Processing {density} density transactions...")
                
                # Encode transactions
                te = TransactionEncoder()
                te_ary = te.fit(transactions).transform(transactions)
                df_encoded = pd.DataFrame(te_ary, columns=te.columns_)
                log_memory(logger, f"After encoding ({density})")
                
                # Adjust support threshold based on density (lower support for extreme)
                density_support = min_support
                if density == 'extreme':
                    density_support = max(min_support * 0.5, 0.01)  # At least 1% support
                    logger.info(f"Using adjusted support threshold {density_support:.4f} for {density} density")
                
                # Run FP-Growth
                itemsets_density = fpgrowth(df_encoded, min_support=density_support, use_colnames=True)
                itemsets_density = itemsets_density.sort_values('support', ascending=False).reset_index(drop=True)
                
                # Filter out common/trivial itemsets by lift BEFORE generating rules
                if len(itemsets_density) > 0:
                    itemsets_density = filter_itemsets_by_lift(
                        itemsets_density, 
                        df_encoded, 
                        MIN_ITEMSET_LIFT, 
                        logger
                    )
                    log_memory(logger, f"After filtering itemsets by lift ({density})")
                
                if len(itemsets_density) > 0:
                    all_itemsets.append(itemsets_density)
                    log_memory(logger, f"After FP-Growth ({density})")
                    
                    # Generate association rules
                    rules_density = association_rules(itemsets_density, metric="confidence", min_threshold=min_confidence)
                    rules_density = rules_density.sort_values('lift', ascending=False).reset_index(drop=True)
                    all_rules.append(rules_density)
                    log_memory(logger, f"After rule generation ({density})")
                else:
                    logger.warning(f"No itemsets remaining after lift filtering for {density} density")
                    continue
                    
            except MemoryError as e:
                logger.error(f"⚠️  Memory error processing {density} density transactions: {e}")
                logger.warning(f"   Skipping {density} density transactions due to memory constraints")
            except Exception as e:
                logger.error(f"⚠️  Error processing {density} density transactions: {e}")
                logger.warning(f"   Skipping {density} density transactions")
        
        # Combine results
        if len(all_itemsets) == 0:
            logger.warning(f"✗ No frequent itemsets for {cohort_id}")
            return {
                'item_type': item_type,
                'cohort_name': cohort_name,
                'age_band': age_band,
                'event_year': event_year,
                'error': 'No frequent itemsets'
            }
        
        # Combine itemsets (deduplicate if needed)
        itemsets = pd.concat(all_itemsets, ignore_index=True)
        itemsets = itemsets.drop_duplicates(subset=['itemsets'])
        itemsets = itemsets.sort_values('support', ascending=False).reset_index(drop=True)
        
        # Combine rules
        if len(all_rules) > 0:
            rules = pd.concat(all_rules, ignore_index=True)
            rules = rules.drop_duplicates(subset=['antecedents', 'consequents'])
            rules = rules.sort_values('lift', ascending=False).reset_index(drop=True)
        else:
            rules = pd.DataFrame()
        
        # Create encoding map
        encoding_map = {}
        for idx, row in itemsets.iterrows():
            itemset = row['itemsets']
            if len(itemset) == 1:
                item = list(itemset)[0]
                encoding_map[item] = {
                    'support': float(row['support']),
                    'rank': int(idx)
                }
        
        # Save to S3
        s3_path = f"{s3_output_base}/{item_type}/cohort_name={cohort_name}/age_band={age_band}/event_year={event_year}"
        
        # Convert frozensets to lists
        itemsets_json = itemsets.copy()
        itemsets_json['itemsets'] = itemsets_json['itemsets'].apply(lambda x: list(x))
        
        rules_json = rules.copy()
        if len(rules) > 0:
            rules_json['antecedents'] = rules_json['antecedents'].apply(lambda x: list(x))
            rules_json['consequents'] = rules_json['consequents'].apply(lambda x: list(x))
        
        # Upload to S3
        s3_client = boto3.client('s3')
        bucket = 'pgxdatalake'
        prefix = s3_path.replace('s3://pgxdatalake/', '')
        
        s3_client.put_object(
            Bucket=bucket,
            Key=f"{prefix}/encoding_map.json",
            Body=json.dumps(encoding_map, indent=2)
        )
        
        s3_client.put_object(
            Bucket=bucket,
            Key=f"{prefix}/itemsets.json",
            Body=itemsets_json.to_json(orient='records', indent=2)
        )
        
        s3_client.put_object(
            Bucket=bucket,
            Key=f"{prefix}/rules.json",
            Body=rules_json.to_json(orient='records', indent=2)
        )
        
        metrics = {
            'item_type': item_type,
            'cohort_name': cohort_name,
            'age_band': age_band,
            'event_year': event_year,
            'unique_items': len(df['item'].unique()),
            'total_transactions': sum(density_counts.values()),
            'density_distribution': density_counts,
            'frequent_itemsets': len(itemsets),
            'association_rules': len(rules),
            'encoding_map_size': len(encoding_map),
            'processing_time_seconds': time.time() - start_time
        }
        
        s3_client.put_object(
            Bucket=bucket,
            Key=f"{prefix}/metrics.json",
            Body=json.dumps(metrics, indent=2)
        )

        # ------------------------------------------------------------------
        # Save local copies for COMBINED view
        # Directory layout: outputs/{cohort_name}/combined/{age_band_fname}/{event_year}/
        # ------------------------------------------------------------------
        try:
            age_band_fname = age_band.replace("-", "_")
            combined_dir = LOCAL_OUTPUT_ROOT / cohort_name / "combined" / age_band_fname / str(event_year)
            combined_dir.mkdir(parents=True, exist_ok=True)

            # encoding_map
            (combined_dir / f"{item_type}_encoding_map.json").write_text(
                json.dumps(encoding_map, indent=2)
            )

            # itemsets
            itemsets_json.to_json(
                combined_dir / f"{item_type}_itemsets.json",
                orient="records",
                indent=2,
            )

            # rules
            rules_json.to_json(
                combined_dir / f"{item_type}_rules.json",
                orient="records",
                indent=2,
            )

            # metrics
            (combined_dir / f"{item_type}_metrics.json").write_text(
                json.dumps(metrics, indent=2)
            )

            logger.info(f"Saved local FP-Growth (combined) outputs under {combined_dir}")
        except Exception as e:
            logger.warning(f"Failed to write local FP-Growth combined outputs: {e}")

        # ------------------------------------------------------------------
        # Target-only FP-Growth: within-target patterns (target == 1)
        # ------------------------------------------------------------------
        try:
            if 'target' in df.columns:
                logger.info("Running target-only FP-Growth (target == 1)...")
                df_target = df[df["target"] == 1].copy()
                if len(df_target) == 0:
                    logger.warning("No target=1 rows; skipping target-only FP-Growth.")
                else:
                    # Build transactions for target-only cohort (no density stratification)
                    tx_target = (
                        df_target.groupby('mi_person_key')['item']
                        .apply(lambda x: sorted(set(x.tolist())))
                        .tolist()
                    )
                    if len(tx_target) < 10:
                        logger.warning(f"Insufficient target-only transactions ({len(tx_target)}); skipping.")
                    else:
                        te_t = TransactionEncoder()
                        te_ary_t = te_t.fit(tx_target).transform(tx_target)
                        df_enc_t = pd.DataFrame(te_ary_t, columns=te_t.columns_)

                        itemsets_t = fpgrowth(
                            df_enc_t,
                            min_support=min_support,
                            use_colnames=True,
                        )
                        if len(itemsets_t) > 0:
                            itemsets_t = filter_itemsets_by_lift(
                                itemsets_t, df_enc_t, MIN_ITEMSET_LIFT, logger
                            )

                        try:
                            rules_t = association_rules(
                                itemsets_t,
                                metric="confidence",
                                min_threshold=min_confidence,
                            )
                        except Exception as e_rules:
                            logger.warning(f"Target-only association_rules failed: {e_rules}")
                            rules_t = pd.DataFrame(columns=['antecedents', 'consequents'])

                        # Convert to JSON-friendly forms
                        itemsets_t_json = itemsets_t.copy()
                        if 'itemsets' in itemsets_t_json.columns:
                            itemsets_t_json['itemsets'] = itemsets_t_json['itemsets'].apply(lambda x: list(x))

                        rules_t_json = rules_t.copy()
                        if 'antecedents' in rules_t_json.columns:
                            rules_t_json['antecedents'] = rules_t_json['antecedents'].apply(lambda x: list(x))
                        if 'consequents' in rules_t_json.columns:
                            rules_t_json['consequents'] = rules_t_json['consequents'].apply(lambda x: list(x))

                        # S3: write alongside combined, with *_target_only.json suffix
                        s3_client.put_object(
                            Bucket=bucket,
                            Key=f"{prefix}/itemsets_target_only.json",
                            Body=itemsets_t_json.to_json(orient='records', indent=2),
                        )
                        s3_client.put_object(
                            Bucket=bucket,
                            Key=f"{prefix}/rules_target_only.json",
                            Body=rules_t_json.to_json(orient='records', indent=2),
                        )

                        # Local: outputs/{cohort_name}/target/{age_band_fname}/{event_year}/
                        target_dir = LOCAL_OUTPUT_ROOT / cohort_name / "target" / age_band_fname / str(event_year)
                        target_dir.mkdir(parents=True, exist_ok=True)

                        itemsets_t_json.to_json(
                            target_dir / f"{item_type}_itemsets_target_only.json",
                            orient="records",
                            indent=2,
                        )
                        rules_t_json.to_json(
                            target_dir / f"{item_type}_rules_target_only.json",
                            orient="records",
                            indent=2,
                        )

                        logger.info(f"Saved target-only FP-Growth outputs under {target_dir}")
        except Exception as e:
            logger.warning(f"Target-only FP-Growth encountered an error: {e}")

        elapsed = time.time() - start_time
        log_memory(logger, "END")
        logger.info(f"✓ {cohort_id} {item_type}: {len(itemsets):,} itemsets, {len(rules):,} rules in {elapsed:.1f}s")

        return metrics
        
    except Exception as e:
        logger.error(f"✗ Failed {cohort_id} {item_type}: {e}")
        return {
            'item_type': item_type,
            'cohort_name': cohort_name,
            'age_band': age_band,
            'event_year': event_year,
            'error': str(e)
        }

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution function."""
    logger = setup_logger()
    
    logger.info("="*80)
    logger.info("COHORT-SPECIFIC FPGROWTH FEATURE IMPORTANCE ANALYSIS")
    logger.info("="*80)
    logger.info(f"Min Support: {MIN_SUPPORT}")
    logger.info(f"Min Confidence: {MIN_CONFIDENCE}")
    logger.info(f"Min Itemset Lift: {MIN_ITEMSET_LIFT} (filtering common/trivial itemsets)")
    logger.info(f"Max Workers: {MAX_WORKERS}")
    logger.info(f"Item Types: {ITEM_TYPES}")
    logger.info(f"S3 Output: {S3_OUTPUT_BASE}")
    logger.info(f"Local Data: {LOCAL_DATA_PATH}")
    logger.info(f"Local Data Exists: {LOCAL_DATA_PATH.exists()}")
    logger.info(f"Model Data Root: {MODEL_DATA_ROOT} (exists={MODEL_DATA_ROOT.exists()})")
    logger.info("="*80)
    
    if not LOCAL_DATA_PATH.exists() and not MODEL_DATA_ROOT.exists():
        logger.error(f"✗ Local data path does not exist: {LOCAL_DATA_PATH}")
        logger.error(f"✗ Model data root does not exist: {MODEL_DATA_ROOT}")
        logger.error(
            "  On EC2, sync from S3 with:\n"
            "    aws s3 sync s3://pgxdatalake/gold/cohorts_F1120/ /mnt/nvme/cohorts/"
        )
        logger.error(
            "  For local development, sync to ./data/cohorts_F1120/ and "
            "either set LOCAL_DATA_PATH accordingly or export LOCAL_DATA_PATH, "
            "or generate filtered model_data first."
        )
        sys.exit(1)
    
    # Generate all cohort combinations
    cohort_jobs = []
    for item_type in ITEM_TYPES:
        for cohort_name in COHORT_NAMES:
            for age_band in AGE_BANDS:
                for event_year in EVENT_YEARS:
                    cohort_jobs.append((item_type, cohort_name, age_band, event_year))
    
    # Apply DRY_RUN limit if enabled
    if DRY_RUN and len(cohort_jobs) > DRY_RUN_LIMIT:
        logger.info(f"⚠️  DRY RUN: Limiting from {len(cohort_jobs)} to {DRY_RUN_LIMIT} cohort combinations")
        cohort_jobs = cohort_jobs[:DRY_RUN_LIMIT]
    
    total_jobs = len(cohort_jobs)
    logger.info(f"Total cohort jobs: {total_jobs}")
    if DRY_RUN:
        logger.info(f"DRY RUN MODE: Processing only {DRY_RUN_LIMIT} combinations (set DRY_RUN = False for full run)")
    else:
        logger.info(f"FULL RUN MODE: Processing all cohorts")
    logger.info("="*80)
    
    # Process cohorts in parallel
    all_metrics = []
    completed = 0
    failed = 0
    
    overall_start = time.time()
    
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all jobs
        future_to_job = {
            executor.submit(
                process_single_cohort,
                item_type, cohort_name, age_band, event_year,
                LOCAL_DATA_PATH, S3_OUTPUT_BASE,
                MIN_SUPPORT, MIN_CONFIDENCE
            ): (item_type, cohort_name, age_band, event_year)
            for item_type, cohort_name, age_band, event_year in cohort_jobs
        }
        
        # Process results as they complete
        for future in as_completed(future_to_job):
            job = future_to_job[future]
            try:
                metrics = future.result()
                all_metrics.append(metrics)
                completed += 1
                
                if 'error' in metrics:
                    failed += 1
                
                # Progress update every 10 completions
                if completed % 10 == 0:
                    elapsed = time.time() - overall_start
                    rate = completed / elapsed
                    remaining = (total_jobs - completed) / rate if rate > 0 else 0
                    logger.info(f"Progress: {completed}/{total_jobs} ({completed/total_jobs*100:.1f}%) - "
                              f"ETA: {remaining/60:.1f} min")
            except Exception as e:
                logger.error(f"✗ Job {job} raised exception: {e}")
                failed += 1
    
    # Final summary
    total_time = time.time() - overall_start
    successful = completed - failed
    
    logger.info("\n" + "="*80)
    logger.info("COHORT ANALYSIS COMPLETE")
    logger.info("="*80)
    logger.info(f"Total Runtime: {total_time/60:.1f} minutes")
    logger.info(f"Total Jobs: {total_jobs}")
    logger.info(f"Successful: {successful} ({successful/total_jobs*100:.1f}%)")
    logger.info(f"Failed: {failed} ({failed/total_jobs*100:.1f}%)")
    logger.info("="*80)
    
    # Summary by item type
    for item_type in ITEM_TYPES:
        item_metrics = [m for m in all_metrics if m['item_type'] == item_type and 'error' not in m]
        if item_metrics:
            total_itemsets = sum(m['frequent_itemsets'] for m in item_metrics)
            total_rules = sum(m['association_rules'] for m in item_metrics)
            logger.info(f"  {item_type}: {len(item_metrics)} cohorts, {total_itemsets:,} itemsets, {total_rules:,} rules")
    
    # EC2 Auto-Shutdown (optional)
    shutdown_ec2(logger)


def shutdown_ec2(logger: logging.Logger, enable: bool = False):
    """
    Automatically shutdown EC2 instance after analysis completes.
    
    Args:
        logger: Logger instance
        enable: Set to True to enable auto-shutdown, False to skip
    """
    if not enable:
        logger.info("\n" + "="*80)
        logger.info("EC2 Auto-Shutdown: DISABLED")
        logger.info("="*80)
        logger.info("To enable auto-shutdown, set enable=True in shutdown_ec2() call")
        logger.info("Instance will continue running.")
        logger.info("\nTo manually stop this instance later:")
        logger.info("  aws ec2 stop-instances --instance-ids $(ec2-metadata --instance-id | cut -d ' ' -f 2)")
        logger.info("Or use AWS Console: EC2 > Instances > Select instance > Instance State > Stop")
        return
    
    logger.info("\n" + "="*80)
    logger.info("Shutting down EC2 instance...")
    logger.info("="*80)
    
    import subprocess
    import requests
    import shutil
    
    # Get instance ID from EC2 metadata service
    try:
        response = requests.get(
            "http://169.254.169.254/latest/meta-data/instance-id",
            timeout=2
        )
        if response.status_code == 200:
            instance_id = response.text.strip()
            logger.info(f"Instance ID: {instance_id}")
            
            # Find AWS CLI
            aws_cmd = shutil.which("aws")
            if not aws_cmd:
                # Try common paths
                for path in ["/usr/local/bin/aws", "/usr/bin/aws", 
                           "/home/ec2-user/.local/bin/aws", 
                           "/home/ubuntu/.local/bin/aws"]:
                    if Path(path).exists():
                        aws_cmd = path
                        break
            
            if aws_cmd:
                # Stop the instance (use terminate-instances for permanent deletion)
                shutdown_cmd = [aws_cmd, "ec2", "stop-instances", "--instance-ids", instance_id]
                
                logger.info(f"Running: {' '.join(shutdown_cmd)}")
                result = subprocess.run(shutdown_cmd, capture_output=True, text=True)
                
                if result.returncode == 0:
                    logger.info("✓ EC2 instance stop command sent successfully")
                    logger.info("Instance will stop in a few moments.")
                    logger.info("Note: This is a STOP (not terminate), so you can restart it later.")
                    if result.stdout:
                        logger.info(f"\nAWS Response:\n{result.stdout}")
                else:
                    logger.error(f"✗ EC2 stop command failed with exit code {result.returncode}")
                    if result.stderr:
                        logger.error(f"Error: {result.stderr}")
                    logger.error("Check AWS credentials and IAM permissions.")
            else:
                logger.error("✗ AWS CLI not found. Cannot shutdown instance.")
                logger.error("Install AWS CLI or ensure it's in your PATH.")
                logger.error("Manual shutdown: aws ec2 stop-instances --instance-ids " + instance_id)
        else:
            logger.error(f"✗ Metadata service returned status code {response.status_code}")
            logger.error("Could not retrieve instance ID.")
    
    except requests.exceptions.RequestException as e:
        logger.error("✗ Could not retrieve instance ID from metadata service.")
        logger.error(f"Error: {e}")
        logger.error("If running on EC2, check that metadata service is accessible.")
        logger.error("\nManual shutdown command:")
        logger.error("  aws ec2 stop-instances --instance-ids <your-instance-id>")
    
    except Exception as e:
        logger.error(f"✗ Unexpected error during shutdown: {e}")


if __name__ == "__main__":
    main()


