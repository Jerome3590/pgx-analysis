#!/usr/bin/env python3
"""
Global FPGrowth Feature Importance Analysis

Processes all cohort data to find global patterns across:
- drug_name (pharmacy events)
- icd_code (medical diagnosis codes)
- cpt_code (medical procedure codes)

Outputs to: s3://pgxdatalake/gold/fpgrowth/global/{item_type}/
"""

import sys
import time
import json
import logging
from pathlib import Path
from typing import List, Dict
from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd
import numpy as np
import boto3
from mlxtend.frequent_patterns import fpgrowth, association_rules
from mlxtend.preprocessing import TransactionEncoder

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from helpers_1997_13.duckdb_utils import get_duckdb_connection

# =============================================================================
# CONFIGURATION
# =============================================================================

# FP-Growth parameters (quality-focused for ML features)
MIN_SUPPORT = 0.01       # Items must appear in 1% of patients (5.7M patients = 57K occurrences)
MIN_CONFIDENCE = 0.4     # 40% confidence - meaningful associations for CatBoost

# Item-specific thresholds (balance coverage vs quality)
MIN_CONFIDENCE_CPT = 0.5 # 50% confidence for CPT - strong procedure associations
MIN_SUPPORT_CPT = 0.02   # 2% support for CPT - focuses on common procedures

# Rule limits (quality over quantity)
MAX_RULES_PER_ITEM_TYPE = 5000  # Top 5000 rules by lift (for ML feature engineering)

# Transaction density bins (based on histogram/percentiles)
DENSITY_BINS = ['low', 'medium', 'high', 'extreme']  # Process in this order

# Itemset filtering (remove common/trivial itemsets)
MIN_ITEMSET_LIFT = 1.1  # Filter itemsets with lift < 1.1 (items are independent/not interesting)

# Target-focused rule mining
TARGET_FOCUSED = True  # Only generate rules that predict target outcomes
TARGET_ICD_CODES = ['F11.20', 'F11.21', 'F11.22', 'F11.23', 'F11.24', 'F11.25', 'F11.29']  # Opioid dependence codes
TARGET_HCG_LINES = [
    "P51 - ER Visits and Observation Care",
    "O11 - Emergency Room",
    "P33 - Urgent Care Visits"
]  # ED visits (HCG Line codes)
TARGET_PREFIXES = ['TARGET_ICD:', 'TARGET_ED:']  # Prefixes for target items in transactions

ITEM_TYPES = ['drug_name', 'icd_code', 'cpt_code']
S3_OUTPUT_BASE = "s3://pgxdatalake/gold/fpgrowth/global"
LOCAL_DATA_PATH = Path("/mnt/nvme/cohorts")  # Instance storage (NVMe SSD for fast I/O)

# =============================================================================
# SETUP LOGGING
# =============================================================================

def setup_logger(name: str = 'global_fpgrowth') -> logging.Logger:
    """Setup logger with console output."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger


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

# =============================================================================
# EXTRACTION FUNCTIONS
# =============================================================================

def extract_global_items(local_data_path: Path, item_type: str, logger: logging.Logger) -> List[str]:
    """
    Extract all unique items of specified type from local cohort data.
    
    Args:
        local_data_path: Path to local cohort parquet files
        item_type: 'drug_name', 'icd_code', or 'cpt_code'
        logger: Logger instance
    
    Returns:
        List of unique items
    """
    logger.info(f"Extracting global {item_type}s from local cohort data...")
    start_time = time.time()
    
    # Get DuckDB connection
    con = get_duckdb_connection(logger=logger)
    
    # Build glob pattern for all parquet files
    parquet_pattern = str(local_data_path / "**" / "cohort.parquet")
    
    # Build query based on item type
    if item_type == 'drug_name':
        query = f"""
        SELECT DISTINCT drug_name as item
        FROM read_parquet('{parquet_pattern}', hive_partitioning=1)
        WHERE drug_name IS NOT NULL 
          AND drug_name != ''
          AND event_type = 'pharmacy'
        ORDER BY item
        """
    elif item_type == 'icd_code':
        # Collect from all ICD diagnosis columns (first 5 for memory efficiency)
        query = f"""
        WITH all_icds AS (
            SELECT primary_icd_diagnosis_code as icd 
            FROM read_parquet('{parquet_pattern}', hive_partitioning=1) 
            WHERE primary_icd_diagnosis_code IS NOT NULL AND event_type = 'medical'
            UNION ALL
            SELECT two_icd_diagnosis_code as icd 
            FROM read_parquet('{parquet_pattern}', hive_partitioning=1) 
            WHERE two_icd_diagnosis_code IS NOT NULL AND event_type = 'medical'
            UNION ALL
            SELECT three_icd_diagnosis_code as icd 
            FROM read_parquet('{parquet_pattern}', hive_partitioning=1) 
            WHERE three_icd_diagnosis_code IS NOT NULL AND event_type = 'medical'
            UNION ALL
            SELECT four_icd_diagnosis_code as icd 
            FROM read_parquet('{parquet_pattern}', hive_partitioning=1) 
            WHERE four_icd_diagnosis_code IS NOT NULL AND event_type = 'medical'
            UNION ALL
            SELECT five_icd_diagnosis_code as icd 
            FROM read_parquet('{parquet_pattern}', hive_partitioning=1) 
            WHERE five_icd_diagnosis_code IS NOT NULL AND event_type = 'medical'
        )
        SELECT DISTINCT icd as item FROM all_icds WHERE icd != '' ORDER BY item
        """
    elif item_type == 'cpt_code':
        query = f"""
        SELECT DISTINCT procedure_code as item
        FROM read_parquet('{parquet_pattern}', hive_partitioning=1)
        WHERE procedure_code IS NOT NULL 
          AND procedure_code != ''
          AND event_type = 'medical'
        ORDER BY item
        """
    else:
        raise ValueError(f"Unknown item_type: {item_type}")
    
    logger.info(f"Running query for {item_type}...")
    df = con.execute(query).df()
    con.close()
    
    items = df['item'].tolist()
    
    elapsed = time.time() - start_time
    logger.info(f"✓ Extracted {len(items):,} unique {item_type}s in {elapsed:.1f}s")
    
    return items


def create_global_transactions(local_data_path: Path, item_type: str, logger: logging.Logger) -> List[List[str]]:
    """
    Create patient-level transactions from local cohort data.
    
    Args:
        local_data_path: Path to local cohort parquet files
        item_type: 'drug_name', 'icd_code', or 'cpt_code'
        logger: Logger instance
    
    Returns:
        List of transactions (each transaction is a list of items)
    """
    logger.info(f"Creating global {item_type} transactions...")
    start_time = time.time()
    
    # Get DuckDB connection
    con = get_duckdb_connection(logger=logger)
    
    # Build glob pattern for all parquet files
    parquet_pattern = str(local_data_path / "**" / "cohort.parquet")
    
    # Build query based on item type
    if item_type == 'drug_name':
        query = f"""
        SELECT mi_person_key, drug_name as item
        FROM read_parquet('{parquet_pattern}', hive_partitioning=1)
        WHERE drug_name IS NOT NULL AND drug_name != '' AND event_type = 'pharmacy'
        """
    elif item_type == 'icd_code':
        # Collect from first 5 ICD diagnosis columns for memory efficiency
        query = f"""
        WITH all_icds AS (
            SELECT mi_person_key, primary_icd_diagnosis_code as icd 
            FROM read_parquet('{parquet_pattern}', hive_partitioning=1) 
            WHERE primary_icd_diagnosis_code IS NOT NULL AND event_type = 'medical'
            UNION ALL
            SELECT mi_person_key, two_icd_diagnosis_code as icd 
            FROM read_parquet('{parquet_pattern}', hive_partitioning=1) 
            WHERE two_icd_diagnosis_code IS NOT NULL AND event_type = 'medical'
            UNION ALL
            SELECT mi_person_key, three_icd_diagnosis_code as icd 
            FROM read_parquet('{parquet_pattern}', hive_partitioning=1) 
            WHERE three_icd_diagnosis_code IS NOT NULL AND event_type = 'medical'
            UNION ALL
            SELECT mi_person_key, four_icd_diagnosis_code as icd 
            FROM read_parquet('{parquet_pattern}', hive_partitioning=1) 
            WHERE four_icd_diagnosis_code IS NOT NULL AND event_type = 'medical'
            UNION ALL
            SELECT mi_person_key, five_icd_diagnosis_code as icd 
            FROM read_parquet('{parquet_pattern}', hive_partitioning=1) 
            WHERE five_icd_diagnosis_code IS NOT NULL AND event_type = 'medical'
        )
        SELECT mi_person_key, icd as item FROM all_icds WHERE icd != ''
        """
    elif item_type == 'cpt_code':
        query = f"""
        SELECT mi_person_key, procedure_code as item
        FROM read_parquet('{parquet_pattern}', hive_partitioning=1)
        WHERE procedure_code IS NOT NULL AND procedure_code != '' AND event_type = 'medical'
        """
    else:
        raise ValueError(f"Unknown item_type: {item_type}")
    
    logger.info(f"Loading {item_type} events...")
    df = con.execute(query).df()
    con.close()
    
    # Group by patient and create item lists
    logger.info(f"Assigning Transaction_Density to {len(df):,} rows...")
    df = assign_transaction_density(df, logger)
    
    elapsed = time.time() - start_time
    logger.info(f"✓ Created transactions with density assignment in {elapsed:.1f}s")
    
    # Return dataframe with density column (caller will process by density)
    return df

# =============================================================================
# FPGROWTH PROCESSING
# =============================================================================

def process_item_type(
    item_type: str,
    local_data_path: Path,
    s3_output_base: str,
    min_support: float,
    min_confidence: float,
    logger: logging.Logger
) -> Dict:
    """
    Process a single item type end-to-end: extract, create transactions, run FP-Growth, save results.
    
    Returns:
        Dictionary with processing metrics
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"Processing {item_type.upper()}")
    logger.info(f"{'='*80}")
    
    overall_start = time.time()
    
    try:
        # Step 1: Extract items
        items = extract_global_items(local_data_path, item_type, logger)
        
        # Step 2: Create transactions with density assignment
        df = create_global_transactions(local_data_path, item_type, logger)
        
        # Step 3: Process transactions by density in order: low -> medium -> high -> extreme
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
                logger.info(f"Encoding {len(transactions):,} {density} transactions...")
                encode_start = time.time()
                te = TransactionEncoder()
                te_ary = te.fit(transactions).transform(transactions)
                df_encoded = pd.DataFrame(te_ary, columns=te.columns_)
                encode_time = time.time() - encode_start
                logger.info(f"✓ Encoded to {df_encoded.shape} matrix in {encode_time:.1f}s")
                
                # Adjust support threshold based on density (lower support for extreme)
                density_support = min_support
                if density == 'extreme':
                    density_support = max(min_support * 0.5, 0.01)  # At least 1% support
                    logger.info(f"Using adjusted support threshold {density_support:.4f} for {density} density")
                
                # Run FP-Growth
                logger.info(f"Running FP-Growth on {density} transactions (min_support={density_support:.4f})...")
                fpgrowth_start = time.time()
                itemsets_density = fpgrowth(df_encoded, min_support=density_support, use_colnames=True)
                itemsets_density = itemsets_density.sort_values('support', ascending=False).reset_index(drop=True)
                fpgrowth_time = time.time() - fpgrowth_start
                logger.info(f"✓ Found {len(itemsets_density):,} frequent itemsets in {fpgrowth_time:.1f}s")
                
                # Filter out common/trivial itemsets by lift BEFORE generating rules
                if len(itemsets_density) > 0:
                    itemsets_density = filter_itemsets_by_lift(
                        itemsets_density, 
                        df_encoded, 
                        MIN_ITEMSET_LIFT, 
                        logger
                    )
                
                if len(itemsets_density) > 0:
                    all_itemsets.append(itemsets_density)
                else:
                    logger.warning(f"No itemsets remaining after lift filtering for {density} density")
                    continue
                
                # Generate association rules
                if len(itemsets_density) > 0:
                    logger.info(f"Generating association rules (min_confidence={min_confidence})...")
                    rules_start = time.time()
                    rules_density = association_rules(itemsets_density, metric="confidence", min_threshold=min_confidence)
                    rules_density = rules_density.sort_values('lift', ascending=False).reset_index(drop=True)
                    rules_time = time.time() - rules_start
                    logger.info(f"✓ Generated {len(rules_density):,} association rules in {rules_time:.1f}s")
                    all_rules.append(rules_density)
                    
            except MemoryError as e:
                logger.error(f"⚠️  Memory error processing {density} density transactions: {e}")
                logger.warning(f"   Skipping {density} density transactions due to memory constraints")
            except Exception as e:
                logger.error(f"⚠️  Error processing {density} density transactions: {e}")
                logger.warning(f"   Skipping {density} density transactions")
        
        # Combine results
        if len(all_itemsets) == 0:
            logger.warning(f"✗ No frequent itemsets found")
            itemsets = pd.DataFrame()
        else:
            itemsets = pd.concat(all_itemsets, ignore_index=True)
            itemsets = itemsets.drop_duplicates(subset=['itemsets'])
            itemsets = itemsets.sort_values('support', ascending=False).reset_index(drop=True)
            logger.info(f"✓ Combined {len(itemsets):,} total frequent itemsets")
        
        if len(all_rules) == 0:
            rules = pd.DataFrame()
        else:
            rules = pd.concat(all_rules, ignore_index=True)
            rules = rules.drop_duplicates(subset=['antecedents', 'consequents'])
            rules = rules.sort_values('lift', ascending=False).reset_index(drop=True)
            logger.info(f"✓ Combined {len(rules):,} total association rules")
        
        # Step 6: Create encoding map
        logger.info(f"Creating encoding map...")
        encoding_map = {}
        for idx, row in itemsets.iterrows():
            itemset = row['itemsets']
            if len(itemset) == 1:  # Single items only
                item = list(itemset)[0]
                encoding_map[item] = {
                    'support': float(row['support']),
                    'rank': int(idx)
                }
        logger.info(f"✓ Created encoding map with {len(encoding_map):,} items")
        
        # Step 7: Save results to S3
        logger.info(f"Saving results to S3...")
        save_start = time.time()
        
        s3_path = f"{s3_output_base}/{item_type}"
        
        # Convert frozensets to lists for JSON serialization
        itemsets_json = itemsets.copy()
        itemsets_json['itemsets'] = itemsets_json['itemsets'].apply(lambda x: list(x))
        
        rules_json = rules.copy()
        rules_json['antecedents'] = rules_json['antecedents'].apply(lambda x: list(x))
        rules_json['consequents'] = rules_json['consequents'].apply(lambda x: list(x))
        
        # Upload to S3
        s3_client = boto3.client('s3')
        bucket = 'pgxdatalake'
        prefix = s3_path.replace('s3://pgxdatalake/', '')
        
        # Save encoding map
        s3_client.put_object(
            Bucket=bucket,
            Key=f"{prefix}/encoding_map.json",
            Body=json.dumps(encoding_map, indent=2)
        )
        
        # Save itemsets
        s3_client.put_object(
            Bucket=bucket,
            Key=f"{prefix}/itemsets.json",
            Body=itemsets_json.to_json(orient='records', indent=2)
        )
        
        # Save rules
        s3_client.put_object(
            Bucket=bucket,
            Key=f"{prefix}/rules.json",
            Body=rules_json.to_json(orient='records', indent=2)
        )
        
        # Save metrics
        total_transactions = sum(density_counts.values())
        
        metrics = {
            'item_type': item_type,
            'min_support': min_support,
            'min_confidence': min_confidence,
            'unique_items': len(items),
            'total_transactions': total_transactions,
            'density_distribution': density_counts,
            'frequent_itemsets': len(itemsets),
            'association_rules': len(rules),
            'encoding_map_size': len(encoding_map),
            'processing_time_seconds': {
                'extraction': extraction_time,
                'encoding': encoding_time_total,
                'fpgrowth': fpgrowth_time_total,
                'rules': rules_time_total,
                'total': time.time() - overall_start
            }
        }
        
        s3_client.put_object(
            Bucket=bucket,
            Key=f"{prefix}/metrics.json",
            Body=json.dumps(metrics, indent=2)
        )
        
        save_time = time.time() - save_start
        logger.info(f"✓ Saved results to {s3_path} in {save_time:.1f}s")
        
        total_time = time.time() - overall_start
        logger.info(f"\n✓ {item_type.upper()} COMPLETE in {total_time:.1f}s")
        logger.info(f"  - {len(items):,} unique items")
        logger.info(f"  - {len(transactions):,} patient transactions")
        logger.info(f"  - {len(itemsets):,} frequent itemsets")
        logger.info(f"  - {len(rules):,} association rules")
        
        return metrics
        
    except Exception as e:
        logger.error(f"✗ Failed to process {item_type}: {e}", exc_info=True)
        return {'item_type': item_type, 'error': str(e)}

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution function."""
    logger = setup_logger()
    
    logger.info("="*80)
    logger.info("GLOBAL FPGROWTH FEATURE IMPORTANCE ANALYSIS")
    logger.info("="*80)
    logger.info(f"Min Support: {MIN_SUPPORT}")
    logger.info(f"Min Confidence: {MIN_CONFIDENCE}")
    logger.info(f"Min Itemset Lift: {MIN_ITEMSET_LIFT} (filtering common/trivial itemsets)")
    logger.info(f"Item Types: {ITEM_TYPES}")
    logger.info(f"S3 Output: {S3_OUTPUT_BASE}")
    logger.info(f"Local Data: {LOCAL_DATA_PATH}")
    logger.info(f"Local Data Exists: {LOCAL_DATA_PATH.exists()}")
    logger.info("="*80)
    
    if not LOCAL_DATA_PATH.exists():
        logger.error(f"✗ Local data path does not exist: {LOCAL_DATA_PATH}")
        logger.error(
            "  On EC2, sync from S3 with:\n"
            "    aws s3 sync s3://pgxdatalake/gold/cohorts_F1120/ /mnt/nvme/cohorts/"
        )
        logger.error(
            "  For local development, sync to ./data/cohorts_F1120/ and "
            "either set LOCAL_DATA_PATH accordingly or export LOCAL_DATA_PATH."
        )
        sys.exit(1)
    
    # Check which item types need processing (skip if already in S3)
    all_metrics = []
    overall_start = time.time()
    skipped = 0
    
    # Helper function to check S3 existence
    def check_s3_results_exist(s3_output_base: str, item_type: str) -> bool:
        """Check if results already exist in S3 (by checking for metrics.json)."""
        s3 = boto3.client('s3')
        key = f"gold/fpgrowth/global/{item_type}/metrics.json"
        try:
            s3.head_object(Bucket='pgxdatalake', Key=key)
            return True
        except:
            return False
    
    items_to_process = []
    for item_type in ITEM_TYPES:
        logger.info(f"Checking {item_type.upper()}...")
        if check_s3_results_exist(S3_OUTPUT_BASE, item_type):
            logger.info(f"  ⏭ Already exists in S3 - SKIPPING")
            skipped += 1
            all_metrics.append({'item_type': item_type, 'status': 'skipped'})
        else:
            logger.info(f"  ▶ Queued for processing")
            items_to_process.append(item_type)
    
    # Process item types SEQUENTIALLY (prevents OOM errors - each job needs ~300-500 GB peak)
    if items_to_process:
        logger.info(f"\n{'='*80}")
        logger.info(f"SEQUENTIAL PROCESSING: {len(items_to_process)} item types")
        logger.info(f"Processing one at a time to avoid memory exhaustion")
        logger.info(f"Expected runtime: 50-85 minutes total")
        logger.info(f"{'='*80}\n")
        
        # Process each item type sequentially
        for idx, item_type in enumerate(items_to_process, 1):
            logger.info(f"\n{'='*80}")
            logger.info(f"Processing {idx}/{len(items_to_process)}: {item_type.upper()}")
            logger.info(f"{'='*80}\n")
            
            try:
                # Use item-specific parameters
                actual_min_support = MIN_SUPPORT_CPT if item_type == 'cpt_code' else MIN_SUPPORT
                actual_min_confidence = MIN_CONFIDENCE_CPT if item_type == 'cpt_code' else MIN_CONFIDENCE
                
                metrics = process_item_type(
                    item_type=item_type,
                    local_data_path=LOCAL_DATA_PATH,
                    s3_output_base=S3_OUTPUT_BASE,
                    min_support=actual_min_support,
                    min_confidence=actual_min_confidence,
                    logger=logger
                )
                all_metrics.append(metrics)
                
                if 'error' not in metrics:
                    logger.info(f"\n✓ {item_type.upper()} COMPLETE:")
                    logger.info(f"  - Frequent itemsets: {metrics['frequent_itemsets']:,}")
                    logger.info(f"  - Association rules: {metrics['association_rules']:,}")
                    logger.info(f"  - TARGET_ICD rules: {metrics.get('rules_by_target', {}).get('TARGET_ICD', 0):,}")
                    logger.info(f"  - TARGET_ED rules: {metrics.get('rules_by_target', {}).get('TARGET_ED', 0):,}")
                    logger.info(f"  - CONTROL rules: {metrics.get('rules_by_target', {}).get('CONTROL', 0):,}")
                    logger.info(f"  - Runtime: {metrics.get('total_time_seconds', 0):.1f}s")
                else:
                    logger.info(f"\n✗ {item_type.upper()} FAILED: {metrics['error']}")
                    
            except Exception as e:
                logger.error(f"\n✗ {item_type.upper()} EXCEPTION: {e}", exc_info=True)
                all_metrics.append({'item_type': item_type, 'error': str(e)})
                
            logger.info(f"\nCompleted {idx}/{len(items_to_process)} item types")
            logger.info("="*80)
    else:
        logger.info("\n⏭ All item types already exist in S3 - nothing to process")
    
    # Final summary
    total_elapsed = time.time() - overall_start
    
    logger.info("\n" + "="*80)
    logger.info("GLOBAL FPGROWTH ANALYSIS - COMPLETE")
    logger.info("="*80)
    logger.info(f"Total time: {total_elapsed:.1f}s ({total_elapsed/60:.1f}min)")
    logger.info(f"Processed: {len([m for m in all_metrics if m.get('status') != 'skipped'])}")
    logger.info(f"Skipped: {skipped}")
    logger.info("\nResults Summary:")
    for m in all_metrics:
        if m.get('status') == 'skipped':
            logger.info(f"  ⏭ {m['item_type']}: SKIPPED (already in S3)")
        elif 'error' not in m:
            logger.info(f"  ✓ {m['item_type']}: {m['frequent_itemsets']:,} itemsets, {m['association_rules']:,} rules")
        else:
            logger.info(f"  ✗ {m['item_type']}: ERROR - {m['error']}")
    logger.info("="*80)
    
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


