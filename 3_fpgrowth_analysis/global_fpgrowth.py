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
import pandas as pd
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

MIN_SUPPORT = 0.01      # Items must appear in 1% of patients
MIN_CONFIDENCE = 0.01   # Rules must have 1% confidence
ITEM_TYPES = ['drug_name', 'icd_code', 'cpt_code']
S3_OUTPUT_BASE = "s3://pgxdatalake/gold/fpgrowth/global"
LOCAL_DATA_PATH = PROJECT_ROOT / "data" / "gold" / "cohorts_F1120"

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
    logger.info(f"Grouping by patient...")
    transactions = (
        df.groupby('mi_person_key')['item']
        .apply(lambda x: sorted(set(x.tolist())))
        .tolist()
    )
    
    elapsed = time.time() - start_time
    logger.info(f"✓ Created {len(transactions):,} patient transactions in {elapsed:.1f}s")
    
    return transactions

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
        
        # Step 2: Create transactions
        transactions = create_global_transactions(local_data_path, item_type, logger)
        
        # Step 3: Encode transactions
        logger.info(f"Encoding {len(transactions):,} transactions...")
        encode_start = time.time()
        te = TransactionEncoder()
        te_ary = te.fit(transactions).transform(transactions)
        df_encoded = pd.DataFrame(te_ary, columns=te.columns_)
        encode_time = time.time() - encode_start
        logger.info(f"✓ Encoded to {df_encoded.shape} matrix in {encode_time:.1f}s")
        
        # Step 4: Run FP-Growth
        logger.info(f"Running FP-Growth (min_support={min_support})...")
        fpgrowth_start = time.time()
        itemsets = fpgrowth(df_encoded, min_support=min_support, use_colnames=True)
        itemsets = itemsets.sort_values('support', ascending=False).reset_index(drop=True)
        fpgrowth_time = time.time() - fpgrowth_start
        logger.info(f"✓ Found {len(itemsets):,} frequent itemsets in {fpgrowth_time:.1f}s")
        
        # Step 5: Generate association rules
        logger.info(f"Generating association rules (min_confidence={min_confidence})...")
        rules_start = time.time()
        rules = association_rules(itemsets, metric="confidence", min_threshold=min_confidence)
        rules = rules.sort_values('lift', ascending=False).reset_index(drop=True)
        rules_time = time.time() - rules_start
        logger.info(f"✓ Generated {len(rules):,} association rules in {rules_time:.1f}s")
        
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
        metrics = {
            'item_type': item_type,
            'min_support': min_support,
            'min_confidence': min_confidence,
            'unique_items': len(items),
            'total_transactions': len(transactions),
            'frequent_itemsets': len(itemsets),
            'association_rules': len(rules),
            'encoding_map_size': len(encoding_map),
            'processing_time_seconds': {
                'extraction': encode_start - overall_start,
                'encoding': encode_time,
                'fpgrowth': fpgrowth_time,
                'rules': rules_time,
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
    logger.info(f"Item Types: {ITEM_TYPES}")
    logger.info(f"S3 Output: {S3_OUTPUT_BASE}")
    logger.info(f"Local Data: {LOCAL_DATA_PATH}")
    logger.info(f"Local Data Exists: {LOCAL_DATA_PATH.exists()}")
    logger.info("="*80)
    
    if not LOCAL_DATA_PATH.exists():
        logger.error(f"✗ Local data path does not exist: {LOCAL_DATA_PATH}")
        logger.error(f"  Run: aws s3 sync s3://pgxdatalake/gold/cohorts_F1120/ data/gold/cohorts_F1120/")
        sys.exit(1)
    
    # Process each item type
    all_metrics = []
    overall_start = time.time()
    
    for item_type in ITEM_TYPES:
        metrics = process_item_type(
            item_type=item_type,
            local_data_path=LOCAL_DATA_PATH,
            s3_output_base=S3_OUTPUT_BASE,
            min_support=MIN_SUPPORT,
            min_confidence=MIN_CONFIDENCE,
            logger=logger
        )
        all_metrics.append(metrics)
    
    # Final summary
    total_time = time.time() - overall_start
    logger.info("\n" + "="*80)
    logger.info("GLOBAL ANALYSIS COMPLETE")
    logger.info("="*80)
    logger.info(f"Total Runtime: {total_time/60:.1f} minutes")
    
    for metrics in all_metrics:
        if 'error' in metrics:
            logger.info(f"  ✗ {metrics['item_type']}: FAILED - {metrics['error']}")
        else:
            logger.info(f"  ✓ {metrics['item_type']}: {metrics['frequent_itemsets']:,} itemsets, {metrics['association_rules']:,} rules")
    
    logger.info("="*80)


if __name__ == "__main__":
    main()


