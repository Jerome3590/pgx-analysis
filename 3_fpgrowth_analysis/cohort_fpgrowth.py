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
from typing import Dict
from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd
import boto3
import psutil
from mlxtend.frequent_patterns import fpgrowth, association_rules
from mlxtend.preprocessing import TransactionEncoder

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from helpers_1997_13.duckdb_utils import get_duckdb_connection

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
COHORTS_TO_PROCESS = ['opioid_ed', 'ed_non_opioid']  # Specify cohorts to process

ITEM_TYPES = ['drug_name', 'icd_code', 'cpt_code']
S3_OUTPUT_BASE = "s3://pgxdatalake/gold/fpgrowth/cohort"
LOCAL_DATA_PATH = Path("/mnt/nvme/cohorts")  # Instance storage (NVMe SSD for fast I/O)

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
        # Get DuckDB connection
        con = get_duckdb_connection(logger=logger)
        
        # Build path to cohort parquet file
        parquet_file = local_data_path / f"cohort_name={cohort_name}" / f"event_year={event_year}" / f"age_band={age_band}" / "cohort.parquet"
        
        if not parquet_file.exists():
            logger.warning(f"✗ Cohort file not found: {parquet_file}")
            return {
                'item_type': item_type,
                'cohort_name': cohort_name,
                'age_band': age_band,
                'event_year': event_year,
                'error': 'File not found'
            }
        
        # Build query based on item type
        if item_type == 'drug_name':
            query = f"""
            SELECT mi_person_key, drug_name as item
            FROM read_parquet('{parquet_file}')
            WHERE drug_name IS NOT NULL AND drug_name != '' AND event_type = 'pharmacy'
            """
        elif item_type == 'icd_code':
            # Collect from first 5 ICD diagnosis columns
            query = f"""
            WITH all_icds AS (
                SELECT mi_person_key, primary_icd_diagnosis_code as icd 
                FROM read_parquet('{parquet_file}') 
                WHERE primary_icd_diagnosis_code IS NOT NULL AND event_type = 'medical'
                UNION ALL
                SELECT mi_person_key, two_icd_diagnosis_code as icd 
                FROM read_parquet('{parquet_file}') 
                WHERE two_icd_diagnosis_code IS NOT NULL AND event_type = 'medical'
                UNION ALL
                SELECT mi_person_key, three_icd_diagnosis_code as icd 
                FROM read_parquet('{parquet_file}') 
                WHERE three_icd_diagnosis_code IS NOT NULL AND event_type = 'medical'
                UNION ALL
                SELECT mi_person_key, four_icd_diagnosis_code as icd 
                FROM read_parquet('{parquet_file}') 
                WHERE four_icd_diagnosis_code IS NOT NULL AND event_type = 'medical'
                UNION ALL
                SELECT mi_person_key, five_icd_diagnosis_code as icd 
                FROM read_parquet('{parquet_file}') 
                WHERE five_icd_diagnosis_code IS NOT NULL AND event_type = 'medical'
            )
            SELECT mi_person_key, icd as item FROM all_icds WHERE icd != ''
            """
        elif item_type == 'cpt_code':
            query = f"""
            SELECT mi_person_key, procedure_code as item
            FROM read_parquet('{parquet_file}')
            WHERE procedure_code IS NOT NULL AND procedure_code != '' AND event_type = 'medical'
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
        
        # Create transactions
        transactions = (
            df.groupby('mi_person_key')['item']
            .apply(lambda x: sorted(set(x.tolist())))
            .tolist()
        )
        
        if len(transactions) < 10:
            logger.warning(f"✗ Insufficient transactions ({len(transactions)}) for {cohort_id}")
            return {
                'item_type': item_type,
                'cohort_name': cohort_name,
                'age_band': age_band,
                'event_year': event_year,
                'error': f'Insufficient transactions: {len(transactions)}'
            }
        
        # Encode transactions
        te = TransactionEncoder()
        te_ary = te.fit(transactions).transform(transactions)
        df_encoded = pd.DataFrame(te_ary, columns=te.columns_)
        log_memory(logger, "After encoding")
        
        # Run FP-Growth
        itemsets = fpgrowth(df_encoded, min_support=min_support, use_colnames=True)
        itemsets = itemsets.sort_values('support', ascending=False).reset_index(drop=True)
        log_memory(logger, "After FP-Growth")
        
        if len(itemsets) == 0:
            logger.warning(f"✗ No frequent itemsets for {cohort_id}")
            return {
                'item_type': item_type,
                'cohort_name': cohort_name,
                'age_band': age_band,
                'event_year': event_year,
                'error': 'No frequent itemsets'
            }
        
        # Generate association rules
        rules = association_rules(itemsets, metric="confidence", min_threshold=min_confidence)
        rules = rules.sort_values('lift', ascending=False).reset_index(drop=True)
        log_memory(logger, "After rule generation")
        
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
            'total_transactions': len(transactions),
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
    logger.info(f"Max Workers: {MAX_WORKERS}")
    logger.info(f"Item Types: {ITEM_TYPES}")
    logger.info(f"S3 Output: {S3_OUTPUT_BASE}")
    logger.info(f"Local Data: {LOCAL_DATA_PATH}")
    logger.info(f"Local Data Exists: {LOCAL_DATA_PATH.exists()}")
    logger.info("="*80)
    
    if not LOCAL_DATA_PATH.exists():
        logger.error(f"✗ Local data path does not exist: {LOCAL_DATA_PATH}")
        logger.error(f"  Run: aws s3 sync s3://pgxdatalake/gold/cohorts_F1120/ data/gold/cohorts_F1120/")
        sys.exit(1)
    
    # Generate all cohort combinations
    cohort_jobs = []
    for item_type in ITEM_TYPES:
        for cohort_name in COHORT_NAMES:
            for age_band in AGE_BANDS:
                for event_year in EVENT_YEARS:
                    cohort_jobs.append((item_type, cohort_name, age_band, event_year))
    
    total_jobs = len(cohort_jobs)
    logger.info(f"Total cohort jobs: {total_jobs}")
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


