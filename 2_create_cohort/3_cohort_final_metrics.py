#!/usr/bin/env python3
"""
3_cohort_final_metrics.py

Generate final metrics for each cohort from gold S3 parquet files.

For each cohort (opioid_ed, non_opioid_ed):
- Target events count
- Control events count
- Distinct patient count by year (target and control)
- Patient transactions by year (target and control)
- Drug frequency counts by year (target and control)

Uses DuckDB to query S3 parquet files and saves all metrics as CSV files.

Usage:
  python 3_cohort_final_metrics.py
  python 3_cohort_final_metrics.py --output-dir ./metrics
  python 3_cohort_final_metrics.py --target-slug F1120
"""

import os
import sys
import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import re

# Project path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from py_helpers.logging_utils import setup_logging
from py_helpers.duckdb_utils import create_simple_duckdb_connection
from py_helpers.s3_utils import S3_BUCKET, get_cohort_parquet_path
import boto3


def discover_cohort_files(target_slug: str = "F1120", bucket: str = S3_BUCKET) -> Dict[str, List[Dict[str, str]]]:
    """
    Discover all cohort parquet files in S3.
    
    Returns:
        Dict mapping cohort_name -> list of dicts with keys: s3_path, event_year, age_band
    """
    s3_client = boto3.client('s3')
    prefix = f"gold/cohorts_{target_slug}/"
    
    cohort_files = {}
    
    logger = logging.getLogger(__name__)
    logger.info(f"Discovering cohort files in s3://{bucket}/{prefix}")
    
    paginator = s3_client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if not key.endswith("cohort.parquet"):
                continue
            
            # Parse path: gold/cohorts_{target}/cohort_name={cohort}/event_year={year}/age_band={band}/cohort.parquet
            match = re.search(r'cohort_name=([^/]+)/event_year=(\d+)/age_band=([^/]+)/cohort\.parquet', key)
            if match:
                cohort_name = match.group(1)
                event_year = match.group(2)
                age_band = match.group(3)
                
                s3_path = f"s3://{bucket}/{key}"
                
                if cohort_name not in cohort_files:
                    cohort_files[cohort_name] = []
                
                cohort_files[cohort_name].append({
                    "s3_path": s3_path,
                    "event_year": event_year,
                    "age_band": age_band
                })
    
    # Log summary
    for cohort_name, files in cohort_files.items():
        logger.info(f"Found {len(files)} files for cohort: {cohort_name}")
    
    return cohort_files


def calculate_cohort_metrics(conn, s3_path: str, cohort_name: str, event_year: str, age_band: str, logger: logging.Logger) -> Dict:
    """
    Calculate metrics for a single cohort partition.
    
    Returns:
        Dict with metrics including target/control counts, patient counts, transactions, drug frequencies
    """
    logger.info(f"Calculating metrics for {cohort_name} / {event_year} / {age_band}")
    
    # Read the parquet file
    try:
        # First, check if target column exists and what values it has
        target_check = conn.sql(f"""
            SELECT DISTINCT target, is_target_case
            FROM read_parquet('{s3_path}')
            LIMIT 10
        """).fetchall()
        
        logger.debug(f"Target values found: {target_check}")
        
        # Calculate target and control event counts
        event_counts = conn.sql(f"""
            SELECT 
                COUNT(*) FILTER (WHERE is_target_case = true) as target_events,
                COUNT(*) FILTER (WHERE is_target_case = false) as control_events,
                COUNT(DISTINCT mi_person_key) FILTER (WHERE is_target_case = true) as target_patients,
                COUNT(DISTINCT mi_person_key) FILTER (WHERE is_target_case = false) as control_patients
            FROM read_parquet('{s3_path}')
        """).fetchone()
        
        target_events, control_events, target_patients, control_patients = event_counts
        
        # Calculate transactions by year (using event_date)
        transactions_by_year = conn.sql(f"""
            SELECT 
                EXTRACT(YEAR FROM event_date) as year,
                COUNT(*) FILTER (WHERE is_target_case = true) as target_transactions,
                COUNT(*) FILTER (WHERE is_target_case = false) as control_transactions,
                COUNT(DISTINCT mi_person_key) FILTER (WHERE is_target_case = true) as target_patients,
                COUNT(DISTINCT mi_person_key) FILTER (WHERE is_target_case = false) as control_patients
            FROM read_parquet('{s3_path}')
            GROUP BY EXTRACT(YEAR FROM event_date)
            ORDER BY year
        """).fetchall()
        
        # Calculate drug frequency counts by year (target and control separately)
        drug_freq_target = conn.sql(f"""
            SELECT 
                EXTRACT(YEAR FROM event_date) as year,
                drug_name,
                COUNT(*) as frequency
            FROM read_parquet('{s3_path}')
            WHERE is_target_case = true 
                AND drug_name IS NOT NULL 
                AND drug_name != ''
            GROUP BY EXTRACT(YEAR FROM event_date), drug_name
            ORDER BY year, frequency DESC
        """).fetchall()
        
        drug_freq_control = conn.sql(f"""
            SELECT 
                EXTRACT(YEAR FROM event_date) as year,
                drug_name,
                COUNT(*) as frequency
            FROM read_parquet('{s3_path}')
            WHERE is_target_case = false 
                AND drug_name IS NOT NULL 
                AND drug_name != ''
            GROUP BY EXTRACT(YEAR FROM event_date), drug_name
            ORDER BY year, frequency DESC
        """).fetchall()
        
        return {
            "cohort_name": cohort_name,
            "event_year": event_year,
            "age_band": age_band,
            "target_events": target_events,
            "control_events": control_events,
            "target_patients": target_patients,
            "control_patients": control_patients,
            "transactions_by_year": transactions_by_year,
            "drug_freq_target": drug_freq_target,
            "drug_freq_control": drug_freq_control
        }
        
    except Exception as e:
        logger.error(f"Error calculating metrics for {s3_path}: {e}")
        raise


def aggregate_metrics_by_cohort(all_metrics: List[Dict], cohort_name: str) -> Dict:
    """
    Aggregate metrics across all partitions for a cohort.
    """
    # Aggregate totals
    total_target_events = sum(m["target_events"] for m in all_metrics)
    total_control_events = sum(m["control_events"] for m in all_metrics)
    
    # Aggregate by year
    year_data = {}
    
    for metrics in all_metrics:
        # Aggregate transactions by year
        for year, target_txns, control_txns, target_pats, control_pats in metrics["transactions_by_year"]:
            year = int(year)
            if year not in year_data:
                year_data[year] = {
                    "target_transactions": 0,
                    "control_transactions": 0,
                    "target_patients": set(),
                    "control_patients": set(),
                    "drug_freq_target": {},
                    "drug_freq_control": {}
                }
            
            year_data[year]["target_transactions"] += target_txns
            year_data[year]["control_transactions"] += control_txns
            
            # Note: We can't get distinct patients across partitions without re-reading all data
            # So we'll aggregate transaction counts but note patient counts are approximate
        
        # Aggregate drug frequencies by year
        for year, drug_name, freq in metrics["drug_freq_target"]:
            year = int(year)
            if year not in year_data:
                year_data[year] = {
                    "target_transactions": 0,
                    "control_transactions": 0,
                    "target_patients": set(),
                    "control_patients": set(),
                    "drug_freq_target": {},
                    "drug_freq_control": {}
                }
            
            if drug_name not in year_data[year]["drug_freq_target"]:
                year_data[year]["drug_freq_target"][drug_name] = 0
            year_data[year]["drug_freq_target"][drug_name] += freq
        
        for year, drug_name, freq in metrics["drug_freq_control"]:
            year = int(year)
            if year not in year_data:
                year_data[year] = {
                    "target_transactions": 0,
                    "control_transactions": 0,
                    "target_patients": set(),
                    "control_patients": set(),
                    "drug_freq_target": {},
                    "drug_freq_control": {}
                }
            
            if drug_name not in year_data[year]["drug_freq_control"]:
                year_data[year]["drug_freq_control"][drug_name] = 0
            year_data[year]["drug_freq_control"][drug_name] += freq
    
    return {
        "cohort_name": cohort_name,
        "total_target_events": total_target_events,
        "total_control_events": total_control_events,
        "year_data": year_data
    }


def save_metrics_to_csv(aggregated_metrics: Dict, output_dir: Path, logger: logging.Logger):
    """
    Save all metrics to CSV files.
    """
    cohort_name = aggregated_metrics["cohort_name"]
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Save event counts summary
    summary_path = output_dir / f"{cohort_name}_event_counts.csv"
    with open(summary_path, 'w') as f:
        f.write("cohort_name,target_events,control_events\n")
        f.write(f"{cohort_name},{aggregated_metrics['total_target_events']},{aggregated_metrics['total_control_events']}\n")
    logger.info(f"Saved event counts to {summary_path}")
    
    # 2. Save patient counts and transactions by year
    year_metrics_path = output_dir / f"{cohort_name}_yearly_metrics.csv"
    with open(year_metrics_path, 'w') as f:
        f.write("cohort_name,year,target_patients,control_patients,target_transactions,control_transactions\n")
        for year in sorted(aggregated_metrics["year_data"].keys()):
            year_info = aggregated_metrics["year_data"][year]
            target_pats = year_info.get("target_patients", 0)
            control_pats = year_info.get("control_patients", 0)
            f.write(f"{cohort_name},{year},{target_pats},{control_pats},{year_info['target_transactions']},{year_info['control_transactions']}\n")
    logger.info(f"Saved yearly metrics to {year_metrics_path}")
    
    # 3. Save drug frequency counts by year (target)
    drug_freq_target_path = output_dir / f"{cohort_name}_drug_frequency_target_by_year.csv"
    with open(drug_freq_target_path, 'w') as f:
        f.write("cohort_name,year,drug_name,frequency\n")
        for year in sorted(aggregated_metrics["year_data"].keys()):
            year_info = aggregated_metrics["year_data"][year]
            for drug_name, freq in sorted(year_info["drug_freq_target"].items(), key=lambda x: x[1], reverse=True):
                f.write(f"{cohort_name},{year},{drug_name},{freq}\n")
    logger.info(f"Saved target drug frequencies to {drug_freq_target_path}")
    
    # 4. Save drug frequency counts by year (control)
    drug_freq_control_path = output_dir / f"{cohort_name}_drug_frequency_control_by_year.csv"
    with open(drug_freq_control_path, 'w') as f:
        f.write("cohort_name,year,drug_name,frequency\n")
        for year in sorted(aggregated_metrics["year_data"].keys()):
            year_info = aggregated_metrics["year_data"][year]
            for drug_name, freq in sorted(year_info["drug_freq_control"].items(), key=lambda x: x[1], reverse=True):
                f.write(f"{cohort_name},{year},{drug_name},{freq}\n")
    logger.info(f"Saved control drug frequencies to {drug_freq_control_path}")


def calculate_distinct_patients_by_year(conn, cohort_files: List[Dict], cohort_name: str, logger: logging.Logger) -> List[Tuple]:
    """
    Calculate distinct patient counts by year across all partitions.
    Uses DuckDB's ability to read multiple parquet files efficiently.
    """
    logger.info(f"Calculating distinct patient counts by year for {cohort_name}")
    
    # Build list of all S3 paths
    s3_paths = [f["s3_path"] for f in cohort_files]
    
    if not s3_paths:
        return []
    
    # Use DuckDB's glob pattern or read_parquet with multiple files
    # For better performance, we'll use UNION ALL but process in chunks if needed
    if len(s3_paths) == 1:
        # Single file - simple query
        query = f"""
            SELECT 
                EXTRACT(YEAR FROM event_date) as year,
                COUNT(DISTINCT mi_person_key) FILTER (WHERE is_target_case = true) as target_patients,
                COUNT(DISTINCT mi_person_key) FILTER (WHERE is_target_case = false) as control_patients
            FROM read_parquet('{s3_paths[0]}')
            GROUP BY year
            ORDER BY year
        """
    else:
        # Multiple files - use UNION ALL
        # DuckDB can handle this efficiently
        union_parts = []
        for path in s3_paths:
            union_parts.append(f"SELECT EXTRACT(YEAR FROM event_date) as year, mi_person_key, is_target_case FROM read_parquet('{path}')")
        
        union_query = " UNION ALL ".join(union_parts)
        
        query = f"""
            SELECT 
                year,
                COUNT(DISTINCT mi_person_key) FILTER (WHERE is_target_case = true) as target_patients,
                COUNT(DISTINCT mi_person_key) FILTER (WHERE is_target_case = false) as control_patients
            FROM (
                {union_query}
            )
            GROUP BY year
            ORDER BY year
        """
    
    try:
        distinct_patients = conn.sql(query).fetchall()
        return distinct_patients
    except Exception as e:
        logger.error(f"Error calculating distinct patients: {e}")
        # Fallback: return empty list
        return []


def main():
    parser = argparse.ArgumentParser(description="Generate final metrics for cohorts")
    parser.add_argument("--output-dir", type=str, default="./cohort_metrics", help="Output directory for CSV files")
    parser.add_argument("--target-slug", type=str, default="F1120", help="Target slug for S3 path (default: F1120)")
    parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    
    args = parser.parse_args()
    
    # Setup logging
    logger, _ = setup_logging("cohort_final_metrics", "all", "all")
    logger.setLevel(getattr(logging, args.log_level.upper()))
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 80)
    logger.info("Cohort Final Metrics Generation")
    logger.info("=" * 80)
    logger.info(f"Target slug: {args.target_slug}")
    logger.info(f"Output directory: {output_dir}")
    
    # Discover cohort files
    cohort_files_dict = discover_cohort_files(target_slug=args.target_slug, bucket=S3_BUCKET)
    
    if not cohort_files_dict:
        logger.warning("No cohort files found. Check S3 bucket and target slug.")
        return
    
    # Create DuckDB connection
    conn = create_simple_duckdb_connection(logger)
    
    try:
        # Process each cohort
        for cohort_name, files in cohort_files_dict.items():
            logger.info("=" * 80)
            logger.info(f"Processing cohort: {cohort_name}")
            logger.info(f"Found {len(files)} partitions")
            logger.info("=" * 80)
            
            # Calculate metrics for each partition
            all_metrics = []
            for file_info in files:
                try:
                    metrics = calculate_cohort_metrics(
                        conn, 
                        file_info["s3_path"],
                        cohort_name,
                        file_info["event_year"],
                        file_info["age_band"],
                        logger
                    )
                    all_metrics.append(metrics)
                except Exception as e:
                    logger.error(f"Failed to process {file_info['s3_path']}: {e}")
                    continue
            
            if not all_metrics:
                logger.warning(f"No metrics calculated for {cohort_name}")
                continue
            
            # Aggregate metrics
            aggregated = aggregate_metrics_by_cohort(all_metrics, cohort_name)
            
            # Calculate distinct patients by year (requires reading all data)
            logger.info(f"Calculating distinct patient counts by year for {cohort_name}...")
            distinct_patients = calculate_distinct_patients_by_year(conn, files, cohort_name, logger)
            
            # Add distinct patient counts to year_data
            for year, target_pats, control_pats in distinct_patients:
                year = int(year)
                if year not in aggregated["year_data"]:
                    aggregated["year_data"][year] = {
                        "target_transactions": 0,
                        "control_transactions": 0,
                        "target_patients": 0,
                        "control_patients": 0,
                        "drug_freq_target": {},
                        "drug_freq_control": {}
                    }
                aggregated["year_data"][year]["target_patients"] = target_pats
                aggregated["year_data"][year]["control_patients"] = control_pats
            
            # Save distinct patient counts by year
            patient_counts_path = output_dir / f"{cohort_name}_distinct_patients_by_year.csv"
            with open(patient_counts_path, 'w') as f:
                f.write("cohort_name,year,target_patients,control_patients\n")
                for year, target_pats, control_pats in distinct_patients:
                    f.write(f"{cohort_name},{year},{target_pats},{control_pats}\n")
            logger.info(f"Saved distinct patient counts to {patient_counts_path}")
            
            # Save all other metrics
            save_metrics_to_csv(aggregated, output_dir, logger)
            
            logger.info(f"[OK] Completed processing {cohort_name}")
        
        logger.info("=" * 80)
        logger.info("All cohorts processed successfully!")
        logger.info(f"Results saved to: {output_dir}")
        logger.info("=" * 80)
        
    finally:
        conn.close()


if __name__ == "__main__":
    main()

