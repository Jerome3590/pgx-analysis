#!/usr/bin/env python3
"""
step1_pre_cohort_audit.py

Pre-cohort creation audit and QA script.
Calculates target averages (F1120 and HCG ED visits) across all partitions
and saves results to S3 for use in control-only cohort creation.

This should run BEFORE cohort creation to ensure target averages are available.

Usage:
  python step1_pre_cohort_audit.py --profile bedrock --save-results
  python step1_pre_cohort_audit.py --all-partitions --save-results
"""

import os
import sys
import argparse
import json
import logging
from datetime import datetime
from collections import defaultdict
from typing import Dict, Any, Optional

# Project path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from helpers_1997_13.logging_utils import setup_logging, save_logs_to_s3
from helpers_1997_13.duckdb_utils import get_duckdb_connection
from helpers_1997_13.constants import S3_BUCKET, AGE_BANDS, EVENT_YEARS
from helpers_1997_13.s3_utils import save_to_s3_json
import boto3


def build_audit_output_path(bucket: str = S3_BUCKET) -> str:
    """Build S3 path for pre-cohort audit results."""
    return f"s3://{bucket}/gold/qa_results/pre_cohort_audit/target_averages.json"


def calculate_target_averages(
    conn,
    logger: logging.Logger,
    aws_profile: Optional[str] = None
) -> Dict[str, Any]:
    """Calculate F1120 and HCG ED visit target averages across all partitions."""
    logger.info("=" * 80)
    logger.info("PRE-COHORT AUDIT: Calculating Target Averages")
    logger.info("=" * 80)
    
    results = {
        'f1120': defaultdict(lambda: defaultdict(int)),
        'hcg_ed': defaultdict(lambda: defaultdict(int)),
    }
    
    ed_hcg_lines = [
        "P51 - ER Visits and Observation Care",
        "O11 - Emergency Room",
        "P33 - Urgent Care Visits"
    ]
    
    try:
        logger.info("Querying all partitions efficiently using glob pattern...")
        
        # Use glob pattern to query all partitions at once
        glob_pattern = f"s3://{S3_BUCKET}/gold/medical/age_band=*/event_year=*/medical_data.parquet"
        
        # Single query for F1120 counts per partition
        logger.info("  Calculating F1120 targets...")
        f1120_query = f"""
        SELECT 
            age_band,
            event_year,
            COUNT(DISTINCT mi_person_key) as distinct_patients
        FROM read_parquet('{glob_pattern}')
        WHERE primary_icd_diagnosis_code = 'F1120'
        GROUP BY age_band, event_year
        ORDER BY age_band, event_year
        """
        
        f1120_results = conn.sql(f1120_query).fetchall()
        for age_band, event_year, count in f1120_results:
            results['f1120'][age_band][int(event_year)] = count
        
        # Single query for HCG ED visit counts per partition
        logger.info("  Calculating HCG ED visit targets...")
        ed_query = f"""
        SELECT 
            age_band,
            event_year,
            COUNT(DISTINCT mi_person_key) as distinct_patients
        FROM read_parquet('{glob_pattern}')
        WHERE hcg_line IN {tuple(ed_hcg_lines)}
        GROUP BY age_band, event_year
        ORDER BY age_band, event_year
        """
        
        ed_results = conn.sql(ed_query).fetchall()
        for age_band, event_year, count in ed_results:
            results['hcg_ed'][age_band][int(event_year)] = count
        
        # Fill in zeros for missing partitions
        for age_band in AGE_BANDS:
            for event_year in EVENT_YEARS:
                if age_band not in results['f1120'] or event_year not in results['f1120'][age_band]:
                    results['f1120'][age_band][event_year] = 0
                if age_band not in results['hcg_ed'] or event_year not in results['hcg_ed'][age_band]:
                    results['hcg_ed'][age_band][event_year] = 0
        
        # Calculate statistics
        logger.info("Calculating statistics...")
        
        # F1120 stats
        f1120_all_counts = [results['f1120'][ab][y] for ab in AGE_BANDS for y in EVENT_YEARS]
        f1120_non_zero = [c for c in f1120_all_counts if c > 0]
        
        f1120_stats = {
            'total_partitions': len(f1120_all_counts),
            'partitions_with_targets': len(f1120_non_zero),
            'partitions_with_zero': len(f1120_all_counts) - len(f1120_non_zero),
            'average': float(sum(f1120_non_zero) / len(f1120_non_zero)) if f1120_non_zero else 0.0,
            'median': float(sorted(f1120_non_zero)[len(f1120_non_zero)//2]) if f1120_non_zero else 0.0,
            'min': int(min(f1120_non_zero)) if f1120_non_zero else 0,
            'max': int(max(f1120_non_zero)) if f1120_non_zero else 0,
        }
        
        # HCG ED stats
        hcg_all_counts = [results['hcg_ed'][ab][y] for ab in AGE_BANDS for y in EVENT_YEARS]
        hcg_non_zero = [c for c in hcg_all_counts if c > 0]
        
        hcg_stats = {
            'total_partitions': len(hcg_all_counts),
            'partitions_with_targets': len(hcg_non_zero),
            'partitions_with_zero': len(hcg_all_counts) - len(hcg_non_zero),
            'average': float(sum(hcg_non_zero) / len(hcg_non_zero)) if hcg_non_zero else 0.0,
            'median': float(sorted(hcg_non_zero)[len(hcg_non_zero)//2]) if hcg_non_zero else 0.0,
            'min': int(min(hcg_non_zero)) if hcg_non_zero else 0,
            'max': int(max(hcg_non_zero)) if hcg_non_zero else 0,
        }
        
        # Combined stats
        combined_counts = [results['f1120'][ab][y] + results['hcg_ed'][ab][y] for ab in AGE_BANDS for y in EVENT_YEARS]
        combined_non_zero = [c for c in combined_counts if c > 0]
        
        combined_stats = {
            'total_partitions': len(combined_counts),
            'partitions_with_targets': len(combined_non_zero),
            'partitions_with_zero': len(combined_counts) - len(combined_non_zero),
            'average': float(sum(combined_non_zero) / len(combined_non_zero)) if combined_non_zero else 0.0,
            'median': float(sorted(combined_non_zero)[len(combined_non_zero)//2]) if combined_non_zero else 0.0,
            'min': int(min(combined_non_zero)) if combined_non_zero else 0,
            'max': int(max(combined_non_zero)) if combined_non_zero else 0,
        }
        
        # Compile audit report
        audit_report = {
            'averages': {
                'f1120': f1120_stats,
                'hcg_ed': hcg_stats,
                'combined': combined_stats,
            },
            'per_partition': {
                'f1120': {ab: {str(y): results['f1120'][ab][y] for y in EVENT_YEARS} for ab in AGE_BANDS},
                'hcg_ed': {ab: {str(y): results['hcg_ed'][ab][y] for y in EVENT_YEARS} for ab in AGE_BANDS},
            },
            'metadata': {
                'computed_at': datetime.now().isoformat(),
                'aws_profile': aws_profile or 'default',
                's3_bucket': S3_BUCKET,
                'age_bands': AGE_BANDS,
                'event_years': EVENT_YEARS,
            }
        }
        
        # Print summary
        logger.info("\n" + "=" * 80)
        logger.info("TARGET AVERAGE SUMMARY")
        logger.info("=" * 80)
        logger.info(f"\nF1120 Targets:")
        logger.info(f"  Average (non-zero): {f1120_stats['average']:,.0f}")
        logger.info(f"  Median (non-zero): {f1120_stats['median']:,.0f}")
        logger.info(f"  Partitions with targets: {f1120_stats['partitions_with_targets']}")
        logger.info(f"  Partitions with zero: {f1120_stats['partitions_with_zero']}")
        
        logger.info(f"\nHCG ED Visit Targets:")
        logger.info(f"  Average (non-zero): {hcg_stats['average']:,.0f}")
        logger.info(f"  Median (non-zero): {hcg_stats['median']:,.0f}")
        logger.info(f"  Partitions with targets: {hcg_stats['partitions_with_targets']}")
        logger.info(f"  Partitions with zero: {hcg_stats['partitions_with_zero']}")
        
        logger.info(f"\nCombined (F1120 + HCG ED) Targets:")
        logger.info(f"  Average (non-zero): {combined_stats['average']:,.0f}")
        logger.info(f"  Median (non-zero): {combined_stats['median']:,.0f}")
        logger.info(f"  Partitions with targets: {combined_stats['partitions_with_targets']}")
        logger.info(f"  Partitions with zero: {combined_stats['partitions_with_zero']}")
        
        return audit_report
        
    except Exception as e:
        logger.error(f"Error calculating target averages: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise


def save_local_config(audit_report: Dict[str, Any], logger: logging.Logger) -> str:
    """Save config file locally for Phase 3 to use."""
    config_file = os.path.join(project_root, 'cohort_target_averages.json')
    try:
        with open(config_file, 'w') as f:
            json.dump(audit_report, f, indent=2)
        logger.info(f"✓ Local config saved: {config_file}")
        return config_file
    except Exception as e:
        logger.warning(f"Could not save local config: {e}")
        return None


def main():
    ap = argparse.ArgumentParser(description="Pre-cohort audit: Calculate target averages")
    ap.add_argument("--profile", help="AWS profile to use (e.g., bedrock)")
    ap.add_argument("--save-results", action="store_true", help="Save audit results to S3")
    ap.add_argument("--save-local", action="store_true", default=True, help="Save local config file (default: True)")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()
    
    # Set AWS profile if provided
    aws_profile = args.profile or os.environ.get('AWS_PROFILE', 'default')
    if args.profile:
        os.environ['AWS_PROFILE'] = args.profile
    
    logger, log_buffer = setup_logging("pre_cohort_audit", "all", "all")
    logger.setLevel(getattr(logging, args.log_level.upper(), logging.INFO))
    
    logger.info(f"Using AWS profile: {aws_profile}")
    
    try:
        conn = get_duckdb_connection()
        
        # Calculate target averages
        audit_report = calculate_target_averages(conn, logger, aws_profile)
        
        # Save local config file (for Phase 3 to load)
        if args.save_local:
            save_local_config(audit_report, logger)
        
        # Save to S3
        if args.save_results:
            s3_path = build_audit_output_path()
            try:
                save_to_s3_json(audit_report, s3_path, logger)
                logger.info(f"✓ Audit report saved to S3: {s3_path}")
            except Exception as e:
                logger.error(f"Failed to save audit report to S3: {e}")
                raise
        
        # Save logs to S3
        try:
            save_logs_to_s3(log_buffer, "pre_cohort_audit", "all", "all")
        except Exception:
            pass
        
        logger.info("\n" + "=" * 80)
        logger.info("PRE-COHORT AUDIT COMPLETED")
        logger.info("=" * 80)
        logger.info(f"\nUse average for control-only cohorts: {audit_report['averages']['combined']['average']:,.0f}")
        logger.info(f"Phase 3 will automatically load this config file.")
        
        conn.close()
        
    except Exception as e:
        logger.error(f"Pre-cohort audit failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == '__main__':
    main()

