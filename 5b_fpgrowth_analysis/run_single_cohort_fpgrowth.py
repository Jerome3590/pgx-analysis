#!/usr/bin/env python3
"""
Run FP-Growth analysis for a single cohort/age_band combination.

This script calls process_single_cohort directly for a specific cohort/age_band.
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import cohort_fpgrowth module from 5b_fpgrowth_analysis
import importlib.util
cohort_fpgrowth_path = PROJECT_ROOT / "5b_fpgrowth_analysis" / "cohort_fpgrowth.py"
spec = importlib.util.spec_from_file_location("cohort_fpgrowth", cohort_fpgrowth_path)
cohort_fpgrowth = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cohort_fpgrowth)

process_single_cohort = cohort_fpgrowth.process_single_cohort
MIN_SUPPORT = cohort_fpgrowth.MIN_SUPPORT
MIN_CONFIDENCE = cohort_fpgrowth.MIN_CONFIDENCE
ITEM_TYPES = cohort_fpgrowth.ITEM_TYPES
S3_OUTPUT_BASE = cohort_fpgrowth.S3_OUTPUT_BASE
MODEL_DATA_ROOT = cohort_fpgrowth.MODEL_DATA_ROOT
LOCAL_DATA_PATH = cohort_fpgrowth.LOCAL_DATA_PATH

def main():
    parser = argparse.ArgumentParser(description="Run FP-Growth for a single cohort/age_band")
    parser.add_argument("--cohort-name", required=True, help="Cohort name (e.g., opioid_ed)")
    parser.add_argument("--age-band", required=True, help="Age band (e.g., 0-12)")
    parser.add_argument("--event-year", default="train", help="Event year (train, 2019, etc.)")
    
    args = parser.parse_args()
    
    # Use model_data if available, otherwise use local data path
    local_data_path = MODEL_DATA_ROOT if MODEL_DATA_ROOT.exists() else LOCAL_DATA_PATH
    
    print(f"Running FP-Growth for {args.cohort_name} / {args.age_band} / {args.event_year}")
    print(f"Using data path: {local_data_path}")
    
    # Process each item type
    for item_type in ITEM_TYPES:
        print(f"\nProcessing {item_type}...")
        try:
            result = process_single_cohort(
                item_type=item_type,
                cohort_name=args.cohort_name,
                age_band=args.age_band,
                event_year=args.event_year,
                local_data_path=local_data_path,
                s3_output_base=S3_OUTPUT_BASE,
                min_support=MIN_SUPPORT,
                min_confidence=MIN_CONFIDENCE
            )
            if 'error' in result:
                print(f"[ERROR] {item_type}: {result['error']}")
            else:
                print(f"[OK] {item_type}: {result.get('itemsets_count', 0)} itemsets, {result.get('rules_count', 0)} rules")
        except Exception as e:
            print(f"[ERROR] {item_type} failed: {e}")
            import traceback
            traceback.print_exc()
    
    print("\nFP-Growth itemsets creation complete!")

if __name__ == "__main__":
    main()

