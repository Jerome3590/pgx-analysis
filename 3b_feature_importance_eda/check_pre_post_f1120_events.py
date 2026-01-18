#!/usr/bin/env python3
"""
Check Pre/Post F1120 Events in model_events.parquet

This script verifies that we have both pre-F1120 and post-F1120 events
in the model_events.parquet file for target patients.
"""

import sys
from pathlib import Path
import duckdb
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from py_helpers.constants import age_band_to_fname

def check_pre_post_f1120_events(cohort: str, age_band: str):
    """Check if we have pre and post F1120 events in model_events.parquet"""
    
    age_band_fname = age_band_to_fname(age_band)
    
    # Find model_events.parquet file
    model_data_paths = [
        PROJECT_ROOT / "4a_model_data" / f"cohort_name={cohort}" / f"age_band={age_band}" / "model_events.parquet",
        PROJECT_ROOT / "model_data" / f"cohort_name={cohort}" / f"age_band={age_band}" / "model_events.parquet",
    ]
    
    model_data_path = None
    for path in model_data_paths:
        if path.exists():
            model_data_path = path
            break
    
    if not model_data_path:
        print(f"❌ Model events file not found. Checked:")
        for path in model_data_paths:
            print(f"  - {path}")
        return
    
    print(f"Checking pre/post F1120 events in: {model_data_path}")
    print(f"{'='*80}")
    
    con = duckdb.connect()
    
    # First, check if we have F1120 events at all
    query1 = f"""
    SELECT 
        COUNT(*) as total_events,
        COUNT(DISTINCT mi_person_key) as unique_patients,
        SUM(CASE WHEN target = 1 THEN 1 ELSE 0 END) as target_events,
        SUM(CASE WHEN target = 0 THEN 1 ELSE 0 END) as control_events
    FROM read_parquet('{model_data_path}')
    """
    
    summary = con.execute(query1).df()
    print(f"\nOverall Summary:")
    print(f"   Total events: {summary['total_events'].iloc[0]:,}")
    print(f"   Unique patients: {summary['unique_patients'].iloc[0]:,}")
    print(f"   Target events: {summary['target_events'].iloc[0]:,}")
    print(f"   Control events: {summary['control_events'].iloc[0]:,}")
    
    # Check for F1120 events
    query2 = f"""
    SELECT 
        COUNT(*) as f1120_events,
        COUNT(DISTINCT mi_person_key) as f1120_patients
    FROM read_parquet('{model_data_path}')
    WHERE target = 1
      AND (
        primary_icd_diagnosis_code LIKE '%F1120%'
        OR two_icd_diagnosis_code LIKE '%F1120%'
        OR three_icd_diagnosis_code LIKE '%F1120%'
        OR four_icd_diagnosis_code LIKE '%F1120%'
        OR five_icd_diagnosis_code LIKE '%F1120%'
        OR six_icd_diagnosis_code LIKE '%F1120%'
        OR seven_icd_diagnosis_code LIKE '%F1120%'
        OR eight_icd_diagnosis_code LIKE '%F1120%'
        OR nine_icd_diagnosis_code LIKE '%F1120%'
        OR ten_icd_diagnosis_code LIKE '%F1120%'
      )
    """
    
    f1120_summary = con.execute(query2).df()
    print(f"\nF1120 Events:")
    print(f"   Total F1120 events: {f1120_summary['f1120_events'].iloc[0]:,}")
    print(f"   Patients with F1120: {f1120_summary['f1120_patients'].iloc[0]:,}")
    
    # Now check pre vs post F1120 events
    query3 = f"""
    WITH target_patients_f1120 AS (
        SELECT DISTINCT
            mi_person_key,
            MIN(CAST(event_date AS DATE)) as first_f1120_date
        FROM read_parquet('{model_data_path}')
        WHERE target = 1
          AND (
            primary_icd_diagnosis_code LIKE '%F1120%'
            OR two_icd_diagnosis_code LIKE '%F1120%'
            OR three_icd_diagnosis_code LIKE '%F1120%'
            OR four_icd_diagnosis_code LIKE '%F1120%'
            OR five_icd_diagnosis_code LIKE '%F1120%'
            OR six_icd_diagnosis_code LIKE '%F1120%'
            OR seven_icd_diagnosis_code LIKE '%F1120%'
            OR eight_icd_diagnosis_code LIKE '%F1120%'
            OR nine_icd_diagnosis_code LIKE '%F1120%'
            OR ten_icd_diagnosis_code LIKE '%F1120%'
          )
        GROUP BY mi_person_key
    ),
    events_with_f1120_date AS (
        SELECT 
            e.mi_person_key,
            e.event_date,
            e.target,
            t.first_f1120_date,
            CASE 
                WHEN e.target = 1 AND t.first_f1120_date IS NOT NULL 
                     AND CAST(e.event_date AS DATE) < t.first_f1120_date 
                THEN 'pre'
                WHEN e.target = 1 AND t.first_f1120_date IS NOT NULL 
                     AND CAST(e.event_date AS DATE) >= t.first_f1120_date 
                THEN 'post'
                WHEN e.target = 1 AND t.first_f1120_date IS NULL
                THEN 'no_f1120'
                ELSE 'control'
            END as timing
        FROM read_parquet('{model_data_path}') e
        LEFT JOIN target_patients_f1120 t ON e.mi_person_key = t.mi_person_key
        WHERE e.target = 1
    )
    SELECT 
        timing,
        COUNT(*) as event_count,
        COUNT(DISTINCT mi_person_key) as patient_count
    FROM events_with_f1120_date
    GROUP BY timing
    ORDER BY timing
    """
    
    timing_summary = con.execute(query3).df()
    print(f"\nPre/Post F1120 Event Distribution (Target Patients Only):")
    print(f"{'='*80}")
    for _, row in timing_summary.iterrows():
        print(f"   {row['timing']:12s}: {row['event_count']:>10,} events in {row['patient_count']:>5,} patients")
    
    # Check sample events
    query4 = f"""
    WITH target_patients_f1120 AS (
        SELECT DISTINCT
            mi_person_key,
            MIN(CAST(event_date AS DATE)) as first_f1120_date
        FROM read_parquet('{model_data_path}')
        WHERE target = 1
          AND (
            primary_icd_diagnosis_code LIKE '%F1120%'
            OR two_icd_diagnosis_code LIKE '%F1120%'
            OR three_icd_diagnosis_code LIKE '%F1120%'
            OR four_icd_diagnosis_code LIKE '%F1120%'
            OR five_icd_diagnosis_code LIKE '%F1120%'
            OR six_icd_diagnosis_code LIKE '%F1120%'
            OR seven_icd_diagnosis_code LIKE '%F1120%'
            OR eight_icd_diagnosis_code LIKE '%F1120%'
            OR nine_icd_diagnosis_code LIKE '%F1120%'
            OR ten_icd_diagnosis_code LIKE '%F1120%'
          )
        GROUP BY mi_person_key
        LIMIT 5
    ),
    sample_patient AS (
        SELECT mi_person_key, first_f1120_date
        FROM target_patients_f1120
        LIMIT 1
    ),
    sample_events AS (
        SELECT 
            e.mi_person_key,
            CAST(e.event_date AS DATE) as event_date,
            t.first_f1120_date,
            CASE 
                WHEN CAST(e.event_date AS DATE) < t.first_f1120_date THEN 'PRE'
                WHEN CAST(e.event_date AS DATE) >= t.first_f1120_date THEN 'POST'
                ELSE 'UNKNOWN'
            END as timing,
            COALESCE(
                e.primary_icd_diagnosis_code,
                e.two_icd_diagnosis_code,
                e.three_icd_diagnosis_code,
                e.drug_name,
                e.procedure_code
            ) as code_or_drug
        FROM read_parquet('{model_data_path}') e
        CROSS JOIN sample_patient t
        WHERE e.mi_person_key = t.mi_person_key
        ORDER BY e.event_date
        LIMIT 20
    )
    SELECT * FROM sample_events
    """
    
    sample_events = con.execute(query4).df()
    if len(sample_events) > 0:
        print(f"\nSample Events for One Patient (First 20 events):")
        print(f"{'='*80}")
        print(f"   Patient: {sample_events['mi_person_key'].iloc[0]}")
        print(f"   First F1120 Date: {sample_events['first_f1120_date'].iloc[0]}")
        print(f"\n   Events:")
        for _, row in sample_events.iterrows():
            print(f"     {row['event_date']} | {row['timing']:4s} | {row['code_or_drug']}")
    
    # Final assessment
    pre_count = timing_summary[timing_summary['timing'] == 'pre']['event_count'].iloc[0] if 'pre' in timing_summary['timing'].values else 0
    post_count = timing_summary[timing_summary['timing'] == 'post']['event_count'].iloc[0] if 'post' in timing_summary['timing'].values else 0
    
    print(f"\n{'='*80}")
    print(f"Assessment:")
    if pre_count > 0 and post_count > 0:
        print(f"   [OK] GOOD: Found both pre-F1120 ({pre_count:,}) and post-F1120 ({post_count:,}) events")
        print(f"   [OK] Pre-target analysis is possible")
    elif pre_count == 0 and post_count > 0:
        print(f"   [WARN] WARNING: No pre-F1120 events found, only post-F1120 ({post_count:,}) events")
        print(f"   [WARN] All features will be flagged as post-target leakage")
        print(f"   [WARN] This suggests events before F1120 were filtered out in Step 4a")
    elif pre_count > 0 and post_count == 0:
        print(f"   [WARN] WARNING: Found pre-F1120 ({pre_count:,}) but no post-F1120 events")
        print(f"   [WARN] This is unusual - check data filtering")
    else:
        print(f"   [ERROR] ERROR: No events found at all")
    
    con.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Check pre/post F1120 events in model_events.parquet")
    parser.add_argument("--cohort", default="opioid_ed", help="Cohort name")
    parser.add_argument("--age-band", default="13-24", help="Age band")
    args = parser.parse_args()
    
    check_pre_post_f1120_events(args.cohort, args.age_band)
