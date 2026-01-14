#!/usr/bin/env python3
"""
Research script to analyze trajectories, time windows, and codes.

This script helps classify:
- Clinical/Procedural: Useful signals (keep)
- Administrative/Post-Event: Noise (filter)
"""

import sys
from pathlib import Path
import pandas as pd
import duckdb
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from py_helpers.constants import COHORT_NAMES, AGE_BANDS


def analyze_time_windows(
    intervals_path: Path,
    min_interval_days: int = 1
) -> Dict:
    """Analyze time window distributions."""
    if not intervals_path.exists():
        return {"error": "Intervals file not found"}
    
    intervals_df = pd.read_parquet(intervals_path)
    
    valid_intervals = intervals_df['days_since_previous'].dropna()
    
    stats = {
        "total_events": len(intervals_df),
        "protocol_events": int(intervals_df['is_protocol_event'].sum()),
        "protocol_pct": float(intervals_df['is_protocol_event'].mean() * 100),
        "non_protocol_events": int((~intervals_df['is_protocol_event']).sum()),
        "non_protocol_pct": float((1 - intervals_df['is_protocol_event'].mean()) * 100),
        "mean_interval_days": float(valid_intervals.mean()),
        "median_interval_days": float(valid_intervals.median()),
        "min_interval_days": float(valid_intervals.min()),
        "max_interval_days": float(valid_intervals.max()),
        "std_interval_days": float(valid_intervals.std()),
    }
    
    # Interval bins
    bins = [0, 1, 3, 7, 14, 30, 90, 365, float('inf')]
    labels = ['<1 day', '1-3 days', '3-7 days', '7-14 days', '14-30 days', '30-90 days', '90-365 days', '>365 days']
    intervals_df['interval_bin'] = pd.cut(valid_intervals, bins=bins, labels=labels, right=False)
    stats["interval_distribution"] = intervals_df['interval_bin'].value_counts().to_dict()
    
    return stats


def analyze_common_sequences(
    model_data_path: Path,
    top_n: int = 20
) -> Dict:
    """Analyze common 2-event sequences."""
    if not model_data_path.exists():
        return {"error": "Model data file not found"}
    
    con = duckdb.connect()
    
    query = f"""
    WITH patient_events AS (
        SELECT 
            mi_person_key,
            event_date,
            target,
            COALESCE(drug_name, '') as drug,
            COALESCE(primary_icd_diagnosis_code, '') as icd,
            COALESCE(procedure_code, '') as cpt,
            ROW_NUMBER() OVER (PARTITION BY mi_person_key ORDER BY event_date) as seq_num
        FROM read_parquet('{model_data_path}')
        WHERE event_date IS NOT NULL
    ),
    events_with_intervals AS (
        SELECT 
            e1.*,
            e2.event_date as prev_event_date,
            DATEDIFF('day', e2.event_date, e1.event_date) as days_since_previous
        FROM patient_events e1
        LEFT JOIN patient_events e2
            ON e1.mi_person_key = e2.mi_person_key
            AND e1.seq_num = e2.seq_num + 1
    )
    SELECT * FROM events_with_intervals
    ORDER BY mi_person_key, seq_num
    """
    
    events_df = con.execute(query).df()
    con.close()
    
    # Extract 2-event sequences
    sequences = []
    for pid in events_df['mi_person_key'].unique():
        patient_events = events_df[events_df['mi_person_key'] == pid].sort_values('seq_num')
        
        for i in range(len(patient_events) - 1):
            curr = patient_events.iloc[i]
            next_event = patient_events.iloc[i + 1]
            
            # Create sequence identifier
            seq_type = []
            if curr['drug']:
                seq_type.append(f"DRUG:{curr['drug']}")
            if curr['icd']:
                seq_type.append(f"ICD:{curr['icd']}")
            if curr['cpt']:
                seq_type.append(f"CPT:{curr['cpt']}")
            
            next_seq_type = []
            if next_event['drug']:
                next_seq_type.append(f"DRUG:{next_event['drug']}")
            if next_event['icd']:
                next_seq_type.append(f"ICD:{next_event['icd']}")
            if next_event['cpt']:
                next_seq_type.append(f"CPT:{next_event['cpt']}")
            
            if seq_type and next_seq_type:
                sequences.append({
                    'sequence': f"{', '.join(seq_type)} -> {', '.join(next_seq_type)}",
                    'days_apart': next_event['days_since_previous'],
                    'is_protocol': next_event['days_since_previous'] < 3 if pd.notna(next_event['days_since_previous']) else False,
                    'target': curr['target']
                })
    
    sequences_df = pd.DataFrame(sequences)
    
    # Most common sequences
    top_sequences = sequences_df['sequence'].value_counts().head(top_n)
    top_sequences_data = []
    for seq, count in top_sequences.items():
        seq_data = sequences_df[sequences_df['sequence'] == seq]
        top_sequences_data.append({
            'sequence': seq,
            'count': int(count),
            'protocol_pct': float(seq_data['is_protocol'].mean() * 100),
            'mean_days': float(seq_data['days_apart'].mean())
        })
    
    # Protocol sequences
    protocol_sequences = sequences_df[sequences_df['is_protocol']]['sequence'].value_counts().head(top_n)
    protocol_sequences_data = []
    for seq, count in protocol_sequences.items():
        seq_data = sequences_df[sequences_df['sequence'] == seq]
        protocol_sequences_data.append({
            'sequence': seq,
            'count': int(count),
            'mean_days': float(seq_data['days_apart'].mean())
        })
    
    # Non-protocol sequences
    non_protocol_sequences = sequences_df[~sequences_df['is_protocol']]['sequence'].value_counts().head(top_n)
    non_protocol_sequences_data = []
    for seq, count in non_protocol_sequences.items():
        seq_data = sequences_df[sequences_df['sequence'] == seq]
        non_protocol_sequences_data.append({
            'sequence': seq,
            'count': int(count),
            'mean_days': float(seq_data['days_apart'].mean())
        })
    
    return {
        "top_sequences": top_sequences_data,
        "protocol_sequences": protocol_sequences_data,
        "non_protocol_sequences": non_protocol_sequences_data
    }


def analyze_codes(
    model_data_path: Path,
    intervals_path: Path,
    top_n: int = 20
) -> Dict:
    """Analyze codes by protocol status."""
    if not model_data_path.exists() or not intervals_path.exists():
        return {"error": "Required files not found"}
    
    con = duckdb.connect()
    
    query = f"""
    WITH model_events AS (
        SELECT 
            mi_person_key,
            event_date,
            drug_name,
            primary_icd_diagnosis_code as icd,
            procedure_code as cpt,
            target
        FROM read_parquet('{model_data_path}')
    ),
    intervals AS (
        SELECT 
            mi_person_key,
            current_event_date,
            is_protocol_event,
            days_since_previous
        FROM read_parquet('{intervals_path}')
    )
    SELECT 
        m.*,
        i.is_protocol_event,
        i.days_since_previous
    FROM model_events m
    INNER JOIN intervals i
        ON m.mi_person_key = i.mi_person_key
        AND m.event_date = i.current_event_date
    """
    
    codes_df = con.execute(query).df()
    con.close()
    
    # Analyze ICD codes
    icd_analysis = codes_df[codes_df['icd'].notna()].groupby('icd').agg({
        'is_protocol_event': ['sum', 'mean', 'count'],
        'days_since_previous': 'mean'
    }).reset_index()
    icd_analysis.columns = ['icd', 'protocol_count', 'protocol_pct', 'total_count', 'mean_days']
    icd_analysis = icd_analysis.sort_values('total_count', ascending=False).head(top_n)
    
    # Analyze CPT codes
    cpt_analysis = codes_df[codes_df['cpt'].notna()].groupby('cpt').agg({
        'is_protocol_event': ['sum', 'mean', 'count'],
        'days_since_previous': 'mean'
    }).reset_index()
    cpt_analysis.columns = ['cpt', 'protocol_count', 'protocol_pct', 'total_count', 'mean_days']
    cpt_analysis = cpt_analysis.sort_values('total_count', ascending=False).head(top_n)
    
    # Analyze Drug names
    drug_analysis = codes_df[codes_df['drug_name'].notna()].groupby('drug_name').agg({
        'is_protocol_event': ['sum', 'mean', 'count'],
        'days_since_previous': 'mean'
    }).reset_index()
    drug_analysis.columns = ['drug', 'protocol_count', 'protocol_pct', 'total_count', 'mean_days']
    drug_analysis = drug_analysis.sort_values('total_count', ascending=False).head(top_n)
    
    return {
        "icd_codes": icd_analysis.to_dict('records'),
        "cpt_codes": cpt_analysis.to_dict('records'),
        "drugs": drug_analysis.to_dict('records')
    }


def check_post_event_leakage(
    model_data_path: Path,
    cohort_name: str
) -> Dict:
    """Check for events after target event date (leakage)."""
    if not model_data_path.exists():
        return {"error": "Model data file not found"}
    
    # Determine target date field based on cohort
    if "opioid" in cohort_name.lower():
        target_date_field = "first_opioid_ed_date"
    else:
        target_date_field = "first_ed_non_opioid_date"
    
    con = duckdb.connect()
    
    query = f"""
    WITH target_events AS (
        SELECT DISTINCT
            mi_person_key,
            target,
            CAST({target_date_field} AS DATE) as target_event_date
        FROM read_parquet('{model_data_path}')
        WHERE target = 1 AND {target_date_field} IS NOT NULL
    ),
    all_events AS (
        SELECT 
            mi_person_key,
            event_date,
            target
        FROM read_parquet('{model_data_path}')
    ),
    events_with_target_dates AS (
        SELECT 
            a.mi_person_key,
            a.event_date,
            a.target,
            t.target_event_date,
            CASE 
                WHEN a.target = 1 AND t.target_event_date IS NOT NULL AND a.event_date >= CAST(t.target_event_date AS TIMESTAMP)
                THEN 1
                ELSE 0
            END as is_post_event
        FROM all_events a
        LEFT JOIN target_events t ON a.mi_person_key = t.mi_person_key
    )
    SELECT 
        COUNT(*) as total_events,
        SUM(is_post_event) as post_event_events,
        SUM(CASE WHEN target = 1 THEN is_post_event ELSE 0 END) as target_post_event_events,
        SUM(CASE WHEN target = 0 THEN 1 ELSE 0 END) as control_events
    FROM events_with_target_dates
    """
    
    leakage_df = con.execute(query).df()
    con.close()
    
    row = leakage_df.iloc[0]
    return {
        "total_events": int(row['total_events']),
        "post_event_events": int(row['post_event_events']),
        "post_event_pct": float(100 * row['post_event_events'] / row['total_events']),
        "target_post_event_events": int(row['target_post_event_events']),
        "control_events": int(row['control_events']),
        "has_leakage": bool(row['post_event_events'] > 0)
    }


def print_research_summary(
    cohort_name: str,
    age_band: str,
    time_window_stats: Dict,
    sequences: Dict,
    codes: Dict,
    leakage: Dict
):
    """Print formatted research summary."""
    print("=" * 80)
    print(f"DTW Filter Research Summary: {cohort_name} / {age_band}")
    print("=" * 80)
    
    # Time Windows
    print("\n1. TIME WINDOWS:")
    print("-" * 80)
    if "error" not in time_window_stats:
        print(f"   Total events: {time_window_stats['total_events']:,}")
        print(f"   Protocol events (< {min_interval_days} days): {time_window_stats['protocol_events']:,} ({time_window_stats['protocol_pct']:.1f}%)")
        print(f"   Non-protocol events: {time_window_stats['non_protocol_events']:,} ({time_window_stats['non_protocol_pct']:.1f}%)")
        print(f"   Mean interval: {time_window_stats['mean_interval_days']:.1f} days")
        print(f"   Median interval: {time_window_stats['median_interval_days']:.1f} days")
    else:
        print(f"   {time_window_stats['error']}")
    
    # Sequences
    print("\n2. COMMON TRAJECTORIES:")
    print("-" * 80)
    if "error" not in sequences:
        print(f"   Top sequences: {len(sequences.get('top_sequences', []))}")
        print(f"   Protocol sequences: {len(sequences.get('protocol_sequences', []))}")
        print(f"   Non-protocol sequences: {len(sequences.get('non_protocol_sequences', []))}")
    else:
        print(f"   {sequences['error']}")
    
    # Codes
    print("\n3. CODE CLASSIFICATION:")
    print("-" * 80)
    if "error" not in codes:
        print(f"   Top ICD codes: {len(codes.get('icd_codes', []))}")
        print(f"   Top CPT codes: {len(codes.get('cpt_codes', []))}")
        print(f"   Top drugs: {len(codes.get('drugs', []))}")
    else:
        print(f"   {codes['error']}")
    
    # Leakage
    print("\n4. POST-EVENT EVENTS (LEAKAGE CHECK):")
    print("-" * 80)
    if "error" not in leakage:
        print(f"   Total events: {leakage['total_events']:,}")
        print(f"   Post-event events: {leakage['post_event_events']:,} ({leakage['post_event_pct']:.1f}%)")
        if leakage['has_leakage']:
            print("   ⚠️  WARNING: Post-event events detected! These should be filtered.")
        else:
            print("   ✓ No post-event events detected.")
    else:
        print(f"   {leakage['error']}")
    
    print("\n" + "=" * 80)


def main():
    """Main research function."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Research trajectories, time windows, and codes for DTW filtering"
    )
    parser.add_argument(
        "--cohort-name",
        type=str,
        required=True,
        help="Cohort name (e.g., opioid_ed)"
    )
    parser.add_argument(
        "--age-band",
        type=str,
        required=True,
        help="Age band (e.g., 0-12)"
    )
    
    args = parser.parse_args()
    
    # Set up paths
    model_data_dir = (
        PROJECT_ROOT
        / "4a_model_data"
        / f"cohort_name={args.cohort_name}"
        / f"age_band={args.age_band}"
    )
    
    model_data_path = model_data_dir / "model_events.parquet"
    
    filter_output_dir = (
        PROJECT_ROOT
        / "4b_dtw_filter"
        / "outputs"
        / args.cohort_name
        / args.age_band.replace("-", "_")
    )
    
    age_band_fname = args.age_band.replace("-", "_")
    intervals_path = filter_output_dir / f"event_intervals_{args.cohort_name}_{age_band_fname}.parquet"
    
    # Run analyses
    time_window_stats = analyze_time_windows(intervals_path)
    sequences = analyze_common_sequences(model_data_path)
    codes = analyze_codes(model_data_path, intervals_path)
    leakage = check_post_event_leakage(model_data_path, args.cohort_name)
    
    # Print summary
    print_research_summary(
        args.cohort_name,
        args.age_band,
        time_window_stats,
        sequences,
        codes,
        leakage
    )


if __name__ == "__main__":
    main()
