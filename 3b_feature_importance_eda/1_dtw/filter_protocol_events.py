#!/usr/bin/env python3
"""
Filter out protocol-like events using DTW time windows.

Events that are too close together (e.g., < 7 days) may indicate standard care
protocols rather than predictive patterns. This script identifies and filters
such events from model_data before feature engineering.

Strategy:
1. Calculate time intervals between consecutive events per patient
2. Identify events that are part of protocol sequences (very short intervals)
3. Filter these events out or mark them for exclusion
4. Create filtered model_data for downstream analysis
"""

import sys
import logging
from pathlib import Path
import pandas as pd
import duckdb
from typing import Optional

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))  # noqa: E402

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def calculate_event_intervals(
    model_data_path: Path,
    min_interval_days: int = 7,
    max_interval_days: Optional[int] = None
) -> pd.DataFrame:
    """
    Calculate time intervals between consecutive events per patient.
    
    Parameters:
    -----------
    model_data_path : Path
        Path to model_events.parquet
    min_interval_days : int
        Minimum interval (days) to consider non-protocol. Events closer than this
        are considered protocol-like.
    max_interval_days : Optional[int]
        Maximum interval (days) to consider. Events further apart may be outliers.
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with event intervals and protocol flags
    """
    logger.info(f"Calculating event intervals from {model_data_path}")
    
    con = duckdb.connect()
    
    query = f"""
    WITH patient_events AS (
        SELECT
            mi_person_key,
            event_date,
            target,
            drug_name,
            primary_icd_diagnosis_code,
            procedure_code,
            ROW_NUMBER() OVER (PARTITION BY mi_person_key ORDER BY event_date) as event_seq
        FROM read_parquet('{model_data_path}')
        WHERE event_date IS NOT NULL
    ),
    event_intervals AS (
        SELECT
            pe1.mi_person_key,
            pe1.event_date as current_event_date,
            pe1.target,
            pe1.drug_name,
            pe1.primary_icd_diagnosis_code,
            pe1.procedure_code,
            pe1.event_seq,
            pe2.event_date as previous_event_date,
            DATEDIFF('day', pe2.event_date, pe1.event_date) as days_since_previous,
            CASE
                WHEN pe2.event_date IS NULL THEN 1  -- First event
                WHEN DATEDIFF('day', pe2.event_date, pe1.event_date) < {min_interval_days} THEN 1  -- Protocol-like
                ELSE 0  -- Non-protocol
            END as is_protocol_event
        FROM patient_events pe1
        LEFT JOIN patient_events pe2
            ON pe1.mi_person_key = pe2.mi_person_key
            AND pe1.event_seq = pe2.event_seq + 1
    )
    SELECT * FROM event_intervals
    ORDER BY mi_person_key, event_seq
    """
    
    intervals_df = con.execute(query).df()
    con.close()
    
    logger.info(f"Calculated intervals for {len(intervals_df)} events")
    logger.info(f"Protocol events (< {min_interval_days} days apart): {(intervals_df['is_protocol_event'] == 1).sum()}")
    logger.info(f"Non-protocol events: {(intervals_df['is_protocol_event'] == 0).sum()}")
    
    return intervals_df


def filter_protocol_events(
    model_data_path: Path,
    output_path: Path,
    min_interval_days: int = 7,
    keep_first_event: bool = True,
    protocol_threshold_pct: float = 0.5
) -> pd.DataFrame:
    """
    Filter out protocol-like events from model_data.
    
    Parameters:
    -----------
    model_data_path : Path
        Input model_events.parquet path
    output_path : Path
        Output filtered model_events.parquet path
    min_interval_days : int
        Minimum interval (days) to consider non-protocol
    keep_first_event : bool
        If True, always keep the first event per patient (even if protocol-like)
    protocol_threshold_pct : float
        If a patient has > this % of protocol events, keep all events (may be
        genuinely high-frequency care)
    
    Returns:
    --------
    pd.DataFrame
        Filtered model_data
    """
    logger.info(f"Filtering protocol events from {model_data_path}")
    
    # Calculate intervals
    intervals_df = calculate_event_intervals(model_data_path, min_interval_days)
    
    # Load original model_data
    con = duckdb.connect()
    original_df = con.execute(f"SELECT * FROM read_parquet('{model_data_path}')").df()
    con.close()
    
    # Merge intervals back to original data
    original_df['event_seq'] = (
        original_df.groupby('mi_person_key')['event_date']
        .rank(method='first', ascending=True)
        .astype(int)
    )
    
    merged = original_df.merge(
        intervals_df[['mi_person_key', 'event_seq', 'is_protocol_event', 'days_since_previous']],
        on=['mi_person_key', 'event_seq'],
        how='left'
    )
    
    # Fill missing values (shouldn't happen, but handle gracefully)
    merged['is_protocol_event'] = merged['is_protocol_event'].fillna(0)
    merged['days_since_previous'] = merged['days_since_previous'].fillna(0)
    
    # Calculate protocol percentage per patient
    patient_protocol_pct = merged.groupby('mi_person_key')['is_protocol_event'].mean()
    merged['patient_protocol_pct'] = merged['mi_person_key'].map(patient_protocol_pct)
    
    # Filter logic:
    # 1. Keep first event if keep_first_event=True
    # 2. Keep events from patients with high protocol % (may be genuinely high-frequency care)
    # 3. Otherwise, exclude protocol events
    if keep_first_event:
        merged['keep_event'] = (
            (merged['event_seq'] == 1) |  # First event
            (merged['patient_protocol_pct'] > protocol_threshold_pct) |  # High-frequency patients
            (merged['is_protocol_event'] == 0)  # Non-protocol events
        )
    else:
        merged['keep_event'] = (
            (merged['patient_protocol_pct'] > protocol_threshold_pct) |  # High-frequency patients
            (merged['is_protocol_event'] == 0)  # Non-protocol events
        )
    
    filtered_df = merged[merged['keep_event']].copy()
    
    # Drop helper columns
    filtered_df = filtered_df.drop(columns=[
        'event_seq', 'is_protocol_event', 'days_since_previous',
        'patient_protocol_pct', 'keep_event'
    ], errors='ignore')
    
    logger.info(f"Filtered {len(original_df)} events -> {len(filtered_df)} events")
    logger.info(f"Removed {len(original_df) - len(filtered_df)} protocol events ({100*(len(original_df) - len(filtered_df))/len(original_df):.1f}%)")
    
    # Save filtered data
    output_path.parent.mkdir(parents=True, exist_ok=True)
    filtered_df.to_parquet(output_path, index=False)
    logger.info(f"Saved filtered model_data to {output_path}")
    
    return filtered_df


def create_protocol_summary(
    intervals_df: pd.DataFrame,
    output_path: Optional[Path] = None
) -> pd.DataFrame:
    """
    Create summary statistics about protocol events.
    
    Returns:
    --------
    pd.DataFrame
        Summary statistics per patient and overall
    """
    # Per-patient summary
    patient_summary = intervals_df.groupby('mi_person_key').agg({
        'is_protocol_event': ['sum', 'mean', 'count'],
        'days_since_previous': ['mean', 'median', 'min', 'max']
    }).reset_index()
    
    patient_summary.columns = [
        'mi_person_key',
        'protocol_event_count',
        'protocol_event_pct',
        'total_events',
        'mean_interval_days',
        'median_interval_days',
        'min_interval_days',
        'max_interval_days'
    ]
    
    if output_path:
        patient_summary.to_csv(output_path, index=False)
        logger.info(f"Saved protocol summary to {output_path}")
    
    return patient_summary


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Filter protocol-like events using DTW time windows")
    parser.add_argument("--cohort-name", type=str, required=True, help="Cohort name (e.g., opioid_ed)")
    parser.add_argument("--age-band", type=str, required=True, help="Age band (e.g., 0-12)")
    parser.add_argument("--min-interval-days", type=int, default=7, help="Minimum interval (days) to consider non-protocol")
    parser.add_argument("--keep-first-event", action="store_true", default=True, help="Always keep first event per patient")
    parser.add_argument("--protocol-threshold-pct", type=float, default=0.5, help="Keep all events if patient has > this % protocol events")
    
    args = parser.parse_args()
    
    model_data_path = (
        PROJECT_ROOT
        / "model_data"
        / f"cohort_name={args.cohort_name}"
        / f"age_band={args.age_band}"
        / "model_events.parquet"
    )
    
    output_path = (
        PROJECT_ROOT
        / "model_data"
        / f"cohort_name={args.cohort_name}"
        / f"age_band={args.age_band}"
        / "model_events_no_protocols.parquet"
    )
    
    summary_path = (
        PROJECT_ROOT
        / "6_dtw_analysis"
        / "outputs"
        / f"protocol_summary_{args.cohort_name}_{args.age_band.replace('-', '_')}.csv"
    )
    
    # Calculate intervals
    intervals_df = calculate_event_intervals(model_data_path, args.min_interval_days)
    
    # Create summary
    create_protocol_summary(intervals_df, summary_path)
    
    # Filter protocol events
    filtered_df = filter_protocol_events(
        model_data_path=model_data_path,
        output_path=output_path,
        min_interval_days=args.min_interval_days,
        keep_first_event=args.keep_first_event,
        protocol_threshold_pct=args.protocol_threshold_pct
    )
    
    print("\n[INFO] Protocol filtering complete!")
    print(f"  Original events: {len(intervals_df)}")
    print(f"  Filtered events: {len(filtered_df)}")
    print(f"  Removed: {len(intervals_df) - len(filtered_df)} ({100*(len(intervals_df) - len(filtered_df))/len(intervals_df):.1f}%)")

