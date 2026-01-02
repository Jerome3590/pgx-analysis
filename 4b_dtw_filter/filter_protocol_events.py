#!/usr/bin/env python3
"""
Filter out protocol-like events using DTW time windows.

Events that are too close together (e.g., < 7 days) may indicate standard care
protocols rather than predictive patterns. This script identifies and filters
such events from model_data before feature engineering.
"""

import sys
import logging
from pathlib import Path
from typing import Optional

import duckdb
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

OUTPUT_ROOT = PROJECT_ROOT / "4b_dtw_filter" / "outputs"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def calculate_event_intervals(
    model_data_path: Path,
    min_interval_days: int = 7,
    max_interval_days: Optional[int] = None,
) -> pd.DataFrame:
    """
    Calculate time intervals between consecutive events per patient.

    Parameters
    ----------
    model_data_path : Path
        Path to model_events.parquet
    min_interval_days : int
        Minimum interval (days) to consider non-protocol. Events closer than this
        are considered protocol-like.
    max_interval_days : Optional[int]
        Maximum interval (days) to consider. Events further apart may be outliers.

    Returns
    -------
    pd.DataFrame
        DataFrame with event intervals and protocol flags
    """
    logger.info("Calculating event intervals from {0}".format(model_data_path))

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
            ROW_NUMBER() OVER (
                PARTITION BY mi_person_key
                ORDER BY event_date
            ) AS event_seq
        FROM read_parquet('{model_data_path}')
        WHERE event_date IS NOT NULL
    ),
    event_intervals AS (
        SELECT
            pe1.mi_person_key,
            pe1.event_date AS current_event_date,
            pe1.target,
            pe1.drug_name,
            pe1.primary_icd_diagnosis_code,
            pe1.procedure_code,
            pe1.event_seq,
            pe2.event_date AS previous_event_date,
            DATEDIFF('day', pe2.event_date, pe1.event_date) AS days_since_previous,
            CASE
                WHEN pe2.event_date IS NULL THEN 1  -- First event
                WHEN DATEDIFF('day', pe2.event_date, pe1.event_date) < {min_interval_days}
                    THEN 1  -- Protocol-like
                ELSE 0  -- Non-protocol
            END AS is_protocol_event
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

    logger.info("Calculated intervals for {0} events".format(len(intervals_df)))
    logger.info(
        "Protocol events (< {0} days apart): {1}".format(
            min_interval_days, (intervals_df["is_protocol_event"] == 1).sum()
        )
    )
    logger.info(
        "Non-protocol events: {0}".format(
            (intervals_df["is_protocol_event"] == 0).sum()
        )
    )

    return intervals_df


def filter_protocol_events(
    model_data_path: Path,
    output_path: Path,
    min_interval_days: int = 7,
    keep_first_event: bool = True,
    protocol_threshold_pct: float = 0.5,
) -> pd.DataFrame:
    """
    Filter out protocol-like events from model_data.

    Parameters
    ----------
    model_data_path : Path
        Input model_events.parquet path
    output_path : Path
        Output filtered model_events.parquet path
    min_interval_days : int
        Minimum interval (days) to consider non-protocol
    keep_first_event : bool
        If True, always keep the first event per patient (even if protocol-like)
    protocol_threshold_pct : float
        If a patient has > this % of protocol events, keep all events

    Returns
    -------
    pd.DataFrame
        Filtered model_data
    """
    logger.info("Filtering protocol events from {0}".format(model_data_path))

    intervals_df = calculate_event_intervals(model_data_path, min_interval_days)

    con = duckdb.connect()
    original_df = con.execute(
        f"SELECT * FROM read_parquet('{model_data_path}')"
    ).df()
    con.close()

    # Rank events per patient; allow missing dates and fill them with 0 so they
    # are treated as non-protocol and always kept.
    event_seq = (
        original_df.groupby("mi_person_key")["event_date"]
        .rank(method="first", ascending=True)
    )
    original_df["event_seq"] = event_seq.fillna(0).astype(int)

    merged = original_df.merge(
        intervals_df[
            [
                "mi_person_key",
                "event_seq",
                "is_protocol_event",
                "days_since_previous",
            ]
        ],
        on=["mi_person_key", "event_seq"],
        how="left",
    )

    merged["is_protocol_event"] = merged["is_protocol_event"].fillna(0)
    merged["days_since_previous"] = merged["days_since_previous"].fillna(0)

    patient_protocol_pct = merged.groupby("mi_person_key")["is_protocol_event"].mean()
    merged["patient_protocol_pct"] = merged["mi_person_key"].map(patient_protocol_pct)

    if keep_first_event:
        merged["keep_event"] = (
            (merged["event_seq"] == 1)
            | (merged["patient_protocol_pct"] > protocol_threshold_pct)
            | (merged["is_protocol_event"] == 0)
        )
    else:
        merged["keep_event"] = (
            (merged["patient_protocol_pct"] > protocol_threshold_pct)
            | (merged["is_protocol_event"] == 0)
        )

    filtered_df = merged[merged["keep_event"]].copy()

    filtered_df = filtered_df.drop(
        columns=[
            "event_seq",
            "is_protocol_event",
            "days_since_previous",
            "patient_protocol_pct",
            "keep_event",
        ],
        errors="ignore",
    )

    logger.info("Filtered {0} events -> {1} events".format(len(original_df), len(filtered_df)))
    logger.info(
        "Removed {0} protocol events ({1:.1f}%)".format(
            len(original_df) - len(filtered_df),
            100.0
            * (len(original_df) - len(filtered_df))
            / max(len(original_df), 1),
        )
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    filtered_df.to_parquet(output_path, index=False)
    logger.info("Saved filtered model_data to {0}".format(output_path))

    return filtered_df


def create_protocol_summary(
    intervals_df: pd.DataFrame,
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Create summary statistics about protocol events.

    Returns
    -------
    pd.DataFrame
        Summary statistics per patient and overall
    """
    patient_summary = intervals_df.groupby("mi_person_key").agg(
        {
            "is_protocol_event": ["sum", "mean", "count"],
            "days_since_previous": ["mean", "median", "min", "max"],
        }
    )

    patient_summary = patient_summary.reset_index()
    patient_summary.columns = [
        "mi_person_key",
        "protocol_event_count",
        "protocol_event_pct",
        "total_events",
        "mean_interval_days",
        "median_interval_days",
        "min_interval_days",
        "max_interval_days",
    ]

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        patient_summary.to_csv(output_path, index=False)
        logger.info("Saved protocol summary to {0}".format(output_path))

    return patient_summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Filter protocol-like events using DTW time windows"
    )
    parser.add_argument(
        "--cohort-name",
        type=str,
        required=True,
        help="Cohort name (e.g., opioid_ed)",
    )
    parser.add_argument(
        "--age-band",
        type=str,
        required=True,
        help="Age band (e.g., 0-12)",
    )
    parser.add_argument(
        "--min-interval-days",
        type=int,
        default=7,
        help="Minimum interval (days) to consider non-protocol",
    )
    parser.add_argument(
        "--keep-first-event",
        action="store_true",
        default=True,
        help="Always keep first event per patient",
    )
    parser.add_argument(
        "--protocol-threshold-pct",
        type=float,
        default=0.5,
        help=(
            "Keep all events if patient has > this percent protocol events "
            "(default: 0.5)"
        ),
    )

    args = parser.parse_args()

    model_data_path = (
        PROJECT_ROOT
        / "4a_model_data"
        / f"cohort_name={args.cohort_name}"
        / f"age_band={args.age_band}"
        / "model_events.parquet"
    )

    output_path = (
        PROJECT_ROOT
        / "4a_model_data"
        / f"cohort_name={args.cohort_name}"
        / f"age_band={args.age_band}"
        / "model_events_no_protocols.parquet"
    )

    age_band_fname = args.age_band.replace("-", "_")

    # Output paths for audit artifacts
    audit_dir = OUTPUT_ROOT / args.cohort_name / age_band_fname
    audit_dir.mkdir(parents=True, exist_ok=True)

    summary_path = audit_dir / f"protocol_summary_{args.cohort_name}_{age_band_fname}.csv"
    intervals_path = audit_dir / f"event_intervals_{args.cohort_name}_{age_band_fname}.parquet"

    intervals_df = calculate_event_intervals(model_data_path, args.min_interval_days)

    # Persist full event-level intervals with protocol flags for audit/exploration
    intervals_df.to_parquet(intervals_path, index=False)
    logger.info("Saved event-level intervals to {0}".format(intervals_path))

    # Per-patient summary
    create_protocol_summary(intervals_df, summary_path)

    filtered_df = filter_protocol_events(
        model_data_path=model_data_path,
        output_path=output_path,
        min_interval_days=args.min_interval_days,
        keep_first_event=args.keep_first_event,
        protocol_threshold_pct=args.protocol_threshold_pct,
    )

    print("\n[INFO] Protocol filtering complete!")
    print(f"  Original events: {len(intervals_df)}")
    print(f"  Filtered events: {len(filtered_df)}")
    print(
        "  Removed: {0} ({1:.1f}%)".format(
            len(intervals_df) - len(filtered_df),
            100.0
            * (len(intervals_df) - len(filtered_df))
            / max(len(intervals_df), 1),
        )
    )

