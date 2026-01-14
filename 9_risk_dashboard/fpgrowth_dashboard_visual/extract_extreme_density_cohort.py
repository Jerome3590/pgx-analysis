#!/usr/bin/env python3
"""
Extract an "extreme density" cohort from model_events and remove these patients
from the main 4a_model_data event set for a given cohort / age_band.

Definition of extreme density:
- Reuses the same Transaction_Density logic as cohort_fpgrowth.assign_transaction_density
- Based on medical_code transactions (combined ICD + CPT codes) over the TRAIN years.

Outputs:
- New cohort under model_data/cohort_name={source}_extreme_density/age_band={age_band}/model_events.parquet
- Updated model_data/cohort_name={source}/age_band={age_band}/model_events.parquet with extreme patients removed
- CSV listing extreme-density mi_person_key values for auditing/visualization.
"""

import argparse
import logging
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# Local copies of core configuration and helpers from cohort_fpgrowth,
# now operating on the canonical 4a_model_data event store.
MODEL_DATA_ROOT = PROJECT_ROOT / "4a_model_data"
TRAIN_YEARS = [2016, 2017, 2018]
DENSITY_BINS = ["low", "medium", "high", "extreme"]


def setup_logger(name: str = "extract_extreme_density") -> logging.Logger:
    """Setup a basic logger with console output."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


def assign_transaction_density(df: pd.DataFrame, logger: logging.Logger) -> pd.DataFrame:
    """
    Calculate transaction sizes per patient and assign Transaction_Density bins.

    This mirrors the logic from cohort_fpgrowth.assign_transaction_density so that
    the definition of "extreme" density is consistent with FP-Growth.
    """
    logger.info("Calculating transaction sizes per patient for density assignment...")
    transaction_sizes = (
        df.groupby("mi_person_key")["item"].size().reset_index(name="transaction_size")
    )

    sizes = transaction_sizes["transaction_size"].values
    p25 = float(np.percentile(sizes, 25))
    p50 = float(np.percentile(sizes, 50))
    p75 = float(np.percentile(sizes, 75))
    p95 = float(np.percentile(sizes, 95))

    logger.info("Transaction size percentiles:")
    logger.info("  P25: %.1f items", p25)
    logger.info("  P50 (median): %.1f items", p50)
    logger.info("  P75: %.1f items", p75)
    logger.info("  P95: %.1f items", p95)
    logger.info("  Max: %s items", f"{int(max(sizes)):,}")

    def _assign_density(size: int) -> str:
        if size <= p25:
            return "low"
        if size <= p50:
            return "medium"
        if size <= p95:
            return "high"
        return "extreme"

    transaction_sizes["Transaction_Density"] = transaction_sizes["transaction_size"].apply(
        _assign_density
    )

    density_counts = transaction_sizes["Transaction_Density"].value_counts()
    logger.info("Transaction density distribution:")
    for density in DENSITY_BINS:
        count = int(density_counts.get(density, 0))
        pct = (count / len(transaction_sizes)) * 100 if len(transaction_sizes) > 0 else 0.0
        logger.info("  %s: %s (%.1f%%)", density, f"{count:,}", pct)

    df_with_density = df.merge(
        transaction_sizes[["mi_person_key", "Transaction_Density", "transaction_size"]],
        on="mi_person_key",
        how="left",
    )
    return df_with_density


def _get_model_events_path(cohort_name: str, age_band: str) -> Path:
    return (
        MODEL_DATA_ROOT
        / f"cohort_name={cohort_name}"
        / f"age_band={age_band}"
        / "model_events.parquet"
    )


def extract_extreme_density_cohort(
    cohort_name: str,
    age_band: str,
    output_cohort_name: str | None = None,
) -> None:
    """
    Identify extreme-density patients (by medical_code) and:
    - write a new extreme-density cohort under 4a_model_data
    - remove them from the source cohort's model_events in 4a_model_data.
    """
    logger: logging.Logger = setup_logger("extract_extreme_density")

    if output_cohort_name is None:
        output_cohort_name = f"{cohort_name}_extreme_density"

    logger.info(
        "Extracting extreme-density cohort from %s / %s into %s",
        cohort_name,
        age_band,
        output_cohort_name,
    )

    model_events_path = _get_model_events_path(cohort_name, age_band)
    if not model_events_path.exists():
        logger.error("Source model_events not found at %s", model_events_path)
        raise SystemExit(1)

    # ------------------------------------------------------------------
    # Step 1: Load medical_code-style events for TRAIN years
    # ------------------------------------------------------------------
    con = duckdb.connect(database=":memory:")

    # Match TRAIN window used in FP-Growth (2016–2018)
    years_sql = ", ".join(str(y) for y in TRAIN_YEARS)
    event_filter = f"event_year IN ({years_sql})"

    model_events_str = str(model_events_path).replace("\\", "/")

    # NOTE: Earlier versions of model_events included an event_type column to
    # distinguish medical vs pharmacy events and this query filtered on
    # event_type = 'medical'. The current 4a_model_data schema does not carry
    # event_type, so we approximate "medical_code" density by looking at all
    # ICD diagnosis columns plus procedure_code for events in the TRAIN years,
    # independent of event_type.
    query_medical = f"""
    WITH all_med_codes AS (
        SELECT mi_person_key, primary_icd_diagnosis_code AS code
        FROM read_parquet('{model_events_str}')
        WHERE primary_icd_diagnosis_code IS NOT NULL
          AND primary_icd_diagnosis_code != ''
          AND {event_filter}
        UNION ALL
        SELECT mi_person_key, two_icd_diagnosis_code AS code
        FROM read_parquet('{model_events_str}')
        WHERE two_icd_diagnosis_code IS NOT NULL
          AND two_icd_diagnosis_code != ''
          AND {event_filter}
        UNION ALL
        SELECT mi_person_key, three_icd_diagnosis_code AS code
        FROM read_parquet('{model_events_str}')
        WHERE three_icd_diagnosis_code IS NOT NULL
          AND three_icd_diagnosis_code != ''
          AND {event_filter}
        UNION ALL
        SELECT mi_person_key, four_icd_diagnosis_code AS code
        FROM read_parquet('{model_events_str}')
        WHERE four_icd_diagnosis_code IS NOT NULL
          AND four_icd_diagnosis_code != ''
          AND {event_filter}
        UNION ALL
        SELECT mi_person_key, five_icd_diagnosis_code AS code
        FROM read_parquet('{model_events_str}')
        WHERE five_icd_diagnosis_code IS NOT NULL
          AND five_icd_diagnosis_code != ''
          AND {event_filter}
        UNION ALL
        SELECT mi_person_key, six_icd_diagnosis_code AS code
        FROM read_parquet('{model_events_str}')
        WHERE six_icd_diagnosis_code IS NOT NULL
          AND six_icd_diagnosis_code != ''
          AND {event_filter}
        UNION ALL
        SELECT mi_person_key, seven_icd_diagnosis_code AS code
        FROM read_parquet('{model_events_str}')
        WHERE seven_icd_diagnosis_code IS NOT NULL
          AND seven_icd_diagnosis_code != ''
          AND {event_filter}
        UNION ALL
        SELECT mi_person_key, eight_icd_diagnosis_code AS code
        FROM read_parquet('{model_events_str}')
        WHERE eight_icd_diagnosis_code IS NOT NULL
          AND eight_icd_diagnosis_code != ''
          AND {event_filter}
        UNION ALL
        SELECT mi_person_key, nine_icd_diagnosis_code AS code
        FROM read_parquet('{model_events_str}')
        WHERE nine_icd_diagnosis_code IS NOT NULL
          AND nine_icd_diagnosis_code != ''
          AND {event_filter}
        UNION ALL
        SELECT mi_person_key, ten_icd_diagnosis_code AS code
        FROM read_parquet('{model_events_str}')
        WHERE ten_icd_diagnosis_code IS NOT NULL
          AND ten_icd_diagnosis_code != ''
          AND {event_filter}
        UNION ALL
        SELECT mi_person_key, procedure_code AS code
        FROM read_parquet('{model_events_str}')
        WHERE procedure_code IS NOT NULL
          AND procedure_code != ''
          AND {event_filter}
    )
    SELECT mi_person_key, code AS item
    FROM all_med_codes
    """

    logger.info("Loading medical_code-style events for density computation...")
    df_med = con.execute(query_medical).df()
    con.close()

    if df_med.empty:
        logger.warning(
            "No medical_code events found for %s / %s in TRAIN years; nothing to extract",
            cohort_name,
            age_band,
        )
        return

    logger.info("Loaded %d rows for density computation", len(df_med))

    # ------------------------------------------------------------------
    # Step 2: Assign Transaction_Density and identify extreme patients
    # ------------------------------------------------------------------
    df_med = assign_transaction_density(df_med, logger)
    extreme_ids = (
        df_med.loc[df_med["Transaction_Density"] == "extreme", "mi_person_key"]
        .drop_duplicates()
        .sort_values()
    )

    n_extreme = len(extreme_ids)
    logger.info("Identified %d extreme-density patients for %s / %s", n_extreme, cohort_name, age_band)

    if n_extreme == 0:
        logger.info("No extreme-density patients to extract; exiting")
        return

    # Save list of extreme patients for auditing / visualization
    age_band_fname = age_band.replace("-", "_")
    out_dir = model_events_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    ids_csv_path = out_dir / f"extreme_density_patients_{age_band_fname}.csv"
    extreme_ids.to_frame(name="mi_person_key").to_csv(ids_csv_path, index=False)
    logger.info("Wrote extreme-density patient list to %s", ids_csv_path)

    # ------------------------------------------------------------------
    # Step 3: Split original model_events into (extreme) and (filtered)
    # ------------------------------------------------------------------
    logger.info("Splitting model_events into extreme and filtered cohorts...")
    con2 = duckdb.connect(database=":memory:")
    con2.register(
        "extreme_ids",
        pd.DataFrame({"mi_person_key": extreme_ids}),
    )

    con2.execute(
        "CREATE TABLE events AS SELECT * FROM read_parquet(?)",
        [model_events_str],
    )

    # Extreme cohort events
    extreme_events_path = (
        MODEL_DATA_ROOT
        / f"cohort_name={output_cohort_name}"
        / f"age_band={age_band}"
        / "model_events.parquet"
    )
    extreme_events_path.parent.mkdir(parents=True, exist_ok=True)

    extreme_events_str = str(extreme_events_path).replace("\\", "/")
    con2.execute(
        f"""
        COPY (
            SELECT e.*
            FROM events e
            JOIN extreme_ids x USING (mi_person_key)
        )
        TO '{extreme_events_str}'
        (FORMAT PARQUET)
        """
    )
    logger.info("Wrote extreme-density cohort events to %s", extreme_events_path)

    # Filtered (non-extreme) events replace the original model_events
    backup_path = model_events_path.with_name("model_events_with_extreme.parquet")
    if not backup_path.exists():
        model_events_path.replace(backup_path)
        logger.info("Backed up original model_events to %s", backup_path)
    else:
        logger.info("Backup already exists at %s; original will be overwritten in place", backup_path)

    model_events_str_out = str(model_events_path).replace("\\", "/")
    con2.execute(
        f"""
        COPY (
            SELECT e.*
            FROM events e
            LEFT JOIN extreme_ids x USING (mi_person_key)
            WHERE x.mi_person_key IS NULL
        )
        TO '{model_events_str_out}'
        (FORMAT PARQUET)
        """
    )
    logger.info("Rewrote model_events with extreme-density patients removed at %s", model_events_path)

    con2.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Extract an extreme-density cohort from model_events and remove those "
            "patients from the source 4a_model_data event set."
        )
    )
    parser.add_argument(
        "--cohort-name",
        required=True,
        help="Source cohort name (e.g., opioid_ed)",
    )
    parser.add_argument(
        "--age-band",
        required=True,
        help="Age band (e.g., 25-44)",
    )
    parser.add_argument(
        "--output-cohort-name",
        required=False,
        help="Name for the new extreme-density cohort (default: {cohort_name}_extreme_density)",
    )

    args = parser.parse_args()
    extract_extreme_density_cohort(
        cohort_name=args.cohort_name,
        age_band=args.age_band,
        output_cohort_name=args.output_cohort_name,
    )


if __name__ == "__main__":
    main()

