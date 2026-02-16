#!/usr/bin/env python3
"""
Extract an "extreme density" cohort from model_events and remove these patients
from the main 4_model_data event set for a given cohort / age_band.

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
import os
import sys
from pathlib import Path
from typing import Optional, Tuple

import duckdb
import numpy as np
import pandas as pd

# Repo root so 4_model_data and py_helpers are available (same as other dtw scripts)
REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from py_helpers.fe_monitor import mirror_log_to_s3
except ImportError:
    mirror_log_to_s3 = None
try:
    from py_helpers.model_data_paths import resolve_model_events_paths
except ImportError:
    resolve_model_events_paths = None

# On EC2 model data is on NVMe; try /mnt/nvme first, then PGX_DATA_ROOT, then repo.
_MODEL_DATA_ROOTS = [
    Path("/mnt/nvme/4_model_data"),
    *([Path(os.environ.get("PGX_DATA_ROOT", "").strip()) / "4_model_data"] if os.environ.get("PGX_DATA_ROOT", "").strip() else []),
    REPO_ROOT / "4_model_data",
]
TRAIN_YEARS = [2016, 2017, 2018]
DENSITY_BINS = ["low", "medium", "high", "extreme"]


def setup_logger(
    name: str = "extract_extreme_density",
    cohort_name: Optional[str] = None,
    age_band: Optional[str] = None,
) -> Tuple[logging.Logger, Optional[Path]]:
    """Setup logger with console output; optional file handler and log path when cohort/age_band provided (for S3 mirror)."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    log_path: Optional[Path] = None

    if not logger.handlers:
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        if cohort_name is not None and age_band is not None:
            logs_dir = REPO_ROOT / "logs" / "feature_engineering" / "extreme_density_extract"
            logs_dir.mkdir(parents=True, exist_ok=True)
            age_band_fname = age_band.replace("-", "_")
            log_path = logs_dir / f"extract_extreme_density_{cohort_name}_{age_band_fname}.log"
            file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
    return logger, log_path


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
    """Resolve model_events path; EC2 uses underscore in partition (age_band=75_84). Try underscore first, then hyphen."""
    band_underscore = age_band.replace("-", "_") if "-" in age_band else age_band
    band_hyphen = age_band.replace("_", "-") if "_" in age_band else age_band
    for root in _MODEL_DATA_ROOTS:
        if not root.exists():
            continue
        for band in (band_underscore, band_hyphen):
            d = root / f"cohort_name={cohort_name}" / f"age_band={band}"
            for name in ("model_events_no_protocols.parquet", "model_events.parquet"):
                p = d / name
                if p.exists():
                    return p
    return (
        _MODEL_DATA_ROOTS[-1]
        / f"cohort_name={cohort_name}"
        / f"age_band={age_band}"
        / "model_events.parquet"
    )


def _get_model_events_paths_and_from_sql(cohort_name: str, age_band: str) -> tuple[list[Path], str]:
    """Resolve 1 or 2 model_events paths (85-114 = 85-94 + 95-114). Return (paths, FROM-clause SQL)."""
    if resolve_model_events_paths and REPO_ROOT.exists():
        paths = resolve_model_events_paths(REPO_ROOT, cohort_name, age_band)
        if paths and all(p.exists() for p in paths):
            norm = [str(p).replace("\\", "/") for p in paths]
            if len(norm) == 1:
                return paths, f"read_parquet('{norm[0]}')"
            if len(norm) == 2:
                return paths, f"(SELECT * FROM read_parquet('{norm[0]}') UNION ALL SELECT * FROM read_parquet('{norm[1]}'))"
    single = _get_model_events_path(cohort_name, age_band)
    if single.exists():
        return [single], f"read_parquet('{str(single).replace(chr(92), '/')}')"
    return [], ""


def extract_extreme_density_cohort(
    cohort_name: str,
    age_band: str,
    output_cohort_name: str | None = None,
    logger: Optional[logging.Logger] = None,
) -> None:
    """
    Identify extreme-density patients (by medical_code) and:
    - write a new extreme-density cohort under 4_model_data
    - remove them from the source cohort's model_events in 4_model_data.
    """
    if logger is None:
        logger, _ = setup_logger("extract_extreme_density")

    if output_cohort_name is None:
        output_cohort_name = f"{cohort_name}_extreme_density"

    logger.info(
        "Extracting extreme-density cohort from %s / %s into %s",
        cohort_name,
        age_band,
        output_cohort_name,
    )

    paths, from_sql = _get_model_events_paths_and_from_sql(cohort_name, age_band)
    if not paths or not from_sql:
        logger.error("Source model_events not found for %s / %s (tried single path and 85-114 = 85-94 + 95-114)", cohort_name, age_band)
        raise SystemExit(1)
    if len(paths) == 2:
        logger.info("Using model_events for 85-114 as union of 85-94 + 95-114")
    # Write path: single partition or new 85-114 when we read from two
    if len(paths) == 2:
        _effective_root = paths[0].parent.parent.parent
        model_events_path = _effective_root / f"cohort_name={cohort_name}" / "age_band=85-114" / "model_events.parquet"
    else:
        model_events_path = paths[0]

    # ------------------------------------------------------------------
    # Step 1: Load medical_code-style events for TRAIN years
    # ------------------------------------------------------------------
    con = duckdb.connect(database=":memory:")
    con.execute(f"CREATE VIEW model_events AS SELECT * FROM {from_sql}")

    # Match TRAIN window used in FP-Growth (2016–2018)
    years_sql = ", ".join(str(y) for y in TRAIN_YEARS)
    event_filter = f"event_year IN ({years_sql})"

    # NOTE: Earlier versions of model_events included an event_type column to
    # distinguish medical vs pharmacy events and this query filtered on
    # event_type = 'medical'. The current 4_model_data schema does not carry
    # event_type, so we approximate "medical_code" density by looking at all
    # ICD diagnosis columns plus procedure_code for events in the TRAIN years,
    # independent of event_type.
    query_medical = f"""
    WITH all_med_codes AS (
        SELECT mi_person_key, primary_icd_diagnosis_code AS code
        FROM model_events
        WHERE primary_icd_diagnosis_code IS NOT NULL
          AND primary_icd_diagnosis_code != ''
          AND {event_filter}
        UNION ALL
        SELECT mi_person_key, two_icd_diagnosis_code AS code
        FROM model_events
        WHERE two_icd_diagnosis_code IS NOT NULL
          AND two_icd_diagnosis_code != ''
          AND {event_filter}
        UNION ALL
        SELECT mi_person_key, three_icd_diagnosis_code AS code
        FROM model_events
        WHERE three_icd_diagnosis_code IS NOT NULL
          AND three_icd_diagnosis_code != ''
          AND {event_filter}
        UNION ALL
        SELECT mi_person_key, four_icd_diagnosis_code AS code
        FROM model_events
        WHERE four_icd_diagnosis_code IS NOT NULL
          AND four_icd_diagnosis_code != ''
          AND {event_filter}
        UNION ALL
        SELECT mi_person_key, five_icd_diagnosis_code AS code
        FROM model_events
        WHERE five_icd_diagnosis_code IS NOT NULL
          AND five_icd_diagnosis_code != ''
          AND {event_filter}
        UNION ALL
        SELECT mi_person_key, six_icd_diagnosis_code AS code
        FROM model_events
        WHERE six_icd_diagnosis_code IS NOT NULL
          AND six_icd_diagnosis_code != ''
          AND {event_filter}
        UNION ALL
        SELECT mi_person_key, seven_icd_diagnosis_code AS code
        FROM model_events
        WHERE seven_icd_diagnosis_code IS NOT NULL
          AND seven_icd_diagnosis_code != ''
          AND {event_filter}
        UNION ALL
        SELECT mi_person_key, eight_icd_diagnosis_code AS code
        FROM model_events
        WHERE eight_icd_diagnosis_code IS NOT NULL
          AND eight_icd_diagnosis_code != ''
          AND {event_filter}
        UNION ALL
        SELECT mi_person_key, nine_icd_diagnosis_code AS code
        FROM model_events
        WHERE nine_icd_diagnosis_code IS NOT NULL
          AND nine_icd_diagnosis_code != ''
          AND {event_filter}
        UNION ALL
        SELECT mi_person_key, ten_icd_diagnosis_code AS code
        FROM model_events
        WHERE ten_icd_diagnosis_code IS NOT NULL
          AND ten_icd_diagnosis_code != ''
          AND {event_filter}
        UNION ALL
        SELECT mi_person_key, procedure_code AS code
        FROM model_events
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

    con2.execute(f"CREATE TABLE events AS SELECT * FROM {from_sql}")

    # Extreme cohort events (same 4_model_data root as source, e.g. PGX_DATA_ROOT on EC2)
    _effective_root = model_events_path.parent.parent.parent
    extreme_events_path = (
        _effective_root
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

    # Filtered (non-extreme) events replace the original model_events (or write new 85-114 partition)
    model_events_path.parent.mkdir(parents=True, exist_ok=True)
    backup_path = model_events_path.with_name("model_events_with_extreme.parquet")
    if model_events_path.exists() and not backup_path.exists():
        model_events_path.replace(backup_path)
        logger.info("Backed up original model_events to %s", backup_path)
    elif model_events_path.exists():
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
            "patients from the source 4_model_data event set."
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
    logger, log_path = setup_logger(
        "extract_extreme_density",
        cohort_name=args.cohort_name,
        age_band=args.age_band,
    )
    try:
        extract_extreme_density_cohort(
            cohort_name=args.cohort_name,
            age_band=args.age_band,
            output_cohort_name=args.output_cohort_name,
            logger=logger,
        )
    finally:
        if mirror_log_to_s3 and log_path and log_path.exists():
            mirror_log_to_s3(
                "extreme_density_extract",
                args.cohort_name,
                args.age_band,
                log_path,
                logger,
            )


if __name__ == "__main__":
    main()
