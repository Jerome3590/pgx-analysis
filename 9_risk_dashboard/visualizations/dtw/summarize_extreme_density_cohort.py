#!/usr/bin/env python3
"""
Summarize an extreme-density cohort created by extract_extreme_density_cohort.py.

For a given cohort / age_band (typically {source}_extreme_density), this script:
- Loads model_events.parquet for the cohort.
- Computes aggregate stats:
  - number of patients
  - total events and events by event_type
  - target prevalence at the patient level
  - distribution of medical transaction sizes (combined ICD + CPT, TRAIN years).
- Creates a patient-level summary CSV for visualization / exploration with:
  - mi_person_key
  - target (patient-level)
  - n_events_total
  - n_events_pharmacy
  - n_events_medical
  - transaction_size_medical (TRAIN years, combined ICD + CPT)
  - admin_icd_event_count, routine_admin (Routine vs no routine from administrative_codes_lookup.json)
- Writes histograms comparing routine admin vs no routine admin (transaction size distribution and outcome rate).
- Writes a JSON file with aggregate summary statistics.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional, Set, Tuple

import duckdb
import numpy as np
import matplotlib.pyplot as plt

# Repo root so 4_model_data and py_helpers are available (same as other dtw scripts)
REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from py_helpers.fe_monitor import mirror_log_to_s3
except ImportError:
    mirror_log_to_s3 = None

MODEL_DATA_ROOT = REPO_ROOT / "4_model_data"
TRAIN_YEARS = [2016, 2017, 2018]

# ICD diagnosis columns in model_events (must match 4_model_data / create_dtw_features)
ICD_DIAGNOSIS_COLUMNS = [
    "primary_icd_diagnosis_code",
    "two_icd_diagnosis_code",
    "three_icd_diagnosis_code",
    "four_icd_diagnosis_code",
    "five_icd_diagnosis_code",
    "six_icd_diagnosis_code",
    "seven_icd_diagnosis_code",
    "eight_icd_diagnosis_code",
    "nine_icd_diagnosis_code",
    "ten_icd_diagnosis_code",
]


def _load_administrative_icd_codes(project_root: Path) -> Set[str]:
    """Load administrative ICD codes from 1b_apcd_event_filter/administrative_codes_lookup.json."""
    path = project_root / "1b_apcd_event_filter" / "administrative_codes_lookup.json"
    if not path.exists():
        return set()
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        codes = data.get("administrative_codes", {}).get("icd", [])
        return set(str(c) for c in codes)
    except Exception:
        return set()


def setup_logger(
    name: str = "summarize_extreme_density",
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
            logs_dir = REPO_ROOT / "logs" / "feature_engineering" / "extreme_density_summarize"
            logs_dir.mkdir(parents=True, exist_ok=True)
            age_band_fname = age_band.replace("-", "_")
            log_path = logs_dir / f"summarize_extreme_density_{cohort_name}_{age_band_fname}.log"
            file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
    return logger, log_path


def _get_model_events_path(cohort_name: str, age_band: str) -> Path:
    return (
        MODEL_DATA_ROOT
        / f"cohort_name={cohort_name}"
        / f"age_band={age_band}"
        / "model_events.parquet"
    )


def summarize_extreme_density_cohort(
    cohort_name: str,
    age_band: str,
    logger: Optional[logging.Logger] = None,
) -> None:
    if logger is None:
        logger, _ = setup_logger()
    logger.info("Summarizing extreme-density cohort %s / %s", cohort_name, age_band)

    model_events_path = _get_model_events_path(cohort_name, age_band)
    if not model_events_path.exists():
        logger.error("model_events not found at %s", model_events_path)
        raise SystemExit(1)

    age_band_fname = age_band.replace("-", "_")
    model_events_str = str(model_events_path).replace("\\", "/")

    con = duckdb.connect(database=":memory:")

    # ------------------------------------------------------------------
    # Aggregate stats: patients, events, event_type breakdown
    # ------------------------------------------------------------------
    logger.info("Computing aggregate event statistics...")
    agg_df = con.execute(
        """
        SELECT
            COUNT(DISTINCT mi_person_key) AS n_patients,
            COUNT(*) AS n_events
        FROM read_parquet(?)
        """,
        [model_events_str],
    ).df()

    n_patients = int(agg_df.iloc[0]["n_patients"])
    n_events = int(agg_df.iloc[0]["n_events"])
    logger.info("Patients: %s", f"{n_patients:,}")
    logger.info("Events: %s", f"{n_events:,}")

    event_type_df = con.execute(
        """
        SELECT event_type, COUNT(*) AS n_events
        FROM read_parquet(?)
        GROUP BY event_type
        ORDER BY n_events DESC
        """,
        [model_events_str],
    ).df()

    for _, row in event_type_df.iterrows():
        logger.info(
            "  event_type=%s n_events=%s",
            row["event_type"],
            f"{int(row['n_events']):,}",
        )

    # ------------------------------------------------------------------
    # Target prevalence at patient level
    # ------------------------------------------------------------------
    logger.info("Computing patient-level target prevalence...")
    target_df = con.execute(
        """
        WITH per_patient AS (
            SELECT
                mi_person_key,
                MAX(target) AS target
            FROM read_parquet(?)
            GROUP BY mi_person_key
        )
        SELECT target, COUNT(*) AS n_patients
        FROM per_patient
        GROUP BY target
        ORDER BY target
        """,
        [model_events_str],
    ).df()

    target_dist = {}
    for _, row in target_df.iterrows():
        t_val = int(row["target"])
        count = int(row["n_patients"])
        target_dist[str(t_val)] = count
        logger.info(
            "  target=%s n_patients=%s (%.1f%%)",
            t_val,
            f"{count:,}",
            (count / n_patients) * 100 if n_patients > 0 else 0.0,
        )

    # ------------------------------------------------------------------
    # Medical transaction sizes (TRAIN years, combined ICD + CPT)
    # ------------------------------------------------------------------
    years_sql = ", ".join(str(y) for y in TRAIN_YEARS)
    event_filter = f"event_year IN ({years_sql})"

    logger.info("Computing medical transaction sizes (TRAIN years, ICD + CPT)...")
    query_medical = f"""
    WITH all_med_codes AS (
        SELECT mi_person_key, primary_icd_diagnosis_code AS code
        FROM read_parquet('{model_events_str}')
        WHERE primary_icd_diagnosis_code IS NOT NULL
          AND primary_icd_diagnosis_code != ''
          AND event_type = 'medical'
          AND {event_filter}
        UNION ALL
        SELECT mi_person_key, two_icd_diagnosis_code AS code
        FROM read_parquet('{model_events_str}')
        WHERE two_icd_diagnosis_code IS NOT NULL
          AND two_icd_diagnosis_code != ''
          AND event_type = 'medical'
          AND {event_filter}
        UNION ALL
        SELECT mi_person_key, three_icd_diagnosis_code AS code
        FROM read_parquet('{model_events_str}')
        WHERE three_icd_diagnosis_code IS NOT NULL
          AND three_icd_diagnosis_code != ''
          AND event_type = 'medical'
          AND {event_filter}
        UNION ALL
        SELECT mi_person_key, four_icd_diagnosis_code AS code
        FROM read_parquet('{model_events_str}')
        WHERE four_icd_diagnosis_code IS NOT NULL
          AND four_icd_diagnosis_code != ''
          AND event_type = 'medical'
          AND {event_filter}
        UNION ALL
        SELECT mi_person_key, five_icd_diagnosis_code AS code
        FROM read_parquet('{model_events_str}')
        WHERE five_icd_diagnosis_code IS NOT NULL
          AND five_icd_diagnosis_code != ''
          AND event_type = 'medical'
          AND {event_filter}
        UNION ALL
        SELECT mi_person_key, six_icd_diagnosis_code AS code
        FROM read_parquet('{model_events_str}')
        WHERE six_icd_diagnosis_code IS NOT NULL
          AND six_icd_diagnosis_code != ''
          AND event_type = 'medical'
          AND {event_filter}
        UNION ALL
        SELECT mi_person_key, seven_icd_diagnosis_code AS code
        FROM read_parquet('{model_events_str}')
        WHERE seven_icd_diagnosis_code IS NOT NULL
          AND seven_icd_diagnosis_code != ''
          AND event_type = 'medical'
          AND {event_filter}
        UNION ALL
        SELECT mi_person_key, eight_icd_diagnosis_code AS code
        FROM read_parquet('{model_events_str}')
        WHERE eight_icd_diagnosis_code IS NOT NULL
          AND eight_icd_diagnosis_code != ''
          AND event_type = 'medical'
          AND {event_filter}
        UNION ALL
        SELECT mi_person_key, nine_icd_diagnosis_code AS code
        FROM read_parquet('{model_events_str}')
        WHERE nine_icd_diagnosis_code IS NOT NULL
          AND nine_icd_diagnosis_code != ''
          AND event_type = 'medical'
          AND {event_filter}
        UNION ALL
        SELECT mi_person_key, ten_icd_diagnosis_code AS code
        FROM read_parquet('{model_events_str}')
        WHERE ten_icd_diagnosis_code IS NOT NULL
          AND ten_icd_diagnosis_code != ''
          AND event_type = 'medical'
          AND {event_filter}
        UNION ALL
        SELECT mi_person_key, procedure_code AS code
        FROM read_parquet('{model_events_str}')
        WHERE procedure_code IS NOT NULL
          AND procedure_code != ''
          AND event_type = 'medical'
          AND {event_filter}
    )
    SELECT mi_person_key, COUNT(*) AS transaction_size_medical
    FROM all_med_codes
    GROUP BY mi_person_key
    """

    tx_df = con.execute(query_medical).df()

    if not tx_df.empty:
        sizes = tx_df["transaction_size_medical"].values
        tx_stats = {
            "min": int(np.min(sizes)),
            "max": int(np.max(sizes)),
            "p25": float(np.percentile(sizes, 25)),
            "p50": float(np.percentile(sizes, 50)),
            "p75": float(np.percentile(sizes, 75)),
            "p95": float(np.percentile(sizes, 95)),
            "mean": float(np.mean(sizes)),
        }
        logger.info("Medical transaction_size_medical stats (TRAIN years):")
        logger.info("  min=%d max=%d mean=%.1f", tx_stats["min"], tx_stats["max"], tx_stats["mean"])
        logger.info(
            "  p25=%.1f p50=%.1f p75=%.1f p95=%.1f",
            tx_stats["p25"],
            tx_stats["p50"],
            tx_stats["p75"],
            tx_stats["p95"],
        )
    else:
        tx_stats = {}
        logger.info("No medical codes found for TRAIN years; transaction_size_medical stats empty.")

    # ------------------------------------------------------------------
    # Patient-level summary table
    # ------------------------------------------------------------------
    logger.info("Building patient-level summary table...")
    patient_events_df = con.execute(
        """
        SELECT
            mi_person_key,
            MAX(target) AS target,
            COUNT(*) AS n_events_total,
            SUM(CASE WHEN event_type = 'pharmacy' THEN 1 ELSE 0 END) AS n_events_pharmacy,
            SUM(CASE WHEN event_type = 'medical' THEN 1 ELSE 0 END) AS n_events_medical
        FROM read_parquet(?)
        GROUP BY mi_person_key
        """,
        [model_events_str],
    ).df()

    # Left join transaction sizes (may be missing for some if no TRAIN-year medical codes)
    patient_summary_df = patient_events_df.merge(
        tx_df, on="mi_person_key", how="left"
    )

    # ------------------------------------------------------------------
    # Admin ICD (routine appointments) count per patient for routine vs no routine
    # ------------------------------------------------------------------
    admin_icd = _load_administrative_icd_codes(REPO_ROOT)
    if admin_icd:
        # Escape single quotes for SQL; build IN list
        admin_list = ", ".join("'" + str(c).replace("'", "''") + "'" for c in sorted(admin_icd))
        icd_conditions = " OR ".join(f"{col} IN ({admin_list})" for col in ICD_DIAGNOSIS_COLUMNS)
        path_str = model_events_str.replace("\\", "/")
        try:
            admin_df = con.execute(
                f"""
                WITH events_with_admin_icd AS (
                    SELECT mi_person_key
                    FROM read_parquet('{path_str}')
                    WHERE {icd_conditions}
                )
                SELECT mi_person_key, COUNT(*)::INTEGER AS admin_icd_event_count
                FROM events_with_admin_icd
                GROUP BY mi_person_key
                """
            ).df()
            patient_summary_df = patient_summary_df.merge(
                admin_df, on="mi_person_key", how="left"
            )
            patient_summary_df["admin_icd_event_count"] = patient_summary_df["admin_icd_event_count"].fillna(0).astype(int)
            patient_summary_df["routine_admin"] = patient_summary_df["admin_icd_event_count"].apply(
                lambda x: "Routine (1+ admin ICD)" if x >= 1 else "No routine (0 admin ICD)"
            )
            logger.info(
                "Routine vs no routine: %s with routine, %s without",
                (patient_summary_df["routine_admin"] == "Routine (1+ admin ICD)").sum(),
                (patient_summary_df["routine_admin"] == "No routine (0 admin ICD)").sum(),
            )
        except Exception as exc:
            logger.warning("Could not compute admin ICD counts: %s", exc)
            admin_icd = set()
    if not admin_icd or "routine_admin" not in patient_summary_df.columns:
        patient_summary_df["admin_icd_event_count"] = 0
        patient_summary_df["routine_admin"] = "Unknown (no admin codes lookup)"

    out_dir = model_events_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    patient_csv_path = out_dir / f"extreme_density_patient_summary_{age_band_fname}.csv"
    patient_summary_df.to_csv(patient_csv_path, index=False)
    logger.info("Wrote patient-level summary CSV to %s", patient_csv_path)

    # ------------------------------------------------------------------
    # Drug / ICD / CPT frequency tables and plots
    # ------------------------------------------------------------------
    logger.info("Computing drug_name frequency for extreme cohort...")
    drug_freq_df = con.execute(
        """
        SELECT
            drug_name,
            COUNT(*) AS n_events
        FROM read_parquet(?)
        WHERE event_type = 'pharmacy'
          AND drug_name IS NOT NULL
          AND drug_name != ''
        GROUP BY drug_name
        ORDER BY n_events DESC
        """,
        [model_events_str],
    ).df()

    drug_freq_path = out_dir / f"extreme_density_drug_frequency_{age_band_fname}.csv"
    drug_freq_df.to_csv(drug_freq_path, index=False)
    logger.info("Wrote drug_name frequency table to %s", drug_freq_path)

    if not drug_freq_df.empty:
        top_n = min(30, len(drug_freq_df))
        top_drugs = drug_freq_df.head(top_n)
        plt.figure(figsize=(10, 8))
        plt.barh(top_drugs["drug_name"][::-1], top_drugs["n_events"][::-1])
        plt.xlabel("Number of events")
        plt.title(f"Top {top_n} drugs (extreme-density cohort {cohort_name} {age_band})")
        plt.tight_layout()
        drug_plot_path = out_dir / f"extreme_density_drug_frequency_top_{age_band_fname}.png"
        plt.savefig(drug_plot_path, dpi=200)
        plt.close()
        logger.info("Wrote drug_name frequency plot to %s", drug_plot_path)

    logger.info("Computing ICD code frequency for extreme cohort...")
    icd_freq_query = f"""
    WITH all_icds AS (
        SELECT primary_icd_diagnosis_code AS icd
        FROM read_parquet('{model_events_str}')
        WHERE primary_icd_diagnosis_code IS NOT NULL
          AND primary_icd_diagnosis_code != ''
          AND event_type = 'medical'
        UNION ALL
        SELECT two_icd_diagnosis_code AS icd
        FROM read_parquet('{model_events_str}')
        WHERE two_icd_diagnosis_code IS NOT NULL
          AND two_icd_diagnosis_code != ''
          AND event_type = 'medical'
        UNION ALL
        SELECT three_icd_diagnosis_code AS icd
        FROM read_parquet('{model_events_str}')
        WHERE three_icd_diagnosis_code IS NOT NULL
          AND three_icd_diagnosis_code != ''
          AND event_type = 'medical'
        UNION ALL
        SELECT four_icd_diagnosis_code AS icd
        FROM read_parquet('{model_events_str}')
        WHERE four_icd_diagnosis_code IS NOT NULL
          AND four_icd_diagnosis_code != ''
          AND event_type = 'medical'
        UNION ALL
        SELECT five_icd_diagnosis_code AS icd
        FROM read_parquet('{model_events_str}')
        WHERE five_icd_diagnosis_code IS NOT NULL
          AND five_icd_diagnosis_code != ''
          AND event_type = 'medical'
        UNION ALL
        SELECT six_icd_diagnosis_code AS icd
        FROM read_parquet('{model_events_str}')
        WHERE six_icd_diagnosis_code IS NOT NULL
          AND six_icd_diagnosis_code != ''
          AND event_type = 'medical'
        UNION ALL
        SELECT seven_icd_diagnosis_code AS icd
        FROM read_parquet('{model_events_str}')
        WHERE seven_icd_diagnosis_code IS NOT NULL
          AND seven_icd_diagnosis_code != ''
          AND event_type = 'medical'
        UNION ALL
        SELECT eight_icd_diagnosis_code AS icd
        FROM read_parquet('{model_events_str}')
        WHERE eight_icd_diagnosis_code IS NOT NULL
          AND eight_icd_diagnosis_code != ''
          AND event_type = 'medical'
        UNION ALL
        SELECT nine_icd_diagnosis_code AS icd
        FROM read_parquet('{model_events_str}')
        WHERE nine_icd_diagnosis_code IS NOT NULL
          AND nine_icd_diagnosis_code != ''
          AND event_type = 'medical'
        UNION ALL
        SELECT ten_icd_diagnosis_code AS icd
        FROM read_parquet('{model_events_str}')
        WHERE ten_icd_diagnosis_code IS NOT NULL
          AND ten_icd_diagnosis_code != ''
          AND event_type = 'medical'
    )
    SELECT icd, COUNT(*) AS n_events
    FROM all_icds
    GROUP BY icd
    ORDER BY n_events DESC
    """
    icd_freq_df = con.execute(icd_freq_query).df()

    icd_freq_path = out_dir / f"extreme_density_icd_frequency_{age_band_fname}.csv"
    icd_freq_df.to_csv(icd_freq_path, index=False)
    logger.info("Wrote ICD frequency table to %s", icd_freq_path)

    if not icd_freq_df.empty:
        top_n = min(30, len(icd_freq_df))
        top_icd = icd_freq_df.head(top_n)
        plt.figure(figsize=(10, 8))
        plt.barh(top_icd["icd"][::-1], top_icd["n_events"][::-1])
        plt.xlabel("Number of events")
        plt.title(f"Top {top_n} ICD codes (extreme-density cohort {cohort_name} {age_band})")
        plt.tight_layout()
        icd_plot_path = out_dir / f"extreme_density_icd_frequency_top_{age_band_fname}.png"
        plt.savefig(icd_plot_path, dpi=200)
        plt.close()
        logger.info("Wrote ICD frequency plot to %s", icd_plot_path)

    logger.info("Computing CPT code frequency for extreme cohort...")
    cpt_freq_df = con.execute(
        """
        SELECT
            procedure_code AS cpt,
            COUNT(*) AS n_events
        FROM read_parquet(?)
        WHERE event_type = 'medical'
          AND procedure_code IS NOT NULL
          AND procedure_code != ''
        GROUP BY procedure_code
        ORDER BY n_events DESC
        """,
        [model_events_str],
    ).df()

    cpt_freq_path = out_dir / f"extreme_density_cpt_frequency_{age_band_fname}.csv"
    cpt_freq_df.to_csv(cpt_freq_path, index=False)
    logger.info("Wrote CPT frequency table to %s", cpt_freq_path)

    if not cpt_freq_df.empty:
        top_n = min(30, len(cpt_freq_df))
        top_cpt = cpt_freq_df.head(top_n)
        plt.figure(figsize=(10, 8))
        plt.barh(top_cpt["cpt"][::-1], top_cpt["n_events"][::-1])
        plt.xlabel("Number of events")
        plt.title(f"Top {top_n} CPT codes (extreme-density cohort {cohort_name} {age_band})")
        plt.tight_layout()
        cpt_plot_path = out_dir / f"extreme_density_cpt_frequency_top_{age_band_fname}.png"
        plt.savefig(cpt_plot_path, dpi=200)
        plt.close()
        logger.info("Wrote CPT frequency plot to %s", cpt_plot_path)

    # ------------------------------------------------------------------
    # Histogram of medical transaction size (per patient)
    # ------------------------------------------------------------------
    if "transaction_size_medical" in patient_summary_df.columns:
        logger.info("Creating histogram of transaction_size_medical...")
        tx_series = patient_summary_df["transaction_size_medical"].dropna()
        if not tx_series.empty:
            plt.figure(figsize=(8, 6))
            plt.hist(tx_series, bins=40, edgecolor="black")
            plt.xlabel("transaction_size_medical (TRAIN years, ICD + CPT)")
            plt.ylabel("Number of patients")
            plt.title(
                f"Distribution of medical transaction size (extreme-density cohort {cohort_name} {age_band})"
            )
            plt.tight_layout()
            hist_path = out_dir / f"extreme_density_transaction_size_hist_{age_band_fname}.png"
            plt.savefig(hist_path, dpi=200)
            plt.close()
            logger.info("Wrote transaction_size_medical histogram to %s", hist_path)
        else:
            logger.info("transaction_size_medical is empty; skipping histogram.")

    # ------------------------------------------------------------------
    # Routine vs no routine: histograms and outcome comparison
    # ------------------------------------------------------------------
    if "routine_admin" in patient_summary_df.columns and patient_summary_df["routine_admin"].nunique() >= 2:
        routine_labels = ["No routine (0 admin ICD)", "Routine (1+ admin ICD)"]
        has_routine = patient_summary_df["routine_admin"].isin(routine_labels)
        df_routine = patient_summary_df[has_routine]
        if not df_routine.empty and "transaction_size_medical" in df_routine.columns:
            tx_col = df_routine["transaction_size_medical"].dropna()
            if not tx_col.empty:
                no_routine_vals = df_routine.loc[df_routine["routine_admin"] == "No routine (0 admin ICD)", "transaction_size_medical"].dropna()
                routine_vals = df_routine.loc[df_routine["routine_admin"] == "Routine (1+ admin ICD)", "transaction_size_medical"].dropna()
                plt.figure(figsize=(9, 6))
                if not no_routine_vals.empty:
                    plt.hist(no_routine_vals, bins=30, alpha=0.6, label="No routine (0 admin ICD)", color="tab:orange", edgecolor="black")
                if not routine_vals.empty:
                    plt.hist(routine_vals, bins=30, alpha=0.6, label="Routine (1+ admin ICD)", color="tab:blue", edgecolor="black")
                plt.xlabel("transaction_size_medical (TRAIN years, ICD + CPT)")
                plt.ylabel("Number of patients")
                plt.title(f"Extreme-density cohort {cohort_name} {age_band}: transaction size by routine vs no routine admin")
                plt.legend()
                plt.tight_layout()
                routine_hist_path = out_dir / f"extreme_density_routine_vs_no_routine_hist_{age_band_fname}.png"
                plt.savefig(routine_hist_path, dpi=200)
                plt.close()
                logger.info("Wrote routine vs no routine histogram to %s", routine_hist_path)

        # Outcome rate by routine vs no routine
        if "target" in df_routine.columns and not df_routine.empty:
            outcome_by_routine = df_routine.groupby("routine_admin", observed=True).agg(
                outcome_rate=("target", "mean"),
                n_patients=("mi_person_key", "count"),
            )
            if len(outcome_by_routine) >= 1:
                plt.figure(figsize=(7, 5))
                x_pos = range(len(outcome_by_routine))
                colors = ["tab:orange", "tab:blue"] if len(outcome_by_routine) == 2 else ["tab:blue"]
                plt.bar(x_pos, outcome_by_routine["outcome_rate"], color=colors[: len(outcome_by_routine)], edgecolor="black")
                plt.xticks(x_pos, outcome_by_routine.index.tolist(), rotation=15, ha="right")
                plt.ylabel("Outcome rate (mean target)")
                plt.xlabel("Routine vs no routine (admin ICD)")
                plt.title(f"Extreme-density cohort {cohort_name} {age_band}: outcome rate by routine admin\n(Lower in Routine group = routine visits associated with fewer extreme events)")
                for i, (_, row) in enumerate(outcome_by_routine.iterrows()):
                    plt.text(i, float(row["outcome_rate"]) + 0.01, f"n={int(row['n_patients'])}", ha="center", fontsize=9)
                plt.tight_layout()
                routine_outcome_path = out_dir / f"extreme_density_routine_vs_no_routine_outcome_{age_band_fname}.png"
                plt.savefig(routine_outcome_path, dpi=200)
                plt.close()
                logger.info("Wrote routine vs no routine outcome bar to %s", routine_outcome_path)
    else:
        logger.info("Skipping routine vs no routine histograms (single group or no admin codes).")

    # ------------------------------------------------------------------
    # Aggregate JSON summary
    # ------------------------------------------------------------------
    summary = {
        "cohort_name": cohort_name,
        "age_band": age_band,
        "n_patients": n_patients,
        "n_events": n_events,
        "event_type_breakdown": {
            str(row["event_type"]): int(row["n_events"])
            for _, row in event_type_df.iterrows()
        },
        "target_distribution": target_dist,
        "medical_transaction_size_stats": tx_stats,
        "train_years": TRAIN_YEARS,
    }
    if "routine_admin" in patient_summary_df.columns:
        routine_counts = patient_summary_df["routine_admin"].value_counts()
        summary["routine_admin_counts"] = {str(k): int(v) for k, v in routine_counts.items()}

    summary_path = out_dir / f"extreme_density_summary_{age_band_fname}.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    logger.info("Wrote aggregate summary JSON to %s", summary_path)

    con.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize an extreme-density cohort for visualization and exploration."
    )
    parser.add_argument(
        "--cohort-name",
        required=True,
        help="Extreme-density cohort name (e.g., opioid_ed_extreme_density)",
    )
    parser.add_argument(
        "--age-band",
        required=True,
        help="Age band (e.g., 25-44)",
    )
    args = parser.parse_args()
    logger, log_path = setup_logger(
        "summarize_extreme_density",
        cohort_name=args.cohort_name,
        age_band=args.age_band,
    )
    try:
        summarize_extreme_density_cohort(
            cohort_name=args.cohort_name,
            age_band=args.age_band,
            logger=logger,
        )
    finally:
        if mirror_log_to_s3 and log_path and log_path.exists():
            mirror_log_to_s3(
                "extreme_density_summarize",
                args.cohort_name,
                args.age_band,
                log_path,
                logger,
            )


if __name__ == "__main__":
    main()
