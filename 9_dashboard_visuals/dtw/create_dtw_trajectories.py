#!/usr/bin/env python3
"""
Create trajectory data for DTW alignment and dashboard visualizations (Step 1 of DTW workflow).

This script extracts patient trajectories from model_data for DTW alignment and visualization.
Part 1 of DTW pipeline: trajectories → DTW alignment (create_dtw_features.py) → visuals.
NOT used for model training - for dashboard visual analysis of SHAP/FFA results.

Output CSV columns (minimal for visualization):
- mi_person_key: Patient identifier
- target: Target outcome (0/1)
- seq_pattern_str: Event-ordered sequence of activity codes (e.g., "DRUG:Med_ICD:F1120_CPT:99213")
- seq_pattern_monthly: Calendar-month sequence; one token per month "YYYY-MM:CODE1_CODE2|..." (codes sorted
  alphabetically within month). Both cohorts = drug only (DRUG: tokens).
- admin_icd_event_count: Count of events with administrative ICD codes (used to identify routine appointments: 1+ = routine, 0 = no routine)
- trajectory_length: Number of events
- trajectory_diversity: Number of unique activities
- dtw_min_distance: Placeholder (NaN); DTW distances computed in create_dtw_features.py (Step 2)
- mean_days_between_events: Mean days between consecutive events in the trajectory (N3: times between sequences)
- days_first_event_to_target: For target=1, days from first event to target date; else NaN (N3)
- temporal_span_days: Days between first and last event in trajectory (0 if single event)
- events_per_month: trajectory_length / (temporal_span_days/30) when span > 0, else NaN
- event_density_bin: Trajectory bin by event density ('low', 'medium', 'high', 'extreme'), aligned with FP-Growth

Model data (parquet) columns needed for N3 time-between metrics:
- Event timestamp: one of event_date, incurred_date, service_date, event_timestamp, claim_date, event_dt (first present in schema is used for ordering and for mean_days_between_events / days_first_event_to_target).
- target (0/1), mi_person_key, drug_name, primary_icd_diagnosis_code, procedure_code.
- Target date column: first_f1120_date (opioid_ed) or first_o11_p_date (non_opioid_ed) or legacy names; used for "days to target" and lookback window.

Requirements:
- 4_model_data (Step 4) with model_events parquet
- 7_shap_analysis and 8_ffa_analysis (Steps 7-8) for SHAP/FFA important codes
- 1b_apcd_event_filter/administrative_codes_lookup.json — administrative ICD codes that identify routine appointments (e.g. well visits, screenings)

Runtime: ~1-2 minutes per cohort/age_band (fast!)
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional, Set, Tuple

import duckdb
import pandas as pd

# Repo root and step folder (9_dashboard_visuals)
REPO_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from py_helpers.fe_monitor import step_block  # noqa: E402
from py_helpers.model_data_paths import (  # noqa: E402
    resolve_model_events_path,
    resolve_model_events_paths,
)
from py_helpers.pipeline_logger import (  # noqa: E402
    setup_pipeline_logger,
    log_step_start,
    log_step_complete,
    PipelineLogger,
)

from py_helpers.duckdb_utils import duckdb_query_df_with_diagnostics  # noqa: E402


def _dtw_output_root(project_root: Path) -> Path:
    """Dashboard visualization outputs (step 10); creation code in 9_dashboard_visuals/dtw."""
    return project_root / "10_risk_dashboard" / "visualizations" / "dtw"


# Expected CSV columns when writing placeholder (empty) output so downstream steps see the file and skip
DTW_TRAJECTORY_CSV_COLUMNS = [
    "mi_person_key",
    "target",
    "seq_pattern_str",
    "seq_pattern_monthly",
    "admin_icd_event_count",
    "medical_event_count_full",
    "medical_utilization_bin",
    "trajectory_length",
    "trajectory_diversity",
    "dtw_min_distance",
    "mean_days_between_events",
    "days_first_event_to_target",
    "temporal_span_days",
    "events_per_month",
    "event_density_bin",
]

# DTW is drug-only for both cohorts (opioid_ed and non_opioid_ed).
POLYPHARMACY_COHORT = "non_opioid_ed"

# Density bins for trajectories (same order as FP-Growth: low -> medium -> high -> extreme)
DENSITY_BINS = ("low", "medium", "high", "extreme")


def _normalize_code_for_match(code: str) -> str:
    """Normalize code for set membership (e.g. F11.20 and F1120 match)."""
    if not code or (isinstance(code, float) and pd.isna(code)):
        return ""
    s = str(code).strip()
    return s.replace(".", "").replace("-", "")


def _split_allowed_codes_by_type(allowed_codes: Set[str]) -> Tuple[Set[str], Set[str], Set[str]]:
    """
    Split SHAP/FFA allowed codes into drug, ICD, and CPT sets using raw (stripped) codes.
    Handles prefixed codes (cpt_01967, icd_F1120, drug_XYZ) and unprefixed fallback.
    """
    drug_set: Set[str] = set()
    icd_set: Set[str] = set()
    cpt_set: Set[str] = set()

    try:
        from py_helpers.shap_ffa_fpgrowth_utils import _parse_feature_name
    except ImportError:
        _parse_feature_name = None

    for c in allowed_codes:
        if not c or (isinstance(c, float) and pd.isna(c)):
            continue
        s = str(c).strip()
        norm = _normalize_code_for_match(s)
        if not norm:
            continue

        if s.startswith("cpt_"):
            cpt_set.add(_normalize_code_for_match(s[4:]))
        elif s.startswith("icd_"):
            icd_set.add(_normalize_code_for_match(s[4:]))
        elif s.startswith("drug_"):
            drug_set.add(_normalize_code_for_match(s[5:]))
        elif _parse_feature_name:
            _type, code = _parse_feature_name(s)
            raw_norm = _normalize_code_for_match(code) if code else norm
            if _type == "cpt":
                cpt_set.add(raw_norm)
            elif _type == "icd":
                icd_set.add(raw_norm)
            elif _type == "drug":
                drug_set.add(raw_norm)
            else:
                # Unknown: add to all three
                drug_set.add(norm)
                icd_set.add(norm)
                cpt_set.add(norm)
        else:
            # No parser: add to all three
            drug_set.add(norm)
            icd_set.add(norm)
            cpt_set.add(norm)

    return drug_set, icd_set, cpt_set


def _load_administrative_icd_codes(project_root: Path) -> Set[str]:
    """Load administrative ICD codes used to identify routine appointments (well visits, screenings) from 1b_apcd_event_filter/administrative_codes_lookup.json."""
    path = project_root / "1b_apcd_event_filter" / "administrative_codes_lookup.json"
    if not path.exists():
        print(f"[WARN] Administrative codes lookup not found at {path}")
        return set()

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        codes = data.get("administrative_codes", {}).get("icd", [])
        # Normalize for matching parquet (dots/dashes stripped)
        return set(_normalize_code_for_match(str(c)) for c in codes if c is not None)
    except Exception as exc:
        print(f"[WARN] Could not load administrative codes: {exc}")
        return set()


def extract_patient_trajectories(
    project_root: Path,
    cohort_name: str,
    age_band: str,
    max_lookback_months: int = 24,
    logger: Optional[PipelineLogger] = None,
) -> Tuple[pd.DataFrame, int]:
    """
    Extract patient trajectories from model_data for visualization.

    Returns (DataFrame, n_events_analyzed). DataFrame has columns:
    - mi_person_key, target, seq_pattern_str, admin_icd_event_count,
    - trajectory_length, trajectory_diversity, ...
    n_events_analyzed is the count of events (rows) that matched the filter and lookback.
    """
    def _log(level: str, msg: str, *args: object) -> None:
        if logger is not None:
            getattr(logger, level)(msg, *args)
        else:
            print("[%s] " % level.upper() + (msg % args if args else msg))

    _log("info", "Extracting trajectories for %s / %s", cohort_name, age_band)

    # Get model_events path
    try:
        model_data_path = resolve_model_events_path(project_root, cohort_name, age_band)
    except Exception:
        model_data_path = None

    if not model_data_path or not model_data_path.exists():
        age_band_fname = age_band.replace("-", "_")
        model_data_dir = project_root / "4_model_data" / cohort_name / age_band_fname
        model_data_path = model_data_dir / "model_events.parquet"

    if not model_data_path.exists():
        _log("error", "Model data not found at %s", model_data_path)
        return pd.DataFrame(), 0

    _log("info", "Using model_events: %s", model_data_path)

    # SHAP/FFA combined allowed codes file is required (same prerequisite as BupaR); we never use all events.
    # For extreme-density cohorts (e.g. non_opioid_ed_extreme_density), use the base cohort's allowed_codes.
    age_band_fname = age_band.replace("-", "_")
    bupar_output_root = project_root / "10_risk_dashboard" / "visualizations" / "bupar"
    allowed_codes_path = bupar_output_root / f"allowed_codes_shap_ffa_{cohort_name}_{age_band_fname}.json"
    if not allowed_codes_path.exists() and cohort_name.endswith("_extreme_density"):
        base_cohort = cohort_name.replace("_extreme_density", "")
        fallback_path = bupar_output_root / f"allowed_codes_shap_ffa_{base_cohort}_{age_band_fname}.json"
        if fallback_path.exists():
            allowed_codes_path = fallback_path
            _log("info", "Using base cohort allowed_codes for extreme: %s", allowed_codes_path)
    if not allowed_codes_path.exists():
        print(
            f"[ERROR] SHAP/FFA allowed codes file is required (prerequisite). Not found: {allowed_codes_path}\n"
            "  Generate the combined allowed_codes file before running DTW (same as BupaR)."
        )
        raise SystemExit(1)
    with open(allowed_codes_path, encoding="utf-8") as f:
        allowed_codes_list = json.load(f)
    allowed_codes = {str(c).strip() for c in allowed_codes_list if c is not None and str(c).strip()}
    if not allowed_codes:
        print(
            f"[ERROR] SHAP/FFA allowed codes file is empty: {allowed_codes_path}\n"
            "  Cannot run DTW without allowed codes."
        )
        raise SystemExit(1)
    _log("info", "Filtering to %d SHAP/FFA important codes (from combined file)", len(allowed_codes))
    drug_set, icd_set, cpt_set = _split_allowed_codes_by_type(allowed_codes)
    use_filter = True

    # Administrative ICD codes identify routine appointments (1+ events with these codes = routine)
    admin_codes = _load_administrative_icd_codes(project_root)
    _log("info", "Loaded %d administrative ICD codes", len(admin_codes))

    # Resolve target date column from parquet schema: use first candidate that exists (alias to target_date in query)
    path_str = str(model_data_path).replace("'", "''")
    con_schema = duckdb.connect(":memory:")
    schema = con_schema.execute(
        f"DESCRIBE SELECT * FROM read_parquet('{path_str}')"
    ).fetchall()
    col_names = {row[0] for row in schema}
    con_schema.close()

    # Ordered candidates per cohort (canonical first; legacy/fallback names that may exist in parquet)
    # For extreme-density cohorts, use base cohort's target date column (same schema as source)
    _cohort_for_target = cohort_name.replace("_extreme_density", "") if cohort_name.endswith("_extreme_density") else cohort_name
    if _cohort_for_target == "opioid_ed":
        candidates = ["first_f1120_date", "first_opioid_ed_date"]
    elif _cohort_for_target == "non_opioid_ed":
        candidates = ["first_o11_p_date", "first_ed_non_opioid_date", "first_opioid_ed_date"]  # last: Step 4 legacy fallback
    else:
        candidates = ["event_date"]

    # DTW expected vs have: log to DTW logs for debugging (e.g. target column / 85-114)
    DTW_EXPECTED_COLUMNS = [
        "target",  # 1=case (target cohort), 0=control; required
        "event_date",
        "mi_person_key",
        "drug_name",
        "primary_icd_diagnosis_code",
        "procedure_code",
    ]
    sorted_cols = sorted(col_names)
    _log("info", "DTW expected (model_events): %s; plus one of target_date: %s", DTW_EXPECTED_COLUMNS, candidates)
    _log("info", "DTW have (model_events columns, %d): %s", len(sorted_cols), sorted_cols)
    missing_expected = [c for c in DTW_EXPECTED_COLUMNS if c not in col_names]
    # event_date can be satisfied by another timestamp column (resolved below)
    if "event_date" in missing_expected:
        event_date_candidates_check = ["event_date", "incurred_date", "service_date", "event_timestamp", "claim_date", "event_dt"]
        if any(c in col_names for c in event_date_candidates_check):
            missing_expected = [c for c in missing_expected if c != "event_date"]
    if missing_expected:
        _log("warning", "DTW model_events missing expected columns: %s", missing_expected)
    if "target" not in col_names:
        _log("error", "model_events has no 'target' column (required: 1=case, 0=control). Columns: %s", sorted_cols)
        return pd.DataFrame(), 0

    target_date_col = None
    for c in candidates:
        if c in col_names:
            target_date_col = c
            break
    _log("info", "DTW target_date: expected one of %s; chosen: %s", candidates, target_date_col)
    if target_date_col is None and _cohort_for_target in ("opioid_ed", "non_opioid_ed"):
        _log("error", "Model data has no target date column. Expected one of %s. Found columns: %s", candidates, sorted_cols)
        return pd.DataFrame(), 0
    if target_date_col is None:
        target_date_col = "event_date"

    # Event date (timestamp) for ordering and N3 time-between: use first available timestamp column
    event_date_candidates = ["event_date", "incurred_date", "service_date", "event_timestamp", "claim_date", "event_dt"]
    event_date_col = None
    for c in event_date_candidates:
        if c in col_names:
            event_date_col = c
            break
    if event_date_col is None:
        _log("error", "Model data has no event timestamp column. Expected one of %s. Found columns: %s", event_date_candidates, sorted_cols)
        return pd.DataFrame(), 0
    _log("info", "DTW event_date (timestamp): chosen column %s for ordering and N3 time-between", event_date_col)

    # Log target distribution in source (row counts) to verify case/control mapping
    con_count = duckdb.connect(":memory:")
    try:
        r = con_count.execute(
            f"SELECT target, COUNT(*) as n FROM read_parquet('{path_str}') GROUP BY target ORDER BY target"
        ).fetchall()
        for (t, n) in r:
            _log("info", "DTW model_events target=%s row count: %s", t, n)
    except Exception as e:
        _log("warning", "Could not get target counts from parquet: %s", e)
    con_count.close()

    # Write an explicit diagnostics artifact for this cohort/age_band so empties are easy to debug
    try:
        diag_dir = _dtw_output_root(project_root) / "feature_engineering"
        diag_dir.mkdir(parents=True, exist_ok=True)
        diag_path = diag_dir / f"dtw_model_events_diagnostics_{cohort_name}_{age_band_fname}.json"

        con_diag = duckdb.connect(":memory:")
        diag = {
            "cohort": cohort_name,
            "age_band": age_band,
            "target": "both",
            "model_events_path": str(model_data_path),
            "event_date_col": event_date_col,
            "target_date_col": target_date_col,
            "model_events_columns": sorted_cols,
        }
        try:
            target_counts = con_diag.execute(
                f"SELECT target, COUNT(*)::BIGINT AS n_rows FROM read_parquet('{path_str}') GROUP BY target ORDER BY target"
            ).fetchall()
            diag["target_row_counts"] = {str(t): int(n) for (t, n) in target_counts}
        except Exception as e:
            diag["target_row_counts_error"] = str(e)

        try:
            drug_counts = con_diag.execute(
                f"""
                SELECT
                    target,
                    SUM(CASE WHEN drug_name IS NOT NULL AND drug_name != '' THEN 1 ELSE 0 END)::BIGINT AS n_drug_rows,
                    COUNT(DISTINCT CASE WHEN drug_name IS NOT NULL AND drug_name != '' THEN CAST(mi_person_key AS VARCHAR) ELSE NULL END)::BIGINT AS n_patients_with_drugs
                FROM read_parquet('{path_str}')
                GROUP BY target
                ORDER BY target
                """
            ).fetchall()
            diag["drug_row_counts_by_target"] = [
                {
                    "target": int(t) if t is not None else None,
                    "n_drug_rows": int(nr) if nr is not None else 0,
                    "n_patients_with_drugs": int(npd) if npd is not None else 0,
                }
                for (t, nr, npd) in drug_counts
            ]
        except Exception as e:
            diag["drug_row_counts_by_target_error"] = str(e)

        try:
            top_drugs_all = con_diag.execute(
                f"""
                SELECT drug_name, COUNT(*)::BIGINT AS n
                FROM read_parquet('{path_str}')
                WHERE drug_name IS NOT NULL AND drug_name != ''
                GROUP BY drug_name
                ORDER BY n DESC
                LIMIT 50
                """
            ).fetchall()
            diag["top_drugs_overall"] = [{"drug_name": str(d), "n": int(n)} for (d, n) in top_drugs_all]
        except Exception as e:
            diag["top_drugs_overall_error"] = str(e)

        try:
            top_drugs_target = con_diag.execute(
                f"""
                SELECT drug_name, COUNT(*)::BIGINT AS n
                FROM read_parquet('{path_str}')
                WHERE target = 1 AND drug_name IS NOT NULL AND drug_name != ''
                GROUP BY drug_name
                ORDER BY n DESC
                LIMIT 50
                """
            ).fetchall()
            diag["top_drugs_target_1"] = [{"drug_name": str(d), "n": int(n)} for (d, n) in top_drugs_target]
        except Exception as e:
            diag["top_drugs_target_1_error"] = str(e)
        con_diag.close()

        with open(diag_path, "w", encoding="utf-8") as f:
            json.dump(diag, f, indent=2)
        _log("info", "Wrote DTW model_events diagnostics: %s", diag_path)
    except Exception as e:
        _log("warning", "Failed to write DTW model_events diagnostics: %s", e)

    # Build SQL query with SHAP/FFA filtering
    con = duckdb.connect(":memory:")

    # Clause that keeps all drug events (no SHAP/FFA filter); used for fallback when filter yields 0 trajectories.
    drug_only_clause = "WHERE drug_name IS NOT NULL AND drug_name != ''"

    if use_filter:
        # DTW is drug-only for both cohorts: only events with allowed drug codes.
        def safe_sql_list(codes: Set[str]) -> str:
            escaped = [f"'{str(c).replace(chr(39), chr(39)+chr(39))}'" for c in codes if c]
            return "(" + ",".join(escaped) + ")" if escaped else "(NULL)"

        filters = []
        if drug_set:
            filters.append(
                f"REPLACE(REPLACE(drug_name, '.', ''), '-', '') IN {safe_sql_list(drug_set)}"
            )
        if filters:
            filter_clause = f"WHERE ({' AND '.join(filters)}) AND drug_name IS NOT NULL AND drug_name != ''"
        else:
            filter_clause = drug_only_clause
        _log("info", "Filter: drug-only, drugs=%d (ICD/CPT excluded for DTW)", len(drug_set))
        if drug_set:
            sample = sorted(drug_set)[:5]
            _log("info", "Allowed drug codes sample (normalized): %s", sample)
    else:
        filter_clause = drug_only_clause

    # Normalize target_date to DATE (parquet may have VARCHAR, TIMESTAMP, or INTEGER YYYYMMDD).
    # Use CAST(col AS DOUBLE) in the integer branch so both CASE branches type-check when col is TIMESTAMP.
    target_date_expr = (
        f"CASE WHEN typeof({target_date_col}) IN ('INTEGER', 'BIGINT') THEN "
        f"make_date(CAST(FLOOR(CAST({target_date_col} AS DOUBLE)/10000.0) AS INTEGER), "
        f"CAST(FLOOR(CAST({target_date_col} AS DOUBLE)/100.0) % 100 AS INTEGER), "
        f"CAST(FLOOR(CAST({target_date_col} AS DOUBLE)) % 100 AS INTEGER)) "
        f"ELSE CAST({target_date_col} AS DATE) END"
    )
    # Same normalization for event timestamp column (used for ordering and N3 time-between)
    event_date_expr = (
        f"CASE WHEN typeof({event_date_col}) IN ('INTEGER', 'BIGINT') THEN "
        f"make_date(CAST(FLOOR(CAST({event_date_col} AS DOUBLE)/10000.0) AS INTEGER), "
        f"CAST(FLOOR(CAST({event_date_col} AS DOUBLE)/100.0) % 100 AS INTEGER), "
        f"CAST(FLOOR(CAST({event_date_col} AS DOUBLE)) % 100 AS INTEGER)) "
        f"ELSE CAST({event_date_col} AS DATE) END"
    )
    # Extract trajectories with cutoff dates (target = before target event, control = all events)
    query = f"""
    WITH patient_events AS (
        SELECT
            CAST(mi_person_key AS VARCHAR) as mi_person_key,
            target,
            ({event_date_expr}) as event_date,
            drug_name,
            primary_icd_diagnosis_code,
            procedure_code,
            ({target_date_expr}) as target_date
        FROM read_parquet('{model_data_path}')
        {filter_clause}
    ),
    filtered_events AS (
        SELECT
            mi_person_key,
            target,
            event_date,
            drug_name,
            primary_icd_diagnosis_code,
            procedure_code
        FROM patient_events
        WHERE
            -- For target patients: only events before target date
            (target = 1 AND event_date < target_date
             AND DATEDIFF('month', event_date, target_date) <= {max_lookback_months})
            -- For control patients: all events
            OR (target = 0)
    ),
    trajectories AS (
        SELECT
            mi_person_key,
            target,
            STRING_AGG('DRUG:' || drug_name, '_' ORDER BY event_date)
                FILTER (WHERE drug_name IS NOT NULL AND drug_name != '') as seq_pattern_str,
            COUNT(*) as trajectory_length,
            COUNT(DISTINCT COALESCE(drug_name, '') || '|' || COALESCE(primary_icd_diagnosis_code, '') || '|' || COALESCE(procedure_code, '')) as trajectory_diversity
        FROM filtered_events
        GROUP BY mi_person_key, target
    )
    SELECT * FROM trajectories
    WHERE seq_pattern_str IS NOT NULL
    """

    _log("info", "Extracting trajectories from model_events...")
    expected_cols = [
        "mi_person_key",
        "target",
        "seq_pattern_str",
        "trajectory_length",
        "trajectory_diversity",
    ]
    expected_types = {
        "mi_person_key": "VARCHAR",
        "target": "INTEGER",
        "seq_pattern_str": "VARCHAR",
        "trajectory_length": "BIGINT",
        "trajectory_diversity": "BIGINT",
    }
    df, diag_main = duckdb_query_df_with_diagnostics(
        con,
        query,
        expected_columns=expected_cols,
        expected_types=expected_types,
    )
    dtw_sql_diagnostics = {
        "cohort": cohort_name,
        "age_band": age_band,
        "target": "both",
        "query_name": "trajectories",
        "diagnostics": diag_main,
    }

    # If SHAP/FFA drug filter matched no events (e.g. allowed codes don't match model_events drug_name normalization),
    # retry using all drug events so we still get trajectories when patients have drug events.
    fallback_diag = None
    if df.empty and use_filter and drug_set:
        try:
            total_drug_events = con.execute(
                f"SELECT COUNT(*) FROM read_parquet('{path_str}') {drug_only_clause}"
            ).fetchone()[0]
            _log(
                "warning",
                "No trajectories with SHAP/FFA drug filter (%d codes). Total drug events in model_events: %d; trying fallback: all drug events.",
                len(drug_set),
                total_drug_events,
            )
        except Exception:
            _log(
                "warning",
                "No trajectories with SHAP/FFA drug filter (%d codes); trying fallback: all drug events.",
                len(drug_set),
            )
        filter_clause = drug_only_clause
        query_fallback = f"""
    WITH patient_events AS (
        SELECT
            CAST(mi_person_key AS VARCHAR) as mi_person_key,
            target,
            ({event_date_expr}) as event_date,
            drug_name,
            primary_icd_diagnosis_code,
            procedure_code,
            ({target_date_expr}) as target_date
        FROM read_parquet('{path_str}')
        {filter_clause}
    ),
    filtered_events AS (
        SELECT
            mi_person_key,
            target,
            event_date,
            drug_name,
            primary_icd_diagnosis_code,
            procedure_code
        FROM patient_events
        WHERE
            (target = 1 AND event_date < target_date
             AND DATEDIFF('month', event_date, target_date) <= {max_lookback_months})
            OR (target = 0)
    ),
    trajectories AS (
        SELECT
            mi_person_key,
            target,
            STRING_AGG('DRUG:' || drug_name, '_' ORDER BY event_date)
                FILTER (WHERE drug_name IS NOT NULL AND drug_name != '') as seq_pattern_str,
            COUNT(*) as trajectory_length,
            COUNT(DISTINCT COALESCE(drug_name, '') || '|' || COALESCE(primary_icd_diagnosis_code, '') || '|' || COALESCE(procedure_code, '')) as trajectory_diversity
        FROM filtered_events
        GROUP BY mi_person_key, target
    )
    SELECT * FROM trajectories
    WHERE seq_pattern_str IS NOT NULL
    """
        df, fallback_diag = duckdb_query_df_with_diagnostics(
            con,
            query_fallback,
            expected_columns=expected_cols,
            expected_types=expected_types,
        )
        if fallback_diag is not None:
            dtw_sql_diagnostics["fallback"] = {
                "query_name": "trajectories_fallback_all_drugs",
                "diagnostics": fallback_diag,
            }
        if not df.empty:
            _log("info", "Fallback succeeded: %d trajectories using all drug events (no SHAP/FFA filter).", len(df))

    # Monthly trajectory: one token per calendar month, drug only for both cohorts.
    monthly_where = "AND drug_name IS NOT NULL AND drug_name != ''"
    _log("info", "Monthly trajectory: drug only (both cohorts)")
    monthly_events_query = f"""
    WITH patient_events AS (
        SELECT
            CAST(mi_person_key AS VARCHAR) as mi_person_key,
            target,
            ({event_date_expr}) as event_date,
            drug_name,
            primary_icd_diagnosis_code,
            procedure_code,
            ({target_date_expr}) as target_date
        FROM read_parquet('{path_str}')
        {filter_clause}
    ),
    filtered_events AS (
        SELECT
            mi_person_key,
            target,
            event_date,
            drug_name,
            primary_icd_diagnosis_code,
            procedure_code
        FROM patient_events
        WHERE
            (target = 1 AND event_date < target_date
             AND DATEDIFF('month', event_date, target_date) <= {max_lookback_months})
            OR (target = 0)
        {monthly_where}
    )
    SELECT
        mi_person_key,
        target,
        event_date,
        'DRUG:' || drug_name as code
    FROM filtered_events
    WHERE drug_name IS NOT NULL AND drug_name != ''
    """
    try:
        monthly_expected_cols = ["mi_person_key", "target", "event_date", "code"]
        monthly_expected_types = {
            "mi_person_key": "VARCHAR",
            "target": "INTEGER",
            "event_date": "DATE",
            "code": "VARCHAR",
        }
        events_df, diag_monthly = duckdb_query_df_with_diagnostics(
            con,
            monthly_events_query,
            expected_columns=monthly_expected_cols,
            expected_types=monthly_expected_types,
        )
        dtw_sql_diagnostics["monthly_events"] = {
            "query_name": "monthly_events",
            "target": "both",
            "diagnostics": diag_monthly,
        }
        if events_df.empty:
            _log(
                "warning",
                "Monthly events query returned 0 rows (cohort=%s age_band=%s). Expected cols=%s; received cols=%s; received types=%s",
                cohort_name,
                age_band,
                monthly_expected_cols,
                diag_monthly.get("received_columns"),
                diag_monthly.get("received_types"),
            )
        if not events_df.empty and "code" in events_df.columns:
            events_df = events_df.dropna(subset=["code"])
        if not events_df.empty:
            events_df["event_month"] = pd.to_datetime(events_df["event_date"]).dt.to_period("M").astype(str)
            # Per (patient, target, month): distinct codes, sort alphabetically, join with '_'
            monthly_bags = (
                events_df.groupby(["mi_person_key", "target", "event_month"])["code"]
                .apply(lambda s: "_".join(sorted(s.unique())))
                .reset_index()
            )
            # Per (patient, target): order by month, format "YYYY-MM:bag", join with '|'
            def _monthly_sequence(bags_df: pd.DataFrame) -> str:
                bags_df = bags_df.sort_values("event_month")
                return "|".join(
                    f"{row['event_month']}:{row['code']}" for _, row in bags_df.iterrows()
                )

            seq_monthly = (
                monthly_bags.groupby(["mi_person_key", "target"])
                .apply(lambda g: _monthly_sequence(g))
                .rename("seq_pattern_monthly")
            )
            df = df.merge(seq_monthly, on=["mi_person_key", "target"], how="left")
        else:
            df["seq_pattern_monthly"] = ""
        if "seq_pattern_monthly" in df.columns:
            df["seq_pattern_monthly"] = df["seq_pattern_monthly"].fillna("")
    except Exception as e:
        _log("warning", "Monthly trajectory failed: %s; leaving seq_pattern_monthly empty", e)
        df["seq_pattern_monthly"] = ""

    # Count events analyzed (filtered_events row count) for logging and status JSON
    count_query = f"""
    WITH patient_events AS (
        SELECT CAST(mi_person_key AS VARCHAR) as mi_person_key, target,
               ({event_date_expr}) as event_date, ({target_date_expr}) as target_date
        FROM read_parquet('{path_str}')
        {filter_clause}
    ),
    filtered_events AS (
        SELECT * FROM patient_events
        WHERE
            (target = 1 AND event_date < target_date
             AND DATEDIFF('month', event_date, target_date) <= {max_lookback_months})
            OR (target = 0)
    )
    SELECT COUNT(*) as n_events FROM filtered_events
    """
    try:
        n_events_analyzed = int(con.execute(count_query).fetchone()[0])
    except Exception as e:
        _log("warning", "Could not get event count: %s; using 0", e)
        n_events_analyzed = 0

    # Time-between metrics (N3: times between sequences) — use event timestamp column for gaps and days to target
    time_query = f"""
    WITH patient_events AS (
        SELECT
            CAST(mi_person_key AS VARCHAR) as mi_person_key,
            target,
            ({event_date_expr}) as event_date,
            ({target_date_expr}) as target_date
        FROM read_parquet('{model_data_path}')
        {filter_clause}
    ),
    filtered_events AS (
        SELECT mi_person_key, target, event_date, target_date
        FROM patient_events
        WHERE
            (target = 1 AND event_date < target_date
             AND DATEDIFF('month', event_date, target_date) <= {max_lookback_months})
            OR (target = 0)
    ),
    ordered AS (
        SELECT
            mi_person_key,
            target,
            target_date,
            event_date,
            LAG(event_date) OVER (PARTITION BY mi_person_key ORDER BY event_date) as prev_event_date,
            FIRST_VALUE(event_date) OVER (PARTITION BY mi_person_key ORDER BY event_date) as first_event_date
        FROM filtered_events
    ),
    gaps AS (
        SELECT
            mi_person_key,
            DATEDIFF('day', prev_event_date, event_date) as gap_days
        FROM ordered
        WHERE prev_event_date IS NOT NULL
    ),
    mean_gap AS (
        SELECT mi_person_key, AVG(gap_days)::DOUBLE as mean_days_between_events
        FROM gaps
        GROUP BY mi_person_key
    ),
    first_to_target AS (
        SELECT
            mi_person_key,
            DATEDIFF('day', MIN(first_event_date), MAX(target_date))::DOUBLE as days_first_event_to_target
        FROM ordered
        WHERE target = 1 AND target_date IS NOT NULL
        GROUP BY mi_person_key
    ),
    all_patients AS (
        SELECT DISTINCT mi_person_key FROM ordered
    )
    SELECT
        a.mi_person_key,
        m.mean_days_between_events,
        f.days_first_event_to_target
    FROM all_patients a
    LEFT JOIN mean_gap m ON a.mi_person_key = m.mi_person_key
    LEFT JOIN first_to_target f ON a.mi_person_key = f.mi_person_key
    """
    try:
        time_expected_cols = ["mi_person_key", "mean_days_between_events", "days_first_event_to_target"]
        time_expected_types = {
            "mi_person_key": "VARCHAR",
            "mean_days_between_events": "DOUBLE",
            "days_first_event_to_target": "DOUBLE",
        }
        time_df, diag_time = duckdb_query_df_with_diagnostics(
            con,
            time_query,
            expected_columns=time_expected_cols,
            expected_types=time_expected_types,
        )
        dtw_sql_diagnostics["time_between"] = {
            "query_name": "time_between",
            "target": "both",
            "diagnostics": diag_time,
        }
        if time_df.empty:
            _log(
                "warning",
                "Time-between query returned 0 rows (cohort=%s age_band=%s). Expected cols=%s; received cols=%s; received types=%s",
                cohort_name,
                age_band,
                time_expected_cols,
                diag_time.get("received_columns"),
                diag_time.get("received_types"),
            )
        if not time_df.empty and "mi_person_key" in time_df.columns:
            df = df.merge(time_df, on="mi_person_key", how="left")
        else:
            df["mean_days_between_events"] = float("nan")
            df["days_first_event_to_target"] = float("nan")
    except Exception as e:
        _log("warning", "Time-between query failed: %s; adding NaN columns", e)
        df["mean_days_between_events"] = float("nan")
        df["days_first_event_to_target"] = float("nan")

    # Temporal span (first to last event) for event density — same filtered events as trajectories
    span_query = f"""
    WITH patient_events AS (
        SELECT
            CAST(mi_person_key AS VARCHAR) as mi_person_key,
            target,
            ({event_date_expr}) as event_date,
            ({target_date_expr}) as target_date
        FROM read_parquet('{path_str}')
        {filter_clause}
    ),
    filtered_events AS (
        SELECT mi_person_key, event_date
        FROM patient_events
        WHERE
            (target = 1 AND event_date < target_date
             AND DATEDIFF('month', event_date, target_date) <= {max_lookback_months})
            OR (target = 0)
    ),
    span AS (
        SELECT
            mi_person_key,
            DATEDIFF('day', MIN(event_date), MAX(event_date))::DOUBLE as temporal_span_days
        FROM filtered_events
        GROUP BY mi_person_key
    )
    SELECT * FROM span
    """
    try:
        span_expected_cols = ["mi_person_key", "temporal_span_days"]
        span_expected_types = {"mi_person_key": "VARCHAR", "temporal_span_days": "DOUBLE"}
        span_df, diag_span = duckdb_query_df_with_diagnostics(
            con,
            span_query,
            expected_columns=span_expected_cols,
            expected_types=span_expected_types,
        )
        dtw_sql_diagnostics["temporal_span"] = {
            "query_name": "temporal_span",
            "target": "both",
            "diagnostics": diag_span,
        }
        if span_df.empty:
            _log(
                "warning",
                "Temporal span query returned 0 rows (cohort=%s age_band=%s). Expected cols=%s; received cols=%s; received types=%s",
                cohort_name,
                age_band,
                span_expected_cols,
                diag_span.get("received_columns"),
                diag_span.get("received_types"),
            )
        if not span_df.empty and "mi_person_key" in span_df.columns:
            df = df.merge(span_df[["mi_person_key", "temporal_span_days"]], on="mi_person_key", how="left")
        else:
            df["temporal_span_days"] = float("nan")
    except Exception as e:
        _log("warning", "Temporal span query failed: %s; adding NaN column", e)
        df["temporal_span_days"] = float("nan")

    # Routine vs no routine and (if added) extreme vs low medical: use TARGET COHORT but FULL UNFILTERED events.
    # Must run while con is still open (admin/medical queries use con).
    # Do not apply drug or SHAP/FFA filters here — we need all event types (medical + pharmacy) so routine admin ICD
    # and medical utilization are defined over full utilization for target and control.
    _log("info", "Counting administrative ICD events (routine appointments) from model_events (full unfiltered, target cohort)...")
    if admin_codes:
        def _safe_sql_list(codes: Set[str]) -> str:
            escaped = [f"'{str(c).replace(chr(39), chr(39)+chr(39))}'" for c in codes if c]
            return "(" + ",".join(escaped) + ")" if escaped else "(NULL)"
        admin_list = _safe_sql_list(admin_codes)
        icd_diag_cols = sorted([c for c in col_names if "icd_diagnosis_code" in c])
        if not icd_diag_cols:
            icd_diag_cols = ["primary_icd_diagnosis_code"] if "primary_icd_diagnosis_code" in col_names else []
        admin_match_conditions = " OR ".join(
            f"REPLACE(REPLACE(COALESCE({c}, ''), '.', ''), '-', '') IN {admin_list}"
            for c in icd_diag_cols
        )
        if not admin_match_conditions:
            admin_match_conditions = "1=0"
        _log("info", "Admin ICD columns used for routine count: %s", icd_diag_cols)
        admin_query = f"""
        WITH patient_events AS (
            SELECT
                CAST(mi_person_key AS VARCHAR) as mi_person_key,
                target,
                ({event_date_expr}) as event_date,
                ({target_date_expr}) as target_date,
                ({admin_match_conditions}) AS is_admin_icd
            FROM read_parquet('{path_str}')
        ),
        filtered_events AS (
            SELECT mi_person_key
            FROM patient_events
            WHERE
                ((target = 1 AND event_date < target_date
                  AND DATEDIFF('month', event_date, target_date) <= {max_lookback_months})
                 OR (target = 0))
                AND is_admin_icd
        )
        SELECT mi_person_key, COUNT(*)::INTEGER as admin_icd_event_count
        FROM filtered_events
        GROUP BY mi_person_key
        """
        try:
            admin_expected_cols = ["mi_person_key", "admin_icd_event_count"]
            admin_expected_types = {"mi_person_key": "VARCHAR", "admin_icd_event_count": "INTEGER"}
            admin_df, diag_admin = duckdb_query_df_with_diagnostics(
                con,
                admin_query,
                expected_columns=admin_expected_cols,
                expected_types=admin_expected_types,
            )
            dtw_sql_diagnostics["admin_icd_count"] = {
                "query_name": "admin_icd_count",
                "target": "both",
                "diagnostics": diag_admin,
            }
            if admin_df.empty:
                _log(
                    "warning",
                    "Admin ICD count query returned 0 rows (cohort=%s age_band=%s). Expected cols=%s; received cols=%s; received types=%s",
                    cohort_name,
                    age_band,
                    admin_expected_cols,
                    diag_admin.get("received_columns"),
                    diag_admin.get("received_types"),
                )
            if not admin_df.empty:
                df = df.merge(admin_df, on="mi_person_key", how="left")
                df["admin_icd_event_count"] = df["admin_icd_event_count"].fillna(0).astype(int)
            else:
                df["admin_icd_event_count"] = 0
        except Exception as e:
            _log("warning", "Admin ICD count query failed: %s; setting admin_icd_event_count=0", e)
            df["admin_icd_event_count"] = 0
    else:
        df["admin_icd_event_count"] = 0

    _log("info", "Counting full unfiltered medical events per patient (for routine × medical utilization)...")
    if "event_type" in col_names:
        medical_where = "event_type = 'medical'"
    else:
        medical_where = "(COALESCE(primary_icd_diagnosis_code, '') != '' OR COALESCE(procedure_code, '') != '')"
    medical_query = f"""
    WITH patient_events AS (
        SELECT
            CAST(mi_person_key AS VARCHAR) as mi_person_key,
            target,
            ({event_date_expr}) as event_date,
            ({target_date_expr}) as target_date
        FROM read_parquet('{path_str}')
        WHERE {medical_where}
    ),
    filtered_events AS (
        SELECT mi_person_key
        FROM patient_events
        WHERE
            ((target = 1 AND event_date < target_date
              AND DATEDIFF('month', event_date, target_date) <= {max_lookback_months})
             OR (target = 0))
    )
    SELECT mi_person_key, COUNT(*)::INTEGER as medical_event_count_full
    FROM filtered_events
    GROUP BY mi_person_key
    """
    try:
        med_expected_cols = ["mi_person_key", "medical_event_count_full"]
        med_expected_types = {"mi_person_key": "VARCHAR", "medical_event_count_full": "INTEGER"}
        medical_df, diag_med = duckdb_query_df_with_diagnostics(
            con,
            medical_query,
            expected_columns=med_expected_cols,
            expected_types=med_expected_types,
        )
        dtw_sql_diagnostics["medical_event_count_full"] = {
            "query_name": "medical_event_count_full",
            "target": "both",
            "diagnostics": diag_med,
        }
        if medical_df.empty:
            _log(
                "warning",
                "Medical event count query returned 0 rows (cohort=%s age_band=%s). Expected cols=%s; received cols=%s; received types=%s",
                cohort_name,
                age_band,
                med_expected_cols,
                diag_med.get("received_columns"),
                diag_med.get("received_types"),
            )
        if not medical_df.empty:
            df = df.merge(medical_df, on="mi_person_key", how="left")
            df["medical_event_count_full"] = df["medical_event_count_full"].fillna(0).astype(int)
        else:
            df["medical_event_count_full"] = 0
    except Exception as e:
        _log("warning", "Medical event count query failed: %s; setting medical_event_count_full=0", e)
        df["medical_event_count_full"] = 0
    if "medical_event_count_full" not in df.columns:
        df["medical_event_count_full"] = 0

    # Bin by full unfiltered medical events to support routine × utilization charts
    try:
        med_count = df["medical_event_count_full"].fillna(0).astype(float)
        p25_m = float(med_count.quantile(0.25))
        p50_m = float(med_count.quantile(0.50))
        p95_m = float(med_count.quantile(0.95))

        def _assign_medical_bin(val: float) -> str:
            if val <= p25_m:
                return "low"
            if val <= p50_m:
                return "medium"
            if val <= p95_m:
                return "high"
            return "extreme"

        df["medical_utilization_bin"] = med_count.apply(_assign_medical_bin)
        _log("info", "Medical utilization bins (full unfiltered): P25=%.0f, P50=%.0f, P95=%.0f", p25_m, p50_m, p95_m)
        for bin_name in DENSITY_BINS:
            n = (df["medical_utilization_bin"] == bin_name).sum()
            pct = 100.0 * n / len(df) if len(df) > 0 else 0
            _log("info", "  %s: %d (%.1f%%)", bin_name, n, pct)
    except Exception as e:
        _log("warning", "Medical utilization binning failed: %s; defaulting to 'low'", e)
        df["medical_utilization_bin"] = "low"

    con.close()

    # Ensure time columns exist
    if "mean_days_between_events" not in df.columns:
        df["mean_days_between_events"] = float("nan")
    if "days_first_event_to_target" not in df.columns:
        df["days_first_event_to_target"] = float("nan")
    # For target=0, days_first_event_to_target should be NaN
    if "target" in df.columns:
        df.loc[df["target"] != 1, "days_first_event_to_target"] = float("nan")
    if "temporal_span_days" not in df.columns:
        df["temporal_span_days"] = float("nan")

    # N3 time-based metrics logging (timestamped event column used for ordering and gaps)
    n_mean_days = int(df["mean_days_between_events"].notna().sum()) if "mean_days_between_events" in df.columns else 0
    n_days_to_target = int(df["days_first_event_to_target"].notna().sum()) if "days_first_event_to_target" in df.columns else 0
    _log("info", "N3 time-based metrics (timestamp column %s): trajectories with mean_days_between_events=%d, with days_first_event_to_target=%d", event_date_col, n_mean_days, n_days_to_target)

    # Event density: events per month (trajectory_length / (span_days/30)); NaN when span <= 0
    span_days = df["temporal_span_days"]
    length = df["trajectory_length"].astype(float)
    df["events_per_month"] = float("nan")
    valid_span = (span_days > 0) & span_days.notna()
    df.loc[valid_span, "events_per_month"] = (
        length.loc[valid_span] / (span_days.loc[valid_span] / 30.0)
    )

    # Bin trajectories by event density (low/medium/high/extreme), aligned with FP-Growth percentiles
    density_value = df["events_per_month"].fillna(0)  # single-event / zero-span -> 0 (low)
    p25 = float(density_value.quantile(0.25))
    p50 = float(density_value.quantile(0.50))
    p95 = float(density_value.quantile(0.95))

    def _assign_density_bin(val):
        if val <= p25:
            return "low"
        if val <= p50:
            return "medium"
        if val <= p95:
            return "high"
        return "extreme"

    df["event_density_bin"] = density_value.apply(_assign_density_bin)
    _log("info", "Event density bins (events_per_month): P25=%.2f, P50=%.2f, P95=%.2f", p25, p50, p95)
    for bin_name in DENSITY_BINS:
        n = (df["event_density_bin"] == bin_name).sum()
        pct = 100.0 * n / len(df) if len(df) > 0 else 0
        _log("info", "  %s: %d (%.1f%%)", bin_name, n, pct)

    # Schema compatibility: dtw_min_distance (not computed here)
    df["dtw_min_distance"] = float("nan")

    # Attach full SQL diagnostics so main() can write them into trajectory_status JSON
    try:
        df.attrs["dtw_sql_diagnostics"] = dtw_sql_diagnostics
    except Exception:
        pass

    _log("info", "Extracted %d patient trajectories", len(df))

    if df.empty:
        _log(
            "warning",
            "No trajectories extracted (cohort=%s age_band=%s). Expected cols=%s; received cols=%s; received types=%s",
            cohort_name,
            age_band,
            expected_cols,
            diag_main.get("received_columns"),
            diag_main.get("received_types"),
        )
        df.attrs["dtw_sql_diagnostics"] = dtw_sql_diagnostics
        return df, n_events_analyzed

    _log("info", "Trajectory summary: events_analyzed=%d, alignments_found=%d", n_events_analyzed, len(df))
    _log("info", "  Mean length: %.1f; mean diversity: %.1f; admin_icd_event_count sum: %s; medical_event_count_full sum: %s", df["trajectory_length"].mean(), df["trajectory_diversity"].mean(), df["admin_icd_event_count"].sum(), df["medical_event_count_full"].sum())
    _log("info", "  DTW output target=1: %d, target=0: %d", (df["target"] == 1).sum(), (df["target"] == 0).sum())

    return df, n_events_analyzed


def main():
    parser = argparse.ArgumentParser(
        description="Extract lightweight trajectory data for DTW visualizations (no distance computations)"
    )
    parser.add_argument("--cohort", "--cohort-name", dest="cohort", required=True, help="Cohort name")
    parser.add_argument("--age-band", required=True, help="Age band")
    parser.add_argument("--max-lookback-months", type=int, default=24,
                       help="Max lookback months for target patients (default: 24)")
    parser.add_argument("--force", action="store_true", help="Force re-run even if output exists")
    parser.add_argument("--project-root", type=Path, default=REPO_ROOT, help="Project root directory")

    args = parser.parse_args()
    project_root = Path(args.project_root)
    age_band_fname = args.age_band.replace("-", "_")
    logger = setup_pipeline_logger(
        step_name="5_dtw",
        cohort=args.cohort,
        age_band=args.age_band,
        script_name="create_dtw_trajectories"
    )

    # Output paths (parquet primary; CSV for backward compatibility)
    output_dir = _dtw_output_root(project_root) / "feature_engineering"
    output_dir.mkdir(parents=True, exist_ok=True)
    base_name = f"dtw_features_{args.cohort}_{age_band_fname}"
    output_path_csv = output_dir / f"{base_name}.csv"
    output_path_parquet = output_dir / f"{base_name}.parquet"

    # Idempotency check (either format present)
    if not args.force and (output_path_parquet.exists() or output_path_csv.exists()):
        logger.info("Output exists at %s or %s; skipping (use --force to re-run)", output_path_parquet, output_path_csv)
        return

    with step_block("5_dtw", "create_dtw_trajectories", logger=logger.logger):
        logger.info("Starting DTW trajectories for %s / %s", args.cohort, args.age_band)
        df, n_events_analyzed = extract_patient_trajectories(
            project_root=project_root,
            cohort_name=args.cohort,
            age_band=args.age_band,
            max_lookback_months=args.max_lookback_months,
            logger=logger,
        )
        n_alignments_found = len(df)
        logger.info("Drug events analyzed: %d; trajectories (alignments) found: %d", n_events_analyzed, n_alignments_found)

        if df.empty:
            logger.warning("No trajectories extracted; writing placeholder artifacts and continuing pipeline.")
            # Placeholder parquet + CSV so create_dtw_features finds the file and skips gracefully
            placeholder_df = pd.DataFrame(columns=DTW_TRAJECTORY_CSV_COLUMNS)
            placeholder_df.to_parquet(output_path_parquet, index=False)
            placeholder_df.to_csv(output_path_csv, index=False)
            added_path_csv = output_dir / f"dtw_added_features_{args.cohort}_{age_band_fname}.csv"
            added_path_parquet = output_dir / f"dtw_added_features_{args.cohort}_{age_band_fname}.parquet"
            placeholder_df.to_csv(added_path_csv, index=False)
            placeholder_df.to_parquet(added_path_parquet, index=False)
            # Status JSON with event/alignment counts so downstream and logs are clear
            status_path = output_dir / f"trajectory_status_{args.cohort}_{age_band_fname}.json"
            diagnostics = {}
            try:
                diagnostics = getattr(df, "attrs", {}).get("dtw_sql_diagnostics", {}) if df is not None else {}
            except Exception:
                diagnostics = {}
            status = {
                "skipped": True,
                "n_events_analyzed": n_events_analyzed,
                "n_alignments_found": 0,
                "message": f"Events analyzed: {n_events_analyzed}; alignments (trajectories) found: 0. No trajectories extracted (check model_events, allowed_codes, or filter).",
                "cohort": args.cohort,
                "age_band": args.age_band,
                "target": "both",
                "sql_diagnostics": diagnostics,
            }
            with open(status_path, "w", encoding="utf-8") as f:
                json.dump(status, f, indent=2)
            logger.info("Wrote placeholder parquet/CSV and %s (events=%d, alignments=0)", status_path.name, n_events_analyzed)
            logger.log_summary()
            return
        # Save parquet (primary) and CSV (backward compatibility)
        df.to_parquet(output_path_parquet, index=False)
        df.to_csv(output_path_csv, index=False)
        logger.info("Saved %d trajectories to %s and %s", len(df), output_path_parquet, output_path_csv)
        logger.info("Columns: %s", list(df.columns))

        # Status JSON with counts (for create_dtw_features and reporting)
        status_path = output_dir / f"trajectory_status_{args.cohort}_{age_band_fname}.json"
        diagnostics = {}
        try:
            diagnostics = getattr(df, "attrs", {}).get("dtw_sql_diagnostics", {}) if df is not None else {}
        except Exception:
            diagnostics = {}
        status = {
            "skipped": False,
            "n_events_analyzed": n_events_analyzed,
            "n_alignments_found": n_alignments_found,
            "message": f"Events analyzed: {n_events_analyzed}; alignments (trajectories) found: {n_alignments_found}.",
            "cohort": args.cohort,
            "age_band": args.age_band,
            "target": "both",
            "sql_diagnostics": diagnostics,
        }
        with open(status_path, "w", encoding="utf-8") as f:
            json.dump(status, f, indent=2)

        # Also write dtw_added_features (expected by create_dtw_visuals.py)
        added_path_csv = output_dir / f"dtw_added_features_{args.cohort}_{age_band_fname}.csv"
        added_path_parquet = output_dir / f"dtw_added_features_{args.cohort}_{age_band_fname}.parquet"
        df.to_csv(added_path_csv, index=False)
        df.to_parquet(added_path_parquet, index=False)
        logger.info("Also saved to %s and %s (for create_dtw_visuals.py)", added_path_parquet, added_path_csv)
    
    logger.log_summary()


if __name__ == "__main__":
    main()
