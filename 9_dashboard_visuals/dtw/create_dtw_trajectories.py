#!/usr/bin/env python3
"""
Create trajectory data for DTW alignment and dashboard visualizations (Step 1 of DTW workflow).

This script extracts patient trajectories from model_data for DTW alignment and visualization.
Part 1 of DTW pipeline: trajectories → DTW alignment (create_dtw_features.py) → visuals.
NOT used for model training - for dashboard visual analysis of SHAP/FFA results.

Output CSV columns (minimal for visualization):
- mi_person_key: Patient identifier
- target: Target outcome (0/1)
- seq_pattern_str: Sequence of activity codes (e.g., "DRUG:Med_ICD:F1120_CPT:99213")
- admin_icd_event_count: Count of administrative ICD codes (routine vs no routine)
- trajectory_length: Number of events
- trajectory_diversity: Number of unique activities
- dtw_min_distance: Placeholder (NaN); DTW distances computed in create_dtw_features.py (Step 2)
- mean_days_between_events: Mean days between consecutive events in the trajectory (N3: times between sequences)
- days_first_event_to_target: For target=1, days from first event to target date; else NaN (N3)

Requirements:
- 4_model_data (Step 4) with model_events parquet
- 7_shap_analysis and 8_ffa_analysis (Steps 7-8) for SHAP/FFA important codes
- 1b_apcd_event_filter/administrative_codes_lookup.json for routine analysis

Runtime: ~1-2 minutes per cohort/age_band (fast!)
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Set, Tuple

import duckdb
import pandas as pd

# Repo root and step folder (9_dashboard_visuals)
REPO_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from py_helpers.fe_monitor import step_block  # noqa: E402
from py_helpers.model_data_paths import resolve_model_events_path  # noqa: E402
from py_helpers.pipeline_logger import (  # noqa: E402
    setup_pipeline_logger,
    log_step_start,
    log_step_complete,
    PipelineLogger,
)


def _dtw_output_root(project_root: Path) -> Path:
    """Dashboard visualization outputs (step 10); creation code in 9_dashboard_visuals/dtw."""
    return project_root / "10_risk_dashboard" / "visualizations" / "dtw"


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
    """Load administrative ICD codes from 1b_apcd_event_filter/administrative_codes_lookup.json."""
    path = project_root / "1b_apcd_event_filter" / "administrative_codes_lookup.json"
    if not path.exists():
        print(f"[WARN] Administrative codes lookup not found at {path}")
        return set()

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        codes = data.get("administrative_codes", {}).get("icd", [])
        return set(str(c) for c in codes)
    except Exception as exc:
        print(f"[WARN] Could not load administrative codes: {exc}")
        return set()


def extract_patient_trajectories(
    project_root: Path,
    cohort_name: str,
    age_band: str,
    max_lookback_months: int = 24,
) -> pd.DataFrame:
    """
    Extract patient trajectories from model_data for visualization.

    Returns DataFrame with columns:
    - mi_person_key
    - target
    - seq_pattern_str
    - admin_icd_event_count
    - trajectory_length
    - trajectory_diversity
    """
    print(f"\n{'='*60}")
    print(f"Extracting trajectories for {cohort_name} / {age_band}")
    print(f"{'='*60}")

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
        print(f"[ERROR] Model data not found at {model_data_path}")
        return pd.DataFrame()

    print(f"[INFO] Using model_events: {model_data_path}")

    # SHAP/FFA combined allowed codes file is required (same prerequisite as BupaR); we never use all events.
    age_band_fname = age_band.replace("-", "_")
    bupar_output_root = project_root / "10_risk_dashboard" / "visualizations" / "bupar" / "outputs"
    allowed_codes_path = bupar_output_root / f"allowed_codes_shap_ffa_{cohort_name}_{age_band_fname}.json"
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
    print(f"[INFO] Filtering to {len(allowed_codes)} SHAP/FFA important codes (from combined file)")
    drug_set, icd_set, cpt_set = _split_allowed_codes_by_type(allowed_codes)
    use_filter = True

    # Get administrative ICD codes for routine vs no routine analysis
    admin_codes = _load_administrative_icd_codes(project_root)
    print(f"[INFO] Loaded {len(admin_codes)} administrative ICD codes")

    # Resolve target date column from parquet schema (Step 4 writes canonical names; support legacy)
    path_str = str(model_data_path).replace("'", "''")
    con_schema = duckdb.connect(":memory:")
    schema = con_schema.execute(
        f"DESCRIBE SELECT * FROM read_parquet('{path_str}')"
    ).fetchall()
    col_names = {row[0] for row in schema}
    con_schema.close()

    if cohort_name == "opioid_ed":
        # Canonical: first_f1120_date (Step 4); legacy: first_opioid_ed_date
        target_date_col = "first_f1120_date" if "first_f1120_date" in col_names else "first_opioid_ed_date"
    elif cohort_name == "non_opioid_ed":
        # Canonical: first_o11_p_date (Step 4); legacy: first_ed_non_opioid_date
        target_date_col = "first_o11_p_date" if "first_o11_p_date" in col_names else "first_ed_non_opioid_date"
    else:
        target_date_col = "event_date"  # fallback

    if cohort_name in ("opioid_ed", "non_opioid_ed") and target_date_col not in col_names:
        print(
            f"[ERROR] Model data at {model_data_path} has no target date column. "
            f"Expected one of: opioid_ed: first_f1120_date/first_opioid_ed_date; "
            f"non_opioid_ed: first_o11_p_date/first_ed_non_opioid_date. Found columns: {sorted(col_names)}"
        )
        raise SystemExit(1)
    print(f"[INFO] Using target date column: {target_date_col}")

    # Build SQL query with SHAP/FFA filtering
    con = duckdb.connect(":memory:")

    if use_filter:
        # Build filter expressions (OR semantics: drug OR any ICD OR CPT)
        # Only include non-empty filters to avoid syntax errors
        filters = []
        if drug_set:
            drug_filter = " OR ".join([f"REPLACE(REPLACE(drug_name, '.', ''), '-', '') = '{d}'" for d in list(drug_set)[:50]])
            filters.append(drug_filter)
        if icd_set:
            icd_filter = " OR ".join([
                f"REPLACE(REPLACE(primary_icd_diagnosis_code, '.', ''), '-', '') = '{i}'" for i in list(icd_set)[:50]
            ])
            filters.append(icd_filter)
        if cpt_set:
            cpt_filter = " OR ".join([f"REPLACE(REPLACE(procedure_code, '.', ''), '-', '') = '{c}'" for c in list(cpt_set)[:50]])
            filters.append(cpt_filter)

        if filters:
            filter_clause = f"WHERE ({' OR '.join(filters)})"
        else:
            filter_clause = ""
    else:
        filter_clause = ""

    # Extract trajectories with cutoff dates (target = before target event, control = all events)
    query = f"""
    WITH patient_events AS (
        SELECT
            CAST(mi_person_key AS VARCHAR) as mi_person_key,
            target,
            event_date,
            drug_name,
            primary_icd_diagnosis_code,
            procedure_code,
            {target_date_col} as target_date
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
            STRING_AGG(
                CASE
                    WHEN drug_name IS NOT NULL AND drug_name != '' THEN 'DRUG:' || drug_name
                    WHEN primary_icd_diagnosis_code IS NOT NULL AND primary_icd_diagnosis_code != ''
                        THEN 'ICD:' || primary_icd_diagnosis_code
                    WHEN procedure_code IS NOT NULL AND procedure_code != '' THEN 'CPT:' || procedure_code
                    ELSE NULL
                END,
                '_'
                ORDER BY event_date
            ) FILTER (WHERE
                drug_name IS NOT NULL OR
                primary_icd_diagnosis_code IS NOT NULL OR
                procedure_code IS NOT NULL
            ) as seq_pattern_str,
            COUNT(*) as trajectory_length,
            COUNT(DISTINCT COALESCE(drug_name, '') || '|' || COALESCE(primary_icd_diagnosis_code, '') || '|' || COALESCE(procedure_code, '')) as trajectory_diversity
        FROM filtered_events
        GROUP BY mi_person_key, target
    )
    SELECT * FROM trajectories
    WHERE seq_pattern_str IS NOT NULL
    """

    print("[INFO] Extracting trajectories from model_events...")
    df = con.execute(query).df()

    # Time-between metrics (N3: times between sequences)
    time_query = f"""
    WITH patient_events AS (
        SELECT
            CAST(mi_person_key AS VARCHAR) as mi_person_key,
            target,
            event_date,
            {target_date_col} as target_date
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
    )
    SELECT
        m.mi_person_key,
        m.mean_days_between_events,
        f.days_first_event_to_target
    FROM mean_gap m
    LEFT JOIN first_to_target f ON m.mi_person_key = f.mi_person_key
    """
    try:
        time_df = con.execute(time_query).df()
        if not time_df.empty and "mi_person_key" in time_df.columns:
            df = df.merge(time_df, on="mi_person_key", how="left")
        else:
            df["mean_days_between_events"] = float("nan")
            df["days_first_event_to_target"] = float("nan")
    except Exception as e:
        print(f"[WARN] Time-between query failed: {e}; adding NaN columns")
        df["mean_days_between_events"] = float("nan")
        df["days_first_event_to_target"] = float("nan")

    con.close()

    # Ensure time columns exist
    if "mean_days_between_events" not in df.columns:
        df["mean_days_between_events"] = float("nan")
    if "days_first_event_to_target" not in df.columns:
        df["days_first_event_to_target"] = float("nan")
    # For target=0, days_first_event_to_target should be NaN
    if "target" in df.columns:
        df.loc[df["target"] != 1, "days_first_event_to_target"] = float("nan")

    # Schema compatibility: dtw_min_distance (not computed here)
    df["dtw_min_distance"] = float("nan")

    print(f"[INFO] Extracted {len(df)} patient trajectories")

    if df.empty:
        print("[WARN] No trajectories extracted")
        return df

    # Compute admin_icd_event_count for each patient
    print("[INFO] Computing administrative ICD event counts...")

    def count_admin_icds(seq_str):
        """Count administrative ICD codes in sequence."""
        if not seq_str or pd.isna(seq_str):
            return 0
        count = 0
        for token in seq_str.split('_'):
            if token.startswith('ICD:'):
                icd_code = token[4:]
                if icd_code in admin_codes:
                    count += 1
        return count

    df['admin_icd_event_count'] = df['seq_pattern_str'].apply(count_admin_icds)

    print("[INFO] Trajectory summary:")
    print(f"  - Mean length: {df['trajectory_length'].mean():.1f}")
    print(f"  - Mean diversity: {df['trajectory_diversity'].mean():.1f}")
    print(f"  - Admin ICD events (routine): {df['admin_icd_event_count'].sum()}")
    print(f"  - Target=1: {(df['target']==1).sum()}, Target=0: {(df['target']==0).sum()}")

    return df


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

    # Output path
    output_dir = _dtw_output_root(project_root) / "outputs" / "feature_engineering"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"dtw_features_{args.cohort}_{age_band_fname}.csv"

    # Idempotency check
    if not args.force and output_path.exists():
        logger.info("Output exists at %s; skipping (use --force to re-run)", output_path)
        return

    with step_block("5_dtw", "create_dtw_trajectories", logger=logger.logger):
        logger.info("Starting DTW trajectories for %s / %s", args.cohort, args.age_band)
        # Extract trajectories
        df = extract_patient_trajectories(
            project_root=project_root,
            cohort_name=args.cohort,
            age_band=args.age_band,
            max_lookback_months=args.max_lookback_months,
        )

        if df.empty:
            logger.error("No trajectories extracted. Check inputs and logs.")
            logger.log_summary()
            sys.exit(1)

        # Save
        df.to_csv(output_path, index=False)
        logger.info("Saved %d trajectories to %s", len(df), output_path)
        logger.info("Columns: %s", list(df.columns))

        # Also copy to dtw_added_features (expected by create_dtw_visuals.py)
        added_path = output_dir / f"dtw_added_features_{args.cohort}_{age_band_fname}.csv"
        df.to_csv(added_path, index=False)
        logger.info("Also saved to %s (for create_dtw_visuals.py)", added_path)
    
    logger.log_summary()


if __name__ == "__main__":
    main()
