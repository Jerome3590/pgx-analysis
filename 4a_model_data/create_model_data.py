#!/usr/bin/env python3
"""
Create model-ready event-level data filtered to important features, with
**within-cohort controls** constructed from gold medical/pharmacy tables.

This script is intentionally DuckDB + Parquet only for event-level data
to avoid pandas memory pressure on large cohorts:

1. Reads aggregated feature-importance CSVs from:
     - 3_feature_importance/outputs/
     - or, if not present locally, from 3_feature_importance/from_s3/by_cohort/**/
2. Extracts the `feature` column (e.g., `item_99284`, `item_AMOXICILLIN`) and
   strips the `item_` prefix to get raw item codes.
3. For each (cohort_name, age_band) combination in those files, it:
   - reads Step-2 cohort parquet files for the target cohort from local disk,
     typically under:
       PROJECT_ROOT/data/gold_cohorts/
         cohort_name={cohort_name}/event_year={year}/age_band={age_band}/cohort.parquet
   - reads gold medical / pharmacy events for the same age band and years:
       PROJECT_ROOT/data/gold_medical/age_band={age_band}/event_year={year}/*.parquet
       PROJECT_ROOT/data/gold_pharmacy/age_band={age_band}/event_year={year}/*.parquet
   - builds:
       * **cases** (target = 1):
           - patients with is_target_case = 1 in the cohort tables
           - events filtered by feature importance
             (drug_name, all ICD diagnosis columns, procedure_code)
       * **controls** (target = 0):
           - patients drawn from gold medical/pharmacy for the same age band
           - must have no opioid ICD codes in any diagnosis column across
             all their medical events, using OPIOID_ICD_CODES
           - must not appear in the case set for this cohort/age band
           - all medical + pharmacy events are kept (no FI-based filtering)
         Controls are sampled to maintain an approximate DEFAULT_SAMPLE_RATIO
         (e.g., 5:1) control:case patient ratio.
   - writes the combined events to:
       4a_model_data/cohort_name={cohort_name}/age_band={age_band}/model_events.parquet
     with an event-level `target` column.

This output is then used as input for:
 - FP-Growth (pattern mining on important features plus within-cohort controls)
 - BupaR (process mining / event-log analysis)
 - DTW (trajectory analysis on filtered event sequences)
 - Final models (Step 6)
"""

import os
import shutil
import sys
from pathlib import Path
from typing import List, Tuple

import duckdb
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from py_helpers.constants import (
    ALL_ICD_DIAGNOSIS_COLUMNS,
    OPIOID_ICD_CODES,
    DEFAULT_SAMPLE_RATIO,
    get_opioid_icd_sql_condition,
)


OUTPUTS_DIR = PROJECT_ROOT / "3_feature_importance" / "outputs"
MODEL_DATA_ROOT = PROJECT_ROOT / "4a_model_data"


def resolve_local_cohort_root() -> Path:
    """
    Resolve the root directory containing Step-2 **gold cohort** parquet files.

    Priority:
      1. LOCAL_DATA_PATH environment variable (if set)
      2. PROJECT_ROOT/data/gold_cohorts
    """
    env_path = os.getenv("LOCAL_DATA_PATH")
    if env_path:
        root = Path(env_path)
        if root.exists():
            return root

    default_root = PROJECT_ROOT / "data" / "gold_cohorts"
    return default_root


def resolve_local_medical_root() -> Path:
    """
    Resolve the root directory containing gold medical event parquet files.

    Default:
      PROJECT_ROOT/data/gold_medical
    """
    env_path = os.getenv("LOCAL_MEDICAL_PATH")
    if env_path:
        root = Path(env_path)
        if root.exists():
            return root

    return PROJECT_ROOT / "data" / "gold_medical"


def resolve_local_pharmacy_root() -> Path:
    """
    Resolve the root directory containing gold pharmacy event parquet files.

    Default:
      PROJECT_ROOT/data/gold_pharmacy
    """
    env_path = os.getenv("LOCAL_PHARMACY_PATH")
    if env_path:
        root = Path(env_path)
        if root.exists():
            return root

    return PROJECT_ROOT / "data" / "gold_pharmacy"


def parse_aggregated_filename(path: Path) -> Tuple[str, str]:
    """
    Parse cohort_name and age_band from an aggregated CSV filename.

    Current pattern (from 3_feature_importance/outputs or from_s3/by_cohort):
        {cohort_name}_{age_band_fname}_aggregated_feature_importance.csv

    Example:
        opioid_ed_0_12_aggregated_feature_importance.csv
        -> cohort_name = opioid_ed
        -> age_band    = 0-12
    """
    stem = path.stem  # e.g. opioid_ed_0_12_aggregated_feature_importance
    parts = stem.split("_")

    # Expect pattern: {cohort_name}_{age_band_fname}_aggregated_feature_importance
    # where age_band_fname is something like "0_12" or "13_24".
    if len(parts) < 5:
        raise ValueError(f"Unexpected aggregated filename format: {path.name}")

    cohort_name_tokens = parts[:-5]
    age_band_tokens = parts[-5:-3]

    cohort_name = "_".join(cohort_name_tokens)
    age_band_fname = "_".join(age_band_tokens)

    # Convert age_band_fname (e.g., 13_24) back to canonical age_band (13-24)
    age_band = age_band_fname.replace("_", "-")
    return cohort_name, age_band


def get_important_items(agg_csv: Path) -> List[str]:
    """Read aggregated feature-importance CSV and return item codes (no 'item_' prefix)."""
    df = pd.read_csv(agg_csv)
    if "feature" not in df.columns:
        raise ValueError(f"'feature' column not found in {agg_csv}")

    items = (
        df["feature"]
        .astype(str)
        .str.replace("^item_", "", regex=True)
        .unique()
        .tolist()
    )
    return items


def filter_cohort_events_for_items(
    cohort_name: str,
    age_band: str,
    important_items: List[str],
    years: List[int],
    output_root: Path,
    local_cohort_root: Path,
    local_medical_root: Path,
    local_pharmacy_root: Path,
    sample_ratio: float = DEFAULT_SAMPLE_RATIO,
) -> None:
    """
    Build model-ready event data for a single cohort/age-band and write to 4a_model_data/.

    For the given (cohort_name, age_band, years), this function:
      - reads the Step-2 cohort parquet(s) for that cohort from LOCAL storage
        (gold cohorts),
      - for **cases** (is_target_case = 1 in the cohort tables):
          * keeps only events where ANY of the item-bearing columns match an
            important item (drug_name, all ICD diagnosis columns, procedure_code),
      - for **controls** (target = 0):
          * selects patients from gold medical/pharmacy in the same age_band/years
          * excludes any patient who has an opioid ICD code in any diagnosis
            column across their medical events
          * excludes any patient whose mi_person_key is in the case set
          * samples patients to approximate `sample_ratio` controls per case
          * keeps all medical + pharmacy events for selected controls
      - writes the combined events to:
            4a_model_data/cohort_name={cohort_name}/age_band={age_band}/model_events.parquet
        with an event-level `target` column.

    All heavy lifting is done in DuckDB; pandas is not used for event-level data.
    """
    if not important_items:
        print(f"[WARN] No important items for {cohort_name}/{age_band}; skipping.")
        return

    # Build list of local cohort parquet paths for this cohort/age_band across years
    cohort_parquet_paths: List[str] = []
    for year in years:
        p = (
            local_cohort_root
            / f"cohort_name={cohort_name}"
            / f"event_year={year}"
            / f"age_band={age_band}"
            / "cohort.parquet"
        )
        if p.exists():
            cohort_parquet_paths.append(str(p))
        else:
            print(
                f"[INFO] Local cohort parquet not found for {cohort_name}/{age_band}/{year}: {p}"
            )

    if not cohort_parquet_paths:
        print(
            f"[WARN] No local cohort parquet files found for {cohort_name}/{age_band} "
            f"across years {years}. Did you run aws s3 sync into {local_root}?"
        )
        return

    # Build lists of gold medical and pharmacy parquet paths (globs) for this age_band across years
    medical_parquet_paths: List[str] = []
    pharmacy_parquet_paths: List[str] = []

    for year in years:
        medical_parent = (
            local_medical_root
            / f"age_band={age_band}"
            / f"event_year={year}"
        )
        pharmacy_parent = (
            local_pharmacy_root
            / f"age_band={age_band}"
            / f"event_year={year}"
        )

        medical_glob = medical_parent / "*.parquet"
        pharmacy_glob = pharmacy_parent / "*.parquet"

        if medical_parent.exists():
            medical_parquet_paths.append(str(medical_glob))
        else:
            print(
                f"[INFO] Gold medical not found for age_band={age_band}, year={year}: {medical_parent}"
            )

        if pharmacy_parent.exists():
            pharmacy_parquet_paths.append(str(pharmacy_glob))
        else:
            print(
                f"[INFO] Gold pharmacy not found for age_band={age_band}, year={year}: {pharmacy_parent}"
            )

    if not medical_parquet_paths:
        print(
            f"[WARN] No gold medical parquet files found for age_band={age_band} "
            f"across years {years}. Controls cannot be constructed; skipping."
        )
        return

    if not pharmacy_parquet_paths:
        print(
            f"[WARN] No gold pharmacy parquet files found for age_band={age_band} "
            f"across years {years}. Controls cannot be fully constructed; skipping."
        )
        return

    # Use DuckDB to read and filter in one pass
    con = duckdb.connect()
    cohort_paths_literal = ", ".join(f"'{p}'" for p in cohort_parquet_paths)
    gold_medical_paths_literal = ", ".join(f"'{p}'" for p in medical_parquet_paths)
    gold_pharmacy_paths_literal = ", ".join(f"'{p}'" for p in pharmacy_parquet_paths)
    all_control_paths_literal = ", ".join(
        f"'{p}'" for p in (medical_parquet_paths + pharmacy_parquet_paths)
    )

    item_list_literal = ", ".join(f"'{v}'" for v in important_items)

    # Build ICD diagnosis conditions dynamically from ALL_ICD_DIAGNOSIS_COLUMNS
    icd_conditions = " OR ".join(
        f"{col} IN ({item_list_literal})" for col in ALL_ICD_DIAGNOSIS_COLUMNS
    )

    print(
        f"[INFO] Building model events for {cohort_name}/{age_band} "
        f"from {len(cohort_parquet_paths)} cohort files, "
        f"{len(medical_parquet_paths)} medical globs, "
        f"{len(pharmacy_parquet_paths)} pharmacy globs, "
        f"using {len(important_items)} important items."
    )

    out_dir = (
        output_root
        / f"cohort_name={cohort_name}"
        / f"age_band={age_band}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "model_events.parquet"

    # Idempotency / Windows-friendly: if the file already exists, assume this
    # cohort/age_band has been built successfully and skip overwriting. This
    # avoids DuckDB attempting to delete an open file and throwing WinError 32.
    if out_path.exists():
        print(
            f"[INFO] model_events.parquet already exists for {cohort_name}/{age_band}; "
            f"skipping rebuild of this partition."
        )
        con.close()
        return

    # Derive a common set of columns present in both cohort and control sources,
    # so that set operations (UNION ALL) are well-defined.
    cohort_cols = [
        row[0]
        for row in con.execute(
            f"DESCRIBE SELECT * FROM read_parquet([{cohort_paths_literal}])"
        ).fetchall()
    ]
    control_cols = [
        row[0]
        for row in con.execute(
            f"DESCRIBE SELECT * FROM read_parquet([{all_control_paths_literal}], union_by_name=True)"
        ).fetchall()
    ]
    common_cols = [c for c in cohort_cols if c in control_cols]
    if not common_cols:
        print(
            f"[WARN] No common columns between cohort and control sources for "
            f"{cohort_name}/{age_band}; skipping."
        )
        con.close()
        return

    common_cols_sql = ", ".join(common_cols)
    common_cols_sql_control = ", ".join(f"c.{c}" for c in common_cols)

    # 1. Case patients from gold cohorts
    case_patients_query = f"""
        CREATE TEMP TABLE case_patients AS
        SELECT DISTINCT mi_person_key
        FROM read_parquet([{cohort_paths_literal}])
        WHERE is_target_case = 1
    """
    con.execute(case_patients_query)

    # Check number of cases; if zero, skip
    n_cases = con.execute("SELECT COUNT(*) FROM case_patients").fetchone()[0]
    if n_cases == 0:
        print(
            f"[WARN] No case patients found for {cohort_name}/{age_band}; skipping."
        )
        con.close()
        return

    # 2. Control candidates from gold medical + pharmacy, excluding patients
    #    with opioid ICDs and excluding case patients.
    opioid_condition = get_opioid_icd_sql_condition(table_alias="ue")

    control_candidates_query = f"""
        CREATE TEMP TABLE control_candidates AS
        WITH unified_gold_events AS (
            SELECT
                mi_person_key,
                primary_icd_diagnosis_code,
                two_icd_diagnosis_code,
                three_icd_diagnosis_code,
                four_icd_diagnosis_code,
                five_icd_diagnosis_code,
                six_icd_diagnosis_code,
                seven_icd_diagnosis_code,
                eight_icd_diagnosis_code,
                nine_icd_diagnosis_code,
                ten_icd_diagnosis_code
            FROM read_parquet([{gold_medical_paths_literal}])
            UNION ALL
            SELECT
                mi_person_key,
                NULL AS primary_icd_diagnosis_code,
                NULL AS two_icd_diagnosis_code,
                NULL AS three_icd_diagnosis_code,
                NULL AS four_icd_diagnosis_code,
                NULL AS five_icd_diagnosis_code,
                NULL AS six_icd_diagnosis_code,
                NULL AS seven_icd_diagnosis_code,
                NULL AS eight_icd_diagnosis_code,
                NULL AS nine_icd_diagnosis_code,
                NULL AS ten_icd_diagnosis_code
            FROM read_parquet([{gold_pharmacy_paths_literal}])
        ),
        per_patient_icd_check AS (
            SELECT
                mi_person_key,
                MAX(
                    CASE
                        WHEN {opioid_condition} THEN 1
                        ELSE 0
                    END
                ) AS has_opioid_icd
            FROM unified_gold_events ue
            GROUP BY mi_person_key
        )
        SELECT
            pp.mi_person_key
        FROM per_patient_icd_check pp
        LEFT JOIN case_patients cp
            ON pp.mi_person_key = cp.mi_person_key
        WHERE
            pp.has_opioid_icd = 0
            AND cp.mi_person_key IS NULL
    """
    con.execute(control_candidates_query)

    n_candidate_controls = con.execute(
        "SELECT COUNT(*) FROM control_candidates"
    ).fetchone()[0]
    if n_candidate_controls == 0:
        print(
            f"[WARN] No eligible control patients found for {cohort_name}/{age_band}; "
            f"using cases only."
        )
        # In this degenerate case, just build case-only events.
        final_query = f"""
            COPY (
                SELECT
                    *,
                    1 AS target
                FROM read_parquet([{cohort_paths_literal}])
                WHERE
                    is_target_case = 1 AND (
                        drug_name IN ({item_list_literal}) OR
                        {icd_conditions} OR
                        procedure_code IN ({item_list_literal})
                    )
            ) TO '{str(out_path)}'
            (FORMAT PARQUET)
        """
        con.execute(final_query)
        con.close()
        print(
            f"[INFO] Wrote case-only model_events.parquet for {cohort_name}/{age_band}: {out_path}"
        )
        return

    # 3. Sample control patients to maintain approximate sample_ratio:1 control:case
    desired_controls = int(sample_ratio * n_cases)
    if desired_controls <= 0:
        desired_controls = n_candidate_controls
    else:
        desired_controls = min(desired_controls, n_candidate_controls)

    con.execute(
        f"""
        CREATE TEMP TABLE control_patients AS
        SELECT mi_person_key
        FROM control_candidates
        ORDER BY random()
        LIMIT {desired_controls}
        """
    )

    # 4. Construct case and control events and write to Parquet
    case_events_query = f"""
        SELECT
            {common_cols_sql},
            1 AS target
        FROM read_parquet([{cohort_paths_literal}])
        WHERE
            is_target_case = 1 AND (
                drug_name IN ({item_list_literal}) OR
                {icd_conditions} OR
                procedure_code IN ({item_list_literal})
            )
    """

    control_events_query = f"""
        SELECT
            {common_cols_sql_control},
            0 AS target
        FROM read_parquet([{all_control_paths_literal}], union_by_name=True) c
        JOIN control_patients cp
            ON c.mi_person_key = cp.mi_person_key
    """

    final_query = f"""
        COPY (
            {case_events_query}
            UNION ALL
            {control_events_query}
        ) TO '{str(out_path)}'
        (FORMAT PARQUET)
    """

    con.execute(final_query)
    con.close()

    print(f"[INFO] Wrote model_events.parquet for {cohort_name}/{age_band}: {out_path}")


def main() -> None:
    # Ensure local directories exist (idempotent: we overwrite per file)
    MODEL_DATA_ROOT.mkdir(exist_ok=True)

    # Discover aggregated feature-importance CSVs
    aggregated_files = sorted(
        OUTPUTS_DIR.glob("*_aggregated_feature_importance.csv")
    )
    # Fallback: if outputs/ is empty locally, look under from_s3/by_cohort where
    # we may have downloaded aggregated feature-importance CSVs from S3.
    if not aggregated_files:
        alt_root = PROJECT_ROOT / "3_feature_importance" / "from_s3" / "by_cohort"
        if alt_root.exists():
            aggregated_files = sorted(
                alt_root.rglob("*_aggregated_feature_importance.csv")
            )
    if not aggregated_files:
        print(
            f"[WARN] No aggregated feature-importance CSVs found in "
            f"{OUTPUTS_DIR} or in 3_feature_importance/from_s3/by_cohort"
        )
        return

    # Default years: match feature-importance temporal setup (2016–2018 train, 2019 test)
    YEARS = [2016, 2017, 2018, 2019]

    local_cohort_root = resolve_local_cohort_root()
    local_medical_root = resolve_local_medical_root()
    local_pharmacy_root = resolve_local_pharmacy_root()

    for agg_path in aggregated_files:
        try:
            cohort_name, age_band = parse_aggregated_filename(agg_path)
        except ValueError as e:
            print(f"[WARN] Skipping {agg_path.name}: {e}")
            continue

        print(
            f"\n=== Processing cohort={cohort_name}, age_band={age_band} "
            f"from {agg_path.name} ==="
        )
        important_items = get_important_items(agg_path)
        if not important_items:
            print(
                f"[WARN] No important items extracted from {agg_path.name}; "
                f"skipping {cohort_name}/{age_band}."
            )
            continue

        filter_cohort_events_for_items(
            cohort_name=cohort_name,
            age_band=age_band,
            important_items=important_items,
            years=YEARS,
            output_root=MODEL_DATA_ROOT,
            local_cohort_root=local_cohort_root,
            local_medical_root=local_medical_root,
            local_pharmacy_root=local_pharmacy_root,
            sample_ratio=DEFAULT_SAMPLE_RATIO,
        )


if __name__ == "__main__":
    main()


