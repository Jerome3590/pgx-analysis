#!/usr/bin/env python3
"""
Create model-ready event-level data filtered to important features.

This script:
1. Reads aggregated feature-importance CSVs from `3_feature_importance/outputs/`
   (files named like: {cohort}_{age_band}_aggregated_feature_importance.csv)
2. Extracts the `feature` column (e.g., `item_99284`, `item_AMOXICILLIN`) and
   strips the `item_` prefix to get raw item codes.
3. Filters the GOLD cohorts parquet data to only events where ANY of the
   event-level item columns match one of the important items:
   - drug_name
   - primary_icd_diagnosis_code
   - two_icd_diagnosis_code
   - three_icd_diagnosis_code
   - four_icd_diagnosis_code
   - five_icd_diagnosis_code
   - procedure_code
4. Writes the filtered event-level data to:
   model_data/cohort_name={cohort_name}/age_band={age_band}/model_events.parquet

This output can then be used as input for:
 - FP-Growth (pattern mining on important features only)
 - bupaR (process mining / event-log analysis)
 - DTW (trajectory analysis on filtered event sequences)

Local data path resolution mirrors the feature-importance utilities:
 - Use LOCAL_DATA_PATH env var if set
 - Otherwise, try project-root-relative `data/cohorts_F1120`
 - Finally, fall back to EC2 path `/mnt/nvme/cohorts`
"""

import os
from pathlib import Path
from typing import List, Tuple

import duckdb
import pandas as pd


PROJECT_ROOT = Path(__file__).parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "3_feature_importance" / "outputs"
MODEL_DATA_ROOT = PROJECT_ROOT / "model_data"


def resolve_local_data_path() -> Path:
    """Resolve the local cohorts data path."""
    env_path = os.getenv("LOCAL_DATA_PATH")
    if env_path:
        return Path(env_path)

    # Project-relative default (Windows/Linux dev)
    project_data = PROJECT_ROOT / "data" / "cohorts_F1120"
    if project_data.exists():
        return project_data

    # EC2 default
    return Path("/mnt/nvme/cohorts")


def parse_aggregated_filename(path: Path) -> Tuple[str, str]:
    """
    Parse cohort_name and age_band from an aggregated CSV filename.

    Current pattern (from 3_feature_importance/outputs):
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
    # Suffix has exactly 3 tokens: "aggregated", "feature", "importance".
    if len(parts) < 5:
        raise ValueError(f"Unexpected aggregated filename format: {path.name}")
    # cohort_name may contain underscores; everything before the last 5 tokens
    # (age_band_fname + 3-word suffix) belongs to cohort_name.
    cohort_name_tokens = parts[:-5]  # e.g. ['opioid', 'ed']
    age_band_tokens = parts[-5:-3]  # e.g. ['0', '12']

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
    local_data_path: Path,
    years: List[int],
    output_root: Path,
) -> int:
    """
    Filter GOLD cohort event-level data by important items and write to model_data/.

    - Reads cohort.parquet for the given (cohort_name, age_band, event_year ∈ years)
    - Keeps rows where ANY of the item-bearing columns match an important item
    - Writes combined events to:
        model_data/cohort_name={cohort_name}/age_band={age_band}/model_events.parquet
    """
    con = duckdb.connect()

    item_list_literal = ", ".join(f"'{v}'" for v in important_items)
    if not item_list_literal:
        print(f"[WARN] No important items for {cohort_name}/{age_band}; skipping.")
        return

    filtered_frames = []

    for year in years:
        parquet_path = (
            local_data_path
            / f"cohort_name={cohort_name}"
            / f"event_year={year}"
            / f"age_band={age_band}"
            / "cohort.parquet"
        )

        if not parquet_path.exists():
            print(f"[INFO] Missing parquet for {cohort_name}/{age_band}/{year}: {parquet_path}")
            continue

        print(f"[INFO] Filtering events for {cohort_name}/{age_band}/{year} using {len(important_items)} items...")

        query = f"""
            SELECT *
            FROM read_parquet('{parquet_path}')
            WHERE
                drug_name IN ({item_list_literal}) OR
                primary_icd_diagnosis_code IN ({item_list_literal}) OR
                two_icd_diagnosis_code IN ({item_list_literal}) OR
                three_icd_diagnosis_code IN ({item_list_literal}) OR
                four_icd_diagnosis_code IN ({item_list_literal}) OR
                five_icd_diagnosis_code IN ({item_list_literal}) OR
                six_icd_diagnosis_code IN ({item_list_literal}) OR
                seven_icd_diagnosis_code IN ({item_list_literal}) OR
                eight_icd_diagnosis_code IN ({item_list_literal}) OR
                nine_icd_diagnosis_code IN ({item_list_literal}) OR
                ten_icd_diagnosis_code IN ({item_list_literal}) OR
                procedure_code IN ({item_list_literal})
        """
        df_year = con.execute(query).df()
        if not df_year.empty:
            df_year["event_year"] = year
            filtered_frames.append(df_year)

    con.close()

    if not filtered_frames:
        print(f"[WARN] No matching events found for {cohort_name}/{age_band} across years {years}.")
        return 0

    filtered_all = pd.concat(filtered_frames, ignore_index=True)

    out_dir = (
        output_root
        / f"cohort_name={cohort_name}"
        / f"age_band={age_band}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "model_events.parquet"

    print(f"[INFO] Writing filtered events to {out_path} ({len(filtered_all)} rows)...")
    # Use DuckDB's Python API to write parquet directly
    con_out = duckdb.connect()
    con_out.register("filtered_all", filtered_all)
    con_out.execute(
        f"COPY filtered_all TO '{str(out_path)}' (FORMAT 'parquet')"
    )
    con_out.close()
    print(f"[INFO] Done for {cohort_name}/{age_band}.")

    # Return number of unique patients in the filtered target set
    if "mi_person_key" in filtered_all.columns:
        return int(filtered_all["mi_person_key"].nunique())
    return 0


def export_control_events_for_age_band(
    control_cohort: str,
    age_band: str,
    n_target_patients: int,
    local_data_path: Path,
    years: List[int],
    output_root: Path,
) -> None:
    """
    Export control cohort events for the given age band into model_data/,
    downsampled to achieve approximately a 5:1 control:target ratio at the
    patient level.

    Control events keep the full column space and are NOT filtered by the
    important item list; we just randomly sample control patients.
    """
    out_dir = (
        output_root
        / f"cohort_name={control_cohort}"
        / f"age_band={age_band}"
    )
    out_path = out_dir / "model_events.parquet"

    if out_path.exists():
        print(f"[INFO] Control model_events already exists for {control_cohort}/{age_band}; skipping export.")
        return

    if n_target_patients <= 0:
        print(f"[WARN] No target patients for {age_band}; skipping control export.")
        return

    # Use DuckDB end-to-end to avoid loading full control cohort into memory.
    con = duckdb.connect()

    # Build a DuckDB view over all available control parquet files for this age band / years.
    parquet_paths = []
    for year in years:
        parquet_path = (
            local_data_path
            / f"cohort_name={control_cohort}"
            / f"event_year={year}"
            / f"age_band={age_band}"
            / "cohort.parquet"
        )
        if parquet_path.exists():
            parquet_paths.append(str(parquet_path))
        else:
            print(f"[INFO] Missing control parquet for {control_cohort}/{age_band}/{year}: {parquet_path}")

    if not parquet_paths:
        print(f"[WARN] No control events found for {control_cohort}/{age_band} across years {years}.")
        con.close()
        return

    # Create a temp view combining all years
    paths_literal = ", ".join(f"'{p}'" for p in parquet_paths)
    con.execute(
        f"CREATE TEMP VIEW control_all AS SELECT * FROM read_parquet([{paths_literal}])"
    )

    # Ensure mi_person_key is present
    has_key = con.execute(
        "SELECT 1 FROM information_schema.columns "
        "WHERE table_name = 'control_all' AND column_name = 'mi_person_key' LIMIT 1"
    ).fetchone()

    if not has_key:
        print(f"[WARN] Control data for {control_cohort}/{age_band} lacks mi_person_key; exporting all rows.")
        out_dir.mkdir(parents=True, exist_ok=True)
        con.execute(
            f"COPY (SELECT * FROM control_all) TO '{str(out_path)}' (FORMAT 'parquet')"
        )
        con.close()
        print(f"[INFO] Done exporting control for {control_cohort}/{age_band}.")
        return

    # Compute desired control patient count (5:1 ratio), bounded by available controls
    n_control = con.execute(
        "SELECT COUNT(DISTINCT mi_person_key) FROM control_all"
    ).fetchone()[0]
    desired = min(n_control, 5 * n_target_patients)
    if desired <= 0:
        print(f"[WARN] Computed desired control count <= 0 for {control_cohort}/{age_band}; skipping export.")
        con.close()
        return

    print(f"[INFO] Sampling {desired} control patients (available={n_control}, target={n_target_patients})...")

    con.execute(
        f"""
        CREATE TEMP TABLE sampled_ids AS
        SELECT mi_person_key
        FROM (
            SELECT DISTINCT mi_person_key FROM control_all
        )
        USING SAMPLE {desired} ROWS
        """
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Writing control events to {out_path} (target patients={n_target_patients})...")
    con.execute(
        f"""
        COPY (
            SELECT *
            FROM control_all
            WHERE mi_person_key IN (SELECT mi_person_key FROM sampled_ids)
        ) TO '{str(out_path)}' (FORMAT 'parquet')
        """
    )
    con.close()
    print(f"[INFO] Done exporting control for {control_cohort}/{age_band}.")


def main():
    local_data_path = resolve_local_data_path()
    print(f"[INFO] Using local cohorts data from: {local_data_path}")
    MODEL_DATA_ROOT.mkdir(exist_ok=True)

    aggregated_files = sorted(
        OUTPUTS_DIR.glob("*_aggregated_feature_importance.csv")
    )
    if not aggregated_files:
        print(f"[WARN] No aggregated feature-importance CSVs found in {OUTPUTS_DIR}")
        return

    # Only process TARGET cohorts for item-based filtering.
    # For now, we hard-code opioid_ed as the target cohort and non_opioid_ed as control.
    TARGET_COHORTS = {"opioid_ed"}
    CONTROL_COHORT = "non_opioid_ed"

    # Default years: match feature-importance temporal setup (2016–2018 train, 2019 test)
    YEARS = [2016, 2017, 2018, 2019]

    for agg_path in aggregated_files:
        try:
            cohort_name, age_band = parse_aggregated_filename(agg_path)
        except ValueError as e:
            print(f"[WARN] Skipping {agg_path.name}: {e}")
            continue

        if cohort_name not in TARGET_COHORTS:
            print(f"[INFO] Skipping non-target cohort '{cohort_name}' from {agg_path.name}")
            continue

        print(f"\n=== Processing {cohort_name} / {age_band} from {agg_path.name} ===")
        important_items = get_important_items(agg_path)
        if not important_items:
            print(f"[WARN] No important items extracted from {agg_path.name}; skipping.")
            continue

        n_target_patients = filter_cohort_events_for_items(
            cohort_name=cohort_name,
            age_band=age_band,
            important_items=important_items,
            local_data_path=local_data_path,
            years=YEARS,
            output_root=MODEL_DATA_ROOT,
        )

        # Also export the matching control cohort (downsampled),
        # so downstream analyses can see both target and control for this age band.
        if n_target_patients > 0:
            export_control_events_for_age_band(
                control_cohort=CONTROL_COHORT,
                age_band=age_band,
                n_target_patients=n_target_patients,
                local_data_path=local_data_path,
                years=YEARS,
                output_root=MODEL_DATA_ROOT,
            )


if __name__ == "__main__":
    main()


