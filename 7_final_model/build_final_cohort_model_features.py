#!/usr/bin/env python3
"""
Build final training feature table for a given cohort and age band.

This script merges, for a specified `(cohort_name, age_band)`:
- Base target patient list from `model_data`
- BupaR sequence and time-to-event features
- DTW trajectory features (prototype DTW distances)

Outputs a patient-level CSV (one row per `mi_person_key`) under:
  `7_final_model/outputs/{cohort_name}/{age_band_fname}/{cohort_name}_{age_band_fname}_train_final_features.csv`

Currently supported cohorts:
- `opioid_ed`  – expects BupaR files with F1120 naming (`pre_f1120`, `post_f1120`, `time_to_f1120`)
- `non_opioid_ed` – expects BupaR files with HCG naming (`pre_hcg`, `time_to_hcg`)
"""

import argparse
from pathlib import Path

import duckdb
import pandas as pd


def build_final_features(project_root: Path, cohort_name: str, age_band: str) -> None:
    """
    Build final features for a specific cohort and age band.

    Parameters
    ----------
    project_root : Path
        Project root directory.
    cohort_name : str
        Cohort identifier, e.g. "opioid_ed" or "non_opioid_ed".
    age_band : str
        Age band string, e.g. "0-12", "13-24", "65-74".
    """
    age_band_fname = age_band.replace("-", "_")

    # ------------------------------------------------------------------
    # Source 1: Base target patient list from model_data
    # ------------------------------------------------------------------
    model_data_path = (
        project_root
        / "model_data"
        / f"cohort_name={cohort_name}"
        / f"age_band={age_band}"
        / "model_events.parquet"
    )

    if not model_data_path.exists():
        raise FileNotFoundError(f"model_data parquet not found: {model_data_path}")

    con = duckdb.connect()
    base_df = con.execute(
        f"""
        SELECT DISTINCT mi_person_key
        FROM read_parquet('{model_data_path}')
        WHERE target = 1
        """
    ).df()
    con.close()

    print(f"[INFO] Loaded {len(base_df)} target patients from {model_data_path}")

    # ------------------------------------------------------------------
    # Source 2: BupaR patient features (pre/post + time-to-event)
    # ------------------------------------------------------------------
    bupar_root = (
        project_root
        / "5_bupaR_analysis"
        / "outputs"
        / cohort_name
        / age_band_fname
        / "features"
    )

    # Filenames depend on cohort
    if cohort_name == "opioid_ed":
        pre_bupar_csv = (
            bupar_root
            / f"{cohort_name}_{age_band_fname}_train_target_pre_f1120_patient_features_bupar.csv"
        )
        post_bupar_csv = (
            bupar_root
            / f"{cohort_name}_{age_band_fname}_train_target_post_f1120_patient_features_bupar.csv"
        )
        time_to_bupar_csv = (
            bupar_root
            / f"{cohort_name}_{age_band_fname}_train_target_time_to_f1120_features_bupar.csv"
        )
    elif cohort_name == "non_opioid_ed":
        pre_bupar_csv = (
            bupar_root
            / f"{cohort_name}_{age_band_fname}_train_target_pre_hcg_patient_features_bupar.csv"
        )
        # Polypharmacy pipeline does not define post-HCG features (descriptive only)
        post_bupar_csv = None
        time_to_bupar_csv = (
            bupar_root
            / f"{cohort_name}_{age_band_fname}_train_target_time_to_hcg_features_bupar.csv"
        )
    else:
        raise ValueError(
            f"Unsupported cohort_name for BupaR feature merging: {cohort_name}"
        )

    if not pre_bupar_csv.exists():
        raise FileNotFoundError(f"Pre-target BupaR features not found: {pre_bupar_csv}")
    if time_to_bupar_csv is None or not time_to_bupar_csv.exists():
        raise FileNotFoundError(
            f"Time-to-event BupaR features not found: {time_to_bupar_csv}"
        )

    pre_df = pd.read_csv(pre_bupar_csv)
    time_to_df = pd.read_csv(time_to_bupar_csv)
    post_df = None

    if post_bupar_csv is not None and post_bupar_csv.exists():
        post_df = pd.read_csv(post_bupar_csv)
    elif cohort_name == "opioid_ed":
        # For opioid_ed we expect post-F1120; for safety, require it
        raise FileNotFoundError(f"Post-target BupaR features not found: {post_bupar_csv}")

    # In BupaR outputs, the ID column is case_id
    if "case_id" in pre_df.columns:
        pre_df = pre_df.rename(columns={"case_id": "mi_person_key"})
    if post_df is not None and "case_id" in post_df.columns:
        post_df = post_df.rename(columns={"case_id": "mi_person_key"})
    if "case_id" in time_to_df.columns:
        time_to_df = time_to_df.rename(columns={"case_id": "mi_person_key"})

    msg = (
        f"[INFO] Loaded BupaR pre-target features for {len(pre_df)} patients, "
        f"time-to-event features for {len(time_to_df)} patients"
    )
    if post_df is not None:
        msg += f", post-target features for {len(post_df)} patients"
    print(msg)

    # ------------------------------------------------------------------
    # Source 3: DTW trajectory features (prototype distances)
    # ------------------------------------------------------------------
    dtw_root = (
        project_root
        / "6_dtw_analysis"
        / "outputs"
        / cohort_name
        / age_band_fname
        / "features"
    )
    dtw_csv = dtw_root / f"{cohort_name}_{age_band_fname}_train_target_dtw_features.csv"

    if not dtw_csv.exists():
        raise FileNotFoundError(f"DTW features not found: {dtw_csv}")

    dtw_df = pd.read_csv(dtw_csv)
    print(f"[INFO] Loaded DTW features for {len(dtw_df)} patients")

    # ------------------------------------------------------------------
    # Merge all features on mi_person_key
    # ------------------------------------------------------------------
    merged = base_df.merge(pre_df, on="mi_person_key", how="left")
    if post_df is not None:
        merged = merged.merge(post_df, on="mi_person_key", how="left", suffixes=("_pre", "_post"))
    merged = (
        merged
        .merge(time_to_df, on="mi_person_key", how="left")
        .merge(dtw_df, on="mi_person_key", how="left")
    )

    out_dir = project_root / "7_final_model" / "outputs" / cohort_name / age_band_fname
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{cohort_name}_{age_band_fname}_train_final_features.csv"

    print(f"[INFO] Writing final feature table to {out_path} ({len(merged)} rows)")
    merged.to_csv(out_path, index=False)
    print("[INFO] Done.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build final patient-level feature table for a cohort/age_band, "
            "combining model_data targets with BupaR and DTW features."
        )
    )
    parser.add_argument(
        "--project-root",
        type=str,
        default=".",
        help="Project root path (default: current directory)",
    )
    parser.add_argument(
        "--cohort-name",
        type=str,
        default="opioid_ed",
        help="Cohort name (e.g. opioid_ed, non_opioid_ed). Default: opioid_ed",
    )
    parser.add_argument(
        "--age-band",
        type=str,
        default="0-12",
        help="Age band string, e.g. '0-12', '13-24', '65-74'. Default: 0-12",
    )
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    build_final_features(project_root, cohort_name=args.cohort_name, age_band=args.age_band)


if __name__ == "__main__":
    main()


