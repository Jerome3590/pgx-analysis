#!/usr/bin/env python3
"""
Build final training feature table for cohort 1 (opioid_ed), age band 0-12.

This script merges:
- Base target patient list from model_data
- BupaR pre-/post-F1120 sequence features
- DTW trajectory features (prototype DTW distances)

Output:
- 7_final_model/outputs/opioid_ed_0_12/opioid_ed_0_12_train_final_features.csv

The resulting table is patient-level (one row per mi_person_key) and is intended
to be used as input to final modeling scripts (CatBoost / RF with 200 splits).
"""

import argparse
from pathlib import Path

import duckdb
import pandas as pd


def build_final_features(project_root: Path) -> None:
    cohort_name = "opioid_ed"
    age_band = "0-12"
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
    # Source 2: BupaR pre-/post-F1120 patient features
    # ------------------------------------------------------------------
    bupar_root = (
        project_root
        / "5_bupaR_analysis"
        / "outputs"
        / cohort_name
        / age_band_fname
        / "features"
    )

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

    if not pre_bupar_csv.exists():
        raise FileNotFoundError(f"Pre-F1120 BupaR features not found: {pre_bupar_csv}")
    if not post_bupar_csv.exists():
        raise FileNotFoundError(f"Post-F1120 BupaR features not found: {post_bupar_csv}")
    if not time_to_bupar_csv.exists():
        raise FileNotFoundError(
            f"Time-to-F1120 BupaR features not found: {time_to_bupar_csv}"
        )

    pre_df = pd.read_csv(pre_bupar_csv)
    post_df = pd.read_csv(post_bupar_csv)
    time_to_df = pd.read_csv(time_to_bupar_csv)

    # In BupaR outputs, the ID column is case_id
    if "case_id" in pre_df.columns:
        pre_df = pre_df.rename(columns={"case_id": "mi_person_key"})
    if "case_id" in post_df.columns:
        post_df = post_df.rename(columns={"case_id": "mi_person_key"})
    if "case_id" in time_to_df.columns:
        time_to_df = time_to_df.rename(columns={"case_id": "mi_person_key"})

    print(
        f"[INFO] Loaded BupaR pre-F1120 features for {len(pre_df)} patients, "
        f"post-F1120 features for {len(post_df)} patients, "
        f"and time-to-F1120 features for {len(time_to_df)} patients"
    )

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
    merged = (
        base_df
        .merge(pre_df, on="mi_person_key", how="left")
        .merge(post_df, on="mi_person_key", how="left", suffixes=("_pre", "_post"))
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
            "Build final patient-level feature table for opioid_ed, age 0-12, "
            "combining model_data targets with BupaR and DTW features."
        )
    )
    parser.add_argument(
        "--project-root",
        type=str,
        default=".",
        help="Project root path (default: current directory)",
    )
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    build_final_features(project_root)


if __name__ == "__main__":
    main()


