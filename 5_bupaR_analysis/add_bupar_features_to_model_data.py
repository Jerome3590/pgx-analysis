import argparse
from pathlib import Path

import duckdb
import pandas as pd


def add_bupar_features(
    project_root: Path,
    cohort_name: str,
    age_band: str,
    train_label: str = "train",
) -> None:
    """
    Merge per-patient BupaR features (pre/post F1120) into a tabular dataset
    derived from model_data for a given cohort/age_band.

    This script is currently wired for cohort 1 (opioid_ed), age 0-12, but the
    paths are parameterized so it can be generalized later.
    """

    age_band_fname = age_band.replace("-", "_")

    # Paths
    model_data_path = (
        project_root
        / "model_data"
        / f"cohort_name={cohort_name}"
        / f"age_band={age_band}"
        / "model_events.parquet"
    )

    bupar_output_dir = (
        project_root
        / "5_bupaR_analysis"
        / "outputs"
        / cohort_name
        / age_band_fname
    )

    pre_features_csv = (
        bupar_output_dir
        / f"{cohort_name}_{age_band_fname}_{train_label}_target_pre_f1120_patient_features_bupar.csv"
    )
    post_features_csv = (
        bupar_output_dir
        / f"{cohort_name}_{age_band_fname}_{train_label}_target_post_f1120_patient_features_bupar.csv"
    )

    if not model_data_path.exists():
        raise FileNotFoundError(f"model_data parquet not found: {model_data_path}")

    if not pre_features_csv.exists():
        raise FileNotFoundError(f"Pre-F1120 BupaR features not found: {pre_features_csv}")

    if not post_features_csv.exists():
        raise FileNotFoundError(f"Post-F1120 BupaR features not found: {post_features_csv}")

    print(f"[INFO] Reading model_data from {model_data_path}")
    con = duckdb.connect()
    # Build a patient-level base table (one row per mi_person_key)
    base_df = con.execute(
        f"""
        SELECT DISTINCT mi_person_key
        FROM read_parquet('{model_data_path}')
        WHERE target = 1
        """
    ).df()
    con.close()

    print(f"[INFO] Loaded {len(base_df)} unique target patients from model_data")

    print(f"[INFO] Reading pre-F1120 features from {pre_features_csv}")
    pre_df = pd.read_csv(pre_features_csv)

    print(f"[INFO] Reading post-F1120 features from {post_features_csv}")
    post_df = pd.read_csv(post_features_csv)

    # Expect case_id column from BupaR outputs; rename to mi_person_key for consistency
    if "case_id" in pre_df.columns:
        pre_df = pre_df.rename(columns={"case_id": "mi_person_key"})
    if "case_id" in post_df.columns:
        post_df = post_df.rename(columns={"case_id": "mi_person_key"})

    # Merge features
    merged = (
        base_df
        .merge(pre_df, on="mi_person_key", how="left")
        .merge(post_df, on="mi_person_key", how="left")
    )

    out_dir = (
        project_root
        / "5_bupaR_analysis"
        / "outputs"
        / cohort_name
        / age_band_fname
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f"{cohort_name}_{age_band_fname}_{train_label}_target_bupar_patient_features_merged.csv"
    print(f"[INFO] Writing merged BupaR features to {out_path} ({len(merged)} rows)")
    merged.to_csv(out_path, index=False)
    print("[INFO] Done.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Merge BupaR pre/post-F1120 patient features into a tabular dataset "
            "derived from model_data (currently tested on cohort 1, age 0-12)."
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
        help="Cohort name (default: opioid_ed)",
    )
    parser.add_argument(
        "--age-band",
        type=str,
        default="0-12",
        help="Age band (default: 0-12)",
    )
    parser.add_argument(
        "--train-label",
        type=str,
        default="train",
        help="Training window label used in filenames (default: train)",
    )

    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    add_bupar_features(
        project_root=project_root,
        cohort_name=args.cohort_name,
        age_band=args.age_band,
        train_label=args.train_label,
    )


if __name__ == "__main__":
    main()


