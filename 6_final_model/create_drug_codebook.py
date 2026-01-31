"""
Create a numeric codebook for all distinct drug names observed in the
event-level model data for a given (cohort, age_band).

For each unique `drug_name`, we compute the full string-based encoding
from `py_helpers.categorical_encoding.encode_drug_name_series` and save
the result as a table:

  - drug_id: integer ID (0-based)
  - drug_name_raw: original string as found in model_events.parquet
  - drug_name_normalized: lowercase / stripped variant used for encoding
  - all encoded numeric features (one column per dimension)

Output path:
  6_final_model/outputs/{cohort}/{age_band_fname}/
      {cohort}_{age_band_fname}_drug_codebook.csv

This codebook can be used to interpret SHAP and FFA outputs by mapping
drug-related feature values back to representative drug strings.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import duckdb
import pandas as pd

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from py_helpers.constants import age_band_to_fname  # type: ignore
from py_helpers.categorical_encoding import encode_drug_name_series  # type: ignore


def build_drug_codebook(cohort: str, age_band: str) -> pd.DataFrame:
    age_band_fname = age_band_to_fname(age_band)
    events_path = (
        PROJECT_ROOT
        / "4_model_data"
        / f"cohort_name={cohort}"
        / f"age_band={age_band}"
        / "model_events.parquet"
    )
    if not events_path.exists():
        raise FileNotFoundError(f"Model data not found: {events_path}")

    # Use DuckDB to pull distinct drug names efficiently from the Parquet file
    con = duckdb.connect()
    df_drugs = con.execute(
        f"""
        SELECT DISTINCT drug_name
        FROM read_parquet('{events_path}')
        WHERE drug_name IS NOT NULL AND TRIM(drug_name) <> ''
        """
    ).df()
    con.close()

    if df_drugs.empty:
        raise ValueError(f"No non-empty drug_name values found in {events_path}")

    df_drugs["drug_name_raw"] = df_drugs["drug_name"].astype(str)
    # Normalized version used for encoding; keep both for traceability
    df_drugs["drug_name_normalized"] = (
        df_drugs["drug_name_raw"].str.strip().str.lower()
    )

    # Compute numeric encodings for the normalized names
    enc = encode_drug_name_series(df_drugs["drug_name_normalized"], prefix="drug")

    codebook = pd.concat([df_drugs[["drug_name_raw", "drug_name_normalized"]], enc], axis=1)
    codebook.insert(0, "drug_id", range(codebook.shape[0]))

    return codebook


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a numeric drug-name codebook for a cohort/age_band."
    )
    parser.add_argument("--cohort", required=True, help="Cohort name, e.g. opioid_ed")
    parser.add_argument("--age_band", required=True, help="Age band, e.g. 13-24")
    args = parser.parse_args()

    codebook = build_drug_codebook(args.cohort, args.age_band)

    age_band_fname = age_band_to_fname(args.age_band)
    out_dir = (
        PROJECT_ROOT / "6_final_model" / "outputs" / args.cohort / age_band_fname
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = (
        out_dir / f"{args.cohort}_{age_band_fname}_drug_codebook.csv"
    )
    codebook.to_csv(out_path, index=False)
    print(f"Saved drug codebook to {out_path}")

    # Mirror into central feature_encoding_outputs folder for consistent access
    fe_base = PROJECT_ROOT / "feature_encoding_outputs" / args.cohort / age_band_fname
    fe_base.mkdir(parents=True, exist_ok=True)
    fe_path = fe_base / f"{args.cohort}_{age_band_fname}_drug_codebook.csv"
    codebook.to_csv(fe_path, index=False)
    print(f"Saved drug codebook to {fe_path}")


if __name__ == "__main__":
    main()

