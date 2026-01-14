"""
Create a lookup table that maps numeric feature indices to feature names and
high-level descriptions, for a given (cohort, age_band) final feature table.

This is meant to support:
- Interpreting model coefficients / feature importances.
- FFA / rules tables where features are referred to by index.

Output:
  6_final_model/outputs/{cohort}/{age_band_fname}/
      {cohort}_{age_band_fname}_feature_lookup.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import json
import re
import sys

import pandas as pd

# Ensure project root is on path so we can import py_helpers
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from py_helpers.constants import age_band_to_fname  # type: ignore


def _classify_feature(name: str) -> Tuple[str, str]:
    """
    Best-effort classification of a feature into a group and a short
    human-readable description, based on naming conventions.
    """
    # ID / label
    if name == "mi_person_key":
        return ("id", "Patient identifier (string key)")
    if name == "target":
        return ("label", "Binary outcome label (0=control, 1=target event)")

    # FP-Growth itemsets
    if name.startswith("drug_name_itemset_") and name.endswith("_match"):
        return ("fpgrowth_drug_itemset", "Indicator for matching a specific drug-name FP-Growth itemset")
    if name == "drug_name_itemsets_matched_count":
        return ("fpgrowth_drug_summary", "Number of drug-name FP-Growth itemsets matched by the patient")
    if name == "drug_name_itemsets_max_support":
        return ("fpgrowth_drug_summary", "Maximum support among matched drug-name FP-Growth itemsets")

    if name.startswith("icd_code_itemset_") and name.endswith("_match"):
        return ("fpgrowth_icd_itemset", "Indicator for matching a specific ICD-code FP-Growth itemset")
    if name == "icd_code_itemsets_matched_count":
        return ("fpgrowth_icd_summary", "Number of ICD FP-Growth itemsets matched by the patient")
    if name == "icd_code_itemsets_max_support":
        return ("fpgrowth_icd_summary", "Maximum support among matched ICD FP-Growth itemsets")

    if name.startswith("cpt_code_itemset_") and name.endswith("_match"):
        return ("fpgrowth_cpt_itemset", "Indicator for matching a specific CPT-code FP-Growth itemset")
    if name == "cpt_code_itemsets_matched_count":
        return ("fpgrowth_cpt_summary", "Number of CPT FP-Growth itemsets matched by the patient")
    if name == "cpt_code_itemsets_max_support":
        return ("fpgrowth_cpt_summary", "Maximum support among matched CPT FP-Growth itemsets")

    if name.startswith("medical_code_itemset_") and name.endswith("_match"):
        return ("fpgrowth_med_itemset", "Indicator for matching a specific combined medical-code FP-Growth itemset")
    if name == "medical_code_itemsets_matched_count":
        return ("fpgrowth_med_summary", "Number of combined medical-code (ICD+CPT) FP-Growth itemsets matched")
    if name == "medical_code_itemsets_max_support":
        return ("fpgrowth_med_summary", "Maximum support among matched combined medical-code FP-Growth itemsets")

    # ICD structural encodings (primary diagnosis)
    if name.startswith("mean_icd_primary_") or name.startswith("max_icd_primary_"):
        return ("icd_primary_structural", "Aggregated structural feature derived from primary ICD diagnosis codes")

    # CPT structural encodings
    if name.startswith("mean_cpt_base_") or name.startswith("max_cpt_base_"):
        return ("cpt_structural", "Aggregated structural feature derived from CPT procedure codes")

    # ICD/CPT generic code encodings
    if name.startswith("mean_icd_code_") or name.startswith("max_icd_code_"):
        return ("icd_code_structural", "Aggregated structural feature derived from generic ICD code string properties")
    if name.startswith("mean_cpt_code_") or name.startswith("max_cpt_code_"):
        return ("cpt_code_structural", "Aggregated structural feature derived from generic CPT code string properties")

    # Drug-name structural encodings
    if name.startswith("mean_drug_") or name.startswith("max_drug_"):
        return ("drug_name_structural", "Aggregated structural feature derived from drug-name string properties")

    # DTW
    if "dtw_" in name:
        return ("dtw", "Feature derived from DTW trajectory analysis")

    # BupaR
    if "bupaR_" in name or "eventlog" in name.lower():
        return ("bupar", "Feature derived from BupaR process-mining outputs")

    # PGx
    if name.startswith("pgx_") or "pgx" in name.lower():
        return ("pgx", "Pharmacogenomics (PGx) feature")

    # Trajectory-level summaries already in model_data / DTW
    if "trajectory" in name.lower():
        return ("trajectory", "Trajectory-level summary feature")

    # Fallbacks for count-like features
    if name.startswith("n_") or name.endswith("_count"):
        return ("count", "Count-based feature (visits, events, codes, etc.)")

    # Generic fallback
    return ("other", "Other engineered feature (see generating module for details)")


def build_feature_lookup(cohort: str, age_band: str) -> pd.DataFrame:
    age_band_fname = age_band_to_fname(age_band)
    base = Path("6_final_model") / "outputs" / cohort / age_band_fname
    features_path = base / f"{cohort}_{age_band_fname}_train_final_features_no_leakage.csv"
    if not features_path.exists():
        raise FileNotFoundError(f"Final features file not found: {features_path}")

    df = pd.read_csv(features_path, nrows=1)

    # Reconstruct the numeric feature column order exactly as in run_final_model.py
    # (train_and_evaluate): drop id/label, then keep only numeric dtypes.
    feature_cols = [c for c in df.columns if c not in ("mi_person_key", "target")]
    numeric_feature_cols = [
        c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])
    ]

    # Best-effort loading of FP-Growth itemset definitions so we can attach
    # itemset contents (e.g. actual CPT/ICD/drug codes) to the lookup table.
    itemset_base = (
        Path("5b_fpgrowth_analysis")
        / "outputs"
        / cohort
        / "target"
        / age_band_fname
        / "train"
    )

    def _load_itemsets(kind: str) -> dict[int, list[str]]:
        path = itemset_base / f"{kind}_itemsets_target_only.json"
        if not path.exists():
            return {}
        try:
            with path.open() as f:
                arr = json.load(f)
        except Exception:
            return {}
        return {i: rec.get("itemsets", []) for i, rec in enumerate(arr)}

    drug_itemsets = _load_itemsets("drug_name")
    icd_itemsets = _load_itemsets("icd_code")
    cpt_itemsets = _load_itemsets("cpt_code")
    med_itemsets = _load_itemsets("medical_code")

    rows = []
    for idx, name in enumerate(numeric_feature_cols):
        group, desc = _classify_feature(name)
        itemset_type = ""
        itemset_contents: list[str] | None = None

        m = re.match(r"^(drug_name|icd_code|cpt_code|medical_code)_itemset_(\d+)_match$", name)
        if m:
            kind, idx_str = m.group(1), m.group(2)
            k = int(idx_str)
            if kind == "drug_name":
                itemset_type = "drug_name"
                itemset_contents = drug_itemsets.get(k, [])
            elif kind == "icd_code":
                itemset_type = "icd_code"
                itemset_contents = icd_itemsets.get(k, [])
            elif kind == "cpt_code":
                itemset_type = "cpt_code"
                itemset_contents = cpt_itemsets.get(k, [])
            elif kind == "medical_code":
                itemset_type = "medical_code"
                itemset_contents = med_itemsets.get(k, [])

        rows.append(
            {
                "feature_index": idx,
                "feature_name": name,
                "group": group,
                "description": desc,
                "itemset_type": itemset_type,
                "itemset_items": "|".join(itemset_contents) if itemset_contents else "",
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a numeric feature index→name lookup table for the final model."
    )
    parser.add_argument("--cohort", required=True, help="Cohort name, e.g. opioid_ed")
    parser.add_argument("--age_band", required=True, help="Age band, e.g. 13-24")
    args = parser.parse_args()

    lookup_df = build_feature_lookup(args.cohort, args.age_band)
    age_band_fname = age_band_to_fname(args.age_band)
    out_dir = Path("6_final_model") / "outputs" / args.cohort / age_band_fname
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.cohort}_{age_band_fname}_feature_lookup.csv"
    lookup_df.to_csv(out_path, index=False)
    print(f"Saved feature lookup to {out_path}")

    # Also mirror into a central feature_encoding_outputs folder for
    # cross-module inspection and documentation.
    fe_base = PROJECT_ROOT / "feature_encoding_outputs" / args.cohort / age_band_fname
    fe_base.mkdir(parents=True, exist_ok=True)
    fe_path = fe_base / f"{args.cohort}_{age_band_fname}_feature_lookup.csv"
    lookup_df.to_csv(fe_path, index=False)
    print(f"Saved feature lookup to {fe_path}")


if __name__ == "__main__":
    main()

