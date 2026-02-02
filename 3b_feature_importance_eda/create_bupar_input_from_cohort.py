#!/usr/bin/env python3
"""
Build BupaR input event data from cohort data + Step 3a aggregated feature importance + target.

This allows Step 3b BupaR (target leakage identification) to run before Step 4.
Uses gold cohort parquet, gold medical/pharmacy, and 3a aggregated FI to produce
model_events-like parquet with the same schema (target, event_year, drug_name, ICDs, procedure_code).

Output: 3b_feature_importance_eda/outputs/cohorts/input_model_data/cohort_name={slug}/age_band={band}/model_events.parquet
R script (create_bupar_outputs_*.R) looks for this path first, then falls back to Step 4 output.
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]  # 3b_feature_importance_eda -> repo root
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Step 4 logic for building event table from cohort + medical + pharmacy
from py_helpers.constants import get_cohort_slug
from py_helpers.env_utils import get_data_root

# Import from 4_model_data (same project)
sys.path.insert(0, str(PROJECT_ROOT))
from importlib.util import spec_from_file_location, module_from_spec
_create_model_data = PROJECT_ROOT / "4_model_data" / "create_model_data.py"
_spec = spec_from_file_location("create_model_data", _create_model_data)
_mod = module_from_spec(_spec)
_spec.loader.exec_module(_mod)
filter_cohort_events_for_items = _mod.filter_cohort_events_for_items
resolve_local_cohort_root = _mod.resolve_local_cohort_root
resolve_local_medical_root = _mod.resolve_local_medical_root
resolve_local_pharmacy_root = _mod.resolve_local_pharmacy_root
get_important_items = _mod.get_important_items
DEFAULT_SAMPLE_RATIO = getattr(_mod, "DEFAULT_SAMPLE_RATIO", 5.0)


def main():
    parser = argparse.ArgumentParser(
        description="Build BupaR input from cohort data + 3a aggregated FI + target"
    )
    parser.add_argument("--cohort", required=True, help="Cohort name (e.g. opioid_ed, non_opioid_ed)")
    parser.add_argument("--age-band", required=True, dest="age_band", help="Age band (e.g. 13-24)")
    args = parser.parse_args()

    cohort_name = args.cohort
    age_band = args.age_band
    age_band_fname = age_band.replace("-", "_")
    years = [2016, 2017, 2018, 2019]

    # 3a aggregated feature importance path
    outputs_3a = PROJECT_ROOT / "3a_feature_importance" / "outputs"
    agg_csv = (
        outputs_3a
        / cohort_name
        / age_band
        / f"{cohort_name}_{age_band_fname}_aggregated_feature_importance.csv"
    )
    if not agg_csv.exists():
        print(f"[ERROR] Step 3a aggregated FI not found: {agg_csv}")
        print("        Run Step 3a for this cohort/age_band first (2_feature_importance.ipynb).")
        sys.exit(1)

    important_items = get_important_items(agg_csv)
    if not important_items:
        print(f"[WARN] No important items in {agg_csv}; building with all events (no FI filter).")

    # Output under 3b so R finds it first
    output_root = PROJECT_ROOT / "3b_feature_importance_eda" / "outputs"
    local_cohort_root = resolve_local_cohort_root()
    local_medical_root = resolve_local_medical_root()
    local_pharmacy_root = resolve_local_pharmacy_root()

    print(f"[INFO] Building BupaR input from cohort data + 3a aggregated FI + target")
    print(f"       Cohort: {cohort_name}, age_band: {age_band}")
    print(f"       Important items: {len(important_items)}")
    print(f"       Output: {output_root}/cohorts/input_model_data/cohort_name={get_cohort_slug(age_band)}/age_band={age_band}/model_events.parquet")

    filter_cohort_events_for_items(
        cohort_name=cohort_name,
        age_band=age_band,
        important_items=important_items,
        years=years,
        output_root=output_root,
        local_cohort_root=local_cohort_root,
        local_medical_root=local_medical_root,
        local_pharmacy_root=local_pharmacy_root,
        sample_ratio=DEFAULT_SAMPLE_RATIO,
        control_exclusions=None,
        skip_s3_download=True,  # Build from cohort + 3a FI only; do not pull from Step 4 S3
    )
    print("[INFO] BupaR input built successfully.")


if __name__ == "__main__":
    main()
