#!/usr/bin/env python3
"""
Convert 3a_feature_importance CSV outputs to dashboard JSONs (index + heatmaps + single-age feature lists).
Run from repo root after Step 3a; used by 4_dashboard_visuals and deploy.
Outputs: feature_importance_index.json, {cohort}/plots/{cohort}_{model}_fi_heatmap.json, {cohort}/plots/{cohort}_{model}_{age_band}_fi.json
"""
import argparse
import sys
from pathlib import Path

# Allow running from repo root or py_helpers
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from py_helpers.feature_importance_heatmap import build_fi_dashboard_jsons


def main():
    ap = argparse.ArgumentParser(description="Convert 3a FI CSVs to dashboard JSONs (filter by cohort/model/age_band).")
    ap.add_argument("--outputs-base", type=Path, default=REPO_ROOT / "3a_feature_importance" / "outputs", help="Base dir for 3a outputs")
    ap.add_argument("--top-n", type=int, default=50, help="Top N features per age band for heatmaps")
    ap.add_argument("--single-top-n", type=int, default=100, help="Top N for single-age-band JSONs")
    args = ap.parse_args()
    if not args.outputs_base.exists():
        print(f"Outputs base not found: {args.outputs_base}", file=sys.stderr)
        sys.exit(1)
    result = build_fi_dashboard_jsons(
        args.outputs_base,
        top_n=args.top_n,
        single_band_top_n=args.single_top_n,
    )
    print("Index:", result["index"])
    print("Written", len(result["written"]), "files:")
    for p in result["written"][:20]:
        print(" ", p)
    if len(result["written"]) > 20:
        print(" ... and", len(result["written"]) - 20, "more")
    return 0


if __name__ == "__main__":
    sys.exit(main())
