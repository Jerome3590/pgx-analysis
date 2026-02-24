#!/usr/bin/env python3
"""
Build CPT code dictionary from SHAP/FFA allowed codes (final feature importance).

Reads all allowed_codes_shap_ffa_{cohort}_{age_band}.json files, extracts CPT codes,
and writes cpt_code_dictionary.csv with columns: cpt_code, first_three, definition.
Only CPT codes that have final feature importance (appear in allowed codes) are included.
Definitions are left empty for you to fill from AMA/CMS or your reference.

Usage (from repo root):
  python 9_dashboard_visuals/fpgrowth/code_mappings/build_cpt_dictionary_from_allowed_codes.py
  python 9_dashboard_visuals/fpgrowth/code_mappings/build_cpt_dictionary_from_allowed_codes.py --project-root /path/to/pgx-analysis
"""

import argparse
import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
BUPAR_OUTPUTS = REPO_ROOT / "10_risk_dashboard" / "visualizations" / "bupar" / "outputs"


def _normalize_code(s: str) -> str:
    if not s or (isinstance(s, float) and str(s) == "nan"):
        return ""
    return str(s).strip().replace(".", "").replace("-", "")


def _collect_cpt_from_allowed_files(project_root: Path) -> set:
    """Collect all unique normalized CPT codes from allowed_codes_shap_ffa_*.json."""
    out_dir = project_root / "10_risk_dashboard" / "visualizations" / "bupar" / "outputs"
    if not out_dir.exists():
        return set()
    try:
        from py_helpers.shap_ffa_fpgrowth_utils import _parse_feature_name
    except ImportError:
        _parse_feature_name = None

    cpt_set = set()
    for path in sorted(out_dir.glob("allowed_codes_shap_ffa_*.json")):
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
        for c in raw:
            s = str(c).strip() if c is not None else ""
            if not s:
                continue
            norm = _normalize_code(s)
            if not norm:
                continue
            if s.startswith("cpt_"):
                cpt_set.add(_normalize_code(s[4:]))
            elif _parse_feature_name:
                typ, code = _parse_feature_name(s)
                if typ == "cpt":
                    raw_norm = _normalize_code(code) if code else norm
                    cpt_set.add(raw_norm)
    return cpt_set


def main() -> int:
    ap = argparse.ArgumentParser(description="Build CPT dictionary from allowed codes (final feature importance)")
    ap.add_argument("--project-root", type=Path, default=REPO_ROOT, help="Project root (default: auto)")
    ap.add_argument("-o", "--output", type=Path, default=None, help="Output CSV (default: cpt_code_dictionary.csv in this dir)")
    args = ap.parse_args()
    project_root = args.project_root.resolve()
    if args.output is None:
        out_path = Path(__file__).resolve().parent / "cpt_code_dictionary.csv"
    else:
        out_path = args.output.resolve()

    cpt_codes = _collect_cpt_from_allowed_files(project_root)
    if not cpt_codes:
        print("No CPT codes found in allowed_codes_shap_ffa_*.json. Run pipeline to generate allowed codes first.", file=sys.stderr)
        return 1

    # Preserve existing definitions if file exists
    existing_definitions = {}
    if out_path.exists():
        with open(out_path, encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                c = (row.get("cpt_code") or "").strip()
                d = (row.get("definition") or "").strip()
                if c:
                    existing_definitions[c] = d

    rows = []
    for code in sorted(cpt_codes):
        first_three = code[:3] if len(code) >= 3 else code
        definition = existing_definitions.get(code, "")
        rows.append((code, first_three, definition))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cpt_code", "first_three", "definition"])
        w.writerows(rows)

    print(f"Wrote {len(rows)} CPT codes (with final feature importance) to {out_path}")
    print("Fill the 'definition' column from AMA/CMS or your reference (e.g. https://www.cms.gov/medicare/physician-fee-schedule/search).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
