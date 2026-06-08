#!/usr/bin/env python3
"""One-off helper: rename dashboard causal → scenario in notebooks and markdown."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# Order matters: longer / more specific patterns first.
REPLACEMENTS: list[tuple[str, str]] = [
    ("10_risk_dashboard/visualizations/causal", "10_risk_dashboard/visualizations/scenario"),
    ("visualizations/{bupar,dtw,fpgrowth,cohort_pgx,causal}", "visualizations/{bupar,dtw,fpgrowth,cohort_pgx,scenario}"),
    ("visualizations/{causal,bupar,dtw,fpgrowth,cohort_pgx}", "visualizations/{scenario,bupar,dtw,fpgrowth,cohort_pgx}"),
    ("feature_importance,causal,cohort_pgx", "feature_importance,scenario,cohort_pgx"),
    ("cohort_pgx/networks/, causal/)", "cohort_pgx/networks/, scenario/)"),
    ("visualizations/causal/", "visualizations/scenario/"),
    ("visualizations/causal", "visualizations/scenario"),
    ("GET /visualizations/causal", "GET /visualizations/scenario"),
    ("causal_data.json", "scenario_data.json"),
    ("causal_data_url", "scenario_data_url"),
    ("CAUSAL_VISUALS", "SCENARIO_VISUALS"),
    ("upload_causal_script", "upload_scenario_script"),
    ("combined_causal", "combined_scenario"),
    ("causal_jsons", "scenario_jsons"),
    ("causal_base", "scenario_base"),
    ('print("--- Causal ---")', 'print("--- Scenario ---")'),
    ("Causal (sample JSON preview)", "Scenario (sample JSON preview)"),
    ("combine SHAP+FFA / Causal step", "combine SHAP+FFA / Scenario step"),
    ("FP-Growth, Causal, and Cohort PGx", "FP-Growth, Scenario, and Cohort PGx"),
    ("# --- Causal (dashboard_data.json", "# --- Scenario (dashboard_data.json"),
    ('_vis / "causal"', '_vis / "scenario"'),
    ('base / "causal"', 'base / "scenario"'),
    ("Upload Causal dashboard", "Upload Scenario dashboard"),
    ("checks Feature Importance, Causal,", "checks Feature Importance, Scenario,"),
    ("FI, Causal, BupaR", "FI, Scenario, BupaR"),
    ("and causal data to S3", "and scenario data to S3"),
    ("uploads causal JSON", "uploads scenario JSON"),
    ("Causal Analysis tab", "Scenario Analysis tab"),
    ("Causal tab", "Scenario tab"),
    ("**Causal**", "**Scenario**"),
    ("| Causal |", "| Scenario |"),
    ("Causal, BupaR", "Scenario, BupaR"),
    ("BupaR, DTW, FP-Growth, Causal", "BupaR, DTW, FP-Growth, Scenario"),
    ("Causal (served from SHAP/FFA", "Scenario Analysis (served from SHAP/FFA"),
    ("Causal (FFA, SHAP", "Scenario (FFA, SHAP"),
    ("causal and cohort_pgx use hyphen", "scenario and cohort_pgx use hyphen"),
    ("# **Causal (Scenario Analysis tab)**", "# **Scenario Analysis tab**"),
    ("# 1. **Setup** – Resolve paths (scripts in `9_dashboard_visuals/`; outputs under `10_risk_dashboard/visualizations/{causal,bupar,dtw,fpgrowth,cohort_pgx}/`).",
     "# 1. **Setup** – Resolve paths (scripts in `9_dashboard_visuals/`; outputs under `10_risk_dashboard/visualizations/{scenario,bupar,dtw,fpgrowth,cohort_pgx}/`)."),
    (".../causal/{cohort}/{age_band}/{bin}/", ".../scenario/{cohort}/{age_band}/{bin}/"),
    (".../causal/{cohort}/{age_band}/", ".../scenario/{cohort}/{age_band}/"),
    ("Default causal path is per-bin", "Default scenario path is per-bin"),
    ("# ── FFA causal —", "# ── FFA scenario factors —"),
]

PROTECTED = (
    "ffa_causal_factors",
    "causal_responsibility",
    "n_causal_features",
    "causal_csv",
    "causal_csv_error",
    "causal_importance",
    "causal-synergy",
    "causally related",
    "Causal Drivers",
    "Causal Calculator",
    "Causal Temporal",
    "Causal Rules",
    "causal analysis tab.png",
    "test_causal",
    "btnLoadCausal",
    "causal-analysis",
    "causal-n-event-bin",
    "causal_factors",
    "/causal/importance",
    "/causal/interactions",
    "POST /causal",
)

NOTEBOOKS = [
    REPO / "3_model_train_shap_ffa.ipynb",
    REPO / "4_dashboard_visuals.ipynb",
    REPO / "5_build_and_deploy.ipynb",
]

MARKDOWN = [
    REPO / "README_execution_workflow.md",
    REPO / "README_dashboard_visuals.md",
    REPO / "10_risk_dashboard/data_preparation/README.md",
    REPO / "10_risk_dashboard/README.md",
    REPO / "9_dashboard_visuals/README.md",
]


def _apply(text: str) -> str:
    for old, new in REPLACEMENTS:
        text = text.replace(old, new)
    return text


def _patch_file(path: Path) -> bool:
    raw = path.read_text(encoding="utf-8")
    patched = _apply(raw)
    if patched != raw:
        path.write_text(patched, encoding="utf-8")
        return True
    return False


def _patch_notebook(path: Path) -> int:
    nb = json.loads(path.read_text(encoding="utf-8"))
    n = 0
    for cell in nb.get("cells", []):
        src = cell.get("source", [])
        if not src:
            continue
        joined = "".join(src)
        new_joined = _apply(joined)
        if new_joined != joined:
            cell["source"] = new_joined.splitlines(keepends=True)
            if cell["source"] and not cell["source"][-1].endswith("\n"):
                cell["source"][-1] += "\n"
            n += 1
    if n:
        path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return n


def main() -> int:
    py_scripts = list((REPO / "py_helpers" / "vs_code_jupyter_notebook_scripts").glob("*.py"))
    changed: list[str] = []
    for p in py_scripts + MARKDOWN:
        if p.exists() and _patch_file(p):
            changed.append(str(p.relative_to(REPO)))
    for nb in NOTEBOOKS:
        if nb.exists():
            cells = _patch_notebook(nb)
            if cells:
                changed.append(f"{nb.relative_to(REPO)} ({cells} cells)")
    # Deduplicate stale dual cleanup paths in notebook 3 script
    nb3_script = REPO / "py_helpers/vs_code_jupyter_notebook_scripts/3_model_train_shap_ffa_windows_local_test.py"
    if nb3_script.exists():
        t = nb3_script.read_text(encoding="utf-8")
        t2 = re.sub(
            r'(PROJECT_ROOT / "10_risk_dashboard" / "visualizations" / "scenario" / cohort,\n\s*)'
            r'PROJECT_ROOT / "10_risk_dashboard" / "visualizations" / "scenario" / cohort,\n',
            r"\1",
            t,
        )
        t2 = re.sub(
            r'(f"s3://\{S3_BUCKET\}/visualizations/scenario/\{cohort\}/",\n\s*)'
            r'f"s3://\{S3_BUCKET\}/visualizations/scenario/\{cohort\}/",\n',
            r"\1",
            t2,
        )
        if t2 != t:
            nb3_script.write_text(t2, encoding="utf-8")
            changed.append("deduped 3_model_train cleanup paths")
    print("Updated:")
    for c in changed:
        print(f"  - {c}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
