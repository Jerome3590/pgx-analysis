#!/usr/bin/env python3
"""
Export Chapter 6 manuscript figures for the PGx Risk Dashboard (all tabs).

1) **Pipeline visuals** — Copies the same PNG assets the deployed dashboard loads
   (see `10_risk_dashboard/visualizations/dashboard_visual_objects.json` and
   `10_risk_dashboard/backend/lambda_function.py` visualization handlers).
   Sources mirror local outputs from Step 9 (`9_dashboard_visuals` / notebook 4/5).

2) **Tab 2 (risk score)** — Copies `manuscript/figures/ch05/fig_dashboard.pdf`, the same
   representative screenshot used in Chapter 5.

3) **Optional `--ui`** — Starts `python -m http.server` in `10_risk_dashboard/frontend`
   and uses Playwright to screenshot the static UI (tabs load from `tabs/*.html`; API
   calls may fail—layouts are still authentic). Requires:
     pip install playwright
     playwright install chromium

Usage (repo root):

  python manuscript/scripts/export_ch6_dashboard_figures.py --cohort opioid_ed --age-band 25-44
  python manuscript/scripts/export_ch6_dashboard_figures.py --cohort opioid_ed --age-band 25-44 --ui

Outputs go to: manuscript/figures/ch06/
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
MANUSCRIPT = REPO_ROOT / "manuscript"
OUT_DIR = MANUSCRIPT / "figures" / "ch06"
FRONTEND = REPO_ROOT / "10_risk_dashboard" / "frontend"
VIS = REPO_ROOT / "10_risk_dashboard" / "visualizations"
CH5_DASHBOARD_PDF = MANUSCRIPT / "figures" / "ch05" / "pgx_dashboard.pdf"


def _age_fname(age_band: str) -> str:
    return age_band.replace("-", "_")


def _base(cohort: str, age_band: str) -> str:
    return f"{cohort}_{_age_fname(age_band)}"


def _copy_if_exists(src: Path, dst: Path) -> bool:
    if not src.is_file():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def _first_existing(paths: Iterable[Path]) -> Optional[Path]:
    for p in paths:
        if p.is_file():
            return p
    return None


def copy_pipeline_pngs(cohort: str, age_band: str, density_bin: Optional[str]) -> list[Tuple[str, bool]]:
    """Copy BupaR, FP-Growth, DTW PNGs. Try per-bin BupaR/FP paths first when density_bin is set."""
    age_band_fname = _age_fname(age_band)
    b = _base(cohort, age_band)
    results: list[Tuple[str, bool]] = []

    # ── BupaR: overall activity frequency (dashboard: activity_frequency_image) ──
    bupar_candidates: list[Path] = []
    if density_bin:
        bupar_candidates.append(
            VIS / "bupar" / cohort / age_band_fname / "density" / density_bin / "plots" / f"{b}_overall_activity_frequency.png"
        )
    bupar_candidates.extend(
        [
            VIS / "bupar" / cohort / age_band_fname / "plots" / f"{b}_overall_activity_frequency.png",
        ]
    )
    src_b = _first_existing(bupar_candidates)
    ok = _copy_if_exists(src_b, OUT_DIR / "ch06_tab4_bupar_overall_activity_frequency.png") if src_b else False
    results.append(("BupaR overall activity frequency PNG", ok))

    # ── FP-Growth: itemsets bar chart + target rules network (same keys as Lambda) ──
    fpg_dir_candidates: list[Path] = []
    if density_bin:
        fpg_dir_candidates.append(VIS / "fpgrowth" / cohort / age_band / "density" / density_bin / "plots")
    fpg_dir_candidates.extend(
        [
            VIS / "fpgrowth" / cohort / age_band / "plots",
            VIS / "fpgrowth" / "outputs" / cohort / age_band_fname / "plots",
        ]
    )
    fpg_plots: Optional[Path] = None
    for d in fpg_dir_candidates:
        if d.is_dir():
            fpg_plots = d
            break
    itemsets_name = f"{b}_drug_name_combined_top_itemsets.png"
    network_name = f"{b}_drug_name_target_rules_network.png"
    ok_item = False
    ok_net = False
    if fpg_plots:
        ok_item = _copy_if_exists(fpg_plots / itemsets_name, OUT_DIR / "ch06_tab4_fpgrowth_top_itemsets.png")
        ok_net = _copy_if_exists(fpg_plots / network_name, OUT_DIR / "ch06_tab4_fpgrowth_target_rules_network.png")
    results.append(("FP-Growth top itemsets PNG", ok_item))
    results.append(("FP-Growth target rules network PNG", ok_net))

    # ── DTW: trajectory analysis PNG (dashboard fallbacks) ──
    dtw_candidates: list[Path] = []
    if density_bin:
        dtw_candidates.append(
            VIS / "dtw" / cohort / age_band_fname / "density" / density_bin / "plots" / f"dtw_trajectory_analysis_{b}.png"
        )
    dtw_candidates.append(VIS / "dtw" / cohort / age_band_fname / "plots" / f"dtw_trajectory_analysis_{b}.png")
    src_d = _first_existing(dtw_candidates)
    ok_d = _copy_if_exists(src_d, OUT_DIR / "ch06_tab4_dtw_trajectory_analysis.png") if src_d else False
    results.append(("DTW trajectory analysis PNG", ok_d))

    return results


def copy_tab2_pdf() -> bool:
    """Same representative Risk Score panel as Chapter 5."""
    dst = OUT_DIR / "ch06_tab2_risk_score.pdf"
    if not CH5_DASHBOARD_PDF.is_file():
        return False
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(CH5_DASHBOARD_PDF, dst)
    return True


def run_playwright_ui(port: int = 8765) -> list[Tuple[str, bool]]:
    """Screenshot primary dashboard tabs (static HTML shell; optional API failures)."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("Playwright not installed. Run: pip install playwright && playwright install chromium", file=sys.stderr)
        return [("playwright", False)]

    if not (FRONTEND / "index.html").is_file():
        print(f"Frontend not found: {FRONTEND / 'index.html'}", file=sys.stderr)
        return [("frontend", False)]

    server = subprocess.Popen(
        [sys.executable, "-m", "http.server", str(port)],
        cwd=str(FRONTEND),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    time.sleep(1.2)
    results: list[Tuple[str, bool]] = []
    url = f"http://127.0.0.1:{port}/index.html"

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1400, "height": 900})
            page.goto(url, wait_until="networkidle", timeout=120_000)
            page.wait_for_timeout(2500)

            # Tab 1 flow: Risk Assessment (age + code summary) then Drugs (code pickers)
            page.screenshot(path=str(OUT_DIR / "ch06_tab1_risk_assessment_ui.png"), full_page=False)
            results.append(("ch06_tab1_risk_assessment_ui.png", True))

            page.evaluate("switchTab('drugs')")
            page.wait_for_timeout(1500)
            page.screenshot(path=str(OUT_DIR / "ch06_tab1_drugs_codes_ui.png"), full_page=False)
            results.append(("ch06_tab1_drugs_codes_ui.png", True))

            # Tab 3: Causal Analysis (What-If / deprescribing context in Chapter 5)
            page.evaluate("switchTab('causal-analysis')")
            page.wait_for_timeout(2000)
            page.screenshot(path=str(OUT_DIR / "ch06_tab3_causal_analysis_ui.png"), full_page=False)
            results.append(("ch06_tab3_causal_analysis_ui.png", True))

            # Tab 5: PGx Patient Card
            page.evaluate("switchTab('pgx-card')")
            page.wait_for_timeout(2000)
            page.screenshot(path=str(OUT_DIR / "ch06_tab5_pgx_card_ui.png"), full_page=False)
            results.append(("ch06_tab5_pgx_card_ui.png", True))

            # Exploratory row: optional single screenshot of BupaR tab panel (developer tab matching Tab 4 narrative)
            page.evaluate("switchTab('bupar-visualizations')")
            page.wait_for_timeout(2000)
            page.screenshot(path=str(OUT_DIR / "ch06_tab4_bupar_dashboard_shell_ui.png"), full_page=False)
            results.append(("ch06_tab4_bupar_dashboard_shell_ui.png", True))

            browser.close()
    finally:
        server.terminate()
        try:
            server.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server.kill()

    return results


def main() -> int:
    ap = argparse.ArgumentParser(description="Export CH6 dashboard tab figures from pipeline + optional UI snapshots.")
    ap.add_argument("--cohort", default="opioid_ed", help="Cohort partition (default: opioid_ed)")
    ap.add_argument("--age-band", default="25-44", help="Age band with hyphen (default: 25-44)")
    ap.add_argument(
        "--density-bin",
        default=None,
        choices=("low", "medium", "high", "extreme"),
        help="Optional density stratum for per-bin BupaR/FP/DTW paths",
    )
    ap.add_argument("--ui", action="store_true", help="Also capture Playwright screenshots of local frontend")
    ap.add_argument("--port", type=int, default=8765, help="http.server port for --ui")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {OUT_DIR}")
    ok_pdf = copy_tab2_pdf()
    print(f"[{'OK' if ok_pdf else 'SKIP'}] Tab 2 risk score PDF (from CH5 pgx_dashboard)")

    for label, ok in copy_pipeline_pngs(args.cohort, args.age_band, args.density_bin):
        print(f"[{'OK' if ok else 'MISS'}] {label}")

    if args.ui:
        print("Capturing UI with Playwright (local http.server + Chromium)…")
        for label, ok in run_playwright_ui(port=args.port):
            print(f"[{'OK' if ok else 'FAIL'}] {label}")

    missing = []
    for name in (
        "ch06_tab2_risk_score.pdf",
        "ch06_tab4_bupar_overall_activity_frequency.png",
        "ch06_tab4_fpgrowth_top_itemsets.png",
        "ch06_tab4_fpgrowth_target_rules_network.png",
        "ch06_tab4_dtw_trajectory_analysis.png",
    ):
        if not (OUT_DIR / name).is_file():
            missing.append(name)

    if missing:
        print()
        print("Missing pipeline assets (sync EC2 outputs or run 9_dashboard_visuals for this cohort/age_band):")
        for m in missing:
            print(f"  - {m}")
        print("See: 9_dashboard_visuals/run_dashboard_visuals.py and .cursorrules (dashboard visuals).")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
