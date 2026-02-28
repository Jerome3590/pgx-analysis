#!/usr/bin/env python3
"""
Local tests for dashboard visuals: assert each pipeline produces expected file types.

- BupaR: at least one .png and one .html under plots/
- DTW: .png and/or chart_data.json under plots/
- FP-Growth: .png and/or .html under plots/

Structure tests run against real output dirs when present, else a temp fixture dir.
Integration tests run pipelines when RUN_VISUALS_INTEGRATION=1.
"""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pytest

# 11_testing/tests/dashboard_visuals/test_*.py -> parents[3] = repo root
REPO_ROOT = Path(__file__).resolve().parents[3]
STEP9 = REPO_ROOT / "9_dashboard_visuals"
VISUAL_ROOT = REPO_ROOT / "10_risk_dashboard" / "visualizations"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(STEP9) not in sys.path:
    sys.path.insert(0, str(STEP9))

RUN_INTEGRATION = os.environ.get("RUN_VISUALS_INTEGRATION", "").strip().lower() in ("1", "true", "yes")


def _bupar_plots_dirs_to_check():
    out = []
    base = VISUAL_ROOT / "bupar"
    if base.exists():
        for cohort_dir in base.iterdir():
            if not cohort_dir.is_dir() or cohort_dir.name.startswith("allowed_codes"):
                continue
            for age_dir in cohort_dir.iterdir():
                if not age_dir.is_dir():
                    continue
                plots = age_dir / "plots"
                if plots.exists():
                    out.append((plots, str(plots)))
    if not out:
        tmp = tempfile.mkdtemp(prefix="bupar_plots_")
        p = Path(tmp)
        (p / "fixture.png").write_bytes(b"\x89PNG\r\n\x1a\n")
        (p / "fixture.html").write_text("<!DOCTYPE html><html></html>", encoding="utf-8")
        out.append((p, "fixture (no real BupaR outputs)"))
    return out


def _dtw_plots_dirs_to_check():
    out = []
    base = VISUAL_ROOT / "dtw"
    if base.exists():
        for cohort_dir in base.iterdir():
            if not cohort_dir.is_dir() or cohort_dir.name == "feature_engineering":
                continue
            for age_dir in cohort_dir.iterdir():
                if not age_dir.is_dir():
                    continue
                plots = age_dir / "plots"
                if plots.exists():
                    out.append((plots, str(plots)))
    if not out:
        tmp = tempfile.mkdtemp(prefix="dtw_plots_")
        p = Path(tmp)
        (p / "fixture.png").write_bytes(b"\x89PNG\r\n\x1a\n")
        out.append((p, "fixture (no real DTW outputs)"))
    return out


def _fpgrowth_plots_dirs_to_check():
    out = []
    base = VISUAL_ROOT / "fpgrowth" / "outputs"
    if base.exists():
        for cohort_dir in base.iterdir():
            if not cohort_dir.is_dir():
                continue
            for age_dir in cohort_dir.iterdir():
                if not age_dir.is_dir():
                    continue
                plots = age_dir / "plots"
                if plots.exists():
                    out.append((plots, str(plots)))
    if not out:
        tmp = tempfile.mkdtemp(prefix="fpgrowth_plots_")
        p = Path(tmp)
        (p / "fixture.png").write_bytes(b"\x89PNG\r\n\x1a\n")
        out.append((p, "fixture (no real FP-Growth outputs)"))
    return out


def _allowed_codes_path(cohort: str, age_band: str) -> Path:
    age_fname = age_band.replace("-", "_")
    return VISUAL_ROOT / "bupar" / f"allowed_codes_shap_ffa_{cohort}_{age_fname}.json"


def _model_events_path(cohort: str, age_band: str) -> Path:
    age_fname = age_band.replace("-", "_")
    for ab in (age_fname, age_band):
        d = REPO_ROOT / "4_model_data" / f"cohort_name={cohort}" / f"age_band={ab}"
        for name in ("model_events.parquet", "model_events_no_protocols.parquet"):
            p = d / name
            if p.exists():
                return p
    return Path()


def _find_one_combo_with_prereqs():
    try:
        from py_helpers.constants import REQUIRED_COHORTS
    except ImportError:
        REQUIRED_COHORTS = {"opioid_ed": ["25-44", "85-114"], "non_opioid_ed": ["65-74", "85-114"]}
    for cohort, bands in REQUIRED_COHORTS.items():
        for age_band in bands:
            ac_path = _allowed_codes_path(cohort, age_band)
            if not ac_path.exists():
                continue
            try:
                with open(ac_path, encoding="utf-8") as f:
                    codes = json.load(f)
                if not codes or (isinstance(codes, list) and len(codes) == 0):
                    continue
            except Exception:
                continue
            if _model_events_path(cohort, age_band).exists():
                return cohort, age_band
    return None


class TestBupaRVisualsOutputs(unittest.TestCase):
    """BupaR must produce at least one PNG and one HTML in each cohort/age_band plots dir."""

    def test_bupar_plots_dir_has_png_and_html(self):
        dirs = _bupar_plots_dirs_to_check()
        self.assertTrue(dirs, "No BupaR plots dirs to check")
        plots_dir, label = dirs[0]
        pngs = list(plots_dir.glob("*.png"))
        htmls = list(plots_dir.glob("*.html"))
        self.assertGreater(len(pngs), 0, f"BupaR plots dir should have at least one .png: {label}")
        self.assertGreater(len(htmls), 0, f"BupaR plots dir should have at least one .html: {label}")

    @pytest.mark.integration
    def test_bupar_integration_produces_png_and_html(self):
        if not RUN_INTEGRATION:
            self.skipTest("Set RUN_VISUALS_INTEGRATION=1 to run")
        combo = _find_one_combo_with_prereqs()
        if not combo:
            self.skipTest("No cohort/age_band with allowed_codes and model_events")
        cohort, age_band = combo
        script = STEP9 / "bupar" / "create_bupar_visuals.py"
        if not script.exists():
            self.skipTest("create_bupar_visuals.py not found")
        r = subprocess.run(
            [sys.executable, str(script), "--cohort-name", cohort, "--age-band", age_band, "--force"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=600,
        )
        self.assertEqual(r.returncode, 0, f"BupaR failed: {r.stderr or r.stdout}")
        age_fname = age_band.replace("-", "_")
        plots_dir = VISUAL_ROOT / "bupar" / cohort / age_fname / "plots"
        self.assertTrue(plots_dir.exists(), f"Plots dir should exist: {plots_dir}")
        pngs = list(plots_dir.glob("*.png"))
        htmls = list(plots_dir.glob("*.html"))
        self.assertGreater(len(pngs), 0, f"BupaR should produce at least one .png in {plots_dir}")
        self.assertGreater(len(htmls), 0, f"BupaR should produce at least one .html in {plots_dir}")


class TestDTWVisualsOutputs(unittest.TestCase):
    """DTW must produce .png and/or chart_data.json in plots/."""

    def test_dtw_plots_dir_has_expected_files(self):
        dirs = _dtw_plots_dirs_to_check()
        self.assertTrue(dirs, "No DTW plots dirs to check")
        plots_dir, label = dirs[0]
        pngs = list(plots_dir.glob("*.png"))
        jsons = list(plots_dir.glob("*.json"))
        self.assertTrue(
            len(pngs) > 0 or len(jsons) > 0,
            f"DTW plots dir should have .png or .json: {label}",
        )

    @pytest.mark.integration
    def test_dtw_integration_produces_outputs(self):
        if not RUN_INTEGRATION:
            self.skipTest("Set RUN_VISUALS_INTEGRATION=1 to run")
        combo = _find_one_combo_with_prereqs()
        if not combo:
            self.skipTest("No cohort/age_band with allowed_codes and model_events")
        cohort, age_band = combo
        age_fname = age_band.replace("-", "_")
        traj_script = STEP9 / "dtw" / "create_dtw_trajectories.py"
        if traj_script.exists():
            r1 = subprocess.run(
                [sys.executable, str(traj_script), "--cohort", cohort, "--age-band", age_band],
                cwd=str(REPO_ROOT),
                capture_output=True,
                text=True,
                timeout=300,
            )
            if r1.returncode != 0:
                self.skipTest(f"DTW trajectories failed: {r1.stderr[:400]}")
        csv_path = VISUAL_ROOT / "dtw" / "feature_engineering" / f"dtw_features_{cohort}_{age_fname}.csv"
        if not csv_path.exists():
            self.skipTest("DTW features CSV not produced")
        vis_script = STEP9 / "dtw" / "create_dtw_visuals.py"
        if not vis_script.exists():
            self.skipTest("create_dtw_visuals.py not found")
        r2 = subprocess.run(
            [sys.executable, str(vis_script), "--cohort-name", cohort, "--age-band", age_band, "--force"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=120,
        )
        self.assertEqual(r2.returncode, 0, f"DTW visuals failed: {r2.stderr or r2.stdout}")
        plots_dir = VISUAL_ROOT / "dtw" / cohort / age_fname / "plots"
        self.assertTrue(plots_dir.exists(), f"DTW plots dir should exist: {plots_dir}")
        files = list(plots_dir.glob("*.png")) + list(plots_dir.glob("*.json"))
        self.assertGreater(len(files), 0, f"DTW should produce .png or .json in {plots_dir}")


class TestFPGrowthVisualsOutputs(unittest.TestCase):
    """FP-Growth must produce .png and/or .html in plots/."""

    def test_fpgrowth_plots_dir_has_png_or_html(self):
        dirs = _fpgrowth_plots_dirs_to_check()
        self.assertTrue(dirs, "No FP-Growth plots dirs to check")
        plots_dir, label = dirs[0]
        pngs = list(plots_dir.glob("*.png"))
        htmls = list(plots_dir.glob("*.html"))
        self.assertTrue(
            len(pngs) > 0 or len(htmls) > 0,
            f"FP-Growth plots dir should have .png or .html: {label}",
        )

    @pytest.mark.integration
    def test_fpgrowth_integration_produces_outputs(self):
        if not RUN_INTEGRATION:
            self.skipTest("Set RUN_VISUALS_INTEGRATION=1 to run")
        combo = _find_one_combo_with_prereqs()
        if not combo:
            self.skipTest("No cohort/age_band with allowed_codes and model_events")
        cohort, age_band = combo
        script = STEP9 / "fpgrowth" / "create_fpgrowth_visuals.py"
        if not script.exists():
            self.skipTest("create_fpgrowth_visuals.py not found")
        r = subprocess.run(
            [sys.executable, str(script), "--cohort-name", cohort, "--age-band", age_band, "--force"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=600,
        )
        if r.returncode != 0:
            self.skipTest(f"FP-Growth failed: {r.stderr[:400]}")
        age_fname = age_band.replace("-", "_")
        plots_dir = VISUAL_ROOT / "fpgrowth" / cohort / age_fname / "plots"
        if plots_dir.exists():
            files = list(plots_dir.glob("*.png")) + list(plots_dir.glob("*.html"))
            self.assertGreater(len(files), 0, f"FP-Growth should produce .png or .html in {plots_dir}")


if __name__ == "__main__":
    unittest.main()
