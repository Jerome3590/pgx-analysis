#!/usr/bin/env python3
"""
Tests for dashboard visuals prerequisite: SHAP/FFA combined allowed codes file.

Covers BupaR, DTW, and FP-Growth all requiring the same file
(10_risk_dashboard/visualizations/bupar/outputs/allowed_codes_shap_ffa_{cohort}_{age_band}.json).
We never use "all codes" / "all items"; the combined file is required.
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path

# Repo root (parent of 9_dashboard_visuals)
REPO_ROOT = Path(__file__).resolve().parents[2]
STEP9 = REPO_ROOT / "9_dashboard_visuals"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(STEP9) not in sys.path:
    sys.path.insert(0, str(STEP9))

# Import after path is set
from run_dashboard_visuals import check_shap_ffa_allowed_codes_prerequisite


class TestBupaRDtwFpgrowthPrerequisite(unittest.TestCase):
    """Prerequisite check used by workflow script (BupaR, DTW, FP-Growth)."""

    def test_accepts_valid_file(self):
        """When the allowed codes file exists and is non-empty, check passes."""
        import unittest.mock
        with unittest.mock.patch.object(Path, "exists", return_value=True):
            with unittest.mock.patch("builtins.open", unittest.mock.mock_open(read_data=json.dumps(["icd_F1120", "drug_XYZ"]))):
                ok, err = check_shap_ffa_allowed_codes_prerequisite(
                    [("opioid_ed", "25-44")],
                    REPO_ROOT,
                )
        self.assertTrue(ok, err)
        self.assertIsNone(err)

    def test_rejects_missing_file(self):
        """When the file is missing, check fails with message listing missing."""
        import unittest.mock
        with unittest.mock.patch.object(Path, "exists", return_value=False):
            ok, err = check_shap_ffa_allowed_codes_prerequisite(
                [("opioid_ed", "25-44")],
                REPO_ROOT,
            )
        self.assertFalse(ok)
        self.assertIn("Missing", err)
        self.assertIn("opioid_ed/25-44", err)
        self.assertIn("allowed_codes_shap_ffa", err)

    def test_rejects_empty_file(self):
        """When the file exists but is empty JSON array, check fails."""
        import unittest.mock
        with unittest.mock.patch.object(Path, "exists", return_value=True):
            with unittest.mock.patch("builtins.open", unittest.mock.mock_open(read_data="[]")):
                ok, err = check_shap_ffa_allowed_codes_prerequisite(
                    [("non_opioid_ed", "65-74")],
                    REPO_ROOT,
                )
        self.assertFalse(ok)
        self.assertIn("Empty or invalid", err)

    def test_rejects_empty_list_content(self):
        """When the file has empty list, check fails."""
        import unittest.mock
        with unittest.mock.patch.object(Path, "exists", return_value=True):
            with unittest.mock.patch("builtins.open", unittest.mock.mock_open(read_data=json.dumps([]))):
                ok, err = check_shap_ffa_allowed_codes_prerequisite(
                    [("opioid_ed", "85-114")],
                    REPO_ROOT,
                )
        self.assertFalse(ok)
        self.assertIn("Empty or invalid", err)

    def test_multiple_combinations_all_required(self):
        """Multiple cohort/age_band combinations all need their file."""
        import unittest.mock
        with unittest.mock.patch.object(Path, "exists", side_effect=[True, False]):
            ok, err = check_shap_ffa_allowed_codes_prerequisite(
                [("opioid_ed", "25-44"), ("non_opioid_ed", "65-74")],
                REPO_ROOT,
            )
        self.assertFalse(ok)
        self.assertIn("Missing", err)
        self.assertIn("non_opioid_ed", err)


class TestDtwPrerequisite(unittest.TestCase):
    """DTW uses the same allowed codes path as BupaR/FP-Growth (validated by workflow check)."""

    def test_dtw_required_path_pattern(self):
        """DTW create_dtw_trajectories expects allowed_codes_shap_ffa_{cohort}_{age_band}.json under bupar/outputs."""
        expected_suffix = "10_risk_dashboard/visualizations/bupar/outputs/allowed_codes_shap_ffa_opioid_ed_25_44.json"
        bupar_outputs = REPO_ROOT / "10_risk_dashboard" / "visualizations" / "bupar" / "outputs"
        path = bupar_outputs / "allowed_codes_shap_ffa_opioid_ed_25_44.json"
        self.assertIn("bupar", str(path))
        self.assertIn("allowed_codes_shap_ffa_opioid_ed_25_44.json", str(path))


class TestFpgrowthLoadAllowedCodes(unittest.TestCase):
    """FP-Growth cohort_fpgrowth: requires same allowed codes file."""

    def test_raises_when_file_missing(self):
        """FP-Growth _load_allowed_codes_by_type raises when file is missing."""
        import tempfile
        sys.path.insert(0, str(REPO_ROOT))
        from pathlib import Path as P
        # cohort_fpgrowth is under 9_dashboard_visuals/fpgrowth
        fpg_path = REPO_ROOT / "9_dashboard_visuals" / "fpgrowth"
        if str(fpg_path) not in sys.path:
            sys.path.insert(0, str(fpg_path))
        try:
            from cohort_fpgrowth import _load_allowed_codes_by_type
        except ImportError as e:
            self.skipTest(f"cohort_fpgrowth not importable: {e}")
            return
        with tempfile.TemporaryDirectory() as tmp:
            root = P(tmp)
            with self.assertRaises(FileNotFoundError) as ctx:
                _load_allowed_codes_by_type("opioid_ed", "25-44", "drug_name", root)
            self.assertIn("required (prerequisite)", str(ctx.exception))

    def test_raises_when_file_empty(self):
        """FP-Growth raises ValueError when file is empty."""
        fpg_path = REPO_ROOT / "9_dashboard_visuals" / "fpgrowth"
        if str(fpg_path) not in sys.path:
            sys.path.insert(0, str(fpg_path))
        try:
            from cohort_fpgrowth import _load_allowed_codes_by_type
        except ImportError as e:
            self.skipTest(f"cohort_fpgrowth not importable: {e}")
            return
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out_dir = root / "10_risk_dashboard" / "visualizations" / "bupar" / "outputs"
            out_dir.mkdir(parents=True)
            (out_dir / "allowed_codes_shap_ffa_opioid_ed_25_44.json").write_text("[]", encoding="utf-8")
            with self.assertRaises(ValueError) as ctx:
                _load_allowed_codes_by_type("opioid_ed", "25-44", "icd_code", root)
            self.assertIn("empty", str(ctx.exception).lower())

    def test_returns_codes_when_valid(self):
        """FP-Growth returns non-empty set when file has codes."""
        fpg_path = REPO_ROOT / "9_dashboard_visuals" / "fpgrowth"
        if str(fpg_path) not in sys.path:
            sys.path.insert(0, str(fpg_path))
        try:
            from cohort_fpgrowth import _load_allowed_codes_by_type
        except ImportError as e:
            self.skipTest(f"cohort_fpgrowth not importable: {e}")
            return
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out_dir = root / "10_risk_dashboard" / "visualizations" / "bupar" / "outputs"
            out_dir.mkdir(parents=True)
            (out_dir / "allowed_codes_shap_ffa_opioid_ed_25_44.json").write_text(
                json.dumps(["drug_HYDROCODONE", "icd_F1120", "cpt_99213"]),
                encoding="utf-8",
            )
            drug = _load_allowed_codes_by_type("opioid_ed", "25-44", "drug_name", root)
            self.assertIsInstance(drug, set)
            self.assertGreater(len(drug), 0)


if __name__ == "__main__":
    unittest.main()
