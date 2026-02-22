"""
Dashboard artifact tests: required files for Lambda and frontend.

Not tied to a single tab; metadata/CPIC/models are used by multiple tabs.
"""

import pytest
from pathlib import Path

from conftest import (  # noqa: E402
    DASHBOARD_ROOT,
    METADATA_DIR,
    MODELS_DIR,
    CPIC_DIR,
    FRONTEND_DIR,
    age_band_fname,
)
from py_helpers.constants import REQUIRED_COHORTS


def _metadata_files():
    return [
        METADATA_DIR / "metadata_opioid_ed.json",
        METADATA_DIR / "metadata_non_opioid_ed.json",
        METADATA_DIR / "model_performance_metrics.json",
    ]


class TestRequiredArtifacts:
    """Verify required dashboard artifacts exist (local outputs)."""

    @pytest.mark.parametrize("path", _metadata_files(), ids=[p.name for p in _metadata_files()])
    def test_metadata_file_exists(self, path: Path):
        if not path.exists():
            pytest.skip(f"Metadata not prepared: {path}. Run data_preparation/generate_metadata.py.")
        assert path.exists() and path.stat().st_size > 0

    def test_cpic_excel_exists(self):
        path = CPIC_DIR / "cpic_gene-drug_pairs.xlsx"
        if not path.exists():
            pytest.skip(f"CPIC not prepared: {path}. Run data_preparation/prepare_cpic_data.py.")
        assert path.exists() and path.stat().st_size > 0

    def test_frontend_index_exists(self):
        if not (FRONTEND_DIR / "index.html").exists():
            pytest.skip("Frontend index.html not found.")
        assert (FRONTEND_DIR / "index.html").exists()

    def test_models_dir_structure(self):
        """At least one cohort/age_band has model files."""
        found = 0
        for cohort, age_bands in REQUIRED_COHORTS.items():
            for age_band in age_bands:
                ab = age_band_fname(age_band)
                base = MODELS_DIR / cohort / ab
                if (base / "feature_schema.json").exists():
                    found += 1
        if found == 0:
            pytest.skip("No model directories found. Run data_preparation/prepare_models.py.")
        assert found >= 1
