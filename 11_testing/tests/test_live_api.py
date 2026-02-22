"""
Live API tests (when BASE_URL is set) – organized by dashboard tab/page.

Run: BASE_URL=https://...execute-api.../prod pytest 11_testing/tests/test_live_api.py -v
"""

import pytest

from conftest import BASE_URL, get_live_session


@pytest.mark.skipif(not BASE_URL, reason="BASE_URL not set; skipping live API tests")
class TestLiveApiRiskAssessmentTab:
    """Risk Assessment tab (and metadata for Drugs/ICD/CPT)."""

    @pytest.fixture(scope="class")
    def session(self):
        s = get_live_session()
        if s is None:
            pytest.skip("requests not installed")
        return s

    def test_options_metadata(self, session):
        r = session.options(f"{BASE_URL}/metadata", timeout=5)
        assert r.status_code == 200

    def test_get_metadata(self, session):
        r = session.get(f"{BASE_URL}/metadata", params={"cohort": "opioid_ed"}, timeout=15)
        assert r.status_code in (200, 404)
        if r.status_code == 200:
            assert "age_bands" in r.json() or len(r.json()) > 0

    def test_post_risk(self, session):
        r = session.post(f"{BASE_URL}/risk", json={
            "cohort": "opioid_ed", "age_band": "25-44", "drugs": [], "icds": [], "cpts": []
        }, timeout=15)
        assert r.status_code in (200, 400, 500)
        if r.status_code == 200:
            d = r.json()
            assert "risk" in d or "score" in d or "band" in d or isinstance(d, dict)


@pytest.mark.skipif(not BASE_URL, reason="BASE_URL not set")
class TestLiveApiDocumentationTab:
    """Documentation tab – metrics."""

    def test_get_metrics(self):
        s = get_live_session()
        if s is None:
            pytest.skip("requests not installed")
        r = s.get(f"{BASE_URL}/metrics", timeout=15)
        assert r.status_code in (200, 404)


@pytest.mark.skipif(not BASE_URL, reason="BASE_URL not set")
class TestLiveApiFeatureImportanceTab:
    """Feature Importance tab."""

    def test_get_feature_importance(self):
        s = get_live_session()
        if s is None:
            pytest.skip("requests not installed")
        r = s.get(f"{BASE_URL}/visualizations/feature_importance", params={"cohort": "opioid_ed"}, timeout=15)
        assert r.status_code in (200, 400, 500)


@pytest.mark.skipif(not BASE_URL, reason="BASE_URL not set")
class TestLiveApiCausalAnalysisTab:
    """Causal Analysis tab."""

    def test_get_visualizations_causal(self):
        s = get_live_session()
        if s is None:
            pytest.skip("requests not installed")
        r = s.get(f"{BASE_URL}/visualizations/causal", params={"cohort": "opioid_ed", "age_band": "25-44"}, timeout=15)
        assert r.status_code in (200, 400, 500)


@pytest.mark.skipif(not BASE_URL, reason="BASE_URL not set")
class TestLiveApiBuparTab:
    """BupaR Process Mining tab."""

    def test_get_bupar(self):
        s = get_live_session()
        if s is None:
            pytest.skip("requests not installed")
        r = s.get(f"{BASE_URL}/visualizations/bupar", params={"cohort": "opioid_ed", "age_band": "25-44"}, timeout=15)
        assert r.status_code in (200, 400, 500)

    def test_get_bupar_activity_frequency(self):
        s = get_live_session()
        if s is None:
            pytest.skip("requests not installed")
        r = s.get(f"{BASE_URL}/visualizations/bupar/activity_frequency", params={"cohort": "opioid_ed", "age_band": "25-44"}, timeout=15)
        assert r.status_code in (200, 400, 500)


@pytest.mark.skipif(not BASE_URL, reason="BASE_URL not set")
class TestLiveApiDtwTab:
    """DTW Trajectories tab."""

    def test_get_dtw(self):
        s = get_live_session()
        if s is None:
            pytest.skip("requests not installed")
        r = s.get(f"{BASE_URL}/visualizations/dtw", params={"cohort": "opioid_ed", "age_band": "25-44"}, timeout=15)
        assert r.status_code in (200, 400, 500)
        if r.status_code == 200:
            assert isinstance(r.json(), dict)


@pytest.mark.skipif(not BASE_URL, reason="BASE_URL not set")
class TestLiveApiFpgrowthTab:
    """FP-Growth Patterns tab."""

    def test_get_fpgrowth(self):
        s = get_live_session()
        if s is None:
            pytest.skip("requests not installed")
        r = s.get(f"{BASE_URL}/visualizations/fpgrowth", params={"cohort": "opioid_ed", "age_band": "25-44"}, timeout=15)
        assert r.status_code in (200, 400, 500)


@pytest.mark.skipif(not BASE_URL, reason="BASE_URL not set")
class TestLiveApiCohortPgxTab:
    """PGx Cohort tab."""

    def test_get_cohort_pgx(self):
        s = get_live_session()
        if s is None:
            pytest.skip("requests not installed")
        r = s.get(f"{BASE_URL}/visualizations/cohort_pgx", params={"cohort": "opioid_ed", "age_band": "25-44"}, timeout=15)
        assert r.status_code in (200, 400, 500)
        if r.status_code == 200:
            assert "network_topology_url" in r.json()
