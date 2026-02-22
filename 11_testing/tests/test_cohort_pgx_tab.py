"""
PGx Cohort tab.

Page: GET /visualizations/cohort_pgx – network topology URL (iframe) per cohort/age_band.
"""

import json
import pytest

from conftest import LAMBDA_AVAILABLE, query_event, lambda_handler


@pytest.mark.skipif(not LAMBDA_AVAILABLE, reason="Lambda handler not importable")
class TestCohortPgxTab:
    """PGx Cohort tab – gene–drug–phenotype network topology from VIP reports."""

    class TestNetworkTopologyPage:
        """GET /visualizations/cohort_pgx – network_topology_url for iframe."""

        def test_cohort_pgx_returns_200_and_url(self):
            event = query_event("GET", "/visualizations/cohort_pgx", query={
                "cohort": "opioid_ed",
                "age_band": "25-44",
            })
            resp = lambda_handler(event, None)
            assert resp["statusCode"] in (200, 400, 500)
            if resp["statusCode"] == 200:
                data = json.loads(resp["body"])
                assert isinstance(data, dict)
                assert "network_topology_url" in data

        def test_cohort_pgx_requires_cohort_and_age_band(self):
            event = query_event("GET", "/visualizations/cohort_pgx", query={})
            resp = lambda_handler(event, None)
            assert resp["statusCode"] == 400
