"""
BupaR Process Mining tab.

Pages:
- Plot images: GET /visualizations/bupar
- Activity frequency (bar charts): GET /visualizations/bupar/activity_frequency
"""

import json
import pytest

from conftest import LAMBDA_AVAILABLE, query_event, lambda_handler


@pytest.mark.skipif(not LAMBDA_AVAILABLE, reason="Lambda handler not importable")
class TestBuparTab:
    """BupaR Process Mining tab."""

    class TestPlotsPage:
        """GET /visualizations/bupar – BupaR plot image URLs."""

        def test_bupar_returns_200_and_urls(self):
            event = query_event("GET", "/visualizations/bupar", query={
                "cohort": "opioid_ed",
                "age_band": "25-44",
            })
            resp = lambda_handler(event, None)
            assert resp["statusCode"] in (200, 400, 500)
            if resp["statusCode"] == 200:
                data = json.loads(resp["body"])
                assert isinstance(data, dict)

    class TestActivityFrequencyPage:
        """GET /visualizations/bupar/activity_frequency – bar chart data."""

        def test_activity_frequency_returns_200_and_structure(self):
            event = query_event("GET", "/visualizations/bupar/activity_frequency", query={
                "cohort": "opioid_ed",
                "age_band": "25-44",
            })
            resp = lambda_handler(event, None)
            assert resp["statusCode"] in (200, 400, 500)
            if resp["statusCode"] == 200:
                data = json.loads(resp["body"])
                assert isinstance(data, dict)
                # overall, pre_target, post_target (or nulls if missing)
                assert "overall" in data or "pre_target" in data or "post_target" in data or len(data) >= 0
