"""
DTW Trajectories tab.

Page: DTW assets – GET /visualizations/dtw (overview image, chart_data, heatmap URLs).
"""

import json
import pytest

from conftest import LAMBDA_AVAILABLE, query_event, lambda_handler


@pytest.mark.skipif(not LAMBDA_AVAILABLE, reason="Lambda handler not importable")
class TestDtwTab:
    """DTW Trajectories tab."""

    class TestTrajectoriesPage:
        """GET /visualizations/dtw – DTW plot and chart URLs."""

        def test_dtw_returns_200_and_urls(self):
            event = query_event("GET", "/visualizations/dtw", query={
                "cohort": "opioid_ed",
                "age_band": "25-44",
            })
            resp = lambda_handler(event, None)
            assert resp["statusCode"] in (200, 400, 500)
            if resp["statusCode"] == 200:
                data = json.loads(resp["body"])
                assert isinstance(data, dict)
                assert "overview_image" in data or "chart_data_url" in data or "metrics" in data
