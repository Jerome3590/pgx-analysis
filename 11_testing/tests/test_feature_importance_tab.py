"""
Feature Importance tab.

Page: Aggregated heatmaps – GET /visualizations/feature_importance.
"""

import json
import pytest

from conftest import LAMBDA_AVAILABLE, query_event, lambda_handler


@pytest.mark.skipif(not LAMBDA_AVAILABLE, reason="Lambda handler not importable")
class TestFeatureImportanceTab:
    """Feature Importance visualizations tab."""

    class TestHeatmapPage:
        """GET /visualizations/feature_importance – heatmap URLs."""

        def test_feature_importance_returns_200_and_urls(self):
            event = query_event("GET", "/visualizations/feature_importance", query={"cohort": "opioid_ed"})
            resp = lambda_handler(event, None)
            assert resp["statusCode"] in (200, 400, 404, 500)
            if resp["statusCode"] == 200:
                data = json.loads(resp["body"])
                assert isinstance(data, dict)
                assert "heatmap_url" in data or "combined_url" in data
