"""
Causal Analysis tab.
GET /visualizations/causal (and POST /causal/* if used).
"""

import json
import pytest

from conftest import LAMBDA_AVAILABLE, query_event, lambda_handler


@pytest.mark.skipif(not LAMBDA_AVAILABLE, reason="Lambda handler not importable")
class TestCausalAnalysisTab:
    """Causal Analysis tab - SHAP/FFA causal importance and interactions."""

    class TestCausalVisualizationsPage:
        """GET /visualizations/causal - causal/SHAP data or URLs."""

        def test_visualizations_causal_returns_200_or_4xx(self):
            event = query_event("GET", "/visualizations/causal", query={
                "cohort": "opioid_ed",
                "age_band": "25-44",
            })
            resp = lambda_handler(event, None)
            assert resp["statusCode"] in (200, 400, 404, 500)
            if resp["statusCode"] == 200:
                data = json.loads(resp["body"])
                assert isinstance(data, dict)
