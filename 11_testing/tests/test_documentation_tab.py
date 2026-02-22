"""
Documentation tab.

Page: Model performance metrics – GET /metrics (or same-origin static file).
"""

import json
import pytest

from conftest import LAMBDA_AVAILABLE, query_event, lambda_handler


@pytest.mark.skipif(not LAMBDA_AVAILABLE, reason="Lambda handler not importable")
class TestDocumentationTab:
    """Documentation tab – metrics and help content."""

    class TestMetricsPage:
        """GET /metrics – model performance metrics table."""

        def test_metrics_returns_200_or_404(self):
            event = query_event("GET", "/metrics")
            resp = lambda_handler(event, None)
            assert resp["statusCode"] in (200, 404)
            if resp["statusCode"] == 200:
                data = json.loads(resp["body"])
                assert isinstance(data, dict)
