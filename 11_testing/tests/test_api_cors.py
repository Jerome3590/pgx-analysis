"""
Global API behavior: CORS and unsupported routes.

Applies to all dashboard tabs (API Gateway + Lambda).
"""

import pytest

from conftest import LAMBDA_AVAILABLE, query_event, lambda_handler


@pytest.mark.skipif(not LAMBDA_AVAILABLE, reason="Lambda handler not importable")
class TestApiCorsAndRouting:
    """CORS and route handling."""

    def test_options_returns_200_and_cors_headers(self):
        event = {"httpMethod": "OPTIONS", "path": "/prod/metadata", "queryStringParameters": None, "body": None}
        resp = lambda_handler(event, None)
        assert resp["statusCode"] == 200
        assert "Access-Control-Allow-Origin" in resp["headers"]

    def test_unsupported_route_returns_404(self):
        event = query_event("GET", "/nonexistent")
        resp = lambda_handler(event, None)
        assert resp["statusCode"] == 404
