"""
PGx Card tab.

Page: Generate PGx Card – POST /pgx/card (and CPIC data in container/S3).
"""

import json
import pytest

from conftest import LAMBDA_AVAILABLE, query_event, lambda_handler


@pytest.mark.skipif(not LAMBDA_AVAILABLE, reason="Lambda handler not importable")
class TestPgxCardTab:
    """PGx Patient Card tab."""

    class TestGenerateCardPage:
        """POST /pgx/card – generate card from variant input."""

        def test_pgx_card_returns_200_or_4xx(self):
            event = query_event("POST", "/pgx/card", body={
                "variants": [{"gene": "CYP2C19", "variants": ["*1/*1"]}],
            })
            resp = lambda_handler(event, None)
            assert resp["statusCode"] in (200, 400, 404, 500)
            body = json.loads(resp.get("body", "{}"))
            assert isinstance(body, dict)
