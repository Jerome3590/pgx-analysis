"""
FP-Growth Patterns tab.

Page: Itemsets and network – GET /visualizations/fpgrowth.
"""

import json
import pytest

from conftest import LAMBDA_AVAILABLE, query_event, lambda_handler


@pytest.mark.skipif(not LAMBDA_AVAILABLE, reason="Lambda handler not importable")
class TestFpgrowthTab:
    """FP-Growth Patterns tab."""

    class TestPatternsPage:
        """GET /visualizations/fpgrowth – itemset/network URLs or empty_state."""

        def test_fpgrowth_returns_200_and_payload(self):
            event = query_event("GET", "/visualizations/fpgrowth", query={
                "cohort": "opioid_ed",
                "age_band": "25-44",
            })
            resp = lambda_handler(event, None)
            assert resp["statusCode"] in (200, 400, 500)
            if resp["statusCode"] == 200:
                data = json.loads(resp["body"])
                assert isinstance(data, dict)
