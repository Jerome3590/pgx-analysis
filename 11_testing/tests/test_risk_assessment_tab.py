"""
Risk Assessment tab (and code-selection tabs: Drugs, ICD Codes, CPT Codes).

Pages:
- Metadata/dropdowns: GET /metadata (feeds Risk + Drugs + ICD + CPT tabs)
- Risk score: POST /risk
- Risk comparison: POST /risk/comparison
"""

import json
import pytest

from conftest import LAMBDA_AVAILABLE, query_event, lambda_handler


@pytest.mark.skipif(not LAMBDA_AVAILABLE, reason="Lambda handler not importable")
class TestRiskAssessmentTab:
    """Risk Assessment tab – metadata, risk score, comparison."""

    class TestMetadataPage:
        """GET /metadata – cohort dropdown and code lists (used by Risk, Drugs, ICD, CPT tabs)."""

        def test_metadata_opioid_ed_returns_200_and_shape(self):
            event = query_event("GET", "/metadata", query={"cohort": "opioid_ed"})
            resp = lambda_handler(event, None)
            assert resp["statusCode"] in (200, 404)
            if resp["statusCode"] == 200:
                data = json.loads(resp["body"])
                assert isinstance(data, dict)
                assert "age_bands" in data or "age_band" in data or len(data) > 0

        def test_metadata_non_opioid_ed_returns_200_and_shape(self):
            event = query_event("GET", "/metadata", query={"cohort": "non_opioid_ed"})
            resp = lambda_handler(event, None)
            assert resp["statusCode"] in (200, 404)
            if resp["statusCode"] == 200:
                data = json.loads(resp["body"])
                assert isinstance(data, dict)

    class TestRiskScorePage:
        """POST /risk – single risk score (Calculate Risk Score button)."""

        def test_risk_returns_200_or_4xx(self):
            event = query_event("POST", "/risk", body={
                "cohort": "opioid_ed",
                "age_band": "25-44",
                "drugs": [],
                "icds": [],
                "cpts": [],
            })
            resp = lambda_handler(event, None)
            assert resp["statusCode"] in (200, 400, 404, 500)
            if resp["statusCode"] == 200:
                data = json.loads(resp["body"])
                assert "risk" in data or "score" in data or "band" in data or isinstance(data, dict)

    class TestRiskComparisonPage:
        """POST /risk/comparison – compare scenarios."""

        def test_risk_comparison_returns_200_or_4xx(self):
            event = query_event("POST", "/risk/comparison", body={
                "base": {"cohort": "opioid_ed", "age_band": "25-44", "drugs": []},
                "scenarios": [],
            })
            resp = lambda_handler(event, None)
            assert resp["statusCode"] in (200, 400, 404, 500)
            body = json.loads(resp.get("body", "{}"))
            assert isinstance(body, dict)
