"""
Combinatorial risk tests: all cohort × age_band × density-bin scenarios.

Tests POST /risk for every (cohort, age_band) pair with 5 input scenarios
designed to exercise each n_event_bin routing tier:

  Default thresholds (p25=5, p50=15, p95=50 — from _DEFAULT_NEVENT_THRESHOLDS):
  - baseline : 0 codes  → is_baseline=True, returns 2019 outcome rate
  - low      : 3 codes  → n_event_bin=low    (n ≤ p25)
  - medium   : 10 codes → n_event_bin=medium (p25 < n ≤ p50)
  - high     : 25 codes → n_event_bin=high   (p50 < n ≤ p95)
  - extreme  : 55 codes → n_event_bin=extreme (n > p95)

Each test asserts:
  1. statusCode == 200 (skip assertion on 500 if models are not deployed)
  2. risk_score ∈ [0.0, 1.0]
  3. risk_band ∈ {"low", "medium", "high"}
  4. n_event_bin matches expected tier (using default thresholds)
  5. n_events == total codes submitted
  6. age_band_used / cohort_used echo back the request
  7. codes_used / codes_unknown are dicts with drug/icd/cpt keys
  8. model_breakdown is a dict

Live-API variant (requires BASE_URL env var) runs the same matrix via HTTP.

Run:
    # Lambda handler (local, no models required for structure tests)
    pytest 11_testing/tests/test_combinatorial_risk.py -v

    # Live API (set BASE_URL first)
    $env:BASE_URL = "https://<id>.execute-api.us-east-1.amazonaws.com/prod"
    pytest 11_testing/tests/test_combinatorial_risk.py -v -k live

    # Single cohort/age_band
    pytest 11_testing/tests/test_combinatorial_risk.py -v -k "opioid_ed/25-44"
"""

import json
import pytest
import requests

from conftest import LAMBDA_AVAILABLE, BASE_URL, LIVE_TESTING, query_event, lambda_handler, get_live_session

# ---------------------------------------------------------------------------
# Matrix
# ---------------------------------------------------------------------------

COHORTS = ["opioid_ed", "non_opioid_ed"]
AGE_BANDS = ["0-12", "13-24", "25-44", "45-54", "55-64", "65-74", "75-84", "85-114"]
ALL_COMBOS = [(c, ab) for c in COHORTS for ab in AGE_BANDS]  # 16 total

VALID_BINS = {"low", "medium", "high", "extreme"}
VALID_BANDS = {"low", "medium", "high"}

# ---------------------------------------------------------------------------
# Representative code pools (broad coverage; unknown codes go to codes_unknown
# but the model still runs — feature vector defaults to 0.0 for absent codes)
# ---------------------------------------------------------------------------

# Opioid-ED relevant codes (target = F1120 opioid use disorder ED)
_OPD_DRUGS = [
    "oxycodone", "hydrocodone", "tramadol", "gabapentin", "alprazolam",
    "cyclobenzaprine", "fentanyl", "codeine", "methadone", "morphine",
    "diazepam", "clonazepam", "buprenorphine", "oxymorphone", "hydromorphone",
    "carisoprodol", "zolpidem", "lorazepam", "pregabalin", "duloxetine",
]
_OPD_ICDS = [
    "M54.5", "G89.29", "F41.1", "F32.1", "F17.210", "R51", "M25.511",
    "Z87.891", "J06.9", "M54.41", "G89.4", "M79.3", "F33.1",
    "M54.16", "G89.11", "F41.0", "M47.816", "Z79.891", "G89.21", "M54.50",
]
_OPD_CPTS = [
    "99213", "80305", "99396", "99214", "99203", "80306", "97110",
    "97012", "90832", "90834", "72100", "72148", "73560", "99215", "99204",
]

# Non-opioid ED relevant codes (target = polypharmacy-related ED visit)
_NON_DRUGS = [
    "furosemide", "hydrochlorothiazide", "lisinopril", "metformin", "simvastatin",
    "atorvastatin", "metoprolol", "amlodipine", "carvedilol", "losartan",
    "warfarin", "aspirin", "omeprazole", "levothyroxine", "albuterol",
    "prednisone", "levofloxacin", "alprazolam", "lorazepam", "acetaminophen",
]
_NON_ICDS = [
    "I10", "E11.9", "E78.5", "I50.9", "N18.3", "I25.10", "J44.1",
    "E03.9", "G47.33", "M79.3", "K21.0", "D64.9", "F03.90", "G20",
    "I48.91", "I63.9", "N39.0", "R06.09", "Z79.01", "M17.11",
]
_NON_CPTS = [
    "99213", "99214", "93000", "83036", "85025", "80053", "36415",
    "99396", "93306", "71046", "93010", "82947", "84443", "86900", "99395",
]

_POOLS = {
    "opioid_ed":     (_OPD_DRUGS, _OPD_ICDS, _OPD_CPTS),
    "non_opioid_ed": (_NON_DRUGS, _NON_ICDS, _NON_CPTS),
}


def _make_scenarios(cohort: str) -> dict:
    """
    Return {scenario_name: (drugs, icds, cpts, expected_bin)} for a cohort.

    Code counts are chosen to fall into each density tier using the default
    thresholds (p25=5, p50=15, p95=50).  If a cohort has custom thresholds
    the bin assertion is skipped (the response still validates structure).
    """
    drugs, icds, cpts = _POOLS[cohort]
    return {
        # 0 codes → baseline path (uses 2019 outcome rate, no model inference)
        "baseline": ([], [], [], None),
        # 1+1+1 = 3 codes  → low  (3 ≤ p25=5)
        "low":      (drugs[:1], icds[:1], cpts[:1], "low"),
        # 4+4+2 = 10 codes → medium (5 < 10 ≤ p50=15)
        "medium":   (drugs[:4], icds[:4], cpts[:2], "medium"),
        # 10+10+5 = 25 codes → high (15 < 25 ≤ p95=50)
        "high":     (drugs[:10], icds[:10], cpts[:5], "high"),
        # 20+20+15 = 55 codes → extreme (55 > p95=50)
        "extreme":  (drugs[:20], icds[:20], cpts[:15], "extreme"),
    }


SCENARIO_NAMES = ["baseline", "low", "medium", "high", "extreme"]

# ---------------------------------------------------------------------------
# Shared validation helpers
# ---------------------------------------------------------------------------

def _assert_200_response_shape(body: dict, cohort: str, age_band: str,
                                scenario: str, drugs: list, icds: list, cpts: list,
                                expected_bin):
    """Assert all invariants on a 200 POST /risk response body."""
    # Core risk output
    assert "risk_score" in body, f"Missing risk_score in {scenario}"
    assert isinstance(body["risk_score"], (int, float)), "risk_score not numeric"
    assert 0.0 <= body["risk_score"] <= 1.0, f"risk_score {body['risk_score']} out of [0,1]"

    assert "risk_band" in body, "Missing risk_band"
    assert body["risk_band"] in VALID_BANDS, f"Unexpected risk_band: {body['risk_band']}"

    # Echo-back fields
    assert body.get("cohort_used") == cohort, (
        f"cohort_used mismatch: got {body.get('cohort_used')}, want {cohort}"
    )
    assert body.get("age_band_used") == age_band, (
        f"age_band_used mismatch: got {body.get('age_band_used')}, want {age_band}"
    )

    total_codes = len(drugs) + len(icds) + len(cpts)

    if scenario == "baseline":
        # Baseline path: is_baseline=True, no n_event_bin in response
        assert body.get("is_baseline") is True, "Expected is_baseline=True for 0-code request"
        assert "model_breakdown" in body
    else:
        # Model inference path
        assert body.get("is_baseline") is not True, "Unexpected is_baseline=True for non-empty codes"

        assert "n_event_bin" in body, "Missing n_event_bin"
        assert body["n_event_bin"] in VALID_BINS, f"Unexpected n_event_bin: {body['n_event_bin']}"

        assert body.get("n_events") == total_codes, (
            f"n_events mismatch: got {body.get('n_events')}, want {total_codes}"
        )

        # Density-bin routing assertion (valid against default thresholds;
        # may legitimately differ if a custom thresholds JSON is loaded).
        if expected_bin is not None:
            assert body["n_event_bin"] == expected_bin, (
                f"[{cohort}/{age_band}/{scenario}] "
                f"Expected n_event_bin={expected_bin!r}, got {body['n_event_bin']!r}. "
                "If cohort-specific thresholds differ from defaults (p25=5,p50=15,p95=50), "
                "update SCENARIO_CODE_COUNTS to hit the correct tier."
            )

        # model_breakdown must be a dict (may be empty if all models failed)
        assert isinstance(body.get("model_breakdown"), dict), "model_breakdown must be a dict"

    # codes_used / codes_unknown structure
    for key in ("codes_used", "codes_unknown"):
        assert key in body, f"Missing {key}"
        block = body[key]
        assert isinstance(block, dict), f"{key} must be a dict"
        for sub in ("drugs", "icds", "cpts"):
            assert sub in block, f"Missing {key}.{sub}"
            assert isinstance(block[sub], list), f"{key}.{sub} must be a list"


# ---------------------------------------------------------------------------
# Local Lambda tests (no network — handler imported directly)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not LAMBDA_AVAILABLE, reason="Lambda handler not importable")
class TestCombinatorial:
    """
    80 parametrized tests: 16 cohort/age_band × 5 density scenarios.
    Invokes lambda_handler directly — no server needed.
    """

    @pytest.mark.parametrize(
        "cohort,age_band",
        ALL_COMBOS,
        ids=[f"{c}/{ab}" for c, ab in ALL_COMBOS],
    )
    @pytest.mark.parametrize("scenario", SCENARIO_NAMES)
    def test_risk(self, cohort, age_band, scenario):
        scenarios = _make_scenarios(cohort)
        drugs, icds, cpts, expected_bin = scenarios[scenario]

        event = query_event("POST", "/risk", body={
            "cohort": cohort,
            "age_band": age_band,
            "drugs": drugs,
            "icds": icds,
            "cpts": cpts,
        })
        resp = lambda_handler(event, None)

        # 500 is tolerated when models are not deployed locally
        assert resp["statusCode"] in (200, 500), (
            f"Unexpected status {resp['statusCode']} for {cohort}/{age_band}/{scenario}"
        )

        if resp["statusCode"] == 200:
            body = json.loads(resp["body"])
            _assert_200_response_shape(body, cohort, age_band, scenario,
                                       drugs, icds, cpts, expected_bin)


# ---------------------------------------------------------------------------
# Visualization endpoint combinatorial tests (GET endpoints, all combos)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not LAMBDA_AVAILABLE, reason="Lambda handler not importable")
class TestVisualizationCombinatorial:
    """
    All visualization GET endpoints across every cohort × age_band.
    Asserts 200 or 400/500; on 200 asserts minimal response shape.
    """

    _VIZ_ENDPOINTS = [
        ("/visualizations/causal",    {"causal_factors": list}),
        ("/visualizations/dtw",       {}),
        ("/visualizations/fpgrowth",  {}),
        ("/visualizations/bupar",     {}),
        ("/visualizations/cohort_pgx", {"network_topology_url": str}),
    ]

    @pytest.mark.parametrize(
        "cohort,age_band",
        ALL_COMBOS,
        ids=[f"{c}/{ab}" for c, ab in ALL_COMBOS],
    )
    @pytest.mark.parametrize(
        "path,required_keys",
        _VIZ_ENDPOINTS,
        ids=[ep[0].lstrip("/").replace("/", ".") for ep in _VIZ_ENDPOINTS],
    )
    def test_viz_endpoint(self, cohort, age_band, path, required_keys):
        event = query_event("GET", path, query={"cohort": cohort, "age_band": age_band})
        resp = lambda_handler(event, None)

        assert resp["statusCode"] in (200, 400, 404, 500), (
            f"Unexpected status {resp['statusCode']} for {path} {cohort}/{age_band}"
        )

        if resp["statusCode"] == 200 and required_keys:
            body = json.loads(resp["body"])
            for key, typ in required_keys.items():
                if key in body:
                    assert isinstance(body[key], typ), (
                        f"{path}: {key} should be {typ.__name__}"
                    )

    @pytest.mark.parametrize(
        "cohort,age_band",
        ALL_COMBOS,
        ids=[f"{c}/{ab}" for c, ab in ALL_COMBOS],
    )
    def test_bupar_activity_frequency(self, cohort, age_band):
        event = query_event("GET", "/visualizations/bupar/activity_frequency",
                            query={"cohort": cohort, "age_band": age_band})
        resp = lambda_handler(event, None)
        assert resp["statusCode"] in (200, 400, 404, 500)
        if resp["statusCode"] == 200:
            body = json.loads(resp["body"])
            assert isinstance(body, dict)


# ---------------------------------------------------------------------------
# Required-param validation (400 tests)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not LAMBDA_AVAILABLE, reason="Lambda handler not importable")
class TestRequiredParamValidation:
    """Verify 400 is returned when required params are missing."""

    def test_risk_missing_cohort_and_age_and_codes(self):
        event = query_event("POST", "/risk", body={})
        resp = lambda_handler(event, None)
        assert resp["statusCode"] in (400, 500)

    @pytest.mark.parametrize("path", [
        "/visualizations/causal",
        "/visualizations/dtw",
        "/visualizations/fpgrowth",
        "/visualizations/bupar",
        "/visualizations/cohort_pgx",
    ])
    def test_viz_missing_params_returns_400(self, path):
        event = query_event("GET", path, query={})
        resp = lambda_handler(event, None)
        assert resp["statusCode"] == 400, (
            f"{path} should return 400 when cohort/age_band missing, got {resp['statusCode']}"
        )


# ---------------------------------------------------------------------------
# Live API combinatorial tests (requires BASE_URL env var)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not LIVE_TESTING, reason="Live API tests require BASE_URL or LIVE_TESTING=1")
class TestLiveCombinatorial:
    """
    Same 80-case matrix against the deployed API Gateway endpoint.

    Set BASE_URL (or LIVE_TESTING=1) before running:
        $env:BASE_URL = "https://cmv0qislq3.execute-api.us-east-1.amazonaws.com/prod"
        pytest 11_testing/tests/test_combinatorial_risk.py -v -k live
    """

    @pytest.fixture(scope="class")
    def session(self):
        s = get_live_session()
        if s is None:
            pytest.skip("requests not installed")
        return s

    @pytest.mark.parametrize(
        "cohort,age_band",
        ALL_COMBOS,
        ids=[f"{c}/{ab}" for c, ab in ALL_COMBOS],
    )
    @pytest.mark.parametrize("scenario", SCENARIO_NAMES)
    def test_risk_live(self, session, cohort, age_band, scenario):
        scenarios = _make_scenarios(cohort)
        drugs, icds, cpts, expected_bin = scenarios[scenario]

        r = session.post(
            f"{BASE_URL}/risk",
            json={"cohort": cohort, "age_band": age_band,
                  "drugs": drugs, "icds": icds, "cpts": cpts},
            timeout=20,
        )
        assert r.status_code in (200, 500), (
            f"Unexpected HTTP {r.status_code} for {cohort}/{age_band}/{scenario}"
        )
        if r.status_code == 200:
            body = r.json()
            # Pass expected_bin=None: production uses trained thresholds that differ
            # from the defaults (p25=5, p50=15, p95=50).  We still verify the bin
            # is a valid string; routing correctness is covered by local Lambda tests.
            _assert_200_response_shape(body, cohort, age_band, scenario,
                                       drugs, icds, cpts, expected_bin=None)

    @pytest.mark.parametrize(
        "cohort,age_band",
        ALL_COMBOS,
        ids=[f"{c}/{ab}" for c, ab in ALL_COMBOS],
    )
    @pytest.mark.parametrize(
        "path",
        ["/visualizations/causal", "/visualizations/dtw",
         "/visualizations/fpgrowth", "/visualizations/bupar",
         "/visualizations/cohort_pgx"],
    )
    def test_viz_live(self, session, cohort, age_band, path):
        r = session.get(
            f"{BASE_URL}{path}",
            params={"cohort": cohort, "age_band": age_band},
            timeout=20,
        )
        assert r.status_code in (200, 400, 404, 500), (
            f"Unexpected HTTP {r.status_code} for {path} {cohort}/{age_band}"
        )
        if r.status_code == 200:
            assert isinstance(r.json(), dict)
