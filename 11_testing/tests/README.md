# Tests (all in one place)

All project tests live under **11_testing/tests/**.

## Dashboard (final) tests

- **conftest.py** – Shared paths, Lambda handler, `query_event`, `get_live_session`.
- **test_artifacts.py** – Metadata, CPIC, frontend, models.
- **test_api_cors.py** – CORS and 404 for unsupported routes.
- **test_risk_assessment_tab.py** – Risk Assessment (+ Drugs/ICD/CPT metadata).
- **test_pgx_card_tab.py** – PGx Card.
- **test_documentation_tab.py** – Documentation (metrics).
- **test_feature_importance_tab.py** – Feature Importance.
- **test_causal_analysis_tab.py** – Causal Analysis.
- **test_bupar_tab.py** – BupaR Process Mining.
- **test_dtw_tab.py** – DTW Trajectories.
- **test_fpgrowth_tab.py** – FP-Growth Patterns.
- **test_cohort_pgx_tab.py** – PGx Cohort (network topology).
- **test_live_api.py** – Live API (when `BASE_URL` set), by tab.

## Dashboard visuals (step 9) tests

- **dashboard_visuals/** – Prerequisite and output structure tests for BupaR, DTW, FP-Growth.

## Running

From repo root:

```bash
pytest 11_testing/tests/ -v
pytest 11_testing/tests/test_risk_assessment_tab.py -v
BASE_URL=https://.../prod pytest 11_testing/tests/test_live_api.py -v
```

Or use **11_testing/run_tests.ps1** / **run_tests.bat** (Windows).
