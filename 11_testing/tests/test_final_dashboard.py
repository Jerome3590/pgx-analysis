"""
Final dashboard test suite – entry point and overview.

Tests are organized by dashboard tab and page under 11_testing/tests:

- test_artifacts.py         – Required metadata, CPIC, frontend, models (shared)
- test_api_cors.py          – CORS and unsupported route (global API)
- test_risk_assessment_tab.py – Risk Assessment (+ Drugs/ICD/CPT metadata)
- test_pgx_card_tab.py      – PGx Card
- test_documentation_tab.py  – Documentation (metrics)
- test_feature_importance_tab.py – Feature Importance
- test_causal_analysis_tab.py   – Causal Analysis
- test_bupar_tab.py         – BupaR Process Mining
- test_dtw_tab.py           – DTW Trajectories
- test_fpgrowth_tab.py      – FP-Growth Patterns
- test_cohort_pgx_tab.py    – PGx Cohort (network topology)
- test_live_api.py          – Live API (when BASE_URL set), by tab

Run from repo root:
  pytest 11_testing/tests/ -v
  pytest 11_testing/tests/test_risk_assessment_tab.py -v
  BASE_URL=https://.../prod pytest 11_testing/tests/test_live_api.py -v
"""

# This file does not define tests; it documents the layout.
# Pytest collects tests from all test_*.py in this directory.
