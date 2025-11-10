# CatBoost Analysis - PGx

This module hosts CatBoost training and analysis pipelines for APCD cohorts.

## Goals

- Build cohort-level CatBoost models for target outcomes (target_code).
- Standardize feature extraction across pharmacy (drug_name) and medical (ICD/CPT) domains.
- Produce model explanations to guide feature reduction and clinical review.

## TODOs

- [ ] Feature importance exploration: identify which ICD diagnostic codes, CPT codes, and drug_name most strongly predict our target_code(s) of interest at the cohort level. Use model-based importance and SHAP summaries to filter to a manageable set for downstream visuals.

## Data Inputs

- Gold cohort partitions: `s3://pgxdatalake/gold/cohorts_clean/...`
- Supporting frequency artifacts (optional):
  - Target code frequencies: `s3://pgxdatalake/gold/target_code/`
  - Drug frequency outputs (if used): local CSV or S3

## Scripts

- `run_ade_targets.py`: Orchestrates ADE/ED CatBoost runs across age bands/years.
- `run_catboost_*.py`: Entry points for specific target setups (opioid ED, ADE ED, etc.).

## Notebook Integration

- See `docs/notebook_calls.md` for calls and patterns.
- Reusable plotting utilities: `1_apcd_input_data/7_target_visuals.py`.

## Notes

- Ensure gold cohort partitions are available before training.
- Prefer fewer, well-regularized features; enable model explanations to guide feature reduction.
- Post-model, revisit ICD/CPT/Drug heatmaps with top features only.


