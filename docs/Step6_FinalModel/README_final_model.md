# Step 6 Final Model

This page is a documentation mirror for Step 6. The authoritative workflow is maintained in:

- `6_final_model/README.md`
- `6_final_model/README_final_model_implementation.md`
- `3_model_train_shap_ffa.ipynb`

## Current production workflow

The production entry point is `3_model_train_shap_ffa.ipynb`.

- **Step 4 model data**
  - Runs for both `opioid_ed` and `non_opioid_ed`.
  - Skips existing `model_events.parquet` when `FORCE_STEP4_ALL = False` and `FORCE_STEP4_NON_OPIOID = False`.

- **Step 5 PGx analysis**
  - Runs for both cohorts.
  - Normally uses `FORCE_STEP5 = False` so completed outputs/checkpoints can skip.

- **Step 6 final model**
  - Runs `6_final_model/run_final_model.py` for both cohorts and all age bands.
  - Default train mode is `--train-mode per_bin`.
  - With `CLEAN_STEP6_DOWNSTREAM_ARTIFACTS = False`, `FORCE_STEP6 = False`, and `FORCE_STEP6_REBUILT_ONLY = False`, complete local/S3 artifacts are not retrained.
  - Missing or incomplete artifacts regenerate.

- **Downstream attribution**
  - SHAP, FFA, and Combine use `DOWNSTREAM_ANALYSIS_COHORTS`.
  - Current final workflow runs downstream attribution for both `opioid_ed` and `non_opioid_ed` after Step 6 artifacts are available.

## Model behavior

- Final features are built from Step 4 model data, Step 3 feature importance, PGx counts, demographics, and allowed schema features.
- FPGrowth, BupaR, and DTW outputs are dashboard/protocol-filtering inputs, not final model feature columns.
- Training uses temporal separation: 2016-2018 for training/MC-CV, 2019 for holdout evaluation.
- Candidate models include XGBoost, XGBoost RF, CatBoost, and Ensemble.
- Selection is by PR-AUC mean first, then recall mean.
- CatBoost uses symmetric trees and has guards for constant/invalid categorical feature matrices.
- Per-density-bin training falls back to pooled full-cohort artifacts for sparse bins and writes `INFERENCE_SOURCE.txt`.

## Output layout

Primary local output:

```text
6_final_model/outputs/{cohort}/{age_band_fname}/
```

Per-bin output:

```text
6_final_model/outputs/{cohort}/{age_band_fname}/bin_models/{low|medium|high|extreme}/
```

Primary S3 output:

```text
s3://pgxdatalake/gold/final_model/{cohort}/{age_band}/
```

## Notes on older docs

Older references to `7_final_model`, `final_model.ipynb` as the production entry point, or FP-Growth/BupaR/DTW-derived model columns are legacy context and should not be used as the current operating workflow.
