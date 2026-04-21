# py_helpers — Shared Python Utilities

Shared utilities used across all pipeline steps. Import from project root after adding it to `sys.path`:

```python
from py_helpers.pipeline_logger import setup_pipeline_logger
from py_helpers.checkpoint_utils import upload_file_to_s3
```

---

## Active utilities (production pipeline)

### Logging & monitoring

| File | Key exports | Used by |
|------|-------------|---------|
| `pipeline_logger.py` | `PipelineLogger`, `setup_pipeline_logger` | Steps 9 (BupaR, DTW, FP-Growth, Cohort PGx) |
| `fe_monitor.py` | `mirror_log_to_s3` | Steps 4, 5, 6, 7, 8 — S3 log mirroring after completion |
| `logging_utils.py` | Extended logging helpers | Supplementary logging patterns |

**Log pattern:** All steps write to `logs/{step_name}/` at repo root; `mirror_log_to_s3` uploads to `s3://pgx-repository/{step_name}_log/{cohort}/{age_band}/`. See `docs/CrossStep_Development/README_logging.md`.

### Pipeline control

| File | Key exports | Used by |
|------|-------------|---------|
| `checkpoint_utils.py` | `upload_file_to_s3`, `save_step_checkpoint`, `check_step_checkpoint` | All steps — S3 idempotency checkpoints |
| `workflow_sync_checkpoint.py` | `sync_from_s3`, `sync_to_s3` | Notebook sync cells |
| `pipeline_utils.py` | Pipeline orchestration helpers | Notebook wrappers |
| `s3_upload_tracker.py` | Batch S3 upload tracking with retry | Step 9 visual artifact uploads |

### Data access

| File | Key exports | Used by |
|------|-------------|---------|
| `s3_utils.py` | `download_from_s3`, `upload_to_s3`, `list_s3_files`, `sync_s3_to_local` | All steps |
| `duckdb_utils.py` | `get_duckdb_conn`, DuckDB S3 config helpers | Steps 2, 3a, 4, 6, 9 |
| `data_utils.py` | Data loading, cleaning, schema helpers | Steps 1a, 2, 4 |
| `constants.py` | `AGE_BANDS`, `COHORTS`, `age_band_to_fname`, S3 bucket paths | All steps |
| `model_data_paths.py` | Canonical path resolver for model inputs/outputs | Steps 4–8 |
| `file_resolver.py` | Resolve file paths across S3/local with fallback | All steps |

### Feature importance

| File | Key exports | Used by |
|------|-------------|---------|
| `feature_importance_utils.py` | MC-CV runner, aggregation, feature ranking | Step 3a |
| `feature_importance_model_utils.py` | Model wrappers for MC-CV | Step 3a |
| `feature_importance_heatmap.py` | Aggregated feature importance heatmap (features × age bands) | Step 3a + dashboard |
| `feature_importance_eda_utils.py` | BupaR post-target leakage analysis | Step 3b |
| `feature_importance_filters.py` | Feature filtering rules | Step 3b, 3c |
| `mc_cv_utils.py` | Monte Carlo cross-validation infrastructure | Step 3a |

### Model & inference

| File | Key exports | Used by |
|------|-------------|---------|
| `model_utils.py` | Model loading, saving, format conversion helpers | Steps 6, 7, 8 |
| `categorical_encoding.py` | Binary/count encoding for item_* features | Steps 4, 6 |
| `feature_utils.py` | Feature vector assembly, alignment | Steps 4, 6 |
| `event_density_utils.py` | `DENSITY_BINS`, `assign_n_event_bin`, threshold I/O | Steps 6–9 + Lambda |

### Dashboard visuals

| File | Key exports | Used by |
|------|-------------|---------|
| `shap_ffa_fpgrowth_utils.py` | `get_shap_ffa_allowed_codes_combined`, `write_shap_ffa_allowed_codes_for_bupar`, `get_final_feature_importance_codes` | Step 9 (BupaR, DTW, FP-Growth gating) |
| `visualization_utils.py` | Plot helpers, color palettes, layout utilities | Step 9 visuals |
| `create_fpgrowth_visualizations.py` | FP-Growth network and itemset charts | Step 9 FP-Growth |
| `fpgrowth_utils.py` | FP-Growth mining wrappers | Step 9 FP-Growth |
| `pgx_dashboard_visuals.py` | Orchestrator: runs BupaR → DTW → FP-Growth end-to-end | Notebook 4 alternative |

### Cohort & drug utilities

| File | Key exports | Used by |
|------|-------------|---------|
| `cohort_utils.py` | Cohort sampling, ratio enforcement, QA helpers | Step 2 |
| `drug_utils.py` | Drug name standardization, PGx mapping | Steps 1a, 5 |
| `env_utils.py` | `is_linux`, `get_xgb_cpu_nthread`, EC2/local detection | All steps |
| `rscript_utils.py` | `run_rscript` — invoke R scripts from Python | Step 9 BupaR |
| `aws_utils.py` | IAM, STS, session helpers | Deployment |

### QA & diagnostics

| File | Key exports | Used by |
|------|-------------|---------|
| `check_cohort_parquet_controls.py` | Validate cohort control ratios | Step 2 QA |
| `check_model_events_controls.py` | Validate model_events control distribution | Step 4 QA |
| `notebook_utils.py` | Jupyter display helpers | Notebooks |

---

## `__init__.py` exports

`py_helpers/__init__.py` re-exports frequently used symbols so callers can do:

```python
from py_helpers import AGE_BANDS, age_band_to_fname, COHORTS
```

Check `__init__.py` for the current export list; not all utilities are re-exported.

---

## VS Code / Jupyter notebook scripts

`vs_code_jupyter_notebook_scripts/` contains `# %%`-delimited equivalents of the workflow notebooks for running in VS Code without Jupyter. These mirror notebooks 0–5 and are kept in sync manually:

| Script | Mirrors |
|--------|---------|
| `0_config_and_pipeline.py` | `0_config_and_pipeline.ipynb` |
| `1_cohort_workflow.py` | `1_cohort_workflow.ipynb` |
| `2_feature_importance.py` | `2_feature_importance.ipynb` |
| `3_model_train_shap_ffa.py` | `3_model_train_shap_ffa.ipynb` |
| `4_dashboard_visuals.py` | `4_dashboard_visuals.ipynb` |
| `5_build_and_deploy.py` | `5_build_and_deploy.ipynb` |

---

## Related documentation

- `docs/CrossStep_Development/README_logging.md` — logging architecture and S3 mirror paths
- `docs/CrossStep_Development/README_event_density_bins.md` — n_event_bin API
- `docs/CrossStep_Workflow/README_file_resolver.md` — file_resolver path resolution logic
