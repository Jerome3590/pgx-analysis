# DTW workflow: S3 checkpoints and alignment with pipeline

## What `6_dtw_checkpoint` is

`6_dtw_checkpoint` is a **best-effort mirror** of the DTW output CSV written by `create_dtw_visuals.py`:

- **Location:** `s3://pgx-repository/6_dtw_checkpoint/{cohort}/{age_band}/dtw_added_features_{cohort}_{age_band}.csv`
- **Written by:** `py_helpers.fe_monitor.mirror_checkpoint_to_s3()` with `feature_step="6_dtw"` (called from `create_dtw_visuals.py` after writing the local file).
- **Purpose:** Observability (e.g. `check_dtw_s3_status.py`) and optional recovery if local/NVMe output is lost.
- **Not used for skip logic:** Nothing in the repo checks for this path to decide whether to run DTW. Idempotency for DTW is **local only**: `create_dtw_visuals` skips when the local output file exists (`out_path.exists()`).

So `6_dtw_checkpoint` is a **separate convention** from the main pipeline checkpoints.

## Pipeline checkpoints (actual final workflow)

The rest of the pipeline uses **`py_helpers.checkpoint_utils`**:

- **Location:** `s3://pgx-repository/pipeline_checkpoints/{step_name}/{cohort}/{age_band}/checkpoint.json`
- **Step names:** e.g. `3a_feature_importance`, `4_model_data`, `6_final_model`, `9_dashboard_models`, `9_dashboard_cpic`.
- **Purpose:** Idempotency — notebooks call `check_step_checkpoint_exists()` before running and `save_step_checkpoint()` after success so steps can be skipped on re-run and status is visible in one place.

Dashboard visuals (notebook 4) currently **do not** use this system for BupaR, DTW, or FP-Growth; they rely on local output existence and `FORCE_RERUN`.

## Aligning DTW with the pipeline

To match the actual final workflow you can:

1. **Use pipeline checkpoints for DTW**  
   In `create_dtw_visuals.py`: after a successful run, call `save_step_checkpoint("10_dashboard_visuals", cohort_name, age_band, ...)` from `checkpoint_utils`. Optionally in `4_dashboard_visuals.ipynb`: before running DTW for a (cohort, age_band), call `check_step_checkpoint_exists("10_dashboard_visuals", cohort, age_band)` and skip if True (unless `FORCE_RERUN`).

2. **Treat `6_dtw_checkpoint` as optional or remove it**  
   - **Keep:** Continue mirroring the CSV to `6_dtw_checkpoint` for status scripts and artifact backup.  
   - **Remove:** Stop calling `mirror_checkpoint_to_s3("6_dtw", ...)` so the single source of truth for “DTW done” is `pipeline_checkpoints/10_dashboard_visuals/...` (and optionally pgxdatalake outputs).

3. **Status script**  
   `check_dtw_s3_status.py` reports from `pipeline_checkpoints/10_dashboard_visuals/` and treats `6_dtw_checkpoint` as optional/legacy.

## Summary

| Location | System | Used for skip? | Purpose |
|----------|--------|----------------|---------|
| `pipeline_checkpoints/{step}/{cohort}/{age_band}/checkpoint.json` | checkpoint_utils | Yes (for steps that use it) | Idempotency and status |
| `6_dtw_checkpoint/{cohort}/{age_band}/*.csv` | fe_monitor | No | Mirror of DTW CSV; status/recovery only |

DTW uses `pipeline_checkpoints` with step name `10_dashboard_visuals` (pipeline phase 10). Optionally dropping or keeping `6_dtw_checkpoint` is up to you.
