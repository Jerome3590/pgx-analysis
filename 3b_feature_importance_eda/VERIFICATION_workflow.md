# Step 3b Workflow Verification

## Correctness

- **Step boundaries:** Step 3b uses only Step 1, 2, 3 artifacts. No 4_model_data.
- **Target:** Built by `create_bupar_input_from_cohort.py` from gold cohort + gold medical/pharmacy + 3a aggregated FI (admin codes removed). Event-level only; output under `3b_feature_importance_eda/outputs/cohorts/input_model_data/cohort_name={slug}/age_band={band}/model_events.parquet`.
- **Control:** Sought in (1) NVMe gold `data_root/gold/cohorts/input_model_data/...`, (2) 3b outputs. Created only when missing; when created, written under 3b with `--output-root` and filtered by `--aggregated-fi-csv` (same feature set as target).
- **Aggregated FI:** Required everywhere (workflow, R scripts, create_bupar_input, control creation). Missing FI causes early exit with clear message.
- **Gold safety:** Control under gold is never deleted; only 3b-output copies are removed when invalid or when recreating for ratio.

## Idempotency

- **Target build:** `filter_cohort_events_for_items()` (in create_model_data) skips if `out_path.exists()`; `create_bupar_input_from_cohort` always calls it, so re-run skips rebuild.
- **Target build (orchestration):** `run_bupar_post_target_analysis.py` checks if target parquet already exists at 3b path; if so, skips calling the build script.
- **Control build:** `create_control_cohort_model_data.py` returns immediately if `out_path.exists()`.
- **Control (R):** If control exists at gold or 3b, it is used. Recreate only when file missing or (when path is under 3b) invalid/ratio failed; never delete gold.
- **Workflow:** Re-running runs sync (optional), then build (idempotent), then R (reads existing parquet). No duplicate outputs; overwrites only when explicitly recreating 3b control.

## Efficiency

- **Target:** Build skipped if 3b target parquet exists (orchestrator) and again inside `filter_cohort_events_for_items` (Python). No redundant DuckDB write.
- **Control:** Lookup order is gold then 3b; one candidate match. Creation only when absent. S3 download for control skipped when in Step 3b (output_root_3b set).
- **Aggregated FI:** Resolved once per run; fail fast if missing so no wasted work downstream.
- **No 4_model_data I/O** in Step 3b.

## Summary

| Check           | Status |
|----------------|--------|
| Correct (step boundaries, paths, FI required) | Yes |
| Idempotent (re-run safe, no unintended overwrites) | Yes |
| Efficient (skip existing target/control, no gold delete) | Yes |
