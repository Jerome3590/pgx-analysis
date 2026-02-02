# Create Cohort Pipeline – Check Summary

## Entry points

| Script | Purpose |
|--------|--------|
| **0_create_cohort.py** | Single partition: `--age-band`, `--event-year`, `--cohort` (opioid_ed \| ed_non_opioid \| both). Runs phases 1–4 and writes cohort parquets to S3. |
| **run_series_opioid_ed.py** | All partitions for opioid_ed (all age bands × event years). Uses `--skip-existing` to skip partitions already in S3. |
| **run_series_ed_non_opioid.py** | Same for ed_non_opioid. |
| **check_s3_cohort_completion.py** | Utility: pipeline state from pgx-repository, list cohort parquets in pgxdatalake. |

## Phases (0_create_cohort.py)

1. **Phase 1 – Data preparation:** Gold views (medical/pharmacy) for age_band/event_year.
2. **Phase 2 – Event processing:** Unified event fact table, drug exposure, tagged events.
3. **Phase 3 – Cohort creation:** OPIOID_ED and ED_NON_OPIOID cohorts (target cases + sampled controls, `is_target_case` 0/1).
4. **Phase 4 – Finalization:** QA, write cohort parquets to local staging, sync to S3 via `aws s3 cp`.

## S3 paths (gold/cohorts)

- **Layout:** `s3://pgxdatalake/gold/cohorts/cohort_name={cohort}/event_year={year}/age_band={band}/cohort.parquet`
- **Cohort names:** `opioid_ed`, `ed_non_opioid` (normalized to `non_opioid_ed` in path for ed_non_opioid).
- **Source of truth:** `py_helpers.s3_utils.get_cohort_parquet_path(cohort_name, age_band, event_year)`.

## Fixes applied (pipeline check)

1. **Skip when `--cohort both`:** Skip is based on **both** opioid_ed and ed_non_opioid parquets existing for that age_band/event_year (using `get_cohort_parquet_path`). Previously only one path was checked and it used cohort name `"both"`, so skip never applied.
2. **Skip when `--cohort opioid_ed` or `ed_non_opioid`:** Unchanged: skip if that cohort’s parquet exists (via `get_output_paths`, which uses normalized cohort name for `cohort_parquet`).
3. **cohort_utils S3 paths:** `check_cohort_exists` and `check_existing_cohorts` now use **gold/cohorts** and the same partition order (cohort_name → event_year → age_band) via `get_cohort_parquet_path` and `s3_exists`, so they match phase4 and `--skip-existing` in the series scripts.
4. **Completion state:** On successful run, `output_for_state` is set from `get_cohort_parquet_path` (when `--cohort both`) or `get_output_paths` (when a single cohort), so `mark_pipeline_completed` always receives a valid output path.

## Run examples

```bash
# Single partition, both cohorts
python 2_create_cohort/0_create_cohort.py --age-band 25-44 --event-year 2016 --cohort both

# All partitions for opioid_ed, skip existing
python 2_create_cohort/run_series_opioid_ed.py --skip-existing

# Check S3 status from local
python 2_create_cohort/check_s3_cohort_completion.py [--outputs] [--profile NAME]
```
