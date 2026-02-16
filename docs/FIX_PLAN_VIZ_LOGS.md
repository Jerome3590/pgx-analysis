# Fix plan: visualization pipeline issues and ERROR_PARAMS logging

This document lists fix plans for each issue found in the synced visualization logs, and how **missing or mismatched parameters** are now logged as `[ERROR_PARAMS]` so follow-on runs can be corrected.

---

## 1. FP-Growth: model_data not found

### Issue
- Some runs reported `model_events.parquet` not found even though **BupaR and DTW use the same model_events successfully**. That can happen if the FP-Growth subprocess resolved paths from a different base (e.g. different working directory or inferred repo root).

### Fix (implemented)
- **Same project root as BupaR/DTW:** The dashboard runner now passes `--project-root` (REPO_ROOT) into the FP-Growth subprocess (`run_single_cohort_fpgrowth.py`). FP-Growth uses that for `resolve_model_events_path(s)`, so it looks in the same 3b / 4_model_data / NVMe locations as BupaR and DTW. No separate model-data step is required for FP-Growth beyond having model_events available where BupaR/DTW already find them.
- If you invoke `run_single_cohort_fpgrowth.py` manually (e.g. from a notebook), pass `--project-root` with the repo root so path resolution matches the rest of the pipeline.

### ERROR_PARAMS logging (implemented)
- When TRAIN model_data is not found, the pipeline now logs:
  - **Python (cohort_fpgrowth):** `[ERROR_PARAMS] step=4_fpgrowth ... paths_checked=path1 | path2 | ...` and `[ERROR_PARAMS] step=4_fpgrowth path_listings: path1 -> parent contents: [...] ; path2 -> parent missing ; ...` (directory listing for each path’s parent).
  - **Python (run_single_cohort_fpgrowth):** prints `[ERROR_PARAMS]` with the result dict including `paths_checked`, `path_listings` (and `cohort_name`, `age_band`, `item_type`, `error`).
- Use `paths_checked` and `path_listings` to see which paths were tried and what files exist in each parent directory; create or symlink model data at one of them, or add the correct path to the resolver.

---

## 2. FP-Growth: insufficient transactions / no itemsets

### Issue
- For some cohort/age_bands (e.g. opioid_ed 0-12), model data exists but the number of transactions per density bin is below the minimum, so no frequent itemsets are produced and the step reports "No frequent itemsets" and creates 0 plots.

### Fix plan
1. **Accept as expected** for very small cohorts (e.g. 0-12) where counts are low; no code change required.
2. **Or relax thresholds** in `cohort_fpgrowth.py` (e.g. minimum transactions per density bin) if the goal is to still produce some itemsets for small cohorts.
3. **Re-run** is only needed if you change thresholds; otherwise treat as "completed with 0 plots".

### ERROR_PARAMS logging
- Existing warnings already identify cohort/age_band and item_type; no additional ERROR_PARAMS needed for "insufficient transactions" (not a missing-path issue).

---

## 3. FP-Growth: Itemset creation failed (SyntaxError / returncode=1)

### Issue
- Some logs show `Itemset creation failed (returncode=1)` with stderr `SyntaxError: '(' was never closed`. This indicates an older or broken version of the script was run.

### Fix plan
1. **Ensure the latest code** is deployed on the runner (pull from repo; no legacy paths like `4b_dtw_analysis` for DTW).
2. **Re-run** after deploying the current `9_dashboard_visuals/fpgrowth/` scripts.

### ERROR_PARAMS logging (implemented)
- On exception in `run_single_cohort_fpgrowth`, the script now prints `[ERROR_PARAMS]` with `cohort_name`, `age_band`, `item_type`, `error=<exception message>`. Use this to correlate with the failing cohort/age_band.

---

## 4. BupaR: model data not found

### Issue
- If `resolve_model_events_paths()` returns empty or a non-existent path, BupaR aborts with "Model data (model_events.parquet) not found".

### Fix plan
1. Same as FP-Growth: ensure model data exists at one of the resolved paths (3b or 4_model_data under NVMe / PGX_DATA_ROOT / project root), or set `PGX_DATA_ROOT` correctly.
2. Re-run the cohort/age_band after fixing.

### ERROR_PARAMS logging (implemented)
- In `create_bupar_visuals.py`, when model data is not found we now log:
  - `[ERROR_PARAMS] step=5_bupar ... paths_checked=path1 | path2 | ...` and `[ERROR_PARAMS] step=5_bupar path_listings: path1 -> parent contents: [...] ; ...` (directory listing for each path’s parent).
- Use `paths_checked` and `path_listings` to fix the path or create model data at one of those locations.

---

## 5. BupaR: process_matrix skipped (missing value where TRUE/FALSE needed)

### Issue
- R reports: `Note: process_matrix skipped due to error: missing value where TRUE/FALSE needed`. The rest of BupaR (outputs, other plots) still runs; only the process matrix step is skipped.
- **Parsing/prefixes:** process_matrix uses the **same** event log as the rest of the script (same `activity` column with `DRUG:` / `ICD:` / `CPT:` prefixes from event log creation). There is no separate parsing or item_/drug_/icd_/cpt_ mismatch; the failure is from **NA** in timestamp, activity, or case_id causing logical checks inside process_matrix to see missing values.

### Fix (implemented)
- In both `create_bupar_outputs_opioid_ed.R` and `create_bupar_outputs_non_opioid_ed.R`: filter to `target_eventlog_valid` with `!is.na(timestamp), !is.na(activity), !is.na(case_id)` before calling `process_matrix()`, and only call process_matrix when `n_events(...) > 0` and `n_cases(...) > 0`. This matches the fix in `3b_feature_importance_eda/1_bupaR/create_bupar_outputs_opioid_ed.R` and avoids the "missing value where TRUE/FALSE needed" error.

### ERROR_PARAMS logging (implemented)
- In both `create_bupar_outputs_opioid_ed.R` and `create_bupar_outputs_non_opioid_ed.R`, when `process_matrix()` fails we now print:
  - `[ERROR_PARAMS] step=5_bupar step=process_matrix cohort_name=... age_band=... error=<conditionMessage(e)>`
- Use this to identify cohort/age_band and the exact R error for follow-on fixes (e.g. NA handling).

---

## 6. DTW: features CSV not found

### Issue
- `create_dtw_visuals.py` only **loads** the DTW features CSV; it does not run `create_dtw_features.py` or `create_predictive_time_features.py`. If the CSV is missing (e.g. DTW feature step was never run or failed), visuals are skipped with a warning.

### Fix plan
1. **Run DTW feature creation** for this cohort/age_band **before** running dashboard visuals:
   - Scripts live under `9_dashboard_visuals/dtw/`: `create_predictive_time_features.py`, `create_dtw_features.py` (or the equivalent in your repo layout).
   - Ensure the **runner invokes these scripts from the correct directory** (e.g. `9_dashboard_visuals/dtw/`), not from a non-existent path like `4b_dtw_analysis` (which caused "can't open file" in some logs).
2. Ensure **model data exists** for the cohort/age_band (same as FP-Growth/BupaR); DTW feature scripts use the same `resolve_model_events_path` logic.
3. Re-run the DTW feature step, then re-run `create_dtw_visuals.py`.

### ERROR_PARAMS logging (implemented)
- In `create_dtw_visuals.py`, when the DTW features CSV is missing we now print:
  - `[ERROR_PARAMS] step=6_dtw ... expected_path=... fix=Run create_dtw_features.py ...` and `[ERROR_PARAMS] step=6_dtw path_listings: expected_path -> parent contents: [...]` (directory listing for the expected path’s parent).
- Use `expected_path` and `path_listings` to see what files exist in that folder and run the correct script, then re-run.

---

## 7. DTW: wrong script path (4b_dtw_analysis, 5d_dtw_analysis, or script not found)

### Issue
- Some logs show: `can't open file '...\\4b_dtw_analysis\\create_predictive_time_features.py': No such file or directory` or `ModuleNotFoundError: No module named 'py_helpers'`. These come from a **runner** (e.g. notebook or build script) that invokes DTW with an incorrect path or environment.

### Fix plan
1. **Use the correct script paths** in any runner (notebook, `run_dashboard_visuals.py`, or 5_build_and_deploy):
   - `create_predictive_time_features.py` → `9_dashboard_visuals/dtw/create_predictive_time_features.py`
   - `create_dtw_features.py` → use the script under `9_dashboard_visuals/dtw/` (or the path where it actually lives in the repo).
2. **Run with cwd = repo root** and ensure `PYTHONPATH` includes the repo root so `py_helpers` can be imported.
3. **Log ERROR_PARAMS on subprocess failure:** any runner that invokes these scripts should, on non-zero returncode or exception, log something like:
   - `[ERROR_PARAMS] step=6_dtw cohort_name=... age_band=... error=script failed script_path=... returncode=... stderr=...`

### ERROR_PARAMS logging (recommended)
- If you have a central runner that calls `create_predictive_time_features` or `create_dtw_features` as a subprocess, add on failure:
  - `[ERROR_PARAMS] step=6_dtw cohort_name=... age_band=... error=... script_path=... returncode=...`
- This allows correcting script path or environment on the next run.

---

## 8. DTW: DuckDB conversion error (empty string to INT32)

### Issue
- One log showed: `duckdb.duckdb.ConversionException: Could not convert string '' to INT32`. This is a data-type issue in the DTW feature or model data (e.g. a column that should be numeric contains empty strings).

### Fix plan
1. **Locate the query** that reads the column causing the conversion error (likely in `create_predictive_time_features.py` or `create_dtw_features.py`).
2. **Coerce or filter** empty strings before casting to INT32 (e.g. `NULLIF(col, '')` or filter out empty strings in SQL; or in Python, replace '' with None and use nullable Int32).
3. Re-run the cohort/age_band after the fix.

### ERROR_PARAMS logging (recommended)
- In the DTW feature scripts, when catching DuckDB (or pandas) conversion errors, log:
  - `[ERROR_PARAMS] step=6_dtw cohort_name=... age_band=... error=ConversionError ... column=...`
- This was not added in this pass; consider adding in a future change where the exception is caught.

---

## Summary: using ERROR_PARAMS in follow-on runs

- **Grep logs for `[ERROR_PARAMS]`** to get a list of missing/mismatched parameters (paths_checked, path_listings, expected_path, cohort_name, age_band, error message). `path_listings` shows the directory contents at each path checked.
- **Fix** the path, environment, or data as indicated (create model_data at one of `paths_checked`, run the correct script from `9_dashboard_visuals/dtw/`, fix PYTHONPATH, or fix data types).
- **Re-run** the affected cohort/age_band (or full pipeline) after corrections.

All new `[ERROR_PARAMS]` lines are written so they appear in the same logs as the rest of the step (Python logger or R stdout), making it easy to sync logs from S3 and analyze failures for the next run.
