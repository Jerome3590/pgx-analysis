# CPU utilization profile vs. local and S3 logs (pgx-dashboard-1b)

Correlation of the **CPU utilization graph** (instance `i-0c968462d413a1028`, pgx-dashboard-1b, ~02:30–14:00 UTC) with **local logs** and **S3-synced logs** in this repo.

---

## Log sources used

| Source | Location | Notes |
|--------|----------|--------|
| **S3-synced viz logs** | `logs/viz_sync/` | Synced from `s3://pgx-repository/{4_fpgrowth_log,5_bupar_log,6_dtw_log}/`; EC2 run timestamps (UTC). |
| **Local FP-Growth** | `logs/fpgrowth_*.txt` | Local runs (if any). |
| **Dashboard visuals logs** | `9_dashboard_visuals/logs/` | e.g. `feature_engineering/5_bupar/`, `feature_engineering/dtw/` (various dates). |

S3 log paths referenced in the logs themselves:

- **FP-Growth:** `s3://pgx-repository/4_fpgrowth_log/{cohort}/{age_band}/fpgrowth_{cohort}_{age_band}.log`
- **BupaR:** uploads to dashboard bucket (e.g. `s3://jerome-dixon.io/vcu/pgx-risk-calculator/bupar/{cohort}/{age_band}/plots/`)
- **DTW:** `s3://pgx-repository/6_dtw_log/...`

---

## Timeline from S3-synced logs (2026-02-16 UTC, EC2/linux)

### FP-Growth (4_fpgrowth_log) — sequential

| Time window (UTC) | What |
|-------------------|------|
| **07:12:30 – 07:12:39** | All 16 (cohort, age_band) FP-Growth jobs **start** (opioid_ed then non_opioid_ed). |
| **07:12 – 08:31** | Cohorts run one after another; first completions and S3 uploads (e.g. opioid_ed 25-44 **completed 08:31:55**). |
| **08:31 – 08:54** | Remaining cohorts finish; last log activity **08:54:43 – 08:54:52** (all 16 logs). |
| **08:54:40** | Extra run for opioid_ed 25-44 (retry or second pass; itemsets not found, 0 plots). |
| **10:53:34 – 10:53:37** | Single run: **opioid_ed 0-12** (quick, ~3 s; no itemsets). |

So **FP-Growth occupies 07:12 – 08:55 UTC** (~1 h 43 min), with one later run at 10:53.

### BupaR (5_bupar_log) — parallel

| Time window (UTC) | What |
|-------------------|------|
| **10:45:09** | All 18 BupaR jobs **start** at once (parallel). |
| **10:45:27 – 10:53:34** | Jobs finish over ~8 min; slowest **opioid_ed 25-44** ends **10:53:33** (duration_sec=503). |
| **10:53:34** | BupaR batch effectively done; FP-Growth opioid_ed 0-12 starts immediately after. |

So **BupaR occupies 10:45 – 10:53 UTC** (~8 min).

### DTW (6_dtw_log)

- Logs in `logs/viz_sync/6_dtw_log/` are from **2025-12-31 / 2026-01-01** and **Windows** (local dev), not from EC2 on 2026-02-16.
- **No DTW run on pgx-dashboard-1b** for the 2026-02-16 UTC window in these logs.

---

## Match to CPU profile (02:30 – 14:00 UTC)

| CPU feature | Time (UTC) | Match from logs |
|------------|------------|------------------|
| **Sustained ~50%** | **06:15 – 07:15** | No viz step in our logs before **07:12**. So either: (1) another process (sync, model data, or other pipeline), or (2) different day. **07:12** is the **start of FP-Growth** and sits at the **end** of this 50% block. |
| **Low** | 07:15 – 08:30 | FP-Growth is running (07:12 – 08:55). Possible if FP-Growth is I/O-bound or single-threaded so average CPU stays low despite one cohort at a time. |
| **Ramp + 99.9% spike** | **08:30 – 09:15** | FP-Growth still running until **08:54–08:55**. The **09:15 spike** has **no direct event** in our viz logs (no step at 09:15). Could be: same run’s final cleanup, another job, or a different day. |
| **Moderate ~10–20%** | 09:15 – 11:00 | Could be post–FP-Growth I/O (uploads, plots) or other background work. |
| **Sustained ~10–15%** | **11:00 – 14:00** | **BupaR batch 10:45 – 10:53** falls in this window. R (BupaR) often shows moderate CPU. After 10:53, light activity (e.g. FP-Growth 0-12, dashboard, or idle). |

Summary:

- **07:12 – 08:55 UTC:** FP-Growth (sequential) on EC2; matches the period that includes the ramp and the end of the 50% block; the **09:15** spike is not directly explained by these viz logs.
- **10:45 – 10:53 UTC:** BupaR (parallel); matches the **11:00–14:00** moderate-CPU band (10:45–10:53 is the active part).

---

## Local logs (this machine)

- **`logs/fpgrowth_non_opioid_ed_75_84.txt`**, **`logs/fpgrowth_opioid_ed_0_12.txt`** — local FP-Growth runs; timestamps may be local time, not EC2/UTC.
- **`9_dashboard_visuals/logs/feature_engineering/5_bupar/*.log`** — last write **2026-02-15**; likely local or earlier EC2 runs, not the 2026-02-16 EC2 run above.

For **pgx-dashboard-1b** and the CPU graph, the **authoritative correlation** is with **S3-synced logs** under `logs/viz_sync/` (EC2, UTC).

---

## S3 outputs (from log lines)

- **FP-Growth logs:** `s3://pgx-repository/4_fpgrowth_log/{cohort}/{age_band}/fpgrowth_{cohort}_{age_band}.log`
- **BupaR plots:** e.g. `s3://jerome-dixon.io/vcu/pgx-risk-calculator/bupar/{cohort}/{age_band}/plots/` (14 files per cohort in the 25-44 example).
- **DTW:** `s3://pgx-repository/6_dtw_log/...` (no 2026-02-16 EC2 run in current sync).

To refresh the correlation after a new run: re-sync from S3 into `logs/viz_sync/`, then re-run the first/last timestamp extraction (or grep for `START time=` / `END_OK` / `Uploaded to`) to update this timeline.

---

## Recommendations (from this analysis)

### 1. Fix model data and paths so viz jobs succeed

- **FP-Growth (update):** Model events are available where BupaR and DTW use them. The dashboard runner now passes `--project-root` (REPO_ROOT) into the FP-Growth subprocess so it uses the same path resolution (3b / 4_model_data / NVMe) as BupaR; no separate model-data step is needed. If FP-Growth still reports not found, ensure the runner invokes it with `--project-root` set to the repo root.
- **FP-Growth (if not using --project-root):** Many cohorts failed with “model_data not found.” Ensure `model_events.parquet` exists on EC2 at one of the resolved paths (see `[ERROR_PARAMS]` / `path_listings` in logs). Run the model-data step (or sync from S3) **before** the dashboard-viz run, then re-run FP-Growth for failed cohort/age_bands.
- **DTW:** Viz step only loads CSVs; it does not create them. Run **DTW feature creation** on EC2 (`9_dashboard_visuals/dtw/create_dtw_features.py` and related scripts) with correct paths and `PYTHONPATH` (repo root), then run `create_dtw_visuals`. See `docs/FIX_PLAN_VIZ_LOGS.md` for path and conversion-error fixes.

### 2. Use ERROR_PARAMS and path_listings on every run

- After each run, **grep logs for `[ERROR_PARAMS]`** (and `path_listings`) to get missing paths, expected_path, and cohort/age_band. Fix paths or data, then re-run only the affected combinations.
- Keeps follow-on runs focused and avoids re-running successful steps.

### 3. Make the CPU profile interpretable next time

- **Tag or log pipeline phase** (e.g. “PIPELINE_PHASE=FP-Growth” at start/end of the step) so the next CPU graph can be matched to a known phase. Optionally log one line per major phase (sync, BupaR, FP-Growth, DTW) with UTC timestamp.
- **Identify the 06:15–07:15 and 09:15 activity:** Log what runs *before* FP-Growth (e.g. S3 sync, model-data prep). If the 99% spike recurs, add a single log line immediately before/after the heaviest CPU call (e.g. FP-Growth itemset mining) with cohort/age_band and timestamp so you can tie the spike to a specific job.

### 4. BupaR process_matrix (optional)

- All BupaR runs completed but **process_matrix** was skipped (“missing value where TRUE/FALSE needed”). To enable it: filter the event log to drop NA in timestamp/activity/case_id before `process_matrix()` (see `3b_feature_importance_eda/1_bupaR` scripts). Low priority if the rest of BupaR output is sufficient.

### 5. Run order and resource use

- **Order:** Model data (and any sync) → BupaR (parallel) → FP-Growth (sequential) → DTW visuals (if CSVs exist). This matches current logs and avoids FP-Growth running without data.
- **FP-Growth:** Already sequential for memory; keep one cohort at a time. If the 99% spike is FP-Growth, it’s expected for a single cohort’s mining phase; no change needed unless you want to throttle (e.g. limit threads) to smooth CPU.

### 6. Re-sync and re-correlate after changes

- After fixing paths and re-running, **sync S3 logs** into `logs/viz_sync/` again and regenerate the first/last timestamp table (or run a small script that extracts START/END from each log). Update `docs/CPU_PROFILE_LOG_CORRELATION.md` so the next CPU profile can be matched with minimal guesswork.
