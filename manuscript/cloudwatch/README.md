# CloudWatch CLI snapshots (CH_5 Lambda benchmarks)

**Purpose:** Keep the **raw CLI output** (or saved logs) from the CloudWatch pull that fed **`{#tbl-benchmarks-cw}`** in `CH_5/ch05_bmic.qmd` (operational telemetry; **`{#tbl-benchmarks}`** is the separate synthetic-target table), so we know **what was measured** and can diff after the next deploy.

## Retention

- **Keep** JSON/text from `aws cloudwatch …` (or copied log excerpts) here **until** the next **`prepare_models.py` / Lambda redeploy** and a **new** benchmark pull.
- On the next run: **add a new dated file** (or subfolder) *or* replace the previous snapshot—your choice; just don’t delete the only record until the manuscript table is updated for that run.

## Files

| File | Role |
|:-----|:-----|
| `LAST_RUN.txt` | **ISO-8601 completion time** of the pull + which scripts were run. |
| `benchmark_snapshot.json` | Structured summary from **`lambda_timing.py`**, **`lambda_timing2.py`**, **`lambda_timing3.py`** (repo: `manuscript/scripts/metrics/`). |
| `lambda_timing*_20260331.txt` | Captured stdout from each script (rename date stamp on future runs). |
| `*.json` | Optional: raw `aws cloudwatch get-metric-statistics` output. |
| `*.log.txt` | Optional: filtered `REPORT` / `INIT_REPORT` lines pasted from the console. |

**Rerun (local, AWS creds required):**

```powershell
cd C:\Projects\pgx-analysis
python manuscript/scripts/metrics/lambda_timing.py
python manuscript/scripts/metrics/lambda_timing2.py
python manuscript/scripts/metrics/lambda_timing3.py
```

Then copy stdout into new `lambda_timing*_YYYYMMDD.txt`, update `benchmark_snapshot.json` and `LAST_RUN.txt`, and refresh the **CloudWatch operational snapshot** in `CH_5/ch05_bmic.qmd` (`{#tbl-benchmarks-cw}`).

## Related

- Manuscript: `CH_5/ch05_bmic.qmd` — **`{#tbl-benchmarks}`** (synthetic) vs **`{#tbl-benchmarks-cw}`** (this folder).
- `../METRICS.md` — CloudWatch checklist and log group name.
- `../NEXT_STEPS.md` — dated completion line pointing here.
