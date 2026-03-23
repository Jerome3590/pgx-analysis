# Pipeline Logging — Architecture and Troubleshooting Guide

This document describes the final logging approach used across all pipeline steps, how logs are stored locally (EC2) and in S3, and how to use logs for troubleshooting.

---

## Final Logging Architecture

### Guiding Principles

1. **File-first**: All structured log output goes to a file. Console is reserved for high-visibility summary lines only.
2. **Consistent paths**: Every step writes to `logs/{step_name}/` at the repo root on EC2, and mirrors to `s3://pgx-repository/{step_name}_log/{cohort}/{age_band}/` on completion.
3. **Append mode**: Log files are appended (`mode="a"`) so re-runs accumulate history. Steps using `setup_pipeline_logger` timestamp their filenames instead (write mode).
4. **Best-effort S3**: S3 mirroring is always wrapped in `try/except` — a missing S3 bucket or network failure never blocks the pipeline.

---

## EC2 Local Log Locations

All logs write to `{repo_root}/logs/{step_name}/` regardless of which directory the script is run from.

| Step | Script | Log path | Filename pattern |
|------|--------|----------|-----------------|
| **4** | `create_model_data.py` | `logs/4_model_data/` | `create_model_data_{cohort}_{ab}.log` |
| **5** | `run_analysis.py` | `logs/5_pgx_analysis/` | `pgx_{cohort}_{ab}.log` |
| **6** | `run_final_model.py` | `logs/6_final_model/` | `final_model_{cohort}_{ab}.log` |
| **7** | `run_shap_analysis.py` | `logs/7_shap_analysis/` | `shap_{cohort}_{ab}[_{bin}].log` |
| **8** | `XGBoostSymbolicExplainer` | `logs/8_ffa_analysis/` | `ffa_{cohort}_{ab}.log` |
| **9 DTW** | `create_dtw_*.py` | `logs/9_dtw/` | `{script}_{cohort}_{ab}_{timestamp}.log` |
| **9 DTW** | `extract/summarize_extreme_density.py` | `logs/9_dtw/` | `{extract\|summarize}_extreme_density_{cohort}_{ab}.log` |
| **9 FPGrowth** | `create_fpgrowth_visuals.py` | `logs/9_fpgrowth/` | `create_fpgrowth_visuals_{cohort}_{ab}_{timestamp}.log` |
| **9 BupaR** | `create_bupar_visuals.py` | `logs/9_bupar/` | `create_bupar_visuals_{cohort}_{ab}_{timestamp}.log` |
| **9 cohort_pgx** | `fetch_vip_reports.py`, `build_network_topology.py` | `logs/9_cohort_pgx/` | `{script}_{cohort}_{ab}_{timestamp}.log` |

> **Steps 1a, 1b, 2, 3a, 3b, 10** have no file logging — these are data transformation scripts or notebooks where console/print output is sufficient.

---

## S3 Mirror Locations

Every active step mirrors its log to S3 on completion (best-effort):

```
s3://pgx-repository/{step_name}_log/{cohort}/{age_band}/{filename}
```

| Step | S3 prefix |
|------|-----------|
| 4 | `s3://pgx-repository/4_model_data_log/` |
| 5 | `s3://pgx-repository/5_pgx_analysis_log/` |
| 6 | `s3://pgx-repository/6_final_model_log/` |
| 7 | `s3://pgx-repository/7_shap_analysis_log/` |
| 8 | `s3://pgx-repository/8_ffa_analysis_log/` |
| 9 DTW | `s3://pgx-repository/9_dtw_log/` |
| 9 FPGrowth | `s3://pgx-repository/9_fpgrowth_log/` |
| 9 BupaR | `s3://pgx-repository/9_bupar_log/` |
| 9 cohort_pgx | `s3://pgx-repository/9_cohort_pgx_log/` |

S3 mirroring is implemented via `mirror_log_to_s3()` in `py_helpers/fe_monitor.py`.

---

## Console Output — What You See

Console output is intentionally minimal. Only high-visibility summary lines print to stdout:

| Event | Console line |
|-------|-------------|
| **Job start** (step 9 scripts) | `[9_dtw] opioid_ed/13-24 started — log: logs/9_dtw/create_dtw_features_opioid_ed_13_24_20260323_030000.log` |
| **Job complete** (step 9 scripts) | `[9_dtw] opioid_ed/13-24 done (4m32s) — OK \| warnings=0 errors=0` |
| **Job complete with errors** | `[9_dtw] opioid_ed/13-24 done (1m12s) — ERROR \| warnings=1 errors=2` + first 3 error lines |
| **WARNING+ messages** | Printed via console handler at `WARNING` level |

Steps 4–7 are **file-only** — no console output at all. Monitor those via the log file or S3.

---

## Two Logging Patterns

### Pattern A — Manual `_get_logger()` (Steps 4, 5, 6, 7)

Used by the main pipeline scripts for steps 4–7. Implemented inline in each script's `main()` or `_get_logger()` helper.

```python
logs_dir = PROJECT_ROOT / "logs" / "4_model_data"
logs_dir.mkdir(parents=True, exist_ok=True)
log_path = logs_dir / f"create_model_data_{cohort}_{age_band_fname}.log"
logger = logging.getLogger(f"4_model_data.{cohort}.{age_band_fname}")
logger.setLevel(logging.INFO)
if not logger.handlers:
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
logger.propagate = False
```

**Characteristics:**
- File handler only (no console)
- Append mode — re-runs accumulate in the same file
- No timestamp in filename — easy to `tail -f` a known path
- S3 mirror called explicitly at end of `main()` via `mirror_log_to_s3(...)`
- Format: `2026-03-23 03:00:00,123 - INFO - message`

### Pattern B — `setup_pipeline_logger()` (Step 9 Scripts)

Used by dashboard visualization scripts. Implemented in `py_helpers/pipeline_logger.py`.

```python
from py_helpers.pipeline_logger import setup_pipeline_logger

pl = setup_pipeline_logger(
    step_name="9_dtw",
    cohort=args.cohort,
    age_band=args.age_band,
    script_name="create_dtw_features",
)
pl.info("Processing %s rows", len(df))
pl.warning("Missing data in %d records", missing_count)
pl.log_summary()  # prints completion line to console + mirrors to S3
```

**Characteristics:**
- File handler at DEBUG level (verbose), console handler at WARNING level
- Timestamped filename — each run creates a new file (write mode)
- S3 mirror triggered automatically by `pl.log_summary()` at end of run
- Tracks `pl.errors` and `pl.warnings` lists for structured summary
- Returns a `PipelineLogger` wrapper with `.info()`, `.warning()`, `.error()`, `.exception()` methods
- Format: `2026-03-23 03:00:00 | INFO     | 9_dtw.opioid_ed.13-24.create_dtw_features | message`

### Pattern C — FFA Explainer Classes (Step 8)

Used by `XGBoostSymbolicExplainer` and `CatBoostSymbolicExplainer`. Logs to two locations simultaneously when `cohort` is set in `PathConfig`.

```python
path_config = PathConfig(
    model_path=...,
    output_dir="8_ffa_analysis/outputs/opioid_ed/13_24/xgboost",
    age_band="13-24",
    cohort="opioid_ed",          # enables standard log location
)
explainer = XGBoostSymbolicExplainer(path_config)
# ... run analysis ...
explainer.mirror_logs_to_s3()    # call after analysis completes
```

**Characteristics:**
- Primary log: `{output_dir}/axp_analysis.log` — co-located with analysis outputs
- Secondary log: `logs/8_ffa_analysis/ffa_{cohort}_{ab}.log` — standard location (only when `cohort` set)
- S3 mirror triggered by explicit `explainer.mirror_logs_to_s3()` call
- `cohort` was added to `PathConfig` specifically to enable standard-location logging

---

## Troubleshooting with Logs

### Finding the right log

```bash
# Most recent run for a specific cohort/age band
ls -lt logs/6_final_model/ | grep opioid_ed_13_24 | head -5

# All errors across a step
grep -i "error\|exception\|traceback" logs/4_model_data/create_model_data_opioid_ed_13_24.log

# Step 9 — list runs in chronological order (timestamped filenames)
ls -lt logs/9_dtw/ | grep opioid_ed_13_24
```

### Tailing a live run

Steps 4–7 use append mode with known filenames — easy to tail:
```bash
tail -f logs/6_final_model/final_model_opioid_ed_13_24.log
tail -f logs/7_shap_analysis/shap_opioid_ed_13_24_extreme.log
```

Step 9 scripts use timestamped filenames. Find the active one:
```bash
ls -lt logs/9_dtw/ | head -3   # most recently modified
tail -f logs/9_dtw/$(ls -t logs/9_dtw/ | head -1)
```

### Fetching logs from S3

```bash
# Download step 6 log for a cohort/age_band
aws s3 cp s3://pgx-repository/6_final_model_log/opioid_ed/13-24/ ./local_logs/ --recursive

# List all available step 7 logs
aws s3 ls s3://pgx-repository/7_shap_analysis_log/opioid_ed/

# Quick check last 50 lines of a remote log
aws s3 cp s3://pgx-repository/6_final_model_log/opioid_ed/13-24/final_model_opioid_ed_13_24.log - | tail -50
```

### Common failure patterns

| Symptom | Where to look | What to search for |
|---------|--------------|-------------------|
| Step 6 fails silently | `logs/6_final_model/final_model_{cohort}_{ab}.log` | `ERROR`, `Traceback` |
| SHAP skipping all models | `logs/7_shap_analysis/shap_{cohort}_{ab}_{bin}.log` | `[SKIP]`, `not found` |
| FFA producing no rules | `8_ffa_analysis/outputs/{cohort}/{ab}/xgboost/axp_analysis.log` | `empty`, `no rules` |
| DTW visual missing | `logs/9_dtw/{script}_{cohort}_{ab}_{timestamp}.log` | `WARNING`, `ERROR` |
| Step 4 model data empty | `logs/4_model_data/create_model_data_{cohort}_{ab}.log` | `0 rows`, `no data` |
| S3 mirror failed | Any log file above | `Failed to mirror log` |

### Checking if a step completed successfully

Steps with S3 mirrors write a final info line before mirroring. If S3 has a log, the step completed (or errored cleanly):

```bash
# Step completed if log exists in S3
aws s3 ls s3://pgx-repository/6_final_model_log/opioid_ed/13-24/

# Step 9 — check for completion banner in log
grep "done (" logs/9_dtw/create_dtw_features_opioid_ed_13_24_*.log | tail -1
# Output: [9_dtw] opioid_ed/13-24 done (4m32s) — OK | warnings=0 errors=0
```

---

## Adding Logging to a New Step

Use **Pattern A** for main pipeline runners with `--cohort`/`--age_band` CLI args:

```python
import logging
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def _get_logger(cohort: str, age_band: str):
    logs_dir = PROJECT_ROOT / "logs" / "N_step_name"
    logs_dir.mkdir(parents=True, exist_ok=True)
    ab_fname = age_band.replace("-", "_")
    log_path = logs_dir / f"step_name_{cohort}_{ab_fname}.log"
    logger = logging.getLogger(f"N_step_name.{cohort}.{ab_fname}")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    logger.propagate = False
    return logger, log_path

# In main():
logger, log_path = _get_logger(args.cohort, args.age_band)
# ... run pipeline ...
try:
    from py_helpers.fe_monitor import mirror_log_to_s3
    mirror_log_to_s3("N_step_name", args.cohort, args.age_band, log_path, logger)
except Exception:
    pass
```

Use **Pattern B** (`setup_pipeline_logger`) for dashboard visualization scripts in `9_dashboard_visuals/`.

---

## Key Source Files

| File | Purpose |
|------|---------|
| `py_helpers/pipeline_logger.py` | `PipelineLogger` class + `setup_pipeline_logger()` (Pattern B) |
| `py_helpers/fe_monitor.py` | `mirror_log_to_s3()` — S3 upload utility |
| `py_helpers/logging_utils.py` | Legacy utilities (`create_cohort_logger`, `create_feature_importance_logger`) — **not called in active pipeline**, retained for reference |
| `8_ffa_analysis/base_symbolic_explainer.py` | `mirror_logs_to_s3()` method on explainer base class |
| `8_ffa_analysis/xgboost_axp_explainer.py` | `PathConfig.cohort` field + secondary `logs/8_ffa_analysis/` handler |
