# -*- coding: utf-8 -*-
# Auto-generated from 1_cohort_workflow.ipynb (VS Code Python notebook script format)

# %% [markdown]
# # Cohort Creation Workflow (Step 2)
#
# Final workflow for creating cohorts. One cell per cohort series; run Configuration once, then the cohort cell(s) you need.
#
# ## Order of operations
#
# 1. **Configuration** — Set project root and paths (run once).
# 2. **Cohort 1 (OPIOID_ED)** — Builds all age-band × event-year partitions. Each partition run writes **both** OPIOID_ED and POLYPHARMACY parquets to S3.
# 3. **Cohort 2 (POLYPHARMACY)** — Same partitions; use to force-rebuild polypharmacy only or run on another machine. With `--skip-existing`, partitions that already have both cohorts are skipped.
# 4. **Check status** — Pipeline state and durations in pgx-repository; cohort parquets in pgxdatalake. Use `--outputs` for file listing; use `--logs` for build logs.
#
# ## Cohorts
#
# | Series | Script | Target |
# |--------|--------|--------|
# | **OPIOID_ED** | `run_series_opioid_ed.py` | F11.20 (opioid use disorder) |
# | **POLYPHARMACY** | `run_series_ed_non_opioid.py` | Time-windowed HCG ED events |
#
# ## Prerequisites
#
# - **Step 1a** — APCD input data in gold (medical/pharmacy).
# - **Step 1b** (optional) — Event filter with `--before-cohorts`; cohort creation uses filtered gold when present.

# %% [markdown]
# ## Configuration

# %%
import sys
from pathlib import Path

# Project root (notebook at repo root)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

PYTHON_BIN = Path(sys.executable)
LOG_DIR = PROJECT_ROOT / "2_create_cohort" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

print(f"Project root: {PROJECT_ROOT}")
print(f"Python: {PYTHON_BIN}")
print(f"Log dir: {LOG_DIR}")

# %% [markdown]
# ## Cohort 1: OPIOID_ED
#
# Runs the OPIOID_ED series for all age bands and event years. Each partition writes both OPIOID_ED and POLYPHARMACY parquets to S3. Partitions that already have the cohort output are skipped (`--skip-existing`).

# %%
import subprocess

script = PROJECT_ROOT / "2_create_cohort" / "run_series_opioid_ed.py"
cmd = [str(PYTHON_BIN), str(script), "--skip-existing", "--concurrent-workers", "1"]
print("Running:", " ".join(cmd))
subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=True)

# %% [markdown]
# ## Cohort 2: POLYPHARMACY (ed_non_opioid)
#
# Runs the POLYPHARMACY series for all age bands and event years. Partitions that already have both cohort outputs are skipped (`--skip-existing`). Use this cell to run polypharmacy-only or to force-rebuild (omit `--skip-existing` in the code cell if needed).

# %%
import subprocess

script = PROJECT_ROOT / "2_create_cohort" / "run_series_ed_non_opioid.py"
cmd = [str(PYTHON_BIN), str(script), "--skip-existing", "--concurrent-workers", "1"]
print("Running:", " ".join(cmd))
subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=True)

# %% [markdown]
# ## Check S3 completion status
#
# Lists pipeline state (pgx-repository) with status and **duration** per partition, and cohort parquet counts in pgxdatalake. Add `--outputs` to the script args to list each parquet with LastModified and size; add `--logs` to list build logs from pgx-repository.

# %%
import subprocess
script = PROJECT_ROOT / "2_create_cohort" / "check_s3_cohort_completion.py"
# Use --outputs to list each cohort.parquet with LastModified and size
subprocess.run([str(PYTHON_BIN), str(script), "--outputs"], cwd=str(PROJECT_ROOT))

# %%
