# README: 32-core / 1TB EC2 Cohort Pipeline (Two Jobs, Sequential Partitions)

This runbook is optimized for a **32 vCPU, 1TB RAM EC2 instance** with **NVMe**. The goal is:

* run **two jobs concurrently** (one per cohort type)
* process **age_band × event_year** sequentially *inside each job*
* prioritize heavy partitions (**25–44** and **65–74**)
* avoid CPU oversubscription and DuckDB memory thrash

## 0) Summary of the required updates

### Update A — DuckDB threads must be configurable (not hard-coded to 1)

In `duckdb_utils.py`, `create_simple_duckdb_connection()` currently forces:

* `PRAGMA threads=1`

Change it to use an env var (default safe):

```python
threads = int(os.getenv("PGX_DUCKDB_THREADS", "1"))
conn.sql(f"PRAGMA threads={threads}")
```

Source: `py_helpers/duckdb_utils.py`

### Update B — Memory limit clamp must allow large RAM instances

`calculate_memory_limit_per_worker()` clamps per-worker memory to max **4GB**, which is too small for heavy joins on 1TB RAM.

Change clamp to allow larger limits on EC2:

```python
# old: per_worker_gb = max(0.5, min(4.0, per_worker_gb))
per_worker_gb = max(4.0, min(256.0, per_worker_gb))
```

Also: explicitly pass or set total workers to **2** (since we run 2 jobs total):

* set `PGX_TOTAL_WORKERS=2` (recommended) and read it in the function
  OR
* call `create_duckdb_conn(..., total_workers=2)` from the job entrypoint.

Source: `py_helpers/duckdb_utils.py`

### Update C — Do not multiply parallelism (separate "job concurrency" vs "in-job workers")

In orchestrators, keep these separate:

* `JOB_PARALLELISM` = number of concurrent cohort processes (here: 2)
* `--concurrent-workers` = internal workers used by your cohort script (recommend: 1–2)
* DuckDB threading is controlled separately via `PGX_DUCKDB_THREADS`

Do **not** reuse the same variable for all three.

### Update D — Optional: reduce noise and avoid collisions

* Make profiling filenames include `{cohort}/{age_band}/{event_year}` if profiling is enabled.
* For logs, include milliseconds or PID to avoid collisions.

---

## 1) Recommended CPU & memory allocation on this instance

We reserve ~4 cores for OS/network overhead.

We run **two processes**:

### Process 1 (heavier): `ed_non_opioid`

* `PGX_DUCKDB_THREADS=16`
* `PGX_DUCKDB_MEMORY_LIMIT=256GB` (or allow auto via per-worker calc with clamp)
* `--concurrent-workers=1` (or 2 if your script overlaps I/O safely)

### Process 2 (lighter): `opioid_ed`

* `PGX_DUCKDB_THREADS=12`
* `PGX_DUCKDB_MEMORY_LIMIT=192GB`
* `--concurrent-workers=1` (or 2)

This uses ~28 threads total, leaving headroom for OS + uploads.

---

## 2) NVMe setup and temp directories

DuckDB spills should go to NVMe.

Expected temp base:

* `/mnt/nvme/duckdb_tmp`

`duckdb_utils.get_worker_temp_dir()` already prefers NVMe when available. Source: `py_helpers/duckdb_utils.py`

### One-time prep (shell)

```bash
sudo mkdir -p /mnt/nvme/duckdb_tmp
sudo chown -R "$USER":"$USER" /mnt/nvme/duckdb_tmp
```

---

## 3) Job ordering (heavy partitions first)

Within each cohort job, process age bands in this order:

1. `25-44`
2. `65-74`
3. remaining bands (descending expected size)

Years: 2016–2019 (or your configured list).

---

## 4) Notebook run cells (two concurrent jobs)

Below assumes you are in a Jupyter notebook on the EC2 instance.

### Cell 1 — Define common paths and env (Python)

```python
import os
from pathlib import Path

PROJECT_ROOT = Path("/home/pgx3874/pgx-analysis")  # adjust to your path
PYTHON_BIN = "/home/pgx3874/jupyter-env/bin/python3.11"  # adjust to your Python path

# Always use NVMe spill directory for DuckDB
os.environ["DUCKDB_TMP_DIRECTORY"] = "/mnt/nvme/duckdb_tmp"

# Two-job plan: treat total workers as 2 for memory calc
os.environ["PGX_TOTAL_WORKERS"] = "2"

# Optional: enable DuckDB profiling (only if debugging performance)
# os.environ["PGX_ENABLE_DUCKDB_PROFILING"] = "1"
```

### Cell 2 — Launch `ed_non_opioid` (shell via notebook)

**IMPORTANT**: `0_create_cohort.py` processes **one partition at a time** (one age_band + one event_year). 

Use the wrapper script `run_series_ed_non_opioid.py` which processes all partitions **sequentially in heavy-first order**:

```bash
%%bash
set -euo pipefail

export PGX_DUCKDB_THREADS=16
export PGX_DUCKDB_MEMORY_LIMIT=256GB
export PGX_TOTAL_WORKERS=2

cd /home/pgx3874/pgx-analysis

# Launch ed_non_opioid
nohup /home/pgx3874/jupyter-env/bin/python3.11 2_create_cohort/run_series_ed_non_opioid.py \
  --skip-existing \
  --concurrent-workers 1 \
  --python-bin /home/pgx3874/jupyter-env/bin/python3.11 \
  > logs/ed_non_opioid_run.log 2>&1 &
echo "ed_non_opioid PID: $!"
```

**Processing order**: `25-44` → `65-74` → `45-54` → `55-64` → `75-84` → `85-114` → `13-24` → `0-12`

### Cell 3 — Launch `opioid_ed` (shell via notebook)

Use the wrapper script `run_series_opioid_ed.py` which processes all partitions **sequentially in heavy-first order**:

```bash
%%bash
set -euo pipefail

export PGX_DUCKDB_THREADS=12
export PGX_DUCKDB_MEMORY_LIMIT=192GB
export PGX_TOTAL_WORKERS=2

cd /home/pgx3874/pgx-analysis

# Launch opioid_ed
nohup /home/pgx3874/jupyter-env/bin/python3.11 2_create_cohort/run_series_opioid_ed.py \
  --skip-existing \
  --concurrent-workers 1 \
  --python-bin /home/pgx3874/jupyter-env/bin/python3.11 \
  > logs/opioid_ed_run.log 2>&1 &
echo "opioid_ed PID: $!"
```

**Processing order**: `25-44` → `65-74` → `45-54` → `55-64` → `75-84` → `85-114` → `13-24` → `0-12`

### Cell 4 — Monitor logs

```bash
%%bash
tail -n 50 logs/ed_non_opioid_run.log
```

```bash
%%bash
tail -n 50 logs/opioid_ed_run.log
```

### Cell 5 — Live follow (optional)

```bash
%%bash
tail -f logs/ed_non_opioid_run.log
```

Stop with Ctrl+C in notebook output.

---

## 5) Wrapper scripts for sequential processing

**IMPORTANT**: `0_create_cohort.py` processes **one partition at a time** (one age_band + one event_year). 

Two wrapper scripts are provided to process all partitions sequentially in heavy-first order:

* `run_series_ed_non_opioid.py` - Processes all ed_non_opioid partitions
* `run_series_opioid_ed.py` - Processes all opioid_ed partitions

### Wrapper script features

* **Heavy-first ordering**: Processes `25-44` and `65-74` first
* **Sequential processing**: One partition at a time (no parallelization within job)
* **Skip existing**: Use `--skip-existing` to skip partitions already in S3
* **Progress tracking**: Shows progress and summary at end

### Usage

```bash
# Process all ed_non_opioid partitions (heavy first)
python 2_create_cohort/run_series_ed_non_opioid.py --skip-existing

# Process all opioid_ed partitions (heavy first)
python 2_create_cohort/run_series_opioid_ed.py --skip-existing
```

### Processing order

Both wrappers process partitions in this order:

1. **Heavy partitions first**: `25-44`, `65-74`
2. **Medium partitions**: `45-54`, `55-64`, `75-84`, `85-114`
3. **Light partitions**: `13-24`, `0-12`

Within each age band, processes years: `2016` → `2017` → `2018` → `2019`

---

## 6) Safety: locks and existing cohorts

You already have helpers to avoid reprocessing when cohorts exist and to detect locks. Source: `py_helpers/cohort_utils.py`

Before running, you can optionally list what needs processing:

```python
from py_helpers.cohort_utils import check_existing_cohorts

jobs = check_existing_cohorts(
    age_bands=["25-44","65-74","45-54","55-64","75-84","85-114","13-24","0-12"],
    event_years=[2016,2017,2018,2019]
)
len(jobs), jobs[:5]
```

---

## 7) Recommended defaults (final)

Use these until you have evidence to change:

* Two concurrent jobs total
* `ed_non_opioid`: `PGX_DUCKDB_THREADS=16`, `PGX_DUCKDB_MEMORY_LIMIT=256GB`, `--concurrent-workers=1`
* `opioid_ed`: `PGX_DUCKDB_THREADS=12`, `PGX_DUCKDB_MEMORY_LIMIT=192GB`, `--concurrent-workers=1`
* Heavy age bands first: `25-44`, `65-74`
* NVMe temp spill: `/mnt/nvme/duckdb_tmp`

---

## 8) Why this works

* DuckDB uses threads internally for the big joins/aggregations.
* By keeping **one process per cohort** and processing partitions sequentially, you avoid:

  * multi-process temp directory contention
  * S3 request storms
  * Python-worker oversubscription
* Large per-job memory caps prevent repeated spills / thrash on heavy age bands.

---

## 9) Verification

After starting both jobs, verify they're running correctly:

```bash
# Check processes
ps aux | grep "0_create_cohort.py" | grep -v grep

# Check memory usage
ps aux | grep "0_create_cohort.py" | grep -v grep | awk '{print $2, $6/1024 "MB"}'

# Check logs for worker count
grep "concurrent workers" logs/*.log

# Check for errors
grep -i error logs/*.log | tail -20
```

---

## 10) Troubleshooting

### Issue: Jobs not starting

* Check Python path: `which python` or `which python3`
* Check script path: `ls -la 2_create_cohort/0_create_cohort.py`
* Check permissions: `chmod +x 2_create_cohort/0_create_cohort.py`

### Issue: Memory errors

* Verify `PGX_TOTAL_WORKERS=2` is set
* Check actual memory limits in logs: `grep "memory limit" logs/*.log`
* Reduce `PGX_DUCKDB_THREADS` if still having issues

### Issue: Too slow

* Check CPU usage: `top` or `htop`
* Verify NVMe is being used: `df -h /mnt/nvme`
* Check for S3 throttling in logs
* Consider increasing `PGX_DUCKDB_THREADS` (but stay under 28 total)

### Issue: Duplicate processing

* Check for lock files: `aws s3 ls s3://pgxdatalake/cohorts/locks/`
* Verify `check_existing_cohorts()` is working correctly
* Check logs for "already exists" messages
