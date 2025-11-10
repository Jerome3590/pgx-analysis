### PGx Pipeline Parallelization Guide

This guide explains how we configure and use parallelization across the PGx data pipeline. It covers setup, configuration, running a single step, chaining multiple steps, and running modeling (CatBoost and others) in parallel.

---

## Overview

- We parallelize using Python's `concurrent.futures.ProcessPoolExecutor` (multi-process). This means one OS process per partition/job.
- We intentionally do not use `ThreadPoolExecutor` for compute/DB steps due to the GIL and potential CPU oversubscription.
- Each worker process creates its own DuckDB connection and uses 1 DuckDB thread by default (configurable) to avoid over-subscription (many processes × many threads).
- Worker counts are environment-driven with CLI overrides for repeatable orchestration on EC2 or local.

Key env variables:

- PGX_WORKERS_MEDICAL: default medical workers (e.g., 16)
- PGX_WORKERS_PHARMACY: default pharmacy workers (e.g., 48)
- PGX_THREADS_PER_WORKER: DuckDB threads per worker (default 1)

Precedence: CLI flag overrides env; env overrides hardcoded fallback.

---

## Process pools vs threads

- **ProcessPoolExecutor (preferred):**
  - Bypasses the Python GIL; good for CPU-bound compute and DB work.
  - Each worker has an isolated DuckDB connection and memory space.
  - Scales predictably on multi-core machines and across large partition sets.

- **ThreadPoolExecutor (when to use):**
  - Use only for lightweight I/O tasks (e.g., polling SQS, small metadata checks) where GIL-bound work is minimal.
  - Not recommended for DuckDB query execution or large S3 COPY operations.

## DuckDB threads

- Controlled per worker via `PRAGMA threads=<N>`; we default to `1`.
- Set with env `PGX_THREADS_PER_WORKER` or `--threads` flags where supported.
- Rationale: With many worker processes, 1 DuckDB thread each prevents CPU oversubscription and stabilizes S3 bandwidth.

Example inside our scripts (per worker):

```sql
PRAGMA threads=1; -- or value from PGX_THREADS_PER_WORKER / --threads
```

---

## Setup

1) Python and dependencies

```bash
python --version                    # 3.10/3.11 recommended
pip install duckdb boto3 pandas     # if not already present on the machine
```

2) AWS credentials and region

- Ensure the environment where you run the pipeline has valid AWS credentials (instance profile, env vars, or ~/.aws config).
- S3 region is set in DuckDB via `SET s3_region='us-east-1'` (already handled by scripts).

3) Environment variables (recommended defaults)

```bash
export PGX_WORKERS_MEDICAL=16
export PGX_WORKERS_PHARMACY=48
export PGX_THREADS_PER_WORKER=1
```

These match the cohort pipeline’s scaling: many processes with 1 thread each for stable S3 throughput and predictable memory.

---

## Configuration (per script)

### 1) Target frequency analysis (`1_apcd_input_data/6_target_frequency_analysis.py`)

- Flags:
  - `--workers`: processes (default from `PGX_WORKERS_MEDICAL` or 1)
  - `--min-year`, `--max-year`
  - `--codes-of-interest`

Env-aware behavior:

- If `--workers` not set, script reads `PGX_WORKERS_MEDICAL`. Each worker uses 1 DuckDB thread by default or from `PGX_THREADS_PER_WORKER`.

Under the hood, the script enumerates partitions and runs analysis per-partition in a `ProcessPoolExecutor`, then reduces partial results.

### 2) Code normalization updates (`1_apcd_input_data/7_update_codes.py`)

- Flags:
  - `--workers`: global override for both datasets
  - `--workers-medical`: default from `PGX_WORKERS_MEDICAL` (fallback 16)
  - `--workers-pharmacy`: default from `PGX_WORKERS_PHARMACY` (fallback 48)
  - `--threads`: DuckDB threads per worker (default from `PGX_THREADS_PER_WORKER` or 1)
  - Filters: `--years`, `--age-bands`

Precedence:

1) CLI flag
2) Environment variable
3) Hardcoded default (medical=16, pharmacy=48, threads=1)

Under the hood, the script enumerates partitions and submits each file to a `ProcessPoolExecutor`; every worker opens its own DuckDB connection and writes updates in-place.

---

## Run a single step (example)

### Example A: Target frequency analysis (parallel across medical partitions)

```bash
export PGX_WORKERS_MEDICAL=16
export PGX_THREADS_PER_WORKER=1

python 1_apcd_input_data/6_target_frequency_analysis.py \
  --workers ${PGX_WORKERS_MEDICAL} \
  --min-year 2016 --max-year 2020
```

Outputs:

- Local CSVs in `1_apcd_input_data/` (ICD by position, ICD aggregated, CPT by field)
- S3 Parquet: `s3://pgxdatalake/gold/target_code/target_code_latest.parquet`, plus CPT aggregated Parquet

### Example B: Update codes (ICD/CPT/drug) across partitions

```bash
export PGX_WORKERS_MEDICAL=16
export PGX_WORKERS_PHARMACY=48
export PGX_THREADS_PER_WORKER=1

python 1_apcd_input_data/7_update_codes.py \
  --years "2016,2017,2018,2019,2020" \
  --workers-medical ${PGX_WORKERS_MEDICAL} \
  --workers-pharmacy ${PGX_WORKERS_PHARMACY} \
  --threads ${PGX_THREADS_PER_WORKER}
```

Notes:

- Each worker modifies its assigned S3 partition in-place; writes are idempotent with `OVERWRITE_OR_IGNORE` policies.
- Use filters (`--years`, `--age-bands`) for targeted reprocessing.

---

## Minimal ProcessPool example for ETL-style jobs

This is a generic pattern we use across steps that process partitions independently:

```python
from concurrent.futures import ProcessPoolExecutor, as_completed
import os, subprocess

def run_partition_job(filename: str) -> int:
    # Example shell-out; replace with direct function call where appropriate
    return subprocess.call([
        'python', '1_apcd_input_data/7_update_codes.py',
        '--years', '2016,2017',
        '--workers-medical', os.getenv('PGX_WORKERS_MEDICAL', '16'),
        '--workers-pharmacy', os.getenv('PGX_WORKERS_PHARMACY', '48'),
        '--threads', os.getenv('PGX_THREADS_PER_WORKER', '1'),
    ])

partitions = [
    's3://pgxdatalake/gold/medical/age_band=65-74/event_year=2019/medical_data.parquet',
    's3://pgxdatalake/gold/medical/age_band=55-64/event_year=2020/medical_data.parquet',
]

max_workers = int(os.getenv('PGX_WORKERS_MEDICAL', '16'))
with ProcessPoolExecutor(max_workers=max_workers) as ex:
    futures = [ex.submit(run_partition_job, p) for p in partitions]
    for f in as_completed(futures):
        rc = f.result()
        if rc != 0:
            print(f'Partition job failed with code {rc}')
```

Prefer direct function invocation over subprocess where possible for lower overhead; some steps expose per-partition helpers internally.

---

## Chain multiple steps

The pattern is to run high-level phases in sequence with parallelization inside each phase.

```bash
#!/usr/bin/env bash
set -euo pipefail

export PGX_WORKERS_PHARMACY=48
export PGX_WORKERS_MEDICAL=16
export PGX_THREADS_PER_WORKER=1

echo "🚀 Phase 1: Global Imputation (single process)"
python 1_apcd_input_data/2_global_imputation.py \
  --pharmacy-input s3://pgxdatalake/silver/pharmacy/**/*.parquet \
  --medical-input  s3://pgxdatalake/silver/medical/**/*.parquet \
  --output-root    s3://pgxdatalake/silver/imputed \
  --lookahead-years 5 \
  --no-demographics-lookup \
  --log-level INFO

echo "🚀 Phase 2: Pharmacy (parallel)"
python 1_apcd_input_data/3_apcd_clean.py \
  --job pharmacy \
  --pharmacy-input s3://pgxdatalake/silver/imputed/pharmacy_partitioned/**/*.parquet \
  --output-root s3://pgxdatalake/gold/pharmacy \
  --min-year 2016 --max-year 2020 \
  --workers ${PGX_WORKERS_PHARMACY} \
  --run-mode subprocess \
  --pharmacy-script 1_apcd_input_data/3a_clean_pharmacy.py \
  --log-level INFO

echo "🚀 Phase 2b: Medical (parallel)"
python 1_apcd_input_data/3_apcd_clean.py \
  --job medical \
  --medical-input s3://pgxdatalake/silver/imputed/medical_partitioned/**/*.parquet \
  --output-root s3://pgxdatalake/gold/medical \
  --min-year 2016 --max-year 2020 \
  --workers ${PGX_WORKERS_MEDICAL} \
  --run-mode subprocess \
  --medical-script 1_apcd_input_data/3b_clean_medical.py \
  --log-level INFO

echo "🚀 Target frequency analysis (parallel)"
python 1_apcd_input_data/6_target_frequency_analysis.py \
  --workers ${PGX_WORKERS_MEDICAL} \
  --min-year 2016 --max-year 2020

echo "🚀 Code normalization updates (parallel)"
python 1_apcd_input_data/7_update_codes.py \
  --years "2016,2017,2018,2019,2020" \
  --workers-medical ${PGX_WORKERS_MEDICAL} \
  --workers-pharmacy ${PGX_WORKERS_PHARMACY} \
  --threads ${PGX_THREADS_PER_WORKER}

echo "✅ Pipeline chain complete"
```

Tips:

- Use `set -euo pipefail` to fail fast and keep logs clean.
- Consider logging to files per phase (tee to `logs/`) for easier monitoring.

---

## Modeling (CatBoost and others) in parallel

We can execute modeling runs (CatBoost, FP-Growth variants, etc.) concurrently using a process pool. Keep 1 DuckDB thread if the modeling step reads from S3 via DuckDB.

### Example: Parallel CatBoost runs by cohort/partition (ProcessPoolExecutor)

```python
from concurrent.futures import ProcessPoolExecutor, as_completed
import os, subprocess

os.environ.setdefault('PGX_THREADS_PER_WORKER', '1')  # safe default

def run_catboost_job(age_band: str, event_year: int, cohort: str):
    cmd = [
        'python', '5_catboost_analysis/run_dual_catboost_pipeline.py',
        '--age-band', age_band,
        '--event-year', str(event_year),
        '--cohort', cohort,
        '--log-level', 'INFO',
    ]
    return subprocess.run(cmd, check=True)

jobs = [
    ('65-74', 2019, 'opioid_ed'),
    ('65-74', 2019, 'ed_non_opioid'),
    ('55-64', 2020, 'opioid_ed'),
]

max_workers = 6  # tune based on CPU/RAM
with ProcessPoolExecutor(max_workers=max_workers) as ex:
    futures = [ex.submit(run_catboost_job, *j) for j in jobs]
    for f in as_completed(futures):
        f.result()
print('All CatBoost jobs done.')
```

### Example: Parallel runs via shell (background jobs)

```bash
python 5_catboost_analysis/run_catboost_ade.py --log-level INFO &
python 5_catboost_analysis/run_opioid_targets.py --log-level INFO &
wait
```

Guidelines:

- Start with conservative `max_workers` (e.g., 4–8) and monitor memory.
- Prefer 1 DuckDB thread per process when reading from S3 (env: `PGX_THREADS_PER_WORKER=1`).

---

## Troubleshooting & Tips

- High CPU but low throughput: reduce DuckDB threads per worker to 1 and increase process count.
- OOM or memory pressure: decrease `--workers` and/or split work (years/age bands).
- Long S3 writes: ensure instance bandwidth and S3 permissions are adequate; logs will include COPY progress in some steps.
- Idempotency: Re-running update steps is safe; outputs for the same partition will be overwritten or ignored per COPY settings.

---

## Summary

- Multi-process parallelization with 1 DuckDB thread per worker gives the best balance for our I/O-bound S3 work.
- Use env vars to standardize scaling across environments, and CLI flags to override per run.
- Apply the same orchestration pattern for ETL, analysis, and modeling jobs.


