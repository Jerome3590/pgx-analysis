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

### 2) Code normalization updates (`1_apcd_input_data/8_update_codes.py`)

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

python 1_apcd_input_data/7_target_frequency_analysis.py \
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

python 1_apcd_input_data/8_update_codes.py \
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
python 1_apcd_input_data/7_target_frequency_analysis.py \
  --workers ${PGX_WORKERS_MEDICAL} \
  --min-year 2016 --max-year 2020

echo "🚀 Code normalization updates (parallel)"
python 1_apcd_input_data/8_update_codes.py \
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

## Feature Importance Analysis Parallelization

The Feature Importance Analysis pipeline uses `joblib.Parallel` with process-based parallelism for Monte Carlo Cross-Validation (MC-CV) splits. This section covers the optimized configuration for EC2 instances.

### Configuration (32-core, 1TB RAM EC2)

**Worker Configuration:**
```python
import multiprocessing
# Optimized for EC2: 32 cores, 1TB RAM
# Use 28 workers (leave 4 cores for system/OS overhead)
N_WORKERS = max(1, multiprocessing.cpu_count() - 4)
```

**XGBoost Model Parameters:**
```python
MODEL_PARAMS = {
    'xgboost': {
        'max_depth': 6,
        'learning_rate': 0.1,
        'n_estimators': 250,
        'subsample': 1.0,
        'colsample_bytree': 1.0,
        'random_seed': 42,
        'n_jobs': 2,  # Use 2 threads per model (28 workers × 2 = 56 threads on 32 cores)
        'tree_method': 'hist',  # Faster than exact, more accurate than approx
        'early_stopping_rounds': 10,  # Enable early stopping for faster training
    },
    'xgboost_rf': {
        'max_depth': 6,
        'learning_rate': 0.1,
        'n_estimators': 250,
        'subsample': 0.8,
        'max_features': None,
        'random_seed': 42,
        'n_jobs': 2,  # Use 2 threads per model
        'tree_method': 'hist',
        'early_stopping_rounds': 10,
    },
}
```

**Note**: For smaller systems (<16 cores), use `n_jobs=1` and `N_WORKERS = cpu_count() - 4` to avoid oversubscription.

### Joblib Configuration

The pipeline uses `joblib.Parallel` with the following optimizations:

- **Backend**: `loky` (process-based) on Linux/Mac, `threading` on Windows
- **Temp folder**: `/tmp` instead of `/dev/shm` for better persistence on long-running jobs
- **Memmapping**: Disabled (`max_nbytes=None`) to avoid shared memory issues
- **Fallback**: Automatic retry with reduced parallelism if initial execution fails

### Worker Process Logging

The Feature Importance pipeline implements **multiprocessing-safe logging** to ensure logs from parallel worker processes are visible and correctly attributed. This is critical for debugging and monitoring long-running parallel jobs.

#### Architecture

**Main Process Setup** (`py_helpers/logging_utils.py`):
- Creates a unique logger with timestamp and process ID to prevent collisions
- Configures three handlers:
  1. **Memory buffer**: Captures logs for later S3 upload
  2. **Console handler** (`AutoFlushHandler`): Writes to stdout with immediate flush
  3. **File handler**: Writes to timestamped log file in `logs/` directory
- Log file naming: `feature_importance_{cohort_name}_{age_band}_{year}_{timestamp}.log`

**Worker Process Setup** (`py_helpers/mc_cv_utils.py`):
- Each worker process calls `_setup_worker_logging()` when initialized
- Worker processes write to the **same log file** as the main process (append mode)
- Each log message is prefixed with `[Worker-ProcessName-PID]` for identification
- Example: `[Worker-LokyProcess-27-268558] 2025-12-22 18:29:30,840 - INFO - ...`

#### Key Features

1. **Worker Identification**: 
   - All log messages include `[Worker-ProcessName-PID]` prefix
   - Process name identifies the backend (e.g., `LokyProcess`, `ForkPoolWorker`)
   - PID allows tracking individual worker processes

2. **Real-Time Visibility**:
   - Logs from all workers appear immediately in stdout (via `AutoFlushHandler`)
   - File handler uses append mode so multiple processes can write safely
   - Python's logging module handles file locking automatically

3. **Progress Tracking**:
   - Each MC-CV split logs:
     - Training start/completion with timing
     - Prediction start/completion with timing
     - Permutation importance progress (every 10% of features)
   - Example log lines:
     ```
     [Worker-LokyProcess-27-268558] [MC-CV] Split 5 (xgboost): training completed in 576.3s
     [Worker-LokyProcess-27-268558] [MC-CV] Split 5 (xgboost): prediction completed in 108.2s
     [Worker-LokyProcess-27-268558] Permutation importance: baseline score=0.042741 on 766298 rows × 1519 features
     ```

4. **Log File Path Passing**:
   - Main process creates log file and passes `log_file_path` to `run_mc_cv_method()`
   - `log_file_path` is passed to each worker via `delayed(run_single_split)(..., log_file_path)`
   - Workers use this path to configure their file handlers

#### Implementation Details

**AutoFlushHandler** (`py_helpers/logging_utils.py`):
```python
class AutoFlushHandler(logging.StreamHandler):
    def emit(self, record):
        super().emit(record)
        self.flush()  # Force flush after each log message
```

**Worker Logging Setup** (`py_helpers/mc_cv_utils.py`):
```python
def _setup_worker_logging(log_file_path: Optional[str] = None):
    worker_id = os.getpid()
    process_name = multiprocessing.current_process().name
    
    # Configure root logger with worker identification
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # Console handler with worker prefix
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter(
        f'[Worker-{process_name}-{worker_id}] %(asctime)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    # File handler (append mode for multi-process writes)
    if log_file_path:
        file_handler = logging.FileHandler(log_file_path, mode='a')
        file_handler.setFormatter(console_formatter)
        root_logger.addHandler(file_handler)
```

#### Benefits

- **Debugging**: Easy to identify which worker process generated each log message
- **Monitoring**: Real-time visibility into parallel progress without waiting for completion
- **Performance Analysis**: Timing information from each worker helps identify bottlenecks
- **Error Tracking**: Worker-specific error messages help isolate issues in parallel execution

#### Example Log Output

```
2025-12-22 18:22:06,749 - INFO - --- Running MC-CV for xgboost (25 splits) ---
2025-12-22 18:22:06,750 - INFO - Using 28 parallel workers for MC-CV splits
[Worker-LokyProcess-18-268549] 2025-12-22 18:22:10,123 - INFO - [MC-CV] Split 0 (xgboost): training model on 3065192 samples × 1519 features
[Worker-LokyProcess-27-268558] 2025-12-22 18:22:10,456 - INFO - [MC-CV] Split 1 (xgboost): training model on 3065192 samples × 1519 features
[Worker-LokyProcess-18-268549] 2025-12-22 18:31:46,789 - INFO - [MC-CV] Split 0 (xgboost): training completed in 576.3s
[Worker-LokyProcess-18-268549] 2025-12-22 18:31:46,890 - INFO - [MC-CV] Split 0 (xgboost): predicting on full holdout (766298 samples)
[Worker-LokyProcess-18-268549] 2025-12-22 18:33:35,012 - INFO - [MC-CV] Split 0 (xgboost): prediction completed in 108.2s
[Worker-LokyProcess-18-268549] 2025-12-22 18:33:35,123 - INFO - [MC-CV] Split 0 (xgboost): starting permutation importance on 766298 rows × 1519 features
[Worker-LokyProcess-18-268549] 2025-12-22 18:29:30,840 - INFO - Permutation importance: baseline score=0.042741 on 766298 rows × 1519 features
```

#### Troubleshooting

**Issue**: Logs from workers not appearing in log file
- **Solution**: Ensure `log_file_path` is passed to `run_mc_cv_method()` and workers have write permissions

**Issue**: Log messages appear without worker identification
- **Solution**: Verify `_setup_worker_logging()` is called in worker processes (should happen automatically via `_log_mc()`)

**Issue**: Log file grows very large
- **Solution**: This is expected for long-running jobs. Consider rotating logs or using `PGX_PERM_MAX_ROWS` to reduce permutation importance logging

### Environment Variables

```bash
# Override joblib temp folder (default: /tmp on Linux, system temp on Windows)
export JOBLIB_TEMP_FOLDER=/tmp

# Limit rows used for permutation importance (reduces memory/time)
export PGX_PERM_MAX_ROWS=50000  # Optional: limits evaluation set size

# Ensure single-threaded operations in workers
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
```

### Performance Characteristics

**Actual Runtime (cohort 2, age 65-74, 32-core EC2, 1TB RAM):**
- Data loading & feature engineering: ~17 minutes
- XGBoost MC-CV training + prediction (25 splits): ~33 minutes
- Permutation importance (25 splits, parallel): ~6-7 hours
- **Total: ~6.5-7.5 hours** (vs. ~4 days before optimizations)

**Performance Improvements:**
- **Training time**: Average 9.6 min/split (range: 6.3-11.6 min) - **~2x faster** than before
- **Early stopping**: Working effectively (1.2x variation indicates models stopping early)
- **Worker count**: 28 workers providing excellent parallelization
- **Threading**: 2 threads per XGBoost model (balanced for 32-core system)
- **Tree method**: `hist` method for better performance

### Timing Analysis

Based on actual log analysis from `non_opioid_ed_65_74_2019` run:

**Training Times (25 splits):**
- Average: 9.6 minutes per split
- Range: 6.3 - 11.6 minutes
- Early stopping working: ~1.2x variation (models converge at different rates)
- **Improvement**: ~2x faster than pre-optimization runs

**Prediction Times (25 splits):**
- Average: 1.8 minutes per split
- Range: 1.1 - 2.2 minutes
- Consistent across splits

**Permutation Importance (Current Bottleneck):**
- Dataset: 766,298 rows × 1,519 features
- Repeats: 3 (default, optimized from 5 for 40% speedup)
- Total predictions per split: 1,519 features × 3 repeats = 4,557 predictions
- Estimated time per split: ~3.5-4 hours (sequential within split, with optimizations)
- With 28 parallel workers: ~3.5-4 hours wall time for all 25 splits

**Overall Timeline:**
1. **Start → Training Complete**: ~33 minutes (all 25 splits in parallel)
2. **Training → Prediction Complete**: ~33 minutes (overlapped with training)
3. **Prediction → Permutation Complete**: ~6-7 hours (bottleneck phase)
4. **Total**: ~6.5-7.5 hours from start to finish

**Comparison to Previous Runs:**
- **Before optimizations**: ~4 days (96 hours) for `non_opioid_ed 65-74` on EC2
- **After optimizations**: 6.5-7.5 hours actual
- **Improvement**: ~13-14x faster overall (from ~4 days to ~7 hours)

**Key Optimizations Applied:**
- **Early stopping**: 2-5x faster training (models stop when converged)
- **Worker count**: 28 workers (40% more parallelism than previous 20)
- **Threading**: 2 threads per XGBoost model (balanced for 32-core system)
- **Tree method**: `hist` method for better performance
- **Joblib configuration**: Persistent temp folder, disabled memmapping

### GPU Configuration (Optional)

For GPU-enabled instances (e.g., `run_cohort_1_25_44.py`):

```python
'xgboost': {
    ...
    'tree_method': 'gpu_hist',
    'predictor': 'gpu_predictor',
    'device': 'cuda',
    'n_jobs': 1,  # GPU mode: use 1 thread (GPU handles parallelism)
    'early_stopping_rounds': 10,
},
```

### Monitoring

- Check logs regularly for progress updates (every 10% of features, every split completion)
- Monitor memory usage (logs show memory at key checkpoints)
- Watch for early stopping messages (indicates models converging faster)
- Look for worker identification in logs: `[Worker-ForkPoolWorker-1-12345]`

### Troubleshooting

**Joblib Pickling Errors:**
- Ensure `JOBLIB_TEMP_FOLDER` is set to a persistent location (`/tmp` on Linux)
- Check that `/tmp` has sufficient space
- The pipeline automatically falls back to threading backend if process-based fails

**High Memory Usage:**
- Set `PGX_PERM_MAX_ROWS` to limit permutation importance evaluation set size
- Reduce `N_WORKERS` if memory pressure occurs
- Monitor logs for memory checkpoints

**Slow Performance:**
- Verify early stopping is working (check logs for "early stopping" messages)
- Ensure `tree_method='hist'` is set
- Check that worker count matches system capacity (28 for 32-core systems)

---

## SHAP Analysis Parallelization

SHAP (SHapley Additive exPlanations) analysis computes patient-level feature importance. Configuration depends on the SHAP explainer type and dataset size.

### Configuration

**For TreeExplainer (XGBoost, CatBoost):**
```python
# SHAP computation is typically single-threaded per model
# Parallelization happens at the cohort/age-band level
import multiprocessing

# Run multiple cohorts/age-bands in parallel
max_workers = min(4, multiprocessing.cpu_count() // 4)  # Conservative for memory
```

**For KernelExplainer (slower, more general):**
```python
# Use background sampling for faster computation
explainer = shap.KernelExplainer(model.predict, X_train_sample)
shap_values = explainer.shap_values(X_test_sample, nsamples=100)  # Limit samples
```

### Environment Variables

```bash
# Limit SHAP computation samples (for KernelExplainer)
export SHAP_NSAMPLES=100  # Default: 100

# Limit test set size for SHAP computation
export SHAP_MAX_ROWS=1000  # Optional: sample test set for faster computation
```

### Performance Notes

- **TreeExplainer**: Fast for tree-based models, typically single-threaded per model
- **KernelExplainer**: Slower, benefits from sampling (`nsamples` parameter)
- **Parallelization**: Run multiple cohorts/age-bands in parallel using `ProcessPoolExecutor`
- **Memory**: SHAP values can be large (n_samples × n_features), consider sampling

---

## FFA Analysis Parallelization

FFA (Feature Attribution Analysis) uses symbolic explainers for rule-based patient explanations. Configuration focuses on memory management and batch processing.

### Configuration

```python
ANALYSIS_CONFIG = {
    'target_class': 1,
    'top_k_features': 20,
    'min_coverage': 0.8,
    'n_permutations': 100,  # Reduced for faster execution
    'random_seed': 1997,
    'max_samples': 10000,  # Limit data samples to prevent OOM
    'max_explanation_samples': 1000,  # Limit instances for explanation generation
    'n_jobs': 2,  # Limit parallel workers to reduce memory usage
    'batch_size': 100,  # Process explanations in batches
}
```

### Worker Configuration

**For 32-core, 1TB RAM EC2:**
```python
import multiprocessing
# FFA analysis is memory-intensive, use conservative parallelism
N_WORKERS = max(1, multiprocessing.cpu_count() // 8)  # ~4 workers for 32-core system
```

**For smaller systems:**
```python
N_WORKERS = 1-2  # Single or dual worker to avoid memory pressure
```

### Environment Variables

```bash
# Limit explanation samples (reduces memory and computation time)
export FFA_MAX_SAMPLES=10000
export FFA_MAX_EXPLANATION_SAMPLES=1000

# Control parallel workers
export FFA_N_JOBS=2  # Default: 2 workers
```

### Performance Notes

- **Memory-intensive**: Explanation generation requires loading full models and data
- **Batch processing**: Process explanations in batches to manage memory
- **Sampling**: Use `max_samples` and `max_explanation_samples` to limit computation
- **Parallelization**: Limited by memory, not CPU (typically 1-4 workers)

---

## SHAP + FFA Combination Analysis

The combination script (`10_results/combine_shap_ffa_results.py`) aggregates results from both methods. This is a lightweight post-processing step that doesn't require parallelization.

### Configuration

```bash
# Basic usage (single-threaded, fast)
python 10_results/combine_shap_ffa_results.py \
    --cohort non_opioid_ed \
    --age-band 65-74 \
    --output-dir 10_results/outputs
```

### Parallelization Strategy

For multiple cohorts/age-bands, run in parallel:

```python
from concurrent.futures import ProcessPoolExecutor

cohorts = [
    ('non_opioid_ed', '65-74'),
    ('non_opioid_ed', '75-84'),
    ('opioid_ed', '65-74'),
]

max_workers = min(4, len(cohorts))  # Conservative for I/O
with ProcessPoolExecutor(max_workers=max_workers) as ex:
    futures = [
        ex.submit(run_combination, cohort, age_band)
        for cohort, age_band in cohorts
    ]
    for f in as_completed(futures):
        f.result()
```

### Performance Notes

- **I/O bound**: Primarily reads CSV/JSON files and writes results
- **Fast**: Typically completes in seconds to minutes per cohort
- **No special configuration needed**: Default single-threaded execution is sufficient

---

## Row-Level Analysis Configuration

Row-level analysis combines multiple approaches (SHAP, FFA, FPGrowth) for patient-specific insights. Each component has its own parallelization strategy.

### Complete Workflow

```bash
# Step 1: Feature Importance (with sampling for speed)
export PGX_PERM_MAX_ROWS=50000  # Use 50K rows for permutation importance
python 3a_feature_importance/run_cohort_2_65_74.py

# Step 2: FPGrowth Pattern Mining (parallel)
python 10_risk_dashboard/visualizations/fpgrowth/run_single_cohort_fpgrowth.py \
    --cohort-name non_opioid_ed \
    --age-band 65-74

# Step 3: SHAP Analysis (if available)
python 8_final_model/add_shap_analysis.py \
    --cohort non_opioid_ed \
    --age-band 65-74

# Step 4: FFA Analysis (memory-conservative)
export FFA_MAX_SAMPLES=10000
python 8_ffa_analysis/run_full_ffa_analysis.py

# Step 5: Combine Results (single-threaded, fast)
python 10_results/combine_shap_ffa_results.py \
    --cohort non_opioid_ed \
    --age-band 65-74
```

### Environment Variables Summary

```bash
# Feature Importance
export PGX_PERM_MAX_ROWS=50000  # Limit permutation importance evaluation

# SHAP Analysis
export SHAP_NSAMPLES=100  # Limit SHAP samples
export SHAP_MAX_ROWS=1000  # Limit test set size

# FFA Analysis
export FFA_MAX_SAMPLES=10000  # Limit data samples
export FFA_MAX_EXPLANATION_SAMPLES=1000  # Limit explanation instances
export FFA_N_JOBS=2  # Parallel workers

# Joblib (shared across all)
export JOBLIB_TEMP_FOLDER=/tmp  # Persistent temp folder
export OMP_NUM_THREADS=1  # Single-threaded operations
```

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

---

# FP-Growth Specific Parallelization

This section provides comprehensive guidance for executing FP-Growth pipelines using multiprocessing with shared AWS connections to prevent resource exhaustion and improve performance.

## Key Improvements

### 1. Shared Connection Pool
- **Problem**: Multiple AWS connections causing resource exhaustion
- **Solution**: Centralized connection pool in `helpers/aws_utils.py`
- **Benefit**: Reduced memory usage and connection overhead

### 2. Signal 15 Debugging
- **Problem**: Unexplained process terminations (Signal 15)
- **Solution**: Enhanced signal handling with detailed resource monitoring
- **Benefit**: Better debugging and prevention of crashes

### 3. SQS-Based Job Control
- **Problem**: CPU pressure building up as more jobs start simultaneously
- **Solution**: SQS FIFO queue with controlled concurrency
- **Benefit**: Jobs complete before new ones start, preventing CPU overload

## Usage Examples

### Standard Multiprocessing (Recommended for most cases)
```python
from fpgrowth_analysis.run_fpgrowth import execute_parallel_global_fpgrowth

# Optimal execution based on CPU cores
result = execute_parallel_global_fpgrowth(num_workers=8)

# Conservative execution for stability
result = execute_parallel_global_fpgrowth(num_workers=4)

# Single worker for debugging
result = execute_parallel_global_fpgrowth(num_workers=1)
```

### SQS-Based Execution (For CPU control and job management)
```python
from fpgrowth_analysis.run_fpgrowth_sqs import execute_fpgrowth_with_sqs

# Full execution with controlled concurrency
result = execute_fpgrowth_with_sqs(max_concurrent=4)

# Conservative execution for maximum stability
result = execute_fpgrowth_with_sqs(max_concurrent=2)

# Single job execution for debugging
result = execute_fpgrowth_with_sqs(max_concurrent=1)
```



### Command Line Usage

#### Standard Multiprocessing
```bash
# Run with optimal workers
python -c "from fpgrowth_analysis.run_fpgrowth import execute_parallel_global_fpgrowth; execute_parallel_global_fpgrowth(num_workers=8)"

# Run with conservative workers
python -c "from fpgrowth_analysis.run_fpgrowth import execute_parallel_global_fpgrowth; execute_parallel_global_fpgrowth(num_workers=4)"
```

#### SQS-Based Execution
```bash
# Full execution (enqueue + process)
python fpgrowth_analysis/run_fpgrowth_sqs.py --max-concurrent 4

# Enqueue jobs only
python fpgrowth_analysis/run_fpgrowth_sqs.py --mode enqueue

# Process jobs only
python fpgrowth_analysis/run_fpgrowth_sqs.py --mode process --max-concurrent 4

# Conservative execution
python fpgrowth_analysis/run_fpgrowth_sqs.py --max-concurrent 2
```

## Configuration Recommendations

### For High CPU Systems (>16 cores)
- **Standard**: `num_workers=8-12`
- **SQS**: `max_concurrent=4-6`

### For Medium CPU Systems (8-16 cores)
- **Standard**: `num_workers=4-8`
- **SQS**: `max_concurrent=2-4`

### For Low CPU Systems (<8 cores)
- **Standard**: `num_workers=2-4`
- **SQS**: `max_concurrent=1-2`

### For Debugging/Testing
- **Standard**: `num_workers=1`
- **SQS**: `max_concurrent=1`

## Troubleshooting

### Signal 15 Issues
If you encounter Signal 15 terminations:
1. Reduce worker count: `num_workers=4` or `max_concurrent=2`
2. Use SQS-based execution for better control
3. Check system resources with signal debugging enabled

### High CPU Usage
If CPU usage exceeds 90%:
1. Switch to SQS-based execution
2. Reduce concurrent jobs to 2 or 1
3. Monitor with `get_system_resource_status()`

### Memory Issues
If memory usage is high:
1. Use shared connection pool (already implemented)
2. Reduce worker count
3. Enable signal debugging for monitoring

## Performance Comparison

| Method | CPU Control | Job Ordering | Fault Tolerance | Complexity |
|--------|-------------|--------------|-----------------|------------|
| Standard Multiprocessing | Medium | Parallel | Good | Low |
| SQS-Based | High | FIFO | Excellent | Medium |

## Best Practices

1. **Start Conservative**: Begin with fewer workers/concurrent jobs
2. **Monitor Resources**: Use signal debugging for system monitoring
3. **Use SQS for CPU Control**: When CPU usage is consistently high
4. **Test First**: Always test with small worker counts before scaling up
5. **Separate Concerns**: Use SQS enqueue/process modes for complex workflows

## File Structure

```
fpgrowth_analysis/
├── run_fpgrowth.py              # Standard multiprocessing
├── run_fpgrowth_sqs.py          # SQS-based execution
└── MULTIPROCESSING_README.md    # This documentation

helpers/
├── aws_utils.py                 # Shared connection pool & signal debugging
├── sqs_utils.py                 # SQS queue management
└── debug_signal_15.py          # Signal debugging tool
```

## Migration Guide

### From Standard to SQS
If you're experiencing CPU pressure with standard multiprocessing:

```python
# Before (Standard)
from fpgrowth_analysis.run_fpgrowth import execute_parallel_global_fpgrowth
result = execute_parallel_global_fpgrowth(num_workers=8)

# After (SQS)
from fpgrowth_analysis.run_fpgrowth_sqs import execute_fpgrowth_with_sqs
result = execute_fpgrowth_with_sqs(max_concurrent=4)
```

### From High to Conservative Workers
If you're experiencing Signal 15 issues:

```python
# Before (High workers)
result = execute_parallel_global_fpgrowth(num_workers=32)

# After (Conservative)
result = execute_parallel_global_fpgrowth(num_workers=4)
# or
result = execute_fpgrowth_with_sqs(max_concurrent=2)
```

## Jupyter Notebook Execution Warning
### Implementation Example: Group-Based Parallelism

The file `run_fpgrowth_group_pipeline.py` demonstrates the recommended approach for parallelism in this project:

- Uses `ProcessPoolExecutor` (from Python's `concurrent.futures`) to launch each group in a separate process.
- Each process loads its own data subset, runs FP-Growth, and writes results independently.
- This approach is robust for both interactive (Jupyter with Bash cells) and production (Bash/CLI) workflows.
- The implementation avoids thread pools, ensuring true parallelism for CPU-bound workloads and compatibility with AWS resource management.

**Key code pattern:**

```python
from concurrent.futures import ProcessPoolExecutor, as_completed

with ProcessPoolExecutor(max_workers=32) as executor:
    future_to_group = {
        executor.submit(run_group_fpgrowth, group): group for group in group_definitions
    }
    for future in as_completed(future_to_group):
        group = future_to_group[future]
        ... # handle results
```

This pattern is used for all large-scale, parallel FP-Growth jobs in the project.

> **Parallelism Note:**
> The FP-Growth pipeline uses **Process Pools** (not Thread Pools) for parallel execution. This is essential for CPU-bound workloads and ensures that each worker runs in a separate process, avoiding Python's Global Interpreter Lock (GIL) and enabling true parallelism. When running from Jupyter notebooks (using Bash cells) or from the command line, Process Pools provide robust, scalable parallelism. Thread Pools are not suitable for this use case and may lead to poor performance or deadlocks.

**Lesson Learned:**

Running the FP-Growth multiprocessing pipeline directly from a Jupyter notebook using native Python (e.g., `%run`, `!python`, or notebook cells) led to frequent kernel crashes and unpredictable failures. This is due to the way Jupyter manages processes and interacts with Python's multiprocessing and system resources, especially when using shared AWS connections or spawning multiple workers.


**Resolution:**
- Always run the pipeline using a standalone Bash script, from the command line (e.g., `python run_fpgrowth.py ...`), **or** from a Jupyter notebook using a Bash cell (e.g., `!python run_fpgrowth.py ...`).
- Avoid launching multiprocessing jobs from within Jupyter notebooks using native Python code cells.

> **Note:**
> You can still use Jupyter notebooks to launch the pipeline, but you must use a Bash cell (e.g., `!python run_fpgrowth.py ...`) rather than native Python code cells. This ensures the pipeline runs in a separate process, avoiding the multiprocessing and resource issues described above.

**Why:**
- Jupyter's process model is not compatible with Python's multiprocessing for complex, resource-intensive jobs.
- Kernel restarts, memory leaks, and zombie processes are common when running heavy parallel workloads from notebooks.

**Best Practice:**
- Jupyter notebooks can be used for prototyping, visualization, lightweight analysis, and even large-scale runs.
- For any large-scale or production run from a notebook, use a Bash cell (e.g., `!python run_fpgrowth.py ...`) rather than a native Python code cell.
- Bash cell output (including logs) can be displayed directly in the notebook and also written to local log files for review and observability.
- For production or large-scale runs outside of notebooks, Bash scripts or direct command line execution are also recommended for stability and reliability.