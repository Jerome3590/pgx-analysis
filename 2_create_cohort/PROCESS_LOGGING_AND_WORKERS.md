# Process Logging and Worker Count Verification

## Overview

This document describes the logging and process management for cohort creation to help verify:
1. How many processes are actually running
2. Whether `--concurrent-workers` is being passed correctly
3. Memory allocation per process

## Logging Added

### Startup Logging (in `0_create_cohort.py`)

The pipeline now logs:
- **Process ID**: Current process PID
- **Parent Process ID**: Parent process PID (if available)
- **CPU Cores Available**: Total CPU cores on system
- **Concurrent Workers Detected**: From CLI, env vars, or default
- **Active Cohort Creation Processes**: Count and PIDs of all processes running `0_create_cohort.py`
- **Current Process Memory**: RSS memory usage

### Configuration Logging

After DuckDB connection setup:
- **Worker-specific temp directory**: PID-based unique directory
- **Total Concurrent Workers**: For memory calculation
- **NOTE**: This process is 1 of N concurrent workers
- **DuckDB memory limit**: Calculated based on workers
- **DuckDB threads**: Per-worker thread count

## Process Spawning

### Notebook/Orchestration Scripts

When running multiple cohorts in parallel via `ThreadPoolExecutor`:

1. **Set MAX_WORKERS** (e.g., `MAX_WORKERS = 3`)
2. **Pass to subprocess** via `--concurrent-workers` argument
3. **Each subprocess** calculates memory limit: `(60% of total memory) / MAX_WORKERS`

### Example: Correct Implementation

```python
MAX_WORKERS = 3  # Total concurrent workers

def run_cohort(job):
    cmd = [
        python_bin, script_path,
        "--age-band", job["age_band"],
        "--event-year", str(job["event_year"]),
        "--cohort", "both",
        "--concurrent-workers", str(MAX_WORKERS),  # CRITICAL: Pass this!
        # ... other args
    ]
    # ... launch subprocess

with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    future_to_job = {executor.submit(run_cohort, job): job for job in jobs}
```

### Updated Helper Function

The `run_cohort` function in `py_helpers/cohort_utils.py` now:
- Accepts `concurrent_workers` parameter
- Automatically detects from `MAX_WORKERS` or `PGX_COHORT_WORKERS` env vars if not provided
- Passes `--concurrent-workers` to subprocess

## Verification Checklist

When running cohort creation, check logs for:

1. **Process Count**:
   ```
   → [CONFIG] Active cohort creation processes: 3 (PIDs: [12345, 12346, 12347])
   ```

2. **Worker Detection**:
   ```
   → [CONFIG] Using --concurrent-workers=3 from CLI argument
   OR
   → [CONFIG] Detected MAX_WORKERS=3 from environment
   OR
   → [CONFIG] Using default worker count: 3
   ```

3. **Memory Allocation**:
   ```
   → [CONFIG] DuckDB memory limit: 200GB (for 3 workers, 1000GB total system memory, 600GB available for DuckDB)
   ```

4. **Process Identity**:
   ```
   → [CONFIG] Current Process ID: 12345
   → [CONFIG] Parent Process ID: 12340
   → [CONFIG] NOTE: This process is 1 of 3 concurrent workers
   ```

## Common Issues

### Issue: Too Many Processes Running

**Symptom**: More processes than expected (e.g., 6 processes when MAX_WORKERS=3)

**Cause**: 
- Multiple notebook cells running simultaneously
- Previous processes not cleaned up
- Multiple orchestrators running

**Fix**: 
- Check active processes in logs
- Kill orphaned processes: `pkill -f "0_create_cohort.py"`
- Ensure only one orchestrator is running

### Issue: Memory Limit Too High/Low

**Symptom**: OOM kills or underutilized memory

**Cause**: 
- `--concurrent-workers` not passed to subprocess
- Wrong worker count detected

**Fix**: 
- Verify logs show correct worker count
- Ensure `--concurrent-workers` is in subprocess command
- Check environment variables aren't conflicting

### Issue: Processes Not Detected

**Symptom**: "Active cohort creation processes: 0" when processes are running

**Cause**: 
- `psutil` not installed
- Process name doesn't match search pattern

**Fix**: 
- Install `psutil`: `pip install psutil`
- Check process names manually: `ps aux | grep create_cohort`

## Notebook Updates Required

The following notebook functions need to pass `--concurrent-workers`:

### `pgx_cohort_pipeline.ipynb` (legacy; moved to archived/2_create_cohort/)

**Location 1** (around line 1216):
```python
def run_cohort(job):
    cmd = [
        python_bin, script_path,
        "--age-band", job["age_band"],
        "--event-year", str(job["event_year"]),
        "--cohort", "both",
        "--starting-step", "phase1_data_preparation",
        "--operation-type", "concurrent_processing",
        "--log-level", "INFO",
        "--concurrent-workers", str(MAX_WORKERS),  # ADD THIS LINE
    ]
```

**Location 2** (around line 1536):
```python
def run_cohort(job):
    cmd = [
        python_bin, script_path,
        "--age_band", job["age_band"],
        "--event_year", str(job["event_year"]),
        "--concurrent-workers", str(MAX_WORKERS),  # ADD THIS LINE
    ]
```

## Verification Commands

### Check Active Processes
```bash
# Count processes
ps aux | grep "0_create_cohort.py" | grep -v grep | wc -l

# List PIDs
ps aux | grep "0_create_cohort.py" | grep -v grep | awk '{print $2}'

# Check memory per process
ps aux | grep "0_create_cohort.py" | grep -v grep | awk '{print $2, $6/1024 "MB"}'
```

### Check Logs for Worker Count
```bash
# Search for worker count in logs
grep "concurrent workers" logs/*.log

# Search for process counts
grep "Active cohort creation processes" logs/*.log
```
