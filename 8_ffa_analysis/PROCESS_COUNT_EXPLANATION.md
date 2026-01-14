# Why Only 16 Processes Instead of 28?

## Summary

**ProcessPoolExecutor creates workers lazily** - it doesn't spawn all `max_workers` processes at once. It creates them as needed, up to the `max_workers` limit.

## Why You're Seeing 16 Processes

### Possible Reasons:

1. **Lazy Process Creation** (Most Likely)
   - `ProcessPoolExecutor` creates workers **on-demand** as tasks are submitted
   - If you have 16 tasks being processed simultaneously, you'll see 16 worker processes
   - Additional workers are created as more tasks are submitted (up to `max_workers=28`)

2. **System Has 16 Logical Cores**
   - If your system has 16 logical cores (not 32 physical cores), ProcessPoolExecutor might be limiting itself
   - Check: `python -c "import os; print(os.cpu_count())"`

3. **Tasks Completed Quickly**
   - If tasks complete quickly, workers are reused
   - You might see fewer processes because workers finish and are reused for new tasks

4. **Batch Processing**
   - If processing in batches (e.g., 100 instances per batch), only enough workers for the current batch are active

## How ProcessPoolExecutor Works

### Lazy Worker Creation
```python
with ProcessPoolExecutor(max_workers=28) as executor:
    # Workers are NOT all created here
    # They're created lazily as tasks are submitted
    
    futures = [executor.submit(task, i) for i in range(100)]
    # Workers are created on-demand, up to 28 max
    
    # If only 16 tasks are actively running, you'll see 16 processes
    # As tasks complete, workers are reused for remaining tasks
```

### Process Lifecycle
1. **Initial**: 0 worker processes
2. **As tasks submit**: Workers created up to `max_workers` (or number of tasks, whichever is smaller)
3. **During execution**: Workers are reused as tasks complete
4. **At completion**: Workers are cleaned up

## How to Verify

### Check Actual CPU Count
```bash
# Check logical CPU count
python -c "import os; print(f'Logical CPUs: {os.cpu_count()}')"

# Check physical CPU count (Linux)
lscpu | grep "^CPU(s):"

# Check in Python
python -c "import multiprocessing; print(f'CPU count: {multiprocessing.cpu_count()}')"
```

### Check Active Worker Count
```bash
# Count processes during active execution
ps aux | grep "python.*run_full_ffa_analysis" | grep -v grep | wc -l

# See process details
ps aux | grep "python.*run_full_ffa_analysis" | grep -v grep

# Monitor process count over time
watch -n 1 'ps aux | grep "python.*run_full_ffa_analysis" | grep -v grep | wc -l'
```

### Check if Workers Are Being Created
Add logging to see when workers are created:

```python
# In base_symbolic_explainer.py, around line 1052
with ProcessPoolExecutor(max_workers=n_jobs) as executor:
    logger.info(f"ProcessPoolExecutor created with max_workers={n_jobs}")
    
    # Submit tasks
    future_to_idx = {...}
    
    # Check active worker count
    import psutil
    import os
    current_process = psutil.Process(os.getpid())
    children = current_process.children(recursive=True)
    logger.info(f"Active worker processes: {len(children)}")
```

## Expected Behavior

### Scenario 1: 16 Logical Cores
- **System**: 16 logical cores
- **max_workers**: 28
- **Observed**: ~16-17 processes (1 main + 16 workers)
- **Why**: ProcessPoolExecutor might limit to logical cores, or only 16 workers are needed

### Scenario 2: 32 Cores, Lazy Creation
- **System**: 32 cores
- **max_workers**: 28
- **Observed**: 16 processes initially, grows to 28 as more tasks are submitted
- **Why**: Workers created lazily as tasks are submitted

### Scenario 3: Batch Processing
- **Batch size**: 100 instances
- **max_workers**: 28
- **Observed**: ~16 processes per batch
- **Why**: Only enough workers for current batch are active

## What to Check

1. **Verify CPU count:**
   ```bash
   python -c "import os; print(os.cpu_count())"
   ```

2. **Monitor process count during execution:**
   ```bash
   watch -n 1 'ps aux | grep "python.*run_full_ffa_analysis" | grep -v grep | wc -l'
   ```
   - Should see process count fluctuate as workers are created/reused

3. **Check if it's actually using 28 workers:**
   - Look at CPU core utilization - should see ~28 cores busy if 28 workers are active
   - If only 16 cores busy, then only 16 workers are active

## Is This a Problem?

### ✅ **No, if:**
- Tasks are completing efficiently
- CPU utilization is high (~16 cores at 80-100%)
- Overall performance is good
- Workers are being reused effectively

### ⚠️ **Maybe, if:**
- You have 32 cores but only 16 are being used
- Performance is slower than expected
- You want to maximize parallelism

## Solutions

### If You Have 32 Cores But Only 16 Workers:

1. **Verify n_jobs is being used:**
   ```python
   # Check log output for:
   # "Using 28 parallel workers (optimized for CPU utilization)"
   ```

2. **Force more workers:**
   ```bash
   # Explicitly set n_jobs
   python utility_scripts/run_full_ffa_analysis.py \
     --cohort-name opioid_ed \
     --age-band 13-24 \
     --n-jobs 28
   ```

3. **Check for system limits:**
   ```bash
   # Check ulimit for process count
   ulimit -u
   
   # Check if there are any resource limits
   cat /etc/security/limits.conf | grep -i process
   ```

## Conclusion

**16 processes is likely normal** if:
- Your system has 16 logical cores, OR
- ProcessPoolExecutor is creating workers lazily and only 16 are needed for current tasks

**To verify:** Check CPU core utilization - if 16 cores are at high usage, then 16 workers is appropriate. If you have 32 cores and want to use more, ensure `n_jobs=28` is being applied correctly.
