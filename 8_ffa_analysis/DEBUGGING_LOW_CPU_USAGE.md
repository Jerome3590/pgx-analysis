# Debugging Low CPU Usage (Only 2 Cores Active)

## Problem

You're seeing:
- Only 2 CPU cores at 100% (CPUs 30 and 31)
- Only 2 Python processes
- Overall CPU usage: ~6%
- This suggests parallel workers aren't being used

## Possible Causes

### 1. **Currently in Single-Threaded Phase** (Most Likely)

The FFA analysis has several phases, and only some use parallel workers:

**Single-threaded phases:**
- Step 1: Loading model JSON
- Step 2: Extracting feature mappings  
- Step 3: Loading data
- Step 4: Extracting rules from model
- Step 6: Calculating feature importance (aggregation)

**Parallel phases (should use 28 workers):**
- Step 5: Generating AXP explanations (`explain_dataset`)
- Step 7: Causal analysis (`explain_dataset` calls)
- Step 8: Multi-feature interaction analysis (`explain_dataset` calls)

**Check:** Look at your log file to see which step is currently running:
```bash
tail -f 8_ffa_analysis/logs/ffa_analysis_*.log | grep -E "(Step|Phase|explain_dataset|Using.*parallel workers)"
```

### 2. **Workers Not Being Created**

ProcessPoolExecutor creates workers lazily. If tasks complete quickly or there are few tasks, you might not see all workers.

**Check:** Look for this log message:
```
Using 28 parallel workers (optimized for CPU utilization)
```

If you see a different number (like 16 or 2), then `n_jobs` isn't being set correctly.

### 3. **Sequential Execution Instead of Parallel**

If `explain_dataset` is being called with `n_jobs=1` or falling back to sequential mode.

**Check:** Look for these log messages:
```
Using parallel processing with X workers for Y instances
```
OR
```
_explain_dataset_sequential
```

### 4. **Batch Processing**

If processing in small batches (e.g., 16 instances per batch), you might only see 16 workers.

**Check:** Look for batch processing logs:
```
Processing batch 1/N (instances 0-100)...
```

## How to Diagnose

### 1. Check Current Phase
```bash
# Find the most recent log file
LATEST_LOG=$(ls -t 8_ffa_analysis/logs/ffa_analysis_*.log | head -1)

# See what's currently happening
tail -20 $LATEST_LOG | grep -E "(Step|Phase|explain_dataset|Using.*workers)"
```

### 2. Check Worker Configuration
```bash
# Check if n_jobs is being set correctly
grep -E "(n_jobs|parallel workers)" $LATEST_LOG | tail -10
```

### 3. Check Process Count During Active Phase
```bash
# Monitor process count (should see ~29 during parallel phases)
watch -n 1 'ps aux | grep "python.*run_full_ffa_analysis" | grep -v grep | wc -l'
```

### 4. Check CPU Usage During Active Phase
```bash
# Monitor CPU usage (should see ~28 cores busy during parallel phases)
top -p $(pgrep -f run_full_ffa_analysis | head -1)
# Press '1' to see per-core view
```

## Expected Behavior by Phase

| Phase | Expected Processes | Expected CPU Cores | Duration |
|-------|-------------------|-------------------|----------|
| Steps 1-4 (Load/Extract) | 1-2 | 1-2 cores | Minutes |
| Step 5 (AXP Explanations) | ~29 | ~28 cores | 10-30 min |
| Step 6 (Feature Importance) | 1-2 | 1-2 cores | Minutes |
| Step 7 (Causal Analysis) | ~29 | ~28 cores | 5-15 min |
| Step 8 (Interactions) | ~29 | ~28 cores | 10-30 min |

## What to Do

### If You're in Steps 1-4 or Step 6:
**This is normal** - these phases are single-threaded. Wait for Step 5, 7, or 8 to see parallel workers.

### If You're in Step 5, 7, or 8 but Only See 2 Cores:
1. **Check the log** for "Using X parallel workers" - what number does it show?
2. **Verify n_jobs is set correctly:**
   ```bash
   grep "n_jobs" $LATEST_LOG | tail -5
   ```
3. **Check if explain_dataset is being called:**
   ```bash
   grep "explain_dataset\|_explain_dataset" $LATEST_LOG | tail -10
   ```

### If n_jobs Shows 2 Instead of 28:
1. **Check if --n-jobs was passed:**
   ```bash
   ps aux | grep run_full_ffa_analysis | grep -o -- "--n-jobs [0-9]*"
   ```
2. **Check if get_sklearn_n_jobs() is returning a low value:**
   ```python
   python -c "from py_helpers.env_utils import get_sklearn_n_jobs; print(get_sklearn_n_jobs())"
   ```

## Quick Fix

If you want to force 28 workers, explicitly set it:
```bash
python utility_scripts/run_full_ffa_analysis.py \
  --cohort-name opioid_ed \
  --age-band 13-24 \
  --n-jobs 28
```

## Summary

**Most likely:** You're in a single-threaded phase (Steps 1-4 or 6). Wait for Step 5, 7, or 8 to see parallel workers kick in.

**If you're in a parallel phase but only see 2 cores:** Check the log to see what `n_jobs` value is being used and verify `explain_dataset` is being called with parallel execution.
