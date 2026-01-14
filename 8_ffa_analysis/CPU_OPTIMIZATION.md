# FFA Analysis CPU Optimization

## Summary

Optimized FFA analysis to utilize more CPU cores, increasing parallelism from 2-4 workers to 24-28 workers on a 32-core system.

## Problem Identified

**Before optimization:**
- Overall CPU usage: ~6% (only 2 cores heavily utilized)
- Default `n_jobs` limited to 4 workers max
- Causal analysis hardcoded to `n_jobs=1` (single-threaded)
- Multi-feature interaction analysis hardcoded to `n_jobs=1` (single-threaded)
- 30 out of 32 cores idle

**Impact:**
- Causal analysis was extremely slow (single-threaded bottleneck)
- Multi-feature interaction analysis was extremely slow (single-threaded bottleneck)
- Underutilization of available compute resources

## Optimizations Applied

### 1. Increased Default Parallel Workers
**Location:** `utility_scripts/run_full_ffa_analysis.py:126`

**Change:**
- **Before:** `'n_jobs': min(4, max(1, get_sklearn_n_jobs()))` (max 4 workers)
- **After:** `'n_jobs': min(28, max(1, get_sklearn_n_jobs()))` (max 28 workers)

**Rationale:**
- With 32 cores available, use 28 workers (leaves 4 cores free for system processes)
- Significantly increases throughput for parallelizable operations

### 2. Enabled Parallelization in Causal Analysis
**Location:** `utility_scripts/run_full_ffa_analysis.py:1459-1481`

**Change:**
- **Before:** Hardcoded `n_jobs=1` in fallback explanation generation
- **After:** Uses `n_jobs=ANALYSIS_CONFIG.get('n_jobs', 2)` (respects configured value)

**Impact:**
- Causal analysis now uses up to 28 workers instead of 1
- Expected speedup: ~20-25x for explanation generation steps

### 3. Enabled Parallelization in Multi-Feature Interaction Analysis
**Location:** `utility_scripts/run_full_ffa_analysis.py:1986-2058`

**Changes:**
- **Before:** Hardcoded `n_jobs=1` in:
  - Original explanations generation (line 1991)
  - Early stopping check (line 2013)
  - Modified explanations generation (line 2058)
- **After:** All use `n_jobs=ANALYSIS_CONFIG.get('n_jobs', 2)` (respects configured value)

**Impact:**
- Interaction analysis now uses up to 28 workers instead of 1
- Expected speedup: ~20-25x for explanation generation steps

### 4. Updated Logging Messages
**Location:** `utility_scripts/run_full_ffa_analysis.py:702`

**Change:**
- **Before:** "Using {n_jobs} parallel workers (limited for memory efficiency)"
- **After:** "Using {n_jobs} parallel workers (optimized for CPU utilization)"

**Rationale:**
- Reflects the new optimization focus (CPU utilization vs memory conservation)
- With 1TB RAM available, memory is not a constraint

## Expected Performance Improvements

### Causal Analysis
- **Before:** Single-threaded, ~1-2 hours for 100 features
- **After:** 28 workers, ~3-5 minutes for 100 features
- **Speedup:** ~20-25x

### Multi-Feature Interaction Analysis
- **Before:** Single-threaded, ~2-3 hours for 100 combinations
- **After:** 28 workers, ~5-10 minutes for 100 combinations
- **Speedup:** ~20-25x

### Overall FFA Analysis
- **Before:** ~4-6 hours per cohort
- **After:** ~30-60 minutes per cohort
- **Speedup:** ~4-6x overall (bottlenecks removed)

## Memory Considerations

- **Available:** 1TB RAM
- **Current Usage:** ~4.5GB
- **Risk:** Low - ProcessPoolExecutor creates separate processes, each with its own memory space
- **Mitigation:** Batch processing still in place (100 instances per batch)

## Monitoring

To verify the optimization is working:

1. **Check CPU usage:**
   ```bash
   top -p $(pgrep -f run_full_ffa_analysis)
   ```
   - Should see ~24-28 cores at high utilization during explanation generation

2. **Check process count:**
   ```bash
   ps aux | grep run_full_ffa_analysis | wc -l
   ```
   - Should see ~28-30 Python processes (main + workers)

3. **Check log output:**
   - Look for: "Using 28 parallel workers (optimized for CPU utilization)"
   - Explanation generation should complete much faster

## Configuration

The number of workers can be adjusted via `ANALYSIS_CONFIG['n_jobs']`:

```python
ANALYSIS_CONFIG = {
    'n_jobs': 28,  # Adjust based on available cores (default: min(28, get_sklearn_n_jobs()))
    # ... other config
}
```

**Recommendations:**
- **32 cores:** Use 28 workers (leaves 4 for system)
- **16 cores:** Use 12-14 workers
- **8 cores:** Use 6 workers
- **4 cores:** Use 2-3 workers

## Notes

- The grouped comparison method (`_calculate_grouped_causal_effect`) doesn't use `explain_dataset` directly, so it doesn't benefit from this optimization (it's already optimized by grouping instances)
- Early stopping checks still run in parallel, reducing wasted computation
- All explanation generation steps now benefit from parallelization
