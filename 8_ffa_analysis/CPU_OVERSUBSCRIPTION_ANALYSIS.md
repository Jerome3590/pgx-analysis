# CPU Oversubscription Analysis: 56 Workers on 32 Cores

## Scenario

Running two FFA analysis instances simultaneously, each with 28 workers:
- **Instance 1**: 28 workers (cohort 1)
- **Instance 2**: 28 workers (cohort 2)
- **Total**: 56 workers competing for 32 CPU cores
- **Oversubscription ratio**: 56/32 = 1.75x

## What Happens

### 1. **Context Switching Overhead**

**The Problem:**
- Operating system scheduler must time-slice between 56 processes
- Only 32 can run simultaneously (one per core)
- Remaining 24 processes wait in ready queue
- Frequent context switches between processes

**Impact:**
- **CPU overhead**: ~5-15% of CPU time spent on context switching
- **Cache thrashing**: Process switches cause CPU cache misses
- **Memory bandwidth**: More contention for memory access

### 2. **Performance Degradation**

**Expected behavior:**

| Metric | Single Instance (28 workers) | Two Instances (28 each) | Degradation |
|--------|------------------------------|-------------------------|-------------|
| CPU utilization | ~87% (28/32 cores) | ~95-100% (all cores busy) | Higher utilization |
| Context switches/sec | ~10,000-20,000 | ~30,000-50,000 | 2-3x increase |
| Effective throughput | 100% baseline | ~70-85% per instance | 15-30% slower per instance |
| Total throughput | 100% | ~140-170% | Still faster than sequential |

### 3. **Timing Estimates**

**Single cohort (28 workers):**
- Causal analysis: ~3-5 minutes
- Interaction analysis: ~5-10 minutes
- Total: ~30-60 minutes

**Two cohorts simultaneously (28 workers each):**
- Causal analysis: ~4-7 minutes per cohort (15-30% slower)
- Interaction analysis: ~7-14 minutes per cohort (15-30% slower)
- Total: ~45-90 minutes per cohort (but both complete in ~45-90 minutes total)

**Sequential (one at a time):**
- Total: ~60-120 minutes (30-60 minutes × 2)

**Conclusion**: Even with oversubscription, parallel execution is **faster overall** (~45-90 min total vs ~60-120 min sequential), but each individual cohort takes longer.

### 4. **System Behavior**

**What you'll see:**

```bash
# CPU usage will show:
- All 32 cores at 90-100% utilization
- High "steal" time (if in VM) or "wait" time
- Context switches: 30,000-50,000/sec (vs 10,000-20,000 for single instance)
- Load average: 50-60 (vs 25-30 for single instance)
```

**Process states:**
- ~32 processes in "R" (Running) state
- ~24 processes in "S" (Sleeping/Ready) state, waiting for CPU
- Frequent transitions between states

### 5. **Memory Impact**

**Per-process memory:**
- Main process: ~100-200 MB
- Worker processes: ~50-100 MB each
- **Total per instance**: ~1.5-3 GB
- **Two instances**: ~3-6 GB total

**With 1TB RAM available:**
- **Risk**: Very low
- **Impact**: Negligible

### 6. **Disk I/O Impact**

**Separate output directories:**
- Each cohort writes to different paths
- Minimal contention
- **Impact**: Low

**S3 downloads/uploads:**
- Both instances may download SHAP values simultaneously
- S3 handles concurrent requests well
- **Impact**: Low

## Is It Still Worth It?

### ✅ **Yes, if:**
- You want to complete both cohorts as fast as possible
- You're okay with each cohort taking 15-30% longer individually
- You have sufficient memory (you do: 1TB available)
- You're not running other critical workloads

### ❌ **No, if:**
- You need optimal performance per cohort
- You're running other CPU-intensive workloads
- You want to minimize system load
- You prefer predictable, consistent timing

## Comparison Table

| Configuration | Workers per Instance | Total Workers | Per-Cohort Time | Total Time | CPU Efficiency |
|--------------|---------------------|---------------|-----------------|------------|----------------|
| Sequential | 28 | 28 | 30-60 min | 60-120 min | Optimal per cohort |
| Parallel (optimal) | 14 | 28 | 35-70 min | 35-70 min | Optimal overall |
| Parallel (oversubscribed) | 28 | 56 | 45-90 min | 45-90 min | Suboptimal but faster |

## Recommendations

### Option 1: Optimal Parallel Execution (Recommended)
```bash
# Terminal 1
python utility_scripts/run_full_ffa_analysis.py \
  --cohort-name opioid_ed \
  --age-band 55-64 \
  --n-jobs 14

# Terminal 2
python utility_scripts/run_full_ffa_analysis.py \
  --cohort-name non_opioid_ed \
  --age-band 65-74 \
  --n-jobs 14
```
**Result**: 28 total workers, optimal CPU utilization, fastest overall completion

### Option 2: Oversubscribed Parallel Execution
```bash
# Terminal 1 (default 28 workers)
python utility_scripts/run_full_ffa_analysis.py \
  --cohort-name opioid_ed \
  --age-band 55-64

# Terminal 2 (default 28 workers)
python utility_scripts/run_full_ffa_analysis.py \
  --cohort-name non_opioid_ed \
  --age-band 65-74
```
**Result**: 56 total workers, context switching overhead, but still faster than sequential

### Option 3: Sequential Execution
```bash
# Run one at a time
python utility_scripts/run_full_ffa_analysis.py --cohort-name opioid_ed --age-band 55-64
# Wait for completion, then:
python utility_scripts/run_full_ffa_analysis.py --cohort-name non_opioid_ed --age-band 65-74
```
**Result**: Optimal per-cohort performance, but slower overall

## Monitoring Oversubscription

### Check context switches:
```bash
# Before starting
vmstat 1 10

# During execution (should see high "cs" column)
# cs = context switches per second
# Should see 30,000-50,000/sec with oversubscription
```

### Check load average:
```bash
uptime
# Load average should be ~50-60 with 56 workers
# (vs ~25-30 with 28 workers)
```

### Check CPU wait time:
```bash
top
# Look for high "wa" (I/O wait) or "st" (steal time)
# With oversubscription, you'll see processes waiting for CPU
```

## Conclusion

**Running two cohorts with 28 workers each (56 total) will work**, but:

1. **Each cohort takes 15-30% longer** due to context switching overhead
2. **Overall completion is still faster** than sequential (~45-90 min vs 60-120 min)
3. **System will be heavily loaded** (all 32 cores at 90-100% utilization)
4. **Memory usage is fine** (~3-6 GB total, well within 1TB limit)

**Best practice**: Use `--n-jobs 14` per instance for optimal performance (28 total workers, no oversubscription).
