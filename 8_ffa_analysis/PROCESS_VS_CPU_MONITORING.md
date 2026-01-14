# Monitoring FFA Analysis: Processes vs CPU Cores

## Summary

**You'll see BOTH processes AND CPU core utilization** - ProcessPoolExecutor creates separate processes, each of which can use a CPU core.

## What ProcessPoolExecutor Creates

### Process Model (Not Threads)
- **`ProcessPoolExecutor`** creates **separate processes** (not threads)
- Each worker is a **separate Python process** with its own memory space
- Each process can be scheduled on a different CPU core

### What You'll See

#### 1. **Process Count** (via `ps` or `top`)
```bash
ps aux | grep run_full_ffa_analysis | wc -l
# Output: ~29-30 processes per instance
```

**Breakdown:**
- **1 main process**: The parent script (`run_full_ffa_analysis.py`)
- **28 worker processes**: Created by `ProcessPoolExecutor(max_workers=28)`
- **Total**: ~29 processes per instance

**With two instances running:**
- Instance 1: ~29 processes
- Instance 2: ~29 processes
- **Total**: ~58 processes

#### 2. **CPU Core Utilization** (via `top`, `htop`, or `iostat`)
```bash
top -p $(pgrep -f run_full_ffa_analysis | tr '\n' ',' | sed 's/,$//')
```

**What you'll see:**
- **~28 CPU cores** at high utilization (80-100%) per instance
- Each worker process can use a different core
- OS scheduler distributes processes across available cores

**With two instances (28 workers each):**
- All 32 cores at 90-100% utilization
- Context switching between 56 processes competing for 32 cores

## Visual Example

### Single Instance (28 workers)

**Process view (`ps aux`):**
```
pgx3874  12345  89.1  0.1  python3.11 run_full_ffa_analysis.py  (main)
pgx3874  12346  78.9  0.1  python3.11 <defunct>                 (worker 1)
pgx3874  12347  82.3  0.1  python3.11 <defunct>                 (worker 2)
... (26 more worker processes)
```

**CPU core view (`top` or `htop`):**
```
CPU 0:  95%  (worker process)
CPU 1:  92%  (worker process)
CPU 2:  88%  (worker process)
...
CPU 27: 91%  (worker process)
CPU 28:   5%  (system/idle)
CPU 29:   3%  (system/idle)
CPU 30:   4%  (system/idle)
CPU 31:   6%  (system/idle)
```

### Two Instances (28 workers each)

**Process view:**
```
# Instance 1
pgx3874  12345  89.1  0.1  python3.11 run_full_ffa_analysis.py  (main 1)
pgx3874  12346-12373  ...  (28 worker processes for instance 1)

# Instance 2
pgx3874  12374  78.9  0.1  python3.11 run_full_ffa_analysis.py  (main 2)
pgx3874  12375-12402  ...  (28 worker processes for instance 2)

Total: ~58 processes
```

**CPU core view:**
```
CPU 0-31:  90-100%  (all cores busy, context switching between 56 processes)
```

## Key Differences: Processes vs Threads

### ProcessPoolExecutor (What We Use) ✅
- **Separate processes**: Each worker is a separate OS process
- **Separate memory**: Each process has its own memory space
- **Visible in `ps`**: You'll see ~29 processes per instance
- **CPU cores**: Each process can use a different core
- **Overhead**: Higher memory usage, but true parallelism (no GIL)

### ThreadPoolExecutor (Not Used)
- **Threads**: All workers share the same process
- **Shared memory**: All threads share the same memory space
- **Visible in `ps`**: You'd see only 1 process
- **CPU cores**: Limited by Python GIL (Global Interpreter Lock)
- **Overhead**: Lower memory, but limited parallelism in CPU-bound tasks

## Monitoring Commands

### Check Process Count
```bash
# Count total processes
ps aux | grep run_full_ffa_analysis | wc -l
# Expected: ~29 per instance (1 main + 28 workers)

# See all processes
ps aux | grep run_full_ffa_analysis

# See process tree
pstree -p $(pgrep -f run_full_ffa_analysis | head -1)
```

### Check CPU Core Utilization
```bash
# Per-core CPU usage
top -p $(pgrep -f run_full_ffa_analysis | tr '\n' ',' | sed 's/,$//')
# Press '1' to see per-core view

# Or use htop (more visual)
htop -p $(pgrep -f run_full_ffa_analysis | tr '\n' ',' | sed 's/,$//')

# Overall CPU usage
mpstat -P ALL 1 5
```

### Check Both Together
```bash
# Process count
echo "Processes: $(ps aux | grep run_full_ffa_analysis | wc -l)"

# CPU cores utilized (count cores > 50% usage)
mpstat -P ALL 1 1 | awk '$3 > 50 {count++} END {print "Cores > 50%: " count}'
```

## Answer to Your Question

**"Will this show up in CPU cores but not processes?"**

**Answer: It shows up in BOTH:**
- ✅ **Processes**: ~29 processes per instance (visible in `ps`)
- ✅ **CPU cores**: ~28 cores at high utilization (visible in `top`/`htop`)

**Why both?**
- `ProcessPoolExecutor` creates separate processes (not threads)
- Each process can be scheduled on a different CPU core
- So you see both the process count AND the CPU core utilization

## Summary Table

| Metric | Single Instance (28 workers) | Two Instances (28 each) |
|--------|------------------------------|-------------------------|
| **Processes** | ~29 (1 main + 28 workers) | ~58 (2 main + 56 workers) |
| **CPU Cores Used** | ~28 cores at 80-100% | All 32 cores at 90-100% |
| **Memory per Process** | ~50-100 MB | ~50-100 MB |
| **Total Memory** | ~1.5-3 GB | ~3-6 GB |
| **Visible in `ps`** | ✅ Yes (~29 processes) | ✅ Yes (~58 processes) |
| **Visible in `top`** | ✅ Yes (~28 cores busy) | ✅ Yes (all 32 cores busy) |
