# Multiprocessing FP-Growth Pipeline Documentation

## Overview
This document provides comprehensive guidance for executing FP-Growth pipelines using multiprocessing with shared AWS connections to prevent resource exhaustion and improve performance.

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