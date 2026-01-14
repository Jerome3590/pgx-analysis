# DuckDB Optimization Guide

## Overview

This document provides comprehensive guidance for optimizing DuckDB performance in our data processing pipeline, with a focus on partitioned data parallelization and common pitfalls.

## Table of Contents

1. [Worker Configuration: Pharmacy vs Medical](#worker-configuration-pharmacy-vs-medical)
2. [Memory Management: Understanding the Layers](#memory-management-understanding-the-layers)
3. [Partitioned Data Parallelization Strategy](#partitioned-data-parallelization-strategy)
4. [DuckDB Connection Best Practices](#duckdb-connection-best-practices)
5. [Common Issues and Solutions](#common-issues-and-solutions)
6. [Performance Tuning](#performance-tuning)
7. [Troubleshooting Guide](#troubleshooting-guide)

## Worker Configuration: Pharmacy vs Medical

### Key Differences

The pharmacy and medical pipelines require **different worker configurations** due to significant differences in partition sizes and memory requirements:

| Pipeline | Workers | Threads/Worker | Est. Memory/Worker | Total Threads | Reasoning |
|----------|---------|----------------|-------------------|---------------|-----------|
| **Pharmacy** | 48 | 1 | ~1-2GB | 48 | Smaller partitions, can run more in parallel |
| **Medical** | 16 | 1 | ~2-4GB | 16 | Larger partitions, especially 25-44 and 45-54 age bands |

### Why Medical Requires Fewer Workers

**Problem Discovered (October 2025):**
Initial runs with 48 workers (matching pharmacy) caused Out Of Memory (OOM) crashes with return code -9, specifically when processing the largest age bands (25-44, 45-54).

**Root Cause:**
Medical partitions are **2-4x larger** than pharmacy partitions for the same age band/year:
- **Pharmacy**: 200-400 MB per partition
- **Medical**: 500-1,500 MB per partition
- **Medical 25-44**: Up to 2GB+ per partition (largest healthcare cohort)

**Memory Consumption Observed:**
```
System Memory Usage During Processing:
- Idle: 0.7% (996GB available / 1TB total)
- Processing 25-44 with 9 workers: 23.2% (771GB available)
  → ~225GB used for 9 workers = ~25GB per worker peak
- Processing 45-54 with 9 workers: Similar pattern
```

**Iterative Optimization:**
1. **48 workers**: OOM crashes on 25-44 age band
2. **24 workers**: Still OOM crashes on 25-44 age band  
3. **16 workers**: ✅ Success! All partitions completed

### Configuration in Code

**Pharmacy Processing (`3a_clean_pharmacy.py`):**
```bash
python 1_apcd_input_data/3_apcd_clean.py pharmacy --workers 48
```

**Medical Processing (`3b_clean_medical.py`):**
```bash
python 1_apcd_input_data/3_apcd_clean.py medical --workers 16
```

### Thread Management Per Worker

**Critical Fix (October 2025):**
```python
# In helpers_1997_13/duckdb_utils.py
def create_simple_duckdb_connection(logger, tmp_dir=None):
    conn = duckdb.connect(database=':memory:')
    # ... S3 setup ...
    
    # CRITICAL: Set threads to 1 per connection
    conn.sql("SET threads = 1")
    
    return conn
```

**Why This Matters:**
- **Before**: DuckDB auto-detected 32 CPU cores and used all of them per connection
  - 24 workers × 32 threads = **768 threads** competing for 32 cores
  - Massive thread over-subscription, context switching, memory thrashing
- **After**: Each worker uses exactly 1 thread
  - 16 workers × 1 thread = **16 threads** for 32 cores
  - Clean resource allocation, predictable performance

## Memory Management: Understanding the Layers

### The Memory Hierarchy

Memory management in our pipeline operates at **four distinct levels**. Understanding each is critical for preventing OOM crashes.

#### 1. System-Level Memory (1TB EC2 Instance)

**What it is:** Total physical RAM available on the machine

**Metrics:**
```python
import psutil
mem = psutil.virtual_memory()
print(f"Total: {mem.total / (1024**3):.1f}GB")
print(f"Used: {mem.percent}%")
print(f"Available: {mem.available / (1024**3):.1f}GB")
```

**Observed During Processing:**
- **Idle**: 0.7% used (996GB available)
- **Light workload** (0-12, 13-24): 1-5% used (950-990GB available)
- **Heavy workload** (25-44, 45-54): 20-30% used (700-800GB available)

**Danger Zone:** >90% usage → OS starts killing processes (SIGKILL -9)

#### 2. Per-Worker Process Memory

**What it is:** Memory allocated to each subprocess running `clean_medical.py` or `clean_pharmacy.py`

**Components:**
- Python interpreter overhead (~100MB)
- DuckDB connection and buffers
- Loaded partition data
- Intermediate query results
- S3 upload buffers

**Estimated Memory Per Worker:**

| Dataset Type | Small Partitions | Medium Partitions | Large Partitions |
|--------------|------------------|-------------------|------------------|
| **Pharmacy** | 1-2GB (all ages) | 1-2GB (all ages) | 1-2GB (all ages) |
| **Medical 0-12, 13-24, 95-114** | 1-2GB | 1-2GB | 1-2GB |
| **Medical 25-44, 45-54** | 2-3GB | 3-4GB | **4-6GB peak** |
| **Medical 55-64, 65-74** | 1-2GB | 2-3GB | 3-4GB |

**Memory Calculation:**
```
Total System Memory Used = Workers × Peak Memory Per Worker

Example (Medical with 16 workers):
- 12 workers processing small partitions: 12 × 2GB = 24GB
- 4 workers processing 25-44 partitions: 4 × 6GB = 24GB
- Total Peak: ~48GB (5% of 1TB system memory)
```

**Why 16 Workers for Medical:**
```
With 24 workers on 25-44:
- Peak scenario: 24 × 6GB = 144GB
- But 25-44 has 5 years × 24 workers = potential 120 parallel jobs
- Actual: ~8-10 processing 25-44 at once = 60GB
- Plus other age bands = 80-100GB total
- Add OS overhead and buffers = 120-150GB
- System thrashing begins at this point → OOM

With 16 workers on 25-44:
- Peak scenario: 16 × 6GB = 96GB  
- Actual: ~5-6 processing 25-44 at once = 36GB
- Plus other age bands = 50-70GB total
- Safe margin: 930GB still available ✅
```

#### 3. Per-Thread Memory (Inside DuckDB)

**What it is:** Memory allocated per DuckDB thread for query execution

**Before Fix (Auto-detected threads):**
```
- DuckDB detected 32 CPU cores
- Created 32 threads per connection
- Each thread allocated:
  - Hash table buffers
  - Sort buffers  
  - Temporary vectors
  - String heaps
- 32 threads × ~200MB each = ~6GB per connection (just for thread overhead!)
```

**After Fix (threads=1):**
```
- Explicitly set to 1 thread per connection
- Single thread uses:
  - One set of buffers
  - One sort workspace
  - One hash table
- 1 thread × ~200MB = ~200MB per connection (thread overhead)
```

**Memory Savings:**
```
Per worker: 6GB - 0.2GB = 5.8GB saved
For 16 workers: 16 × 5.8GB = 92.8GB total memory saved!
```

**Key Insight:** DuckDB's multi-threading is designed for **single-process workloads**. When using multiprocessing (many workers), you **must** set `threads=1` per connection to avoid over-subscription.

#### 4. Per-Query Memory (Temporary Data)

**What it is:** Memory used during specific SQL operations

**Memory-Intensive Operations:**

1. **`ORDER BY`** (REMOVED in optimization):
   ```sql
   -- OLD (memory-intensive):
   COPY (SELECT * FROM table ORDER BY col1, col2) TO 's3://...'
   -- Loaded entire partition into memory for sorting
   ```

2. **`GROUP BY` with large cardinality**:
   ```sql
   -- High memory if many unique values
   GROUP BY mi_person_key, event_date, diagnosis_code
   ```

3. **`JOIN` operations**:
   ```sql
   -- Both tables loaded into memory
   LEFT JOIN drug_map ON LOWER(p.drug_name) = dm.key
   ```

4. **Window functions**:
   ```sql
   -- Entire partition window held in memory
   ROW_NUMBER() OVER (PARTITION BY mi_person_key ORDER BY event_date)
   ```

**Optimizations Applied:**

1. **Removed ORDER BY from COPY**:
   ```python
   # Before: Load all → sort → write (memory spike)
   COPY (SELECT * FROM table ORDER BY ...) TO 's3://...'
   
   # After: Stream write in chunks (memory efficient)
   COPY (SELECT * FROM table) TO 's3://...'
   (FORMAT PARQUET, ROW_GROUP_SIZE 100000)
   ```

2. **Filter BEFORE computation**:
   ```sql
   -- Before: Compute quality on all data, then filter
   CREATE TABLE enriched AS
   SELECT *, <quality_calcs> FROM medical_filtered
   WHERE age_band = '25-44' AND event_year = 2017
   
   -- After: Filter first, then compute (10-100x less data)
   CREATE TABLE enriched AS
   SELECT *, <quality_calcs> FROM (
       SELECT * FROM medical_filtered  
       WHERE age_band = '25-44' AND event_year = 2017
   )
   ```

3. **Use ROW_GROUP_SIZE for chunked writes**:
   ```python
   # Writes 100K rows at a time instead of entire dataset
   ROW_GROUP_SIZE 100000
   ```

### Memory Monitoring Best Practices

**1. Log Memory at Key Stages:**
```python
import psutil

def log_memory(logger, stage):
    mem = psutil.virtual_memory()
    logger.info(f"💾 Memory at {stage}: {mem.percent}% used, "
                f"{mem.available / (1024**3):.1f}GB available")

# Usage:
log_memory(logger, "before data load")
conn.sql("CREATE TABLE data AS SELECT * FROM 's3://...'")
log_memory(logger, "after data load")
```

**2. Set Memory Thresholds:**
```python
# Warn if memory usage exceeds threshold
if mem.percent > 80:
    logger.warning(f"⚠️  High memory usage: {mem.percent}%")
if mem.percent > 95:
    logger.error(f"🚨 Critical memory: {mem.percent}% - near OOM!")
```

**3. Catch Memory Errors:**
```python
try:
    process_data()
except MemoryError as e:
    logger.error("❌ OUT OF MEMORY")
    mem = psutil.virtual_memory()
    logger.error(f"💾 Memory at failure: {mem.percent}% used")
    sys.exit(137)  # 128 + 9 (SIGKILL equivalent)
```

### Summary: Memory Management Checklist

- [ ] **System Memory**: Monitor with `psutil`, stay below 80% usage
- [ ] **Worker Count**: Scale based on partition sizes (16 for medical, 48 for pharmacy)
- [ ] **Thread Count**: Always set `conn.sql("SET threads = 1")` for multiprocessing
- [ ] **Query Optimization**: Filter before computing, avoid ORDER BY on COPY
- [ ] **Chunked Writes**: Use `ROW_GROUP_SIZE 100000` for large datasets
- [ ] **Memory Logging**: Log at every major step to identify bottlenecks
- [ ] **Error Handling**: Catch MemoryError and log context before exit

## Partitioned Data Parallelization Strategy

### Core Concept

Process data in parallel by dividing it into partitions, where each worker processes a specific age band and year combination. This approach maximizes resource utilization while maintaining data integrity.

### Implementation

```python
# Worker Configuration
workers = min(48, (os.cpu_count() or 8) * 1.5)  # Scale with CPU cores
threads_per_worker = 1  # Avoid thread contention
memory_per_worker = "2GB"  # Conservative memory allocation

# Partition Discovery
def discover_partitions(s3_path, min_year, max_year):
    """Discover age_band/year pairs from partitioned S3 data"""
    # Use boto3 for S3 listing instead of DuckDB discovery
    # Extract age_band and event_year from S3 object keys
    # Return list of (age_band, event_year, estimated_rows) tuples
```

### Benefits

- **Scalability**: Automatically scales with available CPU cores
- **Efficiency**: Each worker processes only relevant data
- **Fault Tolerance**: Individual partition failures don't affect others
- **Resource Optimization**: Memory and CPU usage distributed evenly

## DuckDB Connection Best Practices

### 1. Simplified Connection Setup

**❌ Avoid Complex Chaining:**
```python
# DON'T: Complex chained setup
def init_duckdb():
    conn = duckdb.connect()
    conn = configure_s3(conn)
    conn = configure_memory(conn)
    conn = configure_threads(conn)
    return conn
```

**✅ Use Simple, Direct Setup:**
```python
# DO: Simple, direct setup
def create_simple_duckdb_connection(logger, tmp_dir=None):
    conn = duckdb.connect(database=':memory:')
    
    # Basic S3 setup
    conn.sql("INSTALL httpfs; LOAD httpfs;")
    conn.sql("INSTALL aws; LOAD aws;")
    conn.sql("CALL load_aws_credentials();")
    conn.sql("SET s3_region='us-east-1'")
    conn.sql("SET s3_url_style='path'")
    
    # CRITICAL: Set threads to 1 for multiprocessing environments
    conn.sql("SET threads = 1")
    
    # Let DuckDB handle memory automatically
    return conn
```

### 2. Memory and Thread Management

**❌ Avoid Manual Memory Settings:**
```python
# DON'T: Manual memory settings that can cause conflicts
conn.sql(f"SET memory_limit = '{memory_limit}'")
```

**✅ Let DuckDB Auto-Detect Memory:**
```python
# DO: Let DuckDB handle memory allocation automatically
# DuckDB automatically detects optimal memory settings
```

**✅ BUT: Always Set Threads=1 for Multiprocessing:**
```python
# CRITICAL for parallel worker environments
conn.sql("SET threads = 1")

# Why: DuckDB's auto-detection assumes single-process usage
# With multiprocessing, each worker needs exactly 1 thread
# to avoid over-subscription and memory thrashing
```

**Thread Configuration Rules:**
- **Single-process script** (running one DuckDB job): Let DuckDB auto-detect threads
- **Multi-process orchestrator** (parallel workers): Always set `threads = 1` per connection
- Our pipeline uses multiprocessing → **Always set threads = 1**

### 3. S3 Uploader Configuration (CRITICAL)

**❌ Avoid Manual S3 Uploader Settings:**
```python
# DON'T: Manual S3 uploader settings cause memory issues
conn.sql("SET s3_uploader_max_filesize='5368709120'")
conn.sql("SET s3_uploader_max_parts_per_file='10000'")
```

**✅ Let DuckDB Auto-Configure:**
```python
# DO: Let DuckDB handle S3 upload optimization
# DuckDB automatically configures based on:
# - Available memory
# - File sizes
# - Network conditions
# No manual S3 uploader configuration needed
```

**Key Discovery (October 2025):**
Manual S3 uploader parameters were identified as the **second major cause of memory issues**, after fixing the connection chaining problems. DuckDB's auto-configuration for S3 uploads is superior to manual tuning.

### 4. Connection Isolation

**❌ Avoid Global DuckDB State:**
```python
# DON'T: Global connections that can interfere
global_conn = duckdb.connect()
# Multiple workers using same connection
```

**✅ Use Process-Isolated Connections:**
```python
# DO: Each worker gets its own connection
def worker_process(age_band, event_year):
    conn = create_simple_duckdb_connection(logger)
    # Process data
    conn.close()
```

## Common Issues and Solutions

### 1. Out Of Memory (OOM) - Process Killed with Return Code -9

**Problem:** Workers crash with exit code -9 (SIGKILL) during processing of large age bands

**Symptoms:**
- `Process failed with return code -9`
- System memory usage >90%
- Workers killed without error messages (killed by OS)
- Happens specifically on 25-44, 45-54 age bands

**Root Causes:**
- **Too many workers** for partition sizes (medical partitions are 2-4x larger than pharmacy)
- **Thread over-subscription**: DuckDB auto-detecting all CPU cores per connection
  - 24 workers × 32 threads = 768 threads competing for 32 cores
  - Each thread allocates buffers, causing massive memory consumption
- **Memory-intensive SQL operations**: ORDER BY on large datasets during COPY

**Solution:**
```python
# 1. Reduce workers for medical processing
python 1_apcd_input_data/3_apcd_clean.py medical --workers 16  # Not 48!

# 2. Set threads=1 in DuckDB connection (CRITICAL)
conn.sql("SET threads = 1")

# 3. Remove ORDER BY from COPY statements
# Before:
COPY (SELECT * FROM table ORDER BY col1, col2) TO 's3://...'

# After:
COPY (SELECT * FROM table) TO 's3://...'
(FORMAT PARQUET, ROW_GROUP_SIZE 100000)

# 4. Filter before computing
# Before: compute on all data, then filter
SELECT *, <expensive_calcs> FROM large_table WHERE age_band = '25-44'

# After: filter first, then compute
SELECT *, <expensive_calcs> FROM (
    SELECT * FROM large_table WHERE age_band = '25-44'
)
```

**Memory Calculation:**
```
Medical 25-44 partition: Up to 6GB per worker at peak
16 workers × 6GB = 96GB max (safe on 1TB machine)
24 workers × 6GB = 144GB max (causes OOM)
```

### 2. S3 Uploader Memory Issues

**Problem:** Memory allocation errors during large S3 COPY operations

**Root Causes:**
- Manual S3 uploader configuration interfering with auto-detection
- `SET s3_uploader_max_filesize` and `SET s3_uploader_max_parts_per_file` causing conflicts
- Override of DuckDB's intelligent S3 upload management

**Solution:**
```python
# Remove ALL manual S3 uploader configuration
# Let DuckDB handle S3 uploads automatically
# DuckDB knows best based on available resources
```

### 3. Memory Limit Errors

**Problem:** `Parser Error: Unknown unit for memory: ''`

**Root Causes:**
- Complex chained connection setup
- Memory check queries that corrupt settings
- Global state interference between connections
- Manual memory settings conflicting with auto-detection

**Solution:**
```python
# Use simplified connection setup
# Remove all manual memory/thread settings
# Let DuckDB handle resource allocation automatically
```

### 4. SQL Syntax Errors with Hyphens

**Problem:** `Parser Error: syntax error at or near "-"`

**Root Cause:** DuckDB table names cannot contain hyphens

**Solution:**
```python
# Strategy: Replace hyphens with underscores for table names only
safe_run_id = age_band.replace('-', '_')  # "95-114" → "95_114"
run_id = f"{safe_run_id}_{event_year}_{process_id}"

# Keep hyphens for S3 paths (Hive-style partitioning)
partitioned_path = f"s3://bucket/age_band={age_band}/event_year={event_year}"
```

### 5. Missing Column Errors

**Problem:** `Binder Error: Referenced column "ndc_code" not found`

**Root Cause:** Partitioned data has different column structure than expected

**Solution:**
```python
# Check available columns first
available_columns = conn.sql("DESCRIBE table_name").fetchall()

# Only select available columns
SELECT 
    mi_person_key,
    age_band,
    event_year,
    drug_name,
    incurred_date,  # Available
    gender_source,  # Available
    age_source      # Available
FROM table_name
```

### 6. Duplicate Sessions

**Problem:** Multiple pipeline instances running simultaneously

**Root Causes:**
- Jupyter notebook re-execution without process checking
- Hardcoded log file names causing conflicts
- No unique process identification

**Solution:**
```python
# Use unique process identification
process_id = os.getpid()
run_id = f"{safe_run_id}_{event_year}_{process_id}"

# Use unique log file names
log_file = f"pharmacy_clean_output_{timestamp}_{process_id}.txt"

# Check for existing processes before starting
```

## Performance Tuning

### 1. Worker Configuration

```python
# Optimal worker count
workers = min(48, (os.cpu_count() or 8) * 1.5)

# Conservative memory per worker
memory_per_worker = "2GB"

# Single thread per worker to avoid contention
threads_per_worker = 1
```

### 2. S3 Optimization

```python
# Use S3 listing instead of DuckDB discovery
# More efficient for large datasets
s3_client = boto3.client('s3')
paginator = s3_client.get_paginator('list_objects_v2')
```

### 3. Memory Management

```python
# Clean up temporary objects
conn.sql(f"DROP TABLE IF EXISTS temp_table_{run_id}")

# Use memory-efficient operations
# Process data in chunks if needed
```

## Troubleshooting Guide

### 1. Memory Issues

**Symptoms:**
- `memory_limit` errors
- Out of memory errors
- Process crashes

**Debugging:**
```python
# Check current memory usage
result = conn.sql("SELECT SUM(memory_usage_bytes) FROM duckdb_memory()").fetchone()
print(f"Memory usage: {result[0] / (1024**3):.1f} GB")

# Check memory limit
memory_limit = conn.sql("SELECT current_setting('memory_limit')").fetchone()[0]
print(f"Memory limit: {memory_limit}")
```

### 2. Connection Issues

**Symptoms:**
- Connection timeouts
- S3 access errors
- Authentication failures

**Debugging:**
```python
# Test S3 connectivity
conn.sql("SELECT * FROM 's3://bucket/test.parquet' LIMIT 1")

# Check S3 configuration
region = conn.sql("SELECT current_setting('s3_region')").fetchone()[0]
print(f"S3 region: {region}")
```

### 3. Data Issues

**Symptoms:**
- Missing columns
- Data type errors
- Partition not found

**Debugging:**
```python
# Check table structure
columns = conn.sql("DESCRIBE table_name").fetchall()
for col in columns:
    print(f"{col[0]}: {col[1]}")

# Check partition existence
conn.sql("SELECT COUNT(*) FROM 's3://bucket/partition/**/*.parquet'")
```

## Best Practices Summary

1. **Keep it Simple**: Use direct, simple DuckDB connection setup
2. **Scale Workers Appropriately**: 
   - Pharmacy: 48 workers (small partitions)
   - Medical: 16 workers (large partitions, especially 25-44 and 45-54)
3. **Always Set Threads=1**: Critical for multiprocessing environments to avoid over-subscription
   - `conn.sql("SET threads = 1")` in every worker connection
4. **Let DuckDB Handle Memory**: Avoid manual memory limits, let DuckDB auto-detect
5. **No S3 Uploader Settings**: DuckDB auto-configures S3 uploads optimally
6. **Isolate Connections**: Each worker gets its own connection
7. **Optimize Queries**:
   - Filter BEFORE computing aggregations
   - Remove ORDER BY from COPY statements
   - Use ROW_GROUP_SIZE for chunked writes
8. **Handle Hyphens Properly**: Replace with underscores for table names only
9. **Monitor Memory**: Log at key stages, stay below 80% system memory
10. **Use Unique Identifiers**: Prevent conflicts between workers
11. **Clean Up Resources**: Drop temporary objects when done

## Migration Guide

### From Complex to Simple Setup

1. **Remove Complex Chaining:**
   - Eliminate `init_duckdb()` → `get_duckdb_connection()` → `_configure_duckdb_ec2_settings()`
   - Use direct `duckdb.connect()` calls

2. **Remove Manual Settings:**
   - Delete all `SET memory_limit` and `SET threads` commands
   - Let DuckDB auto-detect optimal settings

3. **Fix Naming Issues:**
   - Replace hyphens with underscores for table names
   - Keep hyphens for S3 paths

4. **Update Column Selection:**
   - Check available columns in partitioned data
   - Update SELECT statements to match available columns

5. **Add Process Isolation:**
   - Use unique process IDs
   - Create separate connections per worker

## Conclusion

The key to successful DuckDB optimization is simplicity and letting DuckDB handle resource management automatically. Complex chaining and manual configuration often lead to conflicts and errors. By following these best practices, you can achieve reliable, high-performance data processing with DuckDB.
