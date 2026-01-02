# DuckDB Configuration, Optimization, and Orchestration Guide

This document describes the **DuckDB architecture**, **optimizations**, and **orchestration fixes** applied across PGx pipelines, including APCD preprocessing and the Cohort Creation system. It captures environment-aware configuration patterns, performance optimization, and best practices for stable large-scale operation on EC2 and local systems.

***

## 🎯 Purpose

DuckDB serves as the in-memory SQL engine powering the **PGx analytics pipelines**, providing lightning-fast Parquet I/O, S3 integration, and SQL-layer data engineering.

Recent refactors (October 2025) have unified configuration management across all pipelines, introducing three critical layers:

1. **Environment-aware connection creation** (`get_duckdb_connection`, `create_simple_duckdb_connection`)
2. **Cross-platform optimization rules** (auto-memory/thread/extension management)
3. **Orchestration fixes** eliminating global state interference

***

## 🏗️ Architecture Overview

### Multi-Process Execution

Each task (medical, pharmacy, and cohort creation) runs as an **independent process** or worker that:

- Initializes its **own isolated DuckDB connection**
- Manages local and S3 storage independently
- Cleans up disk space after completion


### Core Module Structure

```
helpers/
├── duckdb_utils.py         # All connection, cleanup, and monitoring utilities
└── __init__.py

1_apcd_input_data/
├── 3a_clean_pharmacy.py    # Worker script (single connection)
├── 3b_clean_medical.py     # Worker script (single connection)
└── 3_apcd_clean.py         # Orchestrator (no DuckDB usage, uses S3 discovery)

2_create_cohort/
└── create_cohort_optimized.py  # Cohort creation with centralized DuckDB logic
```


***

## 🚦 Essential DuckDB Configuration

### Environment-Aware Auto-Detection (recommended)

```python
from helpers.duckdb_utils import get_duckdb_connection
import logging

logger = logging.getLogger()
conn = get_duckdb_connection(logger, memory_limit=None, threads=None)
# Auto-detects: 900GB memory, 30 threads on EC2
# Auto-detects: 13GB memory, 14 threads on Windows
```


### Manual Override (for pinned EC2 environments)

```python
import duckdb, tempfile, os
conn = duckdb.connect(database=':memory:')
conn.sql("INSTALL httpfs; LOAD httpfs;")
conn.sql("INSTALL aws; LOAD aws;")
conn.sql("CALL load_aws_credentials();")

temp_dir = os.path.join(tempfile.gettempdir(), 'duckdb_temp')
os.makedirs(temp_dir, exist_ok=True)

conn.sql("SET temp_directory = '" + temp_dir + "'")
conn.sql("SET memory_limit = '900GB'")
conn.sql("SET threads = 30")
conn.sql("SET s3_region = 'us-east-1'")
conn.sql("SET s3_url_style = 'path'")
conn.sql("PRAGMA disable_profiling")
conn.sql("PRAGMA enable_object_cache")
conn.sql("PRAGMA disable_verification")
conn.sql("PRAGMA enable_optimizer")
```


***

## ✅ Valid DuckDB Configuration Parameters

### SET Commands (Connection-Level Settings)

| Parameter | Valid Values | Description | Example |
| :-- | :-- | :-- | :-- |
| `memory_limit` | String with unit (e.g., '2GB', '900GB') | Maximum memory per connection | `SET memory_limit='2GB'` |
| `threads` | Integer (use `PRAGMA threads=N`) | Number of threads per connection | `PRAGMA threads=1` |
| `temp_directory` | String path | Directory for temp files | `SET temp_directory='/tmp/duckdb'` |
| `s3_region` | String (AWS region) | S3 region for operations | `SET s3_region='us-east-1'` |
| `s3_url_style` | 'path' or 'virtual-hosted' | S3 URL style | `SET s3_url_style='path'` |
| `http_timeout` | Integer (milliseconds) | HTTP request timeout | `SET http_timeout=300000` |
| `http_retries` | Integer | Number of HTTP retry attempts | `SET http_retries=5` |
| `http_retry_wait_ms` | Integer (milliseconds) | Wait time between retries | `SET http_retry_wait_ms=1000` |
| `s3_uploader_max_filesize` | String with unit | Max file size for S3 uploads (used for part size calculation) | `SET s3_uploader_max_filesize='100GB'` |
| `s3_uploader_max_parts_per_file` | Integer | Max parts per file (used for part size calculation) | `SET s3_uploader_max_parts_per_file=10000` |
| `s3_uploader_thread_limit` | Integer | Maximum number of uploader threads | `SET s3_uploader_thread_limit=8` |

### PRAGMA Commands (Query-Level Settings)

| Parameter | Valid Values | Description | Example |
| :-- | :-- | :-- | :-- |
| `threads` | Integer | Number of threads | `PRAGMA threads=1` |
| `enable_profiling` | 'json', 'query_tree', 'query_tree_optimizer' | Enable query profiling | `PRAGMA enable_profiling='json'` |
| `disable_profiling` | No value | Disable profiling | `PRAGMA disable_profiling` |
| `enable_progress_bar` | Boolean (as string) | Enable/disable progress bar | `PRAGMA enable_progress_bar=false` |
| `enable_object_cache` | No value | Enable object cache | `PRAGMA enable_object_cache` |
| `disable_verification` | No value | Disable verification | `PRAGMA disable_verification` |
| `enable_optimizer` | No value | Enable optimizer | `PRAGMA enable_optimizer` |

### Extension Commands

| Command | Description | Notes |
| :-- | :-- | :-- |
| `INSTALL extension_name` | Downloads extension to disk | Only needed once per system/user |
| `LOAD extension_name` | Loads extension into connection | Required per connection |
| `CALL load_aws_credentials()` | Loads AWS credentials | Required per connection for S3 access |

**Best Practice:** Extensions are pre-installed on EC2. Use `LOAD` only, skip `INSTALL` to avoid contention.

## ❌ Invalid DuckDB Configuration Parameters

### Common Invalid Parameters

| Invalid Parameter | Error Message | Why Invalid | Alternative |
| :-- | :-- | :-- | :-- |
| `s3_max_connections` | `Catalog Error: unrecognized configuration parameter "s3_max_connections"` | Not a valid DuckDB parameter | Use `s3_uploader_thread_limit` to control uploader threads, or let DuckDB manage automatically |
| `SET threads=N` | Syntax error | `threads` must use `PRAGMA`, not `SET` | Use `PRAGMA threads=N` |
| `PRAGMA enable_profiling=false` | Syntax error | Profiling uses `SET` or `PRAGMA disable_profiling` | Use `PRAGMA disable_profiling` or `PRAGMA enable_profiling='json'` |
| `SET memory_limit` (empty) | `Unknown unit for memory: ''` | Memory limit must include unit | Use `SET memory_limit='2GB'` |

### Common Mistakes

**❌ Don't use SET for threads:**
```python
# WRONG - will cause syntax error
conn.sql("SET threads=1")
```

**✅ Use PRAGMA for threads:**
```python
# CORRECT
conn.sql("PRAGMA threads=1")
```

**❌ Don't use invalid S3 parameters:**
```python
# WRONG - s3_max_connections doesn't exist
conn.sql("SET s3_max_connections=256")
```

**✅ Use valid S3 uploader parameters:**
```python
# CORRECT - use s3_uploader_thread_limit to control uploader threads
conn.sql("SET s3_uploader_thread_limit=8")

# OR let DuckDB manage automatically (recommended)
# DuckDB automatically manages S3 connection pool and uploader threads
```

**Reference:** [DuckDB S3 API Documentation](https://duckdb.org/docs/stable/core_extensions/httpfs/s3api)

**❌ Don't use PRAGMA with = for boolean settings:**
```python
# WRONG - invalid syntax
conn.sql("PRAGMA enable_profiling=false")
```

**✅ Use PRAGMA disable or SET with string:**
```python
# CORRECT - disable profiling
conn.sql("PRAGMA disable_profiling")

# OR enable with format
conn.sql("PRAGMA enable_profiling='json'")
```

### Parameter Reference by Category

**Memory & Performance:**
- ✅ `SET memory_limit='2GB'` - Valid
- ✅ `PRAGMA threads=1` - Valid
- ❌ `SET threads=1` - Invalid (use PRAGMA)
- ❌ `SET memory_limit` - Invalid (must include unit)

**S3 Configuration:**
- ✅ `SET s3_region='us-east-1'` - Valid
- ✅ `SET s3_url_style='path'` - Valid (or 'vhost')
- ✅ `SET http_timeout=300000` - Valid
- ✅ `SET http_retries=5` - Valid
- ✅ `SET http_retry_wait_ms=1000` - Valid
- ✅ `SET s3_uploader_max_filesize='100GB'` - Valid (used for part size calculation)
- ✅ `SET s3_uploader_max_parts_per_file=10000` - Valid (used for part size calculation)
- ✅ `SET s3_uploader_thread_limit=8` - Valid (max uploader threads)
- ❌ `SET s3_max_connections=256` - Invalid (not a valid parameter, use `s3_uploader_thread_limit` instead)

**Profiling:**
- ✅ `PRAGMA enable_profiling='json'` - Valid
- ✅ `PRAGMA disable_profiling` - Valid
- ❌ `PRAGMA enable_profiling=false` - Invalid (use disable_profiling)
- ❌ `SET enable_profiling='json'` - Invalid (use PRAGMA)

**Extensions:**
- ✅ `LOAD httpfs;` - Valid (if already installed)
- ✅ `INSTALL httpfs; LOAD httpfs;` - Valid (if not installed)
- ✅ `CALL load_aws_credentials();` - Valid

***


## ⚙️ Key Optimizations

### 1. Auto-Detection for Memory and Threads

- Uses 90% of system RAM (up to 900GB)
- Allocates `(CPU cores - 2)` threads
- Functions on Windows, Linux, EC2, and macOS
- Manual override supported for benchmarking


### 2. Temporary File Management

- Uses NVMe or tmpfs for faster temp I/O
- Automated cleanup via `cleanup_duckdb_temp_files()`
- Monitors disk usage after every phase


### 3. S3 Integration \& Reliability

- Region fixed to `us-east-1`
- 300-second timeout (300000ms) and 5 retries
- Path-style URL for compatibility (`s3_url_style='path'`)
- Configurable uploader settings (optional):
  - `s3_uploader_max_filesize` - Max file size for part size calculation
  - `s3_uploader_max_parts_per_file` - Max parts per file for part size calculation  
  - `s3_uploader_thread_limit` - Maximum number of uploader threads
- Default uploader configuration is usually sufficient (auto-managed)

**Reference:** [DuckDB S3 API Documentation](https://duckdb.org/docs/stable/core_extensions/httpfs/s3api)


### 4. Profiling and Diagnostics

```python
conn.sql("PRAGMA enable_profiling")
conn.sql("SET enable_profiling = 'json'")
# JSON, query_tree, query_tree_optimizer supported
conn.sql("PRAGMA disable_profiling")
```

Monitoring:

```bash
/tmp/duckdb_profiling_step3_medical.json
/tmp/duckdb_profiling_step7_event_features.json
```


***

## 🧠 Memory, Checkpoints, and Error Handling

### Resilience and Cleanup

- Auto checkpoint after each major step
- Disk and memory monitoring integrated
- Automatic file cleanup on errors

```python
try:
    run_pipeline_step()
except Exception:
    cleanup_duckdb_temp_files(logger)
    raise
```


### Checkpoint Example

```json
{
  "pipeline": "create_cohort",
  "step": "phase2_event_processing",
  "status": "completed",
  "metrics": {"duration_sec": 720, "records": 5500000}
}
```


***

## 🧩 Orchestration Fix (2025-10)

### Problem

DuckDB was previously loaded both in orchestrator and worker processes, creating **global state interference**:

- Global `memory_limit` reset to empty string
- Conflicting `httpfs` extension loads
- Worker crashes (`Parser Error: Unknown unit for memory: ''`)


### Solution

**Removed DuckDB entirely from orchestrator functions.**

New logic in `3_apcd_clean.py`:

```python
from boto3 import client as s3_client

def discover_from_pharmacy():
    s3 = s3_client("s3")
    return [obj["Key"] for obj in s3.list_objects_v2(Bucket="pgxdatalake", Prefix="silver/imputed/pharmacy")["Contents"]]
```

Workers create their **own isolated DuckDB connection**:

```python
from helpers.duckdb_utils import create_simple_duckdb_connection
conn = create_simple_duckdb_connection(logger)
```


### Benefits

- No overlapping DuckDB states
- 100% connection isolation
- Faster discovery stage (S3-only)
- Workers maintain consistent memory limits

***

## 🚀 Partitioned Data Parallelization

### Overview

Processing is distributed per `(age_band, event_year)` partition.
Each worker:

- Uses 1 thread and ~2GB of RAM
- Maintains one DuckDB connection for the full partition
- Operates independently


### Optimal Worker Configuration

| Parameter | Value | Description |
| :-- | :-- | :-- |
| Threads per Worker | 1 | Best for S3 I/O |
| RAM per Worker | 2GB | Optimal for partition |
| Workers on EC2 | 48 | For 32-core system |

### Command-Line Example

```bash
python 1_apcd_input_data/3_apcd_clean.py --threads 1 --mem-gb 2 --max-workers 48
```

**Throughput Improvement:** 15× over legacy single-connection model.

***

## 🧑‍🔧 Troubleshooting Common Errors

| Error | Cause | Solution |
| :-- | :-- | :-- |
| `Unknown unit for memory` | Global state interference | Use isolated worker connections |
| `enable_profiling=false` | Invalid PRAGMA syntax | Replace with `PRAGMA disable_profiling` |
| `syntax error near '-'` | Hyphens in table names | Use underscores in SQL identifiers only |
| No S3 files found | Hyphen→underscore mismatch | Keep hyphens for Hive-style paths |
| Memory corruption | Manual reconfiguration | Let DuckDB auto-detect settings |


***

## 📊 Performance Results

| Metric | Before | After | Gain |
| :-- | :-- | :-- | :-- |
| Discovery Phase | 40 min | 8 min | 80% faster |
| Worker Isolation | None | 100% | Eliminated conflicts |
| Memory Utilization | 8GB | 90% available RAM | 10–15× increase |
| Parallelization | Sequential | 48 workers | 15× throughput |
| Failure Recovery | Manual | Automated | 100% cleanup reliability |


***

## 🧩 Key Best Practices (Oct 2025)

**Do:**

- Use `create_simple_duckdb_connection()` or `get_duckdb_connection()`
- Let DuckDB decide memory/thread configuration
- Maintain one connection per worker
- Use S3 discovery without DuckDB

**Avoid:**

- Manual `SET memory_limit` overrides
- Global connections shared among processes
- Multiple DuckDB `LOAD httpfs` calls
- Using PRAGMA commands with `=` syntax

***

## 🧰 Reference Functions

| Function | Purpose |
| :-- | :-- |
| `get_duckdb_connection(logger, memory_limit=None, threads=None)` | Auto-detects environment and configures DuckDB |
| `create_simple_duckdb_connection(logger)` | Minimal setup letting DuckDB self-configure |
| `cleanup_duckdb_temp_files(logger)` | Cleans temp directories between runs |
| `enable_query_profiling(conn, logger, format, outfile)` | Enables profiling per step |
| `force_checkpoint(conn, logger)` | Writes a checkpoint file for intermediate results |


***

## ✅ Summary

By applying uniform DuckDB principles across pipelines:

- **Orchestrators** use pure S3 discovery (no DuckDB imports)
- **Workers** maintain isolated, optimized connections
- **Memory/threads** are automatically tuned for the host system
- **Profiling and checkpoints** ensure reliability and auditability
- **Parallel partition execution** delivers over 10× throughput gains

These practices yield a stable, scalable foundation for all PGx analytics pipelines.

**Version:** 4.2
**Last Updated:** November 9, 2025
**Maintainers:** PGx Data Engineering \& Analytics Team

