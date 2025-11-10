
# PGx Data Pipeline – Comprehensive Architecture and Optimization Guide

This document defines the **complete architecture, configuration, and optimization standards** of the PGx Data Pipeline.
It unifies the strategies across **APCD input**, **imputation**, **cohort creation**, and **analysis pipelines**, featuring a fully **partition-first**, **parallel**, and **resilient** design.

***

## 🎯 Overview

The PGx Data Pipeline transports, transforms, and structures high-volume healthcare data from raw inputs through Gold-tier cohorts ready for analysis.
It achieves **massive parallelization**, **fault-tolerant scaling**, and **standardized modularity** across all stages using a unified execution framework.

***

## 🏗️ Core Architecture

### Logical Flow

```
[Raw S3 Input]
    ↓
APCD Input Processing (partitioned)
    ↓
Global Demographic Imputation
    ↓
Event Fact Table Construction
    ↓
Cohort Fact Creation (5:1 Controls)
    ↓
Analysis & Exports
    ↓
[Gold S3 Output]
```


### Design Components

| Component | Function |
| :-- | :-- |
| **Source Data (Silver Tier)** | Clean, standardized APCD extracts partitioned by age_band × event_year |
| **Processing Engine** | DuckDB + Python using parallelized workers |
| **Transformation** | Sequential modular phases (data → events → cohorts) |
| **Output Storage** | Partitioned Parquet datasets in S3 (“Gold” tier) |
| **Orchestration Layer** | Python-level orchestrator using ProcessPoolExecutor |
| **Monitoring** | Checkpoints in S3 and logging per step |


***

## ⚙️ Implementation Foundation

### Standard Worker Configuration

```python
WORKER_CONFIG = {
  "max_workers": min(48, (os.cpu_count() or 8) * 1.5),
  "threads_per_worker": 1,
  "memory_per_worker": 2,  # GB
  "connection_isolation": True,
  "retry_attempts": 3,
  "timeout_seconds": 3600
}
```

**Key Properties:**

- One-thread-per-worker for optimal S3 throughput (I/O bound workload).
- 2GB of memory allocated per partitioned worker.
- Fully isolated DuckDB instances for memory safety and checkpoint integrity.

### Environment-Aware DuckDB Configuration

- Auto-adapts to EC2 vs local dev without hardcoding limits.
- Use `create_simple_duckdb_connection(logger, tmp_dir)`:
  - Loads `httpfs` and `aws` (LOAD-only), calls `load_aws_credentials()`.
  - Sets `threads = 1` per worker (multiprocessing safe).
  - Lets DuckDB auto-detect memory; avoid manual `memory_limit` in pipeline scripts.
  - Supports per-worker `temp_directory` to avoid file locking and collisions.

Quick reference
- EC2 (32 cores, 1TB RAM): 16 workers medical, 48 workers pharmacy; threads=1 each.
- Workstation (16 cores, 64GB): 12–24 workers total; threads=1 each.
- Laptop (8 cores, 32GB): 8–12 workers total; threads=1 each.

Best practices
- First-time extension setup (one-time): `INSTALL httpfs; INSTALL aws;` then `LOAD` thereafter.
- Subsequent runs: `LOAD httpfs; LOAD aws; CALL load_aws_credentials(); SET s3_region='us-east-1'; SET s3_url_style='path'`.
- Avoid invalid PRAGMA forms that can reset settings; prefer:
  - `PRAGMA enable_object_cache` (no `= true`).
  - Use `SET` only for documented parameters.

***

## 🧩 Partition-First Strategy

All pipeline stages have fully adopted a **partition-first architecture**.
This design divides the dataset into discrete units for independent processing:


| Stage | Partition Key | Implementation Status | Description |
| :-- | :-- | :-- | :-- |
| **APCD Input Processing** | age_band × event_year | ✅ Complete | Each worker handles one partition during medical/pharmacy cleaning. |
| **Global Imputation** | age_band | ✅ Complete | Each age band imputed independently using gold-tier linkages. |
| **Cohort Creation** | age_band × event_year | ✅ Complete | Each cohort-phase run isolates both age/event partitions to ensure reproducibility. |
| **Analysis \& Reporting** | cohort, analysis_type | ✅ Complete | Each analytics job references pre-partitioned cohort exports for scalability. |

### Operational Model

```
┌─────────────────────────────────────────────────────────────┐
│  Discovery: Identify all available partitions                │
│  ├── scan S3 for age_band/event_year combinations            │
│  └── enqueue work jobs                                       │
│                                                             │
│  Processing: Parallel workers per partition                 │
│  ├── 48 workers on 32-core EC2 (1.5× core utilization)      │
│  └── 1 thread, 2GB per worker                               │
│                                                             │
│  Aggregation: Merge and validate outputs                    │
│  └── Produce unified Parquet datasets per stage              │
└─────────────────────────────────────────────────────────────┘
```


***

## 🧠 Data Processing Details

### 1. APCD Input Data

## 📥 Defaults: Partitioned (imputed) Silver Inputs

To improve DuckDB performance and maximize parallelism, the orchestrator now prefers partitioned, pre-imputed Silver-tier inputs as the default entry point for optimized runs. In practice this means the pipeline will use paths like:

```
s3://pgxdatalake/silver/imputed/medical_partitioned/
s3://pgxdatalake/silver/imputed/pharmacy_partitioned/
```

Why this change?
- DuckDB performs best when work is split into many small, independent units (age_band × event_year). Reading partitioned Parquet files reduces memory pressure and enables many short-lived DuckDB instances to run concurrently.
- Partition-first inputs unlock full parallelism (48 workers × 1 thread each) which greatly reduces overall runtime and increases cloud efficiency.
- Global imputation already writes partitioned outputs (`global_imputation.py` saves `*_partitioned` directories under `silver/imputed/`); using those early avoids redundant transformation work.

Compatibility and behavior
- Backwards compatible: operators may still pass the legacy `--raw-medical s3://pgxdatalake/silver/medical/*.parquet` or `--raw-pharmacy` paths. The orchestrator will automatically resolve those to the imputed/partitioned equivalents when available.
- Validation: the input validation routine will attempt both the raw silver path and the imputed/partitioned path. This surfaces clear errors if neither exists.
- Override: to explicitly target a non-partitioned raw path, pass `--raw-medical`/`--raw-pharmacy` with the exact S3 URI you want. The orchestrator will use the value you provide.

Examples

- Run optimized medical cleaning (defaults use partitioned imputed input):

```bash
python3 1_apcd_input_data/3_apcd_clean.py --job medical --output-root s3://pgxdatalake/gold/medical --min-year 2016 --max-year 2020
```

- Force using a raw (non-partitioned) silver path (not recommended for large runs):

```bash
python3 1_apcd_input_data/3_apcd_clean.py --job medical --raw-medical s3://pgxdatalake/silver/medical/*.parquet --output-root s3://pgxdatalake/gold/medical
```

Validation helper
- A lightweight script `scripts/validate_silver_inputs.py` is available to preview which input paths will be used and to optionally check whether previous orchestrator logs exist. Use it in CI or as a preflight check.

Logging
- The orchestrator logs which path was used (imputed vs raw) during discovery and validation so operators can audit decisions in S3 logs.



### 2. Global Demographic Imputation

- Imputation executed by age_band partitions for cross-year consistency.
- Consolidated outputs serve as single lookup datasets across all dependent stages.


### 3. Event \& Cohort Creation

- Each partition generates event tables and 5:1 case-control cohorts.
- Checkpoints and logs stored under:
`s3://pgx-repository/pgx-pipeline-status/create_cohort/{entity_id}/`


### 4. Finalization \& QA

- All partitions merged to final S3 gold paths such as:

```
s3://pgxdatalake/gold/cohorts/opioid_ed/
    age_band=65-74/event_year=2019/opioid_ed_cohort.parquet
```

- Validation reports confirm ratio accuracy, exclusivity, and completeness.

***

## 📊 Resource Allocation Overview

| System | Workers | Threads | Memory/Worker | Total Memory | Core Utilization |
| :-- | :-- | :-- | :-- | :-- | :-- |
| EC2 (32 cores, 1TB) | 48 | 1 | 2GB | 96GB | 150% |
| Workstation (16 cores, 64GB) | 24 | 1 | 2GB | 48GB | 150% |
| Laptop (8 cores, 32GB) | 12 | 1 | 2GB | 24GB | 150% |


***

## 📈 Performance Outcomes

| Metric | Before Optimization | After Optimization | Improvement |
| :-- | :-- | :-- | :-- |
| Partition Coverage | 6.7% | 100% | 15× better |
| Parallelization Level | 3 workers | 48 workers | 16× higher |
| Total Processing Time | 45 batches | 1 batch | 15× faster |
| Memory Efficiency | 24GB | 96GB (distributed) | 4× utilization |
| Core Utilization | 9.4% | 150% | 16× increase |

The full production pipeline processes *45 partitions concurrently* on EC2 within a single pass, reducing total runtime from hours to under 30 minutes.

***

## 🧪 Checkpoints and Resilience

Every data phase incorporates the centralized checkpoint system:

- **Resumable execution:** Recover from failure mid-pipeline.
- **Per-partition metrics:** Record size, phase duration, and completion status.
- **Stored in S3:** Persistent JSON records for audit and progress tracking.

Example:

```json
{
  "pipeline": "create_cohort",
  "entity_id": "OPIOID_ED_65-74_2019",
  "phase": "phase3_cohort_creation",
  "status": "completed",
  "metrics": {"records": 250000, "ratio": "5.0:1"}
}
```


***

## 🧠 QA Standards

Each phase enforces:

- 100% demographics coverage
- No patient overlap between cohorts
- 5:1 control-to-target ratio
- Event classification verification
- Imputation completeness validation

Each QA outcome is logged and versioned alongside partition checkpoints.

***

## 🚧 Fault Tolerance \& Error Handling

- Automatic memory adjustments and retries (up to 3)
- Connection isolation — eliminates shared global states
- Graceful degradation: unaffected partitions continue
- Full cleanup on error (`cleanup_duckdb_temp_files()`)

### Multiprocessing Mode: Spawn vs Fork

**Default Configuration:**
- **Default:** `spawn` mode (more stable with many workers, lower memory usage)
- **Mapping Persistence:** Automatically enabled (`PGX_PERSIST_MAPPINGS=1`) when using spawn mode
- **Temp DB:** Enabled by default (`PGX_USE_TEMP_DB=1`) for better stability

**Spawn Mode (Default - Recommended for AWS Linux EC2):**
- **How it works:** Each worker starts as a fresh Python process, reimports the module
- **Memory:** Workers only get data explicitly passed as arguments (lower memory usage)
- **Mapping Persistence:** Automatically enabled - mappings are saved to temp files, workers load from disk (one copy on disk vs N copies in memory)
- **Startup:** Slower (reimports ~1-2 seconds per worker), but much more stable with high worker counts
- **Best for:** High worker counts (16+), AWS EC2 instances, memory-constrained environments, production stability
- **Platform:** Works on all platforms (Linux, Windows, macOS)

**Fork Mode (Optional - AWS Linux EC2 only):**
- **How it works:** Parent process clones itself using `fork()`, child gets copy-on-write snapshot of parent memory
- **Memory:** Each worker inherits a copy of all parent memory (high memory usage - can cause OOM)
- **Startup:** Very fast (no reimport, ~0.1 seconds per worker), but risky with many workers
- **Best for:** Low worker counts (<16), fast development iterations, when memory is abundant
- **Override:** Set `export PGX_MP_START_METHOD=fork` to use fork mode (not recommended for 28 workers)
- **Platform:** Linux only (including AWS Linux EC2)

**Memory Impact Example:**
```
Fork mode with 28 workers:
- 50MB mappings × 28 workers = 1.4GB just for mappings
- 8GB DuckDB memory × 28 workers = 224GB theoretical max
- Result: High risk of OOM crashes

Spawn mode with 28 workers:
- 50MB mappings saved once to disk
- Workers load from disk as needed
- Result: Much lower memory usage, more stable
```

**Configuration for AWS Linux EC2:**
```bash
# Default (spawn + persist mappings) - RECOMMENDED for 28 workers
# No configuration needed - spawn is now default even on Linux
# This avoids OOM crashes with high worker counts

# For AWS EC2 with 28 workers (your current setup):
# ✅ Use defaults (spawn mode) - no env vars needed
# ✅ PGX_PERSIST_MAPPINGS=1 is automatic (mappings saved to disk)
# ✅ PGX_USE_TEMP_DB=1 is default (disk-backed DuckDB)

# To use fork mode (NOT recommended for 28 workers - high OOM risk):
export PGX_MP_START_METHOD=fork
export PGX_PERSIST_MAPPINGS=0  # Not used in fork mode (mappings passed directly)

# To disable mapping persistence in spawn mode (not recommended):
export PGX_PERSIST_MAPPINGS=0
```

### DuckDB File Locking with Multiprocessing

**Issue:** When using `fork` multiprocessing mode, multiple workers may attempt to access the same DuckDB temp database file, causing lock conflicts:
```
IO Error: Could not set lock on file "/mnt/nvme/duckdb_tmp/worker_76519/duckdb_temp.db": 
Conflicting lock is held in /usr/local/bin/python3.11 (PID 80256)
```

**Solution:** Each DuckDB connection uses a unique temp database file when `use_temp_db=True`:
- Unique DB file per connection: `duckdb_temp_{uuid}.db` instead of shared `duckdb_temp.db`
- Unique worker temp directory: `worker_{pid}_{timestamp}_{uuid}` to ensure isolation
- Automatic cleanup: Temp DB files and WAL files are cleaned up on worker exit via `atexit`

**Best Practices:**
- Use `spawn` mode by default (no file locking issues, lower memory usage)
- `PGX_USE_TEMP_DB=1` is now default (reduces memory pressure)
- Each worker process gets its own isolated temp directory and unique DB files
- Temp directories are automatically cleaned up on process exit

***

## 🧮 Monitoring Metrics

**In-line reporting**:

```
→ Phase 1: 2.5M medical + 5.0M pharmacy records loaded
→ Phase 2: 7.5M events generated
→ Phase 3: Cohorts created (5:1 ratio validated)
→ Phase 4: Final parquet outputs written
```

**Dashboard metrics**:

- Completion % per partition
- CPU \& memory utilization
- Failed/retried partition counts
- Throughput per stage

***

## ✅ Best Practices Summary

1. **Partition Everything:** Every transformation must be scoped by `age_band` and `event_year`.
2. **One Connection, One Partition:** Avoid any shared DuckDB connections.
3. **Use Checkpoints:** Enables resumable execution and reduces rework.
4. **Limit Threads:** One thread per worker improves cloud I/O efficiency.
5. **Deploy Balanced Mode:** `workers = cores × 1.5` is ideal on EC2.
6. **Profile First Runs:** Use DuckDB profiling in JSON mode for optimization diagnostics.

***

## 📚 Linked Documentation

- `DuckDB_Dev_README.md` – DuckDB configuration, optimization, profiling, and troubleshooting (consolidated)
- `Create_Cohort_README.md` – Cohort pipeline logic and event schema
- `Cohort_Pipeline_Updates.md` – Latest modular phase updates
- `Pipeline_Optimization_README.md` – Core standards and resource rules

***

## 🏁 Final Notes

All PGx pipelines now operate under one unified **Partition-First, Modular, Checkpoint-Enabled** framework.
Every component — from data ingestion through cohort generation — runs as independent, fault-tolerant partition operations, fully integrated with DuckDB optimization and automatic S3 scaling.

**Version:** 4.4
**Status:** Production-Ready (All Partition Strategies Implemented)
**Last Updated:** November 9, 2025
**Maintainers:** PGx Data Engineering \& Analytics Team

---
<span style="display:none">[^1][^2][^3][^4][^5][^6][^7][^8][^9]</span>

<div align="center">⁂</div>

[^1]: https://estuary.dev/blog/data-pipeline-architecture/

[^2]: https://cloud.google.com/blog/topics/developers-practitioners/what-data-pipeline-architecture-should-i-use

[^3]: https://bitscopic.com/building-a-scalable-pgx-program-5-workflow-pitfalls-to-avoid/

[^4]: https://www.integrate.io/blog/guide-to-data-pipeline-architecture/

[^5]: https://www.boltic.io/blog/data-pipeline-architecture

[^6]: https://risingwave.com/blog/data-pipeline-architecture-building-blocks-diagrams-and-patterns/

[^7]: https://aampe.com/blog/data-pipeline-architecture-examples-best-practices

[^8]: https://github.com/jackc/pgx

[^9]: https://docs.oracle.com/en/industries/financial-services/ofs-analytical-applications/crime-compliance-studio-application-pack/8.1.2.9.0/csarg/architecture-guide.pdf

