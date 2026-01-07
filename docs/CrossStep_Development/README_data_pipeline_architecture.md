
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

- Reads partitioned Silver-tier data.
- Performs demographic linkage and cleanup.
- Produces partitioned Gold-tier medical/pharmacy outputs.
- 48 concurrent workers process all partitions simultaneously, replacing old sequential runs.


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

## 📊 Final Pipeline: Inputs and Outputs

### Complete Data Flow with File Formats

The production pipeline (Steps 1-9) uses **Parquet as the preferred format** throughout, with CSV maintained for backward compatibility where needed.

#### Step-by-Step Data Flow

```
Step 1-2: Cohort Creation
  Input:  Raw APCD data (Silver tier)
  Output: model_events.parquet (Gold tier)
          Location: s3://pgxdatalake/gold/model_data/{cohort}/{age_band}/model_events.parquet

Step 3: Feature Importance
  Input:  model_events.parquet
  Output: aggregated_feature_importance.csv
          Location: 3_feature_importance/outputs/{cohort}/{age_band}/

Step 4a: Model Data Extraction
  Input:  model_events.parquet
  Output: model_events.parquet (filtered, target/control split)
          Location: 4a_model_data/{cohort}/{age_band}/model_events.parquet

Step 4b: DTW Protocol Filtering ⭐ OPTIMIZED
  Input:  model_events.parquet
  Output: model_events_no_protocols.parquet (Parquet, optimized DuckDB SQL)
          Location: 4b_dtw_filter/outputs/{cohort}/{age_band}/model_events_no_protocols.parquet
          S3: s3://pgxdatalake/gold/dtw_filter/{cohort}/{age_band}/model_events_no_protocols.parquet
  Idempotency Check: Checks for 3 S3 outputs:
    - model_events_no_protocols.parquet
    - protocol_summary_{cohort}_{age_band}.csv
    - event_intervals_{cohort}_{age_band}.parquet
  Optimization: Pure DuckDB SQL, COPY TO parquet, integer-based sequences

Step 5a-5d: Feature Engineering
  Input:  model_events_no_protocols.parquet (preferred)
  Output: Various feature tables (CSV/Parquet depending on step)
          - BupaR features
          - FP-Growth features
          - DTW trajectory features
          - PGx features

Step 6: Final Model Training ⭐ OPTIMIZED
  Input:  Feature tables from Steps 5a-5d
  Output: 
    - CSV: {cohort}_{age_band}_train_final_features_no_leakage.csv (backward compatibility)
    - Parquet: inputs/model_train/final_features.parquet (preferred) ⭐ NEW
    Location: 6_final_model/outputs/{cohort}/{age_band}/
    S3: s3://pgxdatalake/gold/final_model/{cohort}/{age_band}/inputs/model_train/final_features.parquet
  Model Artifacts:
    - final_model_json/{cohort}_{age_band}_best_xgboost_model.json
    - final_model_json/{cohort}_{age_band}_best_catboost_model.cbm
  Idempotency Check: Via checkpoint system (saves checkpoint after model training completes)

Step 7: SHAP Analysis ⭐ OPTIMIZED
  Input:  Final model artifacts + final_features.parquet
  Output:
    - Global importance: {cohort}_{age_band}_shap_global_importance_{model_type}.csv
      (for both xgboost and catboost)
    - Sample values: {cohort}_{age_band}_shap_sample_values_{model_type}.parquet ⭐ Parquet
      (for both xgboost and catboost)
    Location: 7_shap_analysis/outputs/{cohort}/{age_band}/
    S3: s3://pgxdatalake/gold/shap_analysis/{cohort}/{age_band}/
  Idempotency Check: Checks for XGBoost outputs (required); CatBoost outputs optional
  Optimization: Two-pass streamed approach, Parquet for row-level SHAP values

Step 8: FFA Analysis ⭐ OPTIMIZED
  Input:  
    - Final model JSON (XGBoost)
    - final_features.parquet (preferred) or CSV (fallback) ⭐ NEW
    - SHAP global importance CSV (both XGBoost and CatBoost)
    - SHAP sample values Parquet (both XGBoost and CatBoost)
  Output: All Parquet format ⭐ NEW
    - axp_explanations.parquet
    - feature_importance_axp.parquet
    - causal_importance.parquet
    - interaction_analysis.parquet
    Location: 8_ffa_analysis/outputs/{cohort}/{age_band}/{model_type}/
    S3: s3://pgxdatalake/gold/ffa_analysis/{cohort}/{age_band}/{model_type}/
  Idempotency Check: Checks for Parquet outputs per model_type:
    - axp_explanations.parquet (required)
    - feature_importance_axp.parquet (required)
    - causal_importance.parquet (optional)
    - interaction_analysis.parquet (optional)
  Optimization: DuckDB for data loading, Parquet outputs with Snappy compression

Step 9: Risk Dashboard
  Input:  Model artifacts + FFA outputs (Parquet)
  Output: Lambda-ready model packages + S3-hosted UI
```

### File Format Standards

| Step | Primary Format | Secondary Format | Notes |
| :-- | :-- | :-- | :-- |
| **1-2: Cohort Creation** | Parquet | N/A | All outputs Parquet |
| **3: Feature Importance** | CSV | N/A | Small files, CSV acceptable |
| **4a: Model Data** | Parquet | N/A | Event-level data |
| **4b: DTW Filter** | Parquet | N/A | Optimized DuckDB SQL |
| **5a-5d: Features** | Mixed | CSV/Parquet | Step-dependent |
| **6: Final Model** | **Parquet** ⭐ | CSV | **Parquet preferred, CSV for compatibility** |
| **7: SHAP** | **Parquet** ⭐ | CSV | **Row-level SHAP in Parquet** |
| **8: FFA** | **Parquet** ⭐ | N/A | **All outputs Parquet** |
| **9: Dashboard** | JSON/Parquet | N/A | Model artifacts |

### Parquet Optimization Benefits

**Storage Efficiency:**
- **10-100x smaller files** (Snappy compression)
- Reduced S3 storage costs
- Faster S3 upload/download

**Performance:**
- **2-5x faster I/O** operations
- Columnar format enables efficient column pruning
- Better type preservation (no CSV parsing issues)

**Compatibility:**
- Native DuckDB support (no conversion needed)
- Spark/Athena compatible
- Better for downstream analysis tools

### Key Optimization Points

1. **Step 4b (DTW Filter):**
   - Pure DuckDB SQL (no pandas `.apply()`)
   - Integer-based sequences (not strings)
   - Direct `COPY TO parquet` operations
   - **Idempotency:** Checks 3 S3 outputs before running

2. **Step 6 (Final Model):**
   - Saves both CSV and Parquet
   - Parquet location: `inputs/model_train/final_features.parquet`
   - Step 8 checks Parquet first, falls back to CSV
   - **Idempotency:** Via checkpoint system (saves checkpoint after completion)

3. **Step 7 (SHAP):**
   - Two-pass streamed approach
   - Global importance: CSV (small file)
   - Row-level values: Parquet (large file)
   - Generates outputs for both XGBoost and CatBoost
   - **Idempotency:** Checks for XGBoost outputs (required); CatBoost optional

4. **Step 8 (FFA):**
   - All outputs: Parquet with Snappy compression
   - Input loading: DuckDB for efficient Parquet/CSV reading
   - S3 paths: Updated to use `.parquet` extension
   - **Idempotency:** Checks for Parquet outputs per model_type (XGBoost required)

### S3 Storage Structure

```
s3://pgxdatalake/gold/
├── model_data/{cohort}/{age_band}/
│   └── model_events.parquet
├── dtw_filter/{cohort}/{age_band}/
│   └── model_events_no_protocols.parquet
├── final_model/{cohort}/{age_band}/
│   └── inputs/model_train/final_features.parquet
├── shap_analysis/{cohort}/{age_band}/
│   ├── {cohort}_{age_band}_shap_global_importance_xgboost.csv
│   ├── {cohort}_{age_band}_shap_sample_values_xgboost.parquet
│   ├── {cohort}_{age_band}_shap_global_importance_catboost.csv
│   └── {cohort}_{age_band}_shap_sample_values_catboost.parquet
└── ffa_analysis/{cohort}/{age_band}/{model_type}/
    ├── axp_explanations.parquet
    ├── feature_importance_axp.parquet
    ├── causal_importance.parquet
    └── interaction_analysis.parquet
```

### Backward Compatibility

- **CSV files maintained** where needed for legacy code
- **Parquet preferred** for all new operations
- **Automatic fallback** in Step 8 (checks Parquet first, then CSV)
- **Dual output** in Step 6 (both CSV and Parquet saved)

### Idempotency and Checkpoint System

All steps use a unified checkpoint/idempotency system via `py_helpers.checkpoint_utils`:

**Idempotency Flow:**
1. **Local File Checks:** Each step checks for local outputs first
2. **S3 Output Checks:** If local files missing, checks S3 for outputs
3. **Checkpoint Metadata:** Saves checkpoint JSON to S3 after completion
4. **File Format Awareness:** Idempotency checks match actual output formats

**Step-Specific Idempotency:**

| Step | Files Checked | Format | Required/Optional |
| :-- | :-- | :-- | :-- |
| **4b: DTW Filter** | 3 files | `.parquet`, `.csv` | All required |
| **6: Final Model** | Checkpoint only | N/A | Checkpoint metadata |
| **7: SHAP** | XGBoost outputs | `.csv`, `.parquet` | XGBoost required, CatBoost optional |
| **8: FFA** | Per model_type | `.parquet` | 2 required, 2 optional per model |

**Checkpoint Location:**
```
s3://pgx-repository/pipeline_checkpoints/{step_name}/{cohort}/{age_band}/checkpoint.json
```

**Note**: Checkpoints are stored in `pgx-repository` bucket, while outputs are stored in `pgxdatalake` bucket.

**Idempotency Behavior:**
- ✅ If outputs exist locally → Skip step, optionally upload to S3
- ✅ If outputs exist in S3 → Download to local, skip step
- ✅ If checkpoint exists → Skip step (even if some files missing)
- ✅ If nothing exists → Run step, upload outputs, save checkpoint

**File Format Matching:**
- **Step 4b:** Checks `.parquet` and `.csv` (matches actual outputs)
- **Step 7:** Checks `.parquet` for sample values, `.csv` for global importance (matches actual outputs)
- **Step 8:** Checks `.parquet` only (all outputs are Parquet)

**Verification:**
All documented S3 paths match the actual idempotency checks in the code:
- ✅ Step 4b: `s3://pgxdatalake/gold/dtw_filter/{cohort}/{age_band}/model_events_no_protocols.parquet`
- ✅ Step 7: `s3://pgxdatalake/gold/shap_analysis/{cohort}/{age_band}/*.parquet` and `*.csv`
- ✅ Step 8: `s3://pgxdatalake/gold/ffa_analysis/{cohort}/{age_band}/{model_type}/*.parquet`

## 📚 Linked Documentation

- `README_duckdb_optimization.md` – Engine configuration and EC2 tuning
- `README_create_cohort_pipeline.md` – Cohort pipeline logic and event schema
- `Cohort_Pipeline_Updates.md` – Latest modular phase updates
- `Pipeline_Optimization_README.md` – Core standards and resource rules
- `8_ffa_analysis/OPTIMIZATION_REVIEW.md` – FFA analysis optimization details

***

## 🏁 Final Notes

All PGx pipelines now operate under one unified **Partition-First, Modular, Checkpoint-Enabled** framework.
Every component — from data ingestion through cohort generation — runs as independent, fault-tolerant partition operations, fully integrated with DuckDB optimization and automatic S3 scaling.

**Parquet optimization is now standard** across Steps 4b, 6, 7, and 8, providing 10-100x storage reduction and 2-5x I/O performance improvements.

**Version:** 4.5
**Status:** Production-Ready (All Partition Strategies + Parquet Optimization Implemented)
**Last Updated:** January 7, 2026
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

