# Pipeline State Management System

A comprehensive checkpoint and state tracking system for managing complex data pipelines across S3.

## 📍 **Checkpoint Location**

All pipeline checkpoints are stored at:
```
s3://pgx-repository/pgx-pipeline-status/
```

## 🪣 Dual-Bucket Architecture: Status vs Analytics Aggregation

We separate operational status/checkpoints from analytics-oriented run summaries:

- Status and checkpoints (authoritative, resume/retry):
  - Bucket: `pgx-repository`
  - Prefix: `pgx-pipeline-status/<pipeline>/<entity>/...`
  - Written by: `PipelineState` and `GlobalPipelineTracker`

- Aggregated summaries (for BI, dashboards, and ad-hoc analysis):
  - Bucket: `pgxdatalake`
  - Prefix: `pgx_pipeline/<script_name>/run_id=<run_id>/summary.json`
  - Written by: each script at the end of execution

### Transaction Codes (tx) per Script

Each aggregated summary includes a short transaction code identifying the script:

- txt_to_parquet → `txtpq`
- reprocess_txt_to_parquet → `repro`
- global_imputation → `glimp`
- clean_pharmacy → `clnph`
- clean_medical → `clnmd`

Example S3 keys for aggregated summaries:

```text
s3://pgxdatalake/pgx_pipeline/txt_to_parquet/run_id=20251025-013045/summary.json
s3://pgxdatalake/pgx_pipeline/reprocess_txt_to_parquet/run_id=20251025-021159/summary.json
s3://pgxdatalake/pgx_pipeline/global_imputation/run_id=20251025-024512/summary.json
s3://pgxdatalake/pgx_pipeline/clean_pharmacy/run_id=20251025-031830/summary.json
s3://pgxdatalake/pgx_pipeline/clean_medical/run_id=20251025-033401/summary.json
```

Minimal JSON payload (common fields across scripts):

```json
{
  "tx": "txtpq",
  "run_id": "20251025-013045",
  "start_time": "2025-10-25T01:30:45Z",
  "end_time": "2025-10-25T01:52:10Z",
  "duration_sec": 1285.4,
  "status": "success",
  "status_code": 0,
  "totals": { "planned": 120, "converted": 118, "skipped": 2, "errors": 0 },
  "datasets": [ { "dataset": "medical", "planned": 60, "converted": 59, "skipped": 1, "errors": 0 } ]
}
```

Notes:
- Scripts may include additional fields (e.g., `output_path`, `final_count`, `appended`).
- Status field is `success` or `error`; `status_code` is `0` on success, non-zero on error.
- `run_id` is ISO-like `YYYYMMDD-HHMMSS` per execution.

## 🗂️ **File Structure**

```
s3://pgx-repository/pgx-pipeline-status/
├── pharmacy_clean/
│   ├── global_tracker.json                    # Global progress across all age_band/year
│   ├── 55_64_2018/
│   │   ├── state.json                         # Overall pipeline state
│   │   ├── checkpoints/
│   │   │   ├── load_pharmacy_data.json
│   │   │   ├── normalize_data.json
│   │   │   ├── deduplication.json
│   │   │   └── write_output.json
│   │   └── failures/
│   │       └── normalize_data.json            # If step failed
│   └── 65_74_2020/
│       ├── state.json
│       └── checkpoints/...
├── create_cohort/
│   ├── global_tracker.json
│   ├── opioid_ed_65_74_2020/
│   │   ├── state.json
│   │   ├── checkpoints/
│   │   │   ├── load_medical.json
│   │   │   ├── load_pharmacy.json
│   │   │   ├── identify_cases.json
│   │   │   └── create_controls.json
│   │   └── failures/
│   └── ed_non_opioid_55_64_2019/
│       └── ...
└── global_imputation/
    ├── global_tracker.json
    └── pharmacy_medical_imputation/
        └── state.json
```

## 🚀 **Quick Start**

### **1. Basic Usage in Your Pipeline**

```python
from helpers.pipeline_state import PipelineState

def build_optimized_pipeline(age_band, event_year, pharmacy_input, output_root, 
                            conn, logger, log_buffer):
    # Initialize state tracker
    entity_id = f"{age_band}/{event_year}"
    state = PipelineState('pharmacy_clean', entity_id, logger)
    
    # Step 1: Load data
    if not state.is_step_completed('load_pharmacy_data'):
        logger.info("📊 Step 1: Loading pharmacy data...")
        # ... your work here ...
        state.mark_step_completed('load_pharmacy_data', {
            'rows': initial_count,
            'patients': initial_patients
        })
    
    # Step 2: Data normalization  
    if not state.is_step_completed('normalize_data'):
        try:
            logger.info("🔧 Step 2: Normalizing data...")
            # ... your work here ...
            state.mark_step_completed('normalize_data', {
                'rows_before': before_count,
                'rows_after': after_count
            })
        except Exception as e:
            state.mark_step_failed('normalize_data', str(e))
            raise
    
    # Final: Mark entire pipeline complete
    state.mark_pipeline_completed({
        'final_rows': final_count,
        'output_path': output_path
    })
```

### **2. Check if Output Already Exists (Skip Entire Pipeline)**

```python
from helpers.pipeline_state import PipelineState

def build_optimized_pipeline(age_band, event_year, ...):
    entity_id = f"{age_band}/{event_year}"
    state = PipelineState('pharmacy_clean', entity_id, logger)
    
    # Define expected output location
    output_path = f"s3://pgxdatalake/gold/pharmacy/age_band={age_band.replace('-','_')}/event_year={event_year}/data.parquet"
    
    # Ultimate check: if output exists, skip everything
    if PipelineState.check_output_exists(output_path):
        logger.info("✅ Output already exists, skipping entire pipeline")
        state.mark_pipeline_completed({'output': output_path, 'skipped': True})
        return
    
    # Otherwise, proceed with pipeline...
```

## 📊 **Features**

### ✅ **Automatic Resume After Failure**

The system automatically detects completed steps and skips them:

```python
# Run 1: Pipeline fails at step 3
state.mark_step_completed('step1')  # ✅ Saved to S3
state.mark_step_completed('step2')  # ✅ Saved to S3
# Step 3 fails... ❌

# Run 2: Automatic resume
if not state.is_step_completed('step1'):  # ⏭️ Skipped (found checkpoint)
    ...
if not state.is_step_completed('step2'):  # ⏭️ Skipped (found checkpoint)
    ...
if not state.is_step_completed('step3'):  # ▶️ Runs (no checkpoint found)
    ...
```

### ✅ **Individual Step Checkpoints**

Each step gets its own checkpoint file for granular tracking:

```bash
# Check specific step status
aws s3 ls s3://pgx-repository/pgx-pipeline-status/pharmacy_clean/55_64_2020/checkpoints/

# Output:
# normalize_data.json
# deduplication.json
# write_output.json
```

### ✅ **Failure Tracking**

Failed steps are tracked separately for debugging:

```bash
aws s3 ls s3://pgx-repository/pgx-pipeline-status/pharmacy_clean/55_64_2020/failures/

# Output:
# normalize_data.json
```

View failure details:
```bash
aws s3 cp s3://pgx-repository/pgx-pipeline-status/pharmacy_clean/55_64_2020/failures/normalize_data.json -

# Output:
{
  "pipeline_name": "pharmacy_clean",
  "entity_id": "55_64_2020",
  "step_name": "normalize_data",
  "failed_at": "2025-10-15T10:30:45.123456",
  "status": "failed",
  "error": "Parser Error: Unknown unit for memory_limit"
}
```

## 🔍 **Monitoring Pipeline Progress**

### **Check Overall State**

```bash
# View pipeline state for specific entity
aws s3 cp s3://pgx-repository/pgx-pipeline-status/pharmacy_clean/55_64_2020/state.json -
```

Output:
```json
{
  "pipeline_name": "pharmacy_clean",
  "entity_id": "55_64_2020",
  "created_at": "2025-10-15T10:00:00.000000",
  "updated_at": "2025-10-15T10:35:00.000000",
  "status": "completed",
  "completed_steps": [
    {
      "step_name": "load_pharmacy_data",
      "completed_at": "2025-10-15T10:05:00.000000",
      "metadata": {"rows": 1000000}
    },
    {
      "step_name": "normalize_data",
      "completed_at": "2025-10-15T10:15:00.000000",
      "metadata": {"rows_before": 1000000, "rows_after": 950000}
    }
  ],
  "failed_steps": [],
  "metadata": {}
}
```

### **Global Progress Tracking**

```python
from helpers.pipeline_state import GlobalPipelineTracker

tracker = GlobalPipelineTracker('pharmacy_clean', logger)
summary = tracker.get_summary()

print(summary)
# Output:
# {
#   'pipeline_name': 'pharmacy_clean',
#   'total_entities': 25,
#   'pending': 0,
#   'running': 3,
#   'completed': 20,
#   'failed': 2,
#   'entities': ['55_64_2018', '55_64_2019', ...]
# }
```

## 📋 **Best Practices**

### **1. Always Check Output First**

```python
# GOOD: Check if final output exists before any work
output_path = "s3://bucket/output/data.parquet"
if PipelineState.check_output_exists(output_path):
    state.mark_pipeline_completed({'output': output_path, 'skipped': True})
    return

# Then proceed with steps...
```

### **2. Wrap Steps in Try/Except**

```python
# GOOD: Capture failures for debugging
if not state.is_step_completed('normalize_data'):
    try:
        # ... do work ...
        state.mark_step_completed('normalize_data', metadata)
    except Exception as e:
        state.mark_step_failed('normalize_data', str(e))
        raise  # Re-raise to stop pipeline
```

### **3. Include Useful Metadata**

```python
# GOOD: Track meaningful metrics
state.mark_step_completed('normalize_data', {
    'rows_before': 1000000,
    'rows_after': 950000,
    'data_loss_pct': 5.0,
    'duration_seconds': 120
})

# BAD: No metadata
state.mark_step_completed('normalize_data')
```

### **4. Use Descriptive Step Names**

```python
# GOOD: Clear, specific names
state.mark_step_completed('load_pharmacy_data')
state.mark_step_completed('normalize_incurred_dates')
state.mark_step_completed('deduplicate_mi_person_key')

# BAD: Vague names
state.mark_step_completed('step1')
state.mark_step_completed('process_data')
```

## 🔧 **Utility Commands**

### **View All Checkpoints for Entity**

```bash
aws s3 ls s3://pgx-repository/pgx-pipeline-status/pharmacy_clean/55_64_2020/checkpoints/ --recursive
```

### **Clear Checkpoints (Force Re-run)**

```bash
# Remove all checkpoints for specific entity
aws s3 rm s3://pgx-repository/pgx-pipeline-status/pharmacy_clean/55_64_2020/ --recursive

# Or in Python
state = PipelineState('pharmacy_clean', '55-64/2020', logger)
state.reset()
```

### **Check Global Progress**

```bash
aws s3 cp s3://pgx-repository/pgx-pipeline-status/pharmacy_clean/global_tracker.json -
```

## 🎯 **Integration with Existing Code**

### **Update Your `clean_pharmacy.py`**

```python
def build_optimized_pipeline(age_band, event_year, pharmacy_input, 
                            demographics_lookup, output_root, conn, logger, log_buffer,
                            resume: bool = True):
    
    # Add at the beginning
    from helpers.pipeline_state import PipelineState
    entity_id = f"{age_band}/{event_year}"
    state = PipelineState('pharmacy_clean', entity_id, logger)
    
    # Check if output exists
    output_path = f"{output_root}/age_band={age_band.replace('-', '_')}/event_year={event_year}/data.parquet"
    if resume and PipelineState.check_output_exists(output_path):
        logger.info("✅ Output exists, skipping")
        state.mark_pipeline_completed({'output': output_path, 'skipped': True})
        return
    
    # Wrap each existing step
    if not state.is_step_completed('load_pharmacy_data'):
        logger.info("📊 Step 1: Loading pharmacy data...")
        # ... existing code ...
        state.mark_step_completed('load_pharmacy_data', {'initial_count': initial_count})
    
    # ... more steps ...
    
    # At the end
    state.mark_pipeline_completed({'final_count': final_count, 'output': output_path})
```

## 🐛 **Troubleshooting**

### **Issue: Steps not being skipped**

**Check:**
```bash
# Verify checkpoints exist
aws s3 ls s3://pgx-repository/pgx-pipeline-status/pharmacy_clean/55_64_2020/checkpoints/
```

**Solution:** Ensure `state.mark_step_completed()` is being called after successful completion.

### **Issue: Pipeline stuck in "running" status**

**Check:**
```bash
aws s3 cp s3://pgx-repository/pgx-pipeline-status/pharmacy_clean/55_64_2020/state.json -
```

**Solution:** Either the pipeline is still running, or it crashed without calling `mark_pipeline_completed()`. Check logs.

### **Issue: Want to re-run specific step**

**Solution:**
```bash
# Remove specific step checkpoint
aws s3 rm s3://pgx-repository/pgx-pipeline-status/pharmacy_clean/55_64_2020/checkpoints/normalize_data.json

# Pipeline will re-run this step next time
```

## 📚 **Related Documentation**

- **DuckDB Configuration**: `README_duckdb_optimization.md`
- **Logging Utils**: `helpers/logging_utils.py`
- **S3 Utils**: `helpers/s3_utils.py`

---

## 🎉 **Summary**

The Pipeline State Management System provides:

✅ **Automatic resume** after failures  
✅ **Granular step tracking** with individual checkpoints  
✅ **Global progress monitoring** across all entities  
✅ **Failure debugging** with detailed error logs  
✅ **S3-based persistence** for reliability  
✅ **Simple API** that's easy to integrate  

All checkpoints are saved to: **`s3://pgx-repository/pgx-pipeline-status/`**

