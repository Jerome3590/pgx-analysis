

# Cohort Creation Pipeline - Comprehensive Guide

This document provides a **complete reference** for the Cohort Creation Pipeline, combining pipeline design, modular architecture, performance improvements, checkpointing, and usage instructions.

***

## 🎯 Overview

The Cohort Pipeline builds **event-based fact tables** for analytical cohorts used in clinical outcome and drug safety research. It generates two main cohorts:

- **OPIOID_ED:** Patients with opioid-related emergency department visits (targeted by ICD codes, e.g., F1120)
- **ED_NON_OPIOID:** Patients with non-opioid emergency department visits (targeted by HCG line codes)

Each cohort includes **target cases** and **5 matching controls** per case to ensure statistical robustness. The pipeline uses a **dual-target system**:
- **Target 1:** ICD/CPT codes (e.g., F1120 for opioid use disorder)
- **Target 2:** HCG-based ED visit identification (P51, O11, P33 line codes)

When partitions have zero targets, the pipeline creates **control-only cohorts** using pre-computed average target counts to ensure complete coverage for model training.

***

## 🏗️ Architecture and Modular Structure

Following the October 2025 refactor, the pipeline has been fully modularized into **4 clean phases** under `2_create_cohort/phases/`.

### Directory Structure

```
2_create_cohort/
├── create_cohort.py
├── pipeline_steps.py (legacy, deprecated)
└── phases/
    ├── __init__.py
    ├── common.py
    ├── phase1_data_preparation.py
    ├── phase2_event_processing.py
    ├── phase3_cohort_creation.py
    ├── phase4_finalization.py
    └── README.md
```


### Phase Summary

| Phase | File | Function | Description |
| :-- | :-- | :-- | :-- |
| Phase 1 | `phase1_data_preparation.py` | `run_phase1_data_preparation()` | Load and integrate medical + pharmacy data from APCD |
| Phase 2 | `phase2_event_processing.py` | `run_phase2_event_processing()` | Create unified event fact table and drug exposure |
| Phase 3 | `phase3_cohort_creation.py` | `run_phase3_step3_final_cohort_fact()` | Build final cohort fact table (target 5:1 control ratio, statistical independence, balanced temporal windows) |
| Phase 4 | `phase4_finalization.py` | `run_phase4_finalization()` | Validate QA and export to S3 |

**Key Benefits**

- Modular, testable, and maintainable
- Clear separation of concerns
- Backward-compatible imports
- No performance overhead

***

## ⚙️ Checkpoint and Resilience System

All pipeline phases now use the **centralized checkpoint system** to ensure job resilience and resumability.

### Checkpoint Features

- Step-level granularity (per-phase progress)
- Automatic resume after failure
- Metrics tracking (record counts, ratios, durations)
- Stored in S3 under `s3://pgx-repository/pgx-pipeline-status/create_cohort/{entity_id}/`

Example checkpoint JSON:

```json
{
  "pipeline_name": "create_cohort",
  "entity_id": "OPIOID_ED_65-74_2019",
  "status": "running",
  "steps": {
    "phase1_data_preparation": {
      "status": "completed",
      "metrics": {"medical_records": 1500000, "pharmacy_records": 850000}
    }
  }
}
```

**Classes:**

- `PipelineState` and `GlobalPipelineTracker` handle checkpoints, resuming, error logging, and status persistence.

***

## 🧩 DuckDB Configuration Enhancements

DuckDB handling has been fully aligned with the standardized system from the pharmacy pipeline:

- Uses `helpers/duckdb_utils.py`
- Corrects profiling commands (`PRAGMA disable_profiling`)
- Proper memory limit units (`16GB`, `900GB`)
- Full error propagation, no silent failures
- Configurable via command-line options or context settings

***

## 🧠 Event Fact Table Schema

### Core Identifiers

| Field | Description |
| :-- | :-- |
| `mi_person_key` | Patient ID |
| `event_date` | Event timestamp |
| `event_type` | 'medical' or 'pharmacy' |
| `data_source` | Originating data system |
| `age_band`, `event_year` | Stratification filters |

### Key Data Domains

- **Demographics:** imputed age, race, gender, payer type, and location
- **Medical Events:** ICD codes, CCS classification, provider and service metadata, **HCG fields** (hcg_setting, hcg_line, hcg_detail)
- **Pharmacy Events:** drug name, therapeutic class, and exposure timing
- **Cohort Metadata:** target/control indicator, cohort label, **cohort type** (OPIOID_ED, NON_OPIOID_ED, NON_ED), creation timestamp

### Cohort Classification Column

The `cohort` column tracks three types:
- **OPIOID_ED:** Target cases in opioid_ed_cohort (patients with opioid ICD codes in **ANY of the 10 ICD diagnosis columns**)
- **NON_OPIOID_ED:** Target cases in ed_non_opioid_cohort (HCG-based ED visits without opioid codes - checked across **ALL 10 ICD diagnosis columns**)
- **NON_ED:** Controls in both cohorts (non-ED visits)

### Temporal Fields and Drug Window Analysis

The pipeline includes temporal analysis fields that differ between cohorts:

#### Temporal Fields

| Field | Type | Description | OPIOID_ED | ED_NON_OPIOID |
| :-- | :-- | :-- | :-- | :-- |
| `first_opioid_ed_date` | STRING | Date of first opioid ED event (if any) | ✅ Populated | ❌ NULL |
| `first_ed_non_opioid_date` | STRING | Date of first non-opioid ED event (if any) | ❌ NULL | ✅ Populated |
| `days_to_target_event` | INTEGER | Days from event to first target event | ❌ NULL* | ✅ Calculated |
| `event_date` | STRING | Date of the event | ✅ All events | ✅ All events |
| `event_sequence` | INTEGER | Sequential order of events per patient | ✅ All events | ✅ All events |

\* For OPIOID_ED cohort, `days_to_target_event` is NULL. Users can calculate it from `event_date` and `first_opioid_ed_date` if needed.

#### Cohort-Specific Temporal Behavior

**OPIOID_ED Cohort:**
- **Complete Drug History:** Includes ALL drug events for target cases (no time restriction)
- **No Drug Window Filtering:** All pharmacy events are included regardless of timing
- **Temporal Analysis:** Use `event_date` and `event_sequence` for temporal analysis
- **First Target Date:** `first_opioid_ed_date` is populated for all patients with target events
- **Days Calculation:** `days_to_target_event` is NULL; calculate manually if needed:
  ```sql
  SELECT 
    event_date,
    first_opioid_ed_date,
    datediff(first_opioid_ed_date::DATE, event_date::DATE) as days_to_target
  FROM opioid_ed_cohort
  WHERE first_opioid_ed_date IS NOT NULL
  ```

**ED_NON_OPIOID Cohort:**
- **30-Day Lookback Window:** Applied to BOTH target cases AND controls for balanced comparison
- **Target Cases:** 
  - Reference date: First ED_NON_OPIOID event
  - Includes: Medical events OR drug events within 30 days before target
- **Controls:**
  - Reference date: First non-ED medical event (fallback to first medical event if none)
  - Includes: Medical events OR drug events within 30 days before reference date
  - **Balanced temporal windows** ensure fair comparison between targets and controls
- **First Target Date:** `first_ed_non_opioid_date` is populated for target cases only (NULL for controls)
- **Days Calculation:** `days_to_target_event` is pre-calculated for all events
  - For targets: Days to first ED_NON_OPIOID event
  - For controls: Days to reference date (first non-ED medical event)
  - Positive values: Event occurred before reference date (included in 30-day window)
  - Zero: Event occurred on reference date
  - Negative values: Event occurred after reference date (filtered out for drug events)

#### Drug Window Filtering Logic

For ED_NON_OPIOID cohort, the pipeline applies balanced temporal filtering to BOTH targets and controls:
```sql
-- Target cases: include medical events OR drug events within 30 days before target
-- Controls: include medical events OR drug events within 30 days before reference date
WHERE (
  (is_target_case = 1 AND (
    event_type = 'medical' 
    OR (event_type = 'pharmacy' 
        AND days_to_target_event IS NOT NULL 
        AND days_to_target_event >= 0 
        AND days_to_target_event <= 30)
  ))
  OR (is_target_case = 0 AND (
    event_type = 'medical'
    OR (event_type = 'pharmacy' 
        AND days_to_target_event IS NOT NULL 
        AND days_to_target_event >= 0 
        AND days_to_target_event <= 30)
  ))
)
```

This ensures:
- **Balanced Comparison:** Both targets and controls have the same temporal window structure
- **Causality Assessment:** Only drugs prescribed within 30 days before the reference event are considered
- **Risk Window Analysis:** Supports identification of high-risk drug exposure periods
- **Temporal Relationships:** Enables analysis of drug exposure timing relative to reference events
- **Statistical Validity:** Prevents bias from unequal temporal data between targets and controls

***

## 📈 Control Sampling: 5:1 Ratio

Control selection ensures matched demographics:

- Age, gender, and race matching
- Geographical alignment (ZIP/county)
- Payer-type consistency
- **Target cases:** No overlap (opioid patients excluded from ED_NON_OPIOID - **checked across ALL 10 ICD diagnosis columns**)
- **Controls:** Can be reused across cohorts (same control can appear in both OPIOID_ED and ED_NON_OPIOID)

### Statistical Independence

**Important:** Controls are sampled **without replacement WITHIN each cohort** to maintain statistical independence:
- **Within a cohort:** Each control patient appears only once (no reuse within OPIOID_ED or ED_NON_OPIOID)
- **Across cohorts:** Same controls CAN be reused between OPIOID_ED and ED_NON_OPIOID cohorts (they are independent studies)
- Should achieve 5:1 ratio unless partition (age_band + event_year) is genuinely small
- If fewer than 5:1 ratio is available, all available controls are used
- Warnings are logged when ratio falls below 5:1 (expected only for small partitions)
- This ensures valid statistical inference and prevents overfitting in ML models

### Control-Only Cohorts

When a partition has **zero target cases** (no F1120 codes or HCG ED visits), the pipeline creates a **control-only cohort**:
- Uses pre-computed average target count from all partitions
- Samples `avg_targets * 5` controls (maintains 5:1 structure)
- All records marked as `is_target_case = 0` and `target = 0`
- Ensures every partition has a cohort file for complete model training coverage
- Logs clearly indicate "CONTROL-ONLY" status

***

## 🚀 Execution Instructions

### Pre-Computation Step (Required First)

Before running the cohort pipeline, run the target frequency analysis script (which automatically pre-computes target averages):

```bash
# Analyze target codes and automatically pre-compute averages for cohort creation
python 1_apcd_input_data/6_target_frequency_analysis.py --profile bedrock
```

This creates `cohort_target_averages.json` in the project root, which Phase 3 uses for control-only cohort sizing. The pre-computation happens automatically as part of the target frequency analysis.

**Output:** `cohort_target_averages.json` containing:
- Average F1120 targets per partition
- Average HCG ED visit targets per partition
- Combined average (F1120 + HCG ED) for control-only cohorts
- Per-partition counts for reference

### SQL Workflow

```sql
\i phase1_data_preparation.sql
\i phase2_step1_event_fact_table.sql
\i phase2_step2_drug_exposure.sql
\i phase3_step3_final_cohort_fact.sql
\i phase4_complete_pipeline.sql
```


### Python Command-Line Interface

```bash
# Run with pre-computed averages (recommended)
python 0_create_cohort.py --age-band "65-74" --event-year 2016 --cohort both
```


### Advanced Usage

```bash
# With custom AWS profile
python precompute_target_averages.py --profile bedrock

# Pipeline with custom settings
python 0_create_cohort.py \
  --age-band "65-74" \
  --event-year 2019 \
  --cohort both \
  --threads 8 \
  --mem-gb 16 \
  --tmp-dir /tmp/duckdb_cohort
```


***

## 📊 S3 Output Structure

Cohorts are organized **by cohort name first, then by year and age-band partitions**:

```
s3://pgxdatalake/gold/cohorts_{TARGET_NAME}/
├── cohort_name=opioid_ed/
│   ├── event_year=2019/
│   │   ├── age_band=45-54/
│   │   │   └── cohort.parquet
│   │   ├── age_band=55-64/
│   │   │   └── cohort.parquet
│   │   └── ...
│   ├── event_year=2020/
│   │   └── ...
│   └── ...
└── cohort_name=ed_non_opioid/
    ├── event_year=2019/
    │   ├── age_band=45-54/
    │   │   └── cohort.parquet
    │   └── ...
    └── ...
```

**Example:** `s3://pgxdatalake/gold/cohorts_F1120/cohort_name=opioid_ed/event_year=2019/age_band=45-54/cohort.parquet`

**Path Structure:** `cohorts_{TARGET_NAME}/cohort_name={cohort}/event_year={year}/age_band={age_band}/cohort.parquet`

**Note:** All cohorts are saved (including control-only cohorts) to ensure complete coverage. Control-only cohorts are clearly logged with "CONTROL-ONLY" status.


***

## 🧪 QA and Validation Checks

- 100% imputed demographics
- Cohort exclusivity for targets (opioid patients excluded from ED_NON_OPIOID **checked across ALL 10 ICD diagnosis columns**), controls can overlap
- 5:1 control ratio (or control-only cohorts when targets = 0)
- Event classification integrity
- **Dual-target validation:** F1120 ICD codes (checked in **ALL 10 ICD diagnosis columns**) and HCG ED visits
- **HCG field presence:** hcg_setting, hcg_line, hcg_detail loaded from gold tier
- **Cohort column values:** OPIOID_ED, NON_OPIOID_ED, NON_ED
- **Comprehensive ICD checking:** All opioid code validations check all 10 ICD diagnosis columns
- QA summary logged in checkpoints

Example QA log:

```
→ Phase 1 QA: Medical: 2.5M, Pharmacy: 5.0M
→ Phase 2 QA: Event fact table created (7.5M events)
  F1120 records: 36,931 (640 distinct patients)
  HCG ED visits: 125,000 (8,500 distinct patients)
→ Phase 3 QA: Ratio 5.0:1 confirmed
  OPIOID_ED: 640 targets, 3,200 controls
  ED_NON_OPIOID: 8,500 targets, 42,500 controls
→ Phase 4 QA: Pipeline complete
  OPIOID_ED cohort saved (CONTROL-ONLY) to S3  [if zero targets]
```


***

## ⚡ Performance Metrics

| Metric | Original | Optimized | Gain |
| :-- | :-- | :-- | :-- |
| Steps | 15 | 5 | 67% fewer |
| Execution Time | 45–60 min | 20–30 min | 50% faster |
| Memory | High | Medium | 40% less |
| Duplication | High | Low | 80% reduced |

Runs efficiently on 8–16 GB memory and supports 10M+ events per cohort.

***

## 👩‍🔬 Testing and Debugging

### Test a Sample Run

```bash
python create_cohort.py --age-band "65-74" --event-year 2019 --cohort OPIOID_ED
```


### Verify Checkpoints

```python
from helpers.pipeline_state import PipelineState
state = PipelineState("create_cohort", "OPIOID_ED_65-74_2019", logger)
print(state.get_progress())
```


***

## ✅ Best Practices

**Pre-computation:**

- Target averages are automatically computed by `6_target_frequency_analysis.py` (run this before cohort creation)
- Re-run if gold tier data changes significantly
- Check `cohort_target_averages.json` exists before batch runs

**When adding phases:**

- Create `phases/phaseN_<description>.py`
- Use `common.py` utilities
- Add to `__init__.py`
- Include checkpoint and error handling
- Document updates in `phases/README.md`

**When editing:**

- Keep each phase self-contained
- Use logging and checkpointing
- Update documentation and unit tests
- Ensure HCG fields are included in medical data loading
- Maintain dual-target classification logic (ICD codes + HCG ED visits)

**Control-Only Cohorts:**

- Model training code should filter by `is_target_case = 1` if only targets are needed
- Control-only cohorts can be used as negative-only examples
- Consider excluding from training or weighting differently in loss function

***

## 📚 Related References

- `Cohort_Creation_SQL.md` — **Complete SQL reference** for all cohort creation queries
- `S3_PATHS.md` — Checkpoints and run summaries (state + paths)
- `DuckDB_Dev_README.md` — Database performance tuning
- `1_apcd_input_data/Preprocessing_README.md` — Pre-imputation overview
- `phases/README.md` — Phase-level logic reference
- `1_apcd_input_data/6_target_frequency_analysis.py` — Target frequency analysis (includes automatic pre-computation of cohort target averages)
- `control_only_cohort_analysis.md` — Detailed analysis of control-only cohort strategy

***

## 🎯 Target Identification System

### Dual-Target Architecture

The pipeline uses two independent target identification methods:

1. **ICD/CPT Code Targets** (OPIOID_ED cohort):
   - Primary: F1120 (Opioid Use Disorder, Uncomplicated)
   - Configurable via environment variables (see below)
   - Codes normalized to F1120 format in gold tier (no dots, no spaces)
   - **Comprehensive checking:** All 10 ICD diagnosis columns are checked (primary through ten), not just `primary_icd_diagnosis_code`

2. **HCG-Based ED Visit Targets** (ED_NON_OPIOID cohort):
   - Uses Healthcare Cost Group (HCG) line codes:
     - `P51 - ER Visits and Observation Care`
     - `O11 - Emergency Room`
     - `P33 - Urgent Care Visits`
   - Identifies ED visits regardless of diagnosis codes
   - Always classified as `'ed_non_opioid'` in event classification
   - **Opioid exclusion:** All opioid patients are excluded by checking ALL 10 ICD diagnosis columns

### Classification Priority

Event classification follows this priority:
1. **Target ICD/CPT codes** → `'target'` (or `'opioid_ed'` if no dynamic targeting) - **Checks ALL 10 ICD diagnosis columns**
2. **HCG ED visits** → `'ed_non_opioid'`
3. **Other events** → `'non_target'` (or `'ed_non_opioid'` if default mode)

**Important:** The implementation uses a helper function `get_opioid_icd_sql_condition()` from `helpers_1997_13/constants.py` to generate comprehensive SQL that checks all 10 ICD diagnosis columns (`primary_icd_diagnosis_code`, `two_icd_diagnosis_code`, ..., `ten_icd_diagnosis_code`). This ensures no opioid-related events are missed regardless of which diagnosis position the code appears in.

### Environment Variables for Dynamic Targeting

The pipeline supports dynamic target selection via environment variables:

| Variable | Description | Example |
| :-- | :-- | :-- |
| `PGX_TARGET_NAME` | Human-readable target name | `F1120` |
| `PGX_TARGET_ICD_CODES` | Comma-separated ICD codes | `F1120,F1121` |
| `PGX_TARGET_CPT_CODES` | Comma-separated CPT codes | `99281,99282` |
| `PGX_TARGET_ICD_PREFIXES` | Comma-separated ICD prefixes | `F11,F12` |
| `PGX_TARGET_CPT_PREFIXES` | Comma-separated CPT prefixes | `9928` |

**Usage Examples:**

```bash
# Set F1120 as target
export PGX_TARGET_NAME="F1120"
export PGX_TARGET_ICD_CODES="F1120"

# Or use command-line arguments
python 0_create_cohort.py \
  --age-band "65-74" \
  --event-year 2019 \
  --target-name "F1120" \
  --target-icd-codes "F1120"
```

**Note:** When environment variables are set, the pipeline uses generic `'target'`/`'non_target'` classification labels. When unset, it defaults to `'opioid_ed'`/`'ed_non_opioid'` classification.

### DuckDB Parallelization Configuration

The pipeline supports parallelization via environment variables:

| Variable | Description | Default |
| :-- | :-- | :-- |
| `PGX_THREADS_PER_WORKER` | Number of DuckDB threads for query execution | `8` |
| `PGX_S3_UPLOADER_THREAD_LIMIT` | Maximum uploader threads for S3 multi-part uploads | DuckDB default |
| `PGX_S3_UPLOADER_MAX_FILESIZE` | Max file size for part size calculation (e.g., "5368709120" for 5GB) | DuckDB default |
| `PGX_S3_UPLOADER_MAX_PARTS_PER_FILE` | Max parts per file for part size calculation | DuckDB default |

**Important:** `s3_max_connections` is **not** a valid DuckDB configuration parameter and will cause errors. S3 parallelization is handled automatically by DuckDB. Use `s3_uploader_thread_limit` if you need to tune upload performance.

**Example:**
```bash
export PGX_THREADS_PER_WORKER=16
export PGX_S3_UPLOADER_THREAD_LIMIT=16
python 0_create_cohort.py --age-band "65-74" --event-year 2019 --operation-type s3_heavy
```

## 🏁 Summary

The **Cohort Creation Pipeline v4.3+** now features:
- **Modular, checkpoint-enabled architecture** with 4 clean phases
- **Dual-target system** (ICD codes + HCG ED visits) for comprehensive cohort identification
- **Comprehensive ICD diagnosis checking** across all 10 ICD diagnosis columns (primary through ten) to ensure no opioid patients are missed or misclassified
- **Control-only cohort logic** ensuring complete partition coverage for model training
- **Pre-computed averages** for efficient control-only cohort sizing
- **HCG field integration** (hcg_setting, hcg_line, hcg_detail) from gold tier
- **Cohort classification column** (OPIOID_ED, NON_OPIOID_ED, NON_ED) for flexible filtering
- **High-performance DuckDB integration** with optimized memory and query handling

The pipeline achieves improved testability, maintainability, and resilience—while reducing runtime and resource usage by over 50%.

**Last Updated:** 2025-11-15
**Version:** 4.3 (Dual-Target + Control-Only Cohorts + HCG Integration + Comprehensive ICD Diagnosis Checking)
**Status:** Production-Ready
**Authors:** PGx Analytics Engineering Team

---
<span style="display:none">[^1][^2][^3]</span>

<div align="center">⁂</div>

[^1]: Cohort_Pipeline_README.md

[^2]: Cohort_Modularization_README.md

[^3]: Cohort_Pipeline_Updates.md

