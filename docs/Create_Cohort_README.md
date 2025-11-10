

# Cohort Creation Pipeline - Comprehensive Guide

This document provides a **complete reference** for the Cohort Creation Pipeline, combining pipeline design, modular architecture, performance improvements, checkpointing, and usage instructions.

***

## 🎯 Overview

The Cohort Pipeline builds **event-based fact tables** for analytical cohorts used in clinical outcome and drug safety research. It generates two main cohorts:

- **OPIOID_ED:** Patients with opioid-related emergency department visits
- **ED_NON_OPIOID:** Patients with non-opioid emergency department visits

Each cohort includes **target cases** and **5 matching controls** per case to ensure statistical robustness.

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
| Phase 3 | `phase3_cohort_creation.py` | `run_phase3_cohort_creation()` | Build final cohort fact table (5:1 control ratio) |
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
- **Medical Events:** ICD codes, CCS classification, provider and service metadata
- **Pharmacy Events:** drug name, therapeutic class, and exposure timing
- **Cohort Metadata:** target/control indicator, cohort label, creation timestamp

***

## 📈 Control Sampling: 5:1 Ratio

Control selection ensures matched demographics:

- Age, gender, and race matching
- Geographical alignment (ZIP/county)
- Payer-type consistency
- No patient overlap across cohorts

***

## 🚀 Execution Instructions

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
python create_cohort.py --age-band "65-74" --event-year 2016 --cohort both
python create_cohort.py --age-band "75-84" --event-year 2017 --cohort opioid_ed
```


### Advanced Usage

```
python create_cohort.py \
  --threads 8 --mem-gb 16 --tmp-dir /tmp/duckdb_cohort \
  --skip-checkpoints
```


***

## 📊 S3 Output Structure

```
s3://pgxdatalake/gold/cohorts/
├── opioid_ed/age_band={age_band}/event_year={event_year}/
│   └── opioid_ed_cohort.parquet
└── ed_non_opioid/age_band={age_band}/event_year={event_year}/
    └── ed_non_opioid_cohort.parquet
```


***

## 🧪 QA and Validation Checks

- 100% imputed demographics
- Cohort exclusivity (no overlap)
- 5:1 control ratio
- Event classification integrity
- QA summary logged in checkpoints

Example QA log:

```
→ Phase 1 QA: Medical: 2.5M, Pharmacy: 5.0M
→ Phase 2 QA: Event fact table created (7.5M events)
→ Phase 3 QA: Ratio 5.0:1 confirmed
→ Phase 4 QA: Pipeline complete
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

***

## 📚 Related References

- `S3_PATHS.md` — Checkpoints and run summaries (state + paths)
- `DuckDB_Dev_README.md` — Database performance tuning
- `1_apcd_input_data/Preprocessing_README.md` — Pre-imputation overview
- `phases/README.md` — Phase-level logic reference

***

## 🏁 Summary

The **Cohort Creation Pipeline v4.0+** now features a modular, checkpoint-enabled, high-performance architecture, unified with the APCD gold data system. It achieves improved testability, maintainability, and resilience—while reducing runtime and resource usage by over 50%.

**Last Updated:** 2025-10-24
**Version:** 4.1 (Modular + Checkpoint Integration)
**Status:** Production-Ready
**Authors:** PGx Analytics Engineering Team

---
<span style="display:none">[^1][^2][^3]</span>

<div align="center">⁂</div>

[^1]: Cohort_Pipeline_README.md

[^2]: Cohort_Modularization_README.md

[^3]: Cohort_Pipeline_Updates.md

