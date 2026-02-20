# BupaR Process Mining Documentation

## Overview
This document describes how to use model events data for BupaR process mining in R. All scripts in this directory are R-based for consistency and to enable execution in a single R Jupyter notebook kernel. The BupaR analysis uses `model_events.parquet` directly without any preprocessing filtering.

---

## Output Files Manifest

### Expected Outputs Structure

For each `(cohort, age_band)` combination, the following files should be generated:

#### Data Files (`outputs/{cohort}/{age_band}/features/`)

| File Pattern | Description | Required |
|--------------|-------------|----------|
| `{cohort}_{age_band}_train_target_pre_f1120_patient_features_bupar.csv` | Pre-F1120 per-patient features | ✅ Yes |
| `{cohort}_{age_band}_train_target_post_f1120_patient_features_bupar.csv` | Post-F1120 per-patient features | ✅ Yes |
| `{cohort}_{age_band}_train_target_time_to_f1120_features_bupar.csv` | Time-to-F1120 features | ✅ Yes |
| `{cohort}_{age_band}_train_target_traces_bupar.csv` | All trace sequences | ✅ Yes |
| `{cohort}_{age_band}_train_target_traces_top_bupar.csv` | Top (frequent) sequences | ✅ Yes |
| `{cohort}_{age_band}_train_target_traces_rare_bupar.csv` | Rare (unique) sequences | ✅ Yes |
| `{cohort}_{age_band}_train_target_pre_f1120_traces_top_bupar.csv` | Pre-F1120 top sequences | ⚠️ Conditional |
| `{cohort}_{age_band}_train_target_pre_f1120_traces_rare_bupar.csv` | Pre-F1120 rare sequences | ⚠️ Conditional |
| `{cohort}_{age_band}_train_target_post_f1120_traces_top_bupar.csv` | Post-F1120 top sequences | ⚠️ Conditional |
| `{cohort}_{age_band}_train_target_post_f1120_traces_rare_bupar.csv` | Post-F1120 rare sequences | ⚠️ Conditional |
| `{cohort}_{age_band}_train_target_process_matrix_bupar.csv` | Process flow matrix | ⚠️ Optional |

#### Feature Engineering Files (`outputs/feature_engineering/`)

| File Pattern | Description | Required | Created By |
|--------------|-------------|----------|------------|
| `sequence_features_{cohort}_{age_band}.csv` | Sequence features (top/rare indicators) | ✅ Yes | `create_sequence_features.R` |
| `bupaR_added_features_{cohort}_{age_band}.csv` | **Final merged bupaR features ready for model training** | ✅ Yes | `add_bupar_features_to_model_data.R` |

**S3 Locations:**
- Sequence features: `s3://pgxdatalake/gold/feature_engineering/5_bupar/{cohort}/{age_band}/sequence_features_{cohort}_{age_band}.csv`
- Final merged features: `s3://pgxdatalake/gold/feature_engineering/5_bupar/{cohort}/{age_band}/bupaR_added_features_{cohort}_{age_band}.csv`

**Format:** CSV with `mi_person_key` column for joining with `model_data` in final model step.

**Workflow (All R-based for consistency):**
1. R script (`create_bupar_outputs_opioid_ed.R`) generates bupaR outputs (pre/post/time features, traces)
2. `create_sequence_features.R` creates sequence features from top/rare traces → saves `sequence_features_{cohort}_{age_band}.csv`
3. `add_bupar_features_to_model_data.R` merges all features (pre/post/time + sequence) → saves `bupaR_added_features_{cohort}_{age_band}.csv`

**Note:** All scripts in this directory are R-based to ensure consistency and enable execution in a single R Jupyter notebook kernel without switching between languages.

**Example Files:**
- `outputs/opioid_ed/0_12/eventlog_target.csv`
- `outputs/opioid_ed/0_12/eventlog_sankey.csv`
- `outputs/opioid_ed/0_12/eventlog_pre_target.csv`
- `outputs/opioid_ed/0_12/process_features.csv`
- `outputs/opioid_ed/0_12/trace_statistics.csv`

#### Visualization Files (`outputs/plots/`)

| File Pattern | Description | Required |
|--------------|-------------|----------|
| `{cohort}_{age_band}_process_map.png` | Process flow diagram | ✅ Yes |
| `{cohort}_{age_band}_sankey_diagram.png` | Sankey flow diagram (target vs control) | ✅ Yes |
| `{cohort}_{age_band}_trace_frequency.png` | Most frequent traces | ⚠️ Optional |
| `{cohort}_{age_band}_throughput_time.png` | Throughput time distribution | ⚠️ Optional |
| `{cohort}_{age_band}_pre_post_comparison.png` | Pre/post target comparison (opioid_ed) | ⚠️ Conditional |

**Example Files:**
- `outputs/plots/opioid_ed_0_12_process_map.png`
- `outputs/plots/opioid_ed_0_12_sankey_diagram.png`
- `outputs/plots/opioid_ed_0_12_pre_post_comparison.png`

### Completion Checklist

For each cohort/age-band combination:

- [ ] Target event log created
- [ ] Sankey event log created (target + control)
- [ ] Pre/post target logs created (if applicable)
- [ ] Process features extracted
- [ ] Trace statistics computed
- [ ] Process flow visualizations generated
- [ ] Sankey diagrams generated
- [ ] Files uploaded to S3 (if applicable)

---


## 1. Input Format from Model Events (Parquet)

The main input is an event log table (long format) from `model_events.parquet`:

| mi_person_key | activity       | event_date   | ...optional columns... |
|---------------|---------------|-------------|-----------------------|
| 12345         | DRUG:ACETAMINOPHEN  | 2020-01-01  | ...                   |
| 12345         | DRUG:IBUPROFEN      | 2020-01-02  | ...                   |
| 12345         | ICD:F1120           | 2020-01-15  | ...                   |
| 12345         | CPT:80307           | 2020-01-20  | ...                   |

- **Source (Step 3b):** Built by `create_bupar_input_from_cohort.py` from cohort + 3a FI. Path: `3b_feature_importance_eda/outputs/cohort_name={cohort}/age_band={age_band}/model_events.parquet` (synced to S3 `gold/cohorts_model_data/cohort_name={cohort}/age_band={age_band}/`). No 4_model_data (that is created after target leakage removal).
- **Format:** Parquet file with event-level data including ICD codes, CPT codes, and drugs
- **How to use:** This table is the direct input to BupaR for process mining and sequence analysis.
- **Activity Format:** Activities are prefixed with type (e.g., `DRUG:`, `ICD:`, `CPT:`) for easy categorization

---

## 2. Creating a BupaR Event Log

- **In R:**
```r
library(bupaR)
eventlog <- read.csv("cohort_event_log.csv")
eventlog <- eventlog(
  case_id = "mi_person_key",
  activity_id = "drug_name",
  timestamp = "timestamp"
)
```

**Note:** This workflow uses R exclusively. For Python-based process mining, consider using `pm4py`, but note that bupaR is R-only and provides the most comprehensive process mining capabilities for this analysis.

---


## 3. Output Layout for BupaR (Long Table)

- Each row: one drug event for a patient
- Columns: `mi_person_key`, `drug_name`, `timestamp`, plus any cohort or demographic columns
- **Best Practice:** Keep event log long; join to wide encoding table for drug features if needed.

---


## 4. Data Source

- The `model_events.parquet` file contains all events (ICD codes, CPT codes, drugs) in long format.
- The R scripts (`create_bupar_outputs_*.R`) read this parquet file directly using DuckDB.
- Events are transformed into BupaR event log format with activities prefixed by type (DRUG:, ICD:, CPT:).
- No preprocessing filtering is applied - all events from `model_events.parquet` are used.

---

## 5. DuckDB usage and optimization

DuckDB is used throughout the BupaR workflow for speed and to keep heavy work out of R:

| Step | Where | How DuckDB helps |
|------|--------|-------------------|
| **Input build** | `create_bupar_input_from_cohort.py` → `4_model_data/create_model_data.py` | All event filtering, case/control union, and parquet write are done in DuckDB (no pandas for event-level data). |
| **R: load target** | `create_bupar_outputs_*.R` | Single read: one DuckDB query does `WHERE target=1` and UNPIVOT (wide→long) so the parquet file is scanned once instead of twice. |
| **R: control cohort** | `control_cohort_utils.R` | Counts, ratio checks, and control sampling use DuckDB over parquet. |
| **R: connection** | Both R scripts | Connect with `dbConnect(duckdb::duckdb())` (no `config` on driver). Optionally set threads after connect via `dbExecute(con, "SET threads = N")` when `DUCKDB_THREADS` is set (e.g. `4` or `8`). See `docs/CrossStep_Development/README_duckdb_optimization.md` § R (BupaR) DuckDB connection. |
| **Post-target analysis** | `create_bupar_post_target_analysis.py` | Pre/post target analytics and feature tables are computed in DuckDB via SQL over `model_events.parquet`. |

**Pre/post target split in DuckDB:**
- **opioid_ed:** One DuckDB query returns long-form events plus `first_target_date` (first F1120 per patient via a CTE). R filters `event_date < first_target_date` for pre-F1120 and `event_date > first_target_date` for post-F1120, so the split no longer uses event indices in R.
- **non_opioid_ed:** A small DuckDB query returns `(mi_person_key, first_target_date)` using `first_ed_non_opioid_date` or `hcg_line`; that is joined to the long table and used for the same pre/post split.

**Control cohort path alignment:**
- Target and control `model_events` paths use the same layout: `outputs/cohort_name={cohort}/age_band={age_band}/model_events.parquet` and S3 `gold/cohorts_model_data/cohort_name={cohort}/...`. Legacy `cohorts/input_model_data/...` remains as a fallback.

**Tips:**
- **Faster runs:** Set `DUCKDB_THREADS=8` (or your core count) in the environment before running the R scripts.
- **Less memory in R:** The pipeline already avoids loading the full wide table into R; only the long-form event table and BupaR objects are in R memory.
- **Upstream:** Building `model_events.parquet` (Step 3b input) is already DuckDB-only in `4_model_data`; no extra optimization needed there for BupaR.
