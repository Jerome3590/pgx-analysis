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

- **Source:** `4a_model_data/cohort_name={cohort}/age_band={age_band}/model_events.parquet`
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
