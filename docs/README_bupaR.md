# BupaR Process Mining Documentation

## Overview
This document describes how to use the outputs of the FP-Growth cohort pipeline as event logs for BupaR process mining in R or Python.

---


## 1. Input Format from FP-Growth (Long Table)

The main input is an event log table (long format):

| mi_person_key | drug_name      | timestamp   | ...optional columns... |
|---------------|---------------|-------------|-----------------------|
| 12345         | ACETAMINOPHEN  | 2020-01-01  | ...                   |
| 12345         | IBUPROFEN      | 2020-01-02  | ...                   |

- **Source:** `fpgrowth_features/` (partitioned by cohort, age_band, event_year)
- **How to use:** This table is the direct input to BupaR for process mining and sequence analysis.
- **Best Practice:** Join to the wide encoding table if you need drug features in the event log.

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

- **In Python (pm4py):**
```python
import pandas as pd
from pm4py.objects.log.util import dataframe_utils
from pm4py.objects.conversion.log import converter as log_converter

df = pd.read_csv("cohort_event_log.csv")
df = dataframe_utils.convert_timestamp_columns_in_df(df)
event_log = log_converter.apply(df)
```

---


## 3. Outputs and Features from BupaR

For each `(cohort_name, age_band, event_window)` (e.g. `opioid_ed`, `0-12`, `train`), the
notebook `5_bupaR_analysis/bupaR_pipeline.ipynb` now produces:

### 3.1 Event Logs

- **Target-only event log**: `target_eventlog`
  - Activities: `DRUG:<code>`, `ICD:<code>`, `CPT:<code>`
  - Filtered to high-signal codes from FP-Growth target-only itemsets, plus `ICD:F1120`.
- **Combined TARGET + CONTROL event log**: `sankey_eventlog`
  - Same activity alphabet, with an additional `group` attribute (`\"target\"` / `\"control\"`)
  - Used for Sankey-style process maps comparing flows between groups.
- **Pre-/Post-F1120 event logs**:
  - `pre_F1120_eventlog`: events up to and including first `ICD:F1120` per case.
  - `post_F1120_eventlog`: events strictly after first `ICD:F1120`.

### 3.2 Aggregate BupaR Outputs (Traces and Process Matrices)

Saved locally under:

```text
5_bupaR_analysis/outputs/{cohort_name}/{age_band_fname}/
```

and to S3 under:

```text
s3://pgxdatalake/gold/bupar/{cohort_name}/{age_band}/
```

For example, for cohort 1 (`opioid_ed`), age `0-12`, TRAIN window:

- **Target-only traces (pre-FP-Growth)**:
  - `opioid_ed_0_12_train_target_traces_bupar.csv`
- **Target-only process matrix**:
  - `opioid_ed_0_12_train_target_process_matrix_bupar.csv`
- **Combined TARGET + CONTROL process matrix (for Sankey)**:
  - `opioid_ed_0_12_train_combined_process_matrix_bupar.csv`

### 3.3 Per-Patient Features for Tabular Datasets

From the pre-/post-F1120 event logs we derive **patient-level sequence features** and save
them with the same naming conventions:

- **Pre-F1120 features** (events up to and including first `ICD:F1120`):
  - `opioid_ed_0_12_train_target_pre_f1120_patient_features_bupar.csv`
  - Columns (per `case_id` / `mi_person_key`):
    - `pre_n_events` – total events before/at F1120
    - `pre_n_drug_events` – number of `DRUG:` activities
    - `pre_n_icd_events` – number of `ICD:` activities
    - `pre_n_cpt_events` – number of `CPT:` activities
    - `pre_n_unique_activities` – distinct activities before/at F1120

- **Post-F1120 features** (events after first `ICD:F1120`):
  - `opioid_ed_0_12_train_target_post_f1120_patient_features_bupar.csv`
  - Columns (per `case_id` / `mi_person_key`):
    - `post_n_events` – total events after F1120
    - `post_n_drug_events` – number of `DRUG:` activities
    - `post_n_icd_events` – number of `ICD:` activities
    - `post_n_cpt_events` – number of `CPT:` activities
    - `post_n_unique_activities` – distinct activities after F1120

These CSVs are intended to be **joined back into the tabular modeling dataset** on
`mi_person_key` and used as additional sequence-aware features.

---

*This document is focused on BupaR process mining. For FP-Growth logic and outputs, see `README_fpgrowth.md`.*
