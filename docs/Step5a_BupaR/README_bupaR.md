# BupaR Process Mining Documentation

## Overview
This document describes how to use the outputs of the FP-Growth cohort pipeline as event logs for BupaR process mining in R or Python.

---


## 1. Input Source: `model_data` + FP-Growth Itemsets

The main event source for BupaR is the **`model_data` parquet** filtered by cohort/age_band, with
FP-Growth TRAIN **target-only itemsets** used to select a high-signal activity alphabet.

- **Source parquet:**  
  `model_data/cohort_name={cohort_name}/age_band={age_band}/model_events.parquet`  
  (event-level rows with `mi_person_key`, `event_date`, `drug_name`, all 10 ICD diagnosis
  columns, `procedure_code`, `target`, `event_year`, etc.).
- **Itemset filters:**  
  `4_fpgrowth_analysis/outputs/{cohort_name}/target/{age_band_fname}/train/*_itemsets_target_only.json`
  (for `drug_name`, `icd_code`, and `medical_code`).

The BupaR notebooks/scripts build a **long-format event table** and then a BupaR `eventlog`:

- Activities:
  - `DRUG:<code>` from `drug_name`
  - `ICD:<code>` from **all 10 ICD diagnosis columns**
  - `CPT:<code>` from `procedure_code`
- Each row is one activity occurrence with:
  - `case_id = mi_person_key`
  - `activity = DRUG:/ICD:/CPT: + code`
  - `timestamp = event_date`

The activity alphabet is restricted to codes that appear in at least one FP-Growth **target-only**
itemset for the TRAIN window, plus the cohort-specific target ICD (e.g. `F1120` for opioid_ed).

---

## 2. Creating a BupaR Event Log (Conceptual)

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
cohort-specific pipeline (e.g. `bupaR_pipeline_opioid_ed_0_12.ipynb` and
`run_bupar_opioid_ed_0_12.R`) now produces:

### 3.1 Event Logs

Built directly from `model_data` + FP-Growth TRAIN itemsets:

- **Target-only event log**: `target_eventlog`
  - Activities: `DRUG:<code>`, `ICD:<code>`, `CPT:<code>`
  - Filtered to high-signal codes from FP-Growth target-only itemsets, plus the
    **cohort-specific target ICD** (e.g. `ICD:F1120` for opioid_ed).
- **Combined TARGET + CONTROL event log**: `sankey_eventlog`
  - Same activity alphabet, with an additional `group` attribute (`"target"` / `"control"`)
  - Used for Sankey-style process maps comparing flows between groups.
- **Pre-/post-target event logs (for opioid_ed/F1120)**:
  - `pre_target_eventlog`: events up to and including the first target ICD (e.g. `ICD:F1120`).
  - `post_target_eventlog`: events strictly after the first target ICD.
  - All **predictive features** are derived from **pre-target** events only; post-target
    views are for descriptive and QC purposes (avoid target leakage).

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

- **Target-only traces**:
  - `opioid_ed_0_12_train_target_traces_bupar.csv`
- **Target-only process matrix**:
  - `opioid_ed_0_12_train_target_process_matrix_bupar.csv`

In addition, the opioid_ed 0-12 notebook saves **PNG visualizations** under:

- `5_bupaR_analysis/outputs/opioid_ed/0_12/figures/`
  - `opioid_ed_0_12_train_target_traces_bupar.png`
  - `opioid_ed_0_12_train_target_process_matrix_bupar.png`

with the same files mirrored to S3 under `gold/bupar/opioid_ed/0-12/`. These provide quick
visual checks of the most common trajectories and transition hotspots.

### 3.3 Per-Patient Features for Tabular Datasets

From the pre-/post-target event logs we derive **patient-level sequence and timing features** and save
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

- **Time-to-F1120 and time-window features** (opiod_ed/F1120, pre-target only):
  - `opioid_ed_0_12_train_target_time_to_f1120_features_bupar.csv`
  - Columns (per `case_id` / `mi_person_key`):
    - `time_to_F1120_days` – days from the first observed event in the TRAIN window
      to the first `ICD:F1120` event.
    - `n_events_30d`, `n_events_90d`, `n_events_180d` – all events within 30/90/180 days
      before F1120.
    - `n_drug_events_30d/90d/180d` – DRUG activities in each window.
    - `n_icd_events_30d/90d/180d` – ICD activities in each window.
    - `n_cpt_events_30d/90d/180d` – CPT activities in each window.

These CSVs are intended to be **joined back into the tabular modeling dataset** on
`mi_person_key` and used as additional sequence-aware and **time-to-event** features.
In the current workflow, only **pre-target** and **time-window** features are used as
predictive inputs; **post-target** features and visualizations are used for descriptive
assessment of care pathways after the event of interest.

---

*This document is focused on BupaR process mining. For FP-Growth logic and outputs, see `README_fpgrowth.md`.*
