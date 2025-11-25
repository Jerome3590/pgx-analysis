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


## 3. Output Layout for BupaR (Long Table)

- Each row: one drug event for a patient
- Columns: `mi_person_key`, `drug_name`, `timestamp`, plus any cohort or demographic columns
- **Best Practice:** Keep event log long; join to wide encoding table for drug features if needed.

---


## 4. Handoff from FP-Growth

- The FP-Growth cohort pipeline produces event logs in the required long format for BupaR.
- No further transformation is needed if columns match (`mi_person_key`, `drug_name`, `timestamp`).
- For drug features, join event log to wide encoding table on `drug_name`.

---

*This document is focused on BupaR process mining. For FP-Growth logic and outputs, see `FpGROWTH_README.md`.*
