# BupaR Process Mining Documentation

## Research questions this visual answers

- **What care pathways do patients follow in the run-up to the target event?** Process matrices and activity-frequency plots show the most common sequences of diagnoses, procedures, and drugs *before* the anchor (e.g. first opioid-ED encounter or first non-opioid ED encounter).
- **Which activities cluster before vs after the target, and how often?** Pre- vs post-target activity frequency and trace explorer views reveal which codes dominate the “on-ramp” vs the “after” period, by cohort and age band.
- **How do high-risk and low-risk patients differ in their care sequences?** By restricting to **feature-important** codes only, we reduce noise and focus on what the model uses to predict the target—so the visual reflects what is actually driving our target cohorts.

**Feature importance drives this visual.** We use only **SHAP/FFA important** codes (drug, ICD, CPT) as allowed activities. Events whose codes are not in that set are excluded. That keeps process mining focused on the signals that matter for the risk model and makes pathways interpretable.

### How features are filtered and used downstream

- The pipeline writes **allowed codes** (from Step 3b cohort feature importance) to `allowed_codes_shap_ffa_{cohort}_{age_band}.json` (see [../README.md#how-features-are-filtered-by-feature-importance-and-used-downstream](../README.md#how-features-are-filtered-by-feature-importance-and-used-downstream)).
- BupaR **R scripts** read this JSON and filter the event log: only activities (drug, ICD, CPT) that appear in the allowed set are kept; all other events are dropped before building process matrices, activity frequency, and trace explorer.
- Downstream, only these filtered pathways are visualized, so the dashboard shows care sequences that reflect what is driving the target cohorts.

---

## Testing one age band locally

To run BupaR for a **single** cohort/age band (e.g. to verify script changes):

1. **Model data** must exist for that cohort and age band. The script looks for parquet under:
   - `4_model_data/cohort_name=opioid_ed/age_band=0_12/` (or `age_band=0-12`)
   - or `PGX_DATA_ROOT/4_model_data/...` if set
   - or `4a_model_data/...` as fallback  
   Run `4_model_data` (or equivalent) for the desired cohort/age band first if needed.

2. **From repo root**, run the Python driver with `--local-test` (skips SHAP/FFA allowed codes and S3 upload):

   ```bash
   python 9_dashboard_visuals/bupar/create_bupar_visuals.py --cohort-name opioid_ed --age-band 0-12 --force --local-test
   ```

   For **non_opioid_ed** (default age band in R is 65-74):

   ```bash
   python 9_dashboard_visuals/bupar/create_bupar_visuals.py --cohort-name non_opioid_ed --age-band 65-74 --force --local-test
   ```

3. **R only** (no Python): if the allowed-codes file already exists (or you want R to use all codes), run from repo root:

   ```bash
   Rscript 9_dashboard_visuals/bupar/create_bupar_outputs_opioid_ed.R 0-12
   # or
   Rscript 9_dashboard_visuals/bupar/create_bupar_outputs_non_opioid_ed.R 65-74
   ```

   The R script reads `10_risk_dashboard/visualizations/bupar/outputs/allowed_codes_shap_ffa_{cohort}_{age}.json` if present; if missing or empty, it uses all codes in the data.

Outputs go to `10_risk_dashboard/visualizations/bupar/outputs/{cohort}/{age_band_fname}/plots/` and `.../features/`.

---

## Overview
This document describes how to use the outputs of the FP-Growth cohort pipeline as event logs for BupaR process mining in R. All scripts in this directory are R-based for consistency and to enable execution in a single R Jupyter notebook kernel.

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
| `bupaR_added_features_{cohort}_{age_band}.csv` | **Final merged BupaR features for dashboard visualization only (not added to model data)** | ✅ Yes | `add_bupar_features_to_model_data.R` |

**S3 Locations:**
- Sequence features: `s3://pgxdatalake/gold/feature_engineering/5_bupar/{cohort}/{age_band}/sequence_features_{cohort}_{age_band}.csv`
- Final merged features: `s3://pgxdatalake/gold/feature_engineering/5_bupar/{cohort}/{age_band}/bupaR_added_features_{cohort}_{age_band}.csv`

**Format:** CSV with `mi_person_key`; used by dashboard visuals only. We do not add BupaR (or DTW or FP-Growth) features to model data.

**Workflow (All R-based for consistency):**
1. R script (`create_bupar_outputs_opioid_ed.R`) generates bupaR outputs (pre/post/time features, traces)
2. `create_sequence_features.R` creates sequence features from top/rare traces → saves `sequence_features_{cohort}_{age_band}.csv`
3. `add_bupar_features_to_model_data.R` merges all features (pre/post/time + sequence) → saves `bupaR_added_features_{cohort}_{age_band}.csv` (dashboard only; not added to model data)

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

**Note:** This workflow uses R exclusively. For Python-based process mining, consider using `pm4py`, but note that bupaR is R-only and provides the most comprehensive process mining capabilities for this analysis.

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
