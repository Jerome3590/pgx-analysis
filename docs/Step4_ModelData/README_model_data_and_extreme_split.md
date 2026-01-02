## Step 4: Model Data, DTW Protocol Filter, and Extreme-Density Split

This step takes GOLD-tier cohort outputs and produces **model-ready event data** for each `(cohort, age_band)`, then applies **DTW-based protocol filtering** and a standardized **extreme-density split** so that downstream feature engineering and models run on a tractable, non-extreme base cohort.

### 4a – Model-Ready Event Extraction (`4a_model_data/`)

**Goal**: Build compact, analysis-ready `model_events.parquet` files for target and control cohorts.

- **Target (`opioid_ed`)**  
  - Reads aggregated MC‑CV feature importance and recovers raw item codes from `item_*` features.  
  - Filters GOLD event-level data (2016–2019) to rows where **any important item** appears in:
    - `drug_name`
    - ICD diagnosis columns 1–9
    - `procedure_code`  
  - Writes to:  
    - `4a_model_data/cohort_name=opioid_ed/age_band={band}/model_events.parquet`

- **Control (`non_opioid_ed`)**  
  - Loads full, unfiltered cohort events for the same age bands and years.  
  - Samples patients to maintain an approximate **5:1 control:target ratio**.  
  - Keeps **all events** per sampled control patient.  
  - Writes to:  
    - `4a_model_data/cohort_name=non_opioid_ed/age_band={band}/model_events.parquet`

These paired `model_events.parquet` files are the **canonical inputs** for FP-Growth, BupaR, DTW trajectories, PGx, and final models.

### 4b – DTW-Based Protocol Filtering (`4b_dtw_filter/`)

**Goal**: Remove protocol-like events (e.g., routine monitoring) that can dominate temporal structure without adding signal.

- **Script**: `4b_dtw_filter/filter_protocol_events.py`  
- **Input**: `4a_model_data/.../model_events.parquet`  
- **Output**:  
  - `4a_model_data/.../model_events_no_protocols.parquet` – preferred input for BupaR, FP-Growth, DTW trajectories, and extreme-cohort summaries.

DTW-derived time-window rules are used to flag and drop protocol patterns while preserving clinically meaningful variation.

### 4c – Extreme-Density Transaction Split (All Cohorts)

**Goal**: Move patients with extremely dense medical histories into a dedicated `{cohort}_extreme_density` cohort so they **do not drive the main models**, while still being available for exploratory analysis.

- **Scripts**:  
  - `5b_fpgrowth_analysis/extract_extreme_density_cohort.py`  
  - `5b_fpgrowth_analysis/summarize_extreme_density_cohort.py`

#### Extraction (`extract_extreme_density_cohort.py`)

- **Input**:  
  - `4a_model_data/cohort_name={cohort}/age_band={band}/model_events.parquet`
- **Method**:
  - Reconstructs **medical_code transactions** (all ICD positions + CPT) over TRAIN years (2016–2018).  
  - Uses the same `assign_transaction_density` logic as `cohort_fpgrowth.py` to compute per‑patient `transaction_size`.  
  - Bins patients into `low`, `medium`, `high`, and `extreme` density buckets (P25/P50/P75/P95 cut points).  
  - Flags all patients in the `extreme` bucket.
- **Outputs** (per `(cohort, age_band)`):
  - `4a_model_data/cohort_name={cohort}/age_band={band}/extreme_density_patients_{band_fname}.csv`  
  - `4a_model_data/cohort_name={cohort}_extreme_density/age_band={band}/model_events.parquet`  
  - In-place rewrite of base cohort:
    - `model_events_with_extreme.parquet` – backup including all patients  
    - `model_events.parquet` – **updated** with extreme patients removed

The updated `model_events.parquet` is what feeds **all main feature engineering and final models**; the `_extreme_density` cohort is reserved for exploratory FP-Growth, BupaR, and DTW analysis.

#### Summaries (`summarize_extreme_density_cohort.py`)

- **Preferred input** (when present):  
  - `4a_model_data/cohort_name={cohort}_extreme_density/age_band={band}/model_events_no_protocols.parquet`  
  - Else falls back to the extreme `model_events.parquet`.
- **Outputs**:
  - Patient-level summary CSV (events counts, transaction size, target flag)  
  - Drug / ICD / CPT frequency tables + PNG plots  
  - Transaction-size histogram PNG  
  - Aggregate JSON summary (`extreme_density_summary_{band_fname}.json`)

For `opioid_ed`, these summaries are complemented by BupaR process mining in  
`5a_bupaR_analysis/create_bupar_outputs_opioid_ed_extreme.R`, which builds an extreme-only eventlog and plots under `feature_engineering_outputs/5_bupar/opioid_ed_extreme_density/{age_band}/plots`.

### How Step 4 Connects to Later Steps

- **Main cohorts** (with extreme-density patients removed) flow into:
  - FP-Growth (`5b_fpgrowth_analysis/`)
  - BupaR (`5a_bupaR_analysis/`)
  - DTW trajectories (`5d_dtw_analysis/`)
  - PGx (`5c_pgx_analysis/`)
  - Final model (`6_final_model/`)
- **Extreme cohorts** mirror the same feature-engineering stack for **exploratory visualization and transition-risk marker discovery**, but **do not feed the risk calculator models directly**.

