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
  - `9_risk_dashboard/visualizations/fpgrowth/extract_extreme_density_cohort.py`  
  - `9_risk_dashboard/visualizations/fpgrowth/summarize_extreme_density_cohort.py`

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

For `opioid_ed`, these summaries can be complemented by BupaR process mining visualizations in  
`9_risk_dashboard/visualizations/bupar/` for exploratory analysis (visualization-only, not used in models).

#### Z Code Analysis: Routine Examinations in Extreme vs Standard Cohorts

**Motivation**: Z codes (ICD-10 codes starting with "Z") represent "Factors influencing health status and contact with health services" and include routine health examinations, administrative encounters, and follow-up visits. These codes are often associated with preventive care and routine checkups rather than acute medical conditions. Understanding how Z codes differ between extreme and standard cohorts can help identify whether routine care patterns contribute to extreme-density patient profiles.

**Analysis Script**: `4b_dtw_filter/analyze_z_codes_in_cohorts.py`

**Usage**:
```bash
# Analyze Z codes for a specific cohort and age band
python 4b_dtw_filter/analyze_z_codes_in_cohorts.py --cohort opioid_ed --age-band 0-12

# Analyze all age bands for a cohort
python 4b_dtw_filter/analyze_z_codes_in_cohorts.py --cohort opioid_ed
```

**Key Z Code Categories**:
- **Z00**: General health examinations (routine checkups)
- **Z01**: Special examinations (eye, dental, blood pressure, etc.)
- **Z02**: Administrative examinations (pre-employment, insurance, driving license, etc.)
- **Z08/Z09**: Follow-up examinations after treatment
- **Z39**: Postpartum care and examination
- **Z51**: Encounters for aftercare and medical care

**Outputs** (per cohort/age_band):
- `4b_dtw_filter/outputs/z_code_analysis/{cohort}_{age_band}.csv` - Full Z code event data with time windows
- `4b_dtw_filter/outputs/z_code_analysis/z_code_summary_{cohort}_{age_band}.json` - Summary statistics

**Analysis Metrics**:
- **Event counts**: Total Z code events in standard vs extreme cohorts
- **Time windows**: Distribution of days from target event date for Z codes
- **Z code frequency**: Most common Z codes in each cohort type
- **Category distribution**: Breakdown by Z code category (Z00, Z01, Z02, etc.)

**Expected Findings**:
- **Early examinations (Z codes) should reduce extreme cohort size**: Routine examinations and administrative encounters are less likely to drive extreme-density patterns compared to acute medical conditions and complex care trajectories.
- **Time window differences**: Z codes in extreme cohorts may show different temporal patterns (e.g., more routine care before target events, or different follow-up patterns).
- **Administrative vs clinical**: Z02 codes (administrative examinations) are particularly likely to be filtered or show different patterns in extreme cohorts.

**Hypothesis Testing: Time Windows in Extreme vs Standard Cohorts**

**Hypothesis**: Extreme cohorts have larger time windows than standard cohorts.

**Analysis Script**: `4b_dtw_filter/analyze_z_code_time_windows.py`

**Findings** (based on available data):
- For `opioid_ed` / `55-64` (the only cohort/age_band with both standard and extreme data):
  - **Standard cohort**: Mean absolute time window = 194.0 days, Median = 117.0 days
  - **Extreme cohort**: Mean absolute time window = 338.2 days, Median = 281.5 days
  - **Difference**: Extreme cohort has **144.2 days larger** mean time window (164.5 days larger median)
  - **Result**: ✅ **Hypothesis supported** - Extreme cohorts show significantly larger time windows

**Interpretation**:
- Extreme-density patients have Z code events (routine examinations) that occur much further from the target event date compared to standard patients
- This suggests extreme-density patients may have:
  - More routine/preventive care spread over longer time periods
  - Different care patterns with events distributed across wider temporal windows
  - Potentially more administrative/routine encounters that could be filtered

**Integration with DTW Filtering**:
Z codes identified as administrative (particularly Z02 codes for administrative examinations) can be added to the administrative codes lookup table (`4b_dtw_filter/administrative_codes_lookup.json`) to filter them during DTW protocol filtering, further reducing the extreme cohort size and focusing on clinically meaningful events. The larger time windows in extreme cohorts suggest that filtering routine examinations may be particularly effective for reducing extreme-density patterns.

### How Step 4 Connects to Later Steps

- **Main cohorts** (with extreme-density patients removed) flow into:
  - PGx feature engineering (`5_pgx_analysis/`)
  - Final model training (`6_final_model_selection/`)
- **Extreme cohorts** are available for **exploratory visualization analysis** (FP-Growth, BupaR, DTW visualizations in `9_risk_dashboard/visualizations/`), but **do not feed the risk calculator models directly**.

