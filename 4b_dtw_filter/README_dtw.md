## DTW Protocol Filtering (Step 4b)

This directory contains scripts for **DTW protocol filtering** - the first step in the DTW workflow that identifies and filters protocol-like events from model data.

## Background: Research-First Approach to Trajectory Analysis

### Primary Goal: Capture All Trajectories First, Then Research

The DTW workflow follows a **research-first approach** where **all research happens in this `dtw_filter` step (Step 4b)**:

1. **Step 1: Capture ALL Trajectories** (Research Phase - **Happens Here**)
   - Extract **all** time windows and common sequences of events from patient data
   - No filtering at this stage - we need complete trajectory data to understand patterns
   - Identify time intervals between events, sequence patterns, and trajectory characteristics
   - Goal: Build a comprehensive understanding of patient healthcare journeys

2. **Step 2: Research & Classify** (Analysis Phase - **Happens Here**)
   - Analyze captured trajectories to distinguish:
     - **Clinical/Useful sequences**: Patterns that are meaningful for prediction
     - **Non-clinical/Protocol sequences**: Routine care patterns that don't add predictive value
   - Research what patterns are predictive vs. what represents standard care protocols
   - Identify which trajectory characteristics correlate with outcomes
   - **Critical**: This research must happen here because filtering occurs before feature extraction

3. **Step 3: Filter Non-Clinical Patterns** (Filtering Phase - **Happens Here**)
   - Use DTW time window analysis to identify protocol-like events
   - Filter out events that occur too close together (< 7 days) - these often represent standard care protocols
   - Output: `model_events_no_protocols.parquet` - cleaned data for downstream feature engineering
   - **Purpose**: Remove noise from routine care that both targets and controls follow

4. **Step 4: Extract Clinical Features** (`5d_dtw_analysis` - Step 5d)
   - Use filtered data to extract trajectory features that are clinically meaningful
   - Calculate DTW distances to prototype trajectories
   - Extract trajectory characteristics (length, diversity, temporal patterns)
   - **Purpose**: Keep what's good - features that capture predictive patterns from the cleaned data

### Why Research Happens in DTW Filter

**Critical Insight**: `dtw_filter` (Step 4b) runs **before** `dtw_analysis` (Step 5d). This means:

- **All research must happen here**: Since filtering occurs first, we must identify what to filter vs. what to keep during this step
- **Capture everything first**: We need to analyze all trajectories and time windows to make informed filtering decisions
- **Research-driven filtering**: Filtering decisions should be based on analysis of trajectory patterns, not arbitrary thresholds
- **Clinical validation**: We need to research which trajectories are clinically meaningful vs. routine care that both targets and controls follow

### Why This Approach?

- **Complete Understanding**: We need to see all trajectories before deciding what to filter
- **Evidence-Based Filtering**: Research informs which patterns are protocol vs. predictive
- **Iterative Refinement**: As we learn more about trajectories, we can refine filtering thresholds
- **Clinical Validation**: Allows clinical review of trajectory patterns before filtering

### Workflow Summary

```
Model Data (4a)
    ↓
[Capture ALL Trajectories] ← Research Phase (4b)
    ↓
[Research & Classify] ← Analysis Phase (4b)
    ├─→ Clinical/Useful Sequences → Keep for DTW Analysis (5d) → Features
    └─→ Non-Clinical/Protocol Sequences → Filter Out (4b) → Remove
    ↓
Filtered Data (model_events_no_protocols.parquet)
    ↓
DTW Analysis (5d) → Extract Features from Cleaned Data
```

**Key Principle**: Get all trajectories first, then research what goes where (filter vs. feature). **All research happens in `dtw_filter` (Step 4b) because it runs first.**

### Key Scripts

- `create_predictive_time_features.py`  
  - Creates **non-leaky** time-window features from `model_events.parquet`.  
  - For each patient, computes intervals (in days) between consecutive:
    - Drug events (`drug_interval_*`)
    - ICD events (`icd_interval_*`)
    - CPT events (`cpt_interval_*`)  
  - Outputs `predictive_time_features_{cohort}_{age_band}.csv` under `6_dtw_analysis/outputs/feature_engineering/` with columns:
    - `mi_person_key`
    - `drug_interval_count`, `drug_interval_mean`, `drug_interval_median`, `drug_interval_std`, `drug_interval_min`, `drug_interval_max`
    - analogous sets for ICD and CPT.

- `create_dtw_features.py`  
  - Builds patient-level **DTW trajectory features** using FP-Growth itemsets to select an activity alphabet.  
  - Steps:
    1. Loads FP-Growth target-only itemsets from `4_fpgrowth_analysis/outputs/{cohort}/{split_type}/{age_band_fname}/{event_year}/` and derives an allowed code set.
    2. Derives per-patient trajectories from `4a_model_data/.../model_events.parquet` (drug/ICD/CPT activities), excluding F1120 from the final trajectories.
    3. Selects prototype trajectories and computes DTW distances from each patient to each prototype.
    4. Produces distance-based and trajectory-shape features:
       - `dtw_distance_to_prototype_k`, `dtw_min_distance`, `dtw_max_distance`, `dtw_mean_distance`, `dtw_std_distance`
       - `trajectory_length`, `trajectory_diversity`  
  - Writes `dtw_features_{cohort}_{age_band}.csv` under `6_dtw_analysis/outputs/feature_engineering/` and (optionally) to S3:
    - `s3://pgxdatalake/gold/feature_engineering/6_dtw/{cohort}/{age_band}/dtw_features_{cohort}_{age_band}.csv`

- `add_dtw_features_to_model_data.py`  
  - Final aggregation step for DTW features.  
  - Reads `dtw_features_{cohort}_{age_band}.csv` and writes:
    - `dtw_added_features_{cohort}_{age_band}.csv` under `6_dtw_analysis/outputs/feature_engineering/`  
    - Optional upload to:
      - `s3://pgxdatalake/gold/feature_engineering/6_dtw/{cohort}/{age_band}/dtw_added_features_{cohort}_{age_band}.csv`
  - Output is ready to join to model-level data via `mi_person_key`.

- `run_analysis.py`  
  - Orchestration wrapper for a full DTW feature run.  
  - Workflow:
    1. (Optional) Run `create_predictive_time_features.py`.
    2. Run `create_dtw_features.py` for the specified cohort/age band.
    3. Run `add_dtw_features_to_model_data.py` to produce the final merged DTW feature file.

### Inputs

- `4a_model_data/cohort_name={cohort}/age_band={age_band}/model_events.parquet`  
  - Required for **all** scripts in this directory.
- `4_fpgrowth_analysis/outputs/{cohort}/{split_type}/{age_band_fname}/{event_year}/*_itemsets*.json`  
  - Used by `create_dtw_features.py` to restrict trajectories to items that passed FP-Growth screening.

### Outputs

All DTW-related feature files are written under:

- `6_dtw_analysis/outputs/feature_engineering/`
  - `predictive_time_features_{cohort}_{age_band}.csv`
  - `dtw_features_{cohort}_{age_band}.csv`
  - `dtw_added_features_{cohort}_{age_band}.csv`

Each file includes `mi_person_key` for joining with other feature blocks and the final modeling table.

### Typical Workflow

From the project root:

```bash
python 6_dtw_analysis/run_analysis.py --cohort-name opioid_ed --age-band 0-12
```

This will:

1. Create predictive time-window features (unless `--skip-predictive-features` is passed).  
2. Create DTW trajectory features from FP-Growth-filtered trajectories.  
3. Write `dtw_added_features_{cohort}_{age_band}.csv` ready for merging into the final feature matrix.

