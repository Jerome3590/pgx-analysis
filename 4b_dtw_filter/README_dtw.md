## DTW Trajectory and Predictive Time Features

This directory contains scripts to build time-based and DTW-based trajectory features from `4a_model_data` for each `(cohort, age_band)` combination.

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

