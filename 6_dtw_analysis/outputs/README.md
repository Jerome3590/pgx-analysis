This directory contains local DTW trajectory feature outputs.

For each `(cohort_name, age_band, window)` (e.g. `opioid_ed`, `0_12`, `train`),
we store patient-level DTW feature tables such as:
- `{cohort_name}_{age_band_fname}_{window}_target_dtw_features.csv`

These are merged into the final model training dataset in `7_final_model`.***

