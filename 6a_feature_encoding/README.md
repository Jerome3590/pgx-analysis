## 6a_feature_encoding – Cohort-Specific Feature Encoding Artifacts

This step builds **cohort- and age-band-specific encoding artifacts** that are used
by the final model (step 6b), SHAP analysis, and FFA.

For each `(cohort, age_band)` we create:

- A **feature lookup table** mapping numeric feature indices to names and groups.
- A **drug codebook** giving a fixed numeric encoding for every distinct `drug_name`.

### Scripts

The current implementation of 6a lives in the `6_final_model` module:

- `6_final_model/create_feature_lookup.py`
  - Input:
    - `6_final_model/outputs/{cohort}/{age_band_fname}/{cohort}_{age_band_fname}_train_final_features_no_leakage.csv`
  - Outputs:
    - `6_final_model/outputs/{cohort}/{age_band_fname}/{cohort}_{age_band_fname}_feature_lookup.csv`
    - `feature_encoding_outputs/{cohort}/{age_band_fname}/{cohort}_{age_band_fname}_feature_lookup.csv`
  - Contents:
    - `feature_index`, `feature_name`, `group`, `description`
    - For FP-Growth itemsets: `itemset_type` and `itemset_items` (actual drug/ICD/CPT/medical codes).

- `6_final_model/create_drug_codebook.py`
  - Input:
    - `4a_model_data/cohort_name={cohort}/age_band={age_band}/model_events.parquet`
  - Outputs:
    - `6_final_model/outputs/{cohort}/{age_band_fname}/{cohort}_{age_band_fname}_drug_codebook.csv`
    - `feature_encoding_outputs/{cohort}/{age_band_fname}/{cohort}_{age_band_fname}_drug_codebook.csv`
  - Contents:
    - `drug_id`, `drug_name_raw`, `drug_name_normalized`
    - All numeric encoding dimensions from `encode_drug_name_series` for that drug.

### Usage

For each `(cohort, age_band)`:

```bash
# 6a: build encoding artifacts
python 6_final_model/create_feature_lookup.py --cohort opioid_ed --age_band 13-24
python 6_final_model/create_drug_codebook.py   --cohort opioid_ed --age_band 13-24
```

This should be run **before** step 6b (`run_final_model.py`) so that model outputs,
SHAP, and FFA can consistently reference the encoding artifacts in
`feature_encoding_outputs/{cohort}/{age_band_fname}/`.

