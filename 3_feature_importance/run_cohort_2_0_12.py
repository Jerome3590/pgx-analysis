#!/usr/bin/env python3
"""
Run feature importance for cohort 2 (non_opioid_ed), age band 0-12
"""
from helpers_1997_13.feature_importance_utils import run_cohort_analysis

COHORT_NAME = "non_opioid_ed"
AGE_BAND = "0-12"

if __name__ == '__main__':
    print(f"Running feature importance for {COHORT_NAME} / {AGE_BAND}")
    result = run_cohort_analysis(
        cohort_name=COHORT_NAME,
        age_band=AGE_BAND,
        train_years=[2016, 2017, 2018],
        test_year=2019,
        n_splits=200,
        train_prop=0.8,
        n_workers=4,
        scaling_metric='recall',
        model_params=None,
        debug_mode=False,
        output_dir='3_feature_importance/outputs'
    )
    print(result)
