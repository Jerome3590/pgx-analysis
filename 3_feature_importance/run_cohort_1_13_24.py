#!/usr/bin/env python3
"""
Run feature importance for cohort 1 (opioid_ed), age band 13-24
"""

import os
import sys
from pathlib import Path

# Add project root to path so local helpers_1997_13 package is importable
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from helpers_1997_13.feature_importance_utils import run_cohort_analysis

COHORT_NAME = "opioid_ed"
AGE_BAND = "13-24"

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
