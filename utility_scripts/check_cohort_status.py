from __future__ import annotations

from pathlib import Path

from py_helpers.constants import COHORT_NAMES, AGE_BANDS, EVENT_YEARS
from py_helpers.s3_utils import (
    check_feature_importance_results_exist,
    check_cohort_file_exists,
)


def main() -> None:
    """
    Quick status check for feature importance and local cohort/model data
    for the main modeling grid described in README_analysis_workflow.md.

    This is intentionally lightweight: it only checks
      - aggregated feature-importance in S3 (test year 2019)
      - local cohort parquet files under data/cohorts_F1120 (or fallback)
    and prints an ASCII-only summary to the console.
    """
    # Modeling grid we care about right now:
    #   Cohort 1 (opioid_ed): 0-12, 13-24, 25-44, 45-54, 55-64
    #   Cohort 2 (non_opioid_ed): 65-74, 75-84, 85-94
    grid = [
        ("opioid_ed", "0-12"),
        ("opioid_ed", "13-24"),
        ("opioid_ed", "25-44"),
        ("opioid_ed", "45-54"),
        ("opioid_ed", "55-64"),
        ("non_opioid_ed", "65-74"),
        ("non_opioid_ed", "75-84"),
        ("non_opioid_ed", "85-94"),
    ]

    print("Cohort status (feature importance S3 + local cohort parquet):")
    print("cohort, age_band, fi_s3_2019, local_cohort_2016_2019")

    for cohort, band in grid:
        # Check aggregated FI for test year 2019
        fi_ok = check_feature_importance_results_exist(cohort, band, 2019)

        # Check that we have at least one local cohort parquet (train or test years)
        local_any = False
        for year in (2016, 2017, 2018, 2019):
            if check_cohort_file_exists(cohort, band, year):
                local_any = True
                break

        print(
            "{0},{1},{2},{3}".format(
                cohort,
                band,
                "yes" if fi_ok else "no",
                "yes" if local_any else "no",
            )
        )


if __name__ == "__main__":
    main()


