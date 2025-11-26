# ============================================================
# CONSTANTS FOR FEATURE IMPORTANCE ANALYSIS
# ============================================================
# Shared constants matching helpers_1997_13/constants.py

# Age bands for cohort analysis
# Note: These match helpers_1997_13/constants.py
# If cohort files use "75+" instead of granular bands, filter accordingly
AGE_BANDS <- c(
  "0-12", "13-24", "25-44", "45-54", "55-64", "65-74", 
  "75-84", "85-94", "95-114"
)

# Event years for cohort analysis
EVENT_YEARS <- c("2016", "2017", "2018", "2019", "2020")

# Cohort names
# Note: These match the cohort_name partition values in S3/parquet files
COHORT_NAMES <- c("opioid_ed", "non_opioid_ed")

