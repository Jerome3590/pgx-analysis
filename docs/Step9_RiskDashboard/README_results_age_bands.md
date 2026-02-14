# Supported Age Bands

## Overview

The dashboard supports **8 age bands** (0-12 through 85-114). Both **Opioid ED** and **Polypharmacy** cohorts use the same full set of age bands. Age band 0-12 is excluded from risk calculation due to small cohort size (minimum age 13 for risk). The last band **85-114** combines the former 85-94 and 95-114 bands.

## Age Bands (Full Set for Both Cohorts)

| Age Band | Age Range | Risk Supported |
|----------|-----------|----------------|
| 0-12 | Ages 0-12 | ❌ Excluded (metadata/visualizations only) |
| 13-24 | Ages 13-24 | ✅ Supported |
| 25-44 | Ages 25-44 | ✅ Supported |
| 45-54 | Ages 45-54 | ✅ Supported |
| 55-64 | Ages 55-64 | ✅ Supported |
| 65-74 | Ages 65-74 | ✅ Supported |
| 75-84 | Ages 75-84 | ✅ Supported |
| 85-114 | Ages 85-114 | ✅ Supported |

## Cohort Selection (Dashboard)

- **Cohort is chosen by the user** via the **Opioid ED** or **Polypharmacy** tab on the dashboard, not by age.
- Age only selects the **age band** within the chosen cohort (both cohorts have the same bands).
- On the data visualization tabs (Causal, BupaR, DTW, FP-Growth), cohort and age band can be changed independently to view either cohort’s visualizations.

## Age Validation

- **Minimum for risk**: 13 (age band 0-12 not supported for risk calculation)
- **Maximum**: 114
- **Error messages**: Clear feedback when age is out of range

## Model Availability

Models are prepared and deployed per cohort and age band. Both cohorts use the full set of age bands (e.g. 8 bands × 2 cohorts = 16 cohort/age_band combinations, each with 3 model types when trained).
