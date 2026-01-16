# Supported Age Bands

## Overview

The dashboard supports **7 age bands** across 2 cohorts. Age band 0-12 is excluded due to small cohort size. Ages 95-114 are mapped to the 85-94 age band (uses the 85-94 model).

## Opioid ED Risk (Ages 13-64)

| Age Band | Age Range | Status |
|----------|-----------|--------|
| 13-24 | Ages 13-24 | ✅ Supported |
| 25-44 | Ages 25-44 | ✅ Supported |
| 45-54 | Ages 45-54 | ✅ Supported |
| 55-64 | Ages 55-64 | ✅ Supported |
| 0-12 | Ages 0-12 | ❌ Excluded (small cohort) |

## Polypharmacy Risk (Ages 65-114)

| Age Band | Age Range | Status |
|----------|-----------|--------|
| 65-74 | Ages 65-74 | ✅ Supported |
| 75-84 | Ages 75-84 | ✅ Supported |
| 85-94 | Ages 85-94 | ✅ Supported |
| 95-114 | Ages 95-114 | ✅ Mapped to 85-94 (uses 85-94 model) |

## Age Validation

The dashboard validates age input:
- **Minimum**: 13 (age band 0-12 not supported)
- **Maximum**: 114 (ages 95-114 mapped to 85-94)
- **Error messages**: Clear feedback when age is out of range
- **Note**: Ages 95-114 are treated as age band 85-94 due to small cohort size (shown on dashboard)

## Model Availability

Models are prepared and deployed for all 7 supported age bands:
- 4 opioid_ed age bands
- 3 non_opioid_ed age bands

Total: **7 age bands × 3 models = 21 model files** (~23 MB total)

