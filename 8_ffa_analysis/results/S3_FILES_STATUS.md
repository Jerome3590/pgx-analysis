# FFA Analysis S3 Files Status

**Date Checked**: 2026-01-14

## Summary

✅ **ALL REQUIRED FILES PRESENT** - All visualization files are available in S3 for all 7 cohorts.

## Required Files Per Cohort

Each cohort/age_band combination requires 4 parquet files in S3:
- `axp_explanations.parquet` - AXP explanations for instances
- `feature_importance_axp.parquet` - Feature importance scores
- `causal_importance.parquet` - Causal importance analysis
- `interaction_analysis.parquet` - Feature interaction analysis

## Status by Cohort

### Opioid ED (`opioid_ed`)

| Age Band | axp_explanations | feature_importance | causal_importance | interaction_analysis | Status |
|----------|------------------|-------------------|-------------------|---------------------|--------|
| 13-24 | ✅ | ✅ | ✅ | ✅ | Complete |
| 25-44 | ✅ | ✅ | ✅ | ✅ | Complete |
| 45-54 | ✅ | ✅ | ✅ | ✅ | Complete |
| 55-64 | ✅ | ✅ | ✅ | ✅ | Complete |

### Non-Opioid ED (`non_opioid_ed`)

| Age Band | axp_explanations | feature_importance | causal_importance | interaction_analysis | Status |
|----------|------------------|-------------------|-------------------|---------------------|--------|
| 65-74 | ✅ | ✅ | ✅ | ✅ | Complete |
| 75-84 | ✅ | ✅ | ✅ | ✅ | Complete |
| 85-94 | ✅ | ✅ | ✅ | ✅ | Complete |

## S3 Location

All files are located at:
```
s3://pgxdatalake/gold/ffa_analysis/{cohort}/{age_band}/xgboost/{filename}
```

Example:
```
s3://pgxdatalake/gold/ffa_analysis/opioid_ed/13-24/xgboost/axp_explanations.parquet
s3://pgxdatalake/gold/ffa_analysis/opioid_ed/13-24/xgboost/feature_importance_axp.parquet
s3://pgxdatalake/gold/ffa_analysis/opioid_ed/13-24/xgboost/causal_importance.parquet
s3://pgxdatalake/gold/ffa_analysis/opioid_ed/13-24/xgboost/interaction_analysis.parquet
```

## Visualization Readiness

✅ **All cohorts are ready for visualization generation**

The visualization scripts (`8_ffa_analysis/create_visualizations.py`) can access all required data files from S3 to generate:
- Feature importance comparisons
- Coverage and importance metrics
- Explanation statistics
- Model comparison charts
- Cattail visualizations

## Optional Visualization Outputs

Visualization outputs (PNG plots, HTML dashboards) are optional and can be uploaded to:
```
s3://pgxdatalake/gold/ffa_analysis/{cohort}/{age_band}/visualizations/
```

These are generated on-demand and are not required for the core analysis.

## Verification

To verify files, run:
```bash
python3 utility_scripts/check_ffa_s3_files.py
```
