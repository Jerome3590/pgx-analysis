# Lambda ECR Container Storage Analysis

## Current File Sizes (Measured)

### Single Age Band: `opioid_ed` / `0-12`

**Model Files:**
- CatBoost JSON: **1.7 MB**
- XGBoost JSON: **528 KB**
- XGBoost RF JSON: **340 KB**
- Combined JSON: **340 KB**
- Joblib model: **324 KB**
- MC-CV results CSV: **~10 KB**
- **Total per age_band: ~3.2 MB**

**Other Files:**
- Feature importance CSVs: **~750 KB** (all cohorts combined)
- Feature schemas (estimated): **~50 KB per age_band**
- Metadata JSON (estimated): **~100 KB per cohort**

## Storage Breakdown

### Models (Per Age Band)
- **3 JSON models**: ~2.9 MB (CatBoost + XGBoost + XGBoost RF)
- **1 Joblib model**: ~324 KB (backup/alternative format)
- **MC-CV results**: ~10 KB
- **Feature schema**: ~50 KB
- **Total per age_band: ~3.3 MB**

### All Age Bands Projection

**Opioid ED (4 age bands):**
- 13-24: ~3.3 MB
- 25-44: ~3.3 MB
- 45-54: ~3.3 MB
- 55-64: ~3.3 MB
- **Subtotal: ~13.2 MB**

**Polypharmacy (3 age bands):**
- 65-74: ~3.3 MB
- 75-84: ~3.3 MB
- 85-94: ~3.3 MB
- **Subtotal: ~9.9 MB**

**Total Models: ~23 MB** (7 age bands × 3.3 MB)

### Metadata Files
- Feature importance CSVs: **~750 KB**
- Metadata JSON files: **~200 KB** (2 cohorts)
- **Total Metadata: ~1 MB**

### Python Dependencies
- **CatBoost**: ~500 MB
- **XGBoost**: ~200 MB
- **Other (pandas, numpy, joblib, boto3)**: ~300 MB
- **Total Dependencies: ~1 GB**

### Base Lambda Image
- **Python 3.10 base image**: ~500 MB

## Total Container Size Estimate

| Component | Size |
|-----------|------|
| Models (7 age bands) | ~23 MB |
| Metadata | ~1 MB |
| Python Dependencies | ~1 GB |
| Base Lambda Image | ~500 MB |
| **TOTAL** | **~1.5 GB** |

## ECR Limit Check

- **ECR Container Limit**: 10 GB
- **Estimated Usage**: ~1.5 GB
- **Remaining Space**: ~8.5 GB
- **Utilization**: ~15%

## S3 Training Data (Not Included in Lambda)

**Note**: The S3 gold cohort parquet files (~7 GB) are **training data** and do **NOT** need to be included in the Lambda container. These are only used when retraining models, not for inference.

**S3 Cohort Data Summary:**
- Total: ~7.0 GB (90 parquet files)
- opioid_ed: 179 MB
- non_opioid_ed: 6.8 GB
- Per age band: 39 MB - 1.7 GB

## ✅ Conclusion

**Yes, you can easily store all cohort data in Lambda ECR!**

**Important**: You only need the **trained models** (~23 MB), not the training data (7 GB).

### Key Points:

1. **Models are small**: ~3.3 MB per age_band
   - Even with 7 age bands = ~23 MB total
   - Well within limits

2. **Dependencies are the largest component**: ~1 GB
   - CatBoost + XGBoost + other libraries
   - This is standard and expected

3. **Plenty of room for growth**:
   - Current: ~1.5 GB
   - Limit: 10 GB
   - Can add ~5-6x more models/age_bands if needed

4. **All models fit comfortably**:
   - All 7 age bands (4 opioid_ed + 3 polypharmacy)
   - All 3 model types per age_band
   - All metadata and schemas

## Optimization Opportunities (If Needed)

If you need to reduce size further:

1. **Use JSON models only** (skip Joblib): Saves ~324 KB per age_band
2. **Model quantization**: Could reduce model sizes by 30-50%
3. **Selective inclusion**: Only include age_bands you actively use
4. **Compress models**: Gzip compression (models decompress quickly)

## Recommended Container Structure

```
/var/task/
├── lambda_function.py
├── models/
│   ├── opioid_ed/
│   │   ├── 13_24/
│   │   │   ├── catboost.json (1.7 MB)
│   │   │   ├── xgboost.json (528 KB)
│   │   │   ├── xgboost_rf.json (340 KB)
│   │   │   ├── feature_schema.json (50 KB)
│   │   │   └── mc_cv_results.csv (10 KB)
│   │   ├── 25_44/ (~3.3 MB)
│   │   ├── 45_54/ (~3.3 MB)
│   │   └── 55_64/ (~3.3 MB)
│   └── non_opioid_ed/
│       ├── 65_74/ (~3.3 MB)
│       ├── 75_84/ (~3.3 MB)
│       └── 85_94/ (~3.3 MB)
├── metadata/
│   ├── metadata_opioid_ed.json (~100 KB)
│   └── metadata_non_opioid_ed.json (~100 KB)
└── [Python dependencies] (~1 GB)
```

**Total: ~1.5 GB** (well within 10 GB limit)

## Next Steps

1. ✅ **Proceed with ECR deployment** - plenty of space
2. ✅ **Bundle all models** - fits comfortably
3. ✅ **Include all age bands** - no need to exclude any
4. ✅ **Keep both JSON and Joblib** - minimal size impact

Your cohort data files are **very small** compared to the ECR limit. You have plenty of room for all models, metadata, and dependencies!

