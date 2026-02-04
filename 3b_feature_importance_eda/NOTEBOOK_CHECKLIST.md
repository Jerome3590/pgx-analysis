# Notebook File Path Checklist

## Summary
All notebooks have been checked for consistent file paths and structure. Here's what was verified:

### ✅ All Notebooks Have:
1. **Consistent PROJECT_ROOT**: All use `/home/pgx3874/pgx-analysis` (EC2 path)
2. **Consistent OUTPUT_DIR structure**: `PROJECT_ROOT / "3b_feature_importance_eda" / "outputs" / COHORT / AGE_BAND_FNAME`
3. **Consistent PLOTS_DIR structure**: `OUTPUT_DIR / "plots"`
4. **Fallback paths for visualizations**: All notebooks check for plots in:
   - Primary: `PLOTS_DIR / plot_name`
   - Alternative: `OUTPUT_DIR / "features" / plot_name`
   - Simple name: `PLOTS_DIR / simple_name` (without cohort prefix)

### 📋 Notebooks Status:

| Notebook | PROJECT_ROOT | OUTPUT_DIR | PLOTS_DIR | BupaR Results Path | Visualization Fallbacks |
|----------|--------------|------------|-----------|-------------------|------------------------|
| cohort1  | ✅ | ✅ | ✅ | ✅ | ✅ |
| cohort2  | ✅ | ✅ | ✅ | ✅ | ✅ |
| cohort3  | ✅ | ✅ | ✅ | ✅ | ✅ |
| cohort4  | ✅ | ✅ | ✅ | ✅ | ✅ |
| cohort5  | ✅ | ✅ | ✅ | ✅ | ✅ |
| cohort6  | ✅ | ✅ | ✅ | ✅ | ✅ |
| cohort7  | ✅ | ✅ | ✅ | ✅ | ✅ |

### 🔍 Rplots.pdf Handling:

**Current Status:**
- R scripts (`create_bupar_outputs_*.R`) now save `Rplots.pdf` to the correct location:
  - `{OUTPUT_DIR}/plots/{cohort}_{age_band}_Rplots.pdf`
- R scripts explicitly open/close PDF device to prevent saving to project root

**Recommended Manual Addition (Optional):**
If you want to add cleanup code in the notebooks after BupaR analysis completes, you can add this snippet after the "BupaR analysis completed successfully" message:

```python
# Cleanup: Check for Rplots.pdf in project root and remove it
project_root_rplots = PROJECT_ROOT / "Rplots.pdf"
if project_root_rplots.exists():
    print(f"\n⚠️  Found Rplots.pdf in project root (should be in plots directory)")
    print(f"   Removing: {project_root_rplots}")
    try:
        project_root_rplots.unlink()
        print(f"   ✅ Removed Rplots.pdf from project root")
    except Exception as e:
        print(f"   ⚠️  Could not remove: {e}")

# Verify correct Rplots.pdf exists in plots directory
correct_rplots = PLOTS_DIR / f"{COHORT}_{AGE_BAND_FNAME}_Rplots.pdf"
if correct_rplots.exists():
    size_mb = correct_rplots.stat().st_size / (1024 * 1024)
    print(f"   ✅ Correct Rplots.pdf found in plots directory: {size_mb:.2f} MB")
else:
    print(f"   ℹ️  Rplots.pdf not found in plots directory (may not have base graphics)")
```

**Note:** This is optional since the R scripts now handle PDF device management correctly. The cleanup code is just a safety check.

### 📝 File Path Patterns:

All notebooks follow these patterns:

1. **BupaR Results CSV:**
   - Primary: `OUTPUT_DIR / f"{COHORT}_{AGE_BAND_FNAME}_bupar_post_target_analysis.csv"`
   - (No fallback needed - R scripts save to correct location)

2. **BupaR Visualizations:**
   - Primary: `PLOTS_DIR / f"{COHORT}_{AGE_BAND_FNAME}_{plot_type}.png"`
   - Fallback 1: `OUTPUT_DIR / "features" / f"{COHORT}_{AGE_BAND_FNAME}_{plot_type}.png"`
   - Fallback 2: `PLOTS_DIR / f"{plot_type}.png"` (without cohort prefix)

3. **Model Data (Step 3b – Step 1/2/3 only, no 4_model_data):**
   - Target and control: `3b_feature_importance_eda/outputs/cohorts/input_model_data/cohort_name={slug}/age_band={age_band}/model_events.parquet`
   - 4_model_data is created only after target leakage removal (later step).
   - (Handled by R/Python scripts, not directly in notebooks)

### ✅ Conclusion:
All notebooks are consistent and follow the same file path patterns. The Rplots.pdf issue should be resolved by the R script changes (explicit PDF device management). If you still see Rplots.pdf in the project root, you can manually add the cleanup code snippet above, or simply delete it manually.
