# DTW Role: Feature Filtering vs. Feature Engineering

## Key Insight

**DTW is more valuable as a feature filtering mechanism than as a feature engineering tool.**

## Primary Use: Noise Reduction via Protocol Filtering

### What DTW Time Windows Reveal
- **Protocol-like events**: Events occurring < 7 days apart often represent standard care protocols
- **Predictive patterns**: Events with longer intervals (> 7 days) may indicate deviations from standard care
- **Noise reduction**: Filtering out protocol events reduces noise in downstream analyses

### Benefits
1. **Cleaner sequences**: FP-Growth and BupaR analyses focus on meaningful patterns
2. **Better signal**: Removes routine care that both targets and controls follow
3. **Improved model performance**: Less noise = better predictive signal

## Secondary Use: Feature Engineering (Optional)

### DTW Distance Features
- **Value**: Moderate - DTW distances to prototypes showed non-significant differences between targets/controls
- **Interpretation**: Trajectory patterns may represent standard care protocols rather than predictive signals
- **Recommendation**: Include but do not prioritize; focus on filtering first

### When DTW Features Are Useful
- **Large cohorts**: May show more predictive patterns with more data
- **Specific trajectories**: Some rare trajectory patterns may be predictive
- **Complementary signal**: Works alongside other features (FP-Growth, BupaR, PGx)

## Workflow Integration

### Step 1: Protocol Filtering (Primary)

```bash
# Filter protocol events using DTW time windows
python 5_dtw_analysis/filter_protocol_events.py \
    --cohort-name opioid_ed \
    --age-band 0-12 \
    --min-interval-days 7
```

**Output**: `model_events_no_protocols.parquet`

### Step 2: Feature Engineering (Secondary)

```bash
# Create DTW features (optional, lower priority)
python 5_dtw_analysis/create_dtw_features.py \
    --cohort opioid_ed \
    --age_band 0-12
```

**Output**: `dtw_features_{cohort}_{age_band}.csv`

## Updated Workflow Priority

### High Priority (Noise Reduction)
1. ✅ **Protocol filtering**: Use DTW time windows to filter protocol events
2. ✅ **Clean sequences**: Use filtered data for FP-Growth and BupaR
3. ✅ **Better signal**: Focus on deviations from standard care

### Medium Priority (Feature Engineering)
1. ⚠️ **DTW features**: Include but do not prioritize
2. ⚠️ **Trajectory distances**: May be useful for large cohorts
3. ⚠️ **Complementary signal**: Works with other features

## Results Summary

### Protocol Filtering (Cohort 1, Age Band 0-12)
- **85.9% protocol events**: Most events are standard care
- **Filtered data**: Reduces noise for downstream analyses
- **Impact**: Better signal-to-noise ratio

### DTW Features (Cohort 1, Age Band 0-12)
- **Non-significant differences**: Targets and controls have similar trajectories
- **Interpretation**: Trajectory patterns represent protocols, not predictive signals
- **Recommendation**: Use for filtering, not primary features

## Conclusion

**DTW's primary value is in preprocessing/filtering:**
- Identifies protocol-like events using time windows
- Filters noise before feature engineering
- Improves signal quality for FP-Growth, BupaR, and final model

**DTW features are secondary:**
- May be useful for large cohorts
- Complement other features
- Do not prioritize over filtering

## Next Steps

1. **Use filtered data**: Run all feature engineering with `model_events_no_protocols.parquet`
2. **Compare performance**: Test model with and without protocol filtering
3. **Refine threshold**: Adjust `min-interval-days` based on results
4. **Keep DTW features**: Include but do not prioritize in feature selection

