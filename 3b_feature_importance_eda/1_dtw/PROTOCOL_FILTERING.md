# Protocol Event Filtering Using DTW Time Windows

## Overview

Events that occur too close together (e.g., < 7 days apart) often represent standard care protocols rather than predictive patterns. This filtering approach uses DTW time window analysis to identify and exclude such protocol-like events from model training.

## Strategy

### 1. **Time Interval Calculation**
- Calculate time intervals between consecutive events per patient
- Identify events that are part of protocol sequences (very short intervals)
- Default threshold: **7 days** (events closer than this are considered protocol-like)

### 2. **Filtering Logic**
- **Keep first event**: Always keep the first event per patient (even if protocol-like)
- **Keep high-frequency patients**: If a patient has > 50% protocol events, keep all events (may be genuinely high-frequency care)
- **Filter protocol events**: Otherwise, exclude events that are < threshold days apart

### 3. **Rationale**
- **Standard protocols**: Routine follow-ups, medication refills, scheduled tests
- **Predictive patterns**: Events with longer intervals may indicate deviations from standard care
- **High-frequency patients**: Some patients genuinely need frequent care (e.g., chronic conditions)

## Usage

```bash
python 6_dtw_analysis/filter_protocol_events.py \
    --cohort-name opioid_ed \
    --age-band 0-12 \
    --min-interval-days 7 \
    --keep-first-event \
    --protocol-threshold-pct 0.5
```

## Output

1. **Filtered model_data**: `model_data/cohort_name={cohort}/age_band={age_band}/model_events_no_protocols.parquet`
2. **Protocol summary**: `6_dtw_analysis/outputs/protocol_summary_{cohort}_{age_band}.csv`

## Results for Cohort 1, Age Band 0-12

- **Total events**: 5,350
- **Protocol events** (< 7 days apart): 4,597 (85.9%)
- **Non-protocol events**: 753 (14.1%)
- **Events removed**: 17 (0.3%)
- **Events kept**: 5,333 (99.7%)

**Note**: Most events are protocol-like, but the filtering logic preserves most events because:
- First events are always kept
- High-frequency patients (>50% protocol events) keep all events
- Only isolated protocol events are removed

## Integration with Workflow

### Option 1: Use Filtered Data for Feature Engineering
Replace `model_events.parquet` with `model_events_no_protocols.parquet` in:
- FP-Growth analysis
- BupaR sequence analysis
- DTW trajectory analysis
- Final model training

### Option 2: Use Protocol Flag as Feature
Keep all events but add `is_protocol_event` as a feature:
- Allows model to learn which events are protocol vs. predictive
- More flexible approach

## Parameters

- `--min-interval-days`: Minimum interval (days) to consider non-protocol (default: 7)
- `--keep-first-event`: Always keep first event per patient (default: True)
- `--protocol-threshold-pct`: Keep all events if patient has > this % protocol events (default: 0.5)

## Clinical Interpretation

- **< 7 days**: Likely protocol (routine follow-ups, refills)
- **7-30 days**: May be protocol or predictive
- **> 30 days**: More likely predictive (deviations from standard care)

## Next Steps

1. **Test with filtered data**: Re-run feature engineering and model training with `model_events_no_protocols.parquet`
2. **Compare performance**: See if removing protocol events improves model performance
3. **Adjust threshold**: Experiment with different `min-interval-days` values
4. **Analyze protocol patterns**: Use protocol summary to understand standard care patterns

