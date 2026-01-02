# Protocol Event Filtering Using DTW Time Windows

## Background: Research-Driven Filtering

### The Importance of Time Windows and Trajectories

Patient medical histories are temporal sequences of events. Understanding these sequences requires analyzing:

1. **Time Windows**: The intervals between consecutive events reveal whether events are part of standard care protocols (short intervals) or deviations from standard care (longer intervals)
2. **Common Sequence Patterns (Trajectories)**: Repeated sequences of events across patients may represent standard care pathways or predictive patterns
3. **Deviations from Common Patterns**: Patients who follow unusual trajectories may be at higher risk

### Why Research Happens in DTW Filter

**`dtw_filter` (Step 4b) runs before `dtw_analysis` (Step 5d)** in the pipeline. This means:

- **All research must happen here**: Since filtering occurs first, we must identify what to filter vs. what to keep during this step
- **Capture everything first**: We need to analyze all trajectories and time windows to make informed filtering decisions
- **Research-driven decisions**: Filtering should be based on analysis of trajectory patterns, not arbitrary thresholds
- **Clinical validation**: We need to research which trajectories are clinically meaningful vs. routine care that both targets and controls follow

### Research Objectives

Before filtering, we need to understand:

1. **What are the common trajectory patterns?**
   - Which sequences of events appear frequently across patients?
   - Are these standard care protocols or predictive signals?

2. **What time intervals are meaningful?**
   - Which intervals indicate protocol-like care (routine follow-ups, refills)?
   - Which intervals indicate deviations from standard care (predictive signals)?

3. **What should be filtered vs. kept?**
   - Protocol-like events: Routine care that both targets and controls follow (filter out)
   - Predictive patterns: Deviations from standard care that predict outcomes (keep)

4. **Clinical interpretation**
   - Which trajectories represent legitimate risk markers?
   - Which trajectories are just routine care that adds noise?

### The Research Workflow

1. **Extract all trajectories**: Capture complete patient sequences with full time window information
2. **Analyze patterns**: Identify common sequences, time intervals, and trajectory characteristics
3. **Research clinical meaning**: Determine which patterns are protocol-like vs. predictive
4. **Make filtering decisions**: Filter out protocol-like events based on research findings
5. **Preserve what's good**: Keep clinically meaningful trajectories for downstream analysis

## Overview

Events that occur too close together (e.g., < 7 days apart) often represent standard care protocols rather than predictive patterns. This filtering approach uses DTW time window analysis to identify and exclude such protocol-like events from model training.

**Note**: The filtering decisions in this step should be research-driven. Before applying filters, analyze all trajectories and time windows to understand what is clinically useful vs. what is noise.

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
python 5_dtw_analysis/filter_protocol_events.py \
    --cohort-name opioid_ed \
    --age-band 0-12 \
    --min-interval-days 7 \
    --keep-first-event \
    --protocol-threshold-pct 0.5
```

## Output

1. **Filtered model_data**  
   - `model_data/cohort_name={cohort}/age_band={age_band}/model_events_no_protocols.parquet`
2. **Protocol summary (per-patient)**  
   - `5_dtw_analysis/outputs/{cohort}/{age_band}/protocol_summary_{cohort}_{age_band}.csv`
3. **Event-level intervals with flags (audit trail)**  
   - `5_dtw_analysis/outputs/{cohort}/{age_band}/event_intervals_{cohort}_{age_band}.parquet`  
   - Columns include:
     - `mi_person_key`, `current_event_date`, `previous_event_date`
     - `days_since_previous`
     - `is_protocol_event` (1 = protocol-like, 0 = non-protocol)

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

### Option 1: Use Filtered Data for Feature Engineering (Recommended)
**All downstream feature engineering steps automatically prefer `model_events_no_protocols.parquet` if available:**

- **FP-Growth analysis**: Uses filtered data to generate itemsets and association rules from useful signals only (non-protocol events)
- **BupaR sequence analysis**: Uses filtered data for cleaner process mining
- **DTW trajectory analysis**: Uses filtered data for trajectory feature extraction
- **Final model training**: Uses filtered data for better predictive signal

**Note**: FP-Growth scripts (`cohort_fpgrowth.py` and `create_fpgrowth_features.py`) automatically check for `model_events_no_protocols.parquet` first, then fall back to `model_events.parquet` if not available. This ensures itemsets and rules only capture useful signals.

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

