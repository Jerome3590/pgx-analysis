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

**IMPORTANT**: This filter now focuses on **code classification** (administrative vs. medical/pharmacy) rather than time intervals alone. Events are filtered based on whether they are administrative (billing, scheduling, post-event documentation) vs. medical/pharmacy related, regardless of time intervals.

Time window analysis (using `min_interval_days`, default: 1 day) is still performed for research purposes to understand trajectory patterns, but the actual filtering is based on code classification.

**Note**: The filtering decisions in this step should be research-driven. Before applying filters, analyze all trajectories and time windows to understand what is clinically useful vs. what is noise.

## Strategy

### 1. **Code Classification**
- **Administrative codes**: Identified through research as codes that appear primarily in protocol-like sequences (events < 1 day apart)
  - Billing codes (specific CPT codes for billing/documentation)
  - Scheduling codes (appointment scheduling, administrative procedures)
  - Post-event documentation (events after target event date - leakage)
- **Clinical codes**: All other codes (diagnoses, procedures, medications)
  - These are kept regardless of time intervals

### 2. **Research-Based Identification**
- **Time window analysis**: Calculate intervals between consecutive events to identify protocol-like sequences
  - **Default threshold: 1 day** for research (events closer than this are considered protocol-like for code analysis)
  - **Rationale**: With both pharmacy and medical data, events can occur more frequently
- **Code analysis**: Identify which codes appear in > 80% of protocol-like sequences (default threshold)
- **Research outputs**: Generate `code_analysis_protocol_vs_clinical_*.csv` with code classifications

### 3. **Filtering Logic**
- **Filter administrative codes**: Remove events with codes identified as administrative
- **Filter post-event leakage**: Remove events occurring on or after target event date
- **Keep first event**: Always keep the first event per patient (even if administrative)
- **Keep all clinical events**: Medical/pharmacy events are kept even if they occur close together

### 4. **Rationale**
- **Administrative events are noise**: Billing, scheduling, and post-event documentation don't provide predictive signal
- **Clinical events are valuable**: Diagnoses, procedures, and medications should be preserved regardless of timing
- **Time intervals are for research**: Time window analysis helps identify administrative codes, but filtering is code-based

## Usage

```bash
python 4b_dtw_filter/filter_protocol_events.py \
    --cohort-name opioid_ed \
    --age-band 0-12 \
    --min-interval-days 3 \
    --keep-first-event \
    --admin-code-threshold-pct 80.0
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
- **Administrative events filtered**: [varies by cohort - check research outputs]
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

- `--min-interval-days`: Minimum interval (days) for time window analysis in research outputs (default: 1, matches BupaR). Note: Filtering is based on code classification, not time intervals.
- `--keep-first-event`: Always keep first event per patient (even if administrative, default: True)
- `--admin-code-threshold-pct`: Threshold for considering a code administrative from research outputs (codes with > this % in protocol-like sequences are considered administrative, default: 80.0)

## Code Classification Approach

**Important**: Filtering is now based on **code classification** (administrative vs. medical/pharmacy), not time intervals.

### Administrative Event Identification

Administrative events are identified through:
1. **Code pattern analysis**: Research outputs identify which codes appear primarily in administrative contexts
2. **Post-event events**: Events occurring after target event date (leakage)
3. **Billing/scheduling codes**: Specific CPT codes that indicate administrative procedures

### Medical/Pharmacy Event Identification

Medical/pharmacy events include:
1. **Clinical diagnoses**: ICD codes representing actual medical conditions
2. **Medical procedures**: CPT codes for clinical procedures
3. **Pharmacy prescriptions**: Drug codes for medications

### Time Window Analysis (Research Only)

Time window analysis (1-day default) is used for:
1. **Understanding trajectory patterns**: Which sequences occur frequently?
2. **Identifying administrative clustering**: Do administrative events cluster at very short intervals?
3. **Validating code classifications**: Do codes classified as administrative appear in short-interval sequences?

### How to Research and Classify Codes

1. **Review research outputs**: Use the research outputs in `outputs/for_review/` to see code frequencies and patterns
2. **Analyze code distributions**: Which codes appear in administrative vs. clinical contexts?
3. **Validate classifications**: Consult clinical experts on code meanings
4. **Update classification logic**: Refine the `classify_event_as_administrative()` function based on research findings

## Next Steps

1. **Test with filtered data**: Re-run feature engineering and model training with `model_events_no_protocols.parquet`
2. **Compare performance**: See if removing protocol events improves model performance
3. **Adjust threshold**: Experiment with different `min-interval-days` values based on research findings
4. **Analyze protocol patterns**: Use protocol summary and research outputs to understand standard care patterns

