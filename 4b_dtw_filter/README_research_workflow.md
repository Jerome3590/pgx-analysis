# DTW Filter Research Workflow

This document outlines the research workflow for DTW protocol filtering, including how to run the filter for all cohorts and analyze trajectories, time windows, and codes.

## Overview

The DTW filter research phase involves:

1. **Running DTW filter for all cohorts** to generate filtered data and analysis artifacts
2. **Reviewing time windows** to understand event interval distributions
3. **Analyzing trajectories** to identify common sequence patterns
4. **Classifying codes** as clinical vs. administrative
5. **Checking for post-event leakage** (events after target date)

## Step 1: Run DTW Filter for All Cohorts

### Batch Processing

Run DTW filter for all cohort/age band combinations:

```bash
# Run for all cohorts and age bands
python 4b_dtw_filter/run_dtw_filter_all_cohorts.py

# Skip cohorts that already have filtered data
python 4b_dtw_filter/run_dtw_filter_all_cohorts.py --skip-existing

# Custom parameters
python 4b_dtw_filter/run_dtw_filter_all_cohorts.py \
    --min-interval-days 1 \
    --protocol-threshold-pct 0.5

# Specific cohorts/age bands
python 4b_dtw_filter/run_dtw_filter_all_cohorts.py \
    --cohorts opioid_ed non_opioid_ed \
    --age-bands 0-12 13-24 25-44
```

### Individual Processing

Run DTW filter for a single cohort/age band:

```bash
python 4b_dtw_filter/filter_protocol_events.py \
    --cohort-name opioid_ed \
    --age-band 0-12 \
    --min-interval-days 1 \
    --keep-first-event \
    --protocol-threshold-pct 0.5
```

### Outputs

For each cohort/age band, the filter generates:

1. **Filtered model data**: `4a_model_data/cohort_name={cohort}/age_band={age_band}/model_events_no_protocols.parquet`
2. **Event intervals**: `4b_dtw_filter/outputs/{cohort}/{age_band}/event_intervals_{cohort}_{age_band}.parquet`
3. **Protocol summary**: `4b_dtw_filter/outputs/{cohort}/{age_band}/protocol_summary_{cohort}_{age_band}.csv`

## Step 2: Research Trajectories and Time Windows

### Using the Research Script

Run the research script for a specific cohort/age band:

```bash
python 4b_dtw_filter/research_trajectories.py \
    --cohort-name opioid_ed \
    --age-band 0-12
```

This will output:
- Time window statistics (protocol vs non-protocol events, interval distributions)
- Common trajectory sequences (top sequences, protocol sequences, non-protocol sequences)
- Code-level analysis (ICD, CPT, drugs by protocol status)
- Post-event leakage check (events after target date)

### Using the Research Notebook

For interactive analysis, use the Jupyter notebook:

```bash
jupyter notebook 4b_dtw_filter/research_trajectories_and_time_windows.ipynb
```

The notebook provides:
- Interactive visualizations of time windows
- Detailed trajectory analysis
- Code classification tables
- Leakage detection

## Step 3: Research Objectives

### 1. Time Windows

**Research Questions:**
- What intervals between events are meaningful?
- Which intervals indicate protocol-like care (routine follow-ups, refills)?
- Which intervals indicate deviations from standard care (predictive signals)?

**Analysis:**
- Review interval distributions (histograms, statistics)
- Compare protocol vs non-protocol event intervals
- Identify optimal threshold (default: 1 day, matches BupaR time windows)
- Consider cohort-specific thresholds

**Findings to Document:**
- Mean/median interval days
- Protocol event percentage
- Interval distribution by bins (<1 day, 1-3 days, 3-7 days, etc.)
- Recommended threshold for filtering

### 2. Common Trajectories

**Research Questions:**
- Which sequence patterns appear frequently?
- Are these standard care pathways (both targets and controls follow)?
- Are these predictive patterns (deviations from standard care)?

**Analysis:**
- Extract 2-event and 3-event sequences
- Count sequence frequencies
- Compare protocol vs non-protocol sequences
- Analyze sequences by target status

**Findings to Document:**
- Top 20 most common sequences
- Top protocol sequences (< 1 day apart by default)
- Top non-protocol sequences (≥ 1 day apart by default)
- Sequences that appear in both targets and controls (likely protocols)
- Sequences that appear primarily in targets (potentially predictive)

### 3. Code Classification

**Research Questions:**
- Which codes are clinical (diagnoses, procedures, medications)?
- Which codes are administrative (billing, scheduling, post-event documentation)?
- Which codes appear primarily in protocol sequences?

**Analysis:**
- Analyze ICD codes by protocol status
- Analyze CPT codes by protocol status
- Analyze drug names by protocol status
- Identify codes with high protocol percentages (>80%)

**Findings to Document:**
- Top ICD codes with protocol percentages
- Top CPT codes with protocol percentages
- Top drugs with protocol percentages
- Codes to consider for additional filtering (highly administrative)
- Codes to preserve (clinical, low protocol percentage)

### 4. Post-Event Events (Leakage)

**Research Questions:**
- Are there events that occur after the target event date?
- Do these represent leakage (information from the future)?

**Analysis:**
- Check for events after `first_opioid_ed_date` (opioid_ed cohort)
- Check for events after `first_ed_non_opioid_date` (non_opioid_ed cohort)
- Verify cutoff logic matches BupaR analysis

**Findings to Document:**
- Total post-event events
- Post-event event percentage
- Whether leakage is detected
- Recommended cutoff date logic

## Step 4: Document Research Findings

Create a research notes file for each cohort/age band:

```
4b_dtw_filter/research_notes/{cohort}_{age_band}_research.md
```

Include:
- Time window findings and recommended threshold
- Common trajectory patterns (protocol vs predictive)
- Code classifications (clinical vs administrative)
- Leakage check results
- Recommended filter parameters
- Any cohort-specific considerations

## Step 5: Adjust Filter Parameters

Based on research findings, adjust filter parameters if needed:

```bash
# Example: Adjust threshold for specific cohort
python 4b_dtw_filter/filter_protocol_events.py \
    --cohort-name opioid_ed \
    --age-band 0-12 \
    --min-interval-days 10 \  # Adjusted based on research
    --protocol-threshold-pct 0.6  # Adjusted based on research
```

## Step 6: Validate Filtered Data

After filtering, validate the filtered data:

1. **Check filtered data exists**: `model_events_no_protocols.parquet`
2. **Compare event counts**: Original vs filtered
3. **Verify protocol events removed**: Check protocol summary
4. **Test downstream steps**: Ensure FP-Growth and other steps can use filtered data

## Research Checklist

For each cohort/age band:

- [ ] Run DTW filter
- [ ] Review time window statistics
- [ ] Analyze common trajectories
- [ ] Classify codes (clinical vs administrative)
- [ ] Check for post-event leakage
- [ ] Document research findings
- [ ] Adjust filter parameters if needed
- [ ] Re-run filter with adjusted parameters
- [ ] Validate filtered data
- [ ] Verify downstream steps work with filtered data

## Example Research Workflow

```bash
# 1. Run filter for all cohorts
python 4b_dtw_filter/run_dtw_filter_all_cohorts.py --skip-existing

# 2. Research a specific cohort
python 4b_dtw_filter/research_trajectories.py \
    --cohort-name opioid_ed \
    --age-band 0-12

# 3. Review findings and adjust parameters if needed
# 4. Re-run filter with adjusted parameters
python 4b_dtw_filter/filter_protocol_events.py \
    --cohort-name opioid_ed \
    --age-band 0-12 \
    --min-interval-days 7

# 5. Validate filtered data
# 6. Proceed with downstream feature engineering
```

## Next Steps

After completing research and filtering:

1. **FP-Growth**: Will automatically use filtered data (`model_events_no_protocols.parquet`)
2. **BupaR**: Use filtered data for process mining
3. **DTW Analysis**: Use filtered data for trajectory features
4. **Final Model**: Use filtered data for training

All downstream steps automatically prefer filtered data if available.
