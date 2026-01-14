# Step 5d: Trajectory Analysis (DTW)

This folder contains documentation for dynamic time warping trajectory analysis.

## Background: Research-First Approach

The DTW analysis workflow follows a **research-first approach**:

1. **Capture ALL Trajectories First**: Extract all time windows and common sequences of events (no filtering)
2. **Research & Classify**: Analyze trajectories to distinguish clinical/useful vs. non-clinical/protocol patterns
3. **Filter Non-Clinical Patterns** (`4b_dtw_filter`): Remove protocol-like events based on research
4. **Extract Clinical Features** (`5d_dtw_analysis`): Keep what's good - features that capture predictive patterns

**Key Principle**: Get all trajectories first, then research what goes where (filter vs. feature).

See [README_dtw_feature_extraction.md](README_dtw_feature_extraction.md) for detailed background and workflow.

## Documentation

- **[README_dtw_feature_extraction.md](README_dtw_feature_extraction.md)** - Dynamic Time Warping trajectory analysis

## Related Documentation

- **Step 5a**: See [`../Step5a_BupaR/`](../Step5a_BupaR/) for process mining
- **Step 5b**: See [`../Step5b_FPGrowth/`](../Step5b_FPGrowth/) for frequent pattern mining
- **Workflow**: See [`../CrossStep_Workflow/README_analysis_workflow.md`](../CrossStep_Workflow/README_analysis_workflow.md) for overall workflow
- **Main Index**: See [`../README.md`](../README.md) for complete documentation index

