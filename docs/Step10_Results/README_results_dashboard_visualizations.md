# Step 10: Dashboard Visualizations

## Overview

The dashboard includes three advanced visualization systems that complement the risk score by providing insights into patient pathways, frequent patterns, and trajectory similarities. All visualizations are filtered based on user-selected codes (drugs, ICDs, CPTs).

## Visualization Types

### 1. BupaR Process Mining

**Purpose**: Analyze patient pathways and sequences

**Visualizations**:
- **Sankey Diagrams**: Flow diagrams showing transitions between activities (drugs, ICDs, CPTs)
- **Process Matrices**: Transition frequency matrices showing common pathways
- **Trace Frequency Charts**: Bar charts of most frequent patient sequences

**Data Source**: `s3://pgxdatalake/gold/bupar/{cohort}/{age_band}/`

**Filtering**: Shows only pathways containing user-selected codes

**Example Use Cases**:
- "What pathways do patients with Drug X typically follow?"
- "How do patients progress from initial diagnosis to outcome?"
- "What are the most common sequences involving these codes?"

### 2. FP-Growth Frequent Patterns

**Purpose**: Discover association rules and frequent itemsets

**Visualizations**:
- **Association Rules Network**: Sankey diagram showing rules between codes
  - Nodes: Individual codes (drugs, ICDs, CPTs)
  - Edges: Association rules (antecedents → consequents)
  - Edge width: Confidence of the rule
- **Frequent Itemsets Bar Chart**: Top frequent code combinations

**Data Source**: `s3://pgxdatalake/gold/fpgrowth/cohort/cohort_name={cohort}/age_band={age_band}/`

**Filtering**: Shows only rules/itemsets containing user-selected codes

**Example Use Cases**:
- "What codes are frequently associated with Drug X?"
- "What patterns predict high risk?"
- "What are the most common code combinations?"

### 3. DTW Trajectory Clusters

**Purpose**: Analyze patient trajectory similarity and clustering

**Visualizations**:
- **Cluster Size Distribution**: Bar chart showing number of patients per cluster
- **Average DTW Distance**: Line chart showing average distance within clusters
- **Patient Trajectory Timelines**: Timeline visualization of representative trajectories

**Data Source**: `s3://pgxdatalake/gold/dtw_trajectories/{cohort}/{age_band}/`

**Filtering**: Shows only clusters containing trajectories with user-selected codes

**Example Use Cases**:
- "Which trajectory clusters contain patients with these codes?"
- "How similar are patient trajectories?"
- "What are the representative patterns for this patient group?"

## Integration with Risk Score

All visualizations complement the risk score by:

1. **Contextualizing Risk**: Showing how selected codes appear in patient pathways
2. **Pattern Discovery**: Revealing associations and sequences involving selected codes
3. **Trajectory Analysis**: Identifying which patient clusters contain similar patterns
4. **Causal Insights**: Supporting causal analysis by showing pathway impacts

## Data Flow

```
User Selects Codes → Risk Score Calculated → Visualizations Load
                                              ↓
                    Filter by Selected Codes → Display Filtered Visualizations
```

## Technical Implementation

### Frontend

- **Library**: Plotly.js (v2.27.0)
- **Rendering**: Dynamic chart generation based on API responses
- **Filtering**: Client-side filtering of visualization data
- **Lazy Loading**: Visualizations load when Tab 3 is opened

### Backend

- **Endpoints**: `GET /visualizations/{type}` (bupar, fpgrowth, dtw)
- **Filtering**: Server-side filtering based on selected codes
- **Data Sources**: S3 buckets for each visualization type
- **Error Handling**: Graceful degradation if data unavailable

### Data Filtering Logic

1. **BupaR**: Filter process matrices/traces containing selected codes
2. **FP-Growth**: Filter rules/itemsets intersecting with selected codes
3. **DTW**: Filter clusters containing trajectories with selected codes

## Related Documentation

- **[README_results_dashboard_tabs.md](README_results_dashboard_tabs.md)** - Dashboard tab organization and API endpoints
- **[README_results_dashboard.md](README_results_dashboard.md)** - Complete dashboard system overview
- **[../Step5a_BupaR/README_bupaR.md](../Step5a_BupaR/README_bupaR.md)** - BupaR analysis details
- **[../Step5b_FPGrowth/README_fpgrowth.md](../Step5b_FPGrowth/README_fpgrowth.md)** - FP-Growth analysis details
- **[../Step5d_DTW/README_dtw_overview.md](../Step5d_DTW/README_dtw_overview.md)** - DTW analysis details

