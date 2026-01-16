# DTW Data Visualizations

## Overview

DTW (Dynamic Time Warping) analysis can create several types of visualizations to help understand patient trajectory patterns, clustering results, and similarity structures. However, **currently DTW visualizations are NOT automatically created in Step 3b**.

## Available DTW Visualization Scripts

### 1. `dtw_cohort_analysis.py` - Full DTW Analysis with Visualizations

This script performs complete DTW analysis including clustering and creates a **4-panel visualization**:

**Location:** `3b_feature_importance_eda/1_dtw/dtw_cohort_analysis.py`

**Visualizations Created (Single PNG file with 4 subplots):**

#### Panel 1: DTW Similarity Matrix Heatmap
- **Type:** Heatmap
- **Content:** DTW distance matrix between all patient pairs
- **Purpose:** Shows which patients have similar trajectories
- **Color scheme:** Viridis colormap (darker = more similar, lighter = more different)
- **Size:** Part of 2×2 grid in 15" × 12" figure

#### Panel 2: Cluster Size Distribution
- **Type:** Bar chart
- **Content:** Number of patients per cluster
- **Purpose:** Shows cluster balance and size distribution
- **X-axis:** Cluster ID
- **Y-axis:** Number of patients
- **Color:** Sky blue bars

#### Panel 3: Average Sequence Length by Cluster
- **Type:** Bar chart
- **Content:** Average trajectory length for each cluster
- **Purpose:** Shows which clusters have longer/shorter sequences
- **X-axis:** Cluster ID
- **Y-axis:** Average sequence length
- **Color:** Light coral bars

#### Panel 4: Top 10 Most Common Items Across All Clusters
- **Type:** Horizontal bar chart
- **Content:** Most frequently occurring items (drugs/ICD/CPT) across all clusters
- **Purpose:** Identifies common patterns across trajectory clusters
- **X-axis:** Total count
- **Y-axis:** Item names (top 10)
- **Color:** Light green bars

**Output File:** `dtw_analysis_results_{cohort}_{age_band}_{event_year}.png`

**Usage:**
```bash
python 3b_feature_importance_eda/1_dtw/dtw_cohort_analysis.py \
    --cohort opioid_ed \
    --age-band 13-24 \
    --event-year 2019 \
    --n-clusters 5
```

**Note:** This script is separate from the Step 3b pipeline and creates full clustering analysis with visualizations.

### 2. `create_dtw_features.py` - Feature Creation (No Visualizations)

**Location:** `3b_feature_importance_eda/1_dtw/create_dtw_features.py`

**Current Status:** This script does **NOT** create visualizations. It only:
- Extracts patient trajectories
- Computes DTW distances to prototype trajectories
- Creates feature CSV files for model training

**Output:** CSV files only (no PNG/PDF files)

## Current Step 3b Workflow

The Step 3b pipeline currently uses `create_dtw_features.py` which:
- ✅ Creates DTW feature CSV files
- ❌ Does NOT create visualizations

## Potential DTW Visualizations (Not Currently Created)

Based on the DTW analysis capabilities, the following visualizations could be created:

### 1. Trajectory Similarity Heatmap
- **Purpose:** Visualize DTW distance matrix
- **Shows:** Which patients have similar trajectories
- **Format:** Heatmap (viridis colormap)

### 2. Cluster Distribution
- **Purpose:** Show patient distribution across trajectory clusters
- **Shows:** Cluster sizes and balance
- **Format:** Bar chart

### 3. Sequence Length Analysis
- **Purpose:** Compare trajectory lengths across clusters
- **Shows:** Average sequence length per cluster
- **Format:** Bar chart

### 4. Common Items Visualization
- **Purpose:** Identify most frequent items in trajectories
- **Shows:** Top items across all clusters
- **Format:** Horizontal bar chart

### 5. Prototype Trajectory Visualization
- **Purpose:** Show representative trajectories for each cluster
- **Shows:** Archetype patterns
- **Format:** Line plot or sequence diagram

### 6. Trajectory Timeline
- **Purpose:** Visualize patient trajectories over time
- **Shows:** Sequence of events for sample patients
- **Format:** Gantt chart or timeline plot

## Integration Options

To add DTW visualizations to Step 3b:

### Option 1: Add Visualization Step to `create_dtw_features.py`
- Add matplotlib/seaborn plotting code
- Create visualizations after feature creation
- Save to `outputs/{cohort}/{age_band}/plots/dtw_*.png`

### Option 2: Call `dtw_cohort_analysis.py` in Step 3b
- Run full DTW analysis with clustering
- Generate comprehensive visualizations
- More computationally intensive

### Option 3: Create Separate Visualization Script
- Create `create_dtw_visualizations.py`
- Read DTW feature files
- Generate visualizations from features
- Called after feature creation

## Current Status

**DTW Visualizations in Step 3b:** ❌ **Not Created**

**Available Scripts:**
- `dtw_cohort_analysis.py` - Creates visualizations (not used in Step 3b)
- `create_dtw_features.py` - Creates features only (used in Step 3b)

**Recommendation:** If DTW visualizations are needed, consider adding a visualization step to the Step 3b pipeline or running `dtw_cohort_analysis.py` separately.
