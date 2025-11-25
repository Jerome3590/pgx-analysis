# Data Visualizations

Visualization approaches, interpretation, and network analysis for the Prescription Drug Analysis pipeline.

## Overview

The pipeline generates various visualizations to help understand patterns, relationships, and model predictions. This document covers visualization approaches, interpretation guidelines, and common pitfalls.

## Visualization Approaches

The pipeline generates visualizations across multiple analysis methods. Each visualization type serves specific purposes in understanding patterns, relationships, and model predictions.

## Network Visualization Interpretation

❗ **Important Note:**
When interpreting the network visualizations generated from association rules, it's crucial to understand the distinction between correlation and causation.

🔍 **Why?**
Association rules (like those from FpGrowth) represent:
- Statistical co-occurrence between items in transactions
- E.g., "If drug A is present, drug B is also often present"

But they don't establish causality because they:
- Don't control for confounding variables
- Don't establish temporal precedence
- Don't use interventions to test effects

So while an arrow A → B is drawn (based on a rule), this is not a causal arrow — it represents a conditional probability relationship:
```
P(B | A) is high → draw A → B
```

✅ **Why the Visualization Is Still Valuable**
Even without causality:
- Summing support across multiple co-occurrence paths gives a meaningful measure of total association weight
- Directionality reflects rule direction (not causal flow)
- Node centrality indicates clustering or "hub" drugs often present in many co-occurrence patterns
- Edge thickness communicates real signal strength in the data

🧠 **Bottom Line**
| Interpretation | Is Valid? | Explanation |
|----------------|-----------|-------------|
| A → B is causal | ❌ | FpGrowth doesn't model interventions |
| A → B co-occur often | ✅ | Based on high confidence/support |
| Thicker edge = more total co-occurrence | ✅ | Sum(support) reflects total influence |

## Pattern Hashing and Attribution

The pattern mining process uses a sophisticated hashing and attribution system:

### 1. Itemset Hashing
- Each frequent itemset (from FpGrowth) is:
  - Turned into a pipe-separated string (e.g., "drug_x|drug_y")
  - Hashed using MD5 to generate a unique pattern_id
- Results in a pattern_lookup table:
  ```
  | pattern_id                         | itemsets          | support | ...metrics |
  |-----------------------------------|-------------------|---------|------------|
  | a8f72c99e5d1f4...                  | drug_x|drug_y     | 0.042   | ...        |
  | b04dd51b7926e2...                  | drug_z            | 0.089   | ...        |
  ```

### 2. Pattern Attribution
- Each row in the DataFrame has up to MAX_PATTERN_COLUMNS slots:
  ```
  | pattern_1       | pattern_2       | ... | pattern_15     |
  |-----------------|-----------------|-----|----------------|
  | b04dd...         | None            |     |                |
  | a8f72...         | b04dd...        |     |                |
  | None            | None            |     |                |
  ```
- These pattern_* columns reflect which pattern_ids (from pattern_lookup) were attributable to that row

### 3. Metric Merge
- Using merge_pattern_metrics(), for each pattern_i, the corresponding metrics are merged in from pattern_lookup:
  ```
  | pattern_1       | support_1 | confidence_1 | ...
  |-----------------|-----------|--------------|
  | a8f72...         | 0.042     | 0.62         |
  ```
- If a pattern was not matched or None, the row will have NaN or 0.0 after merge

#### Guarantees
- Each row only gets patterns it's eligible for — matched from rule/itemset presence
- Only patterns up to MAX_PATTERN_COLUMNS are attributed per row
- Patterns are attributed based on priority (e.g., support or rule quality) — usually highest scoring come first

#### Pattern Metrics
Each attributed pattern includes associated metrics:
- support_N: Frequency of the pattern in the dataset
- confidence_N: Confidence score for the pattern
- lift_N: Lift score indicating pattern significance
- certainty_N: Certainty factor for the pattern

Example schema after metric merge:
```
| pattern_1 | support_1 | confidence_1 | lift_1 | certainty_1 | pattern_2 | support_2 | ... |
|-----------|-----------|--------------|--------|-------------|-----------|-----------|-----|
| abc123... | 0.034     | 0.62         | 1.1    | 0.44        | def456... | 0.028     | ... |
| None      | NaN       | NaN          | NaN    | NaN         | None      | NaN       | ... |
```

### FPgrowth_Rank Variable

A new variable `FPgrowth_Rank` has been added to track the original ranking of network features. This variable:

- Stores the original rank (0-based index) of each active network feature
- For each row, contains a list of ranks for all network features where value = 1
- Uses -1 to indicate padded features (those added with zeros)

The rank information is valuable because:
- Higher ranked patterns (lower indices) were more frequent in the positive samples
- Helps identify the relative importance of different patterns
- Distinguishes between original and padded features
- Can be used to analyze the relationship between pattern rank and prediction accuracy

Example:
```python
# If network_feature_0 and network_feature_5 are active (value = 1)
# FPgrowth_Rank would contain [0, 5]
# If a feature was padded, its rank would be -1
```

## Association Rules and Co-Usage Analysis

### Purpose

We extract association rules from drug co-occurrence data using FP-Growth, filtering for positive-only patterns. These rules reveal structured relationships among drugs and serve as clinically interpretable features.

### What These Rules Show

Each rule takes the form:
```
IF {antecedent drugs} THEN {consequent drugs}
```

These rules represent:
- **Co-occurrence patterns**: Drugs that frequently appear together
- **Sequential relationships**: Temporal ordering of drug exposures
- **Clinical relevance**: Patterns that may indicate treatment protocols or adverse interactions

## Feature Importance Visualizations

The pipeline generates comprehensive feature importance visualizations from Monte Carlo Cross-Validation (MC-CV) analysis using CatBoost and Random Forest models. These visualizations help identify the most predictive features and understand their relative importance across different model quality metrics.

### 1. Top 50 Features Bar Chart

**File:** `{cohort}_{age}_{year}_top50_features.png`  
**Size:** 12" × 14"  
**Format:** Horizontal bar chart

**Description:**
- Displays the top 50 features ranked by scaled importance
- Features are sorted by `importance_scaled` (Recall-weighted importance)
- Bar height represents the scaled importance value
- Includes subtitle showing mean MC-CV Recall across models

**Interpretation:**
- Higher bars indicate features with greater predictive power
- Features appearing in both CatBoost and Random Forest models typically have higher importance
- Use this chart to identify the most critical features for model predictions

**Example Features Shown:**
- Drug names (e.g., HYDROCODONE-ACETAMINOPHEN, TRAMADOL HCL)
- ICD diagnosis codes (e.g., F11.20, F11.21)
- CPT procedure codes (e.g., 99281, 99282)

### 2. Top 50 Features with Recall Confidence

**File:** `{cohort}_{age}_{year}_top50_with_recall.png`  
**Size:** 12" × 14"  
**Format:** Horizontal bar chart with color gradient

**Description:**
- Shows top 50 features with dual encoding:
  - **Bar height**: Scaled importance value
  - **Bar color**: MC-CV Recall quality (with 95% confidence intervals)
- Color gradient: Orange (lower Recall) → Dark Blue (higher Recall)
- Includes error bars or confidence intervals for Recall estimates

**Interpretation:**
- **High importance + High Recall (Dark Blue)**: Highly predictive and reliable features
- **High importance + Low Recall (Orange)**: Predictive but less reliable features
- **Low importance + High Recall**: Reliable but less predictive features
- Use this chart to balance feature importance with model confidence

**Key Insights:**
- Features with both high importance and high Recall are most valuable
- Features with high importance but low Recall may need further investigation
- Helps prioritize features for clinical interpretation and model refinement

### 3. Normalized vs Recall-Scaled Comparison

**File:** `{cohort}_{age}_{year}_normalized_vs_scaled.png`  
**Size:** 12" × 14"  
**Format:** Side-by-side grouped bar chart

**Description:**
- Compares two importance metrics for top 50 features:
  - **Gray bars**: Normalized importance (raw sum across models)
  - **Blue bars**: Recall-scaled importance (weighted by model quality)
- Features sorted by scaled importance
- Shows impact of quality weighting on feature rankings

**Interpretation:**
- **Large difference**: Model quality weighting significantly affects feature ranking
- **Small difference**: Feature importance is consistent regardless of model quality
- Features that rank higher in scaled importance are more reliable predictors
- Helps understand which features benefit most from quality weighting

**Use Cases:**
- Identify features that are important but unreliable (high normalized, low scaled)
- Find features that are both important and reliable (high in both metrics)
- Understand the impact of model quality weighting on feature selection

### 4. Feature Category Distribution

**File:** `{cohort}_{age}_{year}_category_distribution.png`  
**Size:** 10" × 6"  
**Format:** Bar chart by category

**Description:**
- Shows distribution of top 50 features across three categories:
  - **Drug Names**: Prescription medications
  - **ICD Codes**: Diagnosis codes (e.g., F11.20)
  - **CPT Codes**: Procedure codes (e.g., 99281)
- Color-coded by category (steelblue, darkgreen, darkorange)

**Interpretation:**
- Reveals which feature types dominate the top features
- Helps understand the relative importance of different data sources
- Can identify if certain feature types are underrepresented

**Key Insights:**
- High drug name count: Medication patterns are most predictive
- High ICD code count: Diagnosis patterns are most predictive
- Balanced distribution: Multiple data sources contribute to predictions

### 5. Cross-Age-Band Heatmaps

**File:** `{cohort}_{year}_ageband_heatmap_top50.png`  
**Format:** Heatmap (Features × Age Bands)

**Description:**
- Compares feature importance across multiple age bands
- Rows: Top N features (typically 50)
- Columns: Age bands (e.g., 13-24, 25-44, 45-54, 55-64, 65-74)
- Color intensity: Scaled importance value
- Includes summary metrics (CV, consistency)

**Interpretation:**
- **Dark cells**: High importance for that age band
- **Light cells**: Low importance for that age band
- **Consistent rows**: Universal features (important across all age bands)
- **Variable rows**: Age-specific features (important only for certain age bands)

**Key Insights:**
- **Universal Risk Factors**: Features with consistent high importance across age bands
  - Low coefficient of variation (CV)
  - Suitable for age-agnostic models
- **Age-Specific Features**: Features with variable importance across age bands
  - High CV
  - May require age-stratified models
- **Missing Patterns**: Features important in some age bands but not others

**Use Cases:**
- Identify universal vs age-specific risk factors
- Decide between age-agnostic vs age-stratified models
- Understand how feature importance varies with patient age
- Validate model generalizability across age groups

**Example Analysis:**
```r
source("create_cross_ageband_heatmap.R")

create_ageband_heatmap(
  cohort_name = "opioid_ed",
  event_year = 2016,
  age_bands = c("13-24", "25-44", "45-54", "55-64", "65-74"),
  top_n = 50
)
```

**Outputs:**
- Heatmap visualization: Features × Age bands (color = importance)
- Summary CSV: Variability metrics (CV, consistency, mean importance)
- Insights: Universal vs age-specific features

### Storage Locations

**Local Storage:**
- `outputs/plots/` directory
- Files named: `{cohort}_{age}_{year}_{plot_type}.png`

**S3 Storage:**
- `s3://pgxdatalake/gold/feature_importance/cohort_name={cohort}/age_band={age}/event_year={year}/plots/`
- Automatically uploaded after generation

### Best Practices for Interpretation

1. **Compare Multiple Visualizations**: Use all four charts together for comprehensive understanding
2. **Consider Model Quality**: Prioritize features with both high importance and high Recall
3. **Check Consistency**: Features appearing in multiple visualizations are more reliable
4. **Age Band Analysis**: Use cross-age-band heatmaps to understand generalizability
5. **Clinical Context**: Always interpret feature importance in clinical context

## BupaR Process Mining Visualizations

BupaR process mining visualizations reveal temporal patterns and pathways in patient drug sequences. These visualizations help understand how patients progress through different drug exposure patterns over time.

### 1. Sankey Diagrams

**File:** `sankey_plot.html`  
**Format:** Interactive HTML visualization  
**Purpose:** Visualize flow of patients through drug sequences

**Description:**
- Interactive flow diagram showing patient pathways through drug sequences
- Nodes represent drugs or drug combinations
- Flow width represents number of patients following that pathway
- Color coding can represent different cohorts or outcomes
- Interactive features allow exploration of specific pathways

**Interpretation:**
- **Wide flows**: Common pathways followed by many patients
- **Narrow flows**: Less common but potentially important pathways
- **Branching points**: Decision points where patients diverge into different trajectories
- **Convergence**: Points where different pathways merge

**Use Cases:**
- Identify most common drug sequence patterns
- Compare pathways between target and control groups
- Discover critical decision points in patient trajectories
- Understand flow from initial drug exposure to outcomes

**Example Usage:**
```r
library(bupaR)
library(processmapR)

# Create event log from drug sequences
eventlog <- eventlog(
  data = drug_sequences,
  case_id = "mi_person_key",
  activity_id = "drug_name",
  timestamp = "event_date"
)

# Generate Sankey diagram
sankey <- eventlog %>%
  process_map(type = frequency("relative-consequent"))
```

**Storage Location:**
- `5_bupaR_analysis/sankey_plot.html`

### 2. Process Maps

**File:** `process_map_{type}.png`  
**Format:** Static or interactive process flow diagram  
**Types:** Frequency maps, performance maps, relative-consequent maps

**Description:**
- Visual representation of process flows discovered from event logs
- Nodes represent activities (drugs, procedures, diagnoses)
- Edges represent transitions between activities
- Edge thickness indicates frequency or performance metrics
- Node size can represent activity importance or frequency

**Types:**
1. **Frequency Maps**: Show how often activities occur and transitions happen
2. **Performance Maps**: Show timing information (throughput times, waiting times)
3. **Relative-Consequent Maps**: Show relative frequency of transitions

**Interpretation:**
- **Thick edges**: Common transitions between activities
- **Large nodes**: Frequently occurring activities
- **Long paths**: Complex sequences with many steps
- **Short paths**: Direct transitions between activities

**Use Cases:**
- Compare process flows between target and control groups
- Identify bottlenecks or delays in patient pathways
- Discover common vs. rare pathways
- Understand temporal relationships between events

**Example Usage:**
```r
# Frequency process map
process_map(eventlog, type = frequency("relative-consequent"))

# Performance process map
process_map(eventlog, type = performance(median, "days"))
```

### 3. Dotted Charts

**File:** `dotted_chart.png`  
**Format:** Temporal event visualization  
**Purpose:** Show event timing and sequence across cases

**Description:**
- Each row represents a patient case
- Each dot represents an event (drug, procedure, diagnosis)
- X-axis represents time
- Color coding can represent event types or outcomes
- Shows temporal patterns and sequence variations

**Interpretation:**
- **Vertical alignment**: Events occurring at similar times across patients
- **Horizontal spread**: Variability in event timing
- **Dense regions**: Common event sequences
- **Sparse regions**: Less common or outlier sequences

**Use Cases:**
- Identify temporal patterns in drug sequences
- Compare timing between different patient groups
- Detect outliers or unusual sequences
- Understand variability in patient trajectories

**Example Usage:**
```r
library(processmapR)

# Create dotted chart
dotted_chart(eventlog, x = "absolute", color = "drug_name")
```

### 4. Trace Alignment Visualizations

**File:** `trace_alignment.png`  
**Format:** Aligned sequence comparison  
**Purpose:** Compare patient traces against reference models

**Description:**
- Shows how individual patient traces align with reference process models
- Highlights deviations from expected pathways
- Color coding indicates conformance (green) vs. deviations (red)
- Helps identify patients following expected vs. unexpected paths

**Interpretation:**
- **Aligned traces**: Patients following expected pathways
- **Deviations**: Patients with unusual sequences
- **Common deviations**: Systematic variations from expected patterns
- **Rare deviations**: Outlier patients requiring investigation

**Use Cases:**
- Validate process conformance
- Identify patients with unusual trajectories
- Compare actual vs. expected pathways
- Quality assurance for process models

## DTW Trajectory Visualizations

Dynamic Time Warping (DTW) visualizations help understand patient trajectory similarity and clustering based on drug sequence patterns.

### 1. Similarity Matrix Heatmaps

**File:** `dtw_similarity_matrix.png` (part of multi-panel figure)  
**Format:** Heatmap visualization  
**Size:** Variable based on number of patients

**Description:**
- Shows DTW distances between all patient pairs
- Rows and columns represent individual patients
- Color intensity represents similarity (darker = more similar, lighter = less similar)
- Patients are typically reordered by cluster assignment
- Includes colorbar showing distance scale

**Interpretation:**
- **Dark blocks along diagonal**: Patients in same cluster (similar sequences)
- **Light regions**: Patients with dissimilar sequences
- **Block structure**: Clear cluster boundaries
- **Gradient patterns**: Gradual similarity transitions

**Use Cases:**
- Visualize patient similarity patterns
- Validate cluster assignments
- Identify outlier patients
- Understand cluster structure

**Example:**
```python
from dtw_cohort_analysis import DTWCohortAnalyzer

analyzer = DTWCohortAnalyzer(cohort_name="opioid_ed", age_band="25-44", event_year=2019)
results = analyzer.run_analysis(n_clusters=5, create_plots=True)
# Generates similarity matrix heatmap automatically
```

### 2. Cluster Size Distribution

**File:** `cluster_size_distribution.png` (part of multi-panel figure)  
**Format:** Bar chart  
**Purpose:** Show distribution of patients across clusters

**Description:**
- Bar chart showing number of patients in each cluster
- X-axis: Cluster ID
- Y-axis: Number of patients
- Color coding can represent cluster characteristics

**Interpretation:**
- **Large bars**: Dominant trajectory patterns
- **Small bars**: Rare but potentially important patterns
- **Balanced distribution**: Multiple distinct trajectory types
- **Skewed distribution**: One or few dominant patterns

**Use Cases:**
- Understand cluster composition
- Identify dominant vs. rare trajectory patterns
- Validate clustering quality
- Guide further analysis of specific clusters

### 3. Average Sequence Length by Cluster

**File:** `avg_sequence_length.png` (part of multi-panel figure)  
**Format:** Bar chart  
**Purpose:** Compare sequence complexity across clusters

**Description:**
- Shows average drug sequence length for each cluster
- X-axis: Cluster ID
- Y-axis: Average sequence length (number of drugs)
- Helps understand trajectory complexity

**Interpretation:**
- **Long sequences**: Complex drug exposure patterns
- **Short sequences**: Simple or focused drug patterns
- **Variability**: Different clusters have different complexity levels
- **Clinical relevance**: Longer sequences may indicate polypharmacy

**Use Cases:**
- Compare trajectory complexity
- Identify clusters with complex vs. simple patterns
- Understand polypharmacy patterns
- Guide clinical interpretation

### 4. Most Common Drugs by Cluster

**File:** `most_common_drugs.png` (part of multi-panel figure)  
**Format:** Horizontal bar chart  
**Purpose:** Identify characteristic drugs for each cluster

**Description:**
- Shows top N most common drugs across all clusters
- Horizontal bars sorted by total frequency
- Can be broken down by cluster to show cluster-specific drugs
- Helps identify cluster-defining medications

**Interpretation:**
- **High frequency drugs**: Common across many patients
- **Cluster-specific drugs**: Characteristic of particular trajectory types
- **Drug diversity**: Variety of medications in trajectories
- **Therapeutic patterns**: Drug combinations indicating treatment protocols

**Use Cases:**
- Identify characteristic drugs for each trajectory type
- Understand therapeutic patterns
- Compare drug usage across clusters
- Guide clinical interpretation of trajectory clusters

### 5. Trajectory Timeline Visualizations

**File:** `trajectory_timelines.png`  
**Format:** Multi-panel timeline plots  
**Purpose:** Visualize temporal progression of drug sequences

**Description:**
- Shows drug sequences over time for representative patients from each cluster
- Each panel represents a cluster
- X-axis: Time (days or months)
- Y-axis: Drug names or drug categories
- Color coding can represent drug classes or outcomes

**Interpretation:**
- **Temporal patterns**: How drug sequences evolve over time
- **Cluster differences**: Distinct temporal patterns for each cluster
- **Sequence stability**: Consistent vs. variable patterns
- **Transition points**: When patients move between drug types

**Use Cases:**
- Understand temporal progression of drug sequences
- Compare trajectory archetypes
- Identify critical transition points
- Guide clinical interpretation

**Storage Location:**
- `s3://pgxdatalake/gold/dtw_trajectories/{cohort_name}/{age_band}/{event_year}/visualizations/`

## FFA Analysis Visualizations

Formal Feature Attribution (FFA) visualizations help understand model predictions, feature importance, and causal relationships.

### 1. Cattail Plots

**File:** `cattail_plots/{class}_{feature}.png`  
**Format:** Distribution plots  
**Purpose:** Show feature value distributions for top important features

**Description:**
- Distribution plots showing values of top 10 important features
- Separate plots for each class (target vs. control)
- Shows range, frequency, and thresholds in decision rules
- Helps understand feature value patterns

**Interpretation:**
- **Distribution shape**: Normal, skewed, or bimodal patterns
- **Threshold identification**: Decision boundaries in model rules
- **Class differences**: How feature values differ between classes
- **Outliers**: Unusual feature values requiring investigation

**Use Cases:**
- Understand feature value distributions
- Identify decision thresholds
- Compare feature values between classes
- Detect data quality issues

**Storage Location:**
- `s3://pgxdatalake/gold/final_model/cohort_name={cohort}/age_band={age}/event_year={year}/cattail_plots/`

### 2. Mirror Plots

**File:** `mirror_plots/{feature}_mirror.png`  
**Format:** Side-by-side comparison plots  
**Purpose:** Compare feature importance between classes

**Description:**
- Side-by-side bar charts comparing feature importance for each class
- Left side: Class 0 (controls)
- Right side: Class 1 (targets)
- Shows relative importance and direction of effect
- Helps identify class-specific important features

**Interpretation:**
- **Symmetric patterns**: Features important for both classes
- **Asymmetric patterns**: Class-specific important features
- **Direction**: Whether feature increases or decreases risk
- **Magnitude**: Relative importance for each class

**Use Cases:**
- Compare feature importance between classes
- Identify class-specific risk factors
- Understand differential feature effects
- Guide clinical interpretation

**Storage Location:**
- `s3://pgxdatalake/gold/final_model/cohort_name={cohort}/age_band={age}/event_year={year}/mirror_plots/`

### 3. SHAP Plots

**File:** `shap_plots/{class}_shap_summary.png`  
**Format:** Beeswarm plots and summary plots  
**Purpose:** Show feature attribution for individual predictions

**Description:**
- SHAP (SHapley Additive exPlanations) value visualizations
- Beeswarm plots showing feature contributions for each prediction
- Summary plots showing overall feature importance
- Waterfall plots for individual predictions
- Color coding indicates feature value (low to high)

**Interpretation:**
- **Feature importance**: Vertical position indicates importance
- **Feature value**: Color indicates low (blue) to high (red) values
- **Effect direction**: Positive (right) vs. negative (left) contributions
- **Interaction effects**: How features interact in predictions

**Use Cases:**
- Understand individual prediction explanations
- Identify most important features for predictions
- Compare feature effects across predictions
- Validate model interpretability

**Storage Location:**
- `s3://pgxdatalake/gold/final_model/cohort_name={cohort}/age_band={age}/event_year={year}/shap_plots/`

### 4. Calibration Plots

**File:** `calibration_plots/{class}_calibration.png`  
**Format:** Calibration curve plots  
**Purpose:** Assess model calibration and prediction reliability

**Description:**
- Shows predicted probability vs. observed frequency
- Ideal calibration: 45-degree line (perfectly calibrated)
- Bins predictions into groups and compares to actual outcomes
- Includes confidence intervals for calibration estimates
- Separate plots for each class

**Interpretation:**
- **Above diagonal**: Model overconfident (predictions too high)
- **Below diagonal**: Model underconfident (predictions too low)
- **Close to diagonal**: Well-calibrated predictions
- **Binning patterns**: How calibration varies across probability ranges

**Use Cases:**
- Validate model calibration
- Assess prediction reliability
- Identify probability ranges needing recalibration
- Compare calibration across classes

**Storage Location:**
- `s3://pgxdatalake/gold/final_model/cohort_name={cohort}/age_band={age}/event_year={year}/calibration_plots/`

### 5. Causal Relationship Visualizations

**File:** `causal_relationships.png`  
**Format:** Multi-panel figure  
**Purpose:** Show causal relationships and feature effects

**Description:**
- Multi-panel visualization including:
  1. **Top 20 Features by Causal Importance**: Bar chart ranking features
  2. **Feature Value Distributions**: KDE plots for top 5 causal features
  3. **Correlation Matrix**: Heatmap of correlations between top 10 features
- Shows causal importance, distributions, and relationships
- Helps understand causal structure

**Interpretation:**
- **Causal importance**: Features with strongest causal effects
- **Distribution differences**: How feature values differ between groups
- **Correlations**: Relationships between causal features
- **Confounding**: Potential confounding relationships

**Use Cases:**
- Understand causal relationships
- Identify key causal drivers
- Detect confounding factors
- Guide causal inference

**Example:**
```python
from ffa_analysis import FFAAnalyzer

analyzer = FFAAnalyzer(train_df, test_df, model, explainer)
df_metrics = analyzer.analyze_features()
analyzer.plot_causal_relationships(df_metrics, X_test, save_path="outputs/")
```

**Storage Location:**
- `s3://pgxdatalake/gold/ffa_analysis/cohort_name={cohort}/age_band={age}/event_year={year}/`

## Network Graphs (FP-Growth Association Rules)

Network graphs visualize association rules discovered through FP-Growth pattern mining, showing relationships between drugs, ICD codes, and CPT codes.

### Description

**File:** `network_graph_{cohort}_{age}_{year}.png` or `.html`  
**Format:** Interactive network graph or static visualization  
**Purpose:** Visualize co-occurrence patterns and association rules

**Components:**
- **Nodes**: Represent itemsets (drugs, ICD codes, CPT codes, or combinations)
- **Edges**: Represent association rules or co-occurrence relationships
- **Node size**: Can represent support, importance, or frequency
- **Edge thickness**: Represents rule confidence or support
- **Edge direction**: Represents rule direction (antecedent → consequent)
- **Color coding**: Can represent feature type, cluster, or importance

### Types of Network Graphs

1. **Association Rule Networks**
   - Nodes: Individual items (drugs, codes)
   - Edges: Association rules (IF-THEN relationships)
   - Shows which items frequently co-occur
   - Direction indicates rule direction

2. **Itemset Networks**
   - Nodes: Frequent itemsets (combinations of items)
   - Edges: Subset/superset relationships
   - Shows hierarchical structure of patterns
   - Helps understand pattern relationships

3. **Co-occurrence Networks**
   - Nodes: Items
   - Edges: Co-occurrence frequency
   - Undirected edges showing mutual co-occurrence
   - Simpler representation of item relationships

### Interpretation Guidelines

**Node Centrality:**
- **High centrality**: Items appearing in many patterns (hub items)
- **Low centrality**: Items appearing in few patterns (specialized items)
- **Betweenness**: Items connecting different pattern groups

**Edge Interpretation:**
- **Thick edges**: Strong associations (high confidence/support)
- **Thin edges**: Weak associations (low confidence/support)
- **Direction**: Rule direction (not necessarily causal)
- **Multiple paths**: Redundant or reinforcing relationships

**Cluster Identification:**
- **Dense clusters**: Groups of frequently co-occurring items
- **Sparse regions**: Items that rarely co-occur
- **Bridge nodes**: Items connecting different clusters
- **Isolated nodes**: Items with few associations

### Use Cases

1. **Pattern Discovery**
   - Identify common drug combinations
   - Discover co-occurring diagnoses and procedures
   - Find unexpected associations

2. **Clinical Interpretation**
   - Understand treatment protocols
   - Identify drug interaction patterns
   - Discover diagnostic patterns

3. **Feature Engineering**
   - Identify important itemsets for model features
   - Understand feature relationships
   - Guide feature selection

4. **Quality Assurance**
   - Validate expected clinical patterns
   - Detect data quality issues
   - Identify outliers or anomalies

### Best Practices

1. **Filter by Support**: Focus on patterns with sufficient frequency
2. **Consider Confidence**: Prioritize high-confidence rules
3. **Clinical Context**: Always interpret in clinical context
4. **Avoid Causality Claims**: Remember these show association, not causation
5. **Compare Cohorts**: Compare networks between target and control groups

### Storage Locations

- **Local**: `outputs/network_graphs/`
- **S3**: `s3://pgxdatalake/gold/fpgrowth_analysis/cohort_name={cohort}/age_band={age}/event_year={year}/network_graphs/`

## Visualization Insights

### Pattern Analysis Visualizations

1. **Pattern Clustering Heatmaps**
   - Reveals groups of similar drug patterns
   - Shows hierarchical relationships between patterns
   - Helps identify common co-occurring drug combinations
   - Highlights patterns with similar support levels

2. **Feature Relationship Plots**
   - Shows correlations between different drug features
   - Identifies strongly associated drug pairs
   - Helps understand feature dependencies
   - Reveals potential confounding relationships

3. **Feature Distributions**
   - Shows the distribution of drug usage patterns
   - Identifies common vs. rare drug combinations
   - Helps understand the prevalence of different patterns
   - Reveals potential outliers or unusual patterns

### Cattail Visualizations

1. **Feature Value Distributions**
   - Shows the distribution of values for top 10 important features
   - Helps understand the range and frequency of drug usage
   - Identifies common thresholds in decision rules
   - Reveals potential data quality issues or outliers

2. **Decision Rule Visualizations**
   - Displays the extracted rules from the CatBoost model
   - Shows support and confidence metrics for each rule
   - Helps understand the model's decision-making process
   - Identifies the most influential rules

3. **Pattern Metrics Plots**
   - Shows coverage and confidence of different patterns
   - Helps identify reliable vs. rare patterns
   - Reveals the predictive power of different combinations
   - Aids in understanding pattern stability

### Causal Analysis Visualizations

1. **Causal Relationship Heatmaps**
   - Shows the strength of causal relationships between features
   - Identifies direct and indirect causal effects
   - Helps understand the impact of interventions
   - Reveals potential confounding factors

2. **Feature Importance Plots**
   - Ranks features by their causal importance
   - Shows the relative impact of different drugs
   - Helps identify key drivers of outcomes
   - Aids in understanding feature stability

3. **Correlation Matrix Visualizations**
   - Shows pairwise correlations between top features
   - Helps identify multicollinearity
   - Reveals potential feature redundancies
   - Aids in understanding feature dependencies

## Interpreting the Results

### Pattern Analysis
- **High Support Patterns**: Common drug combinations that appear frequently in the data
- **Hierarchical Clusters**: Groups of related patterns that share similar characteristics
- **Feature Correlations**: Strong associations between different drugs or features

### Cattail Insights
- **Decision Rules**: Clear, interpretable rules that explain model predictions
- **Pattern Coverage**: How widely applicable each pattern is in the dataset
- **Confidence Levels**: How reliable the patterns are in predicting outcomes

### Causal Analysis
- **Causal Relationships**: Direct and indirect effects between features and outcomes
- **Feature Importance**: Relative impact of different features on predictions
- **Confounding Factors**: Variables that may influence both features and outcomes

## Common Pitfalls and Misinterpretations

### Pattern Analysis Pitfalls
- **Confusing correlation with causation**: Association rules show co-occurrence, not causality
- **Overinterpreting low-support patterns**: Rare patterns may not be generalizable
- **Ignoring temporal ordering**: Sequence matters in drug exposure analysis

### Cattail Visualization Pitfalls
- **Misinterpreting decision rules**: Rules describe patterns, not deterministic outcomes
- **Overfitting to training data**: Patterns may not generalize to new data
- **Ignoring confidence intervals**: Uncertainty should be considered in interpretation

### Causal Analysis Pitfalls
- **Assuming causality from correlation**: Additional evidence needed for causal claims
- **Ignoring confounding variables**: Unmeasured confounders may bias results
- **Overinterpreting small effects**: Effect sizes should be considered in context

### Data Quality Issues
- **Missing data**: Patterns may be incomplete due to missing values
- **Data preprocessing artifacts**: Cleaning steps may introduce biases
- **Temporal misalignment**: Events may not be properly ordered chronologically

## Best Practices for Avoiding Pitfalls

1. **Always consider context**: Clinical knowledge should inform interpretation
2. **Validate patterns**: Cross-validate findings across different cohorts or time periods
3. **Report uncertainty**: Include confidence intervals and uncertainty measures
4. **Consider multiple explanations**: Alternative hypotheses should be explored
5. **Document assumptions**: Clearly state assumptions and limitations

## Related Documentation

- [`README_overview.md`](README_overview.md) - Project structure and components
- [`README_data_pipeline.md`](README_data_pipeline.md) - Data processing and cohort creation
- [`README_analysis_workflow.md`](README_analysis_workflow.md) - Feature importance and pattern mining
- [`docs/README_data_visualization.md`](docs/README_data_visualization.md) - Detailed visualization guide

