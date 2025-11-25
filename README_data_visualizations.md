# Data Visualizations

Visualization approaches, interpretation, and network analysis for the Prescription Drug Analysis pipeline.

## Overview

The pipeline generates various visualizations to help understand patterns, relationships, and model predictions. This document covers visualization approaches, interpretation guidelines, and common pitfalls.

## Visualization Approaches

### 1. Venn Diagrams
- Compare frequent itemsets vs. risk itemsets
- Identify overlapping patterns
- Highlight unique patterns in each analysis

### 2. Process Maps with Risk Overlay
- Base process map from BupaR
- Color-coded by risk influence
- Edge thickness based on frequency
- Node size based on FFA importance

### 3. Network Graphs
- Nodes: itemsets
- Edges: co-occurrence relationships
- Dual labels:
  - Process frequency
  - Risk weight
- Color coding for pattern alignment

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

