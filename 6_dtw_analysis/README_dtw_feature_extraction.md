# DTW Analysis and Feature Extraction

This module provides Dynamic Time Warping (DTW) analysis capabilities for comparing drug exposure sequences across patients in cohort datasets, and outlines the features to extract from DTW analysis outputs for use in the final CatBoost/Random Forest model.

## Overview

DTW analysis helps identify patients with similar drug exposure patterns by measuring the similarity between temporal sequences that may vary in timing and length. This is particularly useful for:

- **Patient Clustering**: Group patients with similar drug histories
- **Outlier Detection**: Identify patients with unusual drug sequences
- **Pattern Discovery**: Find common drug exposure patterns
- **Risk Assessment**: Compare new patients to known high-risk patterns
- **Feature Engineering**: Extract trajectory features for machine learning models

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure you have access to the S3 bucket containing cohort data.

## Usage

### Basic Usage

```bash
python dtw_cohort_analysis.py \
    --cohort opioid_ed \
    --age-band "65-74" \
    --event-year 2020 \
    --n-clusters 5
```

### Command Line Arguments

- `--cohort`: Cohort to analyze (`opioid_ed` or `ed_non_opioid`)
- `--age-band`: Age band (e.g., "0-12", "65-74")
- `--event-year`: Event year (e.g., 2020)
- `--n-clusters`: Number of clusters to create (default: 5)
- `--no-plots`: Skip creating visualizations

### Examples

**Analyze opioid ED cohort:**
```bash
python dtw_cohort_analysis.py \
    --cohort opioid_ed \
    --age-band "65-74" \
    --event-year 2020 \
    --n-clusters 6
```

**Analyze ADE cohort without plots:**
```bash
python dtw_cohort_analysis.py \
    --cohort ed_non_opioid \
    --age-band "18-25" \
    --event-year 2019 \
    --n-clusters 4 \
    --no-plots
```

## Output

The analysis generates several outputs:

### 1. Visualizations
- **Similarity Matrix Heatmap**: Shows DTW distances between all patient pairs
- **Cluster Size Distribution**: Bar chart of patients per cluster
- **Average Sequence Length**: Drug sequence length by cluster
- **Most Common Drugs**: Top drugs across all clusters

### 2. S3 Storage
Results are saved to S3 with the following structure:
```
s3://{S3_BUCKET}/dtw_trajectories/{cohort_name}/{age_band}/{event_year}/
├── trajectory_results_{item_type}.json    # Complete analysis results
└── patient_trajectories_{item_type}.parquet # Patient trajectories
```

### 3. Analysis Results

The JSON results include:
- **Metadata**: Analysis parameters and summary statistics
- **Clustering Results**: Cluster assignments and quality metrics
- **Cluster Characteristics**: Detailed analysis of each cluster
- **Trajectory Clusters**: Patient-to-cluster mappings
- **Drug Encoding Map**: Mapping of drug names to numerical IDs (for drugs)

## Key Metrics

### Silhouette Score
Measures cluster quality (range: -1 to 1, higher is better)

### Cluster Characteristics
- Number of patients per cluster
- Average sequence length
- Most common items in each cluster
- Therapeutic class distribution (for drugs)
- Temporal characteristics (days to events)

## Integration with Pipeline

This DTW analysis complements the existing pipeline by:

1. **Working with Drug Event Explosion**: Uses the exploded drug-level data
2. **Supporting BupaR Analysis**: DTW clusters can be used for cluster-specific process mining
3. **Enhancing Feature Engineering**: DTW similarity scores can become model features
4. **Providing Clinical Insights**: Identifies similar patient patterns for clinical decision support

## Advanced Usage

### Custom Analysis

```python
from dtw_cohort_analysis import DTWCohortAnalyzer

# Create analyzer
analyzer = DTWCohortAnalyzer("65-74", 2020, "opioid_ed")

# Run custom analysis
results = analyzer.run_analysis(n_clusters=8, create_plots=True)

# Access results
cluster_results = results['cluster_results']
cluster_characteristics = results['cluster_characteristics']
```

### Batch Processing

```bash
# Process multiple cohorts
for cohort in opioid_ed ed_non_opioid; do
    for age_band in "0-12" "13-17" "18-25" "26-35" "36-45" "46-55" "56-65" "65-74"; do
        python dtw_cohort_analysis.py \
            --cohort $cohort \
            --age-band $age_band \
            --event-year 2020 \
            --n-clusters 5
    done
done
```

## Troubleshooting

### Common Issues

1. **DTW Package Not Available**
   ```
   Error: dtaidistance package not available.
   Install with: pip install dtaidistance
   ```

2. **Memory Issues with Large Datasets**
   - Reduce number of patients by filtering
   - Use smaller number of clusters
   - Consider sampling for initial analysis

3. **S3 Access Issues**
   - Ensure AWS credentials are configured
   - Check S3 bucket permissions
   - Verify cohort data exists

### Performance Tips

- **Small Datasets**: Use default settings
- **Large Datasets**: Consider sampling or filtering
- **Memory Optimization**: Process in batches for very large cohorts
- **Parallel Processing**: Run multiple age bands/cohorts in parallel

---

## Feature Extraction for Final Model

This section outlines the features to extract from DTW analysis outputs for use in the final CatBoost/Random Forest model. These features complement FPGrowth itemsets and BupaR process patterns.

## 🎯 Key Features to Extract

### 1. **Trajectory Cluster Membership** (Categorical)
**What**: Which trajectory cluster does each patient belong to?

**Extraction**:
```python
# From: trajectory_results_{item_type}.json
patient_cluster_map = results['trajectory_clusters']['patient_cluster_map']
# Feature: trajectory_cluster_drug, trajectory_cluster_icd, trajectory_cluster_cpt
```

**Why**: Patients in the same trajectory cluster have similar sequences, which may correlate with outcomes.

**Usage**: 
- Categorical feature for CatBoost
- One-hot encoded for Random Forest
- Multiple versions: `trajectory_cluster_drug`, `trajectory_cluster_icd`, `trajectory_cluster_cpt`

---

### 2. **Trajectory Similarity Scores** (Continuous)
**What**: How similar is this patient's trajectory to other patients (especially target cases)?

**Extraction**:
```python
# Calculate average DTW distance to target cases in same cluster
# Or: distance to cluster archetype
# Feature: avg_dtw_distance_to_targets, dtw_distance_to_archetype
```

**Why**: Patients with trajectories similar to known target cases may be at higher risk.

**Metrics to Extract**:
- `dtw_distance_to_cluster_archetype_{item_type}`: Distance to cluster's representative trajectory
- `avg_dtw_distance_to_targets_{item_type}`: Average distance to all target cases in dataset
- `min_dtw_distance_to_targets_{item_type}`: Minimum distance to any target case
- `dtw_distance_to_closest_target_{item_type}`: Distance to nearest target case

---

### 3. **Trajectory Characteristics** (Continuous/Categorical)
**What**: Basic properties of the patient's trajectory sequence.

**Extraction**:
```python
# From: patient_trajectories_{item_type}.parquet
# Features: trajectory_length, trajectory_diversity, temporal_span, temporal_density
```

**Features**:
- `trajectory_length_{item_type}`: Number of events in sequence
- `trajectory_diversity_{item_type}`: Number of unique items in sequence
- `trajectory_temporal_span_{item_type}`: Days between first and last event
- `trajectory_temporal_density_{item_type}`: Events per month (length / (span / 30))
  - **Note**: Monthly rate is more interpretable for medical procedures (CPT codes) and diagnoses (ICD codes), which are scheduled events rather than daily occurrences. Also provides consistent scale across all item types (drugs, ICD, CPT).
- `trajectory_has_temporal_alignment_{item_type}`: Binary (1 if days_to_target_event available, 0 otherwise)

**Why**: Longer, more diverse trajectories may indicate different risk profiles. Monthly event density provides a clinically meaningful measure of healthcare utilization frequency.

---

### 4. **Cluster-Specific Characteristics** (Continuous)
**What**: Properties of the cluster this patient belongs to.

**Extraction**:
```python
# From: trajectory_patterns in results
# Features: cluster_avg_length, cluster_target_rate, cluster_size
```

**Features**:
- `cluster_size_{item_type}`: Number of patients in this cluster
- `cluster_avg_trajectory_length_{item_type}`: Average sequence length in cluster
- `cluster_target_rate_{item_type}`: Proportion of target cases in cluster
- `cluster_silhouette_score_{item_type}`: Quality of cluster (from DTW analysis)

**Why**: Cluster properties provide context about the patient's trajectory group.

---

### 5. **Archetype Matching** (Continuous)
**What**: How well does the patient's trajectory match the cluster archetype?

**Extraction**:
```python
# Calculate DTW distance to cluster archetype
# Feature: archetype_match_score (1 / (1 + dtw_distance))
```

**Features**:
- `archetype_match_score_{item_type}`: Similarity to cluster archetype (normalized 0-1)
- `archetype_match_rank_{item_type}`: Rank of match quality within cluster (1 = best match)

**Why**: Patients matching archetypes closely may have more typical (or atypical) patterns.

---

### 6. **Temporal Alignment Features** (Continuous, ED_NON_OPIOID only)
**What**: For ED_NON_OPIOID cohort, temporal positioning relative to target event.

**Extraction**:
```python
# From: temporal_positions in patient_trajectories
# Features: first_event_days_before_target, last_event_days_before_target, etc.
```

**Features**:
- `first_event_days_before_target_{item_type}`: Days from first event to target
- `last_event_days_before_target_{item_type}`: Days from last event to target
- `trajectory_temporal_mean_{item_type}`: Mean temporal position
- `trajectory_temporal_std_{item_type}`: Std dev of temporal positions
- `trajectory_temporal_skew_{item_type}`: Skewness of temporal distribution

**Why**: Timing of events relative to target may be predictive.

---

### 7. **Multi-Modal Trajectory Features** (Categorical/Continuous)
**What**: Cross-modal trajectory relationships.

**Extraction**:
```python
# Compare clusters across item types
# Features: drug_icd_cluster_alignment, drug_cpt_cluster_alignment
```

**Features**:
- `drug_icd_cluster_match`: Binary (1 if drug and ICD clusters are correlated)
- `drug_cpt_cluster_match`: Binary (1 if drug and CPT clusters are correlated)
- `multi_modal_cluster_consistency`: Categorical (same cluster across all types, mixed, etc.)

**Why**: Consistency across trajectory types may indicate stronger patterns.

---

## 📊 Feature Extraction Implementation

### Recommended Feature Set

**Patient-Level Features** (one row per `mi_person_key`):

| Feature Name | Type | Source | Description |
|-------------|------|--------|-------------|
| `trajectory_cluster_drug` | Categorical | DTW drug analysis | Drug trajectory cluster ID |
| `trajectory_cluster_icd` | Categorical | DTW ICD analysis | ICD trajectory cluster ID |
| `trajectory_cluster_cpt` | Categorical | DTW CPT analysis | CPT trajectory cluster ID |
| `trajectory_length_drug` | Continuous | Patient trajectories | Number of drug events |
| `trajectory_length_icd` | Continuous | Patient trajectories | Number of ICD events |
| `trajectory_length_cpt` | Continuous | Patient trajectories | Number of CPT events |
| `trajectory_diversity_drug` | Continuous | Patient trajectories | Unique drugs in sequence |
| `trajectory_diversity_icd` | Continuous | Patient trajectories | Unique ICD codes in sequence |
| `trajectory_diversity_cpt` | Continuous | Patient trajectories | Unique CPT codes in sequence |
| `dtw_distance_to_archetype_drug` | Continuous | Similarity matrix | Distance to drug cluster archetype |
| `dtw_distance_to_archetype_icd` | Continuous | Similarity matrix | Distance to ICD cluster archetype |
| `dtw_distance_to_archetype_cpt` | Continuous | Similarity matrix | Distance to CPT cluster archetype |
| `archetype_match_score_drug` | Continuous | Calculated | Normalized match to drug archetype |
| `archetype_match_score_icd` | Continuous | Calculated | Normalized match to ICD archetype |
| `archetype_match_score_cpt` | Continuous | Calculated | Normalized match to CPT archetype |
| `cluster_target_rate_drug` | Continuous | Cluster analysis | Target rate in drug cluster |
| `cluster_target_rate_icd` | Continuous | Cluster analysis | Target rate in ICD cluster |
| `cluster_target_rate_cpt` | Continuous | Cluster analysis | Target rate in CPT cluster |
| `avg_dtw_distance_to_targets_drug` | Continuous | Similarity matrix | Avg distance to target cases |
| `min_dtw_distance_to_targets_drug` | Continuous | Similarity matrix | Min distance to any target |
| `first_event_days_before_target_drug` | Continuous | Temporal positions | Days from first drug to target (ED_NON_OPIOID) |
| `last_event_days_before_target_drug` | Continuous | Temporal positions | Days from last drug to target (ED_NON_OPIOID) |
| `trajectory_temporal_density_drug` | Continuous | Calculated | Events per month (ED_NON_OPIOID) |

**Total: ~20-25 trajectory features per patient**

---

## 🔧 Extraction Script Structure

### Step 1: Load DTW Results
```python
# Load trajectory results for each item type
drug_results = load_from_s3_json("s3://.../trajectory_results_drug.json")
icd_results = load_from_s3_json("s3://.../trajectory_results_icd.json")
cpt_results = load_from_s3_json("s3://.../trajectory_results_cpt.json")

# Load patient trajectories
drug_trajectories = pd.read_parquet("s3://.../patient_trajectories_drug.parquet")
icd_trajectories = pd.read_parquet("s3://.../patient_trajectories_icd.parquet")
cpt_trajectories = pd.read_parquet("s3://.../patient_trajectories_cpt.parquet")
```

### Step 2: Extract Cluster Memberships
```python
# Map patients to clusters
patient_clusters = {
    'drug': drug_results['trajectory_clusters']['patient_cluster_map'],
    'icd': icd_results['trajectory_clusters']['patient_cluster_map'],
    'cpt': cpt_results['trajectory_clusters']['patient_cluster_map']
}
```

### Step 3: Calculate Trajectory Characteristics
```python
# For each patient, calculate:
# - Trajectory length
# - Trajectory diversity
# - Temporal characteristics (if available)
# - Archetype distances
```

### Step 4: Calculate Cluster Properties
```python
# For each cluster, calculate:
# - Cluster size
# - Cluster target rate
# - Cluster average trajectory length
```

### Step 5: Create Feature DataFrame
```python
# Merge all features into patient-level dataframe
# One row per mi_person_key
dtw_features = pd.DataFrame({
    'mi_person_key': patient_ids,
    'trajectory_cluster_drug': ...,
    'trajectory_cluster_icd': ...,
    # ... all other features
})
```

---

## 🎯 Integration with Final Model

### Feature Engineering Pipeline

```
1. Load Cohort Data (with FPGrowth itemsets, BupaR patterns)
   ↓
2. Load DTW Results (trajectory clusters, archetypes, similarities)
   ↓
3. Extract DTW Features (cluster memberships, characteristics, similarities)
   ↓
4. Merge with FPGrowth Features (frequent itemsets, association rules)
   ↓
5. Merge with BupaR Features (process patterns, sequence features)
   ↓
6. Create Final Feature Matrix (patient-level, all features combined)
   ↓
7. Train CatBoost/Random Forest Model
```

### Feature Categories in Final Model

| Category | Features | Count |
|----------|----------|-------|
| **FPGrowth** | Frequent itemsets, association rules | ~100-500 |
| **BupaR** | Process patterns, sequence features | ~50-200 |
| **DTW** | Trajectory clusters, similarities, characteristics | ~20-25 |
| **Demographics** | Age, gender, race, location | ~10-15 |
| **Temporal** | Event dates, temporal windows | ~5-10 |
| **Total** | | ~185-750 |

---

## 📝 Recommendations

### For ED_NON_OPIOID Cohort:
1. **Focus on drug trajectories** (primary research question)
2. **Extract temporal alignment features** (`days_to_target_event` available)
3. **Use archetype matching** to identify typical drug sequences
4. **Compare drug trajectories** to known target patterns

### For OPIOID_ED Cohort:
1. **Focus on ICD and CPT trajectories** (diagnostic/procedure patterns)
2. **Extract multi-modal features** (drug + ICD + CPT cluster alignment)
3. **Use cluster target rates** to identify high-risk trajectory patterns
4. **Compare trajectories** across all three item types

### General Best Practices:
1. **Start with cluster memberships** (simplest, most interpretable)
2. **Add trajectory characteristics** (length, diversity, temporal properties)
3. **Include similarity scores** (distance to targets, archetypes)
4. **Use cluster properties** (target rates, sizes) for context
5. **Test feature importance** in CatBoost to identify most predictive DTW features

---

## 🔍 Feature Validation

### Check Feature Quality:
- **Missing values**: Handle patients not in any cluster (assign to "unknown" cluster)
- **Feature distributions**: Check for highly skewed features
- **Correlation**: Check correlation between DTW features and target
- **Feature importance**: Use CatBoost feature importance to validate DTW features

### Expected Feature Importance:
- **High importance**: `trajectory_cluster_drug`, `cluster_target_rate_drug`, `dtw_distance_to_targets_drug`
- **Medium importance**: `trajectory_length_drug`, `archetype_match_score_drug`
- **Low importance**: `cluster_size_drug`, `trajectory_diversity_drug` (may still be useful)

---

## Future Enhancements

- **Parallel DTW**: Implement parallel processing for large datasets
- **Advanced Clustering**: Add more clustering algorithms
- **Interactive Visualizations**: Create interactive dashboards
- **Real-time Analysis**: Support streaming analysis for new data
- **Integration with BupaR**: Direct integration with process mining workflows

## 📚 References

- DTW Analysis Scripts: `6_dtw_analysis/dtw_cohort_analysis.py`, `6_dtw_analysis/dtw_trajectory_analysis.py`
- Final Model: `7_final_model/README_final_model.md`
- Feature Importance: `3_feature_importance/`
- FPGrowth Analysis: `../4_fpgrowth_analysis/`
- BupaR Analysis: `../5_bupaR_analysis/`

