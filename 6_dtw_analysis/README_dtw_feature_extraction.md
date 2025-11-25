# DTW Feature Extraction for Final Model

## Overview

This document outlines the features to extract from DTW analysis outputs for use in the final CatBoost/Random Forest model. These features complement FPGrowth itemsets and BupaR process patterns.

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

## 📚 References

- DTW Analysis: `6_dtw_analysis/dtw_trajectory_analysis.py`
- DTW Documentation: `docs/DTW_Trajectory_Analysis_README.md`
- Final Model: `7_final_model/`
- Feature Importance: `3_feature_importance/`

