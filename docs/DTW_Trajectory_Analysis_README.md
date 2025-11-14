# DTW Trajectory Analysis: Developing Patient Trajectories

## Overview

Dynamic Time Warping (DTW) analysis adds a crucial dimension to the analysis pipeline by **developing patient trajectories** - temporal sequences that capture how patients progress through their healthcare journey. Unlike FPGrowth (which finds frequent patterns) or BupaR (which analyzes process flows), DTW focuses on **identifying similar patient trajectories** even when sequences vary in timing or length.

## 🎯 Key Advantages of DTW for Trajectory Analysis

### 1. **Handles Variable-Length Sequences**
- **FPGrowth**: Requires fixed patterns, doesn't handle timing variations well
- **BupaR**: Focuses on process flows, less flexible with sequence variations
- **DTW**: Handles sequences of different lengths and timing variations

### 2. **Temporal Warping**
- Aligns sequences that are similar but occur at different speeds
- Example: Patient A takes Drug X → Drug Y → Drug Z over 30 days
- Patient B takes the same sequence over 60 days
- DTW recognizes these as similar trajectories

### 3. **Trajectory Clustering**
- Groups patients with similar healthcare journeys
- Creates trajectory archetypes (representative patterns)
- Enables personalized medicine approaches

### 4. **Multi-Modal Trajectories**
- Can combine drugs, ICD codes, and CPT codes
- Creates comprehensive patient journey models
- Captures complex healthcare pathways

## 🔄 Integration with Existing Pipeline

```
Cohort Creation → FPGrowth Filtering → DTW Trajectories → CatBoost → BupaR
                      ↓                      ↓
                 Frequent Patterns    Patient Clusters
                      ↓                      ↓
                 Feature Filtering    Trajectory Features
```

### How DTW Complements Other Methods

| Method | Purpose | DTW Enhancement |
|--------|---------|----------------|
| **FPGrowth** | Find frequent patterns | Use FPGrowth patterns to initialize trajectory analysis |
| **CatBoost** | Predict outcomes | Add trajectory cluster membership as features |
| **BupaR** | Analyze process flows | Use DTW clusters for cluster-specific process mining |

## 📊 Use Cases

### 1. **ED_NON_OPIOID: Drug Window Trajectories**

**Question**: What are the common drug trajectories leading to non-opioid ED visits?

**DTW Analysis**:
```bash
python 7_dtw_analysis/dtw_trajectory_analysis.py \
    --cohort ed_non_opioid \
    --age-band "65-74" \
    --event-year 2020 \
    --item-type drug \
    --n-clusters 5
```

**What it does**:
- Creates drug trajectories using `days_to_target_event` for temporal alignment
- Clusters patients with similar drug sequences in the 30-day window
- Identifies trajectory archetypes (common patterns)

**Output**:
- Trajectory clusters (groups of similar patients)
- Archetype trajectories (representative patterns)
- Trajectory patterns (common sequences per cluster)

### 2. **OPIOID_ED: ICD/CPT Trajectories**

**Question**: What diagnostic/procedure trajectories predict opioid ED events?

**DTW Analysis**:
```bash
# ICD code trajectories
python 7_dtw_analysis/dtw_trajectory_analysis.py \
    --cohort opioid_ed \
    --age-band "65-74" \
    --event-year 2020 \
    --item-type icd \
    --n-clusters 6

# CPT code trajectories
python 7_dtw_analysis/dtw_trajectory_analysis.py \
    --cohort opioid_ed \
    --age-band "65-74" \
    --event-year 2020 \
    --item-type cpt \
    --n-clusters 6
```

**What it does**:
- Creates ICD/CPT trajectories from historical events
- Clusters patients with similar diagnostic/procedure patterns
- Identifies high-risk trajectory patterns

## 🚀 Enhanced Workflow

### Complete Analysis Pipeline with DTW

```python
# Step 1: FPGrowth - Find frequent patterns
python 3_fpgrowth_analysis/run_fpgrowth_cohort_filtering.py \
    --cohort ED_NON_OPIOID --item-type drug

# Step 2: DTW - Develop patient trajectories
python 7_dtw_analysis/dtw_trajectory_analysis.py \
    --cohort ed_non_opioid --item-type drug --n-clusters 5

# Step 3: CatBoost - Predict with trajectory features
# Add trajectory cluster membership as feature
# Use archetype trajectories for feature engineering

# Step 4: BupaR - Cluster-specific process mining
# Analyze process flows within each DTW cluster
```

## 📈 Trajectory Development Process

### 1. **Temporal Alignment**

For **ED_NON_OPIOID**:
- Uses `days_to_target_event` to align trajectories
- All trajectories end at day 0 (target event)
- Sequences are aligned backwards: 30 → 29 → ... → 1 → 0

For **OPIOID_ED**:
- Uses `event_date` for temporal ordering
- Sequences progress forward in time
- Captures full historical patterns

### 2. **Sequence Encoding**

```python
# Example trajectory encoding
Patient A: [Drug_X, Drug_Y, Drug_Z]
  Temporal: [30, 15, 5]  # days before target

Patient B: [Drug_X, Drug_Y, Drug_Z]
  Temporal: [25, 12, 3]  # days before target

# DTW recognizes these as similar despite timing differences
```

### 3. **Trajectory Clustering**

- Groups patients with similar sequences
- Creates cluster-specific archetypes
- Enables trajectory-based risk assessment

### 4. **Archetype Extraction**

- Identifies representative trajectory per cluster
- Uses median-length trajectory as archetype
- Provides interpretable trajectory patterns

## 🔗 Integration Examples

### Example 1: Trajectory-Enhanced CatBoost

```python
# Load trajectory clusters
from helpers_1997_13.s3_utils import load_from_s3_json

traj_results = load_from_s3_json(
    "s3://pgxdatalake/dtw_trajectories/ed_non_opioid/65-74/2020/trajectory_results_drug.json"
)

# Add trajectory cluster as feature
patient_cluster_map = traj_results['trajectory_clusters']['patient_cluster_map']
cohort_df['trajectory_cluster'] = cohort_df['mi_person_key'].map(patient_cluster_map)

# Use in CatBoost
categorical_features = ['trajectory_cluster', 'drug_name', ...]
```

### Example 2: Cluster-Specific BupaR Analysis

```python
# Load trajectory clusters
traj_results = load_from_s3_json("s3://.../trajectory_results_drug.json")

# Get patients in cluster 0 (high-risk trajectory)
cluster_0_patients = [
    pid for pid, cid in traj_results['trajectory_clusters']['patient_cluster_map'].items()
    if cid == 0
]

# Filter cohort data to cluster 0
cluster_0_data = cohort_df[cohort_df['mi_person_key'].isin(cluster_0_patients)]

# Run BupaR on cluster-specific data
# Analyze process flows within this trajectory cluster
```

### Example 3: Trajectory-Based Risk Prediction

```python
# Compare new patient trajectory to archetypes
def predict_risk(new_patient_trajectory, archetypes):
    """Predict risk based on trajectory similarity to archetypes."""
    risks = {}
    
    for cluster_id, archetype in archetypes.items():
        # Calculate DTW distance to archetype
        distance = dtw.distance(new_patient_trajectory, archetype)
        
        # Map distance to risk (lower distance = higher similarity = higher risk)
        risk = 1.0 / (1.0 + distance)  # Normalized risk score
        risks[cluster_id] = risk
    
    return risks
```

## 📊 Output Structure

```
s3://pgxdatalake/dtw_trajectories/
├── ed_non_opioid/
│   ├── 65-74/
│   │   └── 2020/
│   │       ├── trajectory_results_drug.json      # Complete analysis results
│   │       └── patient_trajectories_drug.parquet # Patient trajectories
│   └── ...
└── opioid_ed/
    ├── 65-74/
    │   └── 2020/
    │       ├── trajectory_results_icd.json
    │       ├── trajectory_results_cpt.json
    │       └── patient_trajectories_*.parquet
    └── ...
```

## 🔍 Key Metrics

### Trajectory Clusters
- **Silhouette Score**: Cluster quality (range: -1 to 1, higher is better)
- **Cluster Sizes**: Number of patients per trajectory cluster
- **Archetype Trajectories**: Representative patterns per cluster

### Trajectory Patterns
- **Average Trajectory Length**: Mean sequence length per cluster
- **Most Common Items**: Frequent items in each cluster
- **Temporal Characteristics**: Average temporal positioning

## 🎯 Research Questions DTW Can Answer

### ED_NON_OPIOID Cohort

1. **What are the common drug trajectories leading to ED visits?**
   - DTW clusters identify trajectory archetypes
   - Archetypes show common sequences (Drug A → Drug B → ED)

2. **Do patients with similar trajectories have similar outcomes?**
   - Compare outcomes within trajectory clusters
   - Identify high-risk trajectory patterns

3. **Can we predict ED visits based on trajectory similarity?**
   - Compare new patients to known trajectory archetypes
   - Use trajectory cluster membership for prediction

### OPIOID_ED Cohort

1. **What diagnostic trajectories predict opioid ED events?**
   - ICD code trajectories show diagnostic pathways
   - Identify high-risk diagnostic sequences

2. **What procedure patterns are associated with opioid ED?**
   - CPT code trajectories show procedure sequences
   - Find procedure patterns leading to opioid ED

3. **Can trajectory clustering improve prediction?**
   - Add trajectory features to CatBoost
   - Improve model performance with trajectory information

## 🛠️ Usage Examples

### Basic Trajectory Analysis

```bash
# Drug trajectories for ED_NON_OPIOID
python 7_dtw_analysis/dtw_trajectory_analysis.py \
    --cohort ed_non_opioid \
    --age-band "65-74" \
    --event-year 2020 \
    --item-type drug \
    --n-clusters 5
```

### Batch Processing

```bash
# Process multiple cohorts and item types
for cohort in opioid_ed ed_non_opioid; do
    for item_type in drug icd cpt; do
        python 7_dtw_analysis/dtw_trajectory_analysis.py \
            --cohort $cohort \
            --age-band "65-74" \
            --event-year 2020 \
            --item-type $item_type \
            --n-clusters 5
    done
done
```

## 📚 References

- **DTW Algorithm**: [dtaidistance Documentation](https://dtaidistance.readthedocs.io/)
- **Trajectory Clustering**: Hierarchical clustering on DTW distances
- **Cohort Creation**: See `docs/Create_Cohort_README.md`
- **FPGrowth Integration**: See `3_fpgrowth_analysis/FPGrowth_Filtering_README.md`

---

*Last Updated: January 2025*
*Module Version: 2.0*
*Compatible with: pgx_analysis pipeline v2+*

