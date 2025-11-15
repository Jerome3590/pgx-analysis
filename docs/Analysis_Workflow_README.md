# Analysis Workflow: FPGrowth → CatBoost → BupaR

## Overview

This document describes the complete analysis workflow for answering key research questions using the cohort data:

1. **ED_NON_OPIOID Cohort**: Does drug window influence target outcome and which drugs are involved? Is there a temporal/ordering aspect?
2. **OPIOID_ED Cohort**: What CPT/ICD Codes and Drugs can be used to predict OPIOID_ED events?

## 📊 Research Questions → Analysis Methods Mapping

| Research Question Component | Analysis Method | Purpose | Output |
|----------------------------|-----------------|---------|--------|
| **ED_NON_OPIOID Cohort** | | | |
| Which drugs are involved? | **FPGrowth Filtering** | Discover frequent drug patterns in 30-day window | Frequent drug itemsets, association rules |
| Temporal/ordering aspect? | **BupaR Pattern Mining** | Analyze drug sequence patterns and process flows | Process flow diagrams, sequence analysis |
| Does drug window influence outcome? | **CatBoost Prediction** | Measure predictive power of drug features | Feature importance rankings, model predictions |
| Patient trajectory patterns? | **DTW Trajectories** | Cluster patients with similar drug sequences | Trajectory clusters, archetype trajectories |
| Formal causality assessment? | **Updated CatBoost** | Feature attribution and causal inference | SHAP/LIME scores, causal effect estimates |
| **OPIOID_ED Cohort** | | | |
| Which ICD codes predict? | **FPGrowth Filtering** | Discover frequent ICD code patterns | Frequent ICD itemsets, association rules |
| Which CPT codes predict? | **FPGrowth Filtering** | Discover frequent CPT code patterns | Frequent CPT itemsets, association rules |
| Which drugs predict? | **FPGrowth Filtering** | Discover frequent drug patterns | Frequent drug itemsets, association rules |
| Predictive patterns? | **BupaR Pattern Mining** | Analyze ICD/CPT/drug sequence patterns | Process flow diagrams, sequence patterns |
| Feature importance? | **CatBoost Prediction** | Rank ICD/CPT/drug features by importance | Feature importance rankings, predictions |
| Predictive trajectories? | **DTW Trajectories** | Cluster patients with similar trajectories | ICD/CPT/drug trajectory clusters |
| Formal causality assessment? | **Updated CatBoost** | Feature attribution and causal inference | SHAP/LIME scores, causal effect estimates |

## 🔄 Complete Analysis Pipeline

```
Cohort Creation → FPGrowth Filtering → BupaR Pattern Mining → CatBoost Prediction → DTW Trajectories → Updated CatBoost (Attribution & Causality)
                      ↓                      ↓                      ↓                      ↓                              ↓
                 Frequent Patterns    Process Flows      Feature Importance    Patient Clusters          Formal Attribution
```

### Step 1: Cohort Creation
- **Output**: Clean cohort parquet files with temporal fields
- **Location**: `s3://pgxdatalake/cohorts_clean/`
- **Key Fields**:
  - ED_NON_OPIOID: `days_to_target_event`, `drug_name`, `first_ed_non_opioid_date`
  - OPIOID_ED: `primary_icd_diagnosis_code`, `procedure_code`, `drug_name`, `first_opioid_ed_date`

### Step 2: FPGrowth Filtering
- **Purpose**: Discover frequent patterns and filter features before modeling
- **Script**: `3_fpgrowth_analysis/run_fpgrowth_cohort_filtering.py`
- **Output**: Frequent itemsets, association rules, encoding maps
- **Use Case**: Initial feature filtering for downstream analysis

### Step 3: BupaR Pattern Mining
- **Purpose**: Analyze temporal sequences and ordering patterns
- **Notebook**: `4_bupaR_analysis/bupaR_pipeline.ipynb`
- **Output**: Process flow diagrams, sequence analysis
- **Use Case**: Identify temporal/ordering aspects and process flows

### Step 4: CatBoost Feature Importance & Prediction
- **Purpose**: Identify which features (drugs, ICD codes, CPT codes) predict outcomes
- **Scripts**: `5_catboost_analysis/run_catboost_*.py`
- **Output**: Feature importance rankings, model predictions
- **Use Case**: Initial prediction models and feature ranking

### Step 5: DTW Patient Trajectories
- **Purpose**: Develop patient trajectories and identify similar healthcare journeys
- **Script**: `7_dtw_analysis/dtw_trajectory_analysis.py`
- **Output**: Trajectory clusters, archetype trajectories, patient trajectory mappings
- **Key Advantage**: Handles variable-length sequences and temporal warping
- **Use Case**: Patient clustering and trajectory-based features

### Step 6: Updated CatBoost (Formal Feature Attribution & Causality)
- **Purpose**: Formal feature attribution and causal inference using all previous analysis
- **Scripts**: Enhanced `5_catboost_analysis/run_catboost_*.py` with SHAP/LIME
- **Input**: FPGrowth patterns + BupaR sequences + DTW clusters + original features
- **Output**: Feature attribution scores, causal effect estimates, enhanced models
- **Use Case**: Formal causality assessment and interpretable predictions

---

## 📊 ED_NON_OPIOID Cohort Analysis

### Research Questions
1. **Does drug window influence target outcome?**
2. **Which drugs are involved?**
3. **Is there a temporal/ordering aspect?**

### Analysis Steps

#### Step 1: FPGrowth Drug Pattern Mining

```bash
# Run FPGrowth for drugs in 30-day window
python 3_fpgrowth_analysis/run_fpgrowth_cohort_filtering.py \
    --cohort ED_NON_OPIOID \
    --item-type drug \
    --age-band "65-74" \
    --event-year 2020
```

**What it does:**
- Extracts drugs within 30-day window before target event
- Finds frequent drug combinations
- Discovers association rules (Drug A → Drug B → ED visit)

**Output:**
- `itemsets_drug.json`: Frequent drug combinations
- `rules_drug.json`: Association rules with confidence/lift
- `encoding_drug.parquet`: Drug encoding map with support metrics

#### Step 2: DTW Trajectory Analysis

```bash
# Run DTW to develop patient trajectories
python 7_dtw_analysis/dtw_trajectory_analysis.py \
    --cohort ed_non_opioid \
    --age-band "65-74" \
    --event-year 2020 \
    --item-type drug \
    --n-clusters 5
```

**What it does:**
- Creates drug trajectories using `days_to_target_event` for temporal alignment
- Clusters patients with similar drug sequences
- Identifies trajectory archetypes (representative patterns)
- Groups patients with similar healthcare journeys

**Output:**
- `trajectory_results_drug.json`: Trajectory clusters and archetypes
- `patient_trajectories_drug.parquet`: Patient trajectory mappings

**Key Advantage:**
- Handles variable-length sequences (unlike FPGrowth)
- Recognizes similar trajectories despite timing differences
- Creates trajectory-based patient clusters

#### Step 3: CatBoost Feature Importance

```bash
# Run CatBoost with FPGrowth-filtered features
python 5_catboost_analysis/run_catboost_ade_ed.py \
    --cohort ED_NON_OPIOID \
    --use-fpgrowth-filter
```

**What it does:**
- Filters features to top frequent drugs from FPGrowth
- Trains CatBoost model to predict ED_NON_OPIOID outcome
- Ranks drugs by feature importance

**Output:**
- Feature importance rankings
- Model predictions
- Drug impact analysis

#### Step 4: BupaR Temporal Analysis

```python
# Load FPGrowth rules for process mining
import json
from helpers_1997_13.s3_utils import load_from_s3_json

rules = load_from_s3_json("s3://.../rules_drug.json")

# Filter high-confidence rules
high_conf_rules = [
    r for r in rules 
    if r.get('confidence', 0) > 0.7
]

# Use in BupaR notebook
# 4_bupaR_analysis/bupaR_pipeline.ipynb
```

**What it does:**
- Creates event log with drug sequences
- Analyzes temporal ordering patterns
- Visualizes process flows (Drug A → Drug B → ED visit)

**Output:**
- Process flow diagrams
- Sequence frequency analysis
- Temporal pattern identification

---

## 📊 OPIOID_ED Cohort Analysis

### Research Question
**What CPT/ICD Codes and Drugs can be used to predict OPIOID_ED events?**

### Analysis Steps

#### Step 1: FPGrowth ICD Code Pattern Mining

```bash
# Run FPGrowth for ICD codes
python 3_fpgrowth_analysis/run_fpgrowth_cohort_filtering.py \
    --cohort OPIOID_ED \
    --item-type icd \
    --age-band "65-74" \
    --event-year 2020
```

**What it does:**
- Extracts all ICD diagnosis codes
- Finds frequent ICD code combinations
- Discovers patterns predictive of opioid ED events

**Output:**
- `itemsets_icd.json`: Frequent ICD combinations
- `rules_icd.json`: ICD association rules
- `encoding_icd.parquet`: ICD encoding map

#### Step 2: FPGrowth CPT Code Pattern Mining

```bash
# Run FPGrowth for CPT codes
python 3_fpgrowth_analysis/run_fpgrowth_cohort_filtering.py \
    --cohort OPIOID_ED \
    --item-type cpt \
    --age-band "65-74" \
    --event-year 2020
```

**What it does:**
- Extracts all CPT procedure codes
- Finds frequent CPT code combinations
- Discovers procedure patterns associated with opioid ED

**Output:**
- `itemsets_cpt.json`: Frequent CPT combinations
- `rules_cpt.json`: CPT association rules
- `encoding_cpt.parquet`: CPT encoding map

#### Step 3: DTW Trajectory Analysis

```bash
# Run DTW for ICD code trajectories
python 7_dtw_analysis/dtw_trajectory_analysis.py \
    --cohort opioid_ed \
    --age-band "65-74" \
    --event-year 2020 \
    --item-type icd \
    --n-clusters 6

# Run DTW for CPT code trajectories
python 7_dtw_analysis/dtw_trajectory_analysis.py \
    --cohort opioid_ed \
    --age-band "65-74" \
    --event-year 2020 \
    --item-type cpt \
    --n-clusters 6
```

**What it does:**
- Creates ICD/CPT trajectories from historical events
- Clusters patients with similar diagnostic/procedure patterns
- Identifies high-risk trajectory patterns
- Enables trajectory-based risk prediction

**Output:**
- Trajectory clusters for ICD and CPT codes
- Archetype trajectories showing common patterns
- Patient trajectory mappings

#### Step 4: CatBoost Predictive Modeling

```bash
# Run CatBoost with ICD/CPT features
python 5_catboost_analysis/run_catboost_opioid_ed.py \
    --cohort OPIOID_ED \
    --use-fpgrowth-filter \
    --features icd cpt drug
```

**What it does:**
- Combines ICD, CPT, and drug features
- Filters to top frequent patterns from FPGrowth
- Trains model to predict opioid ED events
- Ranks features by importance

**Output:**
- Feature importance rankings (ICD, CPT, drugs)
- Model predictions
- Predictive feature identification

---

## 🔗 Integration Examples

### Example 1: Complete ED_NON_OPIOID Analysis

```python
# 1. Run FPGrowth filtering
import subprocess
subprocess.run([
    "python", "3_fpgrowth_analysis/run_fpgrowth_cohort_filtering.py",
    "--cohort", "ED_NON_OPIOID",
    "--item-type", "drug",
    "--age-band", "65-74",
    "--event-year", "2020"
])

# 2. Load FPGrowth results
from helpers_1997_13.s3_utils import load_from_s3_json
itemsets = load_from_s3_json("s3://.../itemsets_drug.json")
rules = load_from_s3_json("s3://.../rules_drug.json")

# 3. Filter to top drugs
top_drugs = set()
for itemset in itemsets[:50]:
    for item in itemset.get('itemsets', []):
        top_drugs.add(item.replace('drug_', ''))

# 4. Use in CatBoost
# Filter cohort data to top_drugs
# Train CatBoost model
# Analyze feature importance

# 5. Use in BupaR
# Create event log with drug sequences
# Analyze temporal patterns
```

### Example 2: Complete OPIOID_ED Analysis

```python
# 1. Run FPGrowth for ICD and CPT
for item_type in ['icd', 'cpt']:
    subprocess.run([
        "python", "3_fpgrowth_analysis/run_fpgrowth_cohort_filtering.py",
        "--cohort", "OPIOID_ED",
        "--item-type", item_type,
        "--age-band", "65-74",
        "--event-year", "2020"
    ])

# 2. Load results
icd_itemsets = load_from_s3_json("s3://.../itemsets_icd.json")
cpt_itemsets = load_from_s3_json("s3://.../itemsets_cpt.json")

# 3. Extract top features
top_icd = extract_top_items(icd_itemsets, top_k=100)
top_cpt = extract_top_items(cpt_itemsets, top_k=100)

# 4. Use in CatBoost
# Combine ICD + CPT features
# Train model
# Rank by importance
```

---

## 📈 Expected Results

### ED_NON_OPIOID Cohort

**FPGrowth Results:**
- Frequent drug combinations in 30-day window
- Association rules showing drug sequences
- Support/confidence metrics for each drug

**CatBoost Results:**
- Feature importance rankings for drugs
- Model performance metrics
- Drug impact on ED visit prediction

**BupaR Results:**
- Process flow diagrams showing drug sequences
- Temporal ordering patterns
- Sequence frequency analysis

### OPIOID_ED Cohort

**FPGrowth Results:**
- Frequent ICD code combinations
- Frequent CPT code combinations
- Association rules for code patterns

**CatBoost Results:**
- Feature importance for ICD codes
- Feature importance for CPT codes
- Combined predictive model
- Top predictive features

---

## 🛠️ Quick Reference

### FPGrowth Filtering
```bash
# ED_NON_OPIOID: Drugs only
python 3_fpgrowth_analysis/run_fpgrowth_cohort_filtering.py --cohort ED_NON_OPIOID --item-type drug

# OPIOID_ED: ICD and CPT
python 3_fpgrowth_analysis/run_fpgrowth_cohort_filtering.py --cohort OPIOID_ED --item-type icd
python 3_fpgrowth_analysis/run_fpgrowth_cohort_filtering.py --cohort OPIOID_ED --item-type cpt
```

### CatBoost Analysis
```bash
# ED_NON_OPIOID
python 5_catboost_analysis/run_catboost_ade_ed.py --cohort ED_NON_OPIOID

# OPIOID_ED
python 5_catboost_analysis/run_catboost_opioid_ed.py --cohort OPIOID_ED
```

### BupaR Analysis
```bash
# Open notebook
jupyter notebook 4_bupaR_analysis/bupaR_pipeline.ipynb
```

---

## 📚 Documentation References

- **FPGrowth Filtering**: `3_fpgrowth_analysis/FPGrowth_Filtering_README.md`
- **CatBoost Analysis**: `3_fpgrowth_analysis/CatBoost_README.md`
- **BupaR Analysis**: `3_fpgrowth_analysis/BupaR_README.md`
- **Cohort Creation**: `docs/Create_Cohort_README.md`

---

*Last Updated: November 15, 2025*
*Pipeline Version: 3.0*

