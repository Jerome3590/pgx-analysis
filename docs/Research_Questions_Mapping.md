# Research Questions Mapping: Cohort Configuration → Analysis Workflow

## Overview

This document maps the research questions to the analysis workflow, verifying that our cohort configuration provides all necessary data to answer each question using the complete analysis pipeline.

## 🔄 Updated Analysis Workflow

```
1. FPGrowth → Filtering (frequent patterns)
2. BupaR → Pattern Mining (process flows)
3. CatBoost → Feature Importance & Prediction (initial models)
4. DTW → Patient Trajectories (similarity clustering)
5. Updated CatBoost → Formal Feature Attribution & Causality (enhanced models)
```

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

---

## 📊 Research Question 1: ED_NON_OPIOID Cohort

### Question
**Does drug window influence target outcome and which drugs are involved? Is there a temporal/ordering aspect?**

### Cohort Configuration ✅

**Available Fields:**
- ✅ `drug_name` - Drug names for all pharmacy events
- ✅ `days_to_target_event` - Temporal positioning (0-30 days before target)
- ✅ `first_ed_non_opioid_date` - Reference date for temporal alignment
- ✅ `event_date` - Absolute dates for sequence analysis
- ✅ `is_target_case` - Target outcome (1=case, 0=control)
- ✅ `target` - Binary outcome variable
- ✅ `therapeutic_class_1/2/3` - Drug classification
- ✅ `event_type` - Distinguishes medical vs pharmacy events

**Temporal Window:**
- ✅ 30-day lookback window applied to both targets and controls
- ✅ Balanced temporal windows (same logic for targets and controls)
- ✅ `days_to_target_event` calculated for all events

### Analysis Workflow Mapping

#### Step 1: FPGrowth Filtering ✅
**Question Component**: "Which drugs are involved?"

**What it does:**
- Discovers frequent drug patterns within 30-day window
- Filters to top frequent drugs before modeling
- Identifies drug combinations associated with outcomes

**Cohort Data Used:**
- `drug_name` (filtered by 30-day window)
- `days_to_target_event` (for temporal filtering)
- `is_target_case` (for outcome association)

**Output:**
- Frequent drug itemsets
- Drug association rules
- Top drugs for feature filtering

**Answer**: ✅ **YES** - Identifies which drugs are involved

---

#### Step 2: BupaR Pattern Mining ✅
**Question Component**: "Is there a temporal/ordering aspect?"

**What it does:**
- Analyzes drug sequence patterns
- Identifies temporal ordering (Drug A → Drug B → ED visit)
- Creates process flow diagrams showing drug sequences

**Cohort Data Used:**
- `drug_name` (sequence of drugs)
- `days_to_target_event` (temporal ordering)
- `event_date` (absolute timing)
- `is_target_case` (outcome)

**Output:**
- Process flow diagrams
- Sequence frequency analysis
- Temporal pattern identification

**Answer**: ✅ **YES** - Identifies temporal/ordering aspects

---

#### Step 3: CatBoost Feature Importance & Prediction ✅
**Question Component**: "Does drug window influence target outcome?"

**What it does:**
- Trains model to predict ED_NON_OPIOID outcome
- Ranks drugs by feature importance
- Measures predictive power of drug features

**Cohort Data Used:**
- `drug_name` (features)
- `days_to_target_event` (temporal features)
- `target` / `is_target_case` (outcome)
- FPGrowth-filtered drug set

**Output:**
- Feature importance rankings
- Model predictions
- Drug impact on outcome

**Answer**: ✅ **YES** - Measures if drug window influences outcome

---

#### Step 4: DTW Patient Trajectories ✅
**Question Component**: "Is there a temporal/ordering aspect?" (trajectory view)

**What it does:**
- Creates patient drug trajectories using `days_to_target_event`
- Clusters patients with similar trajectories
- Identifies trajectory archetypes (common patterns)

**Cohort Data Used:**
- `drug_name` (trajectory items)
- `days_to_target_event` (temporal alignment)
- `first_ed_non_opioid_date` (reference point)
- `is_target_case` (outcome)

**Output:**
- Trajectory clusters
- Archetype trajectories
- Similar patient groups

**Answer**: ✅ **YES** - Identifies temporal trajectories and ordering

---

#### Step 5: Updated CatBoost (Feature Attribution & Causality) ✅
**Question Component**: "Does drug window influence target outcome?" (causal inference)

**What it does:**
- Uses FPGrowth patterns, BupaR sequences, DTW clusters as features
- Performs formal feature attribution (SHAP, LIME)
- Assesses causal relationships (drug window → outcome)

**Cohort Data Used:**
- All previous analysis outputs
- `drug_name` + `days_to_target_event` (temporal drug features)
- `target` / `is_target_case` (outcome)
- Trajectory cluster memberships
- Process flow patterns

**Output:**
- Feature attribution scores
- Causal effect estimates
- Enhanced predictive models

**Answer**: ✅ **YES** - Provides formal attribution and causality assessment

---

### ✅ Complete Answer to Question 1

| Component | Analysis Step | Cohort Data | Answer |
|-----------|---------------|-------------|--------|
| **Which drugs involved?** | FPGrowth Filtering | `drug_name`, `days_to_target_event` | ✅ YES |
| **Temporal/ordering?** | BupaR Pattern Mining | `drug_name`, `days_to_target_event`, `event_date` | ✅ YES |
| **Drug window influence?** | CatBoost Prediction | `drug_name`, `days_to_target_event`, `target` | ✅ YES |
| **Trajectory patterns?** | DTW Trajectories | `drug_name`, `days_to_target_event`, `first_ed_non_opioid_date` | ✅ YES |
| **Causal attribution?** | Updated CatBoost | All above + enhanced features | ✅ YES |

---

## 📊 Research Question 2: OPIOID_ED Cohort

### Question
**What CPT/ICD Codes and Drugs can be used to predict OPIOID_ED events?**

### Cohort Configuration ✅

**Available Fields:**
- ✅ `primary_icd_diagnosis_code` - ICD diagnosis codes
- ✅ `procedure_code` - CPT procedure codes
- ✅ `drug_name` - Drug names for pharmacy events
- ✅ `first_opioid_ed_date` - Reference date for temporal analysis
- ✅ `event_date` - Absolute dates for sequence analysis
- ✅ `is_target_case` - Target outcome (1=case, 0=control)
- ✅ `target` - Binary outcome variable
- ✅ `primary_icd_ccs_level_1/2/3` - ICD classification
- ✅ `therapeutic_class_1/2/3` - Drug classification
- ✅ `event_type` - Distinguishes medical vs pharmacy events

**Temporal Configuration:**
- ✅ All historical ICD/CPT codes included (no filtering)
- ✅ All historical drugs included (no temporal filtering)
- ✅ Full patient history available for pattern analysis

### Analysis Workflow Mapping

#### Step 1: FPGrowth Filtering ✅
**Question Component**: "What CPT/ICD Codes and Drugs?"

**What it does:**
- Discovers frequent ICD code patterns
- Discovers frequent CPT code patterns
- Discovers frequent drug patterns
- Filters to top predictive patterns

**Cohort Data Used:**
- `primary_icd_diagnosis_code` (ICD patterns)
- `procedure_code` (CPT patterns)
- `drug_name` (drug patterns)
- `is_target_case` (outcome association)

**Output:**
- Frequent ICD itemsets
- Frequent CPT itemsets
- Frequent drug itemsets
- Association rules for each type

**Answer**: ✅ **YES** - Identifies which ICD/CPT codes and drugs are involved

---

#### Step 2: BupaR Pattern Mining ✅
**Question Component**: "What patterns predict OPIOID_ED?"

**What it does:**
- Analyzes ICD code sequences leading to opioid ED
- Analyzes CPT code sequences leading to opioid ED
- Analyzes drug sequences leading to opioid ED
- Identifies process flows (ICD → CPT → Drug → OPIOID_ED)

**Cohort Data Used:**
- `primary_icd_diagnosis_code` (ICD sequences)
- `procedure_code` (CPT sequences)
- `drug_name` (drug sequences)
- `event_date` (temporal ordering)
- `is_target_case` (outcome)

**Output:**
- Process flow diagrams for ICD/CPT/Drug
- Sequence patterns leading to opioid ED
- Multi-modal process flows

**Answer**: ✅ **YES** - Identifies predictive patterns

---

#### Step 3: CatBoost Feature Importance & Prediction ✅
**Question Component**: "What can be used to predict OPIOID_ED?"

**What it does:**
- Trains model with ICD, CPT, and drug features
- Ranks features by importance
- Identifies top predictive features

**Cohort Data Used:**
- `primary_icd_diagnosis_code` (ICD features)
- `procedure_code` (CPT features)
- `drug_name` (drug features)
- `target` / `is_target_case` (outcome)
- FPGrowth-filtered feature sets

**Output:**
- Feature importance rankings (ICD, CPT, drugs)
- Model predictions
- Top predictive features

**Answer**: ✅ **YES** - Identifies predictive ICD/CPT codes and drugs

---

#### Step 4: DTW Patient Trajectories ✅
**Question Component**: "What trajectories predict OPIOID_ED?"

**What it does:**
- Creates ICD code trajectories
- Creates CPT code trajectories
- Creates drug trajectories
- Clusters patients with similar trajectories
- Identifies high-risk trajectory patterns

**Cohort Data Used:**
- `primary_icd_diagnosis_code` (ICD trajectories)
- `procedure_code` (CPT trajectories)
- `drug_name` (drug trajectories)
- `event_date` (temporal ordering)
- `first_opioid_ed_date` (reference point)
- `is_target_case` (outcome)

**Output:**
- ICD trajectory clusters
- CPT trajectory clusters
- Drug trajectory clusters
- High-risk trajectory archetypes

**Answer**: ✅ **YES** - Identifies predictive trajectories

---

#### Step 5: Updated CatBoost (Feature Attribution & Causality) ✅
**Question Component**: "What can be used to predict OPIOID_ED?" (causal inference)

**What it does:**
- Uses FPGrowth patterns, BupaR sequences, DTW clusters as features
- Performs formal feature attribution
- Assesses causal relationships (ICD/CPT/Drug → OPIOID_ED)

**Cohort Data Used:**
- All previous analysis outputs
- `primary_icd_diagnosis_code` (ICD features)
- `procedure_code` (CPT features)
- `drug_name` (drug features)
- `target` / `is_target_case` (outcome)
- Trajectory cluster memberships
- Process flow patterns

**Output:**
- Feature attribution for ICD/CPT/drugs
- Causal effect estimates
- Enhanced predictive models

**Answer**: ✅ **YES** - Provides formal attribution and causality

---

### ✅ Complete Answer to Question 2

| Component | Analysis Step | Cohort Data | Answer |
|-----------|---------------|-------------|--------|
| **Which ICD codes?** | FPGrowth Filtering | `primary_icd_diagnosis_code` | ✅ YES |
| **Which CPT codes?** | FPGrowth Filtering | `procedure_code` | ✅ YES |
| **Which drugs?** | FPGrowth Filtering | `drug_name` | ✅ YES |
| **Predictive patterns?** | BupaR Pattern Mining | ICD/CPT/Drug sequences | ✅ YES |
| **Feature importance?** | CatBoost Prediction | ICD/CPT/Drug features | ✅ YES |
| **Trajectory patterns?** | DTW Trajectories | ICD/CPT/Drug trajectories | ✅ YES |
| **Causal attribution?** | Updated CatBoost | All above + enhanced features | ✅ YES |

---

## ✅ Summary: Cohort Configuration Completeness

### ED_NON_OPIOID Cohort ✅

| Required Data | Available in Cohort | Used By |
|---------------|---------------------|---------|
| Drug names | ✅ `drug_name` | FPGrowth, BupaR, CatBoost, DTW |
| Temporal positioning | ✅ `days_to_target_event` | BupaR, CatBoost, DTW |
| Reference date | ✅ `first_ed_non_opioid_date` | DTW |
| Outcome variable | ✅ `target`, `is_target_case` | CatBoost |
| Event dates | ✅ `event_date` | BupaR, DTW |
| Drug classification | ✅ `therapeutic_class_1/2/3` | Feature engineering |

**Status**: ✅ **COMPLETE** - All required data available

---

### OPIOID_ED Cohort ✅

| Required Data | Available in Cohort | Used By |
|---------------|---------------------|---------|
| ICD codes | ✅ `primary_icd_diagnosis_code` | FPGrowth, BupaR, CatBoost, DTW |
| CPT codes | ✅ `procedure_code` | FPGrowth, BupaR, CatBoost, DTW |
| Drug names | ✅ `drug_name` | FPGrowth, BupaR, CatBoost, DTW |
| Reference date | ✅ `first_opioid_ed_date` | DTW |
| Outcome variable | ✅ `target`, `is_target_case` | CatBoost |
| Event dates | ✅ `event_date` | BupaR, DTW |
| ICD classification | ✅ `primary_icd_ccs_level_1/2/3` | Feature engineering |
| Drug classification | ✅ `therapeutic_class_1/2/3` | Feature engineering |

**Status**: ✅ **COMPLETE** - All required data available

---

## 🎯 Workflow Completeness Check

### Step 1: FPGrowth Filtering ✅
- **ED_NON_OPIOID**: ✅ Drugs with temporal window
- **OPIOID_ED**: ✅ ICD codes, CPT codes, drugs
- **Cohort Support**: ✅ All required fields available

### Step 2: BupaR Pattern Mining ✅
- **ED_NON_OPIOID**: ✅ Drug sequences with temporal ordering
- **OPIOID_ED**: ✅ ICD/CPT/drug sequences
- **Cohort Support**: ✅ `event_date`, `days_to_target_event` available

### Step 3: CatBoost Feature Importance ✅
- **ED_NON_OPIOID**: ✅ Drug features + temporal features
- **OPIOID_ED**: ✅ ICD/CPT/drug features
- **Cohort Support**: ✅ All features + `target` outcome available

### Step 4: DTW Trajectories ✅
- **ED_NON_OPIOID**: ✅ Drug trajectories with temporal alignment
- **OPIOID_ED**: ✅ ICD/CPT/drug trajectories
- **Cohort Support**: ✅ Temporal fields + reference dates available

### Step 5: Updated CatBoost (Attribution & Causality) ✅
- **ED_NON_OPIOID**: ✅ Enhanced features from all previous steps
- **OPIOID_ED**: ✅ Enhanced features from all previous steps
- **Cohort Support**: ✅ All data + analysis outputs available

---

## ✅ Final Answer

### Question 1: ED_NON_OPIOID ✅
**"Does drug window influence target outcome and which drugs are involved? Is there a temporal/ordering aspect?"**

**Answer**: ✅ **YES** - Cohort configuration fully supports the complete workflow:
- ✅ FPGrowth identifies which drugs are involved
- ✅ BupaR identifies temporal/ordering aspects
- ✅ CatBoost measures drug window influence
- ✅ DTW develops patient trajectories
- ✅ Updated CatBoost provides formal attribution

### Question 2: OPIOID_ED ✅
**"What CPT/ICD Codes and Drugs can be used to predict OPIOID_ED events?"**

**Answer**: ✅ **YES** - Cohort configuration fully supports the complete workflow:
- ✅ FPGrowth identifies ICD/CPT codes and drugs
- ✅ BupaR identifies predictive patterns
- ✅ CatBoost ranks features by importance
- ✅ DTW identifies predictive trajectories
- ✅ Updated CatBoost provides formal attribution

---

## 📋 Recommendations

### 1. Ensure Analysis Outputs Are Saved ✅
- FPGrowth results: `s3://pgxdatalake/fpgrowth_features/`
- BupaR results: Process flow diagrams and sequences
- CatBoost models: Feature importance and predictions
- DTW results: `s3://pgxdatalake/dtw_trajectories/`

### 2. Feature Engineering Pipeline ✅
- Combine FPGrowth patterns → CatBoost features
- Combine BupaR sequences → CatBoost features
- Combine DTW clusters → CatBoost features
- Create multi-modal feature sets

### 3. Causal Inference Setup ✅
- Use SHAP/LIME for feature attribution
- Implement causal inference methods (propensity scoring, etc.)
- Validate causal relationships with domain experts

---

**Conclusion**: ✅ **The current cohort configuration fully supports answering both research questions using the complete 5-step analysis workflow.**

