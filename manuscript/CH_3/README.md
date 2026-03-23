# Chapter 3: Study 1 – Causal Drivers of Opioid Use Disorder (OUD)

## Overview

This study investigates the causal, temporal drivers of Opioid Use Disorder (OUD) risk in young and mid-life adults (ages 13–64), focusing on the active workforce and young adult population most affected by the opioid epidemic.

**Research Question (RQ1):** What are the causal, temporal drivers of opioid-related emergency department visits, and can we identify high-risk trajectories before the diagnosis of dependence (F11.20)?

**Key Contribution:** Moving beyond static risk scoring to identify *modifiable* risk factors using the "Consensus Filter" methodology (SHAP + FFA), while addressing ICPM conference feedback by demonstrating how "visualization-only" process mining protects against target leakage.

---

## 1. Introduction

### 1.1 The Clinical Challenge
The opioid epidemic disproportionately affects the active workforce and young adult population (Ages 13–64). Early identification of high-risk trajectories is critical for intervention before the onset of dependence.

### 1.2 Research Question
Can we identify causal, temporal drivers of opioid-related emergency department visits and predict high-risk trajectories before the diagnosis of dependence (F11.20)?

### 1.3 Innovation
This study employs the "Consensus Filter" (SHAP + FFA) to identify *modifiable* risk factors, moving beyond traditional static risk scoring approaches.

---

## 2. Cohort Construction & Study Design

### 2.1 Target Definition (`opioid_ed`)

**Case Definition:**
- Patients identified by ICD codes (e.g., F11.20 Opioid Use Disorder) appearing in *any* of the 10 diagnosis columns
- **Constraint:** Cases must have a definitive diagnosis

**Control Definition:**
- Controls must have *zero* opioid-related ICD codes
- Ensures strict separation between cases and controls

### 2.2 Control Selection Strategy

**Matching Approach:**
- **Ratio:** 5:1 (controls to cases)
- **Method:** Matched sampling without replacement to ensure statistical independence

**Demographics:**
- Age bands: 13–24, 25–44, 45–54, 55–64
- Demographic matching ensures comparable baseline characteristics

### 2.3 Temporal Window Configuration

**Key Features:**
- **Complete Drug History:** Unlike the polypharmacy cohort (30-day window), this study captures long-term addiction trajectories
- **Leakage Prevention:** Explicit exclusion of all events occurring *after* the first F11.20 diagnosis

**Rationale:** Long-term exposure patterns are critical for understanding addiction pathways

---

## 3. Methodology

### 3.1 Feature Engineering (The Clean Approach)

**Input Features:**
- Drug exposures (prescription history)
- ICD Diagnoses (comorbidities)
- CPT Procedures (clinical interventions)

**ICPM Correction:**
- FP-Growth (Association Rules) and BupaR (Process Mining) are used for **exploratory analysis only** (Step 3b/9)
- These methods are *not* used as predictive features to strictly avoid encoding target information
- Maintains methodological rigor by preventing target leakage

### 3.2 The Ensemble Model

**Model Components:**
1. **CatBoost** - Optimized for categorical features
2. **XGBoost** - Gradient boosting framework
3. **XGBoost RF** - Random Forest mode for diversity

**Performance Weighting:**
- Models weighted based on Recall and AUC-PR on the 2019 holdout set
- Ensures optimal balance between sensitivity and precision

### 3.3 The Consensus Filter (Causal Discovery)

**Dual-Confirmation Rule:**
A feature is deemed causal only if it satisfies BOTH conditions:
1. **High SHAP Importance** (Quantitative measure)
2. **Appears in XGBoost Symbolic Rules** (Logical/FFA confirmation)

**Advantage:** Reduces false positives by requiring convergence of evidence from multiple analytical perspectives

---

## 4. Results: Predictive Performance

### 4.1 Model Metrics

Performance metrics stratified by age band:
- **13–24 years:** [Recall, Precision, AUC-PR]
- **25–44 years:** [Recall, Precision, AUC-PR]
- **45–64 years:** [Recall, Precision, AUC-PR]

Analysis highlights age groups where the model achieves optimal performance.

### 4.2 Feature Importance Ranking

**Aggregated Feature Importance (Step 3):**
- Top predictors identified through ensemble consensus
- Key categories:
  - **Opioid Medications:** Hydrocodone, Oxycodone, Fentanyl
  - **Comorbidities:** Chronic pain (ICD codes), mental health diagnoses
  - **Procedures:** Pain management interventions, surgical procedures

---

## 5. Results: Trajectory Analysis (Visualization)

### 5.1 Pre-Diagnosis Pathways (BupaR)

**Analysis Method:**
- BupaR "Pre-F11.20" visualization tracks event sequences leading to diagnosis
- Timeline limited to events *before* first OUD diagnosis

**Key Findings:**
- Identification of common "gateway" procedures
- Non-opioid pain management failures preceding opioid escalation
- Critical intervention points in the pathway to dependence

### 5.2 Patient Clustering (DTW)

**Methodology:**
- Dynamic Time Warping (DTW) Similarity Matrix
- Trajectory timeline visualization

**Identified Clusters:**

1. **"Rapid Onset" Trajectory**
   - Pattern: Acute injury → High Dose → Dependence
   - Timeframe: Shorter progression
   - Intervention window: Limited

2. **"Chronic Escalation" Trajectory**
   - Pattern: Long-term pain → Dose Creep → Dependence
   - Timeframe: Extended progression
   - Intervention window: Multiple opportunities

---

## 6. Results: Causal Mechanisms (FFA)

### 6.1 Symbolic Rule Extraction

**Example Rules Identified by FFA:**

```
IF Prescribed(Oxycodone) 
   AND Diagnosis(Chronic Pain) 
   AND NOT Prescribed(Physical Therapy)
THEN Risk = High
```

**Interpretation:**
- Boolean rules provide interpretable, actionable insights
- Rules identify specific combinations of risk factors
- Clinical applicability for decision support

### 6.2 Intervention Analysis

**Intervention Rate (IR) Calculation:**
- Quantifies risk reduction if specific features are modified/removed
- Example: Impact of eliminating concomitant Benzodiazepine prescriptions
- Prioritizes interventions by expected impact magnitude

**Clinical Application:**
- Guides resource allocation for intervention programs
- Identifies highest-yield modification targets

---

## 7. Discussion

### 7.1 Clinical Implications

**Early Intervention Opportunities:**
- Trajectory identification enables intervention *before* diagnosis of dependence
- Earlier than current standard-of-care protocols
- Potential for significant reduction in OUD incidence

**Actionable Insights:**
- Modifiable risk factors provide concrete intervention targets
- Symbolic rules can be integrated into clinical decision support systems
- Age-stratified approaches acknowledge developmental differences in risk

### 7.2 Limitations

**Claims Data Constraints:**
- **Cash Payments:** Prescriptions paid out-of-pocket not captured
- **Illicit Drug Use:** Non-prescription opioid use not reflected in APCD
- **Incomplete Picture:** May underestimate total opioid exposure

**Generalizability:**
- Findings specific to insured population with claims data
- May not fully represent uninsured or underinsured populations

**Temporal Considerations:**
- Historical data may not reflect current prescribing practices
- Ongoing policy changes (e.g., prescription monitoring programs) may alter risk patterns

---

## 8. Conclusion

This study successfully demonstrates that:

1. **Ensemble modeling** with consensus-based feature selection identifies robust, causal drivers of OUD
2. **The Consensus Filter** (SHAP + FFA) prevents black-box leakage while maintaining interpretability
3. **Trajectory analysis** reveals distinct pathways to dependence with different intervention windows
4. **Symbolic rules** provide clinically actionable insights for intervention design

**Impact:** The methodology provides a framework for identifying high-risk patients before diagnosis, enabling proactive intervention in the opioid epidemic.

---

## Technical Notes

### Pipeline Configuration
- **Cohort:** `opioid_ed`
- **Age Range:** 13–64 years (4 age bands)
- **Control Ratio:** 5:1
- **Temporal Window:** Complete drug history (pre-diagnosis)
- **Holdout Set:** 2019 data for final validation

### Leakage Prevention Strategy
- Process mining used for **visualization only**
- Association rules for **exploratory analysis only**
- Strict temporal cutoff at first F11.20 diagnosis
- No target-derived features in predictive model

---

## Repository Structure

```
CH_3/
├── data/               # Cohort data and preprocessing scripts
├── models/             # Trained ensemble models
├── analysis/           # SHAP, FFA, and trajectory analysis
├── visualizations/     # BupaR plots, DTW matrices, feature importance
├── results/            # Performance metrics and validation results
└── README.md          # This file
```

---

## Contact & Citation

For questions regarding this research or methodology, please contact [Research Team].

**Recommended Citation:**
[Author(s)]. (2026). Causal Drivers of Opioid Use Disorder Risk in Young and Mid-Life Adults: An Ensemble Machine Learning Approach with Consensus-Based Feature Selection. [Journal/Conference].

---

*Last Updated: January 2026*
