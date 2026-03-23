# Chapter 4: Study 2 – Polypharmacy and Drug Interactions in Older Adults

## Project Overview

This chapter addresses **Research Question 2** and focuses on the geriatric population (Ages 65–94). It distinguishes itself from Chapter 3 by focusing on **short-term causality (30-day windows)** and **multi-feature interactions** (Drug-Drug Interactions), leveraging the specific "Interaction Analysis" capabilities of the FFA module.

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Cohort Construction & Study Design](#2-cohort-construction--study-design)
3. [Methodology: Interaction-Oriented Modeling](#3-methodology-interaction-oriented-modeling)
4. [Results: Causal Drug Interactions](#4-results-causal-drug-interactions)
5. [Results: Network Analysis](#5-results-network-analysis)
6. [Results: High-Utilization vs High-Risk](#6-results-high-utilization-vs-high-risk)
7. [Discussion](#7-discussion)
8. [Conclusion](#8-conclusion)

---

## 1. Introduction

### 1.1 The Polypharmacy Challenge
- **Scope:** Polypharmacy (5+ concurrent medications) in the aging population
- **Link:** Connection to Adverse Drug Events (ADEs)
- **Context:** References [1.1.2, 452]

### 1.2 Research Question (RQ2)
> "How do multi-drug interactions and temporal sequencing causally influence adverse drug events in the elderly?"

**Reference:** [1.5]

### 1.3 Contribution
Introduce **Multi-Feature Interaction Analysis** via FFA to:
- Move beyond single-drug risk factors
- Identify *synergistic* Drug-Drug Interactions (DDIs)
- Quantify causal relationships between drug combinations and adverse events

---

## 2. Cohort Construction & Study Design

### 2.1 Target Definition (`non_opioid_ed`)

#### Target Identification
- **Primary Target:** Patients identified by Milliman Healthcare Cost Group (HCG) line codes
  - O11: Emergency Room
  - P51: ER Visits
  
#### Strict Exclusion Criteria
- **Opioid-Related ICD Codes:** Explicit exclusion of any patient with opioid-related codes
- **Rationale:** Ensure this study measures *non-addiction* related adverse events (pure polypharmacy risk)

### 2.2 The 30-Day Causality Window

#### Critical Distinction from Chapter 3
- **Chapter 3:** Full-history approach
- **Chapter 4:** **30-day lookback window** for both targets and controls

#### Rationale
- ADEs are typically proximal to the prescription event
- This window isolates the immediate causal drug exposure preceding the emergency visit
- Enables detection of short-term, acute drug interaction effects

### 2.3 Demographics
**Focus Age Bands:**
- 65–74 years
- 75–84 years
- 85–94 years

---

## 3. Methodology: Interaction-Oriented Modeling

### 3.1 Feature Engineering (Drug-Focused)

#### Primary Focus
- Heavy emphasis on **Drug Exposure Features** to isolate pharmacological risks
- Contrast with Chapter 3's broad medical codes approach

#### FP-Growth Usage Clarification
- **Purpose:** Visualization of co-occurrence networks (Step 9)
- **Not Used For:** Input features (prevents leakage of target-specific drug bundles)
- **Role:** Association Rules for *clinical context* and network analysis only

### 3.2 The Ensemble Model

**Performance-Weighted Ensemble:**
- CatBoost
- XGBoost
- XGBoost RF (Random Forest mode)

### 3.3 Multi-Feature Interaction Testing (FFA)

#### Algorithm Description
- Tests feature combinations (pairs and triplets)
- Detects:
  - **Synergy:** Combined Effect > Sum of Individual Effects
  - **Antagonism:** Combined Effect < Sum of Individual Effects

#### Computational Management
- Pre-filtering: Only test features with non-zero SHAP importance
- Reduces computational burden of exhaustive combination testing

---

## 4. Results: Causal Drug Interactions

### 4.1 Top Synergistic Pairs

**Data Source:** `interaction_analysis.parquet`

#### Key Findings
- Specific drug pairs where **Interaction Effect** is positive
- Risk is greater than the sum of individual drug risks
- Examples to highlight:
  - [To be populated with actual results]

### 4.2 The "Intervention Rate" (Causal Importance)

#### Causal Scoring (Scale: 0.0–1.0)
Quantifies the impact of removing specific high-risk medications:
- **Example Targets:**
  - Anticoagulants
  - Benzodiazepines
  
#### Scoring Methods
1. **Probability-Based:** Direct effect on outcome probability
2. **Explainer-Based:** SHAP-derived causal attribution

---

## 5. Results: Network Analysis (Visualization)

### 5.1 The "Hairball" of Polypharmacy

**Visualization Method:** FP-Growth Network Visualizations (Step 9)

#### Visual Evidence
- Dense connectivity of drug co-occurrences in 65+ population
- High-risk drugs (identified by FFA) act as **"Hub Nodes"**
- High betweenness centrality indicates critical positions in prescribing cascades

### 5.2 Association Rules as Clinical Context

**Format:** Antecedent → Consequent

#### Common Prescribing Cascades
- **Example:** Opioid → Laxative
- Correlate with higher risk profiles
- Provide clinical context for interaction findings

---

## 6. Results: High-Utilization vs High-Risk

### 6.1 Extreme Density & Z-Codes

**Data Source:** Z-Code Analysis

#### Key Finding
Patients with "Extreme Density" transactions often have:
- Significantly larger time windows
- High proportion of **Z-Codes (Routine Examinations)**

#### Critical Implication
> High transaction volume alone does not equal high risk

**Protective Factor:**
- Routine care (Z-codes) distinguishes "managed" polypharmacy from "unmanaged" ADE risk
- Regular monitoring and preventive care mitigate polypharmacy risks

---

## 7. Discussion

### 7.1 Clinical Implications (Deprescribing)

#### Actionable Insights
- **"Interaction Effect" Score** can guide pharmacists in prioritizing which drugs to deprescribe first
- Focus on high-synergy pairs rather than simple drug counts
- Enable personalized medication optimization strategies

### 7.2 Limitations

#### Primary Limitation: Claims Data Constraints
- **Missing:** Laboratory values (e.g., kidney function, liver enzymes)
- **Impact:** Lab values modulate actual drug toxicity
- **Consequences:**
  - Cannot assess physiological readiness for drugs
  - May miss individual-level contraindications
  
#### Other Considerations
- Temporal resolution limited to claim dates
- No dosage information in some cases
- Cannot capture medication adherence

---

## 8. Conclusion

### Summary
The pipeline successfully disentangled complex polypharmacy into specific, causally-weighted drug interactions using the 30-day window approach.

### Key Achievements
1. **Causal Attribution:** Moved beyond correlation to identify specific drug interaction effects
2. **Methodological Innovation:** Multi-feature FFA analysis for DDI detection
3. **Clinical Translation:** Actionable intervention rates for deprescribing decisions
4. **Risk Stratification:** Distinguished managed vs. unmanaged polypharmacy

### Research Contributions
- Validated short-term (30-day) causality windows for ADE prediction
- Demonstrated synergistic effects exceed simple additive models
- Provided network-based visualization of polypharmacy complexity

---

## Data Files

### Expected Outputs
- `interaction_analysis.parquet` - Multi-feature interaction results
- `association_rules.csv` - FP-Growth derived prescribing patterns
- `network_visualizations/` - Drug co-occurrence network graphs
- `z_code_analysis.csv` - Routine care vs. high-risk patient analysis
- `causal_scores.parquet` - Intervention rate calculations

---

## Pipeline Configuration

### Cohort: `non_opioid_ed`
- **Temporal Window:** 30 days
- **Age Range:** 65-94 years
- **Exclusions:** Opioid-related ICD codes
- **Target:** Emergency department visits (HCG codes O11, P51)

### Feature Focus
- Primary: Drug exposure codes
- Secondary: High-density transaction patterns
- Control: Z-code (routine care) adjustment

---

## References

To be populated with:
- [1.1.2, 452] - Polypharmacy prevalence literature
- [1.5] - Research question framework
- Additional citations as developed

---

## Manuscript Development Notes

### Key Differentiators from Chapter 3
1. **Temporal Scope:** 30-day vs. full-history
2. **Feature Focus:** Drugs vs. broad medical codes
3. **Analytical Approach:** Interaction analysis vs. single-feature importance
4. **Population:** Geriatric only vs. broader age ranges

### Writing Priorities
1. Clearly establish the 30-day causality rationale
2. Explain synergy detection methodology
3. Balance technical detail with clinical interpretability
4. Emphasize actionable findings for deprescribing

---

*Last Updated: January 16, 2026*
