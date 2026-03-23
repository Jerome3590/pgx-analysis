
## Title: 
Improving Hospital Operational Excellence and Patient Operational Excellence with Precision Medicine: A Case Study Utilizing Virginia’s All Payers Claim Database (APCD) and the FDA's Adverse Drug Event Reporting System (FAERS)

## Project Summary:

The goal of this project is to investigate how Hospitals can improve their reach and clinical capabilities by integrating precision medicine into their respective healthcare system. Specifically, we aim to use Virginia's All Payers Claim Database (APCD) and the FDA's Adverse Drug Event Reporting System (FAERS) to identify opportunities and challenges for implementing precision medicine at the patient level for both hospital and patient operational excellence.

---

## Chapter 1: Introduction & Literature Review

### 1.1 Introduction

#### 1.1.1 The Data-Action Gap in Healthcare
Discuss the explosion of electronic health records and claims data. Despite this abundance, clinical decision support often lacks interpretability and scalability.

#### 1.1.2 The Problem of Adverse Drug Events (ADEs)
Define the scope of preventable harm caused by medication errors, specifically focusing on two major public health crises:

- **Opioid Use Disorder (OUD):** The shift from prescription to dependence.
- **Polypharmacy in Aging Populations:** The complexity of drug interactions in patients over 65.

#### 1.1.3 Thesis Statement
Propose that a multi-stage analytical pipeline—combining scalable engineering (DuckDB), ensemble machine learning (CatBoost/XGBoost), and rigorous causal interpretation (SHAP+FFA)—can transform raw claims data into interpretable, actionable risk signals.

### 1.2 Clinical Background: The Two Cohorts

#### 1.2.1 Opioid Use Disorder (The "Opioid ED" Cohort)
- Review literature on risk factors for opioid dependence in young to mid-life adults (Ages 13–64).
- Discuss the limitations of current static prediction models that fail to capture the *temporal* nature of addiction trajectories.

#### 1.2.2 Polypharmacy and Drug Interactions (The "Non-Opioid ED" Cohort)
- Review the definition of polypharmacy (5+ concurrent medications) and its prevalence in older adults (Ages 65+).
- Discuss the challenge of identifying "synergistic" drug-drug interactions (DDIs) where the combination risk exceeds the sum of individual risks.
- Introduce the concept of "drug event explosion" and temporal windows (30-day lookback) for causality assessment.

### 1.3 Methodological Background: From Black Box to Glass Box

#### 1.3.1 Claims Data Analysis (APCD)
- Describe the Virginia All Payer Claims Database (APCD) as a longitudinal data source.
- **Gap Analysis:** Acknowledge the noise, sparsity, and lack of clinical nuance in claims data, necessitating robust preprocessing and filtering.

#### 1.3.2 Pattern and Process Mining
- **FP-Growth (Association Rules):** Literature review on using Market Basket Analysis for finding drug co-occurrences.
- **BupaR (Process Mining):** Review the use of process mining to map patient journeys and temporal sequences.
- **Critical Pivot:** Explain the methodological shift in this dissertation—moving away from using these patterns as *black-box model features* (which causes leakage) toward using them as *post-hoc explanatory visualizations*.

#### 1.3.3 Causal Machine Learning & Interpretability
- **The Interpretability Challenge:** Review the limitations of standard feature importance (permutation/gain) in high-stakes medical decisions.
- **SHAP (Shapley Additive Explanations):** Discuss SHAP as the gold standard for quantitative attribution.
- **Formal Feature Attribution (FFA):** Introduce the novel application of symbolic logic (AXP) to verify ML predictions.
- **The "Consensus Filter" Hypothesis:** Introduce your specific contribution: requiring a feature to be validated by *both* SHAP (quantitative) and FFA (logical) to be considered a causal driver.

### 1.4 Technical Background: Scalable Healthcare Analytics

#### 1.4.1 The "Big Data" Bottleneck
Discuss the computational failure points of traditional Python/Pandas workflows when handling terabytes of event-level data.

#### 1.4.2 Partition-First Architectures
Review the shift toward partition-based processing (handling data by Age Band × Year) to ensure linear scalability.

#### 1.4.3 Modern OLAP Engines
Briefly introduce DuckDB and its role in enabling single-node analytics on massive datasets.

### 1.5 Research Questions

- **RQ1 (Clinical - Opioids):** What are the causal, temporal drivers of opioid-related emergency department visits in young and mid-life adults?
- **RQ2 (Clinical - Polypharmacy):** How do multi-drug interactions and temporal sequencing causally influence adverse drug events in the elderly population?
- **RQ3 (Technical/System):** Can a scalable, modular pipeline automate the discovery of these risks while strictly preventing target leakage and ensuring privacy?

### 1.6 Dissertation Organization

- **Chapter 2 (Methodology):** Details the "System" contribution—the partition-first pipeline, the Consensus Filter (CatBoost+XGBoost), and the visualization-only architecture.
- **Chapter 3 (Study 1):** Presents findings from the Opioid ED cohort (Ages 13–64).
- **Chapter 4 (Study 2):** Presents findings from the Polypharmacy cohort (Ages 65+).
- **Chapter 5 (Translation):** Demonstrates the translation of these findings into the PGx Risk Dashboard and Patient Card.

---

## Literature Review Status

### Chapter 1 Literature Searches Completed:

#### Core Topics:
1. **Black-Box ML and Clinical Decision Support**: 24 articles
2. **APCD Analysis**: 595 articles  
3. **Pharmacovigilance/Pharmacogenomics**: 114 articles
4. **Interpretability (SHAP/Feature Importance)**: 83 articles

#### Methodological Topics:
5. **FP-Growth & Association Rules**: 922 articles
6. **Process Mining (BupaR)**: 2,790 articles
7. **CatBoost/XGBoost**: 15 articles
8. **Dynamic Time Warping (DTW)**: 220 articles
9. **Temporal Causality**: 1 article
10. **Target Leakage Prevention**: 2 articles

#### Clinical Topics:
11. **Opioid Use Disorder (OUD)**: 2,261 articles
12. **Polypharmacy**: 93 articles
13. **Drug-Drug Interactions (DDIs)**: 3 articles

#### Technical Topics:
14. **DuckDB/OLAP Analytics**: 1,935 articles

**Total: 8,859 articles** across all Chapter 1 topics

### Additional Literature Review Areas:

- **PGx Classification Models**: Risk model classification precision medicine articles
- **Risk Models with EHR/Clinical Decision Support**: Electronic health records and clinical decision support tools
- **Risk Models with FHIR Protocol**: Fast Healthcare Interoperability Resources (FHIR) integration

---

## Project Organization

The project has been reorganized to match the Chapter 1 structure for better flow and topic-based organization:

- **`data/chapter1/`**: All Chapter 1 literature organized by section (1.1, 1.2, 1.3, 1.4)
- **`data/other_chapters/`**: Literature for other chapters (PGx, EHR, FHIR)
- **`scripts/`**: All R and PowerShell scripts for running searches
- **`background/`**: Data dictionaries and reference materials
- **`abstracts/`**: Conference abstracts and presentations

See `PROJECT_STRUCTURE.md` for detailed directory structure and organization principles.
