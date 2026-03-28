**CHAPTER 1 Introduction**  
**1.1 Background: Healthcare Costs, Outcomes, and Overall System Performance**The opioid epidemic remains a leading public health crisis, with opioid use disorder (OUD) and polypharmacy-related adverse drug events (ADEs) heavily burdening emergency department (ED) utilization and driving up overall **healthcare costs and outcomes** 1, 2\. Addressing this crisis requires optimizing **overall system performance**, which relies on the intricate balance of three core pillars: **People, Processes, and Technology**.  
Historically, clinicians (the *People*) have lacked real-time, explainable tools to translate complex patient data into actionable prescribing guidance 3\. However, recent advances in machine learning (ML), process mining, and explainable artificial intelligence (XAI) act as a powerful **Technology/Process lever** 4, 5\. By utilizing tools like SHapley Additive exPlanations (SHAP) and Formal Feature Attribution (FFA), this **Technology/Process lever feeds intelligence directly into the People lever**, functioning as a clinical **OODA Loop** (Observe, Orient, Decide, Act) 6-8. This allows clinical staff to rapidly observe patient trajectories, orient to high-risk drug interactions, and decide on preventative interventions before adverse events occur 3, 9\.  
**1.1.2 EHR/APCD Systems and Actionable Intelligence**To fuel this OODA loop, comprehensive data infrastructure is critical. While Electronic Health Records (**EHR**) provide deep clinical narratives, **APCD (All-Payer Claims Database)** systems serve as a vital, population-level input source, capturing crucial pre-treatment covariates, drug exposures, and longitudinal claims data across care settings 10-12.  
Translating these massive APCD datasets into **Actionable Intelligence** requires a structured, end-to-end data mining framework, conceptually aligned with the **CRISP-DM** (Cross-Industry Standard Process for Data Mining) methodology. As detailed in this project's execution workflow, this process involves rigorous data preparation (APCD event filtering and noise reduction), feature engineering via FP-Growth pattern mining, predictive modeling using gradient boosting algorithms (CatBoost/XGBoost), and deployment of interpretable XAI rules 13-15.  
By applying this structured data mining process to APCD systems, the extracted actionable intelligence is specifically targeted to solve two distinct clinical challenges:

![Figure 1: OODA Phase Word Frequency Distribution — 9,454 literature sources](Literature_Review/data/wordclouds/wordcloud_ooda_grid.png)

*Figure 1. Word frequency distributions segmented by OODA phase across the 9,454-article systematic review corpus. Observe-phase literature (upper-left) centers on diagnostic surveillance and risk monitoring; Orient-phase literature (upper-right) reflects clinical heuristics and decision frameworks; Decide-phase literature (lower-left) emphasizes prediction, explainability, and model interpretability; Act-phase literature (lower-right) captures intervention, deployment, and prescribing guidance. The lexical dominance of "opioid," "pharmacogenomic," and "machine learning" across all four quadrants confirms thematic alignment with the dissertation's core research questions.*

* **Improve Opioid ED Visits (RQ 1):** Utilizing causal-oriented modeling to discover risk factors and predictive features for the *Opioid ED Cohort*, specifically targeting F11.20 (Opioid use disorder with intoxication) 16, 17\.  
* **Improve Polypharmacy (RQ 2):** Uncovering complex, higher-order prescribing patterns and drug-drug interactions that lead to hospitalization, targeting the *Polypharmacy/Non-Opioid ED Cohort* to reduce preventable adverse drug events, particularly in older demographic bands 16, 18, 19\.

**1.2 Research Aims and Questions**The overall aim of this research is to evaluate existing machine learning applications in healthcare and propose a multi-modal data architecture that combines pattern mining, XAI, and clinical decision support to improve overall system performance 20, 21\. Guided by the need to generate actionable intelligence, the primary research questions are:

1. **RQ 1 (Improve Opioid ED Visits):** How can advanced predictive modeling and process mining tools (e.g., BupaR, CatBoost) be optimally applied to APCD data to predict and reduce opioid-related emergency department visits? 16, 22  
2. **RQ 2 (Improve Polypharmacy):** In what ways can the extraction of frequent drug co-occurrence patterns (via FP-Growth) and Formal Feature Attribution (FFA) identify temporal polypharmacy risks to prevent adverse drug event hospitalizations? 14, 18, 23

**1.3 Structure and Content of the Workflow**This research is structured around a highly scalable data mining pipeline running on DuckDB and AWS S3, ensuring data integrity from raw input to deployed intelligence 24, 25\. The workflow encompasses:

* **Phase 1 (Data & Cohort Creation):** APCD/EHR input processing, age imputation, and creating strict 5:1 control-to-case cohorts 24, 26\.  
* **Phase 2 (Risk Modeling & Feature Selection):** Gradient-boosted tree models (CatBoost/XGBoost) are trained and optimized (Optuna), producing updated feature importances.  
* **Phase 3 (Causal Analysis & Explainability):** Feature importances are used for causal analysis (SHAP, FFA), generating clinically interpretable rules and robust attribution.  
* **Phase 4 (Pattern & Process Mining):** Prioritized features from causal analysis are then fed into pattern mining (FP-Growth) and process mining (BupaR) for further exploratory and network analysis.  

> **Note:** In our production workflow, feature importances derived from optimized risk models are first used for causal analysis (explainability), and only then are these prioritized features fed into pattern and process mining for further exploratory analysis. Full methodological details are provided in Chapter 2.

---

**1.4 Systematic Literature Review**A systematic quantitative literature review (SQLR) was conducted in accordance with PRISMA 2020 guidelines to characterize the existing evidence base, identify research gaps, and contextualize the proposed methodology. Nine PubMed search strings were constructed to target the primary research questions and analytical methods addressed in this dissertation, yielding a final corpus of 9,454 unique articles screened for relevance.

**1.4.1 PRISMA Flow**

![Figure 2: PRISMA 2020 Systematic Review Flow — 9,454 articles screened](Literature_Review/figures/fig_prisma_flowchart.png)

*Figure 2. PRISMA 2020 flow diagram for the systematic literature review. Of 9,571 records identified via PubMed API, 151 duplicates were removed. Following automated screening using composite relevance scoring and PyTextRank phrase extraction, 5,839 articles were classified as eligible for inclusion. Full-text retrieval achieved 95.8% coverage (9,056/9,454 articles) via PMC Open-Access API, EuropePMC/CORE free OA scan, and VCU EZProxy. A total of 5,699 articles with full text are included in the final synthesis.*

**1.4.2 Topic Volume and Annual Trends (2021–2025)**

Table 1 shows article counts by search topic and year for the included corpus, restricted to 2021–2025 where publication density is sufficient for trend interpretation. Topics are organized by OODA phase and research question alignment. Rapid growth in Drug-Drug Interactions (+111% from 2021 to 2025), DuckDB/OLAP Analytics (+116%), and Interpretability/SHAP (+4,200%) reflects accelerating methodological development in areas directly supporting this dissertation's analytical framework.

*Table 1. Annual article counts by search topic (included corpus, 2021–2025).*

| Topic | RQ | 2021 | 2022 | 2023 | 2024 | 2025 | Total |
|---|---|---:|---:|---:|---:|---:|---:|
| Opioid Use Disorder | RQ1 | 296 | 254 | 254 | 240 | 321 | 1,473 |
| Drug-Drug Interactions | RQ2 | 190 | 194 | 219 | 258 | 400 | 1,373 |
| DuckDB/OLAP Analytics | Arch | 74 | 81 | 95 | 100 | 160 | 554 |
| FP-Growth / Assoc. Rules | N4 | 77 | 78 | 91 | 75 | 113 | 465 |
| Process Mining (BupaR) | N2/N3 | 63 | 82 | 82 | 68 | 110 | 442 |
| APCD / Claims Analysis | RQ1,2 | 68 | 83 | 78 | 66 | 76 | 408 |
| Target Leakage Prevention | RQ1,2 | 2 | 13 | 10 | 20 | 47 | 115 |
| Pharmacovigilance | RQ1,2 | 13 | 13 | 15 | 24 | 28 | 104 |
| Interpretability / SHAP | N5 | 1 | 5 | 7 | 19 | 43 | 95 |
| Polypharmacy / DDI | RQ2 | 14 | 14 | 15 | 17 | 21 | 88 |
| Dynamic Time Warping | N1 | 0 | 2 | 4 | 2 | 8 | 17 |
| CatBoost / XGBoost | RQ1,2 | 2 | 4 | 1 | 3 | 3 | 13 |
| Temporal Causality | RQ1 | 0 | 2 | 2 | 1 | 2 | 9 |
| Opioid ED Prediction (ML) | RQ1 | 1 | 1 | 0 | 0 | 4 | 6 |
| CPT Code + Opioid Risk | RQ1 | 3 | 0 | 3 | 0 | 1 | 8 |
| Polypharmacy ED Prediction | RQ2 | 0 | 0 | 1 | 2 | 0 | 3 |
| Routine Care Utilization | N1 | 0 | 1 | 1 | 1 | 2 | 6 |

**1.4.3 Research Gaps**

Several topics central to this dissertation's research questions are sparsely represented in the existing literature, providing direct justification for the proposed work. Table 2 enumerates the most significant gaps identified through the systematic review, defined as search topics yielding fewer than 30 included articles across the full 2015–2025 window.

*Table 2. Identified research gaps — topics with critically low literature coverage (<30 articles).*

| Topic | n | RQ | Gap Severity | Implication |
|---|---:|---|---|---|
| Polypharmacy ED Prediction | 3 | RQ2 | **Critical** | The core RQ2 outcome domain — ML prediction of polypharmacy-driven ED visits — is near-absent in existing literature, directly motivating this dissertation's contribution. |
| Opioid ED Visit ML Prediction | 6 | RQ1 | **Critical** | Despite 1,473 articles on OUD broadly, only 6 directly address ML-based prediction of opioid-related ED visits from claims data. |
| Routine Care Utilization Patterns | 6 | N1 | **Significant** | Trajectory-based modeling of routine vs. unplanned care utilization using administrative claims is sparsely represented. |
| CPT Code + Opioid Risk | 8 | RQ1 | **Significant** | Use of CPT procedure codes as predictors of opioid risk in claims-based models has not been systematically explored. |
| Temporal Causality in Claims | 9 | RQ1 | **Significant** | Formal temporal ordering and drug window formalism in claims data lacks established literature depth. |
| CatBoost / XGBoost on Claims | 13 | RQ1,2 | **Moderate** | The specific model class used throughout this dissertation (gradient-boosted trees on APCD claims) has limited published benchmarks. |
| Dynamic Time Warping (Clinical) | 17 | N1 | **Moderate** | DTW applied to longitudinal clinical trajectories from administrative data remains a small niche. |
| Black-Box ML + CDS Integration | 26 | N5 | **Emerging** | Interpretability-first CDS system design is an active but not yet standardized area; few papers address deployment of XAI outputs into clinical workflows. |

