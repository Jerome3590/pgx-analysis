**Chapter 1: Literature Review & Background**

* **The Clinical Challenges:** Explore the dual crises of opioid use disorder (OUD) in younger/mid-age populations and polypharmacy-related adverse events in older adults 1-3. Discuss how both pathways heavily impact Emergency Department (ED) utilization 2, 3\.  
* **Healthcare Data Complexity:** Detail the inherent chaos of electronic health records (EHR) and All-Payer Claims Databases (APCD), focusing on the "warped time" of patient journeys, uneven schedules, and administrative noise 4-6.  
* **Explainable AI (XAI) vs. Black-Box Models:** Contrast traditional machine learning with the need for interpretability in healthcare. Introduce SHAP (quantitative feature attribution) and Formal Feature Attribution (FFA) as methods to extract verifiable, symbolic Boolean logic from tree-based models 7, 8\.  
* **Pharmacogenomics (PGx):** Define the role of the Clinical Pharmacogenetics Implementation Consortium (CPIC) guidelines and how individual genetic variants (SNPs) dictate drug metabolism and personalized dosing 9, 10\.

**Chapter 2: Methodology & Pipeline Architecture**

* **Data Engineering & Processing:** Detail the APCD pipeline using DuckDB's partition-first strategy, enabling massive parallelization and global demographic imputation 11, 12\.  
* **Cohort Design & Temporal Validation:** Explain the dual-target system (ICD/CPT targets vs. HCG-based ED visits) 13, 14\. Highlight the strict temporal validation (training on 2016-2018, holding out 2019, excluding 2020\) and 5:1 statistical independence control sampling 15, 16\.  
* **Target Leakage Prevention:** Explicitly outline the core architectural rule: **Dynamic Time Warping (DTW), BupaR, and FP-Growth are strictly excluded from feature engineering** to prevent target leakage 17-20. Explain how DTW instead acts as a noise filter in Step 1b to remove standard-care administrative protocols 21, 22\.  
* **Feature Selection via Consensus:** Describe the Monte Carlo Cross-Validation (MC-CV) approach that rewards model agreement (CatBoost, XGBoost, XGBoost RF) to isolate the most robust predictive features 23, 24\.

**Chapter 3: Predicting Opioid-Related ED Visits (Opioid ED Cohort)**

* **Cohort Definition:** Focus on patients with opioid-related diagnosis codes (e.g., F1120) across a comprehensive feature space that includes **ICD codes, CPT codes, drug names, drug counts, CPIC drug counts, and n\_events** 14, 25\.  
* **Exploratory Sequence Analysis:** Showcase how BupaR process mining identifies common clinical pathways, while DTW clusters patients into high-risk trajectory archetypes based on their healthcare timelines 26-28.  
* **Causal Feature Attribution:** Detail the application of the "Consensus Filter," which requires a feature to have high CatBoost SHAP importance *and* be describable by XGBoost logical rules before it undergoes counterfactual causal intervention 23, 29\.

**Chapter 4: Polypharmacy and Drug Interactions (Non-Opioid ED Cohort)**

* **Cohort Definition:** Focus on HCG-based ED visits (excluding opioid patients) 14, 30\. Highlight the unique constraint of this model: **the feature space is restricted purely to drug names, drug counts, CPIC counts, and n\_events** to function as a dedicated "drug interaction causal calculator" 31, 32\.  
* **Multi-Feature Combinatorial Analysis:** Explain how the FFA causal pipeline tests all combinations of important features (pairs and triplets) without an arbitrary top-K limit 33, 34\.  
* **Synergy vs. Antagonism:** Demonstrate how the model identifies specific drug-drug interactions that causally increase ED visit risk (positive synergy) or have a neutralizing/protective effect (negative antagonism) 35, 36\.

**Chapter 5: Translation to Practice – The Personalized Medicine (PGx) Risk Dashboard**

* **Clinical Decision Support:** Describe the architecture of the serverless AWS dashboard that distills the complex pipeline into an interactive tool for providers 37, 38\.  
* **Ensemble Risk Scoring:** Explain how the dashboard generates real-time predictions by combining CatBoost, XGBoost, and XGBoost RF using performance-weighted averaging (based on MC-CV PR-AUC and LogLoss metrics) 39, 40\.  
* **Visualizing the Patient Context:** Detail how the dashboard safely surfaces FP-Growth co-occurrence networks, BupaR process flows, and DTW trajectories purely for exploratory visual context alongside the causal factors 41, 42\.  
* **The PGx Patient Card:** Explain the architectural separation of genetic data—how the predictive models use simplified **CPIC drug counts** to reduce complexity, while the dashboard's Tab 4 uses actual **allele frequencies and patient-inputted SNPs** matched against the CPIC master database to generate personalized, anonymous dosing modifications 10, 43, 44\.

