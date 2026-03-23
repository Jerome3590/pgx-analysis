### Phase 1: Foundation and Methodological Architecture

**Chapter 1: Introduction & Systematic Quantitative Literature Review (SQLR)**Following Nina's approach, this chapter establishes the knowledge boundaries using a rigorous SQLR, but pivots the topic to clinical predictive modeling.

* **The Clinical Challenge:** Introduce the dual crises of opioid use disorder (OUD) in younger populations and polypharmacy-related adverse events in older adults, highlighting their impact on Emergency Department (ED) utilization 1\.  
* **The Technical Challenge:** Define the complexities of All-Payer Claims Databases (APCD) and electronic health records, specifically the "warped time" of patient journeys and administrative noise 1\. Contrast traditional black-box machine learning with the necessity for Explainable AI (XAI) using SHAP and Formal Feature Attribution (FFA) 1\.  
* **SQLR Methodology (Nina’s PRISMA Approach):** Map the existing literature landscape using a PRISMA flowchart 2, 3\.  
* *Search Terms:* Define specific inclusion terms (e.g., "Explainable AI", "Pharmacogenomics", "APCD", "Opioid Use Disorder", "Clinical Decision Support") 1, 4\.  
* *Exclusion Criteria:* Explicitly exclude studies that do not use causal attribution or lack clinical validation contexts 4\.  
* **Knowledge Gaps:** Use the SQLR to prove the necessity of your research, demonstrating the lack of models that successfully combine XAI with Pharmacogenomics (PGx) guidelines (like CPIC) for personalized dosing 1\.

**Chapter 2: Methodology & Data Engineering Architecture**This chapter replaces Nina's qualitative "Systems Thinking" chapter with your standard Data Science pipeline architecture.

* **Partition-First Architecture:** Detail your 10-step pipeline, focusing on the use of DuckDB for efficient SQL-based transformations and S3-backed checkpoints to enable resumable, parallelizable jobs 5\.  
* **Cohort Design & Temporal Validation:** Explain the creation of the dual-target cohorts (Opioid vs. Non-Opioid) with a 5:1 statistical independence control sampling 6, 7\. Highlight the strict temporal validation: training on 2016–2018, holding out 2019 for testing, and explicitly excluding 2020 to prevent COVID-19 healthcare disruption leakage 8\.  
* **Target Leakage Prevention:** Clearly state the core architectural rule that Dynamic Time Warping (DTW), BupaR, and FP-Growth are strictly visualization-only and excluded from feature engineering 5, 9\. Explain how event filtering removes post-event administrative leakage *before* cohort creation 6\.  
* **Monte Carlo Cross-Validation (MC-CV):** Describe the feature selection process using CatBoost, XGBoost, and XGBoost RF to aggregate robust importance rankings 10\.

### Phase 2: Core Research (The "Manuscript" Chapters)

**Chapter 3 (Manuscript 1): Predicting Opioid-Related ED Visits & Trajectory Mapping**

* **Focus:** The Opioid ED Cohort (Target: F11.20 \- Opioid use disorder with intoxication) 7\.  
* **Feature Space:** Detail the inclusion of ICD/CPT codes, drug names, drug counts, and CPIC counts 11\.  
* **Exploratory Sequence Analysis:** Showcase your data-driven exploratory mapping. Use BupaR process mining to identify common clinical pathways and DTW to cluster patients into high-risk trajectory archetypes based on their healthcare timelines 11\.  
* **Per-Bin Modeling:** Explain the necessity of training four distinct models based on patient event density (low, medium, high, and extreme) rather than a single full-cohort model, allowing decision boundaries to adjust to patients with fundamentally different clinical activity levels 12, 13\.  
* **Causal Attribution:** Detail the "Consensus Filter," which requires features to have high CatBoost SHAP importance and be describable by XGBoost logical rules before undergoing counterfactual causal intervention 11, 14\.

**Chapter 4 (Manuscript 2): Polypharmacy, Drug Interactions, and Causal Rules**

* **Focus:** The Non-Opioid ED cohort (high-cost/geriatric conditions) 7\.  
* **The Causal Calculator:** Explain the unique constraint of this model: the feature space is restricted purely to drug names, drug counts, and CPIC counts to function as a dedicated "drug interaction causal calculator" 15\.  
* **Multi-Feature Combinatorial Analysis:** Detail how the FFA causal pipeline tests combinations of features (pairs and triplets) without an arbitrary top-K limit to generate interpretable Boolean logic rules 14, 15\.  
* **Synergy vs. Antagonism:** Use FP-Growth co-occurrence networks to visually demonstrate how the model identifies specific drug-drug interactions that causally increase ED visit risk (positive synergy) or have a protective effect (negative antagonism) 15, 16\.

**Chapter 5 (Manuscript 3): Translation to Practice – The PGx Risk Dashboard**

* **Focus:** Real-world clinical deployment and decision support 17\.  
* **Serverless Architecture:** Detail the AWS dashboard deployment using Docker, Lambda, API Gateway, and an S3 frontend 16, 18\.  
* **Ensemble Risk Scoring:** Explain how the dashboard prioritizes PR-AUC mean for imbalanced-class safety and uses proportional probability-weighted averaging across the component models to generate real-time patient risk scores 17, 19, 20\.  
* **The PGx Patient Card:** Detail the separation of genetic data. Explain how the underlying predictive models use simplified CPIC counts to prevent complexity, while the dashboard UI safely matches patient-inputted SNPs against the actual CPIC master database to generate personalized dosing modifications 17\.  
* **Visual Context:** Show how BupaR process flows and DTW trajectories are surfaced strictly for visual context alongside the causal factors 17, 20\.

### Phase 3: Synthesis

**Chapter 6: Conclusion**

* **Synthesis:** Following Nina's wrap-up structure, synthesize the findings from your three manuscript chapters 21\.  
* **Contributions:** Clearly state how your integration of XAI, APCD data processing, and Pharmacogenomics advances the field of clinical predictive modeling 1, 17\.  
* **Limitations & Future Work:** Address pipeline limitations (e.g., APCD data lag) and propose future research directions 21\.

Would you like to start drafting the specific inclusion and exclusion criteria for the Chapter 1 PRISMA flowchart, or would you prefer to map out the technical sections of the DuckDB data pipeline for Chapter 2?  
