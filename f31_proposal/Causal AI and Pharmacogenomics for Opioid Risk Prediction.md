### 1\. Front Matter

* **Project Title:** Explainable Artificial Intelligence and Pharmacogenomics for Opioid and Polypharmacy Risk Prediction: A Causal Modeling Approach using All-Payer Claims Data 1, 2\.  
* **Project Summary / Abstract:** This project addresses the "clinical decision gap" between high-performance predictive models and actionable point-of-care tools for opioid and polypharmacy risk 3\. Utilizing the **Virginia All-Payer Claims Database (APCD)**—covering 6.9 million unique patients—we implement a **partition-first architecture** achieving a **15.1× throughput improvement** 4-6. We apply a novel **Consensus-Causal Filter (SHAP ∩ FFA)** to identify modifiable drivers of adverse drug events (ADEs) 7, 8\. The project culminates in a privacy-first, serverless **PGx Risk Dashboard** providing real-time risk scores and counterfactual "What-If" analysis 9, 10\.  
* **Project Narrative:** This research is critical to public health because it moves beyond technically accurate models to provide **user-centric explainability** for vulnerable populations 11, 12\. By recovering pharmacogenomic (PGx) signals from administrative claims, we provide a scalable framework to reduce the preventable mortality associated with polypharmacy and the opioid epidemic 13, 14\.

### 2\. Specific Aims

**Overall Objective:** To apply a standardized **CRISP-DM** pipeline and the **Clinical OODA Loop** framework to develop and evaluate a "causal calculator" for opioid and polypharmacy risk 11, 15\.

* **Aim 1 (RQ1) – Identify Causal Drivers of Opioid-Related ED Visits:**Develop an ensemble model to predict opioid-related ED visits across seven age bands (13–114) using a strict **2019 prospective holdout** 16, 17\. We will use **Dynamic Time Warping (DTW)** clustering to identify "Rapid-Onset" and "Chronic-Escalation" archetypes to time clinical interventions 18, 19\.  
* **Aim 2 (RQ2) – Develop a "Causal Calculator" for Polypharmacy Interaction Risk:**Extend **Formal Feature Attribution (FFA)** to identify synergistic drug-drug interactions (e.g., Acetaminophen \+ Levofloxacin) in geriatric populations 20, 21\. We will quantify **Intervention Rate (IR) scores** to rank deprescribing priorities based on expected risk reduction 22, 23\.  
* **Aim 3 – Translate Causal Rules to Privacy-First Clinical Decision Support:**Deploy a serverless dashboard on AWS Lambda using **Imputation of Normality** to handle sparse patient inputs 24, 25\. The architecture will satisfy the **HIPAA "minimum-necessary" standard** through ephemeral, stateless compute 26, 27\.

### 3\. Research Strategy

#### Significance

Current medication safety tools rely on simple counts and pairwise checks, missing the **nonlinear risk escalation** of complex regimens 28, 29\. This project addresses the **triple gap** identified in our **PRISMA-compliant** systematic review: the omission of PGx context, the prevalence of temporal leakage, and the failure of deployment translation 30, 31\.

#### Innovation

This project is innovative because it replaces heuristic feature selection with the **Consensus Filter (SHAP ∩ FFA)**, ensuring a feature is designated causal only if confirmed by both distributional importance and structural decision logic 8, 32\. Furthermore, it demonstrates that **PGx signal is recoverable** from administrative claims, utilizing **CPIC Level A/B evidence** as a feature engineering baseline 33, 34\.

#### Approach (Validated Industry Best Practices)

* **Data Standards:** We utilize the **Virginia APCD** (1.8 TB) staged into a **Bronze-to-Gold data lake** 35, 36\.  
* **Methodological Rigor:** Adherence to the **PROBAST checklist** ensures low risk of bias in predictors and analysis 37, 38\.  
* **Engineering Best Practices:** A **partition-first architecture** scanned via **DuckDB** eliminates "shuffle" bottlenecks and enforces the **"B1" balancing constraint** to architecturally prevent temporal target leakage 6, 39, 40\.  
* **Validation:** Models (CatBoost/XGBoost) are tuned via **Optuna (Bayesian TPE)** and validated through **Monte Carlo Cross-Validation (MCCV)** 41-43.

### 4\. Biographical Sketches

#### Candidate: R. Jerome Dixon

* **Position:** PhD Student, Integrative Life Sciences, VCU 44, 45\.  
* **Personal Statement:** My research goal is to bridge administrative data and clinical support using systems theory and military decision theory (OODA loops) 46, 47\.  
* **Contributions:** Developed scalable informatics for all-payer claims (15.1× speedup) and pioneered the **Consensus Filter** for causal discovery 46, 48\.

#### Sponsor: Elvin T. Price, Pharm.D., Ph.D.

* **Position:** Director, Geriatric Pharmacotherapy Program, VCU 49, 50\.  
* **Personal Statement:** Expert in personalized medicine and translational informatics for older adults. My lab provides the clinical environment and APCD infrastructure for this fellowship 49\.  
* **Contributions:** Leader in genomic predictors of cardiovascular outcomes and health equity in precision medicine 51, 52\.

### 5\. Compliance and Data Management

* **Human Subjects:** Claims-based secondary analysis involving de-identified records; **VCU Protocol HM20022300 (IRB waiver)** is in place 35, 53\.  
* **Privacy-by-Design:** The **PGx Risk Dashboard** follows a **"compute, respond, discard"** pattern with zero persistent PII storage 27, 54\.  
* **Data Management & Sharing:** Code and derived network feature specifications will be shared via GitHub (pgx-analysis) to ensure exact computational reproducibility 55, 56\.  
* **Responsible Conduct of Research (RCR):** Training will focus on data privacy, ethics of AI in healthcare, and reproducibility standards 57, 58\.

