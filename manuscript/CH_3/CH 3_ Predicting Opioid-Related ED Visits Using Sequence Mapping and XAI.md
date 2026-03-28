Here is a comprehensive draft for Chapter 3, structured for your target journal (*Clinical and Translational Science*). It combines your README repository structures, specific drafting prompts, and methodological filler/background from the pediatric transplant and VR simulation papers to justify your modeling choices.

### Chapter 3: Translating Claims Data to Clinical Pathways: Sequence Mapping and Explainable AI for Predicting Opioid-Related ED Visits

#### 3.1 Introduction & Research Question

The clinical crisis of opioid-related Emergency Department (ED) visits disproportionately affects the active workforce and young adult population 1, 2\. While it is well understood that patients with Opioid Use Disorder (OUD) navigate complex and highly irregular healthcare journeys, traditional black-box predictive models fail to translate these warped timelines into interpretable clinical interventions 3\. To address this translational gap, this study moves beyond static risk scoring to identify verifiable, modifiable risk factors prior to the onset of dependence 1, 4\.  
This chapter answers the following primary research question: *"What are the causal, temporal drivers of opioid-related emergency department visits, and can we identify high-risk trajectories before the diagnosis of dependence (F11.20) using sequence mapping and explainable AI?"* 2, 5\.

#### 3.2 Cohort Definition & Feature Space

To capture long-term addiction trajectories without encoding target information, the Opioid ED Cohort was bounded by the target variable F11.20 (Opioid use disorder with intoxication) 6, 7\. The study focuses on young and mid-life adults (ages 13–64), stratified into age bands (13–24, 25–44, 45–54, 55–64) to acknowledge developmental differences in risk 4, 8\.  
A matched sampling approach without replacement was utilized to construct the cohort, matching controls to cases at a 5:1 ratio 9, 10\. Controls were strictly required to have zero opioid-related ICD codes to ensure strict separation 10\. To prevent post-target leakage, all events occurring on or after the first F11.20 diagnosis were explicitly excluded from the model-ready dataset 10, 11\.  
The comprehensive feature space utilizes pre-treatment covariates and treatments, including ICD diagnosis codes, CPT procedure codes, drug exposures, drug counts, and the total number of events (n\_events) 7, 12\. Furthermore, pharmacogenomics (PGx) proxies, specifically CPIC drug counts, were engineered into the feature space to aid in precision-medicine predictions 6, 7\.

#### 3.3 Exploratory Sequence Mapping for Clinical Insight

Patients rarely follow linear clinical pathways. To handle the "warped time" of patient journeys and irregular schedules, we leveraged exploratory process mining and sequence analysis 13, 14\.  
Dynamic Time Warping (DTW) was applied to measure the similarity between the temporal sequences of patient measurements 15\. DTW acts as a noise filter to remove standard-care administrative protocols, aligning patient timelines to cluster high-risk individuals into distinct trajectory archetypes 14, 16, 17\. Concurrently, BupaR process mining was utilized to map the most common clinical pathways leading to an F11.20 event, identifying common "gateway" procedures and non-opioid pain management failures that precede opioid escalation 13, 17, 18\.  
While these techniques are essential for translating raw administrative claims into actionable clinical archetypes, association rules (FP-Growth) and process mining are utilized strictly for exploratory analysis and visualization; they are excluded from predictive feature engineering to preserve methodological rigor and prevent target leakage 19, 20\.

#### 3.4 Per-Bin Modeling Architecture for Precision Prediction

Traditional generalized models often fail on highly imbalanced, real-world healthcare data. Patients with very few events (low density) have fundamentally different clinical profiles and feature distributions compared to highly active patients (extreme density) 21, 22\. A single full-cohort model is pulled toward the dominant density group, underperforming on the tails 21\.  
To overcome this, we employed a "per-bin" modeling architecture (train\_per\_bin()). The cohort was divided by patient activity levels into four distinct models based on event density: low, medium, high, and extreme 16, 22\. Stratifying by density allows each model to independently tune its hyperparameters, calibration, and decision boundaries, providing drastically improved Precision-Recall Area Under the Curve (PR-AUC) metrics, particularly for the minority class 16, 21, 22\.  
The core models utilized gradient boosting algorithms: XGBoost (gradient boosting with trees), XGBoost RF (random forest-style boosting), and CatBoost 23, 24\. CatBoost was specifically selected to anchor our feature importance screening due to its advanced handling and validated superior performance for representing categorical variables without requiring excessive one-hot encoding 25, 26\. Models were temporally validated using training data from 2016–2018, with a 2019 holdout test set (excluding 2020 to prevent COVID-19 healthcare disruption bias) 20, 27\. The final ensemble weighted the candidates based on optimal balance between Recall (sensitivity) and PR-AUC 24, 28\.

#### 3.5 Causal Feature Attribution & The Consensus Filter

To extract verifiable, causal drivers of OUD events from tree-based models, we introduced the "Consensus Filter" 24, 29\. Traditional machine learning identifies correlations, but clinical interventions require causality.  
The Consensus Filter requires a feature to simultaneously satisfy two conditions:

1. **High SHAP Importance:** Quantitative global and local feature importance (derived natively from CatBoost and XGBoost binaries) 24, 30\.  
2. **Formal Feature Attribution (FFA):** The feature must be describable by XGBoost logical/symbolic rules 24, 30\.

By requiring the convergence of evidence from these multiple analytical perspectives, we drastically reduce false positives 31\. Only features that pass this dual-confirmation filter undergo counterfactual causal intervention (calculating the Intervention Rate, or IR) to quantify the exact risk reduction achieved if specific features (e.g., concomitant benzodiazepine prescriptions) are removed 32, 33\. This ensures that the model’s logic is clinically actionable and verifiable by healthcare providers 32\.

### Data Needed to Fill in Gaps for Final Publication

To complete the Results and Visuals sections of this manuscript, please provide the following missing data points from your pipeline outputs:

* **Cohort Sizes:** The final N for both the *Cases* and *Controls* in the opioid\_ed cohort after the 5:1 matching and Step 1b event filtering.  
* **Model Performance Metrics (Section 4.1):** The exact numeric outputs for **Recall, Precision, and PR-AUC** on the 2019 holdout set, stratified by the four age bands (13–24, 25–44, 45–54, 55–64) 31\.  
* **Top Aggregated Features (Section 4.2):** The specific, finalized list of top features ranked by the ensemble (exact drug names, ICD codes, and CPT procedures that passed the consensus filter) beyond the generic "Hydrocodone/Fentanyl" placeholders 18\.  
* **Causal Intervention Rates (Section 6.2):** The specific calculated Intervention Rate (IR) impact numbers from your FFA counterfactual testing (e.g., "Removing feature X reduced OUD risk by Y%") 33\.  
* **Visual Output Files (Section 3.6/3.7):** The generated S3/dashboard artifacts to be embedded as figures, specifically:  
* BupaR Pre-F11.20 flow diagrams 18\.  
* DTW Trajectory Cluster graphs showing the "Rapid Onset" vs. "Chronic Escalation" timelines 17\.  
* SHAP summary plots / FFA symbolic rule trees 33, 34\.

