# OODA Loop × CRISP-DM Article Taxonomy

> **OODA Loop** (top level — situational awareness layer)
> **└─ CRISP-DM** (data & analytics execution layer)

```
OODA: Observe  ── sensing, collecting, monitoring
  └─ CRISP-DM: data_understanding  (EHR/claims, FAERS, registries)
  └─ CRISP-DM: data_preparation   (cohort building, preprocessing)

OODA: Orient   ── interpreting, synthesizing, situational awareness
  └─ CRISP-DM: business_understanding  (reviews, guidelines, background)
  └─ CRISP-DM: data_understanding      (descriptive/EDA analysis)

OODA: Decide   ── modeling, pattern recognition, prediction
  └─ CRISP-DM: modeling    (ML, process mining, DDI prediction)
  └─ CRISP-DM: evaluation  (SHAP, AUC, explainability, validation)

OODA: Act      ── intervention, deployment, clinical implementation
  └─ CRISP-DM: deployment  (CDS tools, naloxone, treatment programs)
  └─ CRISP-DM: evaluation  (program evaluation, outcome assessment)
```

## Article Counts by OODA × CRISP-DM

| OODA | CRISP-DM | All | Included |
|------|----------|----:|---------:|
| observe | business_understanding | 842 | 188 |
| observe | data_understanding | 1666 | 606 |
| observe | data_preparation | 9 | 2 |
| observe | modeling | 86 | 75 |
| observe | evaluation | 46 | 31 |
| observe | deployment | 100 | 22 |
| orient | business_understanding | 1450 | 381 |
| orient | data_understanding | 215 | 134 |
| orient | data_preparation | 12 | 5 |
| orient | modeling | 330 | 249 |
| orient | evaluation | 218 | 185 |
| orient | deployment | 91 | 35 |
| decide | business_understanding | 10 | 10 |
| decide | data_understanding | 5 | 3 |
| decide | modeling | 59 | 52 |
| decide | evaluation | 97 | 95 |
| decide | deployment | 1 | 1 |
| act | business_understanding | 1148 | 947 |
| act | data_understanding | 535 | 460 |
| act | data_preparation | 10 | 8 |
| act | modeling | 126 | 120 |
| act | evaluation | 83 | 64 |
| act | deployment | 2315 | 1273 |

## Included Articles by Cell

### Observe → Business Understanding (188)

- [PMC8687093] COVID-19-related adaptations to the implementation and evaluation of a clinic-ba
- [PMC11863954] Trends in Emergency Department, Primary Care, and Behavioral Health Use for Pedi
- [PMC11359928] Pharmacogenomics of 3,4-Methylenedioxymethamphetamine (MDMA): A Narrative Review
- [PMC10585000] Mediation and longitudinal analysis to interpret the association between clozapi
- [PMC10213898] An updated examination of the perception of barriers for pharmacogenomics implem
- [PMC8210499] Conference report: inaugural Pharmacogenomics Access & Reimbursement Symposium.
- [PMC8152514] Combined Impact of Inflammation and Pharmacogenomic Variants on Voriconazole Tro
- [PMC12221022] Carbapenem-resistant Enterobacterales among patients with bloodstream infections
- [PMC10865664] Epidemiology and outcomes of multidrug-resistant bacterial infection in non-cyst
- [PMC9521186] An Italian multicentre distributed data research network to study the use, effec
- [PMC11739748] French-Speaking Network of Pharmacogenetics (RNPGx) Recommendations for Clinical
- [PMC12975074] AI in hematology: A new frontier for nursing practice and patient care.
- [PMC12507812] Artificial intelligence in nursing: a systematic review of attitudes, literacy, 
- [PMC12394176] Use of Clinical Decision Support Systems for Diagnosis and Prognosis of Gastric 
- [PMC12323535] Clinical decision-making and care pathways for people with multiple long-term co
- … and 173 more

### Observe → Data Understanding (606)

- [PMC8172471] Linkage of public health and all payer claims data for population-level opioid r
- [PMC10873771] Clinician Risk Tolerance and Rates of Admission From the Emergency Department.
- [PMC10635336] Building Data Infrastructure for Disease-Focused Health Economics Research.
- [HSHd1a01470] How All-Payer Claims Databases (APCDs) Can be Used to Examine Changes in Profess
- [PMC9340909] Cancer Treatment Data in Central Cancer Registries: When Are Supplemental Data N
- [PMC9252452] APCDs can Provide Important Insights for Surveilling the Opioid Epidemic, With C
- [PMC12598900] Transforming Pharmacovigilance With Pharmacogenomics: Toward Personalized Risk M
- [PMC11546559] A Comparison of Molecular Techniques for Improving the Methodology in the Labora
- [HSH437432f4] Application of the community dialogues method to identify ethical values and pri
- [HSH437432f4] Inaugural Pharmacogenomics Access and Reimbursement Symposium.
- [HSHa17702ec] Transcriptomic analysis reveals zinc-mediated virulence and pathogenicity in mul
- [PMC12857999] Novel approaches in linkage of data sources to explore the associations between 
- [PMC12753065] Engagement and retention in HIV care in rural Southern U.S. using an All-Payer C
- [PMC12794773] Investigating Hearing Loss and Cochlear Implantation Disparities Through the Cou
- [PMC12470924] Delay in Celiac Disease Diagnosis Among Patients with High-Risk Screening Condit
- … and 591 more

### Observe → Data Preparation (2)

- [PMC12141303] Leveraging machine learning in nursing: innovations, challenges, and ethical ins
- [HSHa17702ec] Integrating manual preprocessing with automated feature extraction for improved 

### Observe → Modeling (75)

- [PMC11765990] Exploring the Pharmacogenomic Map of Croatia: PGx Clustering of 522-Patient Coho
- [PMC11333871] Predictors of Health Care Practitioners' Intention to Use AI-Enabled Clinical De
- [PMC8820931] Interpretable Model Based on Pyramid Scene Parsing Features for Brain Tumor MRI 
- [PMC10328834] Machine learning using institution-specific multi-modal electronic health record
- [PMC8562928] Identification of important factors in an inpatient fall risk prediction model t
- [PMC10268447] Prognostic models for short-term annual risk of severe complications and mortali
- [PMC12964832] Explainable machine learning-based 28-day mortality prediction model for elderly
- [PMC12858749] Implementation of machine learning in emergency departments: A systematic review
- [PMC11420103] ChatGPT and generative AI in urology and surgery-A narrative review.
- [PMC11315549] ChatGPT in medicine: A cross-disciplinary systematic review of ChatGPT's (artifi
- [PMC8465577] Feature Explanations in Recurrent Neural Networks for Predicting Risk of Mortali
- [PMC9610299] Benchmarking emergency department prediction models with machine learning and pu
- [PMC6888922] Machine Learning-Based Predictive Modeling of Surgical Intervention in Glaucoma 
- [PMC9748586] New onset delirium prediction using machine learning and long short-term memory 
- [PMC7762727] SMART on FHIR in spine: integrating clinical prediction models into electronic h
- … and 60 more

### Observe → Evaluation (31)

- [HSH437432f4] A collaborative force for precision medicine progress: the STRIPE pharmacogenomi
- [PMC8184708] The International Society of Pharmacovigilance (ISoP) Pharmacogenomic Special In
- [PMC9505214] STRIPE partners in precision medicine series: Patient perspective.
- [PMC6991258] Comparison of Machine Learning Methods With Traditional Models for Use of Admini
- [PMC10868035] Enhancing heart failure treatment decisions: interpretable machine learning mode
- [PMC10529708] Development and Validation of the DOAC Score: A Novel Bleeding Risk Prediction T
- [HSH437432f4] Integrating pharmacogenetics in sport medicine: enhancing treatment precision an
- [PMC11744508] Artificial Intelligence-Guided Inverse Design of Deployable Thermo-Metamaterial 
- [PMC11103848] Towards proactive palliative care in oncology: developing an explainable EHR-bas
- [PMC12648812] Risk prediction models for targeted testing of HIV, hepatitis B and hepatitis C:
- [HSH437432f4] Artificial intelligence-based pharmacological approach in non-small cell lung ca
- [HSH437432f4] Revealing the Future of Pharmacovigilance in Precision Pharmaceutical Monitoring
- [PMC12988761] Development and Internal Validation of a Prediction Model for Major Cardiovascul
- [PMC12875509] Optimizing vedolizumab therapy in ulcerative colitis: A critical synthesis of tr
- [PMC12731251] Artificial Intelligence-Powered Nanosensor Platforms for Non-Invasive Breathomic
- … and 16 more

### Observe → Deployment (22)

- [PMC7902340] A Tutorial for Pharmacogenomics Implementation Through End-to-End Clinical Decis
- [PMC10879811] STRIPE partners in precision medicine: international perspective.
- [PMC12142042] Implementation and integration of a multidisciplinary pharmacogenomics service i
- [PMC9576910] Considerations into pharmacogenomics of COVID-19 pharmacotherapy: Hope, hype and
- [HSHa17702ec] Advancing Clinical Pharmacogenomics Worldwide Through the Clinical Pharmacogenet
- [PMC12482788] Artificial Intelligence in Clinical Decision-Making: A Scoping Review of Rule-Ba
- [9145] Implementation of Electronic Health Record Integration and Clinical Decision Sup
- [PMC6981801] Experience in Developing an FHIR Medical Data Management Platform to Provide Cli
- [PMC7710328] Clinical Implementation of Predictive Models Embedded within Electronic Health R
- [PMC8470765] Leveraging Clinical Decision Support and Integrated Medical-Dental Electronic He
- [PMC8324242] Contemporary clinical decision support standards using Health Level Seven Intern
- [HSHd1a01470] Long-Term Prescription Opioid Use After Injury in Washington State 2015-2018.
- [HSHa17702ec] The association between prescription opioid dispensing and opioid-related morbid
- [HSHa17702ec] Chronic prescription opioid use in pregnancy in the United States.
- [HSHd1a01470] Poststroke Telehealth Utilization Patterns Across Gender, Geography, Insurance, 
- … and 7 more

### Orient → Business Understanding (381)

- [PMC10531471] Exploring Acute Pancreatitis Clinical Pathways Using a Novel Process Mining Meth
- [HSHec106a96] The application of process mining for clinical pathways: a systematic literature
- [PMC12779139] Enhancing quality and decision-making for care pathways: An application of proce
- [PMC12355893] Process mining in healthcare: a tertiary study.
- [PMC10254617] Protocol for improving the costs and outcomes of assistive reproductive technolo
- [PMC9257062] Identifying and Investigating Ambulatory Care Sequences Before Invasive Coronary
- [PMC12746521] Insights Into Patient-Level Exposure to Actionable Pharmacogenomic Medications i
- [PMC10456128] Prevalence, comorbidities, and disease-related complications of rheumatoid arthr
- [PMC12906685] Experiences of Older People Living at Home With Medication Use: A Qualitative St
- [PMC12177157] An improved catalogue for whole-genome sequencing prediction of bedaquiline resi
- [PMC10210380] Detection of blaNDM-1,mcr-1 and MexB in multidrug resistant Pseudomonas aerugino
- [PMC12451093] Development of an Artificial Intelligence Powered Medication Risk Score Calculat
- [PMC12954736] Exploring dietary behaviors among healthcare providers: based on association rul
- [PMC11229869] Discovering hidden patterns: Association rules for cardiovascular diseases in ty
- [PMC10770384] Diabetic foot ulcers risk prediction in patients with type 2 diabetes using clas
- … and 366 more

### Orient → Data Understanding (134)

- [PMC11782707] Comparing Care Pathways Between COVID-19 Pandemic Waves Using Electronic Health 
- [PMC11150892] Data-Driven Exploration of National Health Service Talking Therapies Care Pathwa
- [PMC9810199] Exploring the potential of OMOP common data model for process mining in healthca
- [PMC8852240] Biomarker identification using dynamic time warping analysis: a longitudinal coh
- [PMC11636909] Time-dependent sequential association rule-based survival analysis: A healthcare
- [PMC10870595] Extraction frequent patterns in trauma dataset based on automatic generation of 
- [8623] Characterization of Mitoribosomal Small Subunit unit genes related immune and ph
- [PMC8005434] Cost Analysis of Emergency Department Criteria for Evaluation of Febrile Infants
- [PMC12961099] Analysis of Patients' Mobility Patterns: Insights From a Process Mining-Based Lo
- [PMC12990641] Beyond healthcare access: social deprivation and COVID-19 outcomes in dialysis p
- [PMC12729846] An Integrated Predictive Impact-Enhanced Process Mining Framework for Strategic 
- [PMC12465584] Towards robust electronic health record systems: integrating formal verification
- [PMC11558449] Investigating the service utilization and pathways of patients with alcohol use 
- [PMC11310182] Analyzing Healthcare Processes with Incremental Process Discovery: Practical Ins
- [PMC11149139] Process mining to investigate the relationship between clinical antecedents and 
- … and 119 more

### Orient → Data Preparation (5)

- [PMC12842384] Comment on Iacobescu et al. Evaluating Binary Classifiers for Cardiovascular Dis
- [PMC12191892] Stacked Ensemble Learning for Classification of Parkinson's Disease Using Telemo
- [PMC10805868] Efficacy of MRI data harmonization in the age of machine learning: a multicenter
- [PMC10832072] Enhancing deep learning classification performance of tongue lesions in imbalanc
- [PMC10850917] Trans-Balance: Reducing demographic disparity for prediction models in the prese

### Orient → Modeling (249)

- [HSHec106a96] Clinical pathways discovery for long-term and chronic patients: A process mining
- [PMC11836397] Predicting clinical pathways of traumatic brain injuries (TBIs) through process 
- [HSHec106a96] Enhancing healthcare process analysis through object-centric process mining: Tra
- [PMC9674105] Process mining-driven analysis of COVID-19's impact on vaccination patterns.
- [PMC11069464] Somtimes: self organizing maps for time series clustering and its application to
- [PMC12753315] SynVerse: a modular framework for building and evaluating deep learning-based dr
- [PMC11303792] Exploring the predictive factors of heart disease using rare association rule mi
- [PMC9602561] An Integrated Classification and Association Rule Technique for Early-Stage Diab
- [PMC9322980] The Impact of the Association between Cancer and Diabetes Mellitus on Mortality.
- [PMC12657878] Knockoff-ML: a knockoff machine learning framework for controlled variable selec
- [PMC12901364] HealthProcessAI: a technical framework and proof-of-concept for LLM-enhanced hea
- [HSHec106a96] Population-Based Cancer Screening analysis in Northern Portugal Using Process Mi
- [PMC12080779] Utilizing process mining in quality management: A case study in radiation oncolo
- [PMC12051096] Comparative Process Mining for Identifying the Critical Activities in Sepsis Tra
- [HSHec106a96] Business Process Mining in Healthcare with State-of-the-Art Open Source Tools.
- … and 234 more

### Orient → Evaluation (185)

- [PMC10950232] Explainable artificial intelligence for cough-related quality of life impairment
- [HSH5bd0f614] Early detection of Multidrug Resistance using Multivariate Time Series analysis 
- [PMC7446147] Machine learning models predicting multidrug resistant urinary tract infections 
- [HSH84b3e645] Interpretable machine learning for personalized breast cancer screening recommen
- [PMC12177056] Combinatorial discovery of microtopographical landscapes that resist biofilm for
- [PMC12157015] Methodological Review of Classification Trees for Risk Stratification: An Applic
- [HSH84b3e645] CoxFNN: Interpretable machine learning method for survival analysis.
- [PMC12783857] An interpretable model based on concept and argumentation for tabular data.
- [HSHec106a96] DYNAMITE: Integrating Archetypal Analysis and Process Mining for Interpretable D
- [PMC8104377] MGP-AttTCN: An interpretable machine learning model for the prediction of sepsis
- [PMC12796588] Explainable machine learning for preoperative relapse prediction in molecularly 
- [PMC11909980] Interpretable machine learning modeling of treatment outcomes for silver and flu
- [PMC12146919] Serum calcium-based interpretable machine learning model for predicting anastomo
- [PMC9462964] Interpretable Machine Learning-Based Prediction of Intraoperative Cerebrospinal 
- [PMC7312245] Improving Clinical Translation of Machine Learning Approaches Through Clinician-
- … and 170 more

### Orient → Deployment (35)

- [PMC8561344] Spread and scale of an electronic deprescribing software to improve health outco
- [PMC8758290] Effects of Out-of-Hospital Continuous Nursing on Postoperative Breast Cancer Pat
- [PMC8374661] Barriers to the Use of Clinical Decision Support for the Evaluation of Pulmonary
- [PMC10831391] Clinical Decision Support Tools in the Electronic Medical Record.
- [PMC8961261] Combining adult with pediatric patient data to develop a clinical decision suppo
- [PMC9890353] Patient Perspectives on a Targeted Text Messaging Campaign to Encourage Screenin
- [PMC9377482] Development of an Interoperable and Easily Transferable Clinical Decision Suppor
- [PMC12714456] Effect of Clinical Decision Support Alerts on Anticoagulation Management in Atri
- [HSH84b3e645] Revolutionizing pediatric obesity intervention strategies: From traditional grow
- [HSH84b3e645] Intragastric Balloons in the Management of Obesity: Clinical Decision Support To
- [HSHec106a96] Integrated Perspectives on Clinical Decision Support: A Comparative Analysis of 
- [HSHec106a96] [Paediatric digital clinical decision support for global health].
- [PMC11491615] TrajVis: a visual clinical decision support system to translate artificial intel
- [PMC8868039] A novel generalized fuzzy intelligence-based ant lion optimization for internet 
- [PMC10898661] Applying human-centered design to the construction of a cirrhosis management cli
- … and 20 more

### Decide → Business Understanding (10)

- [PMC7233308] Development and validation of an interpretable predictive model for short-term r
- [PMC12999357] Attitudes, Norms, and Control: What Is Shaping Fijian Children's Physical Activi
- [PMC12870296] Does living alone reshape healthcare use? Longitudinal evidence from older adult
- [PMC12988914] Shaping the Future of Evidence Generation: Real-World Data to Drive Healthcare T
- [PMC11905178] Factors shaping cleaning and disinfection practices during the COVID-19 pandemic
- [PMC8621202] Feature Importance of Acute Rejection among Black Kidney Transplant Recipients b
- [PMC8132202] A risk prediction model of gene signatures in ovarian cancer through bagging of 
- [PMC10026830] A qualitative exploration of dentists' opioid prescribing decisions within U.S. 
- [PMC6011013] Machine Learning Methods for Survival Analysis with Clinical and Transcriptomics
- [PMC10815679] Classification of Obesity among South African Female Adolescents: Comparative An

### Decide → Data Understanding (3)

- [PMC7384633] Automated EHR score to predict COVID-19 outcomes at US Department of Veterans Af
- [PMC6937754] Clinical risk prediction with random forests for survival, longitudinal, and mul
- [PMC7319095] Multimetric feature selection for analyzing multicategory outcomes of colorectal

### Decide → Modeling (52)

- [PMC9835100] Explaining the black-box smoothly-A counterfactual approach.
- [PMC9253566] An Explainable AI Approach for the Rapid Diagnosis of COVID-19 Using Ensemble Le
- [PMC11572196] Construction and SHAP interpretability analysis of a risk prediction model for f
- [PMC12348218] Artificial Intelligence in Hypertrophic Cardiomyopathy: Advances, Challenges, an
- [HSH5bd0f614] Time Series Glucose Level Detection in fuel-cell based sensors Using Machine Lea
- [HSH84b3e645] Exploring the use of association rules in random forest for predicting heart dis
- [PMC12248535] A Meta-Learning-Based Ensemble Model for Explainable Alzheimer's Disease Diagnos
- [PMC11258202] A machine learning framework for interpretable predictions in patient pathways: 
- [HSH6e12adec] Development and validation of an interpretable prediction model for the risk of 
- [HSH6e12adec] Predicting perceived likelihood of future suicide attempts in youth with non-sui
- [PMC6804853] An interpretable machine learning model for preoperative prediction of renal mas
- [HSH6e12adec] An interpretable deep-learning approach to detect biomarkers in anxious-depresse
- [HSH6e12adec] Pain prediction model based on machine learning and SHAP values for elders with 
- [PMC9389270] Prediction of conversion to dementia using interpretable machine learning in pat
- [PMC12020094] Exploring the potential and limitations of deep learning and explainable AI for 
- … and 37 more

### Decide → Evaluation (95)

- [PMC12657415] Evaluating XAI techniques under class imbalance using CPRD data.
- [PMC12465982] Fostering trust and interpretability: integrating explainable AI (XAI) with mach
- [PMC10500028] A historical perspective of biomedical explainable AI research.
- [PMC12546719] Lifestyle data-based multiclass obesity prediction with interpretable ensemble m
- [PMC12955129] Explainable AI for critical care: a systematic review of interpretable models fo
- [PMC12941188] Interpretable Machine Learning with SHAP Identifies Key Biomarkers in a Multi-Fa
- [PMC12965957] Machine Learning-Based Prediction of Institutional Delivery Dropout (IDD) Among 
- [PMC12847264] Interpretable machine learning models for beta thalassemia prediction: an explai
- [PMC12838489] Leveraging laboratory biomarkers to predict urosepsis after upper urinary tract 
- [PMC12630575] Bridging the gap: explainable ai for autism diagnosis and parental support with 
- [HSH6e12adec] Explainable Prediction of ICU Transfer in Acute Pancreatitis: A Neural Network M
- [PMC12397250] Personalized health monitoring using explainable AI: bridging trust in predictiv
- [PMC12013099] Development of explainable artificial intelligence based machine learning model 
- [PMC11917090] Interpretable machine learning models for prolonged Emergency Department wait ti
- [PMC12133505] Sarcopenia prediction model based on machine learning and SHAP values for commun
- … and 80 more

### Decide → Deployment (1)

- [PMC8955774] A Clinical Decision Support System for the Prediction of Quality of Life in ALS.

### Act → Business Understanding (947)

- [PMC10501571] Examining explainable clinical decision support systems with think aloud protoco
- [PMC11031279] Unmet need for medication for opioid use disorder among persons who inject drugs
- [PMC10525011] Does polygenic risk for substance-related traits predict ages of onset and progr
- [PMC9710250] "The idea is to help people achieve greater success and liberty": A qualitative 
- [PMC12846737] Pharmacogenomics-guided personalized medicine in a clinical setting: real-world 
- [PMC8603594] Prevalence of potentially harmful multidrug interactions on medication lists of 
- [PMC12999938] The critical need to implement pharmacogenomics in public health services: Mexic
- [HSH9b503762] Identification of polypharmacy patterns in new-users of metformin using the Apri
- [PMC12094658] Associations of childhood adversity and substance use disorder polygenic scores 
- [PMC11468455] Predicting buprenorphine adherence among patients with opioid use disorder in pr
- [PMC10964694] Liver stiffness and associated risk factors among people with a history of injec
- [PMC10593985] Correlates and Patterns in Use of Medications to Treat Opioid Use Disorder in Ja
- [PMC10233176] Prevalence and predictors of suicidality among adults initiating office-based bu
- [PMC10274123] Prenatal Morphine Exposure Increases Cardiovascular Disease Risk and Programs Ne
- [PMC10870807] The Role of Patient-Reported Social Factors in Promoting Buprenorphine Consisten
- … and 932 more

### Act → Data Understanding (460)

- [PMC8755850] Prevention and Management of Opioid use Disorder and Overdose in Adolescents and
- [PMC11550797] Dealing with adverse drug reactions in the context of polypharmacy using regress
- [PMC11437388] Buprenorphine-Precipitated Withdrawal Among Hospitalized Patients Using Fentanyl
- [PMC10195847] Sex-related differences in the prevalence of substance use disorders, treatment,
- [PMC10416118] New Persistent Opioid Use After Surgery: A Risk Factor for Opioid Use Disorder?
- [PMC8027950] Patient, prescriber, and Community factors associated with filled naloxone presc
- [PMC7488216] COVID-19 risk and outcomes in patients with substance use disorders: analyses fr
- [PMC11971676] Trends in Access to Medications for Opioid Use Disorder.
- [PMC9161014] Racial and Ethnic Disparities in Buprenorphine and Extended-Release Naltrexone F
- [PMC12968371] Older Age Is Associated With Long-Term Retention in Buprenorphine Treatment for 
- [PMC12205390] Comparison of Naltrexone Implant and Oral Buprenorphine Naloxone in The Treatmen
- [PMC11056422] Hospital and long-term opioid use according to analgosedation with fentanyl vs. 
- [PMC10715364] Comparing Antepartum and Postpartum Opioid-Related Maternal Deaths in the State 
- [PMC8817063] Association Between Benzodiazepine and Opioid Prescription and Mortality Among P
- [PMC13001623] Association between extended-release buprenorphine adherence and reduced healthc
- … and 445 more

### Act → Data Preparation (8)

- [PMC10520598] Risk factors for the development of opioid use disorder after first opioid presc
- [PMC11762330] Comparing mental health and substance use disorders in patients receiving durabl
- [PMC8011506] Prenatal Opioid Analgesics and the Risk of Adverse Birth Outcomes.
- [HSH9b503762] MRGCDDI: Multi-Relation Graph Contrastive Learning Without Data Augmentation for
- [HSHa1b5b21f] Long-term opioid use in operatively managed orthopaedic patients with fracture-r
- [PMC9631715] Prescription opioid use after vaginal delivery and subsequent persistent opioid 
- [PMC10107311] Verbal fluency functional magnetic resonance imaging detects anti-seizure effect
- [HSH9b503762] Implementation of WeChat-based patient-doctor interaction in the management of H

### Act → Modeling (120)

- [PMC12706421] A Dose-Aware Model for Revealing Dose-Risk Relationship of Drug-Drug Interaction
- [PMC9754174] Using machine learning to study the effect of medication adherence in Opioid Use
- [PMC8913049] A Data-Driven Medical Decision Framework for Associating Adverse Drug Events wit
- [PMC9898169] A deep learning method to detect opioid prescription and opioid use disorder fro
- [PMC12664681] Augmenting large language models to predict social determinants of mental health
- [PMC11718552] Utility of Candidate Genes From an Algorithm Designed to Predict Genetic Risk fo
- [HSHa1b5b21f] Evaluation of an Opioid Overdose Composite Risk Score Cutoff in Active Duty Mili
- [PMC12881380] A machine learning approach for opioid overdose risk prediction among Alabama Me
- [PMC12319639] Development and evaluation of a machine learning model to predict acute care for
- [PMC9630306] An integrated LSTM-HeteroRGNN model for interpretable opioid overdose risk predi
- [HSH9b503762] ComNet: A Multiview Deep Learning Model for Predicting Drug Combination Side Eff
- [PMC11682051] Predicting the toxic side effects of drug interactions using chemical structures
- [PMC11486503] MSDAFL: molecular substructure-based dual attention feature learning framework f
- [PMC11180398] Accurate prediction of drug combination risk levels based on relational graph co
- [PMC10782925] Drug-drug interaction prediction: databases, web servers and computational model
- … and 105 more

### Act → Evaluation (64)

- [PMC12833109] Explainable machine learning for early diagnosis of esophageal cancer: A feature
- [HSH557cd5d0] Interpretable Machine Learning Models Based on Shapley Additive Explanations for
- [PMC11881125] Interpretable Machine Learning Model to Predict Bone Cement Leakage in Percutane
- [PMC10933094] Predictive model and risk analysis for peripheral vascular disease in type 2 dia
- [PMC8180490] Developing a cognitive dysfunction risk score for use with opioid-dependent pers
- [PMC12706623] Implementation of an Opioid Use Disorder (OUD) Machine-Learning Phenotype in Rea
- [PMC8721167] Prediction of the Drug-Drug Interaction Types with the Unified Embedding Feature
- [PMC12824202] Using explainable machine learning to elucidate social and neurobehavioral risk 
- [PMC12410665] Machine learning model for predicting hepatitis C seroconversion in methadone ma
- [PMC12602038] A longitudinal observational study with ecological momentary assessment and deep
- [PMC11693614] Factors associated with cognitive flexibility in people with opioid-use disorder
- [PMC11428664] Predictive Model for Opioid Use Disorder in Chronic Pain: A Development and Vali
- [HSHa1b5b21f] Buprenorphine Initiation in the Era of High-potency Synthetic Opioids: A Call fo
- [PMC9754617] Race, trust, and COVID-19 vaccine hesitancy in people with opioid use disorder.
- [PMC7835476] Bradycardia Shock Caused by the Combined Use of Carteolol Eye Drops and Verapami
- … and 49 more

### Act → Deployment (1273)

- [PMC10201806] Drug use patterns and factors related to the use and discontinuation of medicati
- [PMC12417399] Polypharmacy driven synergistic toxicities in elderly breast cancer chemotherapy
- [PMC10859042] Preferences for pharmacogenomic testing in polypharmacy patients: a discrete cho
- [HSH9b503762] Medication Optimization Using Pharmacogenomic Testing in a Complex Mental Health
- [PMC12917346] Experiences of People Who Discontinue Long-Acting Injectable Buprenorphine Treat
- [PMC10754187] Impact of Emergency Department-Initiated Buprenorphine on Repeat Emergency Depar
- [PMC10480593] The association of medical providers' attitudes about naloxone and people with o
- [HSHa1b5b21f] Cost-effectiveness of flexible take-home buprenorphine-naloxone versus methadone
- [PMC10063455] Predictors of opioid overdose during the COVID-19 pandemic: The role of relapse,
- [PMC9594511] Experience and response to a randomised controlled trial of extended-release inj
- [PMC9616126] Characteristics and correlates of fentanyl preferences among people with opioid 
- [PMC9031431] Pilot survey of prescription opioid use patterns and engagement with harm-reduct
- [PMC9378535] "Just give them a choice": Patients' perspectives on starting medications for op
- [PMC9154036] Development and assessment of PharmaCheck: an electronic screening tool for the 
- [PMC9540177] Contextualized Drug-Drug Interaction Management Improves Clinical Utility Compar
- … and 1258 more
