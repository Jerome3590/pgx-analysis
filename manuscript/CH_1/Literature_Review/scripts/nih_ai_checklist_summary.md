# NIH AI Checklist × Operational Performance Tags

> **Subset**: Decide+Act articles with crisp_dm_phase = modeling or evaluation
> **N (subset)**: 365  |  **N (included)**: 337
> **12 NIH AI checklist domains**  |  **8 operational performance dimensions**

## NIH AI Checklist Domain Coverage

| Domain | Articles (subset) | Description |
|--------|:-----------------:|-------------|
| `study_design` | 89 | Study design, prospective/retrospective, trial type |
| `data_reporting` | 87 | Training/test data, sample size, splits, cohort |
| `model_transparency` | 49 | Reproducibility, open code, model documentation |
| `bias_fairness` | 112 | Bias assessment, equity, subgroup, demographic parity |
| `performance_metrics` | 199 | AUC, sensitivity/specificity, calibration, F1 |
| `explainability` | 169 | SHAP, LIME, feature importance, interpretability |
| `external_validation` | 51 | Independent/multi-site/prospective validation |
| `uncertainty_quantification` | 28 | CIs, prediction intervals, Bayesian uncertainty |
| `clinical_utility` | 146 | Decision curve, net benefit, NRI/IDI, clinical impact |
| `deployment_implementation` | 104 | EHR integration, workflow, clinical adoption |
| `safety_monitoring` | 130 | Patient safety, model drift, failure modes |
| `regulatory_ethics` | 73 | FDA, IRB, HIPAA, ethics, data governance |

## Score Distribution (# NIH AI domains addressed per article)

| Score | Articles | Bar |
|------:|---------:|-----|
| 0 | 78 | ██████████████████████████ |
| 1 | 61 | ████████████████████ |
| 2 | 14 | ████ |
| 3 | 24 | ████████ |
| 4 | 49 | ████████████████ |
| 5 | 47 | ███████████████ |
| 6 | 36 | ████████████ |
| 7 | 30 | ██████████ |
| 8 | 18 | ██████ |
| 9 | 6 | ██ |
| 10 | 1 |  |
| 11 | 1 |  |

## Operational Performance Coverage

| Dimension | Articles | Description |
|-----------|:--------:|-------------|
| `process_capacity` | 4 | Capacity planning, surge, bottleneck analysis |
| `human_resources` | 177 | Staffing, workforce, clinician workload/burden |
| `cost` | 62 | Cost-effectiveness, economic analysis, ROI |
| `process_throughput` | 53 | Wait time, LOS, turnaround, queue, flow |
| `improved_outcomes` | 12 | General outcome improvement, clinical benefit |
| `improved_healthcare_outcomes` | 43 | Mortality/morbidity/readmission reduction |
| `improved_process_performance` | 8 | Workflow, process optimization, quality metrics |
| `improved_patient_outcomes` | 252 | QoL, functional, patient-reported outcomes |

## High-Coverage Included Articles (NIH AI score ≥ 6 of 12)

| Rank | Article ID | Score | OODA | CRISP-DM | NIH AI Tags | Op Perf Tags |
|-----:|-----------|------:|------|----------|-------------|--------------|
| 1 | [11] Artificial Intelligence in Hypertrophic Cardiomyopathy:… | 11 | decide | modeling | study_design|data_reporting|model_transparency|bias_fairness|performance_metrics|explainability|external_validation|clinical_utility|deployment_implementation|safety_monitoring|regulatory_ethics | human_resources|improved_outcomes|improved_patient_outcomes |
| 2 | [839] Constructing a fall risk prediction model for hospitali… | 10 | decide | modeling | study_design|data_reporting|bias_fairness|performance_metrics|explainability|external_validation|clinical_utility|deployment_implementation|safety_monitoring|regulatory_ethics | cost|improved_healthcare_outcomes|improved_patient_outcomes |
| 3 | [1933] RADEX: a rule-based clinical and radiology data extract… | 9 | act | modeling | study_design|data_reporting|model_transparency|bias_fairness|performance_metrics|explainability|clinical_utility|deployment_implementation|safety_monitoring | human_resources|cost|process_throughput|improved_outcomes|improved_patient_outcomes |
| 4 | [2381] Beyond Atrial Fibrillation: Machine Learning Algorithm … | 9 | decide | modeling | study_design|data_reporting|bias_fairness|performance_metrics|explainability|external_validation|clinical_utility|safety_monitoring|regulatory_ethics | human_resources|improved_patient_outcomes |
| 5 | [4877] Predicting the toxic side effects of drug interactions … | 9 | act | modeling | study_design|data_reporting|bias_fairness|performance_metrics|explainability|clinical_utility|deployment_implementation|safety_monitoring|regulatory_ethics | process_throughput|improved_patient_outcomes |
| 6 | [798] Predicting five-year comorbid bipolar disorder after at… | 9 | decide | evaluation | study_design|model_transparency|bias_fairness|performance_metrics|explainability|clinical_utility|deployment_implementation|safety_monitoring|regulatory_ethics | human_resources|improved_healthcare_outcomes|improved_patient_outcomes |
| 7 | [807] Lifestyle data-based multiclass obesity prediction with… | 9 | decide | evaluation | data_reporting|model_transparency|performance_metrics|explainability|external_validation|uncertainty_quantification|clinical_utility|deployment_implementation|safety_monitoring | improved_healthcare_outcomes|improved_patient_outcomes |
| 8 | [808] Explainable Machine Learning Applied to Bioelectrical I… | 9 | decide | evaluation | study_design|data_reporting|bias_fairness|performance_metrics|explainability|external_validation|uncertainty_quantification|clinical_utility|deployment_implementation | improved_patient_outcomes |
| 9 | [1794] FKSUDDAPre: A drug-disease association prediction frame… | 8 | decide | modeling | study_design|data_reporting|performance_metrics|explainability|clinical_utility|deployment_implementation|safety_monitoring|regulatory_ethics | process_capacity|human_resources|process_throughput|improved_patient_outcomes |
| 10 | [2] Explainable AI for critical care: a systematic review o… | 8 | decide | evaluation | study_design|data_reporting|model_transparency|performance_metrics|explainability|clinical_utility|deployment_implementation|regulatory_ethics | human_resources|improved_patient_outcomes |
| 11 | [2376] AI/ML driven prediction of COPD exacerbations and readm… | 8 | decide | modeling | study_design|data_reporting|bias_fairness|performance_metrics|external_validation|clinical_utility|deployment_implementation|safety_monitoring | cost|improved_patient_outcomes |
| 12 | [2383] Simplified Machine Learning Models Can Accurately Ident… | 8 | decide | modeling | study_design|data_reporting|model_transparency|performance_metrics|external_validation|clinical_utility|safety_monitoring|regulatory_ethics | cost|process_throughput|improved_healthcare_outcomes|improved_patient_outcomes |
| 13 | [2476] Interpretable Machine Learning Framework for Diabetes P… | 8 | decide | evaluation | study_design|data_reporting|model_transparency|performance_metrics|explainability|external_validation|clinical_utility|deployment_implementation | human_resources|improved_patient_outcomes |
| 14 | [3] Explainable artificial intelligence in pancreatic cance… | 8 | decide | evaluation | model_transparency|bias_fairness|performance_metrics|explainability|external_validation|clinical_utility|deployment_implementation|regulatory_ethics | human_resources|improved_healthcare_outcomes|improved_patient_outcomes |
| 15 | [3009] Predictive Model for Opioid Use Disorder in Chronic Pai… | 8 | act | evaluation | study_design|data_reporting|bias_fairness|performance_metrics|external_validation|clinical_utility|safety_monitoring|regulatory_ethics | human_resources|improved_patient_outcomes |
| 16 | [778] Value of an automated machine learning model with post-… | 8 | decide | evaluation | study_design|model_transparency|bias_fairness|performance_metrics|explainability|external_validation|clinical_utility|deployment_implementation | human_resources|improved_patient_outcomes |
| 17 | [783] Development and validation of a machine learning model … | 8 | act | evaluation | study_design|data_reporting|bias_fairness|performance_metrics|explainability|external_validation|clinical_utility|safety_monitoring | human_resources|improved_patient_outcomes |
| 18 | [805] Development and validation of a successful aging predic… | 8 | decide | evaluation | study_design|data_reporting|model_transparency|bias_fairness|performance_metrics|explainability|clinical_utility|deployment_implementation | human_resources|improved_outcomes|improved_patient_outcomes |
| 19 | [811] Precision dosing of voriconazole in immunocompromised c… | 8 | decide | evaluation | study_design|bias_fairness|performance_metrics|explainability|external_validation|clinical_utility|safety_monitoring|regulatory_ethics | human_resources|improved_healthcare_outcomes|improved_patient_outcomes |
| 20 | [812] Construction and validation of a machine learning-based… | 8 | decide | evaluation | study_design|data_reporting|bias_fairness|performance_metrics|explainability|external_validation|clinical_utility|deployment_implementation | human_resources|improved_healthcare_outcomes|improved_patient_outcomes |
| 21 | [815] Development and validation of an interpretable multi-ta… | 8 | decide | evaluation | study_design|data_reporting|performance_metrics|explainability|external_validation|uncertainty_quantification|clinical_utility|deployment_implementation | human_resources|cost|improved_patient_outcomes |
| 22 | [822] Enhancing end-stage renal disease outcome prediction: a… | 8 | decide | evaluation | study_design|data_reporting|bias_fairness|performance_metrics|explainability|clinical_utility|deployment_implementation|safety_monitoring | human_resources|improved_patient_outcomes |
| 23 | [844] Forecasting Patient Early Readmission from Irish Hospit… | 8 | decide | evaluation | model_transparency|bias_fairness|performance_metrics|explainability|clinical_utility|deployment_implementation|safety_monitoring|regulatory_ethics | human_resources|cost|process_throughput|improved_healthcare_outcomes|improved_patient_outcomes |
| 24 | [854] Machine Learning-Based Discrimination of Cardiovascular… | 8 | decide | evaluation | study_design|data_reporting|performance_metrics|explainability|external_validation|clinical_utility|safety_monitoring|regulatory_ethics | human_resources|improved_patient_outcomes |
| 25 | [867] Impact of System and Diagnostic Errors on Medical Litig… | 8 | decide | modeling | study_design|data_reporting|model_transparency|performance_metrics|explainability|clinical_utility|safety_monitoring|regulatory_ethics | human_resources|cost|improved_patient_outcomes |
| 26 | [870] Machine Learning Models for Predicting Influential Fact… | 8 | decide | evaluation | study_design|bias_fairness|performance_metrics|explainability|clinical_utility|deployment_implementation|safety_monitoring|regulatory_ethics | human_resources|improved_process_performance|improved_patient_outcomes |
| 27 | [17] Second opinion machine learning for fast-track pathway … | 7 | decide | modeling | data_reporting|bias_fairness|performance_metrics|explainability|clinical_utility|deployment_implementation|safety_monitoring | human_resources|improved_patient_outcomes |
| 28 | [2377] Exploring the Complexity of Real-World Health Data Reco… | 7 | decide | modeling | data_reporting|bias_fairness|performance_metrics|clinical_utility|deployment_implementation|safety_monitoring|regulatory_ethics | human_resources|improved_patient_outcomes |
| 29 | [2388] Comparative Effectiveness of Machine Learning Approache… | 7 | decide | modeling | study_design|data_reporting|performance_metrics|external_validation|clinical_utility|deployment_implementation|safety_monitoring | human_resources|cost|improved_outcomes|improved_patient_outcomes |
| 30 | [2454] Explainable machine learning for early diagnosis of eso… | 7 | act | evaluation | model_transparency|bias_fairness|performance_metrics|explainability|uncertainty_quantification|clinical_utility|deployment_implementation | human_resources|improved_patient_outcomes |
| 31 | [2482] SHAP-based interpretable machine learning for Parkinson… | 7 | decide | evaluation | bias_fairness|performance_metrics|explainability|external_validation|clinical_utility|deployment_implementation|regulatory_ethics | process_capacity|cost|process_throughput|improved_patient_outcomes |
| 32 | [2534] Interpretable Machine Learning Model to Predict Bone Ce… | 7 | act | evaluation | study_design|data_reporting|performance_metrics|explainability|clinical_utility|deployment_implementation|regulatory_ethics | improved_patient_outcomes |
| 33 | [26] An Explainable AI Approach for the Rapid Diagnosis of C… | 7 | decide | modeling | study_design|performance_metrics|explainability|external_validation|uncertainty_quantification|clinical_utility|deployment_implementation | human_resources|improved_patient_outcomes |
| 34 | [2814] Identifying key risk factors for intentional self-harm,… | 7 | act | modeling | study_design|bias_fairness|performance_metrics|explainability|clinical_utility|deployment_implementation|safety_monitoring | human_resources|improved_patient_outcomes |
| 35 | [2936] Utility of Candidate Genes From an Algorithm Designed t… | 7 | act | modeling | bias_fairness|performance_metrics|external_validation|uncertainty_quantification|clinical_utility|safety_monitoring|regulatory_ethics | improved_patient_outcomes |
| 36 | [4285] Comparative clinical response, safety, and institutiona… | 7 | act | evaluation | study_design|data_reporting|bias_fairness|performance_metrics|explainability|uncertainty_quantification|safety_monitoring | human_resources|cost|process_throughput|improved_healthcare_outcomes|improved_patient_outcomes |
| 37 | [4555] Exploring Adverse Event Associations of Predicted PXR A… | 7 | act | evaluation | bias_fairness|performance_metrics|explainability|external_validation|clinical_utility|safety_monitoring|regulatory_ethics | human_resources|improved_patient_outcomes |
| 38 | [4687] Refining Drug-Induced Cholestasis Prediction: An Explai… | 7 | act | evaluation | data_reporting|model_transparency|performance_metrics|explainability|clinical_utility|safety_monitoring|regulatory_ethics | human_resources|cost|process_throughput|improved_patient_outcomes |
| 39 | [5550] Detection of potential drug-drug interactions for risk … | 7 | act | modeling | data_reporting|bias_fairness|performance_metrics|explainability|uncertainty_quantification|safety_monitoring|regulatory_ethics | human_resources|cost|improved_patient_outcomes |
| 40 | [7505] Cloud-Based Machine Learning Platform to Predict Clinic… | 7 | act | evaluation | study_design|data_reporting|bias_fairness|performance_metrics|uncertainty_quantification|clinical_utility|safety_monitoring | human_resources|improved_healthcare_outcomes|improved_patient_outcomes |
| 41 | [779] Development and validation of a machine learning predic… | 7 | decide | evaluation | study_design|data_reporting|performance_metrics|explainability|external_validation|clinical_utility|safety_monitoring | human_resources|cost|improved_patient_outcomes |
| 42 | [781] Interpretable Machine Learning with SHAP Identifies Key… | 7 | decide | evaluation | bias_fairness|performance_metrics|explainability|uncertainty_quantification|clinical_utility|deployment_implementation|safety_monitoring | human_resources|improved_patient_outcomes |
| 43 | [789] A Multicohort Machine Learning Framework to Predict Mor… | 7 | decide | evaluation | study_design|data_reporting|bias_fairness|performance_metrics|explainability|external_validation|clinical_utility | human_resources|improved_patient_outcomes |
| 44 | [794] Shifting Determinants of Mortality Risk After Orthotopi… | 7 | decide | evaluation | study_design|bias_fairness|performance_metrics|explainability|clinical_utility|safety_monitoring|regulatory_ethics | human_resources|improved_outcomes|improved_patient_outcomes |
| 45 | [796] Leveraging laboratory biomarkers to predict urosepsis a… | 7 | decide | evaluation | study_design|performance_metrics|explainability|external_validation|clinical_utility|deployment_implementation|regulatory_ethics | human_resources|cost|process_throughput|improved_patient_outcomes |
| 46 | [797] Development and Statistical Validation of a Machine Lea… | 7 | decide | evaluation | study_design|data_reporting|model_transparency|bias_fairness|performance_metrics|explainability|clinical_utility | human_resources|improved_healthcare_outcomes|improved_patient_outcomes |
| 47 | [809] Explainable machine learning models for predicting of p… | 7 | decide | evaluation | data_reporting|performance_metrics|explainability|clinical_utility|deployment_implementation|safety_monitoring|regulatory_ethics | human_resources|cost|improved_patient_outcomes |
| 48 | [819] Development and validation of the machine learning mode… | 7 | decide | evaluation | study_design|bias_fairness|performance_metrics|explainability|external_validation|clinical_utility|deployment_implementation | human_resources|improved_healthcare_outcomes|improved_patient_outcomes |
| 49 | [820] Systemic immune-inflammatory biomarkers combined with t… | 7 | decide | evaluation | study_design|data_reporting|model_transparency|performance_metrics|explainability|clinical_utility|deployment_implementation | human_resources|cost|improved_patient_outcomes |
| 50 | [829] Machine learning-based prediction of diabetic periphera… | 7 | decide | evaluation | data_reporting|bias_fairness|performance_metrics|explainability|clinical_utility|deployment_implementation|regulatory_ethics | human_resources|cost|improved_patient_outcomes |
| … | +42 more articles | | | | | |

## All Tagged Articles (score ≥ 1, sorted by score)

- **[11/12]** [11] Artificial Intelligence in Hypertrophic Cardiomyopathy: Advances, Chal  
  NIH: `study_design|data_reporting|model_transparency|bias_fairness|performance_metrics|explainability|external_validation|clinical_utility|deployment_implementation|safety_monitoring|regulatory_ethics`  
  OpPerf: `human_resources|improved_outcomes|improved_patient_outcomes`
- **[10/12]** [839] Constructing a fall risk prediction model for hospitalized patients us  
  NIH: `study_design|data_reporting|bias_fairness|performance_metrics|explainability|external_validation|clinical_utility|deployment_implementation|safety_monitoring|regulatory_ethics`  
  OpPerf: `cost|improved_healthcare_outcomes|improved_patient_outcomes`
- **[9/12]** [1933] RADEX: a rule-based clinical and radiology data extraction tool demons  
  NIH: `study_design|data_reporting|model_transparency|bias_fairness|performance_metrics|explainability|clinical_utility|deployment_implementation|safety_monitoring`  
  OpPerf: `human_resources|cost|process_throughput|improved_outcomes|improved_patient_outcomes`
- **[9/12]** [2381] Beyond Atrial Fibrillation: Machine Learning Algorithm Predicts Stroke  
  NIH: `study_design|data_reporting|bias_fairness|performance_metrics|explainability|external_validation|clinical_utility|safety_monitoring|regulatory_ethics`  
  OpPerf: `human_resources|improved_patient_outcomes`
- **[9/12]** [4877] Predicting the toxic side effects of drug interactions using chemical   
  NIH: `study_design|data_reporting|bias_fairness|performance_metrics|explainability|clinical_utility|deployment_implementation|safety_monitoring|regulatory_ethics`  
  OpPerf: `process_throughput|improved_patient_outcomes`
- **[9/12]** [798] Predicting five-year comorbid bipolar disorder after attention-deficit  
  NIH: `study_design|model_transparency|bias_fairness|performance_metrics|explainability|clinical_utility|deployment_implementation|safety_monitoring|regulatory_ethics`  
  OpPerf: `human_resources|improved_healthcare_outcomes|improved_patient_outcomes`
- **[9/12]** [807] Lifestyle data-based multiclass obesity prediction with interpretable   
  NIH: `data_reporting|model_transparency|performance_metrics|explainability|external_validation|uncertainty_quantification|clinical_utility|deployment_implementation|safety_monitoring`  
  OpPerf: `improved_healthcare_outcomes|improved_patient_outcomes`
- **[9/12]** [808] Explainable Machine Learning Applied to Bioelectrical Impedance for Lo  
  NIH: `study_design|data_reporting|bias_fairness|performance_metrics|explainability|external_validation|uncertainty_quantification|clinical_utility|deployment_implementation`  
  OpPerf: `improved_patient_outcomes`
- **[8/12]** [1794] FKSUDDAPre: A drug-disease association prediction framework based on F  
  NIH: `study_design|data_reporting|performance_metrics|explainability|clinical_utility|deployment_implementation|safety_monitoring|regulatory_ethics`  
  OpPerf: `process_capacity|human_resources|process_throughput|improved_patient_outcomes`
- **[8/12]** [2] Explainable AI for critical care: a systematic review of interpretable  
  NIH: `study_design|data_reporting|model_transparency|performance_metrics|explainability|clinical_utility|deployment_implementation|regulatory_ethics`  
  OpPerf: `human_resources|improved_patient_outcomes`
- **[8/12]** [2376] AI/ML driven prediction of COPD exacerbations and readmissions: a syst  
  NIH: `study_design|data_reporting|bias_fairness|performance_metrics|external_validation|clinical_utility|deployment_implementation|safety_monitoring`  
  OpPerf: `cost|improved_patient_outcomes`
- **[8/12]** [2383] Simplified Machine Learning Models Can Accurately Identify High-Need H  
  NIH: `study_design|data_reporting|model_transparency|performance_metrics|external_validation|clinical_utility|safety_monitoring|regulatory_ethics`  
  OpPerf: `cost|process_throughput|improved_healthcare_outcomes|improved_patient_outcomes`
- **[8/12]** [2476] Interpretable Machine Learning Framework for Diabetes Prediction: Inte  
  NIH: `study_design|data_reporting|model_transparency|performance_metrics|explainability|external_validation|clinical_utility|deployment_implementation`  
  OpPerf: `human_resources|improved_patient_outcomes`
- **[8/12]** [3] Explainable artificial intelligence in pancreatic cancer prediction: f  
  NIH: `model_transparency|bias_fairness|performance_metrics|explainability|external_validation|clinical_utility|deployment_implementation|regulatory_ethics`  
  OpPerf: `human_resources|improved_healthcare_outcomes|improved_patient_outcomes`
- **[8/12]** [3009] Predictive Model for Opioid Use Disorder in Chronic Pain: A Developmen  
  NIH: `study_design|data_reporting|bias_fairness|performance_metrics|external_validation|clinical_utility|safety_monitoring|regulatory_ethics`  
  OpPerf: `human_resources|improved_patient_outcomes`
- **[8/12]** [778] Value of an automated machine learning model with post-hoc explanation  
  NIH: `study_design|model_transparency|bias_fairness|performance_metrics|explainability|external_validation|clinical_utility|deployment_implementation`  
  OpPerf: `human_resources|improved_patient_outcomes`
- **[8/12]** [783] Development and validation of a machine learning model for predicting   
  NIH: `study_design|data_reporting|bias_fairness|performance_metrics|explainability|external_validation|clinical_utility|safety_monitoring`  
  OpPerf: `human_resources|improved_patient_outcomes`
- **[8/12]** [805] Development and validation of a successful aging prediction model for   
  NIH: `study_design|data_reporting|model_transparency|bias_fairness|performance_metrics|explainability|clinical_utility|deployment_implementation`  
  OpPerf: `human_resources|improved_outcomes|improved_patient_outcomes`
- **[8/12]** [811] Precision dosing of voriconazole in immunocompromised children under 2  
  NIH: `study_design|bias_fairness|performance_metrics|explainability|external_validation|clinical_utility|safety_monitoring|regulatory_ethics`  
  OpPerf: `human_resources|improved_healthcare_outcomes|improved_patient_outcomes`
- **[8/12]** [812] Construction and validation of a machine learning-based model predicti  
  NIH: `study_design|data_reporting|bias_fairness|performance_metrics|explainability|external_validation|clinical_utility|deployment_implementation`  
  OpPerf: `human_resources|improved_healthcare_outcomes|improved_patient_outcomes`
- **[8/12]** [815] Development and validation of an interpretable multi-task model to pre  
  NIH: `study_design|data_reporting|performance_metrics|explainability|external_validation|uncertainty_quantification|clinical_utility|deployment_implementation`  
  OpPerf: `human_resources|cost|improved_patient_outcomes`
- **[8/12]** [822] Enhancing end-stage renal disease outcome prediction: a multisourced d  
  NIH: `study_design|data_reporting|bias_fairness|performance_metrics|explainability|clinical_utility|deployment_implementation|safety_monitoring`  
  OpPerf: `human_resources|improved_patient_outcomes`
- **[8/12]** [844] Forecasting Patient Early Readmission from Irish Hospital Discharge Re  
  NIH: `model_transparency|bias_fairness|performance_metrics|explainability|clinical_utility|deployment_implementation|safety_monitoring|regulatory_ethics`  
  OpPerf: `human_resources|cost|process_throughput|improved_healthcare_outcomes|improved_patient_outcomes`
- **[8/12]** [854] Machine Learning-Based Discrimination of Cardiovascular Outcomes in Pa  
  NIH: `study_design|data_reporting|performance_metrics|explainability|external_validation|clinical_utility|safety_monitoring|regulatory_ethics`  
  OpPerf: `human_resources|improved_patient_outcomes`
- **[8/12]** [867] Impact of System and Diagnostic Errors on Medical Litigation Outcomes:  
  NIH: `study_design|data_reporting|model_transparency|performance_metrics|explainability|clinical_utility|safety_monitoring|regulatory_ethics`  
  OpPerf: `human_resources|cost|improved_patient_outcomes`
- **[8/12]** [870] Machine Learning Models for Predicting Influential Factors of Early Ou  
  NIH: `study_design|bias_fairness|performance_metrics|explainability|clinical_utility|deployment_implementation|safety_monitoring|regulatory_ethics`  
  OpPerf: `human_resources|improved_process_performance|improved_patient_outcomes`
- **[7/12]** [17] Second opinion machine learning for fast-track pathway assignment in h  
  NIH: `data_reporting|bias_fairness|performance_metrics|explainability|clinical_utility|deployment_implementation|safety_monitoring`  
  OpPerf: `human_resources|improved_patient_outcomes`
- **[7/12]** [2377] Exploring the Complexity of Real-World Health Data Record Linkage-An E  
  NIH: `data_reporting|bias_fairness|performance_metrics|clinical_utility|deployment_implementation|safety_monitoring|regulatory_ethics`  
  OpPerf: `human_resources|improved_patient_outcomes`
- **[7/12]** [2388] Comparative Effectiveness of Machine Learning Approaches for Predictin  
  NIH: `study_design|data_reporting|performance_metrics|external_validation|clinical_utility|deployment_implementation|safety_monitoring`  
  OpPerf: `human_resources|cost|improved_outcomes|improved_patient_outcomes`
- **[7/12]** [2454] Explainable machine learning for early diagnosis of esophageal cancer:  
  NIH: `model_transparency|bias_fairness|performance_metrics|explainability|uncertainty_quantification|clinical_utility|deployment_implementation`  
  OpPerf: `human_resources|improved_patient_outcomes`
- **[7/12]** [2482] SHAP-based interpretable machine learning for Parkinson's disease seve  
  NIH: `bias_fairness|performance_metrics|explainability|external_validation|clinical_utility|deployment_implementation|regulatory_ethics`  
  OpPerf: `process_capacity|cost|process_throughput|improved_patient_outcomes`
- **[7/12]** [2534] Interpretable Machine Learning Model to Predict Bone Cement Leakage in  
  NIH: `study_design|data_reporting|performance_metrics|explainability|clinical_utility|deployment_implementation|regulatory_ethics`  
  OpPerf: `improved_patient_outcomes`
- **[7/12]** [26] An Explainable AI Approach for the Rapid Diagnosis of COVID-19 Using E  
  NIH: `study_design|performance_metrics|explainability|external_validation|uncertainty_quantification|clinical_utility|deployment_implementation`  
  OpPerf: `human_resources|improved_patient_outcomes`
- **[7/12]** [2814] Identifying key risk factors for intentional self-harm, including suic  
  NIH: `study_design|bias_fairness|performance_metrics|explainability|clinical_utility|deployment_implementation|safety_monitoring`  
  OpPerf: `human_resources|improved_patient_outcomes`
- **[7/12]** [2936] Utility of Candidate Genes From an Algorithm Designed to Predict Genet  
  NIH: `bias_fairness|performance_metrics|external_validation|uncertainty_quantification|clinical_utility|safety_monitoring|regulatory_ethics`  
  OpPerf: `improved_patient_outcomes`
- **[7/12]** [4285] Comparative clinical response, safety, and institutional drug use effi  
  NIH: `study_design|data_reporting|bias_fairness|performance_metrics|explainability|uncertainty_quantification|safety_monitoring`  
  OpPerf: `human_resources|cost|process_throughput|improved_healthcare_outcomes|improved_patient_outcomes`
- **[7/12]** [4555] Exploring Adverse Event Associations of Predicted PXR Agonists Using t  
  NIH: `bias_fairness|performance_metrics|explainability|external_validation|clinical_utility|safety_monitoring|regulatory_ethics`  
  OpPerf: `human_resources|improved_patient_outcomes`
- **[7/12]** [4687] Refining Drug-Induced Cholestasis Prediction: An Explainable Consensus  
  NIH: `data_reporting|model_transparency|performance_metrics|explainability|clinical_utility|safety_monitoring|regulatory_ethics`  
  OpPerf: `human_resources|cost|process_throughput|improved_patient_outcomes`
- **[7/12]** [5550] Detection of potential drug-drug interactions for risk of acute kidney  
  NIH: `data_reporting|bias_fairness|performance_metrics|explainability|uncertainty_quantification|safety_monitoring|regulatory_ethics`  
  OpPerf: `human_resources|cost|improved_patient_outcomes`
- **[7/12]** [7505] Cloud-Based Machine Learning Platform to Predict Clinical Outcomes at   
  NIH: `study_design|data_reporting|bias_fairness|performance_metrics|uncertainty_quantification|clinical_utility|safety_monitoring`  
  OpPerf: `human_resources|improved_healthcare_outcomes|improved_patient_outcomes`
- **[7/12]** [779] Development and validation of a machine learning predictive model for   
  NIH: `study_design|data_reporting|performance_metrics|explainability|external_validation|clinical_utility|safety_monitoring`  
  OpPerf: `human_resources|cost|improved_patient_outcomes`
- **[7/12]** [781] Interpretable Machine Learning with SHAP Identifies Key Biomarkers in   
  NIH: `bias_fairness|performance_metrics|explainability|uncertainty_quantification|clinical_utility|deployment_implementation|safety_monitoring`  
  OpPerf: `human_resources|improved_patient_outcomes`
- **[7/12]** [789] A Multicohort Machine Learning Framework to Predict Mortality in Elder  
  NIH: `study_design|data_reporting|bias_fairness|performance_metrics|explainability|external_validation|clinical_utility`  
  OpPerf: `human_resources|improved_patient_outcomes`
- **[7/12]** [794] Shifting Determinants of Mortality Risk After Orthotopic Heart Transpl  
  NIH: `study_design|bias_fairness|performance_metrics|explainability|clinical_utility|safety_monitoring|regulatory_ethics`  
  OpPerf: `human_resources|improved_outcomes|improved_patient_outcomes`
- **[7/12]** [796] Leveraging laboratory biomarkers to predict urosepsis after upper urin  
  NIH: `study_design|performance_metrics|explainability|external_validation|clinical_utility|deployment_implementation|regulatory_ethics`  
  OpPerf: `human_resources|cost|process_throughput|improved_patient_outcomes`
- **[7/12]** [797] Development and Statistical Validation of a Machine Learning Model for  
  NIH: `study_design|data_reporting|model_transparency|bias_fairness|performance_metrics|explainability|clinical_utility`  
  OpPerf: `human_resources|improved_healthcare_outcomes|improved_patient_outcomes`
- **[7/12]** [809] Explainable machine learning models for predicting of protein-energy w  
  NIH: `data_reporting|performance_metrics|explainability|clinical_utility|deployment_implementation|safety_monitoring|regulatory_ethics`  
  OpPerf: `human_resources|cost|improved_patient_outcomes`
- **[7/12]** [819] Development and validation of the machine learning model for acute exa  
  NIH: `study_design|bias_fairness|performance_metrics|explainability|external_validation|clinical_utility|deployment_implementation`  
  OpPerf: `human_resources|improved_healthcare_outcomes|improved_patient_outcomes`
- **[7/12]** [820] Systemic immune-inflammatory biomarkers combined with the CRP-albumin-  
  NIH: `study_design|data_reporting|model_transparency|performance_metrics|explainability|clinical_utility|deployment_implementation`  
  OpPerf: `human_resources|cost|improved_patient_outcomes`
- **[7/12]** [829] Machine learning-based prediction of diabetic peripheral neuropathy: m  
  NIH: `data_reporting|bias_fairness|performance_metrics|explainability|clinical_utility|deployment_implementation|regulatory_ethics`  
  OpPerf: `human_resources|cost|improved_patient_outcomes`
- **[7/12]** [830] Improving Hepatitis B outcome prediction with ensemble machine learnin  
  NIH: `data_reporting|model_transparency|bias_fairness|performance_metrics|explainability|clinical_utility|deployment_implementation`  
  OpPerf: `human_resources|process_throughput|improved_patient_outcomes`
- **[7/12]** [835] A multicenter study on developing a prognostic model for severe fever   
  NIH: `study_design|data_reporting|performance_metrics|explainability|external_validation|clinical_utility|deployment_implementation`  
  OpPerf: `human_resources|improved_patient_outcomes`
- **[7/12]** [837] Interpretable machine learning models for prolonged Emergency Departme  
  NIH: `study_design|bias_fairness|performance_metrics|explainability|clinical_utility|deployment_implementation|safety_monitoring`  
  OpPerf: `process_throughput|improved_process_performance|improved_patient_outcomes`
- **[7/12]** [841] Combining computed tomography features of left atrial epicardial and p  
  NIH: `study_design|data_reporting|bias_fairness|performance_metrics|explainability|uncertainty_quantification|clinical_utility`  
  OpPerf: `human_resources|process_throughput|improved_patient_outcomes`
- **[7/12]** [848] Prediction of sepsis mortality in ICU patients using machine learning   
  NIH: `study_design|data_reporting|model_transparency|performance_metrics|explainability|uncertainty_quantification|clinical_utility`  
  OpPerf: `human_resources|process_throughput|improved_patient_outcomes`
- **[7/12]** [865] Development of a machine learning model to predict lateral hinge fract  
  NIH: `study_design|data_reporting|performance_metrics|external_validation|uncertainty_quantification|clinical_utility|deployment_implementation`  
  OpPerf: `human_resources|improved_healthcare_outcomes|improved_patient_outcomes`
- **[6/12]** [1091] Exploring the potential and limitations of deep learning and explainab  
  NIH: `data_reporting|bias_fairness|performance_metrics|explainability|deployment_implementation|regulatory_ethics`  
  OpPerf: `human_resources|improved_healthcare_outcomes|improved_patient_outcomes`
- **[6/12]** [12] A Meta-Learning-Based Ensemble Model for Explainable Alzheimer's Disea  
  NIH: `model_transparency|performance_metrics|explainability|external_validation|clinical_utility|deployment_implementation`  
  OpPerf: `human_resources|improved_patient_outcomes`
- **[6/12]** [14] Investigating Protective and Risk Factors and Predictive Insights for   
  NIH: `bias_fairness|performance_metrics|explainability|deployment_implementation|safety_monitoring|regulatory_ethics`  
  OpPerf: `human_resources|improved_patient_outcomes`
- **[6/12]** [15] Explainable AI for Chronic Kidney Disease Prediction in Medical IoT: I  
  NIH: `bias_fairness|performance_metrics|explainability|clinical_utility|deployment_implementation|safety_monitoring`  
  OpPerf: `human_resources|process_throughput|improved_patient_outcomes`
- **[6/12]** [16] The AI-enhanced surgeon - integrating black-box artificial intelligenc  
  NIH: `study_design|performance_metrics|clinical_utility|deployment_implementation|safety_monitoring|regulatory_ethics`  
  OpPerf: `human_resources|process_throughput|improved_patient_outcomes`
- **[6/12]** [1817] Enhanced drug-drug interaction extraction from biomedical text using d  
  NIH: `study_design|performance_metrics|explainability|clinical_utility|deployment_implementation|safety_monitoring`  
  OpPerf: `process_throughput|improved_patient_outcomes`
- **[6/12]** [1823] MediNet: ensemble transfer learning approach for classification of med  
  NIH: `study_design|data_reporting|performance_metrics|explainability|external_validation|safety_monitoring`  
  OpPerf: `human_resources|process_throughput|improved_patient_outcomes`
- **[6/12]** [19] Machine learning models' assessment: trust and performance.  
  NIH: `bias_fairness|explainability|uncertainty_quantification|clinical_utility|deployment_implementation|regulatory_ethics`  
  OpPerf: `human_resources|improved_patient_outcomes`
- **[6/12]** [2128] IoMT Meets Machine Learning: From Edge to Cloud Chronic Diseases Diagn  
  NIH: `model_transparency|bias_fairness|performance_metrics|clinical_utility|deployment_implementation|safety_monitoring`  
  OpPerf: `human_resources|cost|process_throughput|improved_patient_outcomes`
- **[6/12]** [2717] Machine learning model for predicting hepatitis C seroconversion in me  
  NIH: `data_reporting|bias_fairness|performance_metrics|external_validation|clinical_utility|safety_monitoring`  
  OpPerf: `cost|improved_patient_outcomes`
- **[6/12]** [2874] A longitudinal observational study with ecological momentary assessmen  
  NIH: `study_design|bias_fairness|performance_metrics|explainability|clinical_utility|regulatory_ethics`  
  OpPerf: `human_resources|cost|improved_patient_outcomes`
- **[6/12]** [3329] The Skeletal Oncology Research Group Machine Learning Algorithm (SORG-  
  NIH: `data_reporting|bias_fairness|performance_metrics|external_validation|clinical_utility|safety_monitoring`  
  OpPerf: `human_resources|cost|process_throughput|improved_outcomes|improved_healthcare_outcomes|improved_patient_outcomes`
- **[6/12]** [3502] Using machine learning to study the effect of medication adherence in   
  NIH: `study_design|data_reporting|performance_metrics|clinical_utility|deployment_implementation|safety_monitoring`  
  OpPerf: `human_resources|cost|improved_patient_outcomes`
- **[6/12]** [4314] BEACON: predicting side effects and therapeutics outcomes to drugs by   
  NIH: `performance_metrics|explainability|clinical_utility|deployment_implementation|safety_monitoring|regulatory_ethics`  
  OpPerf: `human_resources|improved_patient_outcomes`
- **[6/12]** [4519] Drug-induced liver injury prediction based on graph convolutional netw  
  NIH: `data_reporting|performance_metrics|clinical_utility|deployment_implementation|safety_monitoring|regulatory_ethics`  
  OpPerf: `human_resources|process_throughput|improved_patient_outcomes`
- **[6/12]** [4935] Discovering Severe Adverse Reactions From Pharmacokinetic Drug-Drug In  
  NIH: `study_design|bias_fairness|performance_metrics|clinical_utility|safety_monitoring|regulatory_ethics`  
  OpPerf: `human_resources|improved_patient_outcomes`
- **[6/12]** [5] Explainable Artificial Intelligence for Ovarian Cancer: Biomarker Cont  
  NIH: `bias_fairness|performance_metrics|explainability|external_validation|clinical_utility|deployment_implementation`  
  OpPerf: `human_resources|improved_patient_outcomes`
- **[6/12]** [5066] Machine Learning Prediction of On/Off Target-driven Clinical Adverse E  
  NIH: `study_design|data_reporting|clinical_utility|deployment_implementation|safety_monitoring|regulatory_ethics`  
  OpPerf: `human_resources|improved_patient_outcomes`
- **[6/12]** [5129] MPHGCL-DDI: Meta-Path-Based Heterogeneous Graph Contrastive Learning f  
  NIH: `study_design|data_reporting|explainability|clinical_utility|deployment_implementation|safety_monitoring`  
  OpPerf: `human_resources|improved_patient_outcomes`
- **[6/12]** [5474] MSDRP: a deep learning model based on multisource data for predicting   
  NIH: `data_reporting|model_transparency|performance_metrics|explainability|deployment_implementation|regulatory_ethics`  
  OpPerf: `process_throughput|improved_patient_outcomes`
- **[6/12]** [5545] Using machine learning to develop a clinical prediction model for SSRI  
  NIH: `data_reporting|bias_fairness|performance_metrics|clinical_utility|deployment_implementation|safety_monitoring`  
  OpPerf: `human_resources|cost|improved_healthcare_outcomes|improved_patient_outcomes`
- **[6/12]** [5764] Sex Differences in Clopidogrel Effects Among Young Patients With Acute  
  NIH: `data_reporting|bias_fairness|uncertainty_quantification|clinical_utility|safety_monitoring|regulatory_ethics`  
  OpPerf: `human_resources|improved_healthcare_outcomes|improved_patient_outcomes`
- **[6/12]** [6020] Intestinal microbiota signatures of clinical response and immune-relat  
  NIH: `bias_fairness|performance_metrics|external_validation|uncertainty_quantification|clinical_utility|safety_monitoring`  
  OpPerf: `human_resources|cost|process_throughput|improved_outcomes|improved_patient_outcomes`
- **[6/12]** [6489] Meloxicam methyl group determines enzyme specificity for thiazole bioa  
  NIH: `study_design|performance_metrics|uncertainty_quantification|clinical_utility|deployment_implementation|regulatory_ethics`  
  OpPerf: `cost|process_throughput|improved_patient_outcomes`
- **[6/12]** [738] Exploring the impact of design criteria for reference sets on performa  
  NIH: `study_design|data_reporting|bias_fairness|performance_metrics|safety_monitoring|regulatory_ethics`  
  OpPerf: `human_resources|improved_patient_outcomes`
- **[6/12]** [777] FibreCastML: an open web platform for predicting electrospun nanofibre  
  NIH: `model_transparency|bias_fairness|performance_metrics|explainability|clinical_utility|deployment_implementation`  
  OpPerf: `human_resources|process_throughput|improved_patient_outcomes`
- **[6/12]** [780] Web-based cardiovascular disease risk prediction using machine learnin  
  NIH: `model_transparency|bias_fairness|performance_metrics|explainability|external_validation|safety_monitoring`  
  OpPerf: `human_resources|improved_outcomes|improved_patient_outcomes`
- **[6/12]** [784] Research on children's health prediction based on Improved Grey Wolf O  
  NIH: `model_transparency|bias_fairness|performance_metrics|explainability|external_validation|safety_monitoring`  
  OpPerf: `human_resources|improved_patient_outcomes`
- **[6/12]** [793] An individualized risk prediction tool for ectopic pregnancy within th  
  NIH: `study_design|performance_metrics|explainability|clinical_utility|safety_monitoring|regulatory_ethics`  
  OpPerf: `improved_patient_outcomes`
- **[6/12]** [803] Predicting stillbirth and identifying key maternal risk factors using   
  NIH: `study_design|bias_fairness|performance_metrics|explainability|clinical_utility|safety_monitoring`  
  OpPerf: `human_resources|improved_healthcare_outcomes|improved_patient_outcomes`
- **[6/12]** [816] Personalized health monitoring using explainable AI: bridging trust in  
  NIH: `performance_metrics|explainability|external_validation|clinical_utility|deployment_implementation|safety_monitoring`  
  OpPerf: `human_resources|improved_patient_outcomes`
- **[6/12]** [821] XGBoost models based on non imaging features for the prediction of mil  
  NIH: `data_reporting|bias_fairness|performance_metrics|explainability|clinical_utility|deployment_implementation`  
  OpPerf: `improved_patient_outcomes`
- **[6/12]** [824] A Simplified Machine Learning Model for Predicting Reduced Kidney Func  
  NIH: `study_design|data_reporting|performance_metrics|explainability|clinical_utility|deployment_implementation`  
  OpPerf: `improved_patient_outcomes`
- **[6/12]** [833] Development of explainable artificial intelligence based machine learn  
  NIH: `study_design|performance_metrics|explainability|external_validation|deployment_implementation|regulatory_ethics`  
  OpPerf: `human_resources|improved_patient_outcomes`
- **[6/12]** [8490] Implementation of an Opioid Use Disorder (OUD) Machine-Learning Phenot  
  NIH: `study_design|model_transparency|bias_fairness|performance_metrics|external_validation|deployment_implementation`  
  OpPerf: `human_resources|improved_patient_outcomes`
- **[6/12]** [9248] Predictive model and risk analysis for peripheral vascular disease in   
  NIH: `study_design|data_reporting|model_transparency|bias_fairness|performance_metrics|explainability`  
  OpPerf: `human_resources|cost|improved_patient_outcomes`
- **[5/12]** [1] Explainable AI for mortality prediction: a comparative study using the  
  NIH: `study_design|bias_fairness|performance_metrics|explainability|clinical_utility`  
  OpPerf: `human_resources|improved_outcomes|improved_patient_outcomes`
- **[5/12]** [1777] Leveraging Large Language Models for Adverse Drug Event Detection: A C  
  NIH: `study_design|data_reporting|clinical_utility|safety_monitoring|regulatory_ethics`  
  OpPerf: `human_resources|cost|improved_patient_outcomes`
- **[5/12]** [1820] Global Adoption, Promotion, Impact, and Deployment of AI in Patient Ca  
  NIH: `performance_metrics|clinical_utility|deployment_implementation|safety_monitoring|regulatory_ethics`  
  OpPerf: `human_resources|process_throughput|improved_patient_outcomes`
- **[5/12]** [1869] Prediction of Metastasis in Paragangliomas and Pheochromocytomas Using  
  NIH: `model_transparency|performance_metrics|explainability|clinical_utility|deployment_implementation`  
  OpPerf: `human_resources|improved_patient_outcomes`
- **[5/12]** [2378] Development and validation of prediction models for stroke and myocard  
  NIH: `data_reporting|performance_metrics|external_validation|deployment_implementation|safety_monitoring`  
  OpPerf: `cost|improved_patient_outcomes`
- **[5/12]** [2382] The application of machine learning to predict high-cost patients: A p  
  NIH: `data_reporting|bias_fairness|performance_metrics|uncertainty_quantification|safety_monitoring`  
  OpPerf: `human_resources|cost|improved_healthcare_outcomes|improved_patient_outcomes`
- **[5/12]** [2499] Interpretable machine learning model for predicting anastomotic leak a  
  NIH: `study_design|data_reporting|performance_metrics|explainability|external_validation`  
  OpPerf: `human_resources|cost|improved_patient_outcomes`
- **[5/12]** [2644] Augmenting large language models to predict social determinants of men  
  NIH: `model_transparency|bias_fairness|performance_metrics|clinical_utility|regulatory_ethics`  
  OpPerf: `human_resources|improved_patient_outcomes`
- **[5/12]** [4] Predicting and classifying type 2 diabetes using a transparent ensembl  
  NIH: `model_transparency|performance_metrics|explainability|uncertainty_quantification|clinical_utility`  
  OpPerf: `human_resources|improved_patient_outcomes`
- **[5/12]** [4256] Bradycardia Shock Caused by the Combined Use of Carteolol Eye Drops an  
  NIH: `performance_metrics|explainability|clinical_utility|safety_monitoring|regulatory_ethics`  
  OpPerf: `human_resources|cost|improved_patient_outcomes`
- **[5/12]** [4403] A meta-contrastive learning approach for clinical drug-drug interactio  
  NIH: `data_reporting|model_transparency|performance_metrics|deployment_implementation|safety_monitoring`  
  OpPerf: `human_resources|improved_healthcare_outcomes|improved_patient_outcomes`
- **[5/12]** [4502] A Dose-Aware Model for Revealing Dose-Risk Relationship of Drug-Drug I  
  NIH: `data_reporting|bias_fairness|uncertainty_quantification|safety_monitoring|regulatory_ethics`  
  OpPerf: `human_resources|cost|process_throughput|improved_patient_outcomes`
- **[5/12]** [4528] Leveraging Large Language Models in Extracting Drug Safety Information  
  NIH: `data_reporting|external_validation|clinical_utility|safety_monitoring|regulatory_ethics`  
  OpPerf: `cost|process_throughput|improved_patient_outcomes`
- **[5/12]** [4721] Predicting rare drug-drug interaction events with dual-granular struct  
  NIH: `explainability|external_validation|deployment_implementation|safety_monitoring|regulatory_ethics`  
  OpPerf: `human_resources|improved_healthcare_outcomes|improved_patient_outcomes`
- **[5/12]** [4834] Exploiting question-answer framework with multi-GRU to detect adverse   
  NIH: `study_design|data_reporting|performance_metrics|clinical_utility|safety_monitoring`  
  OpPerf: `human_resources|improved_patient_outcomes`
- **[5/12]** [4921] Precision Adverse Drug Reactions Prediction with Heterogeneous Graph N  
  NIH: `bias_fairness|performance_metrics|clinical_utility|safety_monitoring|regulatory_ethics`  
  OpPerf: `human_resources|cost|improved_healthcare_outcomes|improved_process_performance|improved_patient_outcomes`
- **[5/12]** [4989] MSDAFL: molecular substructure-based dual attention feature learning f  
  NIH: `model_transparency|bias_fairness|performance_metrics|deployment_implementation|safety_monitoring`  
  OpPerf: `human_resources|improved_patient_outcomes`
- **[5/12]** [5186] Preclinical side effect prediction through pathway engineering of prot  
  NIH: `study_design|performance_metrics|deployment_implementation|safety_monitoring|regulatory_ethics`  
  OpPerf: `improved_patient_outcomes`
- **[5/12]** [5202] Familiarity with ChatGPT Features Modifies Expectations and Learning O  
  NIH: `study_design|bias_fairness|performance_metrics|safety_monitoring|regulatory_ethics`  
  OpPerf: `human_resources|improved_patient_outcomes`
- **[5/12]** [5590] Development and Validation of an Explainable Machine Learning-Based Pr  
  NIH: `data_reporting|explainability|external_validation|clinical_utility|safety_monitoring`  
  OpPerf: `process_throughput|improved_patient_outcomes`
- **[5/12]** [5901] SPARSE: a sparse hypergraph neural network for learning multiple types  
  NIH: `model_transparency|explainability|clinical_utility|deployment_implementation|safety_monitoring`  
  OpPerf: `improved_patient_outcomes`
- **[5/12]** [5925] Detecting Drug-Drug Interactions in COVID-19 Patients.  
  NIH: `study_design|bias_fairness|clinical_utility|safety_monitoring|regulatory_ethics`  
  OpPerf: `improved_patient_outcomes`
- **[5/12]** [5950] Comparative study of the adverse event profile of hydroxychloroquine b  
  NIH: `study_design|model_transparency|deployment_implementation|safety_monitoring|regulatory_ethics`  
  OpPerf: `improved_patient_outcomes`
- **[5/12]** [5952] Prediction of Drug-Drug Interaction Using an Attention-Based Graph Neu  
  NIH: `explainability|clinical_utility|deployment_implementation|safety_monitoring|regulatory_ethics`  
  OpPerf: `human_resources|process_throughput|improved_patient_outcomes`
- **[5/12]** [6288] Algebraic graph-assisted bidirectional transformers for molecular prop  
  NIH: `performance_metrics|explainability|clinical_utility|safety_monitoring|regulatory_ethics`  
  OpPerf: `human_resources|improved_patient_outcomes`
- **[5/12]** [7231] Implications of Big Data Analytics, AI, Machine Learning, and Deep Lea  
  NIH: `data_reporting|explainability|deployment_implementation|safety_monitoring|regulatory_ethics`  
  OpPerf: `improved_healthcare_outcomes|improved_patient_outcomes`
- **[5/12]** [782] Machine Learning-Based Prediction of Institutional Delivery Dropout (I  
  NIH: `bias_fairness|performance_metrics|explainability|uncertainty_quantification|clinical_utility`  
  OpPerf: `human_resources|improved_healthcare_outcomes|improved_patient_outcomes`
- **[5/12]** [792] Artificial intelligence-driven diagnosis of autism spectrum disorder i  
  NIH: `data_reporting|model_transparency|bias_fairness|performance_metrics|explainability`  
  OpPerf: `human_resources|cost|improved_patient_outcomes`
- **[5/12]** [800] Bridging the gap: explainable ai for autism diagnosis and parental sup  
  NIH: `data_reporting|model_transparency|performance_metrics|explainability|clinical_utility`  
  OpPerf: `human_resources|improved_patient_outcomes`
- **[5/12]** [802] The impact of feature combinations on machine learning models for in-h  
  NIH: `study_design|data_reporting|performance_metrics|explainability|deployment_implementation`  
  OpPerf: `human_resources|improved_patient_outcomes`
- **[5/12]** [8078] E-CatBoost: An efficient machine learning framework for predicting ICU  
  NIH: `performance_metrics|explainability|clinical_utility|deployment_implementation|safety_monitoring`  
  OpPerf: `human_resources|cost|improved_healthcare_outcomes|improved_patient_outcomes`
- **[5/12]** [818] Machine learning-based nomogram for predicting depressive symptoms in   
  NIH: `bias_fairness|performance_metrics|explainability|clinical_utility|deployment_implementation`  
  OpPerf: `human_resources|improved_healthcare_outcomes|improved_patient_outcomes`
- **[5/12]** [826] Enhancing diabetes risk prediction through focal active learning and m  
  NIH: `bias_fairness|performance_metrics|explainability|safety_monitoring|regulatory_ethics`  
  OpPerf: `human_resources|improved_outcomes|improved_patient_outcomes`
- **[5/12]** [832] Predicting Organ Rejections for Pediatric Heart Transplantations with   
  NIH: `bias_fairness|performance_metrics|explainability|external_validation|clinical_utility`  
  OpPerf: `human_resources|improved_healthcare_outcomes|improved_patient_outcomes`
- **[5/12]** [834] Explainable AI for enhanced accuracy in malaria diagnosis using ensemb  
  NIH: `data_reporting|performance_metrics|explainability|uncertainty_quantification|clinical_utility`  
  OpPerf: `human_resources|cost|improved_patient_outcomes`
- **[5/12]** [838] AI Machine Learning-Based Diabetes Prediction in Older Adults in South  
  NIH: `bias_fairness|performance_metrics|explainability|clinical_utility|deployment_implementation`  
  OpPerf: `human_resources|cost|improved_patient_outcomes`
- **[5/12]** [8493] Development and evaluation of a machine learning model to predict acut  
  NIH: `study_design|bias_fairness|performance_metrics|uncertainty_quantification|safety_monitoring`  
  OpPerf: `human_resources|cost|improved_healthcare_outcomes|improved_patient_outcomes`
- **[5/12]** [850] Machine Learning Prediction of Treatment Response to Biological Diseas  
  NIH: `model_transparency|performance_metrics|explainability|safety_monitoring|regulatory_ethics`  
  OpPerf: `human_resources|cost|improved_patient_outcomes`
- **[5/12]** [8577] All Roads Lead to Rome: Diverse Etiologies of Tricuspid Regurgitation   
  NIH: `model_transparency|bias_fairness|performance_metrics|explainability|clinical_utility`  
  OpPerf: `improved_patient_outcomes`
- **[5/12]** [861] Using machine learning to identify patient characteristics to predict   
  NIH: `study_design|bias_fairness|explainability|clinical_utility|safety_monitoring`  
  OpPerf: `cost|improved_healthcare_outcomes|improved_patient_outcomes`
- **[5/12]** [862] Explainable Machine Learning to Predict Successful Weaning of Mechanic  
  NIH: `bias_fairness|explainability|external_validation|clinical_utility|regulatory_ethics`  
  OpPerf: `human_resources|improved_patient_outcomes`
- **[5/12]** [8651] Deep hybrid model for maternal health risk classification in pregnancy  
  NIH: `study_design|bias_fairness|performance_metrics|external_validation|deployment_implementation`  
  OpPerf: `human_resources|process_throughput|improved_healthcare_outcomes|improved_patient_outcomes`
- **[5/12]** [8659] Dependency Factors in Evidence Theory: An Analysis in an Information F  
  NIH: `data_reporting|performance_metrics|uncertainty_quantification|deployment_implementation|regulatory_ethics`  
  OpPerf: `human_resources|improved_patient_outcomes`
- **[5/12]** [871] Using Explainable Machine Learning to Improve Intensive Care Unit Alar  
  NIH: `bias_fairness|performance_metrics|explainability|deployment_implementation|safety_monitoring`  
  OpPerf: `human_resources|improved_patient_outcomes`
- **[5/12]** [8740] Identification of patients' smoking status using an explainable AI app  
  NIH: `model_transparency|performance_metrics|explainability|clinical_utility|deployment_implementation`  
  OpPerf: `human_resources|improved_patient_outcomes`
- **[5/12]** [8915] The Price of Explainability in Machine Learning Models for 100-Day Rea  
  NIH: `study_design|performance_metrics|explainability|clinical_utility|deployment_implementation`  
  OpPerf: `human_resources|improved_healthcare_outcomes|improved_patient_outcomes`
- **[5/12]** [9] Fostering trust and interpretability: integrating explainable AI (XAI)  
  NIH: `bias_fairness|performance_metrics|explainability|clinical_utility|deployment_implementation`  
  OpPerf: `process_capacity|human_resources|process_throughput|improved_patient_outcomes`
- **[4/12]** [1811] Outcome centred process mapping in healthcare using random forest and   
  NIH: `bias_fairness|explainability|clinical_utility|deployment_implementation`  
  OpPerf: `human_resources|process_throughput|improved_process_performance|improved_patient_outcomes`
- **[4/12]** [1885] Sarcopenia prediction model based on machine learning and SHAP values   
  NIH: `data_reporting|performance_metrics|explainability|clinical_utility`  
  OpPerf: `human_resources|improved_patient_outcomes`
- **[4/12]** [1946] Compact Quantum Cascade Laser-Based Noninvasive Glucose Sensor Upgrade  
  NIH: `performance_metrics|clinical_utility|safety_monitoring|regulatory_ethics`  
  OpPerf: `human_resources|improved_patient_outcomes`
- **[4/12]** [2018] Automatic text classification of drug-induced liver injury using docum  
  NIH: `performance_metrics|explainability|safety_monitoring|regulatory_ethics`  
  OpPerf: `improved_patient_outcomes`
- **[4/12]** [23] A historical perspective of biomedical explainable AI research.  
  NIH: `bias_fairness|explainability|deployment_implementation|safety_monitoring`  
  OpPerf: `human_resources|process_throughput|improved_patient_outcomes`
- **[4/12]** [2384] A machine learning model on Real World Data for predicting progression  
  NIH: `study_design|data_reporting|performance_metrics|safety_monitoring`  
  OpPerf: `human_resources|improved_healthcare_outcomes|improved_patient_outcomes`
- **[4/12]** [2389] Integrating human services and criminal justice data with claims data   
  NIH: `data_reporting|performance_metrics|explainability|deployment_implementation`  
  OpPerf: `human_resources|improved_patient_outcomes`
- **[4/12]** [25] Explaining the black-box smoothly-A counterfactual approach.  
  NIH: `bias_fairness|explainability|deployment_implementation|safety_monitoring`  
  OpPerf: `improved_patient_outcomes`
- **[4/12]** [2548] Clinical application of the "sellar barrier's concept" for predicting   
  NIH: `study_design|data_reporting|performance_metrics|clinical_utility`  
  OpPerf: `human_resources|improved_patient_outcomes`
- **[4/12]** [2633] Using explainable machine learning to elucidate social and neurobehavi  
  NIH: `data_reporting|model_transparency|explainability|deployment_implementation`  
  OpPerf: `improved_patient_outcomes`
- **[4/12]** [3927] Developing a cognitive dysfunction risk score for use with opioid-depe  
  NIH: `study_design|bias_fairness|performance_metrics|deployment_implementation`  
  OpPerf: `human_resources|improved_patient_outcomes`
- **[4/12]** [4260] Prediction of the Drug-Drug Interaction Types with the Unified Embeddi  
  NIH: `model_transparency|performance_metrics|deployment_implementation|safety_monitoring`  
  OpPerf: `human_resources|improved_patient_outcomes`
- **[4/12]** [4294] Knowledge-graph-enhanced multi-scale modeling for drug-drug interactio  
  NIH: `study_design|performance_metrics|deployment_implementation|safety_monitoring`  
  OpPerf: `human_resources|improved_healthcare_outcomes|improved_patient_outcomes`
- **[4/12]** [4662] Predicting DDI-induced pregnancy and neonatal ADRs using sparse PCA an  
  NIH: `data_reporting|performance_metrics|safety_monitoring|regulatory_ethics`  
  OpPerf: `human_resources|improved_patient_outcomes`
- **[4/12]** [4925] [Drug-drug interactions in critically ill patients].  
  NIH: `performance_metrics|clinical_utility|deployment_implementation|safety_monitoring`  
  OpPerf: `improved_patient_outcomes`
- **[4/12]** [4956] DRESS syndrome: an interaction between drugs, latent viruses, and the   
  NIH: `bias_fairness|performance_metrics|clinical_utility|safety_monitoring`  
  OpPerf: `human_resources|cost|improved_healthcare_outcomes|improved_patient_outcomes`
- **[4/12]** [5305] Learning self-supervised molecular representations for drug-drug inter  
  NIH: `performance_metrics|explainability|clinical_utility|safety_monitoring`  
  OpPerf: `human_resources|improved_patient_outcomes`
- **[4/12]** [5341] Drug-drug interaction prediction: databases, web servers and computati  
  NIH: `model_transparency|bias_fairness|deployment_implementation|safety_monitoring`  
  OpPerf: `human_resources|cost|improved_patient_outcomes`
- **[4/12]** [5427] A blinded in vitro analysis of the intrinsic immunogenicity of hepatot  
  NIH: `performance_metrics|clinical_utility|safety_monitoring|regulatory_ethics`  
  OpPerf: `improved_patient_outcomes`
- **[4/12]** [5670] AIMedGraph: a comprehensive multi-relational knowledge graph for preci  
  NIH: `study_design|performance_metrics|clinical_utility|safety_monitoring`  
  OpPerf: `process_throughput|improved_patient_outcomes`
- **[4/12]** [5799] Individualising intensive systolic blood pressure reduction in hyperte  
  NIH: `study_design|bias_fairness|performance_metrics|clinical_utility`  
  OpPerf: `process_throughput|improved_patient_outcomes`
- **[4/12]** [5864] Functional and structural characteristics of HLA-B*13:01-mediated spec  
  NIH: `performance_metrics|uncertainty_quantification|clinical_utility|safety_monitoring`  
  OpPerf: `human_resources|improved_patient_outcomes`
- **[4/12]** [5887] Language-agnostic pharmacovigilant text mining to elicit side effects   
  NIH: `data_reporting|bias_fairness|safety_monitoring|regulatory_ethics`  
  OpPerf: `human_resources|improved_patient_outcomes`
- **[4/12]** [6012] A Data-Driven Medical Decision Framework for Associating Adverse Drug   
  NIH: `model_transparency|clinical_utility|deployment_implementation|safety_monitoring`  
  OpPerf: `human_resources|improved_patient_outcomes`
- **[4/12]** [6155] Machine Learning to Identify Interaction of Single-Nucleotide Polymorp  
  NIH: `data_reporting|deployment_implementation|safety_monitoring|regulatory_ethics`  
  OpPerf: `human_resources|improved_patient_outcomes`
- **[4/12]** [6207] AttentionDDI: Siamese attention-based deep learning method for drug-dr  
  NIH: `model_transparency|performance_metrics|explainability|safety_monitoring`  
  OpPerf: `human_resources|improved_patient_outcomes`
- **[4/12]** [6237] A neural network-based method for polypharmacy side effects prediction  
  NIH: `study_design|model_transparency|performance_metrics|safety_monitoring`  
  OpPerf: `improved_patient_outcomes`
- **[4/12]** [6287] Novel deep learning-based transcriptome data analysis for drug-drug in  
  NIH: `performance_metrics|clinical_utility|deployment_implementation|safety_monitoring`  
  OpPerf: `human_resources|improved_patient_outcomes`
- **[4/12]** [6984] Increasing the ethnic diversity of senior leadership within the Englis  
  NIH: `bias_fairness|performance_metrics|clinical_utility|deployment_implementation`  
  OpPerf: `human_resources|process_throughput|improved_patient_outcomes`
- **[4/12]** [791] Understanding COVID-19 vaccine hesitancy among older adults in post-ze  
  NIH: `bias_fairness|performance_metrics|explainability|clinical_utility`  
  OpPerf: `human_resources|improved_patient_outcomes`
- **[4/12]** [799] Evaluating XAI techniques under class imbalance using CPRD data.  
  NIH: `data_reporting|explainability|deployment_implementation|regulatory_ethics`  
  OpPerf: `human_resources|improved_patient_outcomes`
- **[4/12]** [810] Explainable SHAP-XGBoost models for identifying important social facto  
  NIH: `bias_fairness|performance_metrics|explainability|safety_monitoring`  
  OpPerf: `human_resources|cost|improved_patient_outcomes`
- **[4/12]** [814] Decoding the association between health level and human settlements en  
  NIH: `model_transparency|bias_fairness|explainability|clinical_utility`  
  OpPerf: `human_resources|improved_patient_outcomes`
- **[4/12]** [827] AI-driven analysis by identifying risk factors of VL relapse in HIV co  
  NIH: `bias_fairness|explainability|clinical_utility|safety_monitoring`  
  OpPerf: `human_resources|improved_patient_outcomes`
- **[4/12]** [840] Optimizing hypertension prediction using ensemble learning approaches.  
  NIH: `bias_fairness|performance_metrics|explainability|clinical_utility`  
  OpPerf: `cost|process_throughput|improved_patient_outcomes`
- **[4/12]** [842] Predictive and Interpretable Machine Learning of Economic Burden: The   
  NIH: `study_design|bias_fairness|explainability|safety_monitoring`  
  OpPerf: `human_resources|cost|improved_patient_outcomes`
- **[4/12]** [843] Construction and SHAP interpretability analysis of a risk prediction m  
  NIH: `study_design|performance_metrics|explainability|deployment_implementation`  
  OpPerf: `human_resources|cost|improved_healthcare_outcomes|improved_patient_outcomes`
- **[4/12]** [846] Construction of a machine learning-based prediction model for unfavora  
  NIH: `bias_fairness|performance_metrics|explainability|clinical_utility`  
  OpPerf: `improved_patient_outcomes`
- **[4/12]** [8483] A machine learning approach for opioid overdose risk prediction among   
  NIH: `data_reporting|bias_fairness|performance_metrics|safety_monitoring`  
  OpPerf: `improved_patient_outcomes`
- **[4/12]** [849] Machine learning-enabled prediction of prolonged length of stay in hos  
  NIH: `performance_metrics|explainability|clinical_utility|deployment_implementation`  
  OpPerf: `human_resources|cost|process_throughput|improved_healthcare_outcomes|improved_patient_outcomes`
- **[4/12]** [852] Explainable Artificial Intelligence in Quantifying Breast Cancer Facto  
  NIH: `bias_fairness|performance_metrics|explainability|clinical_utility`  
  OpPerf: `human_resources|cost|improved_patient_outcomes`
- **[4/12]** [856] Machine learning approaches to enhance diagnosis and staging of patien  
  NIH: `data_reporting|performance_metrics|explainability|external_validation`  
  OpPerf: `improved_patient_outcomes`
- **[4/12]** [863] Gray matter volume drives the brain age gap in schizophrenia: a SHAP s  
  NIH: `study_design|data_reporting|model_transparency|explainability`  
  OpPerf: `human_resources|cost|improved_patient_outcomes`
- **[4/12]** [8645] Creation and Validation of an Algorithm for Predicting the Recurrence   
  NIH: `study_design|data_reporting|performance_metrics|safety_monitoring`  
  OpPerf: `human_resources|improved_patient_outcomes`
- **[4/12]** [866] Prediction of conversion to dementia using interpretable machine learn  
  NIH: `study_design|bias_fairness|performance_metrics|explainability`  
  OpPerf: `human_resources|cost|improved_patient_outcomes`
- **[4/12]** [8669] Development and validation of a random forest model for predicting rad  
  NIH: `study_design|data_reporting|performance_metrics|regulatory_ethics`  
  OpPerf: `improved_patient_outcomes`
- **[4/12]** [869] Comparative analysis of explainable machine learning prediction models  
  NIH: `model_transparency|performance_metrics|explainability|deployment_implementation`  
  OpPerf: `human_resources|improved_patient_outcomes`
- **[4/12]** [8714] Feature Genes in Neuroblastoma Distinguishing High-Risk and Non-high-R  
  NIH: `data_reporting|performance_metrics|explainability|uncertainty_quantification`  
  OpPerf: `process_throughput|improved_patient_outcomes`
- **[4/12]** [8826] Non-contact screening system based for COVID-19 on XGBoost and logisti  
  NIH: `performance_metrics|explainability|external_validation|safety_monitoring`  
  OpPerf: `human_resources|improved_patient_outcomes`
- **[3/12]** [13] Synergizing advanced algorithm of explainable artificial intelligence   
  NIH: `performance_metrics|explainability|deployment_implementation`  
  OpPerf: `human_resources|improved_outcomes|improved_patient_outcomes`
- **[3/12]** [1798] Enhancing Public Healthcare Through VADER Sentiment Analysis: A Case S  
  NIH: `performance_metrics|clinical_utility|deployment_implementation`  
  OpPerf: `human_resources|process_throughput|improved_outcomes|improved_process_performance|improved_patient_outcomes`
- **[3/12]** [20] A machine learning framework for interpretable predictions in patient   
  NIH: `performance_metrics|explainability|clinical_utility`  
  OpPerf: `process_throughput|improved_healthcare_outcomes|improved_patient_outcomes`
- **[3/12]** [2093] Helicopter parenting through the lens of reddit: A text mining study.  
  NIH: `bias_fairness|clinical_utility|regulatory_ethics`  
  OpPerf: `human_resources|improved_patient_outcomes`
- **[3/12]** [2169] A deep learning method to detect opioid prescription and opioid use di  
  NIH: `study_design|bias_fairness|performance_metrics`  
  OpPerf: `human_resources|cost|improved_patient_outcomes`
- **[3/12]** [2542] Demographic characteristics, clinical symptoms, biochemical markers an  
  NIH: `bias_fairness|external_validation|uncertainty_quantification`  
  OpPerf: `cost|improved_healthcare_outcomes|improved_patient_outcomes`
- **[3/12]** [3781] Race, trust, and COVID-19 vaccine hesitancy in people with opioid use   
  NIH: `bias_fairness|uncertainty_quantification|clinical_utility`  
  OpPerf: `improved_healthcare_outcomes|improved_patient_outcomes`
- **[3/12]** [4306] Drug-induced gastrointestinal toxicity and barrier integrity: cytoskel  
  NIH: `study_design|bias_fairness|performance_metrics`  
  OpPerf: `human_resources|cost|improved_patient_outcomes`
- **[3/12]** [4700] Can large language models detect drug-drug interactions leading to adv  
  NIH: `performance_metrics|safety_monitoring|regulatory_ethics`  
  OpPerf: `human_resources|improved_patient_outcomes`
- **[3/12]** [5136] Accurate prediction of drug combination risk levels based on relationa  
  NIH: `clinical_utility|deployment_implementation|safety_monitoring`  
  OpPerf: `human_resources|process_throughput|improved_healthcare_outcomes|improved_patient_outcomes`
- **[3/12]** [5517] The allopurinol metabolite, oxypurinol, drives oligoclonal expansions   
  NIH: `performance_metrics|clinical_utility|safety_monitoring`  
  OpPerf: `cost|improved_patient_outcomes`
- **[3/12]** [5552] Transgenic murine models for the study of drug hypersensitivity reacti  
  NIH: `performance_metrics|clinical_utility|regulatory_ethics`  
  OpPerf: `improved_patient_outcomes`

… and 165 more — see `nih_ai_checklist_tags.csv` for full list.