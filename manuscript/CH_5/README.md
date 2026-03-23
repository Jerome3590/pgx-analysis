# Chapter 5: Translation – The PGx Risk Dashboard

## Overview

This repository contains the implementation and documentation for **Chapter 5** of a PhD in Translational Informatics. This chapter serves as the "Implementation Paper," demonstrating how theoretical models and causal findings from Chapters 3 and 4 are operationalized into a production-ready, serverless decision support tool for pharmacogenomic (PGx) risk assessment.

**Key Focus Areas:**
- Addressing the "Last Mile" problem in healthcare AI
- Privacy-first architecture for PII-sensitive data
- Handling real-world data sparsity (partial patient inputs)
- Translating ML models into actionable bedside tools

---

## Chapter Structure & Manuscript Roadmap

### 5.1 Introduction
**Objective:** Frame the translational challenge and chapter contributions.

**Key Points:**
- **The "Last Mile" Problem:** Discuss the challenge of translating high-performance ML models into accessible tools for point-of-care decision-making
- **Chapter Objective:** Develop a privacy-first, serverless application that democratizes access to ensemble risk models (Opioid/Polypharmacy) and PGx guidelines
- **Contribution to Translational Informatics:** Present a scalable architecture that handles "Partial Inputs" (real-world data sparsity) and provides "Privacy-First" PGx insights without storing PII

**Manuscript Checklist:**
- [ ] Define the clinical need and pain point
- [ ] Position the dashboard as a solution
- [ ] State reproducibility and open-source goals

---

### 5.2 System Architecture: Serverless & Scalable

**Design Philosophy:** Hybrid deployment combining static frontend with containerized serverless backend.

#### 5.2.1 Hybrid Deployment Model
| Component | Technology | Rationale |
|-----------|-----------|-----------|
| **Frontend** | S3-hosted static HTML/JS | Low latency, high availability, no backend dependencies |
| **Backend** | AWS Lambda (Docker containers, ECR) | Elastic scaling, cost-effective, bundled models (no S3 latency) |
| **Model Storage** | Lambda Container Image (~1.5GB) | All 21 trained models packaged directly; eliminates runtime S3 download overhead |

**Technical Justification:**
- Lambda Container support allows up to 10GB images
- 7 age bands × 3 models = 21 total models
- Bundling enables sub-100ms inference latency

#### 5.2.2 The "Partition-First" Inheritance
**Routing Logic:** Requests automatically routed based on **Patient Age**

```
Input Age → Select Cohort (Opioid vs. Non-Opioid) 
          → Select Age Band Model (e.g., 65-74)
          → Execute Ensemble Inference
          → Return Risk Score & Interpretation
```

**Edge Case Handling:**
- Ages < 18: Route to youngest available model (18-24)
- Ages > 94: Route to eldest model (85-94)
- Missing age: Require user input (cannot proceed without age as partition key)

**Manuscript Checklist:**
- [ ] Document the age band partitioning strategy
- [ ] Explain computational savings from partitioning
- [ ] Detail edge case handling and clinical validation

---

### 5.3 The Inference Engine: From Model to Prediction

#### 5.3.1 Robust Ensemble Strategy
**Three-Model Voting Block:**
- CatBoost (Gradient boosting on categorical features)
- XGBoost (Gradient boosting on all features)
- XGBoost RF (Random Forest variant)

**Performance-Based Weighting Algorithm:**

$$w_i = \frac{0.5 \times \text{PR-AUC}_i + 0.5 \times \frac{1}{1+\text{LogLoss}_i}}{\sum_j \left(0.5 \times \text{PR-AUC}_j + 0.5 \times \frac{1}{1+\text{LogLoss}_j}\right)}$$

**Consensus Calculation:**
$$\text{Ensemble Risk} = \sum_i w_i \times \text{Prediction}_i$$

*Rationale:* Weights derived from MC-CV validation metrics (Chapter 2/3), ensuring consensus prioritizes models with highest calibration and discrimination.

#### 5.3.2 Handling Real-World Data Sparsity (Partial Inputs)
**The Challenge:**
- Clinicians rarely have patient's full 3-year history during point-of-care
- Missing features would typically require model retraining or exclusion

**The Solution: "Imputation of Normality"**
1. **Accept Sparse Inputs:** API receives minimal data (e.g., Age + 2 Drug Codes)
2. **Fill Missing Features:** Missing trajectory features automatically populated with **training set medians** (representing "typical" patient profile)
3. **Preserve Signal:** Provided ICD/Drug codes drive the marginal risk prediction; imputed values provide baseline context

**Validation Strategy:**
- Demonstrate robustness through sensitivity analysis: vary imputed features, confirm stable predictions
- Show models trained on diverse patterns can reliably predict from partial inputs
- Include real-world case studies where sparse inputs yield actionable insights

**Manuscript Checklist:**
- [ ] Document the imputation algorithm in detail
- [ ] Provide sensitivity analysis showing prediction robustness
- [ ] Include clinical validation examples
- [ ] Compare against full-input predictions

---

### 5.4 Feature 1: Clinical Risk Assessment & Causal "What-If"

#### 5.4.1 The Clinician Interface
**Tabs 1 & 2: Patient Input & Risk Display**
- Dynamic input forms populated by metadata from Feature Importance pipeline (Step 3)
- Real-time risk score updates as clinician adjusts inputs
- Model agreement breakdown showing consensus strength

**Outputs Provided:**
- **Risk Score:** 0-100% probability scale
- **Risk Band:** Low / Moderate / High (with clinical thresholds)
- **Model Agreement:** Percentage of agreement; flags when models disagree (N out of 3 models agree)
- **Top Risk Drivers:** Most influential features for this patient's risk

#### 5.4.2 Causal Interaction Analysis ("What-If" Scenarios)
**Scenario Comparison Feature:**
- User selects a drug to "add" or "remove"
- System calculates counterfactual predictions using FFA results (Step 8)
- Display $\Delta$ Risk (delta) showing predicted change if intervention applied

**User Workflow:**
1. Enter baseline patient profile → Get baseline risk
2. Select hypothetical drug to add/remove
3. System returns: Predicted new risk + $\Delta$ Risk
4. Compare multiple scenarios side-by-side

**Causal Mechanism:**
- Grounded in causal directed acyclic graphs (DAGs) from Chapter 4
- FFA identifies true causal edges (not just associations)
- Counterfactuals respect causal directionality

**Manuscript Checklist:**
- [ ] Explain the causal inference foundation
- [ ] Document "What-If" algorithm
- [ ] Include UI mockups/screenshots
- [ ] Provide clinical use case examples

---

### 5.5 Feature 2: Exploratory Visualizations (Addressing ICPM Feedback)

#### 5.5.1 Contextualizing Risk
**Important Distinction:**
- BupaR and FP-Growth were **excluded from predictive models** (prevent data leakage)
- They are **reintroduced here for clinical context** (exploratory tab)
- Users see NOT included in risk score, but inform interpretation

**Clinical Value:**
- Patient sees the broader context: "typical pathway for patients like you"
- Supports shared decision-making: "here's what we often see in practice"

#### 5.5.2 Visualization Types

| Visualization | Technology | Clinical Purpose |
|---------------|-----------|------------------|
| **BupaR Sankey** | Interactive process mining | Show most probable patient pathways containing selected drugs; illuminate typical disease progressions |
| **FP-Growth Network** | Graph visualization (D3/Cytoscape) | Display "web" of comorbidities and drug interactions associated with patient profile |
| **DTW Trajectory** | Time series clustering plot | Plot patient's inputs against representative trajectory clusters; visualize disease progression stages |

**Interactive Features:**
- Hover to explore edge labels and pathway frequencies
- Filter by comorbidity or time window
- Export visualizations for clinical documentation

**Manuscript Checklist:**
- [ ] Justify why exploratory features don't compromise model integrity
- [ ] Document visualization design choices
- [ ] Include example screenshots with clinical annotations
- [ ] Explain how clinicians interpret each visualization

---

### 5.6 Feature 3: The PGx Patient Card (Personalized Medicine)

#### 5.6.1 Privacy-First Design
**Architecture Principles:**
- **Stateless System:** No patient data stored; all computation in-memory
- **Anonymous Workflow:** Only genetic variants (SNPs) processed; no linkage to identity
- **Ephemeral Results:** User downloads card; nothing persisted on server
- **Compliance:** HIPAA-ready; designed for patient self-service (no clinician input required)

**Technical Implementation:**
- API endpoint accepts user-uploaded genetic data (23andMe format)
- Validates SNP format; maps to known gene nomenclature
- Executes CPIC lookup; streams results to client
- Clears session; returns stateless response

#### 5.6.2 CPIC Integration
**Clinical Pharmacogenomics Implementation Consortium (CPIC) Database:**
- 573 gene-drug pairs with evidence-based guidelines
- Standardized phenotype classifications (Normal/Intermediate/Poor/Ultra-rapid metabolizers)

**Workflow:**
$$\text{User SNPs} \to \text{Phenotype Inference} \to \text{CPIC DB Lookup} \to \text{Drug-Gene Card}$$

**Output: "Drug-Gene Interaction Card"**
- Lists all relevant gene-drug interactions from user's genetic profile
- Includes CPIC recommendation level (A/B/C for actionability)
- Dosing adjustment guidance (if available)
- Exportable PDF for patient to share with healthcare provider

**Examples:**
- *CYP2D6* Poor Metabolizer + Codeine → Recommend alternative opioid
- *TPMT* Heterozygous + Azathioprine → Reduce dose by 50%
- *HLA-B*5701 Positive + Abacavir → Contraindicated; choose alternative

**Clinical Validation:**
- Demonstrate alignment with published CPIC guidelines
- Case studies showing actionable insights
- Patient survey on card comprehensibility

**Manuscript Checklist:**
- [ ] Document CPIC data source and update frequency
- [ ] Explain genotype-to-phenotype algorithm
- [ ] Include example cards (de-identified)
- [ ] Address privacy/security mechanisms
- [ ] Discuss clinical evidence base

---

### 5.7 Deployment & Reliability

#### 5.7.1 Incremental Build System
**Graceful Degradation:**
- Dashboard can be deployed even if some age cohorts are still processing models
- Missing cohorts are **disabled** in the frontend (not available in dropdown)
- System doesn't crash; users see clear message: "Age 45-54 models still processing. Available ranges: 18-44, 55-64, 65+."

**CI/CD Pipeline:**
1. **Model Training:** Triggered for each age band independently
2. **Containerization:** As each band completes, updated Lambda image built
3. **Deployment:** Incremental; frontend updated with available cohorts
4. **Rollback:** If new model fails validation, previous version stays live

**Validation Checkpoints:**
- Model performance thresholds (PR-AUC > 0.60, LogLoss < 0.3)
- Ensemble agreement tests (all 3 models must agree on risk band > 70%)
- Edge case validation (sparse inputs, extreme ages)

#### 5.7.2 Storage Analysis & Feasibility
**Model Footprint Calculation:**

| Component | Size | Notes |
|-----------|------|-------|
| **7 Age Band Cohorts** | | Each with 3 trained models |
| CatBoost models × 7 | ~150 MB | ~21 MB per model |
| XGBoost models × 7 | ~140 MB | ~20 MB per model |
| XGBoost RF models × 7 | ~180 MB | ~26 MB per model |
| Feature Engineering metadata | ~50 MB | Preprocessing pipelines |
| CPIC database snapshot | ~50 MB | Gene-drug pairs & guidelines |
| **Total** | **~570 MB** | — |
| **Docker Image (with OS)** | **~1.5 GB** | ✅ Well within 10 GB Lambda limit |

**Proof of Feasibility:**
- Demonstrate actual container builds with real models
- Benchmark Lambda cold start time (target: < 500ms)
- Measure inference latency per patient (target: < 100ms)

**Manuscript Checklist:**
- [ ] Include actual model size measurements
- [ ] Document deployment pipeline
- [ ] Show performance benchmarks (cold start, inference latency)
- [ ] Discuss scaling strategy (future roadmap)

---

### 5.8 Conclusion

**Chapter Summary:**
The PGx Risk Dashboard demonstrates the complete translational pipeline:

$$\text{Data} \to \text{Models} \to \text{Causal Insights} \to \text{Actionable Bedside Tool}$$

**Key Contributions:**
1. **Serverless Architecture:** Proves ML models can be deployed at scale without complex infrastructure
2. **Privacy-First Design:** Demonstrates clinical-grade tools can operate without storing PII
3. **Handling Real-World Data:** Solves the partial input problem endemic to clinical workflows
4. **Interpretability + Causal Reasoning:** Moves beyond "black box" predictions to explainable, interactive risk stratification
5. **Incremental Deployment:** Shows graceful degradation allows phased rollout

**Translational Impact:**
- Democratizes access to cutting-edge PGx insights
- Supports shared decision-making at point-of-care
- Bridges the gap between research models and clinical practice

**Final Statement:**
This chapter demonstrates that Translational Informatics is not just about publishing accurate models—it's about ensuring those models reach patients and providers in usable, trustworthy, and actionable forms.

**Manuscript Checklist:**
- [ ] Frame translational value clearly
- [ ] Connect to PhD program learning objectives
- [ ] Address limitations (e.g., FDA approval pathway, liability)
- [ ] Discuss future enhancements

---

## Implementation Tracking

### Section 5.1 – Introduction
- [ ] Clinical motivation and problem statement drafted
- [ ] Chapter objectives clearly stated
- [ ] Translational contribution articulated

### Section 5.2 – System Architecture
- [ ] Hybrid deployment model documented with diagrams
- [ ] Partition-first routing logic explained
- [ ] Edge case handling described with examples

### Section 5.3 – Inference Engine
- [ ] Ensemble strategy detailed with pseudocode
- [ ] Performance-based weighting algorithm formalized
- [ ] Partial input imputation algorithm documented
- [ ] Robustness validation completed

### Section 5.4 – Risk Assessment & Causal Features
- [ ] Clinician interface mockups/screenshots included
- [ ] "What-If" causal analysis algorithm documented
- [ ] Clinical use cases provided

### Section 5.5 – Exploratory Visualizations
- [ ] BupaR Sankey visualization examples included
- [ ] FP-Growth network graphs demonstrated
- [ ] DTW trajectory clustering explained
- [ ] Clinical interpretation guidance provided

### Section 5.6 – PGx Patient Card
- [ ] Privacy architecture documented
- [ ] CPIC integration workflow described
- [ ] Example patient cards included (de-identified)
- [ ] Security validation completed

### Section 5.7 – Deployment & Reliability
- [ ] CI/CD pipeline architecture documented
- [ ] Storage analysis with actual measurements included
- [ ] Performance benchmarks collected (cold start, inference latency)
- [ ] Graceful degradation strategy explained

### Section 5.8 – Conclusion
- [ ] Translational contributions summarized
- [ ] Limitations acknowledged
- [ ] Future work outlined

---

## Technical Specifications

**Model Ensemble:**
- 21 total trained models (7 age bands × 3 models)
- Age bands: 18-24, 25-34, 35-44, 45-54, 55-64, 65-74, 75-84, 85-94, 95+
- Algorithms: CatBoost, XGBoost, XGBoost Random Forest

**Supported Age Range:**
- Input: 0-150 years (mapped to nearest available band)
- Primary: 18-94
- Edge cases: < 18 → 18-24 band; > 94 → 85-94 band

**Feature Dimensions:**
- Input ICD-10 codes: dynamic (varies by patient)
- Input Drug codes: dynamic (varies by patient)
- Input trajectories: 3-year history (filled with medians if sparse)
- Total training features: ~500+ derived features

**Inference Latency Targets:**
- Cold start (Lambda): < 500 ms
- Warm inference: < 100 ms per patient
- Dashboard page load: < 2 seconds

**Deployment Environment:**
- AWS Lambda (containerized)
- AWS S3 (static frontend)
- AWS ECR (model image repository)
- Optional: CloudFront (CDN for frontend)

---

## Key References & Integration Points

**From Chapter 3 (Predictive Modeling):**
- Ensemble validation metrics (PR-AUC, LogLoss) → Used for weighting
- Feature importance rankings → Populate dynamic input forms

**From Chapter 4 (Causal Inference):**
- Causal DAGs → Validate "What-If" counterfactual directions
- FFA results (true causal edges) → Grounds scenario comparisons
- Partial effects estimates → Support risk delta calculations

**External Standards:**
- CPIC guidelines (573 gene-drug pairs)
- HIPAA compliance requirements
- Clinical decision support regulations (FDA 21 CFR Part 11)

---

## Manuscript Outline Status

| Section | Status | Notes |
|---------|--------|-------|
| 5.1 Introduction | [ ] Draft | Start with problem framing |
| 5.2 Architecture | [ ] In Progress | Include system diagrams |
| 5.3 Inference Engine | [ ] Planned | Detail ensemble + imputation logic |
| 5.4 Risk Assessment | [ ] Planned | Add UI/UX considerations |
| 5.5 Visualizations | [ ] Planned | Include screenshot gallery |
| 5.6 PGx Patient Card | [ ] Planned | Emphasize privacy design |
| 5.7 Deployment | [ ] Planned | Add performance benchmarks |
| 5.8 Conclusion | [ ] Final | Synthesize translational impact |

---

## Notes for Authors

- **Keep Chapter Focused:** This is implementation, not methods review. Assume readers know Chapters 3-4.
- **Clinical Orientation:** Frame technical decisions through clinician's perspective. "Why does this matter for patient care?"
- **Reproducibility:** Include code references, Git commit hashes, and container image IDs for traceability.
- **Limitations Upfront:** Address FDA approval pathway, validation on external cohorts, liability frameworks early.
- **Iterative Development:** This chapter reflects v1.0. Document future enhancements (mobile app, real-time feedback loops, etc.).

---

**Document Created:** January 2026  
**Chapter:** PhD in Translational Informatics – Final Implementation Paper  
**Contact:** [Add your contact info]
