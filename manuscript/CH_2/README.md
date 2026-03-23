# Chapter 2: Methodology – A Scalable Causal-Oriented Pipeline

## Overview

This project implements a comprehensive, production-grade pipeline for analyzing terabytes of longitudinal healthcare claims data (APCD) to identify causal drug interactions and predict opioid and polypharmacy-related emergency department (ED) visits. The system bridges the gap between big data processing and clinically interpretable, causally-grounded insights through a novel **Consensus Filter** approach combining distributional (SHAP) and structural (FFA) analysis.

**Key Innovation:** A **Partition-First Architecture** enabling 15x throughput improvement and linear scalability, combined with a **Consensus Filter** that extracts causal relationships through model agreement between multiple analytical perspectives.

---

## Core Innovations

### 1. **Partition-First Architecture**
- Transforms monolithic data processing into discrete partitions (Age Band × Year)
- Independent DuckDB workers process specific partitions across distributed EC2 instances
- **Performance:** 15x throughput improvement, linear scalability across 32-core, 1TB RAM infrastructure
- **Resilience:** S3-based state management for checkpoint-based recovery

### 2. **Consensus Filter: A Causal Discovery Framework**
- **Philosophy:** A feature is robust only if identified by *both* distributional and structural analysis
- **SHAP (Quantitative Attribution):** Shapley values measure magnitude and direction of feature contribution
- **Formal Feature Attribution (FFA):** Converts tree-based models into interpretable Boolean rules (e.g., `IF Drug A AND NOT Drug B THEN Risk`)
- **Hybrid Filter:** Prioritizes symbolic rules using SHAP scores for maximum interpretability

### 3. **Visualization-Only Process Mining**
- FP-Growth and BupaR used exclusively for exploratory dashboards
- Removed from predictive models to prevent target leakage
- Critical methodological safeguard for causal validity

---

## Architecture Overview

### Data Infrastructure: Bronze-Silver-Gold Data Lake

```
Raw Text Data (Bronze)
    ↓
Imputed Demographics (Silver)
    ↓
Analysis-Ready Cohorts (Gold) → Partitioned by Age Band × Year
    ↓
Independent DuckDB Workers
    ↓
Feature-Engineered Datasets
```

**Key Stages:**
1. **Bronze Layer:** Raw claims data transformation
2. **Silver Layer:** Demographic imputation and standardization
3. **Gold Layer:** Cohort construction with signal isolation and noise reduction

### Cohort Construction

#### Target Definitions (Mutually Exclusive)
- **Opioid ED:** ICD codes (e.g., F11.20) across all 10 diagnosis columns
- **Polypharmacy (Non-Opioid ED):** Milliman HCG codes (e.g., O11 Emergency Room) with strict opioid exclusion

#### Statistical Controls
- **5:1 Matching Ratio:** 5 control patients per target case
- **Noise Reduction:**
  - DTW Protocol Filtering: Removes administrative/scheduling artifacts
  - Extreme-Density Split: Isolates top 5% high-transaction patients to prevent model bias

---

## Feature Engineering Pipeline

### Step 1: Pharmacogenomics (PGx) Enrichment
- Integration of CPIC guidelines
- Allele frequency databases
- Drug-gene interaction scoring

### Step 2: Feature Selection via Monte Carlo Cross-Validation
- 50+ random train-test splits
- Robust ranking by Recall and AUC-PR
- Noise filtering before final model training
- Strict temporal validation: Training (2016–2018) vs. Holdout Test (2019)
- COVID-19 confounding avoided (2020 excluded)

---

## Ensemble Modeling Engine

### Multi-Model Architecture

Three complementary tree-based models:

1. **CatBoost** – Captures categorical nuances and feature interactions
2. **XGBoost** – Gradient boosting for sequential improvement
3. **XGBoost RF** – Random Forest mode for robustness

### Performance-Based Weighting

Ensemble weights calculated from MC-CV validation phase:
- Composite score: Recall + LogLoss normalization
- Dynamic reweighting based on temporal performance patterns

### Model Validation

- **Train Set:** 2016–2018 longitudinal data
- **Test Set:** 2019 (strict temporal holdout)
- **Exclusion:** 2020 (COVID-19 confounding)

---

## The Consensus Filter: Causal Inference

### Causal Discovery Methodology

#### SHAP-Based Analysis
- Quantifies feature contribution magnitude and direction
- Provides distributional perspective on model decisions

#### Formal Feature Attribution (FFA)
- Converts XGBoost decision trees into interpretable Boolean rules
- Example: `IF (Opioid ≥ 2 prescriptions) AND (NOT Benzodiazepine) THEN Risk ↑`
- Rules extracted and prioritized by SHAP scores

#### Causal Interaction Analysis
**Intervention Rate Metric:** Measures how model logic changes under counterfactual modifications:
- Feature removal
- Median value substitution
- Alternative drug substitutions

---

## System Deployment

### Serverless Architecture
- Ensemble models packaged in AWS Lambda/ECR containers
- Real-time inference capability
- Scalable, cost-efficient deployment

### Privacy-First Design
- Anonymous PGx card generation
- No PII storage requirement
- HIPAA-compliant data handling

### Risk Dashboard
- Exploratory visualization-only process mining (FP-Growth, BupaR)
- Interactive causal rule exploration
- Real-time risk stratification interface

---

## Project Structure

```
CH_2/
├── README.md                          # This file
├── data/
│   ├── bronze/                        # Raw claims data
│   ├── silver/                        # Imputed demographics
│   └── gold/                          # Analysis-ready cohorts
├── src/
│   ├── partition_engine.py            # Partition-First processor
│   ├── cohort_construction.py         # Target/control definitions
│   ├── feature_engineering.py         # PGx enrichment & selection
│   ├── ensemble_models.py             # Model training & weighting
│   ├── consensus_filter.py            # SHAP + FFA causal filter
│   └── deployment/                    # Lambda/container configs
├── notebooks/
│   ├── exploratory_analysis.ipynb     # Dashboard & visualization
│   └── causal_validation.ipynb        # Consensus Filter results
├── tests/
│   ├── test_partition_engine.py
│   ├── test_cohort_construction.py
│   └── test_consensus_filter.py
└── config/
    ├── partitions.yaml                # Age Band × Year definitions
    ├── feature_selection.yaml         # MC-CV parameters
    └── model_weights.yaml             # Ensemble weighting config
```

---

## Key Methodology Contributions

### Problem Addressed
- **Computational Challenge:** Processing terabytes of longitudinal claims data while maintaining patient-level granularity
- **Interpretability Gap:** Bridging black-box ML predictive power with logical explainability for clinical acceptance

### Solutions Provided

1. **Scalable Infrastructure:** 15x throughput improvement through partition-first architecture
2. **Causal Validity:** Consensus Filter prevents target leakage and enforces model agreement
3. **Clinical Interpretability:** Symbolic rule extraction via FFA enables actionable insights
4. **Robust Validation:** Monte Carlo cross-validation with strict temporal safeguards

---

## Key Metrics & Performance

| Metric | Value |
|--------|-------|
| Throughput Improvement | 15x vs. monolithic baseline |
| Scalability | Linear across distributed workers |
| Cross-validation Splits | 50+ random partitions |
| Temporal Validation | 2016–2018 train / 2019 holdout |
| Ensemble Models | 3 (CatBoost, XGBoost, XGBoost RF) |
| Consensus Filter Criteria | SHAP ∩ FFA agreement |

---

## Dependencies & Requirements

### Data Processing
- DuckDB (distributed query engine)
- AWS S3 (state management & checkpointing)
- APCD claims database (input data)

### ML & Analytics
- CatBoost, XGBoost (ensemble models)
- SHAP (explainability)
- Symbolic logic engine for FFA (AXP/Z3)
- scikit-learn (MC-CV utilities)

### Deployment
- AWS Lambda / ECR (serverless inference)
- FastAPI (REST endpoints)
- Docker (containerization)

### Exploratory Tools
- FP-Growth (association mining)
- BupaR (process mining visualization)
- Plotly/Dash (interactive dashboards)

---

## Quick Start

### 1. Data Preparation
```bash
# Stage raw claims data in Bronze layer
python src/data_loader.py --input raw_claims.csv --output s3://bronze-bucket/claims/

# Process to Silver layer (imputation)
python src/silver_processor.py --source s3://bronze-bucket/ --dest s3://silver-bucket/

# Build Gold layer cohorts
python src/gold_cohort_builder.py --source s3://silver-bucket/ --dest s3://gold-bucket/
```

### 2. Partitioned Processing
```bash
# Launch distributed partition processing
python src/partition_engine.py \
  --input s3://gold-bucket/ \
  --age-bands "0-18,19-35,36-50,51-65,65+" \
  --years 2016,2017,2018,2019 \
  --workers 32
```

### 3. Feature Engineering & Model Training
```bash
# Monte Carlo cross-validation feature selection
python src/feature_engineering.py \
  --input s3://partitioned-data/ \
  --pgx-database cpic_alleles.db \
  --cv-splits 50 \
  --output s3://features-bucket/

# Train ensemble with MC-CV weighting
python src/ensemble_models.py \
  --features s3://features-bucket/ \
  --train-years 2016-2018 \
  --test-year 2019 \
  --output models/ensemble/
```

### 4. Causal Inference via Consensus Filter
```bash
# Extract SHAP + FFA agreement
python src/consensus_filter.py \
  --models models/ensemble/ \
  --features s3://features-bucket/ \
  --output causal_insights/consensus_rules.json
```

### 5. Deploy Risk Dashboard
```bash
# Package and deploy
docker build -t risk-dashboard .
aws ecr push risk-dashboard
aws lambda create-function --role arn:aws:iam::xxx --image-uri risk-dashboard:latest
```

---

## Methodological Notes

### Why "Visualization-Only" Process Mining?

FP-Growth and BupaR reveal temporal patterns but introduce **target leakage** when used in predictions. These tools inform exploratory analysis and dashboard features but are explicitly excluded from the predictive ensemble to maintain causal validity.

### Why Consensus Filter (SHAP ∩ FFA)?

No single analytical method is sufficient for causal inference:
- **SHAP alone:** Measures correlation/importance, not causality
- **FFA alone:** Rule extraction can miss distributional nuances
- **SHAP ∩ FFA:** Model agreement provides robustness across complementary analytical perspectives

### Temporal Validation Rationale

- **Training:** 2016–2018 (pre-COVID, stable healthcare patterns)
- **Testing:** 2019 (out-of-sample, same era)
- **Exclusion:** 2020+ (COVID-19 pandemic confounding)

---

## Authors & Citation

**Dissertation Chapter:** Chapter 2 – Methodology: A Scalable Causal-Oriented Pipeline

This work represents research into:
- Large-scale healthcare data processing
- Causal discovery in complex systems
- Explainable machine learning for clinical applications
- Production deployment of interpretable models

---

## License

[Specify appropriate license]

---

## Contact & Support

For questions or contributions, please contact the research team or open an issue in the project repository.

---

## Appendix: Advanced Topics

### A. Partition Strategy Deep Dive
Age Band × Year partitions maximize:
- **Resource Utilization:** Independent DuckDB workers process non-overlapping cohorts
- **Resilience:** Failure in one partition doesn't block others; S3 checkpoints enable recovery
- **Scalability:** Linear relationship between workers and throughput

### B. Causal Interaction Examples
```
Rule 1 (High Risk):
  IF (Opioid prescriptions ≥ 2) AND (NOT Benzodiazepine) AND (Age > 50)
  THEN P(ED visit) ↑↑ (85% confidence)

Rule 2 (Protective):
  IF (PGx poor metabolizer status) AND (Dose adjusted) 
  THEN P(ED visit) ↓ (70% confidence)
```

### C. Infrastructure Specifications
- **Compute:** AWS EC2 (32 cores, 1TB RAM per worker)
- **Storage:** S3 (Bronze/Silver/Gold layer buckets)
- **Processing:** DuckDB with parallel execution
- **Deployment:** Lambda + ECR for serverless inference

---

**Last Updated:** January 2026  
**Status:** Production Research Pipeline
