# PGx Risk Assessment Dashboard - Complete System Overview

## What This System Provides

A comprehensive, production-ready web dashboard that combines **clinical risk prediction** with **pharmacogenomic (PGx) guidance** for personalized medication safety.

---

## 🎯 Two Main Capabilities

### 1. **Risk Assessment Dashboard** (Tab 1)
**Purpose**: Predict risk of adverse drug events using machine learning models

**What Users Get**:
- **Opioid ED Visit Risk Score** (cohort selected via **Opioid ED** tab; full age bands)
  - Predicts risk of F1120 opioid-related emergency department visits
  - Based on patient age, ICD codes, CPT codes, and drug names
  
- **Polypharmacy Risk Score** (cohort selected via **Polypharmacy** tab; full age bands)
  - Predicts risk of polypharmacy-related complications
  - Based on patient age and drug combinations

**Key Features**:
- ✅ **Robust Ensemble Models**: Uses 3 models (CatBoost, XGBoost, XGBoost RF) with performance-based weighting
- ✅ **Cohort from tab**: **Opioid ED** or **Polypharmacy** tab sets cohort; age selects age band within that cohort
- ✅ **Feature-Driven Inputs**: Dropdowns populated from actual feature importances
- ✅ **Scenario Comparison**: Compare risk changes for different drug/ICD/CPT combinations
- ✅ **Model Transparency**: See individual model predictions and ensemble weights
- ✅ **Risk Bands**: Low/Medium/High risk categorization

**Outputs**:
- Risk score (0-100%)
- Risk band (Low/Medium/High)
- Per-model breakdown
- Visual charts showing model contributions
- Comparison scenarios with risk deltas

---

### 2. **PGx Patient Card** (Tab 2)
**Purpose**: Generate personalized pharmacogenomic cards showing drug-gene interactions

**What Users Get**:
- **Anonymous, Generic PGx Card**
  - Shows which drugs may require dosing modifications based on genetic variants
  - Links to CPIC (Clinical Pharmacogenomics Implementation Consortium) guidelines
  - FDA labeling information
  - CPIC evidence levels

**Key Features**:
- ✅ **Privacy-First**: Anonymous, no personal identification required
- ✅ **Master CPIC Database**: Uses official CPIC Excel file (573 pairs, 300 drugs, 121 genes)
- ✅ **Simple Input**: Enter gene variants from ancestry reports (23andMe, Ancestry.com, etc.)
- ✅ **File Upload Support**: CSV, Excel, or text files
- ✅ **Timestamp & IP Tracking**: For audit purposes (not identification)
- ✅ **Optional Patient ID**: Can add identification if desired, but not necessary

**Input Format**:
```
CYP2D6,*1,*2
CYP2C19,*1,*17
CYP2C9,*1,*2
TPMT,*1,*3
```

**Outputs**:
- List of genes tested with variants
- List of drugs requiring dosing modifications
- CPIC guideline URLs for each drug-gene pair
- CPIC evidence levels (A, B, etc.)
- FDA labeling information
- Gene details with allele counts

---

## 🔧 Technical Architecture

### **Frontend** (S3 Static Website)
- **HTML/JavaScript Dashboard**
  - Tabbed interface (Risk Assessment + PGx Card)
  - Interactive forms with searchable dropdowns
  - Real-time risk calculation
  - Visual charts (Plotly)
  - Responsive design

### **Backend** (AWS Lambda with ECR)
- **Serverless API** (API Gateway + Lambda)
  - `/metadata` - Get valid codes for dropdowns
  - `/risk` - Calculate risk scores
  - `/risk/comparison` - Compare scenarios
  - `/pgx/card` - Generate PGx cards

- **Model Storage** (ECR Container - 10GB limit)
  - All trained models (CatBoost, XGBoost, XGBoost RF)
  - Full age band set for both cohorts (0-12 through 85-114; 8 bands; both opioid_ed and non_opioid_ed use same set)
  - Feature schemas with model weights
  - CPIC master Excel file
  - Total: ~1.5 GB (well within limit)

- **Ensemble Approach**
  - Combines predictions from all 3 models
  - Performance-based weighting (from MC-CV results)
  - Composite score: `0.5 × PR-AUC + 0.5 × (1/(1+LogLoss))`
  - Graceful degradation if models fail

---

## 📊 Data Sources

### **Risk Models**
- **Training Data**: S3 gold cohort parquet files (~7 GB)
  - Opioid ED: 179 MB
  - Polypharmacy: 6.8 GB
- **Feature Importances**: From Step 3 analysis
- **Model Weights**: From Step 8 MC-CV results

### **PGx Card**
- **CPIC Master File**: `cpic_gene-drug_pairs.xlsx`
  - 573 gene-drug pairs
  - 300 drugs
  - 121 genes
  - Official CPIC guidelines
  - FDA labeling information

---

## 💡 Use Cases

### **For Healthcare Providers**
1. **Pre-prescription Risk Assessment**
   - Enter patient age and current medications
   - Get risk score before prescribing opioids
   - Compare scenarios (e.g., "What if we add this drug?")

2. **Polypharmacy Management**
   - Assess risk for elderly patients (65+)
   - Identify high-risk drug combinations
   - Make informed decisions about medication adjustments

3. **Pharmacogenomic Guidance**
   - Generate PGx card from patient's genetic test results
   - Identify drugs requiring dose modifications
   - Access CPIC guidelines for clinical decision-making

### **For Patients**
1. **Self-Assessment** (with provider guidance)
   - Understand personal risk factors
   - See how different medications affect risk
   - Get PGx card from ancestry report

2. **Medication Safety**
   - Know which drugs may need dose adjustments
   - Have PGx card for healthcare providers
   - Access evidence-based guidelines

---

## 🎁 Key Benefits

### **1. Evidence-Based Predictions**
- Models trained on real clinical data
- Ensemble approach improves robustness
- Performance-weighted for reliability

### **2. Actionable Insights**
- Clear risk scores and bands
- Specific drug-gene interactions
- Links to clinical guidelines

### **3. Privacy-Focused**
- Anonymous PGx cards
- No personal identification required
- Users control what information to include

### **4. Scalable & Production-Ready**
- Serverless architecture (Lambda + API Gateway)
- ECR container for large models (10GB)
- S3 static hosting for frontend
- Handles all age bands automatically

### **5. Transparent & Explainable**
- See individual model predictions
- Understand ensemble weights
- Access CPIC evidence levels
- Model breakdown charts

---

## 📈 What Problems This Solves

### **Problem 1: Opioid Overdose Risk**
- **Solution**: Predict F1120 opioid ED visit risk before prescribing
- **Impact**: Reduce emergency department visits, improve patient safety

### **Problem 2: Polypharmacy Complications**
- **Solution**: Assess polypharmacy risk for elderly patients
- **Impact**: Prevent adverse drug interactions, optimize medication regimens

### **Problem 3: Pharmacogenomic Guidance**
- **Solution**: Generate personalized PGx cards from genetic data
- **Impact**: Enable precision medicine, prevent adverse reactions

### **Problem 4: Lack of Clinical Decision Support**
- **Solution**: Provide evidence-based risk scores and drug-gene interactions
- **Impact**: Support clinical decision-making with data-driven insights

---

## 🚀 Deployment Status

### **Ready for Production**
- ✅ All models trained and validated
- ✅ Ensemble approach implemented
- ✅ API endpoints defined
- ✅ Frontend dashboard complete
- ✅ PGx card generation functional
- ✅ CPIC data integrated
- ✅ Docker containerization ready
- ✅ Documentation complete

### **Deployment Steps**
1. Build Docker image with models and CPIC data
2. Push to ECR
3. Deploy Lambda function
4. Configure API Gateway
5. Upload dashboard HTML to S3
6. Configure S3 static website hosting

---

## 📋 Summary: What You Get

**A complete, production-ready system that provides**:

1. **Risk Prediction**: ML-powered risk scores for opioid ED visits and polypharmacy
2. **PGx Guidance**: Personalized drug-gene interaction cards
3. **Clinical Decision Support**: Evidence-based recommendations with CPIC guidelines
4. **Privacy Protection**: Anonymous, generic cards with optional identification
5. **Scalability**: Serverless architecture handling all age bands
6. **Transparency**: Model breakdowns and evidence levels
7. **User-Friendly**: Simple web interface, no technical knowledge required

**This enables healthcare providers and patients to make informed decisions about medication safety based on clinical risk prediction and pharmacogenomic evidence.**

