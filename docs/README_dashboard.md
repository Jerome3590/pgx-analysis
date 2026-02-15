# Risk Assessment Dashboard

## Overview

The **Risk Assessment Dashboard** (Step 9) is a production-ready web application that provides comprehensive risk prediction and analysis for pharmacogenomic outcomes. The dashboard combines machine learning models, causal analysis, process mining, and pharmacogenomic insights into an integrated clinical decision support tool.

**Primary Capabilities:**

1. **Risk Assessment** - Predict risk for **Opioid ED** or **Polypharmacy** cohorts across all age bands (0-12 through 85-114)
2. **Causal Analysis** - Explore FFA causal factors and SHAP feature importance
3. **DTW Trajectories** - View patient temporal trajectory patterns
4. **FP-Growth Patterns** - Explore frequent itemsets and association rules
5. **BupaR Process Mining** - View clinical process flows and activity sequences
6. **PGx Patient Card Generator** - Generate pharmacogenomic cards from genetic variants

**Cohorts:**
- **opioid_ed**: Full set of 8 age bands (0-12, 13-24, 25-44, 45-54, 55-64, 65-74, 75-84, 85-114)
- **non_opioid_ed**: Full set of 8 age bands (same as opioid_ed)

**Total: 16 cohort/age_band combinations (2 cohorts × 8 bands)**

---

## Table of Contents

- [Overview](#overview)
- [Dashboard Requirements](#dashboard-requirements)
  - [Required Outputs by Step](#required-outputs-by-step)
  - [Minimum Requirements](#minimum-requirements)
  - [Checking Availability](#checking-availability)
- [Directory Structure](#directory-structure)
- [Core Components](#core-components)
  - [Frontend](#frontend-frontend)
  - [Backend](#backend-backend)
  - [Data Preparation](#data-preparation-data_preparation)
  - [Visualizations](#visualizations-visualizations)
  - [Deployment](#deployment-deployment)
- [Dashboard Features](#dashboard-features)
- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [API Endpoints](#api-endpoints)
- [Data Sources](#data-sources)
- [FP-Growth Network Integration](#fp-growth-network-visualization-integration)
- [Build Process](#build-process)
- [Documentation](#documentation)

---

## Dashboard Requirements

The dashboard build process (`build_dashboard.sh`) requires outputs from multiple pipeline steps. **All cohorts and age bands must be completed before the dashboard can be built.**

### Required Outputs by Step

#### Step 3: Feature Importance (Required for Metadata)

**Location:** `3a_feature_importance/outputs/`

**Required Files:**
- `{cohort}_{age_band}_aggregated_feature_importance.csv`
  - **Used by:** `generate_metadata.py`
  - **Purpose:** Extract top ICD/CPT/Drug codes for dashboard metadata

**Example:**
```
3a_feature_importance/outputs/opioid_ed_13_24_aggregated_feature_importance.csv
```

**Note:** If Feature Importance EDA files are available (`3b_feature_importance_eda/outputs/{cohort}/{age_band}/cohort_feature_importance.csv`), those are prioritized over Step 3 aggregated files.

---

#### Step 6: Final Model Training (Required for Models)

**Location:** `6_final_model/outputs/{cohort}/{age_band}/`

**Required Files:**

1. **Model Files:**
   - `models/catboost.joblib` (or `models/catboost.cbm`)
   - `models/xgboost.joblib` (or `models/xgboost_model.ubj`)
   - **Used by:** `prepare_models.py`
   - **Purpose:** Trained models for risk prediction

2. **Training Data:**
   - `{cohort}_{age_band}_train_final_features_no_leakage.csv`
   - **Used by:** `prepare_models.py`
   - **Purpose:** Extract feature schema and default values

3. **Model Performance Metrics:**
   - `models/{cohort}_{age_band}_mc_cv_results.csv`
   - **Used by:** `prepare_models.py`
   - **Purpose:** Calculate model weights based on MC-CV performance

**Example:**
```
6_final_model/outputs/opioid_ed/13_24/
├── models/
│   ├── catboost.joblib
│   ├── xgboost.joblib
│   └── opioid_ed_13_24_mc_cv_results.csv
└── opioid_ed_13_24_train_final_features_no_leakage.csv
```

---

#### Step 7: SHAP Analysis (Optional but Recommended)

**Location:** `7_shap_analysis/outputs/{cohort}/{age_band}/`

**Required Files:**
- `{cohort}_{age_band}_shap_global_importance_xgboost.csv`
- `{cohort}_{age_band}_shap_sample_values_xgboost.parquet`
- **Used by:** Dashboard for feature explanations
- **Purpose:** SHAP values for individual patient explanations

**S3 Location:** `s3://pgxdatalake/gold/shap_analysis/{cohort}/{age_band}/`

**Example:**
```
7_shap_analysis/outputs/opioid_ed/13_24/
├── opioid_ed_13_24_shap_global_importance_xgboost.csv
└── opioid_ed_13_24_shap_sample_values_xgboost.parquet
```

---

#### Step 8: FFA Analysis (Optional but Recommended)

**Location:** `8_ffa_analysis/outputs/{cohort}/{age_band}/xgboost/`

**Required Files:**
- `axp_explanations.parquet`
- `feature_importance_axp.parquet`
- `causal_importance.parquet` (optional)
- **Used by:** Dashboard for causal explanations
- **Purpose:** Formal Feature Attribution explanations and causal importance

**S3 Location:** `s3://pgxdatalake/gold/ffa_analysis/{cohort}/{age_band}/xgboost/`

**Example:**
```
8_ffa_analysis/outputs/opioid_ed/13_24/xgboost/
├── axp_explanations.parquet
├── feature_importance_axp.parquet
└── causal_importance.parquet
```

---

### Minimum Requirements

**To build a basic dashboard (models only):**
- ✅ Step 3: Feature importance CSV
- ✅ Step 6: Model files + training data + MC-CV results

**To build a full-featured dashboard (with explanations):**
- ✅ Step 3: Feature importance CSV
- ✅ Step 6: Model files + training data + MC-CV results
- ✅ Step 7: SHAP outputs
- ✅ Step 8: FFA outputs

**File Structure Summary:**

For each cohort/age_band combination, you need:

```
{cohort}/{age_band}/
├── Step 3: Feature Importance
│   └── {cohort}_{age_band}_aggregated_feature_importance.csv
│
├── Step 6: Final Model
│   ├── models/
│   │   ├── catboost.joblib
│   │   ├── xgboost.joblib
│   │   └── {cohort}_{age_band}_mc_cv_results.csv
│   └── {cohort}_{age_band}_train_final_features_no_leakage.csv
│
├── Step 7: SHAP Analysis (optional)
│   ├── {cohort}_{age_band}_shap_global_importance_xgboost.csv
│   └── {cohort}_{age_band}_shap_sample_values_xgboost.parquet
│
└── Step 8: FFA Analysis (optional)
    └── xgboost/
        ├── axp_explanations.parquet
        ├── feature_importance_axp.parquet
        └── causal_importance.parquet
```

### Checking Availability

To check what's available for dashboard build:

```bash
# Check Step 6 outputs (required)
ls -la 6_final_model/outputs/opioid_ed/*/models/*.joblib

# Check Step 3 outputs (required for metadata)
ls -la 3a_feature_importance/outputs/*aggregated_feature_importance.csv

# Check Step 7 outputs (optional)
ls -la 7_shap_analysis/outputs/opioid_ed/*/*shap*.csv

# Check Step 8 outputs (optional)
ls -la 8_ffa_analysis/outputs/opioid_ed/*/xgboost/*.parquet
```

**Important Notes:**
- **All cohorts and age bands are required** - the build will fail if any are missing
- Step 6 outputs are **required** - without them, the dashboard cannot make predictions
- Step 3 outputs are **required** for metadata generation
- Steps 7 and 8 are **optional** but recommended for full explanation features
- The build script validates that all required cohorts/age_bands are present before building

---

## Directory Structure

```text
10_risk_dashboard/
├── frontend/                          # Frontend dashboard (user-facing)
│   ├── index.html                     # Main dashboard HTML with all tabs
│   ├── assets/                        # Static assets (CSS, JS, images)
│   ├── dashboard_index_template.html  # Template for dashboard index
│   └── README.md                      # Frontend documentation
│
├── backend/                           # Backend API (Lambda function)
│   ├── lambda_function.py             # AWS Lambda handler (API endpoints)
│   ├── lambda_api_template.py         # API Gateway integration template
│   ├── requirements.txt               # Python dependencies
│   ├── Dockerfile                     # Docker container for Lambda (ECR)
│   └── README.md                      # Backend API documentation
│
├── deployment/                        # Deployment scripts and configs
│   ├── docker_build.sh                # Build and push Docker image to ECR
│   ├── prepare_lambda_dir.py          # Prepare Lambda deployment package (optional)
│   ├── scripts/                       # Additional deployment helper scripts
│   └── README.md                      # Deployment documentation
│
├── data_preparation/                  # Scripts to prepare data for dashboard
│   ├── prepare_models.py              # Package models for Lambda deployment
│   ├── generate_metadata.py           # Generate metadata JSON files
│   ├── prepare_cpic_data.py           # Prepare CPIC data for PGx cards
│   ├── combine_shap_ffa_results.py    # Combine SHAP and FFA results
│   └── README.md                      # Data preparation documentation
│
├── visualizations/                    # Visualization generation scripts
│   ├── dtw/                           # DTW trajectory visualizations
│   │   ├── create_dtw_features.py
│   │   ├── create_dtw_visuals.py
│   │   ├── create_predictive_time_features.py
│   │   ├── dtw_cohort_runner.ipynb
│   │   ├── DTW_FEATURE_ANALYSIS.md
│   │   └── outputs/                   # DTW visualization outputs
│   ├── fpgrowth/                      # FP-Growth pattern visualizations
│   │   ├── create_fpgrowth_visuals.py
│   │   ├── create_plots.py
│   │   ├── create_fpgrowth_features.py
│   │   ├── cohort_fpgrowth.py
│   │   ├── global_fpgrowth.py
│   │   └── outputs/                   # FP-Growth visualization outputs
│   ├── bupar/                         # BupaR process mining visualizations
│   │   ├── create_bupar_visuals.py
│   │   ├── create_bupar_outputs_opioid_ed.R
│   │   ├── create_bupar_outputs_non_opioid_ed.R
│   │   ├── create_plots.R
│   │   ├── build_bupar_eventlogs.R
│   │   └── outputs/                   # BupaR visualization outputs
│   └── README.md                      # Visualization overview
│
├── outputs/                           # Generated outputs for dashboard
│   ├── models/                        # Prepared models for Lambda
│   │   └── {cohort}/{age_band}/
│   │       ├── catboost.joblib
│   │       ├── xgboost.joblib
│   │       ├── xgboost_rf.joblib
│   │       └── feature_schema.json
│   ├── metadata/                      # Metadata JSON files
│   │   ├── metadata_opioid_ed.json
│   │   └── metadata_non_opioid_ed.json
│   ├── cpic/                          # CPIC data files
│   │   └── cpic_gene-drug_pairs.xlsx
│   └── visualizations/                # Visualization images/data
│       ├── dtw/
│       ├── fpgrowth/
│       └── bupar/
│
└── docs/                              # Additional documentation
    ├── API.md                         # API endpoint documentation
    ├── DEPLOYMENT.md                  # Deployment guide
    └── VISUALIZATIONS.md              # Visualization guide
```

### Organizational Rationale

This structure follows a **separation of concerns** approach:

- **Frontend**: All user-facing HTML/CSS/JavaScript
- **Backend**: Lambda function and API logic
- **Deployment**: Scripts and configs for deploying to AWS
- **Data Preparation**: Scripts to prepare models and metadata
- **Visualizations**: Scripts to generate visualization files
- **Outputs**: Centralized location for all generated outputs

**Workflow**: `Data Preparation → Visualizations → Frontend/Backend → Deployment`

This organization provides:
- ✅ **Clear Organization**: Each directory has a single, clear purpose
- ✅ **Easy Maintenance**: Related files are grouped together
- ✅ **Better Onboarding**: New developers can understand structure quickly
- ✅ **Scalable**: Easy to add new features or visualization types

---

## Core Components

### Frontend (`frontend/`)

**Purpose**: User-facing dashboard interface

**Key Files**:
- `index.html` - Main dashboard with all tabs:
  - **Risk Assessment** - Calculate risk scores for **Opioid ED** or **Polypharmacy** (select cohort via tabs); both use full age bands (0-12 through 85-114)
  - **Causal Analysis** - FFA causal factors and SHAP importance with interactive charts
  - **DTW Trajectories** - Patient trajectory patterns, temporal metrics, and sample trajectories
  - **FP-Growth Patterns** - Frequent itemsets, support distributions, and co-occurrence networks
  - **BupaR Process Mining** - Process flows, activity frequencies, Gantt charts, and sequence patterns
  - **PGx Patient Card** - Generate pharmacogenomic cards from genetic variants

**Features**:
- Interactive forms with searchable dropdowns
- Real-time risk calculation
- Visual charts (Plotly.js)
- Responsive design

### Backend (`backend/`)

**Purpose**: Serverless API backend

**Key Files**:
- `lambda_function.py` - Main Lambda handler with all API endpoints
- `Dockerfile` - Container image configuration (ECR)
- `requirements.txt` - Python dependencies

**Features**:
- Model inference (ensemble: CatBoost + XGBoost + XGBoost RF)
- Metadata retrieval
- Visualization data serving
- PGx card generation

### Data Preparation (`data_preparation/`)

**Purpose**: Prepare data for dashboard deployment

**Scripts**:

1. **`prepare_models.py`** - Package models from Step 6 for Lambda deployment
   - **Input:** `6_final_model/outputs/{cohort}/{age_band}/`
   - **Output:** `10_risk_dashboard/outputs/models/`
   - **Function:** Extracts models, feature schemas, and MC-CV results for each cohort/age_band
   - **Configuration:** PGx cohorts (`opioid_ed`, `non_opioid_ed`) with correct age bands

2. **`generate_metadata.py`** - Extract valid codes from Feature Importance files
   - **Primary Input:** `3b_feature_importance_eda/outputs/{cohort}/{age_band}/cohort_feature_importance.csv` (prioritized)
   - **Fallback Input:** `3a_feature_importance/outputs/{cohort}_{age_band}_aggregated_feature_importance.csv`
   - **Output:** `10_risk_dashboard/outputs/metadata/metadata_{cohort}.json`
   - **Function:** Creates dropdown lists of valid ICD/CPT/Drug codes per cohort

3. **`prepare_cpic_data.py`** - Prepare CPIC data for PGx cards
   - **Input:** CPIC reference data
   - **Output:** `10_risk_dashboard/outputs/cpic/`
   - **Function:** Prepares pharmacogenomic drug-gene pairs

4. **`combine_shap_ffa_results.py`** - Combine SHAP and FFA analysis for consensus features
   - **Input:** Steps 7 and 8 outputs
   - **Output:** Combined feature importance analysis
   - **Function:** Identifies high-confidence features from both methods

**Outputs**: All saved to `outputs/` directory

### Visualizations (`visualizations/`)

**Purpose**: Generate visualization files (images, HTML) for dashboard tabs

**Subdirectories**:

1. **`dtw/`** - Dynamic Time Warping trajectory visualizations
   - **Scripts:** `create_dtw_features.py`, `create_dtw_visuals.py`
   - **Outputs:** Trajectory patterns, temporal metrics, sample trajectories

2. **`fpgrowth/`** - Frequent pattern mining visualizations
   - **Scripts:** `create_fpgrowth_visuals.py`, `create_plots.py`
   - **Outputs:** Frequent itemsets, support distributions, co-occurrence networks

3. **`bupar/`** - Process mining visualizations
   - **Scripts:** `create_bupar_visuals.py`, R scripts for process analysis
   - **Outputs:** Process flows, activity frequencies, Gantt charts, sequence patterns

**Outputs**: Saved to `outputs/visualizations/` and uploaded to S3

### Deployment (`deployment/`)

**Purpose**: Deployment automation

**Scripts**:
- `docker_build.sh` - Build and push Docker image to ECR
- `prepare_lambda_dir.py` - Prepare Lambda deployment package (optional)

**Process**:
1. Build Docker container with models and dependencies
2. Push to Amazon ECR (Elastic Container Registry)
3. Deploy Lambda function from ECR image
4. Configure API Gateway endpoints
5. Upload frontend to S3 and configure CloudFront

---

## Dashboard Features

### Visualization Tabs

The dashboard includes the following visualization tabs:

- **Causal Analysis Tab**: Displays FFA causal factors and SHAP importance with interactive charts
- **DTW Trajectories Tab**: Shows patient trajectory patterns, temporal metrics, and sample trajectories
- **FP-Growth Patterns Tab**: Displays frequent itemsets, support distributions, and co-occurrence networks
- **BupaR Process Mining Tab**: Shows process flows, activity frequencies, Gantt charts, and sequence patterns

### Data Preparation

**Model Preparation (`prepare_models.py`)**:
- Output directory: `10_risk_dashboard/outputs/models`
- Configured for PGx cohorts (`opioid_ed`, `non_opioid_ed`) with correct age bands
- Loads models from `6_final_model/outputs/{cohort}/{age_band}/`

**Metadata Generation (`generate_metadata.py`)**:
- Prioritizes Feature Importance EDA `cohort_feature_importance` files (refined features)
- Falls back to Step 3 `aggregated_feature_importance` files if Feature Importance EDA files not available
- Output directory: `10_risk_dashboard/outputs/metadata`
- Uses directory structure: `3b_feature_importance_eda/outputs/{cohort}/{age_band}/`

---

## Quick Start

### 1. Prepare Data for Dashboard

```bash
cd 10_risk_dashboard/data_preparation

# Generate metadata from feature importance files
python generate_metadata.py --all

# Package models for Lambda deployment
python prepare_models.py --all

# Prepare CPIC data for PGx cards
python prepare_cpic_data.py
```

### 2. Generate Visualizations (if not already done)

```bash
cd ../visualizations

# See individual READMEs for each visualization type:
# - visualizations/dtw/README.md
# - visualizations/fpgrowth/README.md
# - visualizations/bupar/README.md
```

### 3. Deploy Dashboard

```bash
cd ../deployment
./docker_build.sh
```

### Individual Component Documentation

- **Frontend**: [frontend/README.md](../10_risk_dashboard/frontend/README.md)
- **Backend**: [backend/README.md](../10_risk_dashboard/backend/README.md)
- **Deployment**: [deployment/README.md](../10_risk_dashboard/deployment/README.md)
- **Data Preparation**: [data_preparation/README.md](../10_risk_dashboard/data_preparation/README.md)
- **Visualizations**: [visualizations/README.md](../10_risk_dashboard/visualizations/README.md)

---

## Architecture

```text
User Browser → S3 Static Site → API Gateway → Lambda (ECR) → Models/Data
```

**Components**:

- **Frontend**: S3-hosted static website (`frontend/index.html`)
- **API Gateway**: RESTful API endpoints
- **Lambda Function**: Serverless backend (ECR container, up to 10GB)
- **Model Storage**: Models packaged in Lambda container (`/var/task/models/`)
- **Data Storage**: S3 for visualization images and large datasets

**Key Features:**

- **Ensemble Models**: CatBoost + XGBoost + XGBoost RF with performance-based weighting
- **Age-Based Selection**: Automatically selects appropriate model based on age
- **Feature-Driven Inputs**: Dropdowns populated from Feature Importance EDA refined feature importances
- **Privacy-First PGx Cards**: Anonymous, generic cards with optional patient ID
- **SHAP + FFA Combination**: Comprehensive patient-level explanations combining quantitative (SHAP) and logical (FFA) methods
- **Consensus Features**: High-confidence features identified by both SHAP and FFA analysis

---

## API Endpoints

### Core Endpoints

#### `GET /metadata`

Get valid age bands and valid codes for dropdowns

**Response:**
- Returns, per cohort, the supported age bands and code lists for the **Drugs / CPT / ICD** tabs
- The dashboard uses these to populate the cohort/age-band grid (e.g., 13-24, 25-44, 45-54, 65-74, 75-84, 85-114) and the tab-specific grids

#### `POST /risk`

Calculate risk score for a given `(cohort, age_band)` and selected codes

**Request Body:**
```json
{
  "cohort": "opioid_ed",
  "age_band": "25-44",
  "drugs": ["DRUG_NAME_1", "DRUG_NAME_2"],
  "icds": ["F1120", "R51"],
  "cpts": ["80305", "99213"]
}
```

**Response:**
- Lambda builds a feature vector using `feature_schema.json` (prepared by `prepare_models.py`)
- Returns ensemble risk plus per-model breakdown for visualization

#### `POST /risk/comparison`

Compare risk scenarios side-by-side

#### `POST /pgx/card`

Generate PGx patient card from genetic variants

### Visualization Endpoints

#### `GET /visualizations/causal`

Get causal analysis visualizations (FFA + SHAP)

**Query Parameters:**
- `cohort` - Cohort name (opioid_ed, non_opioid_ed)
- `age_band` - Age band (13-24, 25-44, etc.)

**Response:**
- Causal factors and SHAP importance data

#### `GET /visualizations/dtw`

Get DTW trajectory visualizations

**Query Parameters:**
- `cohort` - Cohort name
- `age_band` - Age band

**Response:**
- S3 paths to DTW visualization images

#### `GET /visualizations/fpgrowth`

Get FP-Growth pattern visualizations

**Query Parameters:**
- `cohort` - Cohort name
- `age_band` - Age band
- `item_type` - Item type (drug_name, icd_code, cpt_code, medical_code)

**Response:**
- S3 paths to FP-Growth visualization images

#### `GET /visualizations/bupar`

Get BupaR process mining visualizations

**Query Parameters:**
- `cohort` - Cohort name
- `age_band` - Age band

**Response:**
- S3 paths to BupaR visualization images

See [Step9_RiskDashboard/README_results_dashboard.md](Step9_RiskDashboard/README_results_dashboard.md) for complete API documentation.

---

## Data Sources

### Model Outputs

- **Location**: `6_final_model/outputs/{cohort}/{age_band}/`
- **Files**: Model JSONs, joblib files, feature schemas, MC-CV results

### SHAP Outputs

- **S3 Location**: `s3://pgxdatalake/gold/shap_analysis/{cohort}/{age_band}/`
- **Files**: `*_shap_global_importance_xgboost.csv`, `*_shap_sample_values_xgboost.parquet`

### FFA Outputs

- **S3 Location**: `s3://pgxdatalake/gold/ffa_analysis/{cohort}/{age_band}/xgboost/`
- **Files**: `causal_importance.parquet`, `feature_importance_axp.parquet`, `interaction_analysis.parquet`

### Visualization Outputs

**DTW Visualizations:**
- **S3 Location**: `s3://pgxdatalake/gold/feature_importance/{cohort}/{age_band}/plots/`
- **Files**: `dtw_trajectory_analysis_{cohort}_{age_band}.png`, `dtw_sample_trajectories_{cohort}_{age_band}.png`

**FP-Growth Visualizations:**
- **S3 Location**: `s3://pgxdatalake/gold/fpgrowth/{cohort}/{age_band}/plots/`
- **Files**: `*_top20_itemsets.png`, `*_itemset_support.png`, `*_network.html`

**BupaR Visualizations:**
- **S3 Location**: `s3://pgxdatalake/gold/feature_importance/{cohort}/{age_band}/plots/`
- **Files**: `*_overall_activity_frequency.png`, `*_gantt.png`, `*_activity_sequence_top.png`, etc.

**Note**: The Lambda function loads data from S3, so visualization files must be uploaded to S3 before deployment.

---

## FP-Growth Network Visualization Integration

**⚠️ Important**: FP-Growth features are **NOT** used in the final model due to target leakage concerns. However, FP-Growth network visualizations are valuable for **causal analysis and exploratory visualization** in the risk dashboard.

### Overview

FP-Growth network visualizations show:
- **Co-occurrence patterns**: Which drugs, ICD codes, or CPT codes frequently appear together
- **Association rules**: Directed relationships between items (antecedent → consequent)
- **Pattern strength**: Support, confidence, and lift metrics for patterns

### Integration with Causal Analysis

FP-Growth networks complement FFA/SHAP causal analysis by:
1. **Visualizing Feature Relationships**: Show how causal features (from FFA/SHAP) relate to each other
2. **Pattern Discovery**: Identify drug combinations or diagnostic patterns that align with high-importance features
3. **Patient Context**: Show which patterns a patient matches, providing clinical context for risk predictions

### Network Visualization Files

**Location**:
- **Local**: `10_risk_dashboard/visualizations/fpgrowth/outputs/{cohort}/{age_band}/plots/`
- **S3**: `s3://pgxdatalake/gold/fpgrowth/{cohort}/{age_band}/plots/`

**Files**:
- `{cohort}_{age_band}_{item_type}_network.html`: Interactive co-occurrence network
- `{cohort}_{age_band}_{item_type}_rules_network.html`: Interactive association rules network

**Item Types**: `drug_name`, `icd_code`, `cpt_code`, `medical_code`

### Dashboard Integration

#### Option 1: Embed HTML Network Files

```html
<!-- In dashboard HTML -->
<iframe 
  src="https://s3.amazonaws.com/pgxdatalake/gold/fpgrowth/{cohort}/{age_band}/plots/{cohort}_{age_band}_drug_name_network.html"
  width="100%" 
  height="600px"
  frameborder="0">
</iframe>
```

#### Option 2: Load via API Endpoint

```javascript
// In dashboard JavaScript
async function loadFPGrowthNetwork(cohort, ageBand, itemType) {
  const url = `https://s3.amazonaws.com/pgxdatalake/gold/fpgrowth/${cohort}/${ageBand}/plots/${cohort}_${ageBand}_${itemType}_network.html`;
  
  // Load and embed in dashboard
  const response = await fetch(url);
  const html = await response.text();
  document.getElementById('fpgrowth-network-container').innerHTML = html;
}
```

#### Option 3: Combine with Causal Analysis

```javascript
// Show FP-Growth network alongside FFA/SHAP results
function displayCausalAnalysis(patientData, ffaResults, shapResults) {
  // Display FFA/SHAP feature importance
  displayFeatureImportance(ffaResults, shapResults);
  
  // Load and display FP-Growth network for context
  loadFPGrowthNetwork(
    patientData.cohort,
    patientData.ageBand,
    'drug_name'  // or 'icd_code', 'cpt_code'
  );
  
  // Highlight features in network that match high-importance features
  highlightFeaturesInNetwork(
    getTopFeatures(ffaResults, shapResults, topN=20)
  );
}
```

### Network Features

**Interactive Controls**:
- **Node Centrality Filter**: Filter nodes by degree centrality (≥ 0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5)
- **Edge Support Filter**: Filter edges by support threshold
- **Edge Confidence Filter**: Filter edges by confidence (rules networks only)
- **Max Nodes Limit**: Limit display to top N nodes (20, 50, 100, 200, or All)
- **Reset Filters**: Clear all filters

**Visual Encoding**:
- **Node Size**: Represents degree centrality (how connected the node is)
- **Edge Width**: Represents support/confidence (strength of relationship)
- **Node Color**: Can be customized to highlight patient-matched items

### Use Cases

1. **Causal Analysis Visualization**
   - Show FP-Growth network alongside FFA/SHAP feature importance
   - Highlight features that appear in both analyses
   - Visualize relationships between high-importance features

2. **Patient-Specific Context**
   - Show which FP-Growth patterns a patient matches
   - Visualize patient's position in the network
   - Compare patient patterns to target cohort patterns

3. **Clinical Hypothesis Generation**
   - Explore drug combinations of interest
   - Discover diagnostic code patterns
   - Understand treatment sequences

### Related Documentation

- `10_risk_dashboard/visualizations/fpgrowth/README_visualization_only.md`: Why FP-Growth is visualization-only
- `10_risk_dashboard/visualizations/fpgrowth/README.md`: FP-Growth analysis documentation
- `8_ffa_analysis/README.md`: FFA analysis documentation (includes causal importance that reflects SHAP consensus)

---

## Build Process

### 1. Prepare Models

```bash
python 10_risk_dashboard/data_preparation/prepare_models.py --cohort opioid_ed
```

- **Reads from:** Step 6 outputs (`6_final_model/outputs/{cohort}/{age_band}/`)
- **Creates:** `10_risk_dashboard/outputs/models/{cohort}/{age_band}/`

### 2. Generate Metadata

```bash
python 10_risk_dashboard/data_preparation/generate_metadata.py --cohort opioid_ed
```

- **Reads from:** Step 3 outputs (or Step 3b Feature Importance EDA outputs)
- **Creates:** `10_risk_dashboard/outputs/metadata/metadata_{cohort}.json`

### 3. Build Dashboard

```bash
# See 10_risk_dashboard/deployment/ for build/deploy instructions
# Or use archived/utility_scripts/build_dashboard.sh if present
```

- Processes all available cohorts
- Skips missing cohorts gracefully

### 4. Deploy to AWS

```bash
cd 10_risk_dashboard/deployment
./docker_build.sh
```

- Builds Docker container with models and dependencies
- Pushes to Amazon ECR
- Deploys Lambda function
- Configures API Gateway
- Uploads frontend to S3

---

## Documentation

For detailed documentation, see [`Step9_RiskDashboard/`](Step9_RiskDashboard/):

### Main Documentation

- **[README_results_dashboard.md](Step9_RiskDashboard/README_results_dashboard.md)** - Complete dashboard system overview
- **[README_results_value_proposition.md](Step9_RiskDashboard/README_results_value_proposition.md)** - Business value and use cases
- **[README_results_deployment.md](Step9_RiskDashboard/README_results_deployment.md)** - Complete deployment guide (architecture, steps, security)
- **[README_results_prediction.md](Step9_RiskDashboard/README_results_prediction.md)** - Prediction workflow and technical details
- **[README_results_quickstart.md](Step9_RiskDashboard/README_results_quickstart.md)** - Quick start guide for predictions

### Feature Documentation

- **[README_results_pgx_card.md](Step9_RiskDashboard/README_results_pgx_card.md)** - PGx Patient Card feature
- **[README_results_ensemble.md](Step9_RiskDashboard/README_results_ensemble.md)** - Ensemble model approach
- **[README_results_model_weights.md](Step9_RiskDashboard/README_results_model_weights.md)** - Performance-based model weighting

### Deployment Guides

- **[README_results_deployment_ecr.md](Step9_RiskDashboard/README_results_deployment_ecr.md)** - Lambda ECR container deployment
- **[README_results_deployment_cpic.md](Step9_RiskDashboard/README_results_deployment_cpic.md)** - CPIC data deployment

### Reference

- **[README_results_storage.md](Step9_RiskDashboard/README_results_storage.md)** - Storage analysis and container sizing
- **[README_results_age_bands.md](Step9_RiskDashboard/README_results_age_bands.md)** - Supported age bands and mappings

### Technical Documentation

- **[README_checkpoints_and_workflow_resets.md](README_checkpoints_and_workflow_resets.md)** - Checkpoints, refresh mechanisms, and workflow resets
- **[README_file_resolver.md](README_file_resolver.md)** - Universal file resolution across storage layers
- **[README_feature_engineering_and_analysis.md](README_feature_engineering_and_analysis.md)** - Feature engineering pipeline

See [`Step9_RiskDashboard/README.md`](Step9_RiskDashboard/README.md) for complete documentation index.

---

## Summary

The Risk Assessment Dashboard provides:

✅ **Comprehensive Risk Prediction** - Ensemble models for opioid ED and polypharmacy across all age bands  
✅ **Explainable AI** - SHAP and FFA causal analysis for individual patient explanations  
✅ **Process Mining** - BupaR process flows and trajectory analysis  
✅ **Pattern Discovery** - FP-Growth networks for co-occurrence patterns  
✅ **Pharmacogenomics** - CPIC-based patient card generation  
✅ **Production-Ready** - Serverless architecture with ECR containers, API Gateway, and S3 hosting  
✅ **Scalable** - Lambda auto-scaling with up to 10GB container images  

**Next Steps:**
1. Complete all cohort/age_band combinations (Steps 1-8)
2. Prepare models and metadata (data_preparation/)
3. Generate visualizations (visualizations/)
4. Build and deploy dashboard (deployment/)
5. Access dashboard via S3/CloudFront URL
