# Step 9: Risk Dashboard

This directory contains the **production-ready risk assessment dashboard** and deployment artifacts for the PGx analysis pipeline.

## Quick Overview

The dashboard provides multiple capabilities:

1. **Risk Assessment Dashboard** - Predict opioid ED visit risk (ages 13-64) or polypharmacy risk (ages 65-114)
2. **Causal Analysis** - Explore FFA causal factors and SHAP importance
3. **DTW Trajectories** - View patient trajectory patterns
4. **FP-Growth Patterns** - Explore frequent itemsets and association rules
5. **BupaR Process Mining** - View process flows and activity sequences
6. **PGx Patient Card Generator** - Generate pharmacogenomic cards from genetic variants

## Directory Structure

```text
9_risk_dashboard/
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
│   │   ├── create_dtw_visualizations.py
│   │   ├── create_predictive_time_features.py
│   │   ├── add_dtw_features_to_model_data.py
│   │   ├── dtw_cohort_runner.ipynb
│   │   ├── DTW_FEATURE_ANALYSIS.md
│   │   └── outputs/                   # DTW visualization outputs
│   ├── fpgrowth/                      # FP-Growth pattern visualizations
│   │   ├── run_analysis.py
│   │   ├── create_plots.py
│   │   ├── create_fpgrowth_features.py
│   │   ├── cohort_fpgrowth.py
│   │   ├── global_fpgrowth.py
│   │   └── outputs/                   # FP-Growth visualization outputs
│   ├── bupar/                         # BupaR process mining visualizations
│   │   ├── run_analysis.py
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

## Core Components

### Frontend (`frontend/`)

**Purpose**: User-facing dashboard interface

**Key Files**:
- `index.html` - Main dashboard with all tabs:
  - **Risk Assessment** - Calculate risk scores for opioid ED visits (ages 13-64) or polypharmacy risk (ages 65-114)
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
- `prepare_models.py` - Package models from `6_final_model/outputs/{cohort}/{age_band}/` for Lambda deployment
  - Output directory: `9_risk_dashboard/outputs/models`
  - Configured for PGx cohorts (`opioid_ed`, `non_opioid_ed`) with correct age bands
- `generate_metadata.py` - Extract valid codes from Step 3b `cohort_feature_importance` files
  - Prioritizes Step 3b refined features from `3b_feature_importance_eda/outputs/{cohort}/{age_band}/`
  - Falls back to Step 3 aggregated features if Step 3b files not available
  - Output directory: `9_risk_dashboard/outputs/metadata`
- `prepare_cpic_data.py` - Prepare CPIC data for PGx cards
- `combine_shap_ffa_results.py` - Combine SHAP and FFA analysis for consensus features

**Outputs**: All saved to `outputs/` directory

### Visualizations (`visualizations/`)

**Purpose**: Generate visualization files (images, HTML) for dashboard tabs

**Subdirectories**:
- `dtw/` - Dynamic Time Warping trajectory visualizations
- `fpgrowth/` - Frequent pattern mining visualizations
- `bupar/` - Process mining visualizations

**Outputs**: Saved to `outputs/visualizations/` and uploaded to S3

### Deployment (`deployment/`)

**Purpose**: Deployment automation

**Scripts**:
- `docker_build.sh` - Build and push Docker image to ECR
- `prepare_lambda_dir.py` - Prepare Lambda deployment package (optional, for manual preparation)

## Dashboard Features

### Visualization Tabs

The dashboard includes the following visualization tabs:

- **Causal Analysis Tab**: Displays FFA causal factors and SHAP importance with interactive charts
- **DTW Trajectories Tab**: Shows patient trajectory patterns, temporal metrics, and sample trajectories
- **FP-Growth Patterns Tab**: Displays frequent itemsets, support distributions, and co-occurrence networks
- **BupaR Process Mining Tab**: Shows process flows, activity frequencies, Gantt charts, and sequence patterns

### Data Preparation

**Model Preparation (`prepare_models.py`)**:
- Output directory: `9_risk_dashboard/outputs/models`
- Configured for PGx cohorts (`opioid_ed`, `non_opioid_ed`) with correct age bands
- Loads models from `6_final_model/outputs/{cohort}/{age_band}/`

**Metadata Generation (`generate_metadata.py`)**:
- Prioritizes Step 3b `cohort_feature_importance` files (refined features)
- Falls back to Step 3 `aggregated_feature_importance` files if Step 3b files not available
- Output directory: `9_risk_dashboard/outputs/metadata`
- Uses directory structure: `3b_feature_importance_eda/outputs/{cohort}/{age_band}/`

## Quick Start

### 1. Prepare Data for Dashboard

```bash
cd data_preparation
python generate_metadata.py --all
python prepare_models.py --all
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

- **Frontend**: `frontend/README.md`
- **Backend**: `backend/README.md`
- **Deployment**: `deployment/README.md`
- **Data Preparation**: `data_preparation/README.md`
- **Visualizations**: `visualizations/README.md`

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

## Key Features

- **Ensemble Models**: CatBoost + XGBoost + XGBoost RF with performance-based weighting
- **Age-Based Selection**: Automatically selects appropriate model based on age
- **Feature-Driven Inputs**: Dropdowns populated from Step 3b refined feature importances
- **Privacy-First PGx Cards**: Anonymous, generic cards with optional patient ID
- **SHAP + FFA Combination**: Comprehensive patient-level explanations combining quantitative (SHAP) and logical (FFA) methods
- **Consensus Features**: High-confidence features identified by both SHAP and FFA analysis
- **Visualization Tabs**:
  - **Causal Analysis**: FFA causal factors and SHAP importance
  - **DTW Trajectories**: Patient trajectory patterns and temporal metrics
  - **FP-Growth Patterns**: Frequent itemsets, association rules, and co-occurrence networks
  - **BupaR Process Mining**: Process flows, activity sequences, and Gantt charts

## API Endpoints

### Core Endpoints

- **`GET /metadata`** - Get valid age bands and valid codes for dropdowns
  - Returns, per cohort, the supported age bands and code lists for the **Drugs / CPT / ICD** tabs
  - The dashboard uses these to populate the cohort grid (e.g., 13-24, 25-44, 45-54, 65-74, 75-84, 85-94) and the tab-specific grids

- **`POST /risk`** - Calculate risk score for a given `(cohort, age_band)` and selected codes
  - Dashboard sends a JSON body:
    ```json
    {
      "cohort": "opioid_ed",
      "age_band": "25-44",
      "drugs": ["DRUG_NAME_1", "DRUG_NAME_2"],
      "icds": ["F1120", "R51"],
      "cpts": ["80305", "99213"]
    }
    ```
  - Lambda builds a feature vector using `feature_schema.json` (prepared by `prepare_models.py`) and returns ensemble risk plus per-model breakdown for visualization

- **`POST /risk/comparison`** - Compare risk scenarios

- **`POST /pgx/card`** - Generate PGx patient card from genetic variants

### Visualization Endpoints

- **`GET /visualizations/causal`** - Get causal analysis visualizations (FFA + SHAP)
  - Query params: `cohort`, `age_band`
  - Returns: Causal factors and SHAP importance data

- **`GET /visualizations/dtw`** - Get DTW trajectory visualizations
  - Query params: `cohort`, `age_band`
  - Returns: S3 paths to DTW visualization images

- **`GET /visualizations/fpgrowth`** - Get FP-Growth pattern visualizations
  - Query params: `cohort`, `age_band`, `item_type`
  - Returns: S3 paths to FP-Growth visualization images

- **`GET /visualizations/bupar`** - Get BupaR process mining visualizations
  - Query params: `cohort`, `age_band`
  - Returns: S3 paths to BupaR visualization images

See [README_results_dashboard.md](../docs/Step10_Results/README_results_dashboard.md) for complete API documentation.

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

**DTW Visualizations**:
- **S3 Location**: `s3://pgxdatalake/gold/feature_importance/{cohort}/{age_band}/plots/`
- **Files**: `dtw_trajectory_analysis_{cohort}_{age_band}.png`, `dtw_sample_trajectories_{cohort}_{age_band}.png`

**FP-Growth Visualizations**:
- **S3 Location**: `s3://pgxdatalake/gold/fpgrowth/{cohort}/{age_band}/plots/`
- **Files**: `*_top20_itemsets.png`, `*_itemset_support.png`, `*_network.html`

**BupaR Visualizations**:
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
- Local: `9_risk_dashboard/visualizations/fpgrowth/outputs/{cohort}/{age_band}/plots/`
- S3: `s3://pgxdatalake/gold/fpgrowth/{cohort}/{age_band}/plots/`

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

- `visualizations/fpgrowth/README_VISUALIZATION_ONLY.md`: Why FP-Growth is visualization-only
- `visualizations/fpgrowth/README.md`: FP-Growth analysis documentation
- `8_ffa_analysis/README.md`: FFA analysis documentation (includes causal importance that reflects SHAP consensus)

## Documentation

For detailed documentation, see [`docs/Step10_Results/`](../docs/Step10_Results/):

**Main Documentation**:
- **[README_results_dashboard.md](../docs/Step10_Results/README_results_dashboard.md)** - Complete dashboard system overview
- **[README_results_value_proposition.md](../docs/Step10_Results/README_results_value_proposition.md)** - Business value and use cases
- **[README_results_deployment.md](../docs/Step10_Results/README_results_deployment.md)** - Complete deployment guide (architecture, steps, security)
- **[README_results_prediction.md](../docs/Step10_Results/README_results_prediction.md)** - Prediction workflow and technical details
- **[README_results_quickstart.md](../docs/Step10_Results/README_results_quickstart.md)** - Quick start guide for predictions

**Feature Documentation**:
- **[README_results_pgx_card.md](../docs/Step10_Results/README_results_pgx_card.md)** - PGx Patient Card feature
- **[README_results_ensemble.md](../docs/Step10_Results/README_results_ensemble.md)** - Ensemble model approach
- **[README_results_model_weights.md](../docs/Step10_Results/README_results_model_weights.md)** - Performance-based model weighting

**Deployment Guides**:
- **[README_results_deployment_ecr.md](../docs/Step10_Results/README_results_deployment_ecr.md)** - Lambda ECR container deployment
- **[README_results_deployment_cpic.md](../docs/Step10_Results/README_results_deployment_cpic.md)** - CPIC data deployment

**Reference**:
- **[README_results_storage.md](../docs/Step10_Results/README_results_storage.md)** - Storage analysis and container sizing
- **[README_results_age_bands.md](../docs/Step10_Results/README_results_age_bands.md)** - Supported age bands and mappings

See [`docs/Step10_Results/README.md`](../docs/Step10_Results/README.md) for complete documentation index.
