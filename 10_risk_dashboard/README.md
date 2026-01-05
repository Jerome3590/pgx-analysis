## 10_results – Final Results & Dashboard

This directory contains the **production-ready risk assessment dashboard** and deployment artifacts for the PGx analysis pipeline.

### Quick Overview

The dashboard provides two main capabilities:

1. **Risk Assessment Dashboard** - Predict opioid ED visit risk (ages 13-64) or polypharmacy risk (ages 65-114)
2. **PGx Patient Card Generator** - Generate pharmacogenomic cards from genetic variants

### Core Components

- **`index.html`** - Frontend dashboard (HTML/JavaScript)
- **`lambda_function.py`** - AWS Lambda handler (API endpoints)
- **`generate_metadata.py`** - Extract valid codes for dropdowns
- **`prepare_models.py`** - Package models for Lambda deployment
- **`prepare_cpic_data.py`** - Prepare CPIC data for PGx cards
- **`combine_shap_ffa_results.py`** - Combine SHAP and FFA analysis for consensus features
- **`Dockerfile`** - Container image for Lambda (ECR)
- **`requirements.txt`** - Python dependencies

### Quick Start

```bash
# 1. Generate metadata for dropdowns
python generate_metadata.py --all

# 2. Prepare models for deployment
python prepare_models.py --all

# 3. Prepare CPIC data
python prepare_cpic_data.py

# 4. Combine SHAP and FFA results (optional, for comprehensive explanations)
python combine_shap_ffa_results.py --cohort non_opioid_ed --age-band 65-74

# 5. Build Docker container
./docker_build.sh
```

### Documentation

For detailed documentation, see [`docs/Step10_Results/`](../docs/Step10_Results/):

**Main Documentation:**
- **[README_results_dashboard.md](../docs/Step10_Results/README_results_dashboard.md)** - Complete dashboard system overview
- **[README_results_value_proposition.md](../docs/Step10_Results/README_results_value_proposition.md)** - Business value and use cases
- **[README_results_deployment.md](../docs/Step10_Results/README_results_deployment.md)** - Complete deployment guide (architecture, steps, security)
- **[README_results_prediction.md](../docs/Step10_Results/README_results_prediction.md)** - Prediction workflow and technical details
- **[README_results_quickstart.md](../docs/Step10_Results/README_results_quickstart.md)** - Quick start guide for predictions

**Feature Documentation:**
- **[README_results_pgx_card.md](../docs/Step10_Results/README_results_pgx_card.md)** - PGx Patient Card feature
- **[README_results_ensemble.md](../docs/Step10_Results/README_results_ensemble.md)** - Ensemble model approach
- **[README_results_model_weights.md](../docs/Step10_Results/README_results_model_weights.md)** - Performance-based model weighting
- **[README_SHAP_FFA_COMBINATION.md](README_SHAP_FFA_COMBINATION.md)** - SHAP + FFA combination and consensus analysis

**Deployment Guides:**
- **[README_results_deployment_ecr.md](../docs/Step10_Results/README_results_deployment_ecr.md)** - Lambda ECR container deployment
- **[README_results_deployment_cpic.md](../docs/Step10_Results/README_results_deployment_cpic.md)** - CPIC data deployment

**Reference:**
- **[README_results_storage.md](../docs/Step10_Results/README_results_storage.md)** - Storage analysis and container sizing
- **[README_results_age_bands.md](../docs/Step10_Results/README_results_age_bands.md)** - Supported age bands and mappings

See [`docs/Step10_Results/README.md`](../docs/Step10_Results/README.md) for complete documentation index.

### Architecture

```
User Browser → S3 Static Site → API Gateway → Lambda (ECR) → Models/Data
```

### Key Features

- **Ensemble Models**: CatBoost + XGBoost + XGBoost RF with performance-based weighting
- **Age-Based Selection**: Automatically selects appropriate model based on age
- **Feature-Driven Inputs**: Dropdowns populated from actual feature importances
- **Privacy-First PGx Cards**: Anonymous, generic cards with optional patient ID
- **SHAP + FFA Combination**: Comprehensive patient-level explanations combining quantitative (SHAP) and logical (FFA) methods
- **Consensus Features**: High-confidence features identified by both SHAP and FFA analysis
- **FP-Growth Network Visualizations**: Interactive co-occurrence networks for causal analysis (see [FP-Growth Visualization Integration](#fp-growth-network-visualization-integration))

### API Endpoints

- `GET /metadata` - Get valid age bands and valid codes for dropdowns.
  - Returns, per cohort, the supported age bands and code lists for the **Drugs / CPT / ICD** tabs.
  - The dashboard uses these to populate the cohort grid (e.g., 13-24, 25-44, 45-54, 65-74, 75-84, 85-94) and the tab-specific grids.
- `POST /risk` - Calculate risk score for a given `(cohort, age_band)` and selected codes.
  - Dashboard sends a JSON body like:
    ```json
    {
      "cohort": "opioid_ed",
      "age_band": "25-44",
      "drugs": ["DRUG_NAME_1", "DRUG_NAME_2"],
      "icds": ["F1120", "R51"],
      "cpts": ["80305", "99213"]
    }
    ```
  - Lambda builds a feature vector using `feature_schema.json` (prepared by `prepare_models.py`) and returns ensemble risk plus per-model breakdown for visualization.
- `POST /risk/comparison` - Compare risk scenarios
- `POST /pgx/card` - Generate PGx card

See [README_results_dashboard.md](../docs/Step10_Results/README_results_dashboard.md) for complete API documentation.

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

**Location:**
- Local: `10b_fpgrowth_dashboard_visual/outputs/{cohort}/{age_band}/plots/`
- S3: `s3://pgxdatalake/gold/fpgrowth/{cohort}/{age_band}/plots/`

**Files:**
- `{cohort}_{age_band}_{item_type}_network.html`: Interactive co-occurrence network
- `{cohort}_{age_band}_{item_type}_rules_network.html`: Interactive association rules network

**Item Types:** `drug_name`, `icd_code`, `cpt_code`, `medical_code`

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

**Interactive Controls:**
- **Node Centrality Filter**: Filter nodes by degree centrality (≥ 0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5)
- **Edge Support Filter**: Filter edges by support threshold
- **Edge Confidence Filter**: Filter edges by confidence (rules networks only)
- **Max Nodes Limit**: Limit display to top N nodes (20, 50, 100, 200, or All)
- **Reset Filters**: Clear all filters

**Visual Encoding:**
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

- `10b_fpgrowth_dashboard_visual/README_VISUALIZATION_ONLY.md`: Why FP-Growth is visualization-only
- `10b_fpgrowth_dashboard_visual/README.md`: FP-Growth analysis documentation
- `9_combined_shap_ffa/README.md`: Combined causal analysis documentation
