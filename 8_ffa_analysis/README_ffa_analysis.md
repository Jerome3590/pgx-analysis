# Formal Feature Attribution (FFA) Analysis

## Overview

Formal Feature Attribution (FFA) Analysis provides a comprehensive framework for interpreting CatBoost models through symbolic logic extraction, anchored explanations, and causal analysis. This module transforms opaque gradient-boosted decision tree models into interpretable, analyzable symbolic rules suitable for formal verification and causal inference.

**Key Capabilities:**

- **Symbolic Rule Extraction**: Convert CatBoost tree structures into Boolean logic formulas
- **Anchored Explanations (AXP)**: Generate instance-level explanations using rule matching
- **Causal Analysis**: Measure causal responsibility of features through counterfactual analysis
- **Feature Importance**: Calculate importance scores from explanations and causal effects
- **Formal Verification**: Use SAT solvers for consistency checking and minimal explanation extraction

## Architecture

The FFA pipeline follows a three-phase architecture:

```text
┌─────────────────────────────────────────────────────────────┐
│ Phase I: Model Ingestion & Feature Mapping                  │
├─────────────────────────────────────────────────────────────┤
│ • Load CatBoost JSON model                                  │
│ • Parse features_info (float and categorical)              │
│ • Extract CTR (Counter-based Target Statistics) mappings    │
│ • Map feature indices to readable names                     │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ Phase II: Symbolic Logic Extraction                         │
├─────────────────────────────────────────────────────────────┤
│ • Convert tree paths to PySAT formulas                      │
│ • Build CNF (Conjunctive Normal Form) constraints          │
│ • Extract decision rules with conditions                    │
│ • Validate rule consistency using SAT solvers                │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ Phase III: Explanation & Analysis                           │
├─────────────────────────────────────────────────────────────┤
│ • Generate anchored explanations (AXP)                      │
│ • Calculate feature importance from explanations            │
│ • Perform causal analysis via counterfactuals               │
│ • Generate visualizations and reports                       │
└─────────────────────────────────────────────────────────────┘
```

## Key Components

### 1. Model Loading & Validation

- Loads CatBoost models from JSON format
- Validates model structure (trees, features_info, CTR data)
- Extracts feature metadata and mappings

### 2. CTR (Counter-based Target Statistics) Handling

CatBoost uses CTR transformations for categorical features, which require special handling:

- Extract CTR mappings from `ctr_data`
- Map hash values to categorical feature indices
- Resolve CTR split indices to feature names and borders

### 3. Symbolic Rule Extraction

- Converts oblivious tree structures to Boolean formulas
- Uses PySAT for CNF conversion and SAT solving
- Creates human-readable decision rules with conditions

### 4. Anchored Explanations (AXP)

- Matches instances to decision rules
- Generates explanations for target class predictions
- Tracks unmatched instances for coverage analysis

### 5. Causal Analysis

- Measures causal importance by modifying features
- Uses counterfactual reasoning (what-if scenarios)
- Calculates average prediction change per feature

### 6. Feature Importance

- **AXP-based**: Frequency of features in explanation conditions
- **Causal-based**: Average change in predictions when features are modified
- Normalized scores for comparison

## Files

### Notebooks

- **`catboost_feature_attribution_analysis.ipynb`**: Main Jupyter notebook for interactive FFA analysis
  - Complete workflow from model loading to visualization
  - Configurable analysis parameters
  - Step-by-step execution with detailed outputs

### Scripts

- **`ffa_analysis.py`**: Core FFA analysis functions
  - Model validation and structure inspection
  - CTR hash map analysis
  - Feature mapping extraction

- **`catboost_axp_explainer.py`**: Anchored Explanations implementation
  - `CatBoostAXPExplainer` class for generating explanations
  - Path configuration and analysis configuration classes
  - Rule matching and explanation generation

- **`catboost_axp_explainer2.py`**: Alternative explainer implementation (if needed)

## Quick Start

### Prerequisites

**Python Version:** Python 3.11+ (tested with Python 3.11.9 for EC2 compatibility)

```bash
pip install catboost pandas numpy matplotlib seaborn pysat boto3
```

### Basic Usage

1. **Configure paths** in the notebook:

   ```python
   MODEL_CONFIG = {
       'model_json_path': 'path/to/catboost_model.json',
       'model_cbm_path': 'path/to/model.cbm',  # Optional
       'model_info_json': 'path/to/model_info.json',  # Optional
   }
   ```

2. **Load model and generate explanations**:

   ```python
   # See catboost_feature_attribution_analysis.ipynb for complete workflow
   ```

3. **View results**:

   - Feature importance visualizations
   - Causal importance scores
   - Explanation summaries
   - Saved CSV/JSON outputs

## Workflow

1. **Model Ingestion**: Load and validate CatBoost JSON model
2. **Feature Mapping**: Extract and map feature names, CTR data
3. **Rule Extraction**: Convert trees to symbolic formulas
4. **Explanation Generation**: Match instances to rules (AXP)
5. **Causal Analysis**: Measure feature causal importance
6. **Visualization**: Create plots and summary reports
7. **Export**: Save results to CSV/JSON/Parquet

## Outputs

The analysis generates several outputs:

- **Feature Importance (AXP)**: CSV with feature importance scores from explanations
- **Causal Importance**: CSV with causal importance scores from counterfactual analysis
- **Explanations Summary**: CSV with per-instance explanation metadata
- **Visualizations**: PNG plots of feature importance and causal relationships
- **Model Info**: JSON with model metadata and metrics

## Key Concepts

### Anchored Explanations (AXP)

Anchored explanations match instances to decision rules that explain their predictions. Each explanation contains:

- **Matched rule**: The decision rule satisfied by the instance
- **Conditions**: Feature conditions that must be met
- **Prediction**: The rule's predicted outcome

### Causal Importance

Causal importance measures how much changing a feature affects the model's prediction:

- **Counterfactual**: Create modified instances (flip binary, shift numerical)
- **Prediction Change**: Measure change in predicted probability
- **Aggregation**: Average changes across instances

### CTR (Counter-based Target Statistics)

CatBoost transforms categorical features using CTR:

- **Hash Mapping**: Maps category values to hash codes
- **Borders**: Thresholds for CTR value discretization
- **Resolution**: Maps CTR splits back to original feature names

## References

- [CatBoost Model Export Tutorial](https://colab.research.google.com/github/catboost/tutorials/blob/master/model_analysis/model_export_as_json_tutorial.ipynb)
- [PySAT Documentation](https://pysathq.github.io/docs/pysat.pdf)
- [CatBoost Categorical Features](https://catboost.ai/docs/en/features/categorical-features)
- [Formal Methods for ML Interpretability](https://wires.onlinelibrary.wiley.com/doi/full/10.1002/widm.70015)

## Notes

- **Implementation Details**: See `catboost_feature_attribution_analysis.ipynb` for the complete implementation workflow
- **CTR Handling**: CTR mappings are complex; the notebook includes validation and debugging utilities
- **Performance**: Large models (1000+ trees) may require significant memory and computation time
- **S3 Support**: The notebook supports loading models and data from S3 when configured

---

**For detailed implementation steps and code examples, see the `catboost_feature_attribution_analysis.ipynb` notebook.**
