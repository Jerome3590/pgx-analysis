# Formal Feature Attribution (FFA) Analysis

## Overview

Formal Feature Attribution (FFA) Analysis provides a comprehensive framework for interpreting gradient-boosted decision tree models (CatBoost, XGBoost) through symbolic logic extraction, anchored explanations, and causal analysis. This module transforms opaque models into interpretable, analyzable symbolic rules suitable for formal verification and causal inference.

**Key Capabilities:**

- **Symbolic Rule Extraction**: Convert tree structures into Boolean logic formulas
- **SHAP-Augmented Rule Prioritization**: Uses SHAP importance values (from Step 7) to augment and prioritize rules for AXP computation (not to filter them out)
- **Anchored Explanations (AXP)**: Generate instance-level explanations using rule matching with SHAP-prioritized rule sets
- **Causal Analysis**: Measure causal responsibility of features through counterfactual analysis and filter the final rule set
- **Feature Importance**: Calculate importance scores from explanations and causal effects
- **Formal Verification**: Use SAT solvers for consistency checking and minimal explanation extraction

**Important**: The rule extraction process differs by model type:

- **XGBoost**: JSON is used directly to extract rules (no SHAP intermediary needed for rule extraction)
- **CatBoost**: Uses SHAP values as a translation layer due to complex JSON hashing and CTR (Counter-based Target Statistics) that make direct conversion difficult

For both model types, SHAP values are then used to augment and prioritize rules (not filter them out):

1. Extracts all possible rules from the model JSON (directly for XGBoost, via SHAP translation for CatBoost)
2. Uses SHAP importance values (required from Step 7) to augment and prioritize rules for AXP computation
3. Computes AXP explanations from the SHAP-prioritized rule set
4. Causal analysis filters the final rule set based on causal importance

## Architecture

The FFA pipeline follows a four-phase architecture with SHAP as an intermediary:

```text
┌─────────────────────────────────────────────────────────────┐
│ Phase I: Model Ingestion & Feature Mapping                  │
├─────────────────────────────────────────────────────────────┤
│ • Load model JSON (CatBoost/XGBoost)                        │
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
│ • Extract ALL decision rules with conditions                │
│ • Validate rule consistency using SAT solvers                │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ Phase II.5: SHAP-Augmented Rule Prioritization (REQUIRED)   │
├─────────────────────────────────────────────────────────────┤
│ • Load SHAP importance values from Step 7                   │
│ • Score each rule by summing SHAP values of its features    │
│ • Augment rule selection: union of (1) first 100,          │
│   (2) random 100, and (3) all rules with SHAP > 0          │
│ • SHAP values are REQUIRED - raises error if missing       │
│ • Note: SHAP augments/prioritizes rules, does not filter    │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ Phase III: Explanation & Analysis                           │
├─────────────────────────────────────────────────────────────┤
│ • Generate anchored explanations (AXP) from prioritized rules│
│ • Calculate feature importance from explanations            │
│ • Perform causal analysis via counterfactuals               │
│ • Causal analysis filters the final rule set                │
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
- **Why SHAP is needed**: CatBoost's complex JSON hashing and CTR make direct rule extraction difficult. SHAP values act as a translation layer to bridge the gap between the complex JSON structure and interpretable rules.

### 3. Symbolic Rule Extraction

**Model-Specific Approaches:**

- **XGBoost**: JSON is parsed directly to extract tree structures and convert them to Boolean formulas
- **CatBoost**: Due to complex JSON hashing and CTR, SHAP values are used as a translation layer to extract rules from the model structure

**Common Process:**

- Converts tree structures to Boolean formulas
- Uses PySAT for CNF conversion and SAT solving
- Creates human-readable decision rules with conditions
- **Extracts ALL possible rules** from the model (not filtered at this stage)

### 4. SHAP-Augmented Rule Prioritization (REQUIRED)

- **SHAP values are required**: Loads SHAP importance scores from Step 7 (SHAP Analysis)
- **For CatBoost**: SHAP values serve a dual purpose:

  1. **Translation layer**: Helps extract rules from complex JSON hashing and CTR structures
  2. **Prioritization**: Scores and prioritizes rules for AXP computation

- **For XGBoost**: SHAP values are used only for prioritization (rules are extracted directly from JSON)
- **Rule Scoring**: Each rule is scored by summing the SHAP importance values of all features in the rule
- **Rule Selection Logic**: For each instance, augments rule selection using a three-set union approach:
  1. **First 100 rules**: Takes the first 100 matched rules (order-based coverage)
  2. **Random 100 rules**: Takes a random sample of 100 matched rules (diversity through sampling, seed=42)
  3. **SHAP-augmented rules**: Includes all rules where the sum of SHAP importance > 0 (SHAP-important rules)
  4. **Final rule set for AXP**: Union of all three sets (deduplicated) for AXP computation
- **Error Handling**: Raises `FileNotFoundError` or `ValueError` if SHAP data is missing or malformed
- **Key Point**: 
  - **SHAP augments/prioritizes rules** - it does not filter them out. All matched rules are considered.
  - **Causal analysis filters the final rule set** based on causal importance scores.
  - **XGBoost**: Rules are extracted directly from JSON, then SHAP prioritizes them
  - **CatBoost**: SHAP values act as a translation layer to extract rules from complex JSON/CTR, then prioritizes them

### 5. Anchored Explanations (AXP)

- Matches instances to decision rules from the SHAP-prioritized rule set
- Generates explanations for target class predictions using the prioritized rules
- Tracks unmatched instances for coverage analysis
- Uses SAT solvers (Hitman) to compute minimal hitting sets (AXP) from the prioritized rule set

### 6. Causal Analysis

- Measures causal importance by modifying features
- Uses counterfactual reasoning (what-if scenarios)
- Calculates average prediction change per feature
- **Filters the final rule set** based on causal importance scores
- Only rules with significant causal impact are retained in the final output

### 7. Feature Importance

- **AXP-based**: Frequency of features in explanation conditions (from SHAP-prioritized rules)
- **Causal-based**: Average change in predictions when features are modified (from causal-filtered rule set)
- Normalized scores for comparison

## Files

### Notebooks

- **`catboost_feature_attribution_analysis.ipynb`**: Main Jupyter notebook for interactive FFA analysis
  - Complete workflow from model loading to visualization
  - Configurable analysis parameters
  - Step-by-step execution with detailed outputs

### Scripts

- **`run_full_ffa_analysis.py`**: Main script to run complete FFA analysis workflow
  - Loads models (CatBoost, XGBoost, XGBoost RF)
  - Extracts rules using unified schema
  - Generates AXP explanations
  - Calculates feature importance
  - Performs causal analysis

- **`validate_xgboost_rules_vs_shap.py`**: Validation script to compare XGBoost JSON-extracted rules with SHAP values
  - Validates that SHAP values can accurately augment and prioritize the rule set for causal analysis
  - Demonstrates that rules extracted from JSON align well with SHAP importance patterns
  - Confirms that SHAP-augmented rule prioritization (three-set union approach) produces meaningful results
  - Generates correlation statistics and visualization plots
  - Usage: `python validate_xgboost_rules_vs_shap.py --cohort opioid_ed --age-band 13-24`

- **`ffa_analysis.py`**: Core FFA analysis functions
  - Model validation and structure inspection
  - CTR hash map analysis
  - Feature mapping extraction

- **`catboost_axp_explainer.py`**: CatBoost AXP explainer implementation
  - `CatBoostSymbolicExplainer` class for generating explanations
  - Uses SHAP values as translation layer for rule extraction

- **`xgboost_axp_explainer.py`**: XGBoost AXP explainer implementation
  - `XGBoostSymbolicExplainer` class for generating explanations
  - Extracts rules directly from JSON (no SHAP translation layer needed)

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

1. **Model Ingestion**: Load and validate model JSON (CatBoost/XGBoost)
2. **Feature Mapping**: Extract and map feature names, CTR data
3. **Rule Extraction**: Convert trees to symbolic formulas (extracts ALL rules)
4. **SHAP Loading**: Load SHAP importance values from Step 7 (REQUIRED)
5. **Rule Prioritization**: Augment rule selection using SHAP importance (union of first 100 + random 100 + all SHAP > 0)
6. **Explanation Generation**: Match instances to SHAP-prioritized rules and compute AXP
7. **Causal Analysis**: Measure feature causal importance and filter the final rule set
8. **Visualization**: Create plots and summary reports
9. **Export**: Save results to CSV/JSON/Parquet

**Critical Dependency**: Step 7 (SHAP Analysis) must run before Step 8 (FFA Analysis). FFA requires SHAP importance values to augment and prioritize rules. The system will raise an error if SHAP data is not available.

**Important Distinction**: 
- **SHAP augments/prioritizes** rules for AXP computation (does not filter them out)
- **Causal analysis filters** the final rule set based on causal importance scores

## Outputs

The analysis generates several outputs:

- **Feature Importance (AXP)**: CSV with feature importance scores from explanations
- **Causal Importance**: CSV with causal importance scores from counterfactual analysis
- **Explanations Summary**: CSV with per-instance explanation metadata
- **Visualizations**: PNG plots of feature importance and causal relationships
- **Model Info**: JSON with model metadata and metrics

## Key Concepts

### Anchored Explanations (AXP)

Anchored explanations match instances to SHAP-prioritized decision rules that explain their predictions. Each explanation contains:

- **Matched rule**: The decision rule satisfied by the instance (from the SHAP-prioritized rule set)
- **Conditions**: Feature conditions that must be met
- **Prediction**: The rule's predicted outcome

AXP explanations are computed from the prioritized rule set, using SHAP importance to augment rule selection. The final rule set is filtered by causal analysis based on causal importance scores.

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
