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

**Important**: FFA analysis is performed **only for XGBoost models**:

- **XGBoost FFA**: JSON is used directly to extract rules from XGBoost model structure
- **CatBoost FFA**: **NOT performed** due to CatBoost's complex hashing and CTR (Counter-based Target Statistics) for categorical variables that make direct rule extraction difficult
- **CatBoost SHAP**: Used for feature importance filtering (not for FFA rule extraction)

The workflow uses SHAP values from both models to filter and prioritize rules:

1. **XGBoost FFA**: Extracts all possible rules directly from XGBoost JSON model structure
2. **SHAP-Augmented Rule Filtering**: Uses SHAP importance values from **both XGBoost and CatBoost** (required from Step 7) to filter and prioritize rules for AXP computation
3. **AXP Computation**: Computes AXP explanations from the SHAP-filtered rule set
4. **Causal Analysis**: Filters the final rule set based on causal importance scores

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
│ • Hybrid filtering: top-K OR percentile threshold          │
│   - Top 300 rules by SHAP score                             │
│   - OR all rules above 10th percentile (whichever is larger)│
│ • Rule selection: union of (1) first 100,                  │
│   (2) random 100, and (3) top SHAP rules (300 or percentile)│
│ • Limits to ~300-500 unique rules for efficient AXP        │
│ • SHAP values are REQUIRED - raises error if missing       │
│ • Balances performance (limits rules) with coverage        │
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
- **Why CatBoost FFA is not performed**: CatBoost's complex JSON hashing and CTR make direct rule extraction difficult. Instead, CatBoost SHAP values are used for feature importance filtering in XGBoost FFA.

### 3. Symbolic Rule Extraction

**Model-Specific Approaches:**

- **XGBoost**: JSON is parsed directly to extract tree structures and convert them to Boolean formulas
- **CatBoost**: FFA analysis is **NOT performed** due to complex JSON hashing and CTR. CatBoost SHAP values are used for feature importance filtering in XGBoost FFA instead.

**Common Process:**

- Converts tree structures to Boolean formulas
- Uses PySAT for CNF conversion and SAT solving
- Creates human-readable decision rules with conditions
- **Extracts ALL possible rules** from the model (not filtered at this stage)

### 4. SHAP-Augmented Rule Prioritization (REQUIRED)

- **SHAP values are required**: Loads SHAP importance scores from Step 7 (SHAP Analysis) for **both XGBoost and CatBoost**
- **Model-Specific Usage**:
  - **XGBoost FFA**: Rules are extracted directly from XGBoost JSON, then SHAP filters/prioritizes them
  - **CatBoost FFA**: **NOT performed** - CatBoost's complex hashing and CTR make direct rule extraction difficult
  - **CatBoost SHAP**: Used for feature importance filtering in XGBoost FFA (not for CatBoost FFA rule extraction)
- **Rule Scoring**: Each rule is scored by summing the SHAP importance values of all features in the rule
  - Uses SHAP importance from **both XGBoost and CatBoost** to filter/prioritize rules
- **Hybrid Filtering Strategy**: Uses top-K + percentile threshold to balance performance and coverage:
  - **Top-K strategy**: Takes top 300 rules by SHAP score (globally important rules)
  - **Percentile threshold**: Also includes all rules above 10th percentile (safety net for important rules)
  - **Selection**: Uses whichever set is larger (ensures coverage while limiting rule count)
  - **Result**: Limits to ~300-500 unique rules for efficient AXP computation
- **Rule Selection Logic**: For each instance, combines three sets via union:
  1. **First 100 matched rules**: Common patterns (order-based coverage)
  2. **Random 100 matched rules**: Diversity through sampling (seed=42 for reproducibility)
  3. **Top SHAP rules**: Top 300 by SHAP score OR all above 10th percentile (whichever is larger)
     - Uses SHAP importance from both XGBoost and CatBoost
  4. **Final rule set for AXP**: Union of all three sets (deduplicated) → ~300-500 unique rules
- **Error Handling**: Raises `FileNotFoundError` or `ValueError` if SHAP data is missing or malformed
- **Key Points**:

  - **SHAP filters and prioritizes rules** - uses hybrid top-K + percentile approach to limit rule count
  - **Performance optimization**: Limits rule sets to ~300-500 unique rules (down from potentially thousands)
  - **Coverage**: Percentile threshold ensures we don't miss important rules that rank lower globally
  - **Trade-off**: Prioritizes most important rules; may miss rare variants (acceptable for performance)
  - **Causal analysis filters the final rule set** based on causal importance scores
  - **XGBoost FFA only**: FFA analysis is performed only for XGBoost models
  - **CatBoost SHAP**: Used for feature importance filtering, not for FFA rule extraction

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

#### 6.1 Single-Feature Causal Analysis

- Tests individual features one at a time
- Modifies each feature and measures explanation change
- Calculates causal importance per feature

#### 6.2 Multi-Feature Interaction Analysis

- **Purpose**: Tests combinations of features (pairs, triplets, etc.) to detect synergies/antagonisms
- **Feature Selection**: Only includes features with **ANY** importance > 0:
  - SHAP importance > 0 (model-level), OR
  - FFA importance > 0 (explanation-based), OR
  - Causal importance > 0 (individual causal effect)
- **Combinatorial Explosion Reduction**: 
  - Without filtering: 11,060 features → 61 million pairs (impossible!)
  - With filtering: ~20-100 important features → 190-4,950 pairs (manageable)
  - **99.5%+ reduction** in feature count, enabling comprehensive interaction testing
- **Combination Testing**: 
  - Generates all combinations of important features (pairs, triplets, etc.)
  - Tests each combination by modifying all features simultaneously
  - Calculates interaction effect = combined_effect - sum_individual_effects
  - Detects synergies (positive interaction) and antagonisms (negative interaction)
- **No Max Limit**: Tests ALL combinations of important features (no arbitrary cutoff)
- **Output**: `interaction_analysis.parquet` with columns:
  - `feature_combination`: Feature names joined by "|" (e.g., "drug_A|drug_B")
  - `interaction_size`: Number of features in combination (2, 3, etc.)
  - `combined_causal_importance`: Combined effect when all features modified
  - `sum_individual_effects`: Sum of individual univariate effects
  - `interaction_effect`: Difference (combined - individual), measures synergy/antagonism
  - `synergy_type`: positive/negative/neutral

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
5. **Rule Filtering & Prioritization**: Filter rules using hybrid approach:
   - Top 300 rules by SHAP score OR all rules above 10th percentile (whichever is larger)
   - Union with first 100 + random 100 matched rules
   - Limits to ~300-500 unique rules for efficient AXP computation
6. **Explanation Generation**: Match instances to SHAP-filtered rules and compute AXP
7. **Causal Analysis**: Measure feature causal importance and filter the final rule set
8. **Visualization**: Create plots and summary reports
9. **Export**: Save results to CSV/JSON/Parquet

**Critical Dependency**: Step 7 (SHAP Analysis) must run before Step 8 (FFA Analysis). FFA requires SHAP importance values to augment and prioritize rules. The system will raise an error if SHAP data is not available.

**Important Distinction**:

- **SHAP filters and prioritizes** rules for AXP computation using hybrid top-K + percentile approach
  - Top 300 rules by SHAP score OR all above 10th percentile (whichever is larger)
  - Limits rule count to ~300-500 for performance while ensuring coverage
  - May miss rare variants (acceptable trade-off for performance)
- **Causal analysis filters** the final rule set based on causal importance scores

## Outputs

The analysis generates several outputs:

- **Feature Importance (AXP)**: Parquet/CSV with feature importance scores from explanations
- **Causal Importance**: Parquet with causal importance scores from counterfactual analysis
  - Includes `causal_importance` (IR - Intervention Rate), `support` (Support - number of intervenable instances), and `confidence` (fraction that changed)
  - See [`8_ffa_analysis/SUPPORT_CONFIDENCE_METRICS.md`](../../8_ffa_analysis/SUPPORT_CONFIDENCE_METRICS.md) for detailed explanation
- **Explanations Summary**: Parquet/CSV with per-instance explanation metadata
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
