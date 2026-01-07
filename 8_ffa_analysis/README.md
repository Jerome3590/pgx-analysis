## 8_ffa_analysis – Formal Feature Attribution Analysis

This directory contains the **Formal Feature Attribution (FFA) Analysis** framework for interpreting gradient-boosted decision tree models through symbolic logic extraction, anchored explanations, and causal analysis.

### Quick Overview

FFA Analysis transforms opaque models into interpretable symbolic rules suitable for formal verification and causal inference.

**Key Capabilities:**
- **Symbolic Rule Extraction**: Convert tree structures into Boolean logic formulas
- **Model-Specific FFA Implementation**:
  - **XGBoost FFA**: Direct rule extraction from JSON model structure
  - **CatBoost FFA**: **NOT performed** due to CatBoost's complex hashing and CTR (Counter-based Target Statistics) for categorical variables that make direct rule extraction difficult
  - **CatBoost SHAP**: Used for feature importance filtering (not for FFA rule extraction)
- **Anchored Explanations (AXP)**: Generate instance-level explanations using rule matching
  - **Rule Selection Logic**: Union of three sets:
    1. **First 100 matched rules** - Common patterns
    2. **Random sample of 100 matched rules** - Diversity and coverage
    3. **Top-K SHAP rules with percentile threshold** - Hybrid approach:
       - Takes top 300 rules by SHAP importance score
       - OR all rules above 10th percentile threshold (whichever captures more rules)
       - Uses SHAP importance from **both XGBoost and CatBoost** to filter/prioritize rules
       - Balances performance (limits rule count) with coverage (doesn't miss important rules)
  - **SHAP Requirement**: SHAP values from Step 7 (both XGBoost and CatBoost) are required (raises error if not available)
  - **Performance Optimization**: Limits rule sets to ~300-500 unique rules for efficient AXP computation
- **Causal Analysis**: Measure causal responsibility through counterfactual analysis
- **Feature Importance**: Calculate importance scores from explanations and causal effects

### Core Components

- **`run_full_ffa_analysis.py`** - Main script to run complete FFA analysis workflow
- **`ffa_analysis.py`** - Core FFA analysis functions
- **`base_symbolic_explainer.py`** - Base class for unified symbolic explainers
- **`catboost_axp_explainer.py`** - CatBoost-specific explainer implementation
- **`xgboost_axp_explainer.py`** - XGBoost-specific explainer implementation
- **`combined_causal_analysis.py`** - Dual-approach causal analysis (explainer-based + probability-based)
- **`create_visualizations.py`** - Generate static visualizations
- **`interactive_risk_explorer.py`** - Generate interactive Plotly dashboards

### Quick Start

```bash
# Run complete FFA analysis for all models
python run_full_ffa_analysis.py

# Generate visualizations
python create_visualizations.py

# Run combined causal analysis
python combined_causal_analysis.py

# Create interactive dashboards
python interactive_risk_explorer.py
```

### Documentation

For detailed documentation, see [`docs/Step9_FFA/`](../docs/Step9_FFA/):

- **[README_ffa_analysis.md](../docs/Step9_FFA/README_ffa_analysis.md)** - Complete FFA analysis framework overview
- **[README_ffa_causal_analysis.md](../docs/Step9_FFA/README_ffa_causal_analysis.md)** - Dual-approach causal analysis guide
- **[README_ffa_unified_schema.md](../docs/Step9_FFA/README_ffa_unified_schema.md)** - Unified schema for symbolic explainers

See [`docs/Step9_FFA/README.md`](../docs/Step9_FFA/README.md) for complete documentation index.

### Architecture

The FFA pipeline follows three phases:
1. **Model Ingestion & Feature Mapping** - Load models and extract feature information
2. **Symbolic Logic Extraction** - Convert tree paths to PySAT formulas
3. **Explanation & Analysis** - Generate explanations, calculate importance, perform causal analysis

### XGBoost JSON → DataFrame → Rules (Current Implementation)

For the leakage-filtered final models (e.g., `opioid_ed / 13-24`), we now use a **DataFrame-centric** path for XGBoost FFA:

- **Model export (step 6_final_model)**:
  - After MC-CV and final refit, `run_final_model.py` exports an FFA-friendly JSON at:
    - `6_final_model/outputs/{cohort}/{age_band_fname}/final_model_json/{cohort}_{age_band_fname}_final_model_xgboost.json`
  - The JSON has a minimal, explainer-focused schema:
    - `model_type`: `"xgboost"`
    - `feature_names`: list of numeric feature columns used in training (ordered as in the final feature matrix).
    - `trees`: list of **text tree dumps** from `booster.get_dump(dump_format="text")`, one string per tree.

- **Explainer initialization (run_full_ffa_analysis.py)**:
  - When we call `initialize_explainer(...)` for `model_type="xgboost"`:
    - We build a `PathConfig` pointing at the JSON and the leakage-filtered feature CSV from `6_final_model`.
    - We pass the **DataFrame column names** from the final features into the explainer:
      - `feature_names=list(X.columns)`
    - The `XGBoostSymbolicExplainer` receives these names and keeps them as its `feature_names` mapping.

- **JSON → DataFrame conversion (xgboost_axp_explainer.py)**:
  - `fit_from_model_json(model_json)` now:
    - Uses `explainer.feature_names` if already set, otherwise falls back to `model_json["feature_names"]` or infers `f0`, `f1`, ... when needed.
    - Iterates over `model_json["trees"]` (the text dumps) and, for each tree:
      - Parses the dump into a structured tree.
      - Calls `_explode_tree_to_dataframe(parsed_tree, tree_idx)` to turn all root-to-leaf paths into a **pandas DataFrame** (`df_paths`).
        - Each row represents one decision path with feature index, thresholds, and leaf prediction.
      - Calls `_create_rules_from_dataframe(df_paths)` to convert the DataFrame rows into symbolic CNF clauses (`rule_clauses`, `rule_predictions`).
    - This DataFrame path is the primary path; when it fails for a tree, we fall back to a direct recursive traversal of the parsed tree.

- **Why this matters**:
  - The explainer no longer depends on a fragile, version-specific XGBoost JSON schema.
  - Feature names are **guaranteed to align** with the final model’s feature matrix, since they come from the same DataFrame used for training.
  - The DataFrame representation of each tree path makes debugging, inspection, and downstream analysis (e.g., exporting rules or joining back to feature engineering outputs) much easier and more robust across XGBoost versions.

### Output Structure

```
outputs/
├── {cohort}/
│   └── {age_band}/
│       ├── {model_type}/
│       │   ├── analysis_summary.json
│       │   ├── axp_explanations.csv
│       │   └── feature_importance_axp.csv
│       ├── causal_analysis/
│       │   ├── causal_importance_*.csv
│       │   ├── causal_analysis_*.png
│       │   └── intervention_effects_radar_chart.html
│       ├── visualizations/
│       │   ├── feature_importance_comparison.png
│       │   ├── normalized_importance.png
│       │   └── ...
│       └── interactive/
│           ├── dropdown_dashboard.html
│           └── feature_slider_dashboard.html
```

### Key Features

- **XGBoost FFA Only**: FFA analysis is performed only for XGBoost models
  - CatBoost FFA is not performed due to complex hashing and CTR transformations
  - CatBoost SHAP values are used for feature importance filtering in XGBoost FFA
- **SHAP-Augmented Rule Filtering**: Uses SHAP importance from both XGBoost and CatBoost to filter/prioritize rules
- **Unified Schema**: Consistent representation across XGBoost model types
- **Dual Causal Analysis**: Explainer-based and probability-based methods
- **Interactive Dashboards**: Plotly-based risk exploration tools
- **Formal Verification**: SAT solver integration for consistency checking

