# Step 9: Formal Feature Attribution (FFA) Analysis

This folder contains documentation for the Formal Feature Attribution Analysis framework, which provides interpretable explanations and causal analysis for gradient-boosted decision tree models.

## 📋 Documentation Index

### Main Documentation

- **[README_ffa_analysis.md](README_ffa_analysis.md)** - Complete FFA analysis framework overview
  - Architecture and pipeline phases
  - Symbolic rule extraction
  - Anchored explanations (AXP)
  - Feature importance calculation
  - Implementation details

### Causal Analysis

- **[README_ffa_causal_analysis.md](README_ffa_causal_analysis.md)** - Dual-approach causal analysis guide
  - Explainer-based method (FFA)
  - Probability-based method
  - Counterfactual interventions
  - Model ensemble approach
  - Visualization and interpretation

### Technical Details

- **[README_ffa_unified_schema.md](README_ffa_unified_schema.md)** - Unified schema for symbolic explainers
  - Base class architecture
  - Unified DataFrame schema
  - Model-specific implementations
  - Consistency across model types

## 🚀 Quick Start

1. **Overview**: Start with [README_ffa_analysis.md](README_ffa_analysis.md)
2. **Causal Analysis**: See [README_ffa_causal_analysis.md](README_ffa_causal_analysis.md) for dual-approach methodology
3. **Implementation**: Review [README_ffa_unified_schema.md](README_ffa_unified_schema.md) for technical details

## 📚 Related Documentation

- **Step 8**: See [`../Step8_FinalModel/`](../Step8_FinalModel/) for model training details
- **Step 10**: See [`../Step10_Results/`](../Step10_Results/) for dashboard deployment
- **Main Index**: See [`../README.md`](../README.md) for complete documentation index

## 🔑 Key Concepts

- **Formal Feature Attribution (FFA)**: Framework for interpreting tree-based models through symbolic logic
- **Anchored Explanations (AXP)**: Minimal hitting set explanations for individual instances
- **Symbolic Rule Extraction**: Converting tree paths to Boolean logic formulas
- **Causal Analysis**: Measuring causal responsibility through counterfactual interventions
- **Unified Schema**: Consistent representation across CatBoost, XGBoost, and XGBoost RF models

