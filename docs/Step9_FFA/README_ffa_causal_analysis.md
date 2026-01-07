# Combined Causal Analysis Guide - Dual Approach

## Overview

This guide explains how to conduct causal analysis using **TWO complementary approaches** with all three models (CatBoost, XGBoost, XGBoost RF) together for comprehensive feature attribution.

## Dual Approach Methodology

### Approach 1: **Explainer-Based (FFA) Method**
- Uses Formal Feature Attribution (FFA) explainers
- Measures how **explanations change** with interventions
- Focuses on **rule-based** causal effects
- Aligned with symbolic logic and interpretability

### Approach 2: **Probability-Based Method**
- Uses actual model predictions
- Measures how **prediction probabilities change** with interventions
- Focuses on **quantitative** causal effects
- Provides direct probability impact estimates

### Why Both Approaches?

1. **Complementary Insights**: 
   - FFA explains *why* (rules/logic)
   - Probability shows *how much* (magnitude)

2. **Validation**: 
   - Agreement between methods increases confidence
   - Disagreement highlights areas needing investigation

3. **Robustness**: 
   - Different perspectives on the same causal question
   - Reduces method-specific biases

## Model Ensemble

**Note**: FFA analysis is performed only for XGBoost models. CatBoost FFA is not performed due to CatBoost's complex hashing and CTR for categorical variables. CatBoost SHAP values are used for feature importance filtering in XGBoost FFA.

We use XGBoost models for FFA-based causal analysis because:
- **XGBoost FFA**: Direct rule extraction from JSON model structure enables symbolic logic analysis
- **CatBoost SHAP**: Used for feature importance filtering (not for FFA rule extraction)
- **Robustness**: Different models may capture different patterns
- **Consensus**: Features important across all models are more reliable
- **Weighted Aggregation**: Models are weighted by their performance (coverage rate)

## Counterfactual Interventions

For each feature, we create interventions:
- **Remove**: Set feature to median (neutral value)
- **Median**: Set to median value
- **Zero**: Set to zero
- **Increase**: Increase by one standard deviation
- **Decrease**: Decrease by one standard deviation

## Causal Importance Calculation

### Explainer-Based Method:
1. **Baseline**: Get explanations from all models on original data
2. **Intervention**: Apply intervention, get new explanations
3. **Change Measurement**: Count how many explanations change
4. **Aggregation**: Weight by model performance, average across interventions

### Probability-Based Method:
1. **Baseline**: Get probability predictions from all models
2. **Intervention**: Apply intervention, get new probabilities
3. **Change Measurement**: Calculate absolute probability change
4. **Aggregation**: Weight by model performance, average across interventions

### Combined Method:
- Normalizes both methods to [0,1] scale
- Averages normalized scores
- Provides unified causal importance ranking

## Usage

### Basic Usage

```python
python combined_causal_analysis.py
```

### Configuration

Edit the `CAUSAL_CONFIG` in `combined_causal_analysis.py`:

```python
CAUSAL_CONFIG = {
    'target_class': 1,
    'sample_size': 500,  # Number of instances to analyze
    'top_k_features': 20,  # Number of top features
    'intervention_types': ['remove', 'median', 'zero', 'increase'],
    'random_seed': 1997,
}
```

## Outputs

### 1. **CSV Results**

#### Explainer-Based Method:
- `causal_importance_explainer_method.csv`: FFA-based causal scores
  - `causal_importance`: Overall causal importance (average across interventions)
  - `remove_effect`, `median_effect`, `zero_effect`, `increase_effect`: Individual intervention effects

#### Probability-Based Method:
- `causal_importance_probability_method.csv`: Probability-based causal scores
  - `causal_importance`: Overall causal importance (average across interventions)
  - `remove_effect`, `median_effect`, `zero_effect`, `increase_effect`: Individual intervention effects

#### Combined Method:
- `causal_importance_combined.csv`: Unified results from both methods
  - `combined_importance`: Normalized average of both methods
  - `explainer_importance`: Explainer-based score
  - `probability_importance`: Probability-based score
  - All intervention effects from both methods

### 2. **Visualizations**
- `combined_causal_analysis_dual_approach.png`: Comprehensive visualization
  - **Panel 1**: Top features by explainer-based method
  - **Panel 2**: Top features by probability-based method
  - **Panel 3**: Top features by combined method
  - **Panel 4**: Scatter plot comparing both methods
  - **Panels 5-6**: Intervention effects for top features (both methods)

## Interpretation

### Causal Importance Score
- **Higher score** = Feature has stronger causal effect
- **Score = 0** = Feature has no causal effect (changing it doesn't change predictions)
- **Score > 0.1** = Feature has meaningful causal effect

### Intervention Effects
- **Remove/Median**: Shows effect of neutralizing the feature
- **Zero**: Shows effect of completely removing feature signal
- **Increase**: Shows effect of amplifying the feature

## Model Weights

**Note**: FFA-based causal analysis uses only XGBoost models. CatBoost SHAP values are used for feature importance filtering, but CatBoost FFA is not performed.

Models are weighted by their explanation coverage rate:
- **XGBoost**: Weighted by coverage rate (typically ~100%)
- **XGBoost RF**: Weighted by coverage rate (typically ~100%)

Weights are normalized so they sum to 1.0.

## Example Results

```
Top 10 Features by Causal Importance:
1. medical_code_itemsets_max_support: 0.682
2. cpt_code_itemsets_matched_count: 0.610
3. drug_name_itemsets_max_support: 0.528
4. cpt_code_itemsets_max_support: 0.514
...
```

## Next Steps

1. **Feature Selection**: Use causal importance to select features for intervention
2. **Policy Analysis**: Understand which features have the strongest causal effects
3. **Risk Assessment**: Identify modifiable risk factors
4. **Model Improvement**: Focus on features with high causal importance

## Notes

- Causal analysis requires careful interpretation - correlation ≠ causation
- Results are model-dependent - different models may show different causal patterns
- Use domain expertise to validate causal findings
- Consider confounders and mediators in real-world applications

