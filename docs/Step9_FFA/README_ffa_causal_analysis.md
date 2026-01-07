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

### The Consensus Filter: Why CatBoost SHAP + XGBoost FFA Works

While the inability to generate rules for signals found *only* by CatBoost is a technical gap, the documentation suggests this acts as a **"Consensus Filter"** that screens out model-specific artifacts and potential overfitting.

#### 1. The "Model Agreement" Philosophy

The pipeline explicitly values **robustness over sensitivity**. In the feature importance phase, the system uses an aggregation method designed to **"Reward Model Agreement"**.

- **The Mechanism**: Features that are important in multiple models (CatBoost, XGBoost, and XGBoost RF) receive higher scores than those found by a single model.
- **The Benefit**: This approach "reduces the risk of model-specific artifacts". If CatBoost finds a signal that XGBoost (using a different mathematical structure) cannot replicate, there is a higher probability that the signal is an artifact of CatBoost's specific encoding (CTR) rather than a universal biological truth. By "dropping" these signals from the causal analysis, the system inherently filters for higher-confidence features.

#### 2. CatBoost vs. Rare Noise

While CatBoost is designed to handle high-cardinality data, the documentation notes it handles rare categories by **"shrinking them toward the global mean"** to reduce overfitting.

- **The Risk**: Despite this regularization, CatBoost can still overfit to "idiosyncratic" patterns in the training data, particularly with complex categorical interactions.
- **The Safeguard**: XGBoost in this pipeline is configured to use the **"exact" tree method** specifically to preserve full split resolution. If this rigorous exact method cannot find a split that corresponds to the CatBoost signal, it suggests the signal might be weak or dependent on CatBoost's specific "Ordered Target Statistics" transformation.

#### 3. Causal Analysis Requires Strict Logic

The Formal Feature Attribution (FFA) module is designed to produce **"high-confidence candidates for clinical decision-making"**.

- **Logical Necessity**: FFA requires converting a model into symbolic Boolean logic to test counterfactuals (e.g., "If NOT Drug A, then NO Risk").
- **The Benefit**: If a pattern is so complex or model-specific that it cannot be translated into a symbolic rule (via XGBoost), it is likely too opaque or unstable to serve as the basis for a clinical intervention. Excluding these "un-translatable" CatBoost signals ensures that the final causal recommendations are grounded in logic that can be explicitly verified.

#### Summary

By requiring that a feature be **detected by CatBoost** (high SHAP) *and* **describable by XGBoost** (symbolic rule existence), the system ensures that the **FFA Causal Analysis** only focuses on signals robust enough to be found by two fundamentally different tree-building algorithms. This "gap" effectively serves as a quality control mechanism that filters out model-specific artifacts and ensures logical translatability for clinical use.

## Counterfactual Interventions

### Single-Feature Interventions

For each feature, we create interventions:
- **Remove**: Set feature to median (neutral value)
- **Median**: Set to median value
- **Zero**: Set to zero
- **Increase**: Increase by one standard deviation
- **Decrease**: Decrease by one standard deviation

### Multi-Feature Interaction Testing

**Purpose**: Test combinations of features (pairs, triplets, etc.) to detect synergies/antagonisms

**Feature Selection Strategy**:
- Only includes features with **ANY** importance > 0:
  - SHAP importance > 0 (model-level), OR
  - FFA importance > 0 (explanation-based), OR
  - Causal importance > 0 (individual causal effect)
- This dramatically reduces combinatorial explosion:
  - **Without filtering**: 11,060 features → C(11,060, 2) = **61 million pairs** (impossible!)
  - **With filtering**: ~20-100 important features → C(50, 2) = **1,225 pairs** (manageable)
  - **99.5%+ reduction** in feature count

**Combination Testing Process**:
1. Select all features with ANY importance > 0 (SHAP OR FFA OR causal)
2. Generate all combinations of size 2, 3, ..., up to `max_interaction_size` (default: 2)
3. For each combination:
   - Modify ALL features in the combination simultaneously
   - Generate explanations for modified instances
   - Compare with original explanations
   - Calculate combined causal effect
   - Calculate sum of individual effects (from univariate analysis)
   - Calculate interaction effect = combined_effect - sum_individual_effects
4. Filter results by minimum interaction effect threshold (default: 0.01)

**Interaction Effect Interpretation**:
- **Positive interaction** (synergy): Combined effect > sum of individual effects
  - Example: Drug A + Drug B together have stronger effect than A alone + B alone
- **Negative interaction** (antagonism): Combined effect < sum of individual effects
  - Example: Drug A + Drug B together have weaker effect than expected
- **Neutral**: Combined effect ≈ sum of individual effects (no interaction)

**Output**: `interaction_analysis.parquet` contains:
- `feature_combination`: Feature names joined by "|" (e.g., "item_drug_A|item_drug_B")
- `interaction_size`: Number of features in combination (2, 3, etc.)
- `combined_causal_importance`: Combined effect when all features modified together
- `sum_individual_effects`: Sum of individual univariate effects
- `interaction_effect`: Difference (combined - individual), measures synergy/antagonism
- `synergy_type`: positive/negative/neutral
- `n_instances_tested`: Number of instances tested
- `explanation_change_rate`: Fraction of explanations that changed

**Configuration**:
```python
ANALYSIS_CONFIG = {
    'enable_interaction_analysis': False,  # Set to True to enable
    'max_interaction_size': 2,  # Test pairs (2), triplets (3), etc.
    'interaction_sample_size': 50,  # Sample size for interaction testing
    'min_interaction_effect': 0.01,  # Minimum interaction effect to report
    'min_combined_shap_threshold': 0.0,  # Optional: filter by combined SHAP score
    'min_individual_shap_threshold': 0.0,  # Filter: only features with SHAP > 0
}
```

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

