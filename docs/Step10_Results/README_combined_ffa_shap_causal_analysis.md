# Combined FFA, SHAP, and Causal Analysis

## Overview

This document describes the comprehensive patient-level explanation system that combines **Formal Feature Attribution (FFA)**, **SHAP (SHapley Additive exPlanations)**, and **Causal Analysis** to provide complete, multi-perspective insights into model predictions and feature importance.

## Why Combine Three Methods?

### Complementary Perspectives

| Method | What It Measures | Question Answered | Output Type |
|--------|-----------------|-------------------|-------------|
| **SHAP** | Feature contribution (Shapley values) | "How much did feature X contribute?" | Numerical contributions |
| **FFA** | Rule-based explanations (AXP) | "Which rules/conditions led to this prediction?" | Symbolic/logical rules |
| **Causal** | Intervention effects | "What happens if we change feature X?" | Probability/explanation changes |

### Benefits of Integration

1. **Complete Picture**: Quantitative (SHAP) + Logical (FFA) + Counterfactual (Causal)
2. **Validation**: Agreement between methods increases confidence
3. **Robustness**: Different perspectives reduce method-specific biases
4. **Actionability**: Causal analysis shows what can be changed

## Architecture

```
┌─────────────────┐
│  Step 8: SHAP   │──┐
│  Analysis       │  │
└─────────────────┘  │
                      │
┌─────────────────┐  │    ┌──────────────────────────┐
│  Step 9: FFA    │──┼───▶│  Step 10: Combined       │
│  Analysis       │  │    │  Analysis & Dashboard     │
└─────────────────┘  │    └──────────────────────────┘
                      │
┌─────────────────┐  │
│  Step 9: Causal │──┘
│  Analysis       │
└─────────────────┘
```

## Components

### 1. SHAP Analysis (Step 8)

**Purpose**: Quantitative feature contributions per patient

**Outputs**:
- `shap_values.npy`: SHAP values matrix (n_samples × n_features)
- `shap_feature_importance.csv`: Aggregate feature importance

**Key Features**:
- Additive contributions (sum to prediction)
- Positive/negative contributions
- Patient-specific feature drivers

### 2. FFA Analysis (Step 9)

**Purpose**: Rule-based logical explanations per patient

**Outputs**:
- `axp_explanations.csv`: Instance-level explanations with matched rules
- `feature_importance_axp.csv`: Feature importance from rule frequency

**Key Features**:
- Symbolic IF-THEN rules
- Logical reasoning paths
- Interpretable conditions

### 3. Causal Analysis (Step 9)

**Purpose**: Counterfactual "what-if" analysis

**Outputs**:
- `causal_importance_explainer_method.csv`: FFA-based causal scores
- `causal_importance_probability_method.csv`: Probability-based causal scores
- `causal_importance_combined.csv`: Unified causal importance

**Key Features**:
- Intervention effects (remove, median, zero, increase, decrease)
- Dual approach: explainer-based + probability-based
- Model ensemble weighting

## Usage

### Basic Workflow

```bash
# Step 1: Run SHAP analysis (Step 8)
python 8_final_model/add_shap_analysis.py \
    --cohort non_opioid_ed \
    --age-band 65-74

# Step 2: Run FFA analysis (Step 9)
python 9_ffa_analysis/run_full_ffa_analysis.py

# Step 3: Combine SHAP and FFA results (Step 10)
python 10_results/combine_shap_ffa_results.py \
    --cohort non_opioid_ed \
    --age-band 65-74 \
    --output-dir 10_results/outputs
```

### Advanced Options

```bash
python 10_results/combine_shap_ffa_results.py \
    --cohort non_opioid_ed \
    --age-band 65-74 \
    --top-k 20 \                    # Top K features for consensus
    --weight-shap 0.5 \             # Weight for SHAP (default: 0.5)
    --weight-ffa 0.5 \              # Weight for FFA (default: 0.5)
    --n-patients 100 \              # Number of patients to analyze
    --output-dir 10_results/outputs
```

### Parameters

- `--cohort`: Cohort name (e.g., `non_opioid_ed`, `opioid_ed`)
- `--age-band`: Age band (e.g., `65-74`)
- `--output-dir`: Output directory (default: `10_results/outputs`)
- `--top-k`: Number of top features for consensus analysis (default: 20)
- `--weight-shap`: Weight for SHAP in combined importance (default: 0.5)
- `--weight-ffa`: Weight for FFA in combined importance (default: 0.5)
- `--n-patients`: Number of patients to analyze (default: 100)

## Outputs

### 1. Consensus Features (`consensus_features.json`)

Features that appear in top K of both SHAP and FFA:

```json
{
  "consensus_features": ["AMOXICILLIN", "AZITHROMYCIN", ...],
  "shap_only": ["METFORMIN", ...],
  "ffa_only": ["ONDANSETRON", ...],
  "consensus_count": 12,
  "consensus_rate": 0.6
}
```

**Use case**: High-confidence features for clinical decision-making

### 2. Combined Importance (`combined_importance.csv`)

Weighted combination of SHAP and FFA importance scores:

| feature | shap_norm | ffa_norm | combined_importance |
|---------|-----------|----------|---------------------|
| AMOXICILLIN | 0.95 | 0.88 | 0.915 |
| AZITHROMYCIN | 0.82 | 0.91 | 0.865 |
| ... | ... | ... | ... |

**Use case**: Unified feature ranking for model interpretation

### 3. Patient Explanations (`patient_explanations.csv`)

Comprehensive patient-level explanations combining both methods:

| patient_id | shap_top_positive | ffa_features | consensus_features |
|------------|-------------------|--------------|-------------------|
| 12345 | ["AMOXICILLIN", "AZITHROMYCIN"] | ["AMOXICILLIN", "AZITHROMYCIN"] | ["AMOXICILLIN", "AZITHROMYCIN"] |
| ... | ... | ... | ... |

**Use case**: Patient-specific explanations for clinical use

### 4. Summary Report (`summary_report.txt`)

Human-readable summary of all results:

```
================================================================================
SHAP + FFA COMBINED ANALYSIS SUMMARY
================================================================================

CONSENSUS FEATURES:
  - Consensus features: 12
  - SHAP-only features: 8
  - FFA-only features: 8
  - Consensus rate: 60.0%

  High-confidence features (consensus):
    - AMOXICILLIN
    - AZITHROMYCIN
    ...

COMBINED FEATURE IMPORTANCE (Top 10):
  1. AMOXICILLIN: 0.9150 (SHAP: 0.950, FFA: 0.880)
  2. AZITHROMYCIN: 0.8650 (SHAP: 0.820, FFA: 0.910)
  ...
```

### 5. Causal Analysis Outputs

#### Explainer-Based Method (`causal_importance_explainer_method.csv`)

FFA-based causal scores measuring how explanations change with interventions:

| feature | causal_importance | remove_effect | median_effect | zero_effect | increase_effect |
|--------|-------------------|---------------|--------------|-----------|-----------------|
| AMOXICILLIN | 0.682 | 0.15 | 0.12 | 0.18 | 0.23 |
| ... | ... | ... | ... | ... | ... |

#### Probability-Based Method (`causal_importance_probability_method.csv`)

Probability-based causal scores measuring how predictions change:

| feature | causal_importance | remove_effect | median_effect | zero_effect | increase_effect |
|--------|-------------------|---------------|--------------|-----------|-----------------|
| AMOXICILLIN | 0.654 | 0.14 | 0.11 | 0.17 | 0.23 |
| ... | ... | ... | ... | ... | ... |

#### Combined Method (`causal_importance_combined.csv`)

Unified results from both methods:

| feature | combined_importance | explainer_importance | probability_importance | ... |
|---------|---------------------|----------------------|------------------------|-----|
| AMOXICILLIN | 0.668 | 0.682 | 0.654 | ... |
| ... | ... | ... | ... | ... |

## Causal Analysis Methodology

### Dual Approach

#### Approach 1: Explainer-Based (FFA) Method

1. **Baseline**: Get explanations from all models on original data
2. **Intervention**: Apply intervention (e.g., set feature to median)
3. **Change Measurement**: Count how many explanations change
4. **Aggregation**: Weight by model performance, average across interventions

**Measures**: How logical explanations change with interventions

#### Approach 2: Probability-Based Method

1. **Baseline**: Get probability predictions from all models
2. **Intervention**: Apply intervention (e.g., set feature to median)
3. **Change Measurement**: Calculate absolute probability change
4. **Aggregation**: Weight by model performance, average across interventions

**Measures**: How prediction probabilities change with interventions

### Counterfactual Interventions

For each feature, we create interventions:

- **Remove**: Set feature to median (neutral value)
- **Median**: Set to median value
- **Zero**: Set to zero
- **Increase**: Increase by one standard deviation
- **Decrease**: Decrease by one standard deviation

### Model Ensemble

Models are weighted by their explanation coverage rate:

- **CatBoost**: Weighted by coverage rate (typically ~96%)
- **XGBoost**: Weighted by coverage rate (typically ~100%)
- **XGBoost RF**: Weighted by coverage rate (typically ~100%)

Weights are normalized so they sum to 1.0.

## Interpretation Guide

### Consensus Features (High Confidence)

✅ **Use for**: Clinical decision-making, feature prioritization
- Features important from both quantitative (SHAP) and logical (FFA) perspectives
- Highest confidence in importance

### SHAP-Only Features

🔍 **Investigate**: May indicate quantitative importance not captured in rules
- Could be additive effects
- May need rule refinement

### FFA-Only Features

🔍 **Investigate**: May indicate rule-based importance not captured quantitatively
- Could be conditional effects (only matter in specific contexts)
- May need SHAP interaction values

### Combined Importance

✅ **Use for**: Unified feature ranking
- Balances quantitative and logical perspectives
- Adjustable weights based on trust in each method

### Causal Importance

✅ **Use for**: Understanding modifiable risk factors
- **Higher score** = Feature has stronger causal effect
- **Score = 0** = Feature has no causal effect
- **Score > 0.1** = Feature has meaningful causal effect

**Intervention Effects**:
- **Remove/Median**: Shows effect of neutralizing the feature
- **Zero**: Shows effect of completely removing feature signal
- **Increase**: Shows effect of amplifying the feature

## Integration with Dashboard

### API Endpoints

The dashboard provides endpoints for accessing combined analysis results:

```python
# Get patient explanation
GET /explanations/<cohort>/<age_band>/<patient_id>

# Get consensus features
GET /consensus/<cohort>/<age_band>

# Get causal effects
POST /causal
```

### Dashboard Display

The dashboard displays:

1. **Consensus Features Panel**: High-confidence features from both methods
2. **Combined Importance Chart**: Unified feature ranking
3. **Patient Explanation Panel**: Individual patient analysis
4. **Causal Impact Chart**: "What-if" analysis results

### Example API Response

```json
{
  "shap_analysis": {
    "top_positive": ["AMOXICILLIN", "AZITHROMYCIN"],
    "top_negative": ["METFORMIN", "LISINOPRIL"]
  },
  "ffa_analysis": {
    "matched_rules": ["Rule_42", "Rule_15"],
    "features": ["AMOXICILLIN", "AZITHROMYCIN"]
  },
  "consensus_features": ["AMOXICILLIN", "AZITHROMYCIN"],
  "causal_effects": {
    "AMOXICILLIN": {
      "remove_effect": 0.15,
      "increase_effect": 0.23
    }
  }
}
```

## Best Practices

1. **Start with Consensus**: Focus on consensus features for highest confidence
2. **Investigate Differences**: SHAP-only and FFA-only features may reveal important patterns
3. **Adjust Weights**: Tune `--weight-shap` and `--weight-ffa` based on method reliability
4. **Patient-Level Analysis**: Use patient explanations for individual case analysis
5. **Causal Validation**: Use causal analysis to validate feature importance
6. **Regular Updates**: Re-run combination when new SHAP or FFA results are available

## Troubleshooting

### Missing SHAP Results

If SHAP results are not found:
- Check that Step 8 SHAP analysis has been run
- Verify file paths in `8_final_model/outputs/`
- Check for `shap_values.npy` or `shap_feature_importance.csv`

### Missing FFA Results

If FFA results are not found:
- Check that Step 9 FFA analysis has been run
- Verify file paths in `9_ffa_analysis/outputs/`
- Check for `axp_explanations.csv` and `feature_importance_axp.csv`

### Missing Causal Results

If causal results are not found:
- Check that Step 9 causal analysis has been run
- Verify file paths in `9_ffa_analysis/outputs/`
- Check for `causal_importance_*.csv` files

### Low Consensus Rate

If consensus rate is low (<30%):
- May indicate different patterns captured by each method
- Investigate SHAP-only and FFA-only features
- Consider adjusting `--top-k` parameter
- Review if methods are analyzing the same data subset

### Causal Analysis Issues

If causal analysis fails:
- Check that models are loaded correctly
- Verify sufficient memory (causal analysis is memory-intensive)
- Reduce sample size in `CAUSAL_CONFIG`
- Check that interventions are valid for feature types

## Performance Considerations

### Memory Usage

- **SHAP**: Moderate (n_samples × n_features matrix)
- **FFA**: High (requires loading full models and data)
- **Causal**: Very High (requires multiple model evaluations)

### Computation Time

- **SHAP**: Fast (TreeExplainer) to Moderate (KernelExplainer)
- **FFA**: Moderate to Slow (depends on sample size)
- **Causal**: Slow (requires multiple interventions per feature)

### Optimization Tips

1. **Sample Data**: Use `--n-patients` to limit patient analysis
2. **Limit Features**: Use `--top-k` to focus on top features
3. **Batch Processing**: Process cohorts/age-bands in parallel
4. **Caching**: Cache SHAP/FFA results to avoid recomputation

## Related Documentation

- **Step 8**: [`../Step8_FinalModel/`](../Step8_FinalModel/) - SHAP analysis implementation
- **Step 9**: [`../Step9_FFA/`](../Step9_FFA/) - FFA and causal analysis
- **Parallelization**: [`../CrossStep_Development/README_parallelization_pipeline.md`](../CrossStep_Development/README_parallelization_pipeline.md) - Performance optimization
- **Dashboard**: [`README_results_dashboard.md`](README_results_dashboard.md) - Dashboard integration

## Future Enhancements

- [ ] Batch processing for all cohorts (`--all-cohorts`)
- [ ] Interactive visualization of consensus vs. differences
- [ ] Real-time causal "what-if" analysis in dashboard
- [ ] Automated consensus threshold recommendations
- [ ] Export to clinical report format
- [ ] SHAP interaction values integration
- [ ] Multi-model ensemble causal analysis

