# Causal Analysis Guide - Formal Feature Attribution (FFA)

## Overview

This guide explains how to conduct **model-based causal analysis** using Formal Feature Attribution (FFA) to measure how interventions on features affect model explanations and predictions. This approach uses counterfactual reasoning to identify features that causally drive the model's decision-making process.

## Key Concepts

### Model-Based Causal Importance

**What it measures**: "What features causally affect the **model's predictions**?"

**How it works**:
1. **Interventions**: Modify features in the dataset (e.g., remove a drug, set to median)
2. **Counterfactual Comparison**: Generate explanations for original vs. modified instances
3. **Change Measurement**: Calculate fraction of instances where explanations changed
4. **Causal Score**: Higher score = stronger causal effect on model reasoning

**Important Distinction**:
- ✅ **Model-Based Causal Inference** (what we measure): Which features drive the model's decision-making
- ❌ **True Causal Inference** (what we don't measure): Which features causally affect the true outcome (requires RCTs)

### Why This Captures True Signal

Model-based causal importance is more robust than correlation-based methods because:

1. **Filters Spurious Correlations**: Features that correlate but don't causally affect predictions get low scores
2. **Measures Intervention Effects**: Uses counterfactual reasoning ("What if this feature changed?")
3. **Model-Based Causal Inference**: Identifies features the model actually relies on
4. **More Robust Than Simple Importance**: Measures intervention effects rather than just associations

## Data Requirements

### Test Data Required

**FFA analysis requires test data (2019) for validation** - no fallback to training data.

**Why test data?**
- Rules are extracted from training model (2016-2018) - correct
- Rules are validated on test data (2019) - ensures generalizability
- Faster processing - test data is smaller than training data
- Aligns with temporal validation best practices

**Test data paths checked** (in order, highest priority first):
1. **`/mnt/nvme/` drive** (Linux/EC2):
   - `/mnt/nvme/6_final_model/outputs/{cohort}/{age_band}/inputs/model_test/final_features.parquet`
   - `/mnt/nvme/6_final_model/outputs/{cohort}/{age_band}/{cohort}_{age_band}_test_final_features_no_leakage.csv`
   - `/mnt/nvme/data/cohorts/cohort_name={cohort}/event_year=2019/age_band={age_band}/final_features.parquet`
2. **Project root** (fallback):
   - `6_final_model/outputs/{cohort}/{age_band}/inputs/model_test/final_features.parquet`
   - `6_final_model/outputs/{cohort}/{age_band}/{cohort}_{age_band}_test_final_features_no_leakage.csv`
   - `data/cohorts/cohort_name={cohort}/event_year=2019/age_band={age_band}/final_features.parquet`

**If test data not found**: Script exits with error code 1, listing all checked paths.

## Causal Importance Calculation

### Single-Feature Causal Analysis

**Process**:
1. **Baseline**: Get explanations (AXP) from model on original data
2. **Intervention**: Apply intervention, get new explanations
3. **Change Measurement**: Count how many explanations changed
4. **Support Calculation**: Count number of intervenable instances
5. **Confidence Calculation**: Fraction of intervenable instances where intervention caused change
6. **Causal Score**: Same as confidence (fraction of instances with changed explanations)

**Interventions**:
- **Binary features (remove_only mode)**: Remove feature (1→0) on instances where feature=1
- **Binary features (add_only mode)**: Add feature (0→1) on instances where feature=0
- **Continuous features**: Set to median value (neutral baseline)

**Output**: `causal_importance.parquet` contains:
- `feature`: Feature name
- `causal_importance`: Causal importance score (IR - Intervention Rate, 0.0 to 1.0)
- `support`: Number of intervenable instances (Support - number of instances where intervention can be applied)
- `confidence`: Confidence score (0.0 to 1.0) - Same as `causal_importance`
- `median_value`: Median value used for intervention (continuous features)
- `is_binary`: Whether feature is binary (0/1)
- `intervention`: Description of intervention applied

### Metrics Explained

#### Causal Importance (IR(j) - Intervention Rate)
- **Definition**: Fraction of instances with changed explanations after intervention
- **Range**: 0.0 to 1.0
- **Interpretation**:
  - `0.0` = Feature has no causal effect (explanations unchanged)
  - `1.0` = Feature always causes explanation changes (perfect causal effect)
  - `0.5` = Feature causes explanation changes in 50% of instances
  - `> 0.1` = Feature has meaningful causal effect

#### Support (Support(j))
- **Definition**: Number of intervenable instances for feature j
- **Calculation**:
  - **Binary features (remove_only)**: Number of instances where `feature == 1`
  - **Binary features (add_only)**: Number of instances where `feature == 0`
  - **Continuous features**: Total sample size
- **Interpretation**:
  - Higher support = More reliable causal estimate (more instances tested)
  - Low support (< 10) = Less reliable (few instances available for intervention)
- **Use Case**: Used for filtering features with insufficient support (pruning)

#### Confidence
- **Definition**: Fraction of intervenable instances where intervention caused change
- **Calculation**: `confidence = changes / support = causal_importance`
- **Range**: 0.0 to 1.0
- **Interpretation**:
  - `confidence = 1.0` = Feature always affects explanations when present
  - `confidence = 0.5` = Feature affects explanations in 50% of intervenable instances
- **Note**: In our implementation, `confidence` and `causal_importance` are identical

**Example**:
For `item_drug_FUROSEMIDE` with `support = 50` and `confidence = 1.0`:
- 50 instances had the feature present (support)
- Removing it changed the explanation in all 50 instances (confidence = 1.0)
- This indicates a strong, reliable causal relationship


### Multi-Feature Interaction Analysis

**Purpose**: Test combinations of features (pairs, triplets, etc.) to detect synergies/antagonisms

**Feature Selection Strategy**:
- Only includes features with **ANY** importance > 0:
  - SHAP importance > 0 (model-level), OR
  - FFA importance > 0 (explanation-based), OR
  - Causal importance > 0 (individual causal effect)
- **No top_k limit**: All features with any importance signal are included
- Features sorted by combined importance (SHAP + causal + FFA) for prioritization
- **Safe for drug-only features**: All drugs with any importance signal should be tested

**Why this filtering works**:
- **Without filtering**: 11,060 features → C(11,060, 2) = **61 million pairs** (impossible!)
- **With filtering**: ~20-100 important features → C(50, 2) = **1,225 pairs** (manageable)
- **99.5%+ reduction** in feature count

**Cohort-Specific Configuration**:
- **First cohort (`opioid_ed`)**: Tests pairs only (size 2)
- **Second cohort (`non_opioid_ed`/`polypharmacy`)**: Tests pairs and triplets (size 2 and 3)
- Prevents combinatorial explosion for first cohort while allowing higher-order interactions for polypharmacy

**Combination Testing Process**:
1. Select all features with ANY importance > 0 (SHAP OR FFA OR causal)
2. Generate all combinations of size 2, 3, ..., up to `max_interaction_size` (cohort-specific)
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

**Drug Interaction Calculator**: For `non_opioid_ed` cohort (drug-only features), this serves as a drug interaction causal calculator for ED visits:
- Identifies which drug combinations causally increase ED visit risk
- Measures synergy/antagonism effects (positive = synergy, negative = antagonism)
- Output: `interaction_analysis.parquet` with drug-drug and drug-drug-drug interaction effects

**Configuration**:
```python
ANALYSIS_CONFIG = {
    'enable_interaction_analysis': True,  # Enabled by default
    'max_interaction_size': 2,  # Cohort-specific: 2 for opioid_ed, 3 for polypharmacy
    'interaction_sample_size': 50,  # Sample size for interaction testing
    'min_interaction_effect': 0.01,  # Minimum interaction effect to report
    'binary_intervention_mode': 'remove_only',  # remove_only or add_only
}
```

## Technical Implementation

### Rule Grouping for Efficiency

The causal analysis uses **rule grouping** to dramatically improve efficiency:

**How it works**:
1. **Class-Specific Rule Matching**: Rules matched per predicted class (0 or 1)
2. **Deterministic Grouping**: Instances with identical rules AND same predicted class form a group
3. **Efficiency**: O(n) → O(g) where g << n (groups << instances)
   - Typical: 10,000 instances → ~100-500 groups (20-100x reduction)
4. **No Accuracy Loss**: Instances in same group have identical rules, so identical AXP

**Why it's robust for binary outcomes**:
- Respects class boundaries (Class 0 vs Class 1)
- Ensures instances with identical rule patterns get identical explanations
- Handles binary feature interventions correctly (only test instances where feature=1)
- Full AXP recomputation even when rules don't change (detects features in explanations)

### Partial Condition Satisfaction

**Important**: Rules use **strict AND logic** - a rule matches only if **ALL conditions are satisfied**.

- **No partial matching**: If a rule has 3 conditions but only 2 are satisfied → rule does NOT match
- **Deterministic**: Same conditions → same rule matches → same group
- **Sensitive to changes**: Any condition change can alter rule matches and create new groups

**Example**:
```
Rule: (age > 25) AND (drug_count > 3) AND (icd_code == "E11")

Instance A: age=30, drug_count=5, icd_code="E11" → ✅ Rule MATCHES
Instance B: age=30, drug_count=5, icd_code="E10" → ❌ Rule DOES NOT MATCH
Instance C: age=30, drug_count=2, icd_code="E11" → ❌ Rule DOES NOT MATCH
```

Instances with different partial matches end up matching **different, simpler rules** that only require the subset of conditions they satisfy.

### Feature Pruning

**Key principle**:
> **Never prune before you measure causality.  
> Always prune before combinatorics.  
> Only early-stop after you've committed.**

**Current status**:
- ✅ **Binary intervention mode consistency**: Univariate and interaction analysis both use the same `binary_intervention_mode` (default: `remove_only`)
- ⚠️ **Stage 2.5 pruning gate**: NOT YET IMPLEMENTED (highest priority)
- ⚠️ **Stage 3 interaction pruning**: PARTIALLY IMPLEMENTED (only SHAP filtering, missing co-occurrence and capping)

For detailed guides on pruning, see:
- [`8_ffa_analysis/PRUNING_PIPELINE.md`](../../8_ffa_analysis/PRUNING_PIPELINE.md) - Complete pruning stage mapping
- [`8_ffa_analysis/PRUNING_RULES.md`](../../8_ffa_analysis/PRUNING_RULES.md) - Detailed pruning rules

## Usage

### Running FFA Causal Analysis

```bash
# Run complete FFA analysis (includes causal analysis)
python utility_scripts/run_full_ffa_analysis.py \
    --cohort opioid_ed \
    --age_band 13-24 \
    --model-type xgboost

# Force rerun (clears existing outputs)
python utility_scripts/run_full_ffa_analysis.py \
    --cohort opioid_ed \
    --age_band 13-24 \
    --model-type xgboost \
    --force

# Adjust parallel workers (for CPU optimization)
python utility_scripts/run_full_ffa_analysis.py \
    --cohort opioid_ed \
    --age_band 13-24 \
    --model-type xgboost \
    --n-jobs 14
```

### Configuration

Edit `ANALYSIS_CONFIG` in `utility_scripts/run_full_ffa_analysis.py`:

```python
ANALYSIS_CONFIG = {
    'max_samples': None,  # None = use all data
    'n_jobs': 14,  # Number of parallel workers (default: min(28, cpu_count()))
    'enable_interaction_analysis': True,  # Enable multi-feature interactions
    'max_interaction_size': 2,  # Cohort-specific: 2 for opioid_ed, 3 for polypharmacy
    'interaction_sample_size': 50,  # Sample size for interaction testing
    'min_interaction_effect': 0.01,  # Minimum interaction effect to report
    'binary_intervention_mode': 'remove_only',  # remove_only or add_only
}
```

## Outputs

### Single-Feature Causal Analysis

**File**: `8_ffa_analysis/outputs/{cohort}/{age_band}/{model_type}/causal_importance.parquet`

**Schema**:
```python
{
    'feature': str,                    # Feature name
    'causal_importance': float,        # IR(j) - Intervention Rate (0.0 to 1.0)
    'support': int,                    # Support(j) - Number of intervenable instances
    'confidence': float,                # Confidence - Same as causal_importance
    'median_value': float,             # Median value (for continuous features)
    'is_binary': bool,                 # Whether feature is binary
    'intervention': str                 # Intervention description
}
```

**Example**:
```python
{
    'feature': 'item_drug_FUROSEMIDE',
    'causal_importance': 1.000000,      # All interventions caused changes
    'support': 50,                      # 50 instances with feature=1
    'confidence': 1.000000,            # 100% of interventions caused changes
    'median_value': 0.0,               # Not used for binary features
    'is_binary': True,
    'intervention': 'removed (1->0, 50/1000 instances)'
}
```

### Multi-Feature Interaction Analysis

**File**: `8_ffa_analysis/outputs/{cohort}/{age_band}/{model_type}/interaction_analysis.parquet`

**Schema**:
```python
{
    'feature_combination': str,         # Feature names joined by "|"
    'interaction_size': int,            # Number of features (2, 3, etc.)
    'combined_causal_importance': float,  # Combined effect
    'sum_individual_effects': float,    # Sum of individual effects
    'interaction_effect': float,         # Difference (combined - individual)
    'synergy_type': str,                # positive/negative/neutral
    'n_instances_tested': int,          # Number of instances tested
    'explanation_change_rate': float    # Fraction of explanations that changed
}
```

## Interpretation

### Causal Importance Score

**High Score (e.g., 1.0)**:
- Feature always causes explanation changes
- Strong, reliable causal relationship
- Good candidate for clinical decision-making

**Medium Score (e.g., 0.5)**:
- Feature causes explanation changes in 50% of instances
- Moderate causal effect
- May depend on context or other features

**Low Score (e.g., 0.1)**:
- Feature rarely causes explanation changes
- Weak causal effect
- May be spurious correlation

**Zero Score (0.0)**:
- Feature never causes explanation changes
- No causal effect on model reasoning
- Can be safely ignored

### Support and Confidence

**High Support + High Confidence** (e.g., `support = 100`, `confidence = 1.0`):
- Feature appears in many instances AND always affects explanations
- **Conclusion**: Strong, reliable causal relationship

**Low Support + High Confidence** (e.g., `support = 5`, `confidence = 1.0`):
- Feature appears in few instances BUT always affects explanations when present
- **Conclusion**: Potentially important but needs more data to confirm

**High Support + Low Confidence** (e.g., `support = 100`, `confidence = 0.1`):
- Feature appears in many instances BUT rarely affects explanations
- **Conclusion**: Feature is common but not causally important

**Low Support + Low Confidence** (e.g., `support = 5`, `confidence = 0.2`):
- Feature appears in few instances AND rarely affects explanations
- **Conclusion**: Feature is likely not causally important

### Interaction Effects

**Positive Interaction (Synergy)**:
- Combined effect > sum of individual effects
- Example: Drug A + Drug B together have stronger effect than A alone + B alone
- **Clinical implication**: Drug combination increases risk more than expected

**Negative Interaction (Antagonism)**:
- Combined effect < sum of individual effects
- Example: Drug A + Drug B together have weaker effect than expected
- **Clinical implication**: Drug combination may have protective or neutralizing effect

**Neutral**:
- Combined effect ≈ sum of individual effects
- **Clinical implication**: No interaction, effects are additive

## When to Use Causal Importance

**Use causal importance when**:
- ✅ You want to identify features that causally drive model predictions
- ✅ You need to filter out spurious correlations
- ✅ You want robust features for clinical decision-making
- ✅ You need interpretable features that can be expressed as Boolean logic
- ✅ You want to understand drug interactions and synergies

**Don't use causal importance when**:
- ❌ You need true causal inference from observational data (requires RCTs)
- ❌ You want to understand population-level causal effects
- ❌ You need to make policy recommendations without experimental validation

## Model Weights

**Note**: FFA-based causal analysis uses only XGBoost models. CatBoost SHAP values are used for feature importance filtering, but CatBoost FFA is not performed.

Models are weighted by their explanation coverage rate:
- **XGBoost**: Weighted by coverage rate (typically ~100%)
- **XGBoost RF**: Weighted by coverage rate (typically ~100%)

Weights are normalized so they sum to 1.0.

## Example Results

```
Top 10 Features by Causal Importance:
1. item_drug_FUROSEMIDE: 1.000 (support: 50, confidence: 1.000)
2. item_drug_LISINOPRIL: 0.940 (support: 75, confidence: 0.940)
3. item_drug_AMLODIPINE_BESYLATE: 0.920 (support: 60, confidence: 0.920)
...

Top Drug Interactions (Synergy):
1. item_drug_FUROSEMIDE|item_drug_LISINOPRIL: +0.15 (positive synergy)
2. item_drug_AMLODIPINE_BESYLATE|item_drug_OMEPRAZOLE: +0.12 (positive synergy)
...

Top Drug Interactions (Antagonism):
1. item_drug_LEVOTHYROXINE_SODIUM|item_drug_LISINOPRIL: -0.10 (negative synergy)
...
```

## Next Steps

1. **Feature Selection**: Use causal importance to select features for intervention
2. **Policy Analysis**: Understand which features have the strongest causal effects
3. **Risk Assessment**: Identify modifiable risk factors
4. **Drug Interaction Analysis**: Use interaction analysis to identify drug combinations that increase ED visit risk
5. **Model Improvement**: Focus on features with high causal importance

## Notes

- **Causal analysis requires careful interpretation** - correlation ≠ causation
- **Results are model-dependent** - different models may show different causal patterns
- **Use domain expertise** to validate causal findings
- **Consider confounders and mediators** in real-world applications
- **Test data required** - ensures validation on unseen data
- **Support and confidence** help assess reliability of causal estimates

## Related Documentation

- [`8_ffa_analysis/README.md`](../../8_ffa_analysis/README.md) - Complete FFA analysis framework overview
- [`8_ffa_analysis/PRUNING_PIPELINE.md`](../../8_ffa_analysis/PRUNING_PIPELINE.md) - Pruning pipeline and implementation status
- [`docs/Step9_FFA/README_ffa_analysis.md`](README_ffa_analysis.md) - FFA analysis overview