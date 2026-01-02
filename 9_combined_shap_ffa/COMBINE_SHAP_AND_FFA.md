# Combining SHAP and FFA for Comprehensive Row-Level Analysis

## Overview

**SHAP and FFA are complementary, not redundant.** You should use **both together** to get complete patient-level insights, not just look at features that appear in both.

## Key Differences

| Aspect | SHAP | FFA (AXP) |
|--------|------|-----------|
| **What it measures** | Feature contribution (Shapley values) | Rule-based explanations (logical conditions) |
| **Question answered** | "How much did feature X contribute?" | "Which rules/conditions led to this prediction?" |
| **Output format** | Numerical values (positive/negative contributions) | Symbolic rules (IF-THEN conditions) |
| **Granularity** | Per-feature contribution | Per-rule explanation |
| **Interpretability** | Quantitative | Qualitative/logical |

## Why Use Both (Not Just Intersection)

### 1. **Complementary Information**

**SHAP tells you:**
- "Feature A contributed +0.15 to the prediction"
- "Feature B contributed -0.08 (reduced risk)"
- Quantitative magnitude of each feature's impact

**FFA tells you:**
- "Patient matched Rule 42: IF (Feature A > 0.5) AND (Feature B < 0.3) THEN high risk"
- Which specific conditions were satisfied
- Logical reasoning path

### 2. **Different Perspectives**

- **SHAP**: Additive feature contributions (sum to prediction)
- **FFA**: Rule-based logic (discrete conditions)

### 3. **Validation Through Agreement**

- **When they agree**: High confidence in feature importance
- **When they disagree**: Investigate why (may reveal interactions or non-linearities)

## How to Combine SHAP and FFA

### Approach 1: **Consensus Analysis** (High Confidence)

Find features that are important in **both** methods:

```python
import pandas as pd
import numpy as np

# Load SHAP results
shap_importance = pd.read_csv('shap_feature_importance.csv')
shap_top_features = set(shap_importance.head(20)['feature'].values)

# Load FFA results
ffa_importance = pd.read_csv('7_ffa_analysis/outputs/.../feature_importance_axp.csv')
ffa_top_features = set(ffa_importance.head(20)['feature'].values)

# Consensus features (high confidence)
consensus_features = shap_top_features.intersection(ffa_top_features)
print(f"Consensus features (high confidence): {len(consensus_features)}")
print(consensus_features)
```

**Use case**: These are your **most reliable** features - important from both perspectives.

### Approach 2: **Complementary Analysis** (Complete Picture)

Use **both** methods together, not just intersection:

```python
# Combine SHAP and FFA for patient-level analysis
def analyze_patient_with_both_methods(patient_id, shap_values, ffa_explanations):
    """
    Analyze a patient using both SHAP and FFA.
    """
    # SHAP analysis
    patient_shap = shap_values[patient_id]
    top_shap_features = patient_shap.nlargest(10)
    
    # FFA analysis
    patient_ffa = ffa_explanations[ffa_explanations['instance_id'] == patient_id]
    matched_rules = patient_ffa['axp'].values[0]  # List of matched rules
    
    return {
        'patient_id': patient_id,
        'shap_top_features': top_shap_features.to_dict(),
        'ffa_matched_rules': matched_rules,
        'consensus_features': set(top_shap_features.index).intersection(
            extract_features_from_rules(matched_rules)
        )
    }
```

**Use case**: Get complete understanding - quantitative (SHAP) + logical (FFA).

### Approach 3: **Weighted Combination** (Like FFA's Combined Causal Analysis)

Combine normalized scores from both methods:

```python
def combine_shap_and_ffa(shap_importance, ffa_importance, weight_shap=0.5, weight_ffa=0.5):
    """
    Combine SHAP and FFA importance scores.
    """
    # Normalize both to [0, 1]
    shap_norm = (shap_importance['importance'] - shap_importance['importance'].min()) / \
                (shap_importance['importance'].max() - shap_importance['importance'].min() + 1e-10)
    
    ffa_norm = (ffa_importance['importance'] - ffa_importance['importance'].min()) / \
               (ffa_importance['importance'].max() - ffa_importance['importance'].min() + 1e-10)
    
    # Merge
    combined = shap_importance[['feature']].merge(
        ffa_importance[['feature', 'importance']],
        on='feature',
        how='outer',
        suffixes=('_shap', '_ffa')
    )
    
    # Fill missing values with 0
    combined['shap_norm'] = shap_norm.reindex(combined.index).fillna(0)
    combined['ffa_norm'] = ffa_norm.reindex(combined.index).fillna(0)
    
    # Weighted combination
    combined['combined_importance'] = (
        weight_shap * combined['shap_norm'] + 
        weight_ffa * combined['ffa_norm']
    )
    
    return combined.sort_values('combined_importance', ascending=False)
```

**Use case**: Unified ranking that considers both methods.

## Practical Workflow

### Step 1: Run Both Analyses

```bash
# Run FFA analysis
python 7_ffa_analysis/run_full_ffa_analysis.py

# Run SHAP analysis (when implemented in Step 8)
python 8_final_model/add_shap_analysis.py
```

### Step 2: Load Results

```python
import pandas as pd

# FFA results
ffa_explanations = pd.read_csv('7_ffa_analysis/outputs/.../axp_explanations.csv')
ffa_importance = pd.read_csv('7_ffa_analysis/outputs/.../feature_importance_axp.csv')

# SHAP results
shap_values = np.load('8_final_model/outputs/.../shap_values.npy')
shap_importance = pd.read_csv('8_final_model/outputs/.../shap_feature_importance.csv')
```

### Step 3: Patient-Level Analysis

```python
def comprehensive_patient_analysis(patient_id):
    """
    Comprehensive analysis combining SHAP and FFA.
    """
    # SHAP: Quantitative contributions
    patient_shap = shap_values[patient_id]
    shap_contributions = {
        'top_positive': patient_shap.nlargest(5).to_dict(),
        'top_negative': patient_shap.nsmallest(5).to_dict(),
        'total_contribution': patient_shap.sum()
    }
    
    # FFA: Rule-based explanations
    patient_ffa = ffa_explanations[ffa_explanations['instance_id'] == patient_id]
    ffa_explanation = {
        'matched_rules': patient_ffa['axp'].values[0],
        'conditions': patient_ffa['conditions'].values[0],
        'rule_count': len(patient_ffa)
    }
    
    # Extract features from FFA rules
    ffa_features = extract_features_from_rules(ffa_explanation['matched_rules'])
    
    # Consensus
    consensus = set(shap_contributions['top_positive'].keys()).intersection(ffa_features)
    
    return {
        'patient_id': patient_id,
        'shap_analysis': shap_contributions,
        'ffa_analysis': ffa_explanation,
        'consensus_features': consensus,
        'interpretation': generate_interpretation(shap_contributions, ffa_explanation, consensus)
    }
```

### Step 4: Generate Interpretations

```python
def generate_interpretation(shap_contributions, ffa_explanation, consensus):
    """
    Generate human-readable interpretation combining SHAP and FFA.
    """
    interpretation = []
    
    # SHAP insights
    interpretation.append("QUANTITATIVE ANALYSIS (SHAP):")
    for feature, value in shap_contributions['top_positive'].items():
        interpretation.append(f"  - {feature}: +{value:.3f} (increased risk)")
    
    # FFA insights
    interpretation.append("\nLOGICAL ANALYSIS (FFA):")
    interpretation.append(f"  - Matched {len(ffa_explanation['matched_rules'])} rules")
    interpretation.append(f"  - Conditions: {ffa_explanation['conditions']}")
    
    # Consensus
    interpretation.append("\nCONSENSUS:")
    if consensus:
        interpretation.append(f"  - High confidence features: {', '.join(consensus)}")
        interpretation.append("  - These features are important from both perspectives")
    else:
        interpretation.append("  - No direct consensus (may indicate complex interactions)")
    
    return "\n".join(interpretation)
```

## When to Trust Consensus vs. Investigate Differences

### ✅ **Trust Consensus** When:
- Same features appear in top 10 of both methods
- SHAP values align with FFA rule conditions
- Both methods agree on direction (increase/decrease risk)

### 🔍 **Investigate Differences** When:
- Feature important in SHAP but not in FFA (or vice versa)
- SHAP shows positive contribution but FFA rule shows negative condition
- Disagreement may reveal:
  - **Non-linear interactions**: Feature matters in combination (FFA) but not alone (SHAP)
  - **Context-dependent effects**: Feature matters under certain conditions (FFA rules)
  - **Method-specific biases**: One method may miss certain patterns

## Example: Patient Analysis

```python
# Patient 12345 analysis
patient_analysis = comprehensive_patient_analysis('patient_12345')

print(patient_analysis['interpretation'])
```

**Output:**
```
QUANTITATIVE ANALYSIS (SHAP):
  - AMOXICILLIN: +0.15 (increased risk)
  - AZITHROMYCIN: +0.12 (increased risk)
  - METFORMIN: -0.08 (decreased risk)

LOGICAL ANALYSIS (FFA):
  - Matched 3 rules
  - Conditions: [AMOXICILLIN > 0.5, AZITHROMYCIN > 0.3, METFORMIN < 0.2]

CONSENSUS:
  - High confidence features: AMOXICILLIN, AZITHROMYCIN
  - These features are important from both perspectives
```

## Summary

**Don't just look at intersection** - use both methods together:

1. **SHAP**: Quantitative feature contributions
2. **FFA**: Logical rule-based explanations
3. **Consensus**: High-confidence features (when both agree)
4. **Differences**: Investigate for deeper insights

**Best practice**: Use SHAP for "how much" and FFA for "why/how" (rules/conditions).

