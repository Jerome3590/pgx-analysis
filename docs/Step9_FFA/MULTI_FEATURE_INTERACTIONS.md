# Multi-Feature Interaction Testing in FFA and Dashboard

## Overview

This document describes the implementation plan for explicit multi-feature interaction testing in the FFA causal analysis and dashboard. This extends the current univariate causal analysis to capture explicit interactions between multiple features.

## Current Implementation

### Current Causal Analysis (Univariate)

The current implementation:
- Modifies **one feature at a time**
- Measures how AXP explanations change
- Captures interactions **implicitly** through joint AXP rules

**Limitation**: Does not explicitly test multi-feature interactions (e.g., modifying two features simultaneously).

### How Joint Rules Are Currently Captured

1. **AXP Explanations**: Contain multiple features together (joint rules)
2. **Univariate Testing**: When one feature is modified, if it's part of a joint AXP, the entire explanation changes
3. **Implicit Interactions**: Interactions are captured through the explanation change metric

## Proposed Implementation

### Phase 1: Extend FFA Causal Analysis

#### 1.1 Add Multi-Feature Interaction Testing Function

**Location**: `7_ffa_analysis/run_full_ffa_analysis.py`

**New Function**: `perform_multi_feature_causal_analysis()`

```python
def perform_multi_feature_causal_analysis(
    explainer: Any,
    X: pd.DataFrame,
    y: np.ndarray,
    feature_importance_df: pd.DataFrame,
    cohort: str,
    age_band: str,
    max_interaction_size: int = 2,
    top_k_features: int = 20
) -> pd.DataFrame:
    """
    Perform causal analysis testing multi-feature interactions.
    
    Tests combinations of features (pairs, triplets, etc.) to measure
    their combined causal effect.
    
    Args:
        explainer: FFA explainer instance
        X: Feature matrix
        y: Target vector
        feature_importance_df: Feature importance from AXP
        cohort: Cohort name
        age_band: Age band
        max_interaction_size: Maximum number of features to test together (default: 2 for pairs)
        top_k_features: Number of top features to consider for interactions (default: 20)
    
    Returns:
        DataFrame with columns:
        - feature_combination: Tuple of feature names (e.g., ("drug_A", "drug_B"))
        - interaction_size: Number of features in combination (2, 3, etc.)
        - causal_importance: Combined causal effect when all features are modified
        - individual_effects: Sum of individual causal effects
        - interaction_effect: Difference (combined - individual), measures synergy/antagonism
    """
```

**Implementation Strategy**:
1. Select top K features from `feature_importance_df` (by causal importance)
2. Generate all combinations of size 2, 3, ..., up to `max_interaction_size`
3. For each combination:
   - Modify all features in the combination simultaneously
   - Generate explanations for modified instances
   - Compare with original explanations
   - Calculate combined causal effect
   - Calculate sum of individual effects (from univariate analysis)
   - Calculate interaction effect = combined - individual
4. Return results sorted by interaction effect (strongest interactions first)

#### 1.2 Integration with Existing Causal Analysis

**Modify**: `run_full_ffa_analysis.py` → `run_full_analysis_for_model()`

Add option to run multi-feature interaction analysis:

```python
# After univariate causal analysis
causal_df = perform_causal_analysis(...)

# Optionally run multi-feature interaction analysis
if ANALYSIS_CONFIG.get('enable_interaction_analysis', False):
    interaction_df = perform_multi_feature_causal_analysis(
        explainer, X, y, feature_importance_df, cohort, age_band,
        max_interaction_size=ANALYSIS_CONFIG.get('max_interaction_size', 2),
        top_k_features=ANALYSIS_CONFIG.get('interaction_top_k', 20)
    )
    # Save interaction results
    save_interaction_results(model_type, interaction_df, cohort, age_band)
```

#### 1.3 Output Format

**File**: `7_ffa_analysis/outputs/{cohort}/{age_band}/interaction_analysis_{model_type}.csv`

**Columns**:
- `feature_combination`: String representation of feature tuple (e.g., "drug_A|drug_B")
- `interaction_size`: Integer (2, 3, etc.)
- `combined_causal_importance`: Float (effect when all features modified together)
- `sum_individual_effects`: Float (sum of individual univariate effects)
- `interaction_effect`: Float (combined - individual, measures synergy)
- `n_instances_tested`: Integer (number of instances tested)
- `explanation_change_rate`: Float (fraction of explanations that changed)

**S3 Path**: `gold/ffa_analysis/{cohort}/{age_band}/interaction_analysis_{model_type}.csv`

### Phase 2: Dashboard Integration

#### 2.1 Backend API Endpoint

**Location**: `9_risk_dashboard/lambda_function.py`

**New Endpoint**: `POST /causal/interactions`

**Request Body**:
```json
{
  "cohort": "opioid_ed",
  "age_band": "25-44",
  "selected_features": ["item_drug_A", "item_drug_B", "item_icd_X"],
  "max_interaction_size": 2,
  "model_type": "xgboost"  // optional, defaults to best model
}
```

**Response**:
```json
{
  "interactions": [
    {
      "features": ["item_drug_A", "item_drug_B"],
      "interaction_size": 2,
      "combined_effect": 0.35,
      "sum_individual": 0.28,
      "interaction_effect": 0.07,
      "synergy_type": "positive",  // positive = synergy, negative = antagonism
      "explanation_change_rate": 0.42
    },
    ...
  ],
  "top_interactions": [...],  // Top 10 by interaction_effect
  "summary": {
    "total_interactions_tested": 45,
    "positive_synergies": 12,
    "negative_synergies": 8,
    "neutral": 25
  }
}
```

**Implementation**:
```python
def handle_causal_interactions(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    POST /causal/interactions
    
    Returns multi-feature interaction analysis results.
    """
    body = json.loads(event.get("body") or "{}")
    cohort = body.get("cohort")
    age_band = body.get("age_band")
    selected_features = body.get("selected_features", [])
    max_interaction_size = body.get("max_interaction_size", 2)
    model_type = body.get("model_type", "xgboost")  # Default to best model
    
    # Load interaction analysis results from S3
    interaction_df = load_interaction_analysis(cohort, age_band, model_type)
    
    # Filter to selected features if provided
    if selected_features:
        interaction_df = filter_interactions_by_features(
            interaction_df, selected_features
        )
    
    # Filter by max_interaction_size
    interaction_df = interaction_df[
        interaction_df['interaction_size'] <= max_interaction_size
    ]
    
    # Calculate synergy types
    interaction_df['synergy_type'] = interaction_df['interaction_effect'].apply(
        lambda x: 'positive' if x > 0.01 else ('negative' if x < -0.01 else 'neutral')
    )
    
    # Format response
    interactions = interaction_df.to_dict('records')
    top_interactions = interaction_df.nlargest(10, 'interaction_effect').to_dict('records')
    
    summary = {
        'total_interactions_tested': len(interaction_df),
        'positive_synergies': len(interaction_df[interaction_df['synergy_type'] == 'positive']),
        'negative_synergies': len(interaction_df[interaction_df['synergy_type'] == 'negative']),
        'neutral': len(interaction_df[interaction_df['synergy_type'] == 'neutral'])
    }
    
    return _response(200, {
        'interactions': interactions,
        'top_interactions': top_interactions,
        'summary': summary
    })
```

#### 2.2 Frontend Dashboard Tab

**Location**: `9_risk_dashboard/index.html` or `dashboard_index_template.html`

**New Section**: "Feature Interactions" tab or section

**Features**:
1. **Interaction Matrix Visualization**: Heatmap showing interaction effects between feature pairs
2. **Top Interactions List**: Table showing strongest synergies/antagonisms
3. **Interaction Network Graph**: Network visualization showing feature relationships
4. **Filter Controls**: 
   - Select features to analyze
   - Set max interaction size (2, 3, 4)
   - Filter by synergy type (positive, negative, neutral)

**Example UI**:
```html
<div id="interactions-tab">
  <h2>Feature Interactions Analysis</h2>
  
  <div class="controls">
    <label>Max Interaction Size:</label>
    <select id="max-interaction-size">
      <option value="2">Pairs (2 features)</option>
      <option value="3">Triplets (3 features)</option>
      <option value="4">Quadruplets (4 features)</option>
    </select>
    
    <label>Filter by Synergy:</label>
    <select id="synergy-filter">
      <option value="all">All</option>
      <option value="positive">Positive Synergies</option>
      <option value="negative">Antagonisms</option>
    </select>
    
    <button id="btnAnalyzeInteractions">Analyze Interactions</button>
  </div>
  
  <div id="interaction-matrix-chart"></div>
  <div id="top-interactions-table"></div>
  <div id="interaction-network-chart"></div>
</div>
```

#### 2.3 Visualization Components

**1. Interaction Matrix Heatmap**:
- X-axis: Feature 1
- Y-axis: Feature 2
- Color: Interaction effect (red = positive synergy, blue = antagonism)
- Tooltip: Shows combined effect, individual effects, interaction effect

**2. Top Interactions Table**:
- Columns: Feature Combination, Interaction Size, Combined Effect, Individual Sum, Interaction Effect, Synergy Type
- Sortable by interaction effect
- Clickable to highlight in matrix/network

**3. Interaction Network Graph**:
- Nodes: Features
- Edges: Interactions (thickness = interaction effect magnitude)
- Color: Synergy type (green = positive, red = negative, gray = neutral)
- Interactive: Click node to highlight its interactions

### Phase 3: Performance Optimization

#### 3.1 Precomputation Strategy

**Option A**: Precompute all interactions during FFA analysis
- Pros: Fast dashboard queries
- Cons: Large storage, long computation time

**Option B**: Compute on-demand for selected features
- Pros: Flexible, smaller storage
- Cons: Slower dashboard queries

**Recommended**: Hybrid approach
- Precompute top 20 features × top 20 features = 400 pairs during FFA
- Compute higher-order interactions (triplets, etc.) on-demand in dashboard

#### 3.2 Caching Strategy

- Cache interaction results in S3
- Use Lambda cache for frequently accessed interactions
- Invalidate cache when new FFA results are available

### Phase 4: Configuration

**Add to**: `7_ffa_analysis/run_full_ffa_analysis.py` → `ANALYSIS_CONFIG`

```python
ANALYSIS_CONFIG = {
    # ... existing config ...
    
    # Multi-feature interaction analysis
    'enable_interaction_analysis': True,
    'max_interaction_size': 2,  # Test pairs (2), triplets (3), etc.
    'interaction_top_k': 20,    # Top K features to consider for interactions
    'interaction_sample_size': 100,  # Sample size for interaction testing
    'min_interaction_effect': 0.01,  # Minimum interaction effect to report
}
```

## Implementation Steps

1. **Step 1**: Implement `perform_multi_feature_causal_analysis()` in FFA
2. **Step 2**: Integrate with existing causal analysis workflow
3. **Step 3**: Add S3 upload for interaction results
4. **Step 4**: Implement `handle_causal_interactions()` endpoint in Lambda
5. **Step 5**: Add frontend UI components for interaction visualization
6. **Step 6**: Test with sample cohort/age_band
7. **Step 7**: Update documentation

## Benefits

1. **Explicit Interaction Detection**: Directly measures multi-feature synergies/antagonisms
2. **Better Causal Understanding**: Identifies which feature combinations drive predictions
3. **Clinical Insights**: Helps identify drug-drug interactions, comorbidity effects, etc.
4. **Dashboard Enhancement**: Provides interactive exploration of feature relationships

## Limitations

1. **Computational Cost**: Testing all combinations is expensive (O(n^k) for k features)
2. **Sample Size**: May need to limit to top features and small interaction sizes
3. **Interpretation**: Higher-order interactions (>3 features) may be harder to interpret

## Future Enhancements

1. **Conditional Interactions**: Test interactions conditional on other features
2. **Temporal Interactions**: Test interactions across time (e.g., drug A followed by drug B)
3. **Domain-Specific Interactions**: Pre-define clinically relevant interaction sets (e.g., drug-drug interactions from CPIC)

