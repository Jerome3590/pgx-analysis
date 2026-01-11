# Support and Confidence Metrics for Causal Features

## Summary
We now calculate and save **support** and **confidence** metrics for causal features in addition to causal importance.

## Metrics Explained

### 1. Support (Support(j))
**Definition**: Number of intervenable instances for feature j.

**Calculation**:
- **Binary features (remove_only mode)**: Number of instances where `feature == 1`
- **Binary features (add_only mode)**: Number of instances where `feature == 0`
- **Continuous features**: Total sample size (all instances)

**Example**:
- If `item_drug_FUROSEMIDE` appears in 50 instances with value=1, then `support = 50`
- This means we can test the removal effect on 50 instances

**Use Case**: 
- Indicates how many instances are available for intervention
- Higher support = more reliable causal estimate
- Used for filtering features with insufficient support (pruning)

### 2. Confidence
**Definition**: Fraction of intervenable instances where the intervention caused a change in the explanation.

**Calculation**:
```
confidence = causal_importance = change_rate
```

**For binary features**:
- `confidence = changes / support`
- Where `changes` = number of instances where removing the feature changed the explanation
- `support` = number of instances with feature=1 (intervenable instances)

**For continuous features**:
- `confidence = changes / total_instances`
- Where `changes` = number of instances where setting to median changed the explanation
- `total_instances` = total sample size

**Example**:
- If `item_drug_FUROSEMIDE` has `support = 50` and `changes = 50`, then `confidence = 1.0`
- This means removing the feature changed the explanation for ALL 50 instances where it was present

**Use Case**:
- Indicates how consistently the feature affects explanations
- Higher confidence = more reliable causal relationship
- `confidence = 1.0` means the feature always affects explanations when present

### 3. Causal Importance (IR(j))
**Definition**: Intervention Rate - same as confidence in our implementation.

**Note**: `causal_importance` and `confidence` are identical in our current implementation. Both represent the fraction of intervenable instances where the intervention caused a change.

## Output Schema

The `causal_importance.parquet` file now contains:

```python
{
    'feature': str,                    # Feature name
    'causal_importance': float,        # IR(j) - Intervention Rate (0.0 to 1.0)
    'support': int,                    # Support(j) - Number of intervenable instances
    'confidence': float,                # Confidence - Same as causal_importance
    'median_value': float,              # Median value (for continuous features)
    'is_binary': bool,                 # Whether feature is binary
    'intervention': str                 # Intervention description
}
```

## Example Output

For `item_drug_FUROSEMIDE`:
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

## Interpretation

### High Support + High Confidence
- **Example**: `support = 100`, `confidence = 1.0`
- **Meaning**: Feature appears in many instances AND always affects explanations
- **Conclusion**: Strong, reliable causal relationship

### Low Support + High Confidence
- **Example**: `support = 5`, `confidence = 1.0`
- **Meaning**: Feature appears in few instances BUT always affects explanations when present
- **Conclusion**: Potentially important but needs more data to confirm

### High Support + Low Confidence
- **Example**: `support = 100`, `confidence = 0.1`
- **Meaning**: Feature appears in many instances BUT rarely affects explanations
- **Conclusion**: Feature is common but not causally important

### Low Support + Low Confidence
- **Example**: `support = 5`, `confidence = 0.2`
- **Meaning**: Feature appears in few instances AND rarely affects explanations
- **Conclusion**: Feature is likely not causally important

## Relationship to Pruning

Support is used in the pruning stage to filter features:
- **min_present_support**: Minimum number of instances with feature=1 (for remove_only mode)
- **min_absent_support**: Minimum number of instances with feature=0 (for add_only mode)

Features with insufficient support are pruned before causal analysis to avoid unreliable estimates.

## Code Location

- **Support calculation**: Line 1411 (`effective_sample_size = len(X_sample_filtered)`)
- **Confidence calculation**: Line 1506 (`confidence = change_rate`)
- **Output**: Lines 1508-1515 (causal_scores.append)

## Notes

1. **Confidence vs Causal Importance**: In our implementation, these are identical. Both represent the fraction of intervenable instances where the intervention caused a change.

2. **Support Normalization**: Causal importance is normalized by support (not total sample size), which is correct for binary features in remove_only mode.

3. **Consistency**: Both grouped and row-by-row fallback methods use the same normalization (effective_sample_size) for consistency.
