# Binary Feature Causal Importance Fix Verification

## Fix Summary

The fix addresses the issue where binary features (drugs/ICDs) showed `causal_importance = 0.00` even when they appeared in AXPs.

## Fix Location

**File**: `8_ffa_analysis/run_full_ffa_analysis.py`  
**Function**: `_calculate_grouped_causal_effect()`  
**Lines**: 1157-1189

## Fix Logic

1. **Detection**: When analyzing a binary feature, the code checks if the feature appears in the original AXP literals.

2. **Method**: 
   - Iterates through each literal in `original_axp`
   - Looks up the literal in `explainer.id_condition_map` to get `(feat_idx, thresh, direction)`
   - Maps `feat_idx` to feature name via `explainer.feature_names`
   - Checks if the feature name matches the intervened feature
   - Verifies the condition would be invalidated by setting feature=0:
     - `direction == 1 and thresh == 0.0` (feature > 0)
     - OR `direction == 1 and thresh < 1.0` (feature > threshold where threshold < 1)

3. **Counting**: If the feature appears in the AXP, it counts as a change even if `original_axp == modified_axp` (i.e., the AXP computation didn't change).

## Key Code Section

```python
feature_appears_in_axp = False
if is_binary and feat_name and hasattr(explainer, 'id_condition_map') and hasattr(explainer, 'feature_names'):
    # Check if feature appears in original AXP literals
    for lit in original_axp:
        try:
            feat_idx, thresh, direction = explainer.id_condition_map[lit]
            axp_feat_name = explainer.feature_names.get(feat_idx, None)
            if axp_feat_name == feat_name:
                # Check if removing the feature (setting to 0) would invalidate this literal
                if direction == 1 and thresh == 0.0:  # "feature > 0"
                    feature_appears_in_axp = True
                    break
                elif direction == 1 and thresh < 1.0:  # "feature > threshold" where threshold < 1
                    feature_appears_in_axp = True
                    break
        except (KeyError, IndexError, ValueError):
            continue

# Count as change if:
# 1. AXP literals changed (different minimal hitting set)
# 2. Feature appears in original AXP and we're removing it (for binary features)
if original_axp != modified_axp or feature_appears_in_axp:
    total_changes += 1
```

## Verification Checklist

- [x] Fix is implemented in `_calculate_grouped_causal_effect()`
- [x] `is_binary` flag is correctly detected (line 1327)
- [x] `is_binary` is passed to `_calculate_grouped_causal_effect()` (line ~1378)
- [x] Explainer has `id_condition_map` and `feature_names` attributes
- [x] Logic checks for both `thresh == 0.0` and `thresh < 1.0` conditions
- [x] Change is counted even if `original_axp == modified_axp`

## Potential Issues

1. **Missing Attributes**: If `explainer` doesn't have `id_condition_map` or `feature_names`, the fix won't trigger. However, these are set during explainer initialization (lines 608-610, 251-254 in base_symbolic_explainer.py).

2. **Feature Name Mismatch**: If the feature name in the DataFrame doesn't match the feature name in `explainer.feature_names`, the fix won't detect it. This should be handled by proper initialization (line 2417).

3. **Condition Type**: The fix assumes binary features use `direction == 1` (>) conditions. If a binary feature uses `direction == 0` (<=) conditions, it won't be detected. However, for binary features that are present (value=1), the condition should be "feature > 0" or "feature > threshold".

## Testing Recommendation

To verify the fix works:

1. Run Step 8 on a cohort that previously showed all binary features with `causal_importance = 0.00`
2. Check if binary features now have non-zero `causal_importance`
3. Verify that features appearing in AXPs are correctly detected

## Expected Behavior After Fix

- Binary features that appear in AXPs should have `causal_importance > 0`
- Binary features that don't appear in AXPs may still have `causal_importance = 0` (this is expected)
- The fix should correctly detect features even when the AXP computation doesn't change (conservative approximation removed)
