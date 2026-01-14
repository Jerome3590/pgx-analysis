# Unified Schema for Symbolic Explainers

## Overview

**Note**: FFA analysis is performed only for XGBoost models. CatBoost FFA is not performed due to CatBoost's complex hashing and CTR for categorical variables. CatBoost SHAP values are used for feature importance filtering in XGBoost FFA.

All model types (CatBoost, XGBoost, XGBoost RF) use a **unified schema** and **base class** (`BaseSymbolicExplainer`) to ensure consistency and maintainability. However, only XGBoost models are used for FFA analysis in the current workflow.

## Unified DataFrame Schema

All tree paths are represented using the same DataFrame schema:

```python
TREE_PATH_SCHEMA = [
    'tree_idx',      # Index of the tree in the ensemble
    'path_idx',      # Unique identifier for this path (leaf index or path number)
    'step_in_path',  # Step number within this path (0-based)
    'feature_idx',   # Feature index (integer)
    'feature_name',  # Feature name (string)
    'threshold',     # Split threshold (float)
    'direction',     # 0 for <= (left), 1 for > (right)
    'depth',         # Depth in tree (0 = root)
    'leaf_value',    # Leaf value (float)
    'prediction',    # Binary prediction (0 or 1)
    'path_length'    # Total length of this path
]
```

## Base Class: `BaseSymbolicExplainer`

All explainers inherit from `BaseSymbolicExplainer` which provides:

### Common Attributes
- `condition_id_map`: Maps (feature_idx, threshold, direction) → literal_id
- `id_condition_map`: Maps literal_id → (feature_idx, threshold, direction)
- `rule_clauses`: List of rule clauses (each clause is list of literal_ids)
- `rule_predictions`: List of predictions (0 or 1) for each rule
- `feature_names`: Maps feature_idx → feature_name
- `model_json`: Model JSON structure
- `logger`: Logging instance

### Common Methods (Implemented in Base Class)
- `_get_condition_literal()`: Get or create literal ID for a condition
- `_create_rules_from_dataframe()`: **Unified rule creation logic** - converts DataFrame paths to rules
- `_satisfied_rules()`: Find rules satisfied by an instance
- `_compute_axp()`: Compute minimal hitting set (AXP)
- `explain_literals()`: Get AXP literals for instance
- `explain_instance()`: Get readable explanation
- `_literal_to_text()`: Convert literal to human-readable text
- `explain_dataset()`: Generate explanations for dataset
- `_literal_condition_holds()`: Check if literal condition holds
- `_satisfied_clauses_for_instance()`: Find satisfied clauses
- `_enumerate_axps()`: Enumerate minimal hitting sets
- `validate_input_data()`: Validate input data format

### Abstract Methods (Must be implemented by subclasses)
- `fit_from_model_json()`: Parse model JSON and build rules
- `_explode_tree_to_dataframe()`: Convert tree structure to unified DataFrame schema

## Model-Specific Implementations

### CatBoost (`CatBoostSymbolicExplainer`)
- **Tree Types**: Oblivious trees (uses DataFrame approach)
- **Implementation**: `_explode_oblivious_tree_to_dataframe()` converts oblivious tree structure to unified schema
- **Special Handling**: Non-oblivious trees use fallback traversal method
- **FFA Status**: **NOT used in current workflow** - CatBoost FFA is not performed due to complex hashing and CTR. CatBoost SHAP values are used for feature importance filtering in XGBoost FFA instead.

### XGBoost (`XGBoostSymbolicExplainer`)
- **Tree Types**: Standard binary trees (uses DataFrame approach)
- **Implementation**: `_explode_tree_to_dataframe()` converts parsed XGBoost tree to unified schema
- **Special Handling**: Handles both tree dump strings and pre-parsed trees

## Benefits of Unified Schema

1. **Consistency**: All models use the same DataFrame structure for tree paths
2. **Maintainability**: Common logic (rule creation, AXP computation) is in one place
3. **Debugging**: Easy to inspect DataFrame to see all paths from any model type
4. **Extensibility**: Adding new model types only requires implementing `_explode_tree_to_dataframe()`
5. **Testing**: Can test common functionality once for all model types

## Usage Example

```python
from base_symbolic_explainer import BaseSymbolicExplainer, TREE_PATH_SCHEMA
from xgboost_axp_explainer import XGBoostSymbolicExplainer

# Note: CatBoost FFA is not performed in the current workflow
# Only XGBoost explainers are used for FFA analysis
explainer = XGBoostSymbolicExplainer(path_config)

explainer.fit_from_model_json(model_json)

# Uses the unified DataFrame schema
df_paths = explainer._explode_tree_to_dataframe(tree, tree_idx=0)
assert list(df_paths.columns) == TREE_PATH_SCHEMA

# Uses the unified rule creation logic
explainer._create_rules_from_dataframe(df_paths)
    
    # All use the same explanation methods
    explanations = explainer.explain_dataset(X, predictions=y)
```

## Schema Validation

The unified schema ensures:
- ✅ Same column names across all model types
- ✅ Same data types for each column
- ✅ Same rule creation logic
- ✅ Same AXP computation
- ✅ Same explanation format

## Migration Notes

- **Before**: Each explainer had duplicate implementations of common methods
- **After**: Common methods are in `BaseSymbolicExplainer`, model-specific logic is in subclasses
- **Breaking Changes**: None - all existing code continues to work
- **New Features**: Can be added to base class and automatically available to all model types

