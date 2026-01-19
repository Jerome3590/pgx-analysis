# Feature Filtering Approach

## Overview

We use a **safe feature filter** approach that:
1. **Excludes** post-target leakage features (>=80% post-F1120 ratio)
2. **Keeps** ALL features with ANY pre-F1120 presence
3. **Applies** the same feature set to both cases and controls

## Strategy

### Safe Feature Filter (Preferred)

**File**: `{cohort}_{age_band}_safe_feature_filter.json`

**Approach**: Whitelist (positive list)
- **Excludes**: 348 features with >=80% post-F1120 ratio (pure post-target leakage)
- **Keeps**: 682 features with <80% post-F1120 ratio
  - 114 pure predictive (>=80% pre-F1120)
  - 567 mixed timing (any pre-F1120, <80% post-F1120)
  - 1 low pre but not leakage

**Benefits**:
- Maximizes information available to the algorithm
- Prevents target leakage
- Ensures same feature set for cases and controls
- Includes F1120 for target creation

### Implementation

The `filter_and_refine_features.py` script:
1. Loads `safe_feature_filter.json` if available
2. Uses `all_features_to_keep` as a whitelist
3. Falls back to BupaR CSV-based filtering if JSON not found

### Usage

```bash
# Run filter and refine (automatically uses safe_feature_filter.json if available)
python filter_and_refine_features.py --cohort opioid_ed --age-band 13-24
```

The script will:
- Load aggregated feature importance from Step 3
- Apply safe feature filter (whitelist approach)
- Filter non-value-added features (administrative codes from lookup table)
- Output refined `cohort_feature_importance.csv` for Step 4a

### Feature Set Applied

**For Cases (target=1)**:
- Use ONLY features from `all_features_to_keep` (682 features)

**For Controls (target=0)**:
- Use the SAME features from `all_features_to_keep` (682 features)

This ensures fair comparison and prevents bias from different feature sets.

## File Locations

- **Safe Feature Filter JSON**: `3b_feature_importance_eda/outputs/{cohort}/{age_band}/{cohort}_{age_band}_safe_feature_filter.json`
- **Refined Feature Importance**: `3b_feature_importance_eda/outputs/{cohort}/{age_band}/{cohort}_{age_band}_cohort_feature_importance.csv`

## Creating the Safe Feature Filter

Run:
```bash
python create_safe_feature_filter_json.py --cohort opioid_ed --age-band 13-24
```

This creates the JSON file with:
- `all_features_to_keep`: List of 682 features to use
- `all_features_to_exclude`: List of 348 leakage features
- Organized by type (ICD, CPT, Drug) and timing category
