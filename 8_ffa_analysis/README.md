## 8_ffa_analysis – Formal Feature Attribution Analysis

This directory contains the **Formal Feature Attribution (FFA) Analysis** framework for interpreting gradient-boosted decision tree models through symbolic logic extraction, anchored explanations, and causal analysis.

### Quick Overview

FFA Analysis transforms opaque models into interpretable symbolic rules suitable for formal verification and causal inference.

**Key Capabilities:**
- **Symbolic Rule Extraction**: Convert tree structures into Boolean logic formulas
- **Model-Specific FFA Implementation**:
  - **XGBoost FFA**: Direct rule extraction from JSON model structure
  - **CatBoost FFA**: **NOT performed** due to CatBoost's complex hashing and CTR (Counter-based Target Statistics) for categorical variables that make direct rule extraction difficult
  - **CatBoost SHAP**: Used for feature importance filtering (not for FFA rule extraction)
- **Anchored Explanations (AXP)**: Generate instance-level explanations using rule matching
  - **Rule Selection Logic**: Union of five sets:
    1. **First 100 matched rules** - Common patterns
    2. **Random sample of 100 matched rules** - Diversity and coverage
    3. **Top-K SHAP rules with percentile threshold** - Hybrid approach:
       - Takes top 300 rules by SHAP importance score
       - OR all rules above 10th percentile threshold (whichever captures more rules)
       - Uses SHAP importance from **both XGBoost and CatBoost** to filter/prioritize rules
       - Balances performance (limits rule count) with coverage (doesn't miss important rules)
    4. **Fallback SHAP=0 rules** - Up to 100 rules with SHAP=0 that aren't already covered
    5. **Top 100 most frequent rules** - Rules that match most often across dataset (NEW: Set 5)
       - Ensures frequent patterns are captured, matching reference implementation's implicit weighting
       - Rule frequencies are computed once before explanation generation
  - **SHAP Requirement**: SHAP values from Step 7 (both XGBoost and CatBoost) are required (raises error if not available)
  - **Performance Optimization**: Limits rule sets to ~300-500 unique rules for efficient AXP computation
- **Causal Analysis**: Measure causal responsibility through counterfactual analysis
  - **Model-Based Causal Importance**: Measures how interventions on features affect the model's explanations and predictions
  - **Single-Feature Causal Analysis**: Tests individual feature effects
  - **Multi-Feature Interaction Analysis**: Tests combinations of features (pairs, triplets, etc.) to detect synergies/antagonisms
    - **Default Configuration**: Enabled by default (`enable_interaction_analysis: True`)
    - **Cohort-Specific Interaction Sizes**:
      - **First cohort (`opioid_ed`)**: Tests pairs only (size 2)
      - **Second cohort (`non_opioid_ed`/`polypharmacy`)**: Tests pairs and triplets (size 2 and 3)
      - Prevents combinatorial explosion for first cohort while allowing higher-order interactions for polypharmacy
    - **Feature Selection**: Includes ALL features with SHAP > 0 OR FFA > 0 OR causal > 0 (no top_k limit)
      - Features sorted by combined importance (SHAP + causal + FFA) for prioritization
      - Safe for drug-only features where all drugs with any importance signal should be tested
    - **Drug Interaction Calculator**: For `non_opioid_ed` cohort (drug-only features), this serves as a drug interaction causal calculator for ED visits
      - Identifies which drug combinations causally increase ED visit risk
      - Measures synergy/antagonism effects (positive = synergy, negative = antagonism)
      - Output: `interaction_analysis.parquet` with drug-drug and drug-drug-drug interaction effects
  - **SHAP/FFA/Causal Filtering**: Only tests combinations of features with ANY importance > 0 (SHAP OR FFA OR causal)
- **Feature Importance**: Calculate importance scores from explanations and causal effects

### Core Components

- **`utility_scripts/run_full_ffa_analysis.py`** - Main script to run complete FFA analysis workflow (moved to utility_scripts for better organization)
- **`ffa_analysis.py`** - Core FFA analysis functions
- **`base_symbolic_explainer.py`** - Base class for unified symbolic explainers
- **`catboost_axp_explainer.py`** - CatBoost-specific explainer implementation
- **`xgboost_axp_explainer.py`** - XGBoost-specific explainer implementation
- **`combined_causal_analysis.py`** - Dual-approach causal analysis (explainer-based + probability-based)
- **`create_visualizations.py`** - Generate static visualizations
- **`interactive_risk_explorer.py`** - Generate interactive Plotly dashboards

### Quick Start

```bash
# Run complete FFA analysis for all models
python run_full_ffa_analysis.py

# Generate visualizations
python create_visualizations.py

# Run combined causal analysis
python combined_causal_analysis.py

# Create interactive dashboards
python interactive_risk_explorer.py
```

### Documentation

For detailed documentation, see [`docs/Step9_FFA/`](../docs/Step9_FFA/):

- **[README_ffa_analysis.md](../docs/Step9_FFA/README_ffa_analysis.md)** - Complete FFA analysis framework overview
- **[README_ffa_causal_analysis.md](../docs/Step9_FFA/README_ffa_causal_analysis.md)** - Dual-approach causal analysis guide
- **[README_ffa_unified_schema.md](../docs/Step9_FFA/README_ffa_unified_schema.md)** - Unified schema for symbolic explainers

See [`docs/Step9_FFA/README.md`](../docs/Step9_FFA/README.md) for complete documentation index.

### Architecture

The FFA pipeline follows three phases:
1. **Model Ingestion & Feature Mapping** - Load models and extract feature information
2. **Symbolic Logic Extraction** - Convert tree paths to PySAT formulas
3. **Explanation & Analysis** - Generate explanations, calculate importance, perform causal analysis

### XGBoost JSON → DataFrame → Rules (Current Implementation)

For the leakage-filtered final models (e.g., `opioid_ed / 13-24`), we now use a **DataFrame-centric** path for XGBoost FFA:

- **Model export (step 6_final_model)**:
  - After MC-CV and final refit, `run_final_model.py` exports an FFA-friendly JSON at:
    - `6_final_model/outputs/{cohort}/{age_band_fname}/final_model_json/{cohort}_{age_band_fname}_final_model_xgboost.json`
  - The JSON has a minimal, explainer-focused schema:
    - `model_type`: `"xgboost"`
    - `feature_names`: list of numeric feature columns used in training (ordered as in the final feature matrix).
    - `trees`: list of **text tree dumps** from `booster.get_dump(dump_format="text")`, one string per tree.

- **Explainer initialization (run_full_ffa_analysis.py)**:
  - When we call `initialize_explainer(...)` for `model_type="xgboost"`:
    - We build a `PathConfig` pointing at the JSON and the leakage-filtered feature CSV from `6_final_model`.
    - We pass the **DataFrame column names** from the final features into the explainer:
      - `feature_names=list(X.columns)`
    - The `XGBoostSymbolicExplainer` receives these names and keeps them as its `feature_names` mapping.

- **JSON → DataFrame conversion (xgboost_axp_explainer.py)**:
  - `fit_from_model_json(model_json)` now:
    - Uses `explainer.feature_names` if already set, otherwise falls back to `model_json["feature_names"]` or infers `f0`, `f1`, ... when needed.
    - Iterates over `model_json["trees"]` (the text dumps) and, for each tree:
      - Parses the dump into a structured tree.
      - Calls `_explode_tree_to_dataframe(parsed_tree, tree_idx)` to turn all root-to-leaf paths into a **pandas DataFrame** (`df_paths`).
        - Each row represents one decision path with feature index, thresholds, and leaf prediction.
      - Calls `_create_rules_from_dataframe(df_paths)` to convert the DataFrame rows into symbolic CNF clauses (`rule_clauses`, `rule_predictions`).
    - This DataFrame path is the primary path; when it fails for a tree, we fall back to a direct recursive traversal of the parsed tree.

- **Why this matters**:
  - The explainer no longer depends on a fragile, version-specific XGBoost JSON schema.
  - Feature names are **guaranteed to align** with the final model’s feature matrix, since they come from the same DataFrame used for training.
  - The DataFrame representation of each tree path makes debugging, inspection, and downstream analysis (e.g., exporting rules or joining back to feature engineering outputs) much easier and more robust across XGBoost versions.

### Output Structure

```
outputs/
├── {cohort}/
│   └── {age_band}/
│       ├── {model_type}/
│       │   ├── analysis_summary.json
│       │   ├── axp_explanations.csv
│       │   └── feature_importance_axp.csv
│       ├── causal_analysis/
│       │   ├── causal_importance_*.csv
│       │   ├── causal_analysis_*.png
│       │   └── intervention_effects_radar_chart.html
│       ├── visualizations/
│       │   ├── feature_importance_comparison.png
│       │   ├── normalized_importance.png
│       │   └── ...
│       └── interactive/
│           ├── dropdown_dashboard.html
│           └── feature_slider_dashboard.html
```

### Key Features

- **Deterministic robustness due to rules**: Explanations are grounded in symbolic Boolean rules. Same instance → same matched rules → same explanation, making outcomes reproducible and verifiable and well-suited to formal and causal analysis.
- **XGBoost explainer only for rules**: FFA rule extraction and the symbolic explainer use **only the XGBoost model** (best XGB variant from Step 6). CatBoost is not used as an explainer because CatBoost’s hashing of categorical variables makes it too hard to explain symbolically.
- **SHAP values**: Come from the **best model** selected in Step 6:
  - **If best model is XGBoost**: SHAP values and the XGBoost explainer both use that same model.
  - **If best model is CatBoost**: SHAP values use CatBoost plus the best XGBoost (for consensus filtering); rule extraction still uses only the **XGBoost explainer** (best XGB variant).
- **SHAP-Augmented Rule Filtering**: Uses SHAP importance (from the best model(s) as above) to filter/prioritize rules in XGBoost FFA.
- **Unified Schema**: Consistent representation across XGBoost model types
- **Dual Causal Analysis**: Explainer-based and probability-based methods
- **Interactive Dashboards**: Plotly-based risk exploration tools
- **Formal Verification**: SAT solver integration for consistency checking

### The Consensus Filter Philosophy

While CatBoost FFA is not performed due to technical limitations, this design choice functions as a **deliberate quality control mechanism**:

- **Model Agreement**: Features must be detected by CatBoost (SHAP > 0) **AND** describable by XGBoost (symbolic rule existence)
- **Robustness Over Sensitivity**: Filters out model-specific artifacts and overfitting patterns
- **Logical Translatability**: Ensures all causal recommendations can be expressed as interpretable Boolean logic
- **Clinical Confidence**: Produces high-confidence candidates validated by multiple model architectures

**Trade-off**: May miss rare variants found only by CatBoost, but ensures all features entering causal analysis are robust and interpretable. See [`docs/Step10_Results/README_combined_ffa_shap_causal_analysis.md`](../docs/Step10_Results/README_combined_ffa_shap_causal_analysis.md) for detailed explanation.

---

## Model-Based Causal Importance

### Overview

**Model-based causal importance** measures how interventions on features affect the model's explanations and predictions. Unlike correlation-based feature importance (e.g., SHAP, permutation importance), causal importance uses counterfactual reasoning to identify features that causally drive the model's decision-making process.

### What It Measures

Causal importance answers the question: **"When I change this feature, how much does the model's explanation change?"**

The analysis performs interventions on features and measures the resulting change in explanations (AXP - Abductive Explanations):

1. **Interventions**:
   - **Continuous features**: Set to median value (neutral baseline)
   - **Binary features**: Flip values (0→1, 1→0)
   - Creates a counterfactual dataset where the feature is modified

2. **Change Measurement**:
   - Generates explanations (AXP) for original and modified instances
   - Calculates the fraction of instances where explanations changed
   - **Causal Importance Score (IR)** = Fraction of instances with changed explanations
   - **Support** = Number of intervenable instances (instances where intervention can be applied)
   - **Confidence** = Same as causal importance (fraction of intervenable instances that changed)

3. **Higher Score = Stronger Causal Effect**:
   - **Causal Importance (IR)**: 
     - Score of 0.0 = Feature has no causal effect (explanations unchanged)
     - Score of 1.0 = Feature always causes explanation changes (perfect causal effect)
     - Score of 0.5 = Feature causes explanation changes in 50% of instances
   - **Support**:
     - Higher support = More instances tested = More reliable estimate
     - Low support (< 10) = Less reliable (few instances available for intervention)
   - **Confidence**:
     - Same interpretation as causal importance
     - `confidence = 1.0` means feature always affects explanations when present
     - `confidence = 0.5` means feature affects explanations in 50% of intervenable instances

### Why This Captures True Signal

**Model-based causal importance** captures true signal better than correlation-based methods because:

1. **Filters Spurious Correlations**:
   - Features that correlate with the outcome but don't causally affect predictions get low scores
   - Only features that actually drive the model's reasoning receive high scores

2. **Measures Intervention Effects**:
   - Uses counterfactual reasoning: "What if this feature changed?"
   - Tests actual causal mechanisms rather than statistical associations

3. **Model-Based Causal Inference**:
   - Identifies features the model actually relies on for its reasoning
   - Focuses on features that causally affect the model's explanations and predictions

4. **More Robust Than Simple Importance**:
   - More robust than SHAP or permutation importance
   - Measures intervention effects rather than just associations
   - Filters out features that are correlated but not causally relevant

### Important Distinction: Model-Based vs. True Causal Inference

**Model-Based Causal Inference** (what we measure):
- Measures: "What features causally affect the **model's predictions**?"
- Answers: Which features drive the model's decision-making process
- Use case: Understanding model behavior, identifying robust features

**True Causal Inference** (what we don't measure):
- Would measure: "What features causally affect the **true outcome**?"
- Would require: Randomized controlled trials (RCTs) or natural experiments
- Use case: Clinical decision-making, policy recommendations

### Why Rule Grouping is Robust for Binary Outcomes

The grouped comparison approach is particularly robust for binary classification because:

1. **Class-Specific Rule Matching**:
   - Rules are matched per predicted class (0 or 1)
   - Each instance's rules are filtered by its predicted class before grouping
   - Class 0 instances are grouped separately from Class 1 instances

2. **Deterministic Grouping**:
   - Instances with **identical rules AND same predicted class** form a group
   - Same rules → same AXP (for that class), ensuring consistency
   - Group key = sorted tuple of rule IDs + predicted class (implicit)

3. **Efficiency Without Loss of Accuracy**:
   - **Without grouping**: O(n) AXP computations (one per instance)
   - **With grouping**: O(g) AXP computations where g << n (groups << instances)
   - Typical reduction: 10,000 instances → ~100-500 groups (20-100x reduction)
   - **No accuracy loss**: Instances in same group have identical rules, so identical AXP

4. **Handles Binary Feature Interventions**:
   - For binary features, intervention removes feature (1→0) only on instances where feature is present
   - Only instances with feature=1 are tested, ensuring realistic counterfactuals
   - If removal causes rule change → new group → AXP recomputed
   - Even if rules don't change, AXP is recomputed to detect features that appear in explanations

5. **Full AXP Recomputation**:
   - Even when rules don't change after intervention, AXP is recomputed
   - This ensures we detect features that appear in explanations but don't change rule matching
   - Previously, a conservative approximation assumed AXP unchanged when rules didn't change, causing all binary features (drugs/ICDs) to have 0.0 causal importance
   - The fix ensures accurate causal importance for all features, including drugs that appear in AXP

**Bottom Line**: Grouping is robust for binary outcomes because it respects class boundaries and ensures instances with identical rule patterns get identical explanations, while dramatically improving computational efficiency.

### How Partial Condition Satisfaction is Handled

**Important**: Rules use **strict AND logic** - a rule matches only if **ALL conditions are satisfied**.

1. **Rule Matching Logic**:
   - Each rule is a **conjunction** (AND) of conditions: `condition1 AND condition2 AND condition3`
   - A rule matches an instance **only if ALL conditions are true**
   - If a rule has 3 conditions but only 2 are satisfied → **rule does NOT match**
   - There is **no partial matching** - it's all-or-nothing

2. **Example**:
   ```
   Rule: (age > 25) AND (drug_count > 3) AND (icd_code == "E11")
   
   Instance A: age=30, drug_count=5, icd_code="E11" → ✅ Rule MATCHES (all 3 conditions true)
   Instance B: age=30, drug_count=5, icd_code="E10" → ❌ Rule DOES NOT MATCH (only 2/3 conditions true)
   Instance C: age=30, drug_count=2, icd_code="E11" → ❌ Rule DOES NOT MATCH (only 2/3 conditions true)
   ```

3. **Grouping Behavior**:
   - Instances are grouped by their **complete set of matched rules**
   - If two instances have different partial matches (e.g., Instance B vs Instance C above), they may match different rules entirely
   - Instances with **identical complete rule sets** are grouped together
   - This ensures **deterministic grouping** - same rules → same group → same AXP

4. **Intervention Effects**:
   - When a feature is intervened (e.g., set to median), it may cause:
     - A previously matched rule to **no longer match** (one condition now false)
     - A previously unmatched rule to **now match** (one condition now true)
   - This creates a **new group** with different rules → AXP recomputed
   - The grouping correctly captures these changes

5. **Why This is Robust**:
   - **No ambiguity**: Rules either match completely or don't match at all
   - **Deterministic**: Same conditions → same rule matches → same group
   - **Sensitive to changes**: Any condition change can alter rule matches and create new groups
   - **Efficient**: Only fully-matched rules are considered, avoiding complex partial-match logic

**Key Insight**: The strict AND logic means that partial condition satisfaction doesn't create "partial groups" - instead, instances with different partial matches end up matching **different, simpler rules** that only require the subset of conditions they satisfy. This makes the grouping approach both robust and efficient.

**Example of Rule Hierarchy**:
```
Rule A (complex): (age > 25) AND (drug_count > 3) AND (icd_code == "E11")
Rule B (simpler): (age > 25) AND (drug_count > 3)
Rule C (simpler): (age > 25) AND (icd_code == "E11")

Instance with age=30, drug_count=5, icd_code="E10":
- Rule A: ❌ Does NOT match (icd_code condition fails)
- Rule B: ✅ MATCHES (both age and drug_count conditions satisfied)
- Rule C: ❌ Does NOT match (icd_code condition fails)

→ Instance matches Rule B (the simpler rule for the subset it satisfies)
→ Instance is grouped with other instances that match Rule B
```

This is exactly how decision trees work - each rule represents a **complete path** through the tree. If you don't satisfy all conditions in one path, you match a different path (different rule) that corresponds to the conditions you do satisfy. The grouping correctly captures this by grouping instances by their **complete set of matched rules**.

### Technical Implementation

The causal analysis is implemented in `perform_causal_analysis()` in `run_full_ffa_analysis.py`:

1. **Feature Selection**:
   - Filters to features with FFA importance > 0
   - Only analyzes features that appear in model explanations

2. **Intervention Creation**:
   - For each feature, creates modified dataset with intervention applied
   - **Binary features**: Only test instances where feature is present (value=1), remove feature (set to 0)
     - Normalized by `|S_f|` (number of instances with feature=1), not `N` (total instances)
     - Skips features with no instances where feature=1
   - **Continuous features**: Set to median value
   - **Multi-feature interactions**: Only test instances where ALL binary features in combination are present

3. **Explanation Comparison**:
   - Generates AXP explanations for original and modified instances
   - Uses **grouped comparison** for efficiency (instances with same rules grouped together)
   - Measures fraction of instances where explanations changed
   
   **Grouping Robustness for Binary Outcomes**:
   - Rules are **class-specific**: Each instance's rules are matched for its predicted class (0 or 1)
   - Instances with **same rules AND same predicted class** are grouped together
   - AXP is computed once per group, then applied to all instances in that group
   - This is robust for binary classification because:
     - Class 0 instances are grouped separately from Class 1 instances
     - Rules are filtered by predicted class before grouping
     - Same rules → same AXP (for that class), ensuring consistency
   - **Performance benefit**: Reduces computation from O(n) to O(g) where g << n (groups << instances)
   - **Full AXP recomputation**: Even when rules don't change, AXP is recomputed to detect features that appear in explanations but don't change rule matching (fixes conservative approximation issue)

4. **Causal Score Calculation**:
   - `causal_importance` = Fraction of instances with changed explanations (IR - Intervention Rate)
   - `support` = Number of intervenable instances (Support - number of instances where intervention can be applied)
   - `confidence` = Same as `causal_importance` (fraction of intervenable instances that changed)
   - Higher score = Feature has stronger causal effect on model's reasoning
   - Higher support = More reliable causal estimate (more instances tested)

### Output Format

The causal importance results are saved to:
- `outputs/{cohort}/{age_band}/xgboost/causal_importance.parquet`

Columns:
- `feature`: Feature name
- `causal_importance`: Causal importance score (0.0 to 1.0) - **IR(j)** (Intervention Rate)
- `support`: Number of intervenable instances - **Support(j)**
  - For binary features (remove_only): Number of instances where `feature == 1`
  - For binary features (add_only): Number of instances where `feature == 0`
  - For continuous features: Total sample size
- `confidence`: Confidence score (0.0 to 1.0) - Fraction of intervenable instances where intervention caused change
  - Same as `causal_importance` in our implementation
  - `confidence = changes / support`
- `median_value`: Median value used for intervention (continuous features)
- `is_binary`: Whether feature is binary (0/1)
- `intervention`: Description of intervention applied

**Metrics Explained:**
- **Support (Support(j))**: Number of instances available for intervention. Higher support = more reliable causal estimate.
- **Confidence**: Fraction of intervenable instances where the intervention caused a change. `confidence = 1.0` means the feature always affects explanations when present.
- **Causal Importance (IR(j))**: Same as confidence - fraction of instances with changed explanations after intervention.

See [`docs/Step9_FFA/README_ffa_causal_analysis.md`](../docs/Step9_FFA/README_ffa_causal_analysis.md) for detailed explanation of support and confidence metrics.

### Top 10 Causal Importance Features

After Step 8 completes, the workflow automatically prints the top 10 causal importance features:

```
================================================================================
TOP 10 CAUSAL IMPORTANCE FEATURES
================================================================================
   1. feature_name_1                                        1.000000
   2. feature_name_2                                        0.950000
   ...
  10. feature_name_10                                       0.850000
================================================================================
```

These features represent the features that, when changed, most strongly affect the model's explanations and predictions.

**Interpreting Results:**
- **Score of 1.000000**: Feature appears in AXP for all instances where it's present, and removing it always changes the explanation
- **Score of 0.940000**: Feature affects explanations in 94% of intervenable instances
- **Support**: Check the `support` column in `causal_importance.parquet` to see how many instances were tested
  - Higher support = more reliable estimate
  - Low support (< 10) = less reliable (few instances available for intervention)

### When to Use Causal Importance

**Use causal importance when**:
- You want to identify features that causally drive model predictions
- You need to filter out spurious correlations
- You want robust features for clinical decision-making
- You need interpretable features that can be expressed as Boolean logic

**Don't use causal importance when**:
- You need true causal inference from observational data (requires RCTs)
- You want to understand population-level causal effects
- You need to make policy recommendations without experimental validation

### Feature Pruning Pipeline

For detailed guides on **where pruning belongs** in the FFA pipeline and **how to implement pruning rules**, see:

- **[PRUNING_PIPELINE.md](PRUNING_PIPELINE.md)** - Complete pruning stage mapping and implementation status
- **[PRUNING_PIPELINE_DIAGRAM.md](PRUNING_PIPELINE_DIAGRAM.md)** - Visual pipeline diagram with code locations
- **[PRUNING_RULES.md](PRUNING_RULES.md)** - Detailed pruning rules with implementation code examples

**Key principle:**
> **Never prune before you measure causality.  
> Always prune before combinatorics.  
> Only early-stop after you've committed.**

**Current status:**
- ✅ **Binary intervention mode consistency**: Univariate and interaction analysis both use the same `binary_intervention_mode` (default: `remove_only`)
- ⚠️ **Stage 2.5 pruning gate**: NOT YET IMPLEMENTED (highest priority)
- ⚠️ **Stage 3 interaction pruning**: PARTIALLY IMPLEMENTED (only SHAP filtering, missing co-occurrence and capping)
- ⚠️ **Stage 4 runtime pruning**: PARTIALLY IMPLEMENTED (basic mask filtering, missing early stopping)

**Recommended pruning rules:**
1. **Prevalence filter**: Require `#(x=1) ≥ min_support` for binary features (removal mode)
2. **AXP coverage**: Require `coverage ≥ min_coverage` (already computed, not yet used for pruning)
3. **Importance-union**: Test if `SHAP > 0 OR FFA > 0` (currently only checks FFA)
4. **Co-occurrence**: Require `#(A=1 & B=1) ≥ min_cooccur` for interaction pairs
5. **Cap combinations**: Limit to top-K combinations per size to prevent explosion

### References

For detailed technical documentation, see:
- [`docs/Step9_FFA/README_ffa_causal_analysis.md`](../docs/Step9_FFA/README_ffa_causal_analysis.md) - Dual-approach causal analysis guide
- `run_full_ffa_analysis.py` - Implementation of `perform_causal_analysis()`
- `PRUNING_PIPELINE.md` - Feature pruning pipeline guide