# FFA Analysis Pipeline and Timeline

## Overview

This document consolidates information on the FFA pipeline data flow, intermediate/final output locations, and the complete development timeline.

---

## Table of Contents

1. [Pipeline Data Locations](#pipeline-data-locations)
2. [FFA Analysis Timeline](#ffa-analysis-timeline)

---

## Pipeline Data Locations

### Directory Structure

```
8_ffa_analysis/
├── outputs/                          ← Main outputs directory
│   ├── cohort_{name}_ageband_{band}_{timestamp}/  ← Cohort-specific outputs
│   │   ├── axp_explanations.parquet     ← Explanations (pre-pruning)
│   │   ├── ffa_results.parquet          ← FFA results (pre-pruning)
│   │   ├── final_axp_explanations.parquet  ← Explanations (post-pruning)
│   │   ├── final_ffa_results.parquet    ← FFA results (post-pruning)
│   │   ├── pruning_summary.parquet      ← Pruning statistics
│   │   ├── detailed_pruning_log.parquet ← Per-rule pruning details
│   │   ├── causal_analysis.parquet      ← Single-feature causal analysis
│   │   ├── interaction_analysis.parquet ← Multi-feature interaction analysis
│   │   ├── combined_importance.csv      ← Combined SHAP + causal + FFA scores
│   │   ├── final_model/                 ← Final model artifacts
│   │   │   ├── model.pkl                ← Trained XGBoost model
│   │   │   ├── training_metadata.json   ← Training info
│   │   │   └── feature_importance.csv   ← SHAP values
│   │   └── logs/                        ← Execution logs
│   │       ├── run_log.txt              ← Main execution log
│   │       └── error_log.txt            ← Error tracking
│   └── consolidated_results/            ← Consolidated across cohorts
│       ├── all_cohorts_ffa.parquet      ← All cohorts FFA results
│       ├── all_cohorts_interactions.parquet  ← All interactions
│       └── summary_statistics.json      ← Summary metrics
│
├── data_preparation/                 ← Intermediate data processing
│   ├── cohort_prepared/                 ← Prepared cohort data
│   │   ├── cohort_{name}_ageband_{band}.parquet
│   │   └── feature_metadata.json
│   └── model_ready/                     ← Model-ready datasets
│       ├── train_data.parquet
│       ├── test_data.parquet
│       └── feature_list.json
│
└── cache/                            ← Caching for performance
    ├── model_predictions/               ← Cached model predictions
    └── intermediate_results/            ← Cached intermediate computations
```

### Key Output Files

#### 1. Explanation Files

**axp_explanations.parquet** (pre-pruning):
- **Location**: `outputs/cohort_{name}_ageband_{band}_{timestamp}/`
- **Content**: Raw AXP explanations before pruning
- **Schema**:
  - `instance_id`: Unique instance identifier
  - `rule_id`: Rule identifier
  - `features`: List of features in rule
  - `support`: Rule support (instances covered)
  - `confidence`: Rule confidence
  - `lift`: Rule lift
  - `shap_contribution`: SHAP contribution for this rule
  - `frequency_weight`: Frequency-based weight

**final_axp_explanations.parquet** (post-pruning):
- **Location**: `outputs/cohort_{name}_ageband_{band}_{timestamp}/`
- **Content**: AXP explanations after pruning pipeline
- **Schema**: Same as axp_explanations.parquet, but with:
  - Redundant rules removed
  - Low-support rules removed
  - Interaction-based rules added
  - Rules passing all 6 pruning stages

#### 2. FFA Results Files

**ffa_results.parquet** (pre-pruning):
- **Location**: `outputs/cohort_{name}_ageband_{band}_{timestamp}/`
- **Content**: Feature importance scores before pruning
- **Schema**:
  - `feature`: Feature name
  - `ffa_score`: FFA importance score
  - `support`: Number of instances with this feature
  - `rule_count`: Number of rules containing this feature
  - `avg_shap`: Average SHAP contribution
  - `avg_confidence`: Average rule confidence

**final_ffa_results.parquet** (post-pruning):
- **Location**: `outputs/cohort_{name}_ageband_{band}_{timestamp}/`
- **Content**: Feature importance scores after pruning
- **Schema**: Same as ffa_results.parquet, but computed from pruned explanations

#### 3. Pruning Output Files

**pruning_summary.parquet**:
- **Location**: `outputs/cohort_{name}_ageband_{band}_{timestamp}/`
- **Content**: Summary statistics for each pruning stage
- **Schema**:
  - `stage`: Pruning stage (0-6)
  - `rules_before`: Number of rules before stage
  - `rules_after`: Number of rules after stage
  - `rules_removed`: Number removed
  - `removal_rate`: Percentage removed
  - `features_affected`: Features impacted by pruning
  - `execution_time`: Time taken (seconds)

**detailed_pruning_log.parquet**:
- **Location**: `outputs/cohort_{name}_ageband_{band}_{timestamp}/`
- **Content**: Per-rule pruning decisions
- **Schema**:
  - `rule_id`: Rule identifier
  - `stage`: Pruning stage where action taken
  - `action`: 'removed' or 'kept'
  - `reason`: Reason for action (e.g., "redundant", "low support")
  - `rule_features`: Features in the rule
  - `rule_support`: Rule support
  - `rule_confidence`: Rule confidence

#### 4. Causal Analysis Files

**causal_analysis.parquet**:
- **Location**: `outputs/cohort_{name}_ageband_{band}_{timestamp}/`
- **Content**: Single-feature causal intervention analysis
- **Schema**:
  - `feature`: Feature name
  - `intervention_rate`: IR(j) - fraction of changed explanations
  - `avg_explanation_change`: Average change in explanation
  - `instances_tested`: Number of instances tested
  - `is_causal`: Boolean (IR > threshold)
  - `causal_strength`: 'weak', 'moderate', 'strong'

**interaction_analysis.parquet**:
- **Location**: `outputs/cohort_{name}_ageband_{band}_{timestamp}/`
- **Content**: Multi-feature interaction analysis
- **Schema**:
  - `features`: Tuple of feature names (e.g., ('drug_A', 'drug_B'))
  - `interaction_size`: Number of features (2, 3)
  - `combined_ir`: IR(j,k) - combined intervention rate
  - `individual_ir_sum`: Sum of individual IRs
  - `interaction_effect`: IR(j,k) - sum(IR(j))
  - `interaction_type`: 'synergistic', 'antagonistic', 'redundant'
  - `instances_tested`: Number of instances tested
  - `significance`: Statistical significance (if tested)

#### 5. Combined Importance Files

**combined_importance.csv**:
- **Location**: `outputs/cohort_{name}_ageband_{band}_{timestamp}/`
- **Content**: Combined SHAP + causal + FFA importance scores
- **Schema**:
  - `feature`: Feature name
  - `shap_importance`: SHAP-based importance
  - `causal_importance`: Causal intervention rate
  - `ffa_importance`: FFA score
  - `combined_score`: Weighted combination
  - `rank_shap`: Rank by SHAP
  - `rank_causal`: Rank by causal
  - `rank_ffa`: Rank by FFA
  - `rank_combined`: Rank by combined score

#### 6. Model Files

**model.pkl**:
- **Location**: `outputs/cohort_{name}_ageband_{band}_{timestamp}/final_model/`
- **Content**: Serialized trained XGBoost model
- **Format**: Python pickle (joblib)

**training_metadata.json**:
- **Location**: `outputs/cohort_{name}_ageband_{band}_{timestamp}/final_model/`
- **Content**: Training configuration and metrics
- **Schema**:
  ```json
  {
    "cohort": "opioid_ed",
    "ageband": "0_12",
    "timestamp": "20240115_143022",
    "train_samples": 1234,
    "test_samples": 456,
    "features": 789,
    "hyperparameters": {...},
    "performance": {
      "train_auc": 0.85,
      "test_auc": 0.82,
      "train_accuracy": 0.78,
      "test_accuracy": 0.75
    }
  }
  ```

**feature_importance.csv**:
- **Location**: `outputs/cohort_{name}_ageband_{band}_{timestamp}/final_model/`
- **Content**: SHAP-based feature importance
- **Schema**:
  - `feature`: Feature name
  - `shap_mean`: Mean absolute SHAP value
  - `shap_std`: Standard deviation of SHAP values
  - `rank`: Importance rank

### Data Flow Diagram

```
Input Data (cohort parquet files)
    ↓
Data Preparation
    ↓
Model Training → model.pkl, training_metadata.json
    ↓
SHAP Analysis → feature_importance.csv
    ↓
AXP Explanation Generation → axp_explanations.parquet
    ↓
Pruning Pipeline (Stages 0-6) → detailed_pruning_log.parquet
    ↓                           → pruning_summary.parquet
    ↓
Final Explanations → final_axp_explanations.parquet
    ↓
FFA Computation → ffa_results.parquet, final_ffa_results.parquet
    ↓
Causal Analysis → causal_analysis.parquet
    ↓
Interaction Analysis → interaction_analysis.parquet
    ↓
Combined Importance → combined_importance.csv
    ↓
Consolidation → consolidated_results/
```

### File Naming Convention

**Cohort-specific outputs:**
```
outputs/cohort_{cohort_name}_ageband_{age_band}_{timestamp}/
```

**Examples:**
- `outputs/cohort_opioid_ed_ageband_0_12_20240115_143022/`
- `outputs/cohort_non_opioid_ed_ageband_13_24_20240116_091535/`

**Timestamp format:** `YYYYMMDD_HHMMSS`

### Cache Management

**Cache locations:**
- `cache/model_predictions/` - Stores model predictions for repeated analysis
- `cache/intermediate_results/` - Stores intermediate computations for resumption

**Cache invalidation:**
- Automatically invalidated if model or data changes
- Manual deletion: `rm -rf 8_ffa_analysis/cache/`

---

## FFA Analysis Timeline

### Timeline Overview

**Total Duration:** November 2023 → February 2024 (3 months)

### Phase 1: Initial Implementation (November 2023)

**Week 1-2: Basic FFA Setup**
- ✅ **Implemented basic FFA framework** (from research paper)
- ✅ **Rule-based explanation generation** (AXP)
- ✅ **Initial SHAP integration**
- **Output**: Basic working pipeline for single cohort

**Week 3-4: Rule Selection Strategy**
- ✅ **Implemented 5-set union rule selection**
  - Set 1: First 100 matched rules
  - Set 2: Random 100 rules
  - Set 3: Top 300 SHAP-filtered rules
  - Set 4: Fallback SHAP=0 rules
  - Set 5: Top 100 frequent rules
- ✅ **Rule frequency weighting** (by patient frequency)
- **Output**: `RULE_SELECTION_METHODOLOGY.md`, `RULE_FREQUENCY_WEIGHTING.md`

### Phase 2: SHAP Integration (December 2023)

**Week 1-2: SHAP Filtering**
- ✅ **Implemented SHAP-based rule filtering**
- ✅ **SHAP signal validity analysis**
- ✅ **Integrated SHAP into rule selection** (Set 3, Set 4)
- **Output**: `SHAP_FILTERING_ANALYSIS.md`, `SHAP_SIGNAL_VALIDITY.md`

**Week 3-4: Set 5 Implementation**
- ✅ **Implemented frequent rules (Set 5)**
- ✅ **Tested impact on explanation quality**
- ✅ **Validated against baseline (4-set union)**
- **Output**: `SET5_FREQUENT_RULES_IMPLEMENTATION.md`

### Phase 3: Pruning Pipeline (December 2023 - January 2024)

**Week 1-2: Initial Pruning Stages (Stage 0-2)**
- ✅ **Stage 0: SHAP-based filtering** (remove SHAP < 0)
- ✅ **Stage 1a: Redundancy removal** (exact duplicates)
- ✅ **Stage 1b: Subsumption removal** (one rule subset of another)
- ✅ **Stage 2: Support-based pruning** (remove low-support rules)
- **Output**: Initial pruning pipeline

**Week 3-4: Advanced Pruning Stages (Stage 3-6)**
- ✅ **Stage 3: Confidence-based pruning** (remove low-confidence rules)
- ✅ **Stage 4: Length-based pruning** (remove overly complex rules)
- ✅ **Stage 5: Feature interaction analysis**
- ✅ **Stage 6: Final quality check**
- **Output**: Complete 6-stage pruning pipeline

**Week 5-6: Implementation and Documentation**
- ✅ **Implemented all 9 pruning rules**
- ✅ **Created pruning pipeline diagram**
- ✅ **Comprehensive testing across cohorts**
- **Output**: `PRUNING_PIPELINE.md`, `PRUNING_RULES.md`, `PRUNING_IMPLEMENTATION_SUMMARY.md`, `PRUNING_PIPELINE_DIAGRAM.md`

### Phase 4: Causal Analysis (January 2024)

**Week 1-2: Single-Feature Causal Analysis**
- ✅ **Implemented intervention testing** (remove/modify features)
- ✅ **Computed intervention rates** (IR(j))
- ✅ **Classified causal vs non-causal features**
- **Output**: `causal_analysis.parquet` for each cohort

**Week 3-4: Multi-Feature Interaction Analysis**
- ✅ **Implemented interaction testing** (simultaneous interventions)
- ✅ **Computed interaction effects** (synergy, antagonism)
- ✅ **Classified interaction types**
- ✅ **Cohort-specific interaction sizes** (pairs for first, pairs+triplets for second)
- **Output**: `interaction_analysis.parquet` for each cohort

### Phase 5: Optimization (January - February 2024)

**Week 1-2: CPU Optimization**
- ✅ **Parallelization with 28 workers** (14 workers × 2 cohorts)
- ✅ **Achieved 20-25x speedup**
- ✅ **I/O optimization** (Parquet format)
- **Output**: `CPU_OPTIMIZATION.md`

**Week 3-4: Process Management**
- ✅ **Parallel cohort execution guidelines**
- ✅ **Process count explanation** (workers vs CPU cores)
- ✅ **CPU oversubscription analysis** (28 workers on 32 cores)
- ✅ **Monitoring and logging**
- **Output**: `PARALLEL_COHORT_EXECUTION.md`, `PROCESS_COUNT_EXPLANATION.md`, `PROCESS_VS_CPU_MONITORING.md`, `CPU_OVERSUBSCRIPTION_ANALYSIS.md`

### Phase 6: Analysis and Validation (February 2024)

**Week 1-2: Framework Robustness Assessment**
- ✅ **Evaluated framework strengths/limitations**
- ✅ **Validated on population-level dataset** (State of Virginia)
- ✅ **Temporal validation** (2016-2018 train, 2019 test)
- **Output**: `FRAMEWORK_ROBUSTNESS_ASSESSMENT.md`

**Week 3-4: Specialized Analyses**
- ✅ **Polypharmacy interaction analysis**
- ✅ **Combinatorial analysis** (rule vs feature combinations)
- ✅ **Multi-feature interaction analysis**
- **Output**: `POLYPHARMACY_INTERACTION_ANALYSIS_EXPLAINED.md`, `COMBINATORIAL_ANALYSIS.md`, `MULTI_FEATURE_INTERACTIONS.md`

**Week 5: Pipeline Documentation**
- ✅ **Documented data locations**
- ✅ **Created optimization review**
- ✅ **Timeline documentation** (this document!)
- **Output**: `PIPELINE_DATA_LOCATIONS.md`, `OPTIMIZATION_REVIEW.md`, `FFA_ANALYSIS_TIMELINE.md`

### Key Milestones

#### Milestone 1: Working Pipeline ✅ (November 2023)
- Basic FFA framework operational
- Rule-based explanations generated
- Initial SHAP integration

#### Milestone 2: Rule Selection Complete ✅ (December 2023)
- 5-set union rule selection finalized
- SHAP filtering integrated
- Frequency weighting implemented

#### Milestone 3: Pruning Pipeline Complete ✅ (January 2024)
- All 6 stages implemented
- All 9 pruning rules operational
- Comprehensive testing complete

#### Milestone 4: Causal Analysis Complete ✅ (January 2024)
- Single-feature intervention testing
- Multi-feature interaction analysis
- Synergy/antagonism detection

#### Milestone 5: Optimization Complete ✅ (February 2024)
- 20-25x speedup achieved
- Parallel execution operational
- Process management optimized

#### Milestone 6: Full Pipeline Operational ✅ (February 2024)
- End-to-end pipeline validated
- All cohorts processed successfully
- Comprehensive documentation complete

### Current Status (March 2024)

**Production-Ready Features:**
- ✅ Complete FFA pipeline (rule generation → pruning → causal analysis)
- ✅ Parallelized execution (28 workers, 20-25x speedup)
- ✅ Comprehensive pruning (6 stages, 9 rules)
- ✅ Causal and interaction analysis
- ✅ Population-level validation (State of Virginia)
- ✅ Temporal validation (2019 test set)

**In Progress:**
- 🔄 Dashboard integration (Step 9)
- 🔄 Automated reporting
- 🔄 Cross-cohort analysis

**Future Enhancements:**
- ⏳ Rare interaction targeted analysis
- ⏳ External validation (FAERS database)
- ⏳ Clinical workflow integration
- ⏳ Real-time prediction API

### Development Statistics

**Total Development Time:** ~3 months (November 2023 - February 2024)

**Lines of Code:**
- Main pipeline: ~2,000 lines (run_full_ffa_analysis.py)
- Helper functions: ~1,500 lines (symbolic_explainer.py, pruning.py)
- Analysis scripts: ~1,000 lines
- **Total**: ~4,500 lines of production code

**Documentation:**
- 21 markdown files
- ~4,621 lines of documentation
- Consolidated into 5-6 comprehensive documents (docs/Step8_FFA/)

**Testing:**
- 6 cohorts tested (opioid_ed × 3 age bands, non_opioid_ed × 3 age bands)
- Temporal validation (2016-2018 train, 2019 test)
- Population-level dataset (State of Virginia)

### Lessons Learned

#### 1. Rule Combinations vs Feature Combinations
- **Initially thought**: Rule combinations were the bottleneck
- **Discovered**: Feature combinations explode exponentially (C(N,k))
- **Solution**: Cohort-specific interaction sizes, pruning, parallelization

#### 2. SHAP Filtering is Critical
- **Initially**: Generated 10,000+ rules per instance
- **After SHAP filtering**: Reduced to ~300-500 rules
- **Impact**: 20-30x speedup in explanation generation

#### 3. Pruning Must Be Multi-Stage
- **Initially**: Single-pass redundancy removal
- **Discovered**: Needed sequential stages (each stage creates new redundancies)
- **Solution**: 6-stage pipeline with iterative pruning

#### 4. Parallelization Requires Careful Tuning
- **Initially**: Used all 32 cores (32 workers)
- **Discovered**: CPU oversubscription and context switching overhead
- **Solution**: 28 workers (optimal for 32 cores with I/O waits)

#### 5. Co-occurrence Filtering is Essential but Slow
- **Initially**: Tested all 36M combinations
- **Discovered**: Co-occurrence pruning is bottleneck (single-threaded)
- **Solution**: Increased SHAP thresholds, added combination caps

#### 6. Population-Level Data is Critical
- **Key advantage**: State of Virginia dataset provides statistical power
- **Impact**: Enables detection of moderate-to-large interactions
- **Validation**: Temporal validation (2019 test) confirms generalizability

---

## Summary

**Pipeline Data Flow:**
1. Input: Cohort parquet files
2. Model training: XGBoost models with SHAP
3. Explanation generation: AXP with pruning
4. Causal analysis: Single and multi-feature interventions
5. Output: Final importance scores and interaction analysis

**Key Output Files:**
- `final_axp_explanations.parquet` - Pruned explanations
- `final_ffa_results.parquet` - Final importance scores
- `causal_analysis.parquet` - Causal effects
- `interaction_analysis.parquet` - Drug interactions
- `combined_importance.csv` - Integrated importance scores

**Timeline:** 3 months development (November 2023 - February 2024)

**Current Status:** Production-ready pipeline with comprehensive validation

---

## Related Documentation

**Complete FFA Documentation:**
- [README_ffa_methodology.md](README_ffa_methodology.md) - Rule selection, SHAP filtering, frequency weighting
- [README_ffa_pruning.md](README_ffa_pruning.md) - 6-stage pruning pipeline with 9 rules
- [README_ffa_optimization.md](README_ffa_optimization.md) - Performance optimization and parallelization
- [README_ffa_interactions.md](README_ffa_interactions.md) - Combinatorial and interaction analysis
- [README_ffa_causal_analysis.md](README_ffa_causal_analysis.md) - Causal intervention methodology
- [README_ffa_analysis.md](README_ffa_analysis.md) - Analysis results and findings
- [README_ffa_overview.md](README_ffa_overview.md) - High-level overview and architecture
- [README_ffa_unified_schema.md](README_ffa_unified_schema.md) - Unified output schema
- [MULTI_FEATURE_INTERACTIONS.md](MULTI_FEATURE_INTERACTIONS.md) - Multi-feature interaction details

**Input Data Sources:**
- [Step 1: APCD Input Data](../Step1_Input) - Raw data preparation
- [Step 2: Cohort Creation](../Step2_Cohort) - Analytical cohort generation
- [Data Pipeline Workflow](../CrossStep_Development/README_data_pipeline_workflow.md) - Data processing workflow
- [Data Pipeline Architecture](../CrossStep_Development/README_data_pipeline_architecture.md) - Architecture and design

**Output Applications:**
- [Step 9: Risk Dashboard](../../10_risk_dashboard/docs) - Clinical dashboard implementation
