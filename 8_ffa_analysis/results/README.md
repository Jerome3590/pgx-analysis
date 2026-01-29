# FFA Analysis Results

This directory contains summarized FFA (Formal Feature Attribution) analysis results for completed cohorts.

## Files

### Summary Document
- **FFA_RESULTS_SUMMARY.md** - Comprehensive summary of causal factors and interactions for all completed cohorts

### Causal Importance Files
- **causal_importance_opioid_ed_13_24.parquet** - Top causal factors for ages 13-24
- **causal_importance_opioid_ed_25_44.parquet** - Top causal factors for ages 25-44
- **causal_importance_opioid_ed_45_54.parquet** - Top causal factors for ages 45-54

### Interaction Analysis Files
- **interaction_analysis_opioid_ed_13_24.parquet** - Feature interactions for ages 13-24
- **interaction_analysis_opioid_ed_25_44.parquet** - Feature interactions for ages 25-44
- **interaction_analysis_opioid_ed_45_54.parquet** - Feature interactions for ages 45-54

## Data Schema

### Causal Importance Parquet Files
Columns:
- `feature` - Feature name
- `causal_importance` - Causal importance score (0-1)
- `support` - Number of instances where feature can be intervened
- `confidence` - Confidence score
- `median_value` - Median value of the feature
- `is_binary` - Whether feature is binary
- `intervention` - Intervention value used

### Interaction Analysis Parquet Files
Columns:
- `feature_combination` - Combined features (e.g., "drug_A|drug_B")
- `interaction_size` - Number of features in combination (2, 3, etc.)
- `combined_causal_importance` - Combined causal effect
- `sum_individual_effects` - Sum of individual univariate effects
- `interaction_effect` - Difference (combined - individual), measures synergy/antagonism
- `n_instances_tested` - Number of instances tested
- `explanation_change_rate` - Fraction of explanations that changed
- `synergy_type` - Type of interaction (synergy/antagonism)
- `binary_intervention_mode` - Intervention mode used

## Key Findings

### Universal Top Causal Factors:
1. **n_events** - Number of events (always #1)
2. **pgx_num_drugs** - PGx drug count
3. **item_drug_GABAPENTIN** - Present in all age bands
4. **item_drug_NARCAN** - Present in all age bands
5. **item_drug_BUPRENORPHINE_HYDROCHLORI** - Present in all age bands

### Top Interactions:
- Most interactions involve **n_events** (number of events)
- Strong interaction: **n_events | pgx_num_drugs**
- Most show negative interaction effects (synergy)

See **FFA_RESULTS_SUMMARY.md** for detailed analysis.

## Reading the Files

### Using Python with DuckDB:
```python
import duckdb
con = duckdb.connect()
df = con.execute("SELECT * FROM read_parquet('causal_importance_opioid_ed_13_24.parquet')").df()
con.close()
```

### Using Python with Pandas:
```python
import pandas as pd
df = pd.read_parquet('causal_importance_opioid_ed_13_24.parquet')
```

## Data Source Verification

**✅ CONFIRMED**: These files were generated on 2026-01-14 using **test data (2019)**.

**Evidence from log file** (`ffa_analysis_20260114_074605.log`):
- `Using data source: test (2019) [synced from S3]`
- Data path: `/mnt/nvme/gold/model_training_data/cohort_name=opioid_ed/event_year=2019/age_band=45-54/final_features.parquet`
- The `event_year=2019` in the path confirms test data was used

**What this means:**
- Rules were extracted from the training model (2016-2018)
- Rules were validated on **unseen test data (2019)** - ensuring generalizability
- Causal importance and interactions reflect patterns validated on test data

## Last Updated
2026-01-14
