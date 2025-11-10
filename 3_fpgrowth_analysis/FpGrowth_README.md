# FP-Growth Analysis Module

## Overview

The FP-Growth Analysis Module implements a comprehensive drug pattern mining system using the Frequent Pattern Growth (FP-Growth) algorithm. This module provides **two complementary approaches** for drug association analysis in the pharmacovigilance pipeline:

## ⚠️ Cohort-Specific Feature Generation (v2+)

**All cohort-specific features, metrics, and encodings are now built from scratch for each cohort.**

- No global drug names or metrics are reused for cohort-specific outputs.
- Each cohort's drug features, metrics, and encodings are generated using only the drugs present in that cohort's data.
- This ensures true independence and correctness for downstream process mining and ML.

The pipeline logic for cohort processing now matches the group/global pipeline, but always operates on cohort-specific data.

1. **Global FP-Growth**: Creates universal drug encoding features for machine learning models
2. **By-Cohort FP-Growth**: Discovers cohort-specific drug patterns for process mining

## 🎯 Purpose & Use Cases

### Global FP-Growth → CatBoost Feature Engineering
- **Purpose**: Creates universal drug encoding features that work across all cohorts
- **Why Essential**: CatBoost models require consistent feature spaces - same drug encodings across training/validation/test sets
- **Output**: Global drug encoding map (`s3://pgxdatalake/global_fpgrowth/drug_encoding_map.json`)
- **Benefit**: Population-level drug pattern insights become numerical features for ML

### By-Cohort FP-Growth → BupaR Process Mining
- **Purpose**: Discovers cohort-specific drug patterns and treatment sequences
- **Why Essential**: Different cohorts (ED vs non-ED, age groups) have fundamentally different care pathways
- **Output**: Cohort-specific association rules and network visualizations
- **Benefit**: Reveals how prescribing patterns differ between populations

## 📁 Module Structure

```
fpgrowth_analysis/
├── README.md                          # This documentation
├── run_fpgrowth.py                     # Core FP-Growth implementation
├── fpgrowth_by_cohort.py              # By-cohort pipeline orchestrator
├── fpgrowrh_global.py                 # Global pipeline script
└── fpgrowth_global_notebook.ipynb     # Interactive Jupyter analysis
```

## 🚀 Quick Start

### Option 1: Interactive Jupyter Notebook (Recommended)
```bash
# Open the comprehensive notebook
jupyter notebook fpgrowth_global_notebook.ipynb
```

The notebook includes:
- Global FP-Growth pipeline (Cells 1-11)
- By-cohort FP-Growth analysis (Cells 12-13)
- Comparative analysis (Cell 14)

### Option 2: Command Line Execution

#### Global FP-Growth
```python
from fpgrowth_analysis.run_fpgrowth import run_fpgrowth_global, create_fpgrowth_logger

logger = create_fpgrowth_logger()
results = run_fpgrowth_global(logger)
print(f"Processed {results['unique_drugs']} drugs, found {results['itemsets']} itemsets")
```

#### By-Cohort FP-Growth
```python
from fpgrowth_analysis.fpgrowth_by_cohort import process_features

# Process specific cohort
process_features(
    use_builtin=True, 
    max_workers=2, 
    support_threshold=0.05, 
    cohort_name="ed_non_opioid"
)
```

#### Individual Cohort Processing
```bash
python run_fpgrowth.py --path "s3://pgxdatalake/cohort_clean/ed_non_opioid/..." --support 0.05
```

## 🔧 Technical Implementation

### Core Functions (`run_fpgrowth.py`)

#### Global Pipeline Functions:
- `run_fpgrowth_global(logger)` - Main global pipeline
- `extract_global_drug_names(logger)` - Extract all unique drugs
- `create_global_drug_transactions(logger)` - Create patient-level transactions
- `create_global_encoding_map(logger, itemsets, rules)` - Generate encoding map
- `save_global_fpgrowth_results(logger, itemsets, rules, encoding_map)` - Save to S3

#### By-Cohort Functions:
- `feature_engineer(cohort_name, age_band, event_year, paths, logger)` - Process single cohort
- `process_cohort_feature_engineer(...)` - Orchestrate cohort processing

#### Utility Functions:
- `create_fpgrowth_logger(name)` - Logger with same pattern as create_cohort.py
- `load_global_encoding_map(logger)` - Load saved encoding map
- `get_drug_metrics_from_rules(drug, rules, logger)` - Extract drug-specific metrics

### Dependencies

#### Required Libraries:
```python
# Core ML libraries
from mlxtend.frequent_patterns import fpgrowth, association_rules
from mlxtend.preprocessing import TransactionEncoder

# Project utilities
from helpers.common_imports import s3_client, get_logger, pd
from helpers.duckdb_utils import get_duckdb_connection, execute_duckdb_query
from helpers.drug_name_utils import encode_drug_name, save_drug_encoding_map
from helpers.s3_utils import save_to_s3_json, save_to_s3_parquet
from helpers.fpgrowth_utils import *
from helpers.visualization_utils import create_network_visualization
```

#### Installation:
```bash
pip install mlxtend pandas numpy duckdb boto3
```

## 📊 Pipeline Workflows

### Global FP-Growth Workflow:
1. **Extract** → Get all unique drug names from pharmacy dataset
2. **Transform** → Create patient-level drug transactions
3. **Encode** → Apply TransactionEncoder for FP-Growth input format
4. **Mine** → Run FP-Growth algorithm (min_support=0.005)
5. **Rules** → Generate association rules (min_confidence=0.01)
6. **Encode** → Create drug encoding map with FP-Growth metrics
7. **Save** → Store results to S3 for downstream ML models

### By-Cohort Workflow:
1. **Discover** → List available cohort paths from S3
2. **Check** → Verify which cohorts need processing
3. **Process** → Run FP-Growth for each cohort individually
4. **Extract** → Generate cohort-specific drug tokens
5. **Mine** → Apply FP-Growth with fallback support levels
6. **Build Features** → For each cohort, build all drug metrics/features (linguistics, support, confidence, etc.) from scratch using only that cohort's data
7. **Visualize** → Create network graphs for drug associations
8. **Store** → Save cohort-specific results partitioned by cohort/age/year

## 📈 Output Structure

### Global Results (`s3://pgxdatalake/global_fpgrowth/`):
```
global_fpgrowth/
├── drug_encoding_map.json     # Universal drug encodings for ML
├── global_itemsets.json       # All frequent drug combinations
├── global_rules.json          # Association rules with confidence/lift
└── global_metrics.json        # Summary statistics
```

### By-Cohort Results (Multiple S3 locations):
```
fpgrowth_features/             # Processed feature datasets
├── cohort_name=ed_non_opioid/
│   ├── age_band=25-34/
│   │   └── event_year=2020/
│   └── age_band=35-44/
└── cohort_name=opioid_ed/

itemsets/                      # Frequent itemsets by cohort
rules/                         # Association rules by cohort
drug_networks/                 # Network visualizations
feature_manifests/             # Processing metadata
```

## 🏃‍♂️ Parallelization & Multiprocessing
### Sequential Cohort Processing (Recommended for Large Jobs)

To avoid overloading the system, you can process one cohort at a time, using all available processes for that cohort, then move to the next cohort after completion. This ensures you never exceed your process limit and makes logs/outputs easier to manage.

**Pattern:**

```python
from concurrent.futures import ProcessPoolExecutor, as_completed

def build_jobs_for_cohort(cohort):
    # Return a list of job dicts for all age_band/event_year for this cohort
    ...

for cohort in ["opioid_ed", "non_opioid_ed"]:
    jobs = build_jobs_for_cohort(cohort)
    with ProcessPoolExecutor(max_workers=30) as executor:
        futures = [executor.submit(run_single_cohort, job) for job in jobs]
        for future in as_completed(futures):
            result = future.result()
            print(f"Cohort {cohort} job complete: {result}")
    print(f"All jobs for {cohort} complete.")
```

**Benefits:**
- Never more than 30 jobs running at once
- No resource contention between cohorts
- Easier to debug and monitor

For details on parallel and SQS-based execution, see [`Parallelization_README.md`](./Parallelization_README.md).

## ⚙️ Configuration

### Global FP-Growth Parameters:
```python
# Configurable in run_fpgrowth_global()
min_support = 0.005           # Minimum support threshold
min_confidence = 0.01         # Minimum confidence for rules
timeout = 300                 # Processing timeout (seconds)
```

### By-Cohort Parameters:
```python
# Configurable in fpgrowth_by_cohort.py
support_threshold = 0.05      # Higher threshold for cohort-specific patterns
max_workers = 2               # Parallel processing workers
TOP_K = 25                    # Top frequent itemsets to extract
```

## 🔍 Key Differences: Global vs By-Cohort

| Aspect | Global FP-Growth | By-Cohort FP-Growth |
|--------|------------------|-------------------|
| **Data Scope** | Entire pharmacy dataset | Individual cohorts |
| **Purpose** | ML feature engineering | Process mining analysis |
| **Support Threshold** | 0.005 (lower for coverage) | 0.05 (higher for significance) |
| **Output Format** | Universal encoding map | Cohort-specific patterns |
| **Use Case** | CatBoost consistent features | BupaR pathway analysis |
| **Granularity** | Population-level patterns | Cohort-specific patterns |
| **Processing** | Single large job | Multiple parallel jobs |

## 🛠️ Usage Examples

### Example 1: Complete Pipeline Execution
```python
# Run both pipelines
from fpgrowth_analysis.run_fpgrowth import run_fpgrowth_global
from fpgrowth_analysis.fpgrowth_by_cohort import process_features

# Global analysis
logger = create_fpgrowth_logger()
global_results = run_fpgrowth_global(logger)

# By-cohort analysis
for cohort in ["ed_non_opioid", "opioid_ed"]:
    process_features(cohort_name=cohort, support_threshold=0.05)
```

### Example 2: Loading Results for Analysis
```python
# Load global encoding map
encoding_map = load_global_encoding_map(logger)

# Use in CatBoost feature engineering
def encode_patient_drugs(drug_list, encoding_map):
    return [encoding_map.get(drug, "X000000000000000") for drug in drug_list]
```

### Example 3: Analyzing Cohort-Specific Patterns
```python
# Load cohort results
cohort_paths = list_cohort_input(cohort_name="ed_non_opioid")
for path in cohort_paths:
    # Process cohort-specific itemsets and rules
    # Use for BupaR process mining
```

## 📋 Prerequisites

### Data Requirements:
- **Pharmacy Clean Data**: `s3://pgxdatalake/pharmacy_clean/**/*.parquet`
- **Cohort Data**: `s3://pgxdatalake/cohort_clean/**/*.parquet`
- **Required Columns**: `mi_person_key`, `drug_name`, `event_date`

### System Requirements:
- **Python**: 3.8+
- **Memory**: 8GB+ RAM for global analysis
- **Storage**: S3 write access to `pgxdatalake` bucket
- **Network**: AWS S3 connectivity

### Environment Setup:
```bash
# Ensure project root is in Python path
export PYTHONPATH=/path/to/pgx-analysis:$PYTHONPATH

# Install dependencies
pip install -r requirements.txt
```

## 🔧 Troubleshooting

### Common Issues:

#### 1. mlxtend Import Error
```bash
pip install mlxtend==0.21.0
```

#### 2. Memory Issues with Global Analysis
```python
# Reduce data scope or increase memory
# Consider sampling for development
```

#### 3. S3 Permission Errors
```bash
# Verify AWS credentials and S3 bucket access
aws s3 ls s3://pgxdatalake/
```

#### 4. DuckDB Connection Issues
```python
# Check DuckDB installation and S3 extensions
# Ensure proper authentication
```

### Debug Mode:
```python
# Enable verbose logging
logger = create_fpgrowth_logger()
logger.setLevel(logging.DEBUG)
```

## 🔗 Using FP-Growth Results in Downstream Analysis

This section provides comprehensive examples of how to integrate FP-Growth results into your CatBoost and BupaR pipelines.

### 🤖 CatBoost Feature Engineering Integration

#### Loading Global Encoding Map

```python
# Load the global drug encoding map for consistent features across all cohorts
from fpgrowth_analysis.run_fpgrowth import load_global_encoding_map, create_fpgrowth_logger

logger = create_fpgrowth_logger()
encoding_map = load_global_encoding_map(logger)
print(f"Loaded encoding map with {len(encoding_map)} drugs")
```

#### Transform Patient Drug Lists

```python
def encode_patient_drugs(drug_list, encoding_map):
    """Transform patient drug lists into numerical features for CatBoost"""
    return [encoding_map.get(drug, "X000000000000000") for drug in drug_list]

# Apply to patient data
df['drug_encodings'] = df['drug_list'].apply(
    lambda drugs: encode_patient_drugs(drugs, encoding_map)
)

# Create feature columns for CatBoost
df['encoded_drug_features'] = df['drug_encodings'].apply(
    lambda encodings: ','.join(encodings) if encodings else ''
)
```

#### CatBoost Integration Example

```python
from catboost import CatBoostClassifier, Pool

# Use encoded drug features as categorical features
categorical_features = ['encoded_drug_features', 'age_band', 'cohort_type']

# Create CatBoost training pool
train_pool = Pool(
    X_train, 
    y_train, 
    cat_features=categorical_features
)

# Train model with drug pattern features
model = CatBoostClassifier(
    iterations=1000,
    learning_rate=0.1,
    depth=6,
    cat_features=categorical_features,
    verbose=100
)

model.fit(train_pool)
```

### 🔄 BupaR Process Mining Integration

#### Loading Cohort-Specific Results

```python
import json
from helpers.s3_utils import get_output_paths

def load_cohort_fpgrowth_results(cohort_name, age_band, event_year):
    """Load cohort-specific FP-Growth results for process mining"""
    
    output_paths = get_output_paths(cohort_name, age_band, event_year, "pgxdatalake")
    
    # Load itemsets
    itemsets_response = s3_client.get_object(
        Bucket='pgxdatalake',
        Key=output_paths['itemsets_json'].replace('s3://pgxdatalake/', '')
    )
    itemsets = json.loads(itemsets_response['Body'].read())
    
    # Load association rules
    rules_response = s3_client.get_object(
        Bucket='pgxdatalake', 
        Key=output_paths['rules_json'].replace('s3://pgxdatalake/', '')
    )
    rules = json.loads(rules_response['Body'].read())
    
    return itemsets, rules

# Example usage
itemsets, rules = load_cohort_fpgrowth_results("ed_non_opioid", "45-54", "2020")
print(f"Loaded {len(itemsets)} itemsets and {len(rules)} rules")
```

#### Process Flow Analysis

```python
def analyze_drug_pathways(rules_data):
    """Analyze drug prescription pathways for process mining"""
    
    pathways = []
    for rule in rules_data:
        if rule.get('confidence', 0) > 0.7:  # High confidence rules
            antecedents = rule.get('antecedents', [])
            consequents = rule.get('consequents', [])
            
            pathway = {
                'from_drugs': antecedents,
                'to_drugs': consequents,
                'confidence': rule.get('confidence', 0),
                'support': rule.get('support', 0),
                'lift': rule.get('lift', 0)
            }
            pathways.append(pathway)
    
    return pathways

# Analyze pathways for process mining
pathways = analyze_drug_pathways(rules)
print(f"Found {len(pathways)} high-confidence drug pathways")
```

#### BupaR Event Log Creation

```python
def create_bupar_event_log(itemsets, rules, cohort_data):
    """Create event log for BupaR process mining from FP-Growth results"""
    
    event_log = []
    
    for patient_id, patient_data in cohort_data.groupby('mi_person_key'):
        drug_sequence = patient_data.sort_values('event_date')['drug_name'].tolist()
        
        # Use FP-Growth patterns to identify process stages
        for i, drug in enumerate(drug_sequence):
            # Find relevant patterns
            relevant_patterns = [
                rule for rule in rules 
                if drug in rule.get('antecedents', []) or drug in rule.get('consequents', [])
            ]
            
            event = {
                'case_id': patient_id,
                'activity': drug,
                'timestamp': patient_data.iloc[i]['event_date'],
                'pattern_support': max([p.get('support', 0) for p in relevant_patterns] or [0]),
                'sequence_position': i + 1
            }
            event_log.append(event)
    
    return pd.DataFrame(event_log)

# Create event log for BupaR
event_log = create_bupar_event_log(itemsets, rules, cohort_df)
```

### 📊 Cross-Pipeline Validation

#### Validate Results Availability

```python
def validate_fpgrowth_integration():
    """Validate that FP-Growth results are ready for downstream use"""
    
    validation_results = {
        'global_ready': False,
        'cohort_ready': False,
        'errors': []
    }
    
    try:
        # Check global encoding map
        logger = create_fpgrowth_logger()
        encoding_map = load_global_encoding_map(logger)
        validation_results['global_ready'] = len(encoding_map) > 0
        print(f"✓ Global encoding map: {len(encoding_map)} drugs")
        
    except Exception as e:
        validation_results['errors'].append(f"Global encoding error: {e}")
        print(f"✗ Global encoding map error: {e}")
    
    try:
        # Check cohort results
        cohort_paths = list_cohort_input(cohort_name="ed_non_opioid", bucket_name="pgxdatalake")
        validation_results['cohort_ready'] = len(cohort_paths) > 0
        print(f"✓ Cohort results: {len(cohort_paths)} paths available")
        
    except Exception as e:
        validation_results['errors'].append(f"Cohort results error: {e}")
        print(f"✗ Cohort results error: {e}")
    
    return validation_results

# Run validation
validation = validate_fpgrowth_integration()
print(f"Integration ready: {validation['global_ready'] and validation['cohort_ready']}")
```

### 🎯 Pipeline Integration Checklist

#### Before CatBoost Analysis

- [ ] Global FP-Growth completed successfully
- [ ] Global encoding map accessible via `load_global_encoding_map()`
- [ ] Drug features transformed using encoding map
- [ ] Categorical features properly configured in CatBoost

#### Before BupaR Analysis

- [ ] By-cohort FP-Growth completed for target cohorts
- [ ] Cohort-specific itemsets and rules accessible
- [ ] Event logs created with FP-Growth pattern enrichment
- [ ] Process flow analysis configured

#### Integration Verification

```python
# Quick integration test
def test_integration():
    """Test both integration paths"""
    
    print("Testing CatBoost integration...")
    try:
        encoding_map = load_global_encoding_map(create_fpgrowth_logger())
        test_drugs = ['ACETAMINOPHEN', 'IBUPROFEN']
        encoded = [encoding_map.get(drug, "X000000000000000") for drug in test_drugs]
        print(f"✓ CatBoost integration: {encoded}")
    except Exception as e:
        print(f"✗ CatBoost integration error: {e}")
    
    print("Testing BupaR integration...")
    try:
        itemsets, rules = load_cohort_fpgrowth_results("ed_non_opioid", "45-54", "2020")
        print(f"✓ BupaR integration: {len(itemsets)} itemsets, {len(rules)} rules")
    except Exception as e:
        print(f"✗ BupaR integration error: {e}")

# Run integration test
test_integration()
```

### 🚀 Next Steps After Integration

1. **CatBoost Pipeline**: Use encoded drug features in model training and prediction
2. **BupaR Pipeline**: Analyze process flows with FP-Growth pattern enrichment
3. **Cross-Validation**: Compare global patterns vs cohort-specific patterns
4. **Model Evaluation**: Assess impact of FP-Growth features on model performance

### Integration Points Summary

- **Input Dependencies**: Completed pharmacy and cohort data cleaning
- **Output Consumers**: CatBoost feature engineering and BupaR analysis
- **Checkpoint Integration**: Uses same checkpoint system as other pipeline modules
- **Error Handling**: Comprehensive validation and fallback mechanisms

## 📚 References

- **FP-Growth Algorithm**: [MLxtend Documentation](https://rasbt.github.io/mlxtend/user_guide/frequent_patterns/fpgrowth/)
- **Association Rules**: [MLxtend Association Rules](https://rasbt.github.io/mlxtend/user_guide/frequent_patterns/association_rules/)
- **Drug Encoding**: See `helpers/drug_name_utils.py`
- **Network Visualization**: See `helpers/visualization_utils.py`

## 📞 Support

For questions or issues:

1. Check the Jupyter notebook for interactive examples
2. Review logs in S3 for detailed error information
3. Verify data availability and S3 permissions
4. Ensure all dependencies are properly installed

---

*Last Updated: August 3, 2025*
*Module Version: 2.0*
*Compatible with: pgx_analysis pipeline v2+*
