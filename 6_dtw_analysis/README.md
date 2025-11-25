# DTW Analysis for Drug Sequence Similarity

This module provides Dynamic Time Warping (DTW) analysis capabilities for comparing drug exposure sequences across patients in cohort datasets.

## Overview

DTW analysis helps identify patients with similar drug exposure patterns by measuring the similarity between temporal sequences that may vary in timing and length. This is particularly useful for:

- **Patient Clustering**: Group patients with similar drug histories
- **Outlier Detection**: Identify patients with unusual drug sequences
- **Pattern Discovery**: Find common drug exposure patterns
- **Risk Assessment**: Compare new patients to known high-risk patterns

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure you have access to the S3 bucket containing cohort data.

## Usage

### Basic Usage

```bash
python dtw_cohort_analysis.py \
    --cohort opioid_ed \
    --age-band "65-74" \
    --event-year 2020 \
    --n-clusters 5
```

### Command Line Arguments

- `--cohort`: Cohort to analyze (`opioid_ed` or `ed_non_opioid`)
- `--age-band`: Age band (e.g., "0-12", "65-74")
- `--event-year`: Event year (e.g., 2020)
- `--n-clusters`: Number of clusters to create (default: 5)
- `--no-plots`: Skip creating visualizations

### Examples

**Analyze opioid ED cohort:**
```bash
python dtw_cohort_analysis.py \
    --cohort opioid_ed \
    --age-band "65-74" \
    --event-year 2020 \
    --n-clusters 6
```

**Analyze ADE cohort without plots:**
```bash
python dtw_cohort_analysis.py \
    --cohort ed_non_opioid \
    --age-band "18-25" \
    --event-year 2019 \
    --n-clusters 4 \
    --no-plots
```

## Output

The analysis generates several outputs:

### 1. Visualizations
- **Similarity Matrix Heatmap**: Shows DTW distances between all patient pairs
- **Cluster Size Distribution**: Bar chart of patients per cluster
- **Average Sequence Length**: Drug sequence length by cluster
- **Most Common Drugs**: Top drugs across all clusters

### 2. S3 Storage
Results are saved to S3 with the following structure:
```
s3://{S3_BUCKET}/dtw_analysis/{cohort_name}/{age_band}/{event_year}/
├── dtw_results.json          # Complete analysis results
└── patient_sequences.parquet # Patient drug sequences
```

### 3. Analysis Results

The JSON results include:
- **Metadata**: Analysis parameters and summary statistics
- **Clustering Results**: Cluster assignments and quality metrics
- **Cluster Characteristics**: Detailed analysis of each cluster
- **Drug Encoding Map**: Mapping of drug names to numerical IDs

## Key Metrics

### Silhouette Score
Measures cluster quality (range: -1 to 1, higher is better)

### Cluster Characteristics
- Number of patients per cluster
- Average drug sequence length
- Most common drugs in each cluster
- Therapeutic class distribution
- Temporal characteristics (days to events)

## Integration with Pipeline

This DTW analysis complements the existing pipeline by:

1. **Working with Drug Event Explosion**: Uses the exploded drug-level data
2. **Supporting BupaR Analysis**: DTW clusters can be used for cluster-specific process mining
3. **Enhancing Feature Engineering**: DTW similarity scores can become model features
4. **Providing Clinical Insights**: Identifies similar patient patterns for clinical decision support

## Advanced Usage

### Custom Analysis

```python
from dtw_cohort_analysis import DTWCohortAnalyzer

# Create analyzer
analyzer = DTWCohortAnalyzer("65-74", 2020, "opioid_ed")

# Run custom analysis
results = analyzer.run_analysis(n_clusters=8, create_plots=True)

# Access results
cluster_results = results['cluster_results']
cluster_characteristics = results['cluster_characteristics']
```

### Batch Processing

```bash
# Process multiple cohorts
for cohort in opioid_ed ed_non_opioid; do
    for age_band in "0-12" "13-17" "18-25" "26-35" "36-45" "46-55" "56-65" "65-74"; do
        python dtw_cohort_analysis.py \
            --cohort $cohort \
            --age-band $age_band \
            --event-year 2020 \
            --n-clusters 5
    done
done
```

## Troubleshooting

### Common Issues

1. **DTW Package Not Available**
   ```
   Error: dtaidistance package not available.
   Install with: pip install dtaidistance
   ```

2. **Memory Issues with Large Datasets**
   - Reduce number of patients by filtering
   - Use smaller number of clusters
   - Consider sampling for initial analysis

3. **S3 Access Issues**
   - Ensure AWS credentials are configured
   - Check S3 bucket permissions
   - Verify cohort data exists

### Performance Tips

- **Small Datasets**: Use default settings
- **Large Datasets**: Consider sampling or filtering
- **Memory Optimization**: Process in batches for very large cohorts
- **Parallel Processing**: Run multiple age bands/cohorts in parallel

## Future Enhancements

- **Parallel DTW**: Implement parallel processing for large datasets
- **Advanced Clustering**: Add more clustering algorithms
- **Interactive Visualizations**: Create interactive dashboards
- **Real-time Analysis**: Support streaming analysis for new data
- **Integration with BupaR**: Direct integration with process mining workflows 