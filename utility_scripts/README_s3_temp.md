# S3 Upload Monitoring Tool

Local command-line tool to monitor EC2 uploads to S3 for dashboard visualizations.

## Quick Start

```bash
# Check current S3 status for all visualizations
python utility_scripts/monitor_s3_uploads.py --check-all

# Watch for new uploads in real-time (checks every 30 seconds)
python utility_scripts/monitor_s3_uploads.py --watch --interval 30

# Find all missing uploads
python utility_scripts/monitor_s3_uploads.py --find-missing

# Check specific cohort/age_band
python utility_scripts/monitor_s3_uploads.py --cohort opioid_ed --age-band 1-0-12
```

## What It Does

- **Monitors S3 buckets** for dashboard visualization uploads (BupaR, DTW, FP-Growth)
- **Tracks completion status** for all cohort/age_band combinations
- **Detects missing files** by comparing expected vs actual uploads
- **Real-time watching** to see uploads as they happen from EC2
- **Generates reports** in JSON format for integration with other tools

## Expected Files Per Visualization Type

### BupaR (6 files per cohort/age_band)
- `{cohort}_{age_band}_activity_frequency_interactive.html` ⭐ Interactive with year dropdown
- `{cohort}_{age_band}_trace_explorer_interactive.html` ⭐ Interactive with year dropdown
- `{cohort}_{age_band}_process_matrix_interactive.html` ⭐ Interactive with year dropdown
- `{cohort}_{age_band}_activity_frequency.png` (static fallback)
- `{cohort}_{age_band}_trace_explorer.png` (static fallback)
- `{cohort}_{age_band}_process_matrix.png` (static fallback)

### DTW (4 files per cohort/age_band)
- `dtw_trajectory_cluster_interactive_{cohort}_{age_band}.html`
- `dtw_trajectory_analysis_{cohort}_{age_band}.png`
- `dtw_sample_trajectories_{cohort}_{age_band}.png`
- `chart_data.json`

### FP-Growth (8 files per cohort/age_band)
- `{cohort}_{age_band}_drug_name_itemsets_interactive.html`
- `{cohort}_{age_band}_drug_name_network_interactive.html`
- `{cohort}_{age_band}_icd_code_itemsets_interactive.html`
- `{cohort}_{age_band}_icd_code_network_interactive.html`
- `{cohort}_{age_band}_cpt_code_itemsets_interactive.html`
- `{cohort}_{age_band}_cpt_code_network_interactive.html`
- `{cohort}_{age_band}_medical_code_itemsets_interactive.html`
- `{cohort}_{age_band}_medical_code_network_interactive.html`

**Total**: 18 files × 2 cohorts × 8 age bands = **288 files expected**

## Usage Examples

### 1. Check All Uploads

```bash
python utility_scripts/monitor_s3_uploads.py --check-all
```

Output shows:
- Overall completion status
- Files found vs expected
- Breakdown by visualization type
- List of incomplete combinations

### 2. Watch Mode (Real-Time Monitoring)

```bash
python utility_scripts/monitor_s3_uploads.py --watch --interval 30
```

Continuously monitors S3 and shows:
- `[12:34:56] 🆕 5 new file(s) uploaded!`
- Which cohort/age_band/viz_type received uploads
- Runs until you press Ctrl+C

### 3. Find Missing Uploads

```bash
python utility_scripts/monitor_s3_uploads.py --find-missing --output status/missing.json
```

Shows all missing files and saves detailed report to JSON.

### 4. Check Specific Cohort

```bash
python utility_scripts/monitor_s3_uploads.py --cohort opioid_ed --age-band 1-0-12
```

Shows detailed status for that specific combination.

### 5. Check Only One Visualization Type

```bash
python utility_scripts/monitor_s3_uploads.py --check-all --viz-type fpgrowth
```

Checks only FP-Growth uploads across all cohorts.

## Command-Line Options

```
positional arguments:
  None

optional arguments:
  -h, --help            show help message and exit
  
Actions:
  --check-all           Check status for all cohorts/age_bands
  --find-missing        Find all missing uploads
  --watch               Continuously monitor for changes
  
Filters:
  --cohort {opioid_ed,non_opioid_ed}
                        Check specific cohort
  --age-band AGE_BAND   Check specific age band (e.g., 1-0-12)
  --viz-type {bupar,dtw,fpgrowth}
                        Check specific visualization type
  
Options:
  --interval SECONDS    Watch mode check interval (default: 30)
  --output PATH         Save report to JSON file
  --profile PROFILE     AWS CLI profile name
  --bucket BUCKET       S3 bucket (default: jerome-dixon.io)
  --prefix PREFIX       S3 prefix (default: vcu/pgx-risk-calculator)
```

## Output Examples

### Check All Output

```
================================================================================
S3 Upload Status - 2026-02-16T14:23:45.123456+00:00
================================================================================
Bucket: s3://jerome-dixon.io/vcu/pgx-risk-calculator

Overall Summary:
  Total cohort/age_band combinations: 16
  Complete: 12 (75.0%)
  Incomplete: 4 (25.0%)
  Files: 216/288 found
  Missing: 72 files

By Visualization Type:
  BUPAR:
    Complete: 14/16 (87.5%)
    Files: 84/96 found, 12 missing
  DTW:
    Complete: 16/16 (100.0%)
    Files: 64/64 found, 0 missing
  FPGROWTH:
    Complete: 10/16 (62.5%)
    Files: 68/128 found, 60 missing

Incomplete Combinations (4):
  non_opioid_ed / 1-0-12: bupar, fpgrowth
    Missing bupar: 2 files
    Missing fpgrowth: 8 files
  ...
```

### Watch Mode Output

```
Starting watch mode (checking every 30 seconds)...
Press Ctrl+C to stop

[14:23:45] Initial check: 216/288 files found
[14:24:15] No changes (216/288 files)
[14:24:45] 🆕 8 new file(s) uploaded!
  non_opioid_ed / 1-0-12 - fpgrowth: +8 files
[14:25:15] No changes (224/288 files)
^C
Watch mode stopped.
```

### Missing Uploads Report

```
================================================================================
Missing Uploads Report - 2026-02-16T14:23:45.123456+00:00
================================================================================
Total incomplete combinations: 4
Total missing files: 72

FPGROWTH - 4 incomplete combination(s):
  non_opioid_ed / 1-0-12: 8 missing
    - non_opioid_ed_1_0_12_drug_name_itemsets_interactive.html
    - non_opioid_ed_1_0_12_drug_name_network_interactive.html
    - non_opioid_ed_1_0_12_icd_code_itemsets_interactive.html
    ... and 5 more
  ...

Report saved to: status/missing_uploads.json
```

## Integration with EC2 Workflow

### During EC2 Visualization Generation

On your **local machine**, run in watch mode:
```bash
python utility_scripts/monitor_s3_uploads.py --watch --interval 30
```

As EC2 completes each visualization and uploads to S3, you'll see real-time updates locally.

### After EC2 Completes

Check final status:
```bash
python utility_scripts/monitor_s3_uploads.py --check-all --output status/final_status.json
```

Find any missing uploads:
```bash
python utility_scripts/monitor_s3_uploads.py --find-missing --output status/missing_uploads.json
```

### Re-run Only Missing Combinations on EC2

Use the missing uploads report to target specific cohort/age_band combinations:
```bash
# From missing_uploads.json, identify incomplete combinations
# Then re-run only those on EC2
```

## Requirements

- **Python 3.7+**
- **boto3**: `pip install boto3`
- **AWS credentials** configured (via `~/.aws/credentials` or environment variables)
- **Permissions**: `s3:ListBucket`, `s3:GetObject` on dashboard bucket

## Configuration

Update these variables in the script if needed:

```python
# S3 Configuration
S3_DASHBOARD_BUCKET = "jerome-dixon.io"  # Your dashboard bucket
S3_DASHBOARD_PREFIX = "vcu/pgx-risk-calculator"  # Your prefix

# Expected cohorts and age bands
REQUIRED_COHORTS = {
    "opioid_ed": ['0-12', '13-24', '25-44', '45-54', '55-64', '65-74', '75-84', '85-114'],
    "non_opioid_ed": ['0-12', '13-24', '25-44', '45-54', '55-64', '65-74', '75-84', '85-114']
}
```

## Troubleshooting

### "Error: boto3 required"
```bash
pip install boto3
```

### "NoCredentialsError"
Configure AWS credentials:
```bash
aws configure
# or
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
```

### "Access Denied"
Ensure your AWS credentials have S3 read permissions for the dashboard bucket.

### Watch mode shows no changes
- Verify EC2 is actually uploading to S3
- Check bucket/prefix configuration matches EC2 upload targets
- Confirm AWS credentials have access to the bucket

## Tips

1. **Run watch mode during EC2 processing** to see uploads in real-time
2. **Save reports to status/** directory (gitignored) for historical tracking
3. **Use --viz-type** to focus on specific visualization types during debugging
4. **Increase --interval** for less frequent checks (reduces API calls)
5. **Check specific combinations** when re-running failed uploads

## Related Files

- **Upload scripts**: `9_dashboard_visuals/{bupar,dtw,fpgrowth}/create_*_visuals.py`
- **Dashboard notebook**: `4_dashboard_visuals.ipynb`
- **Deploy script**: `5_build_and_deploy.ipynb`
- **S3 utilities**: `py_helpers/s3_utils.py`, `py_helpers/checkpoint_utils.py`

---

**Status tracking made easy!** Monitor your EC2 uploads locally in real-time. 🚀
