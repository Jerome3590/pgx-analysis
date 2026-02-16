# S3 Upload Tracking System

Local tracking system for monitoring S3 uploads from visualization notebooks.

## Overview

The S3 Upload Tracker provides a local JSON-based tracking system to monitor which visualization outputs have been successfully uploaded to S3. This helps:

- **Monitor upload progress** across cohorts, age bands, and visualization types
- **Identify missing uploads** for specific combinations
- **Track upload failures** with error messages
- **Generate summary reports** for upload status
- **Maintain audit trail** of all upload attempts

## Quick Start

### 1. Initialize Tracker

```python
from py_helpers.s3_upload_tracker import S3UploadTracker

tracker = S3UploadTracker("status/s3_upload_tracker.json")
```

### 2. Track Uploads

```python
from pathlib import Path
from py_helpers.checkpoint_utils import upload_file_to_s3

# Upload with automatic tracking
upload_file_to_s3(
    local_path=Path("outputs/opioid_ed_1_0_12_bupar.html"),
    s3_path="s3://pgx-dashboard/bupar/opioid_ed/1-0-12/plots/opioid_ed_1_0_12_bupar.html",
    logger=logger,
    tracker=tracker,
    viz_type="bupar",
    cohort="opioid_ed",
    age_band="1-0-12"
)
```

### 3. View Status

```python
# Print summary report
tracker.print_summary()

# Check for missing uploads
tracker.print_missing_uploads(
    expected_cohorts=["opioid_ed", "non_opioid_ed"],
    expected_age_bands=["1-0-12", "1-13-24", "1-25-44", "1-45-54", 
                       "1-55-64", "1-65-74", "1-75-84", "1-85-114"],
    expected_viz_types=["bupar", "dtw", "fpgrowth"],
    expected_item_types=["drug_name", "icd_code", "cpt_code", "medical_code"]
)
```

### 4. Command Line Status Check

```bash
python -m py_helpers.s3_upload_tracker
```

## API Reference

### S3UploadTracker Class

#### `__init__(tracker_file: str = "status/s3_upload_tracker.json")`

Initialize tracker with JSON file path.

#### `log_upload(...)`

Log an upload attempt with full metadata:

```python
tracker.log_upload(
    local_path="outputs/file.html",
    s3_path="s3://bucket/key/file.html",
    visualization_type="bupar",  # bupar, dtw, fpgrowth, feature_importance
    cohort="opioid_ed",          # opioid_ed, non_opioid_ed
    age_band="1-0-12",
    item_type="drug_name",       # Optional: for FP-Growth
    file_size_mb=2.5,
    success=True,
    error=None,                  # Error message if failed
    metadata={"year": 2016}      # Optional additional metadata
)
```

#### `get_uploads(...)`

Query uploads with filters:

```python
uploads = tracker.get_uploads(
    visualization_type="bupar",
    cohort="opioid_ed",
    age_band="1-0-12",
    item_type="drug_name",
    success_only=True
)
```

Returns list of upload records matching filters.

#### `get_summary()`

Get summary statistics:

```python
summary = tracker.get_summary()
# Returns:
# {
#     "total_uploads": 100,
#     "successful_uploads": 95,
#     "failed_uploads": 5,
#     "by_visualization_type": {...},
#     "by_cohort": {...},
#     "by_age_band": {...},
#     "total_size_mb": 1250.5
# }
```

#### `print_summary()`

Print formatted summary report to console.

#### `get_missing_uploads(...)`

Identify missing uploads based on expected combinations:

```python
missing = tracker.get_missing_uploads(
    expected_cohorts=["opioid_ed", "non_opioid_ed"],
    expected_age_bands=["1-0-12", "1-13-24", ...],
    expected_viz_types=["bupar", "dtw", "fpgrowth"],
    expected_item_types=["drug_name", "icd_code", ...]  # Optional
)
# Returns:
# {
#     "bupar": ["opioid_ed/1-0-12", "non_opioid_ed/1-13-24"],
#     "dtw": ["opioid_ed/1-25-44"],
#     "fpgrowth": ["opioid_ed/1-0-12/drug_name", ...]
# }
```

#### `print_missing_uploads(...)`

Print formatted missing uploads report.

#### `clear_tracker()`

Clear all tracking data (use with caution).

### Integration with checkpoint_utils

The `upload_file_to_s3()` function in `checkpoint_utils.py` now accepts optional tracking parameters:

```python
from pathlib import Path
from py_helpers.checkpoint_utils import upload_file_to_s3
from py_helpers.s3_upload_tracker import S3UploadTracker

tracker = S3UploadTracker()

success = upload_file_to_s3(
    local_path=Path("outputs/file.html"),
    s3_path="s3://bucket/key",
    logger=logger,
    check_exists=True,
    tracker=tracker,           # Pass tracker instance
    viz_type="bupar",          # Required with tracker
    cohort="opioid_ed",        # Required with tracker
    age_band="1-0-12",         # Required with tracker
    item_type="drug_name"      # Optional
)
```

## Tracking Data Structure

The tracker JSON file has the following structure:

```json
{
  "uploads": [
    {
      "timestamp": "2026-02-16T10:30:45.123456",
      "local_path": "outputs/opioid_ed_1_0_12_bupar.html",
      "s3_path": "s3://pgx-dashboard/bupar/...",
      "visualization_type": "bupar",
      "cohort": "opioid_ed",
      "age_band": "1-0-12",
      "item_type": null,
      "file_size_mb": 2.5,
      "success": true,
      "error": null,
      "metadata": {}
    }
  ],
  "last_updated": "2026-02-16T10:30:45.123456",
  "summary": {
    "total_uploads": 100,
    "successful_uploads": 95,
    "failed_uploads": 5,
    "by_visualization_type": {...},
    "by_cohort": {...},
    "by_age_band": {...},
    "total_size_mb": 1250.5
  }
}
```

## Example Workflows

### Track All BupaR Uploads

```python
from pathlib import Path
from py_helpers.s3_upload_tracker import S3UploadTracker
from py_helpers.checkpoint_utils import upload_file_to_s3

tracker = S3UploadTracker()

cohorts = ["opioid_ed", "non_opioid_ed"]
age_bands = ["1-0-12", "1-13-24", "1-25-44", "1-45-54", 
             "1-55-64", "1-65-74", "1-75-84", "1-85-114"]

for cohort in cohorts:
    for age_band in age_bands:
        age_band_fname = age_band.replace("-", "_")
        
        # Upload activity frequency
        local_file = Path(f"9_dashboard_visuals/bupar/outputs/{cohort}/{age_band}/plots/"
                         f"{cohort}_{age_band_fname}_activity_frequency_interactive.html")
        s3_path = f"s3://pgx-dashboard/bupar/{cohort}/{age_band}/plots/{local_file.name}"
        
        upload_file_to_s3(
            local_path=local_file,
            s3_path=s3_path,
            logger=logger,
            tracker=tracker,
            viz_type="bupar",
            cohort=cohort,
            age_band=age_band
        )

# Print status
tracker.print_summary()
```

### Track FP-Growth Uploads

```python
from pathlib import Path
from py_helpers.s3_upload_tracker import S3UploadTracker
from py_helpers.checkpoint_utils import upload_file_to_s3

tracker = S3UploadTracker()

item_types = ["drug_name", "icd_code", "cpt_code", "medical_code"]

for cohort in ["opioid_ed", "non_opioid_ed"]:
    for age_band in ["1-0-12", "1-13-24", ...]:
        for item_type in item_types:
            age_band_fname = age_band.replace("-", "_")
            
            # Upload itemsets
            local_file = Path(f"outputs/{cohort}_{age_band_fname}_{item_type}_itemsets_interactive.html")
            s3_path = f"s3://pgx-dashboard/fpgrowth/{cohort}/{age_band}/plots/{local_file.name}"
            
            upload_file_to_s3(
                local_path=local_file,
                s3_path=s3_path,
                logger=logger,
                tracker=tracker,
                viz_type="fpgrowth",
                cohort=cohort,
                age_band=age_band,
                item_type=item_type
            )

tracker.print_summary()
tracker.print_missing_uploads(
    expected_cohorts=["opioid_ed", "non_opioid_ed"],
    expected_age_bands=["1-0-12", ...],
    expected_viz_types=["fpgrowth"],
    expected_item_types=item_types
)
```

### Query Upload History

```python
from py_helpers.s3_upload_tracker import S3UploadTracker

tracker = S3UploadTracker()

# Get all successful DTW uploads
dtw_uploads = tracker.get_uploads(
    visualization_type="dtw",
    success_only=True
)

print(f"Found {len(dtw_uploads)} successful DTW uploads")
for upload in dtw_uploads:
    print(f"  {upload['cohort']}/{upload['age_band']} - {upload['timestamp']}")

# Get failed uploads
failed = tracker.get_uploads(success_only=False)
failed = [u for u in failed if not u["success"]]

print(f"\nFailed uploads: {len(failed)}")
for upload in failed:
    print(f"  {upload['visualization_type']} - {upload['cohort']}/{upload['age_band']}")
    print(f"    Error: {upload['error']}")
```

## Best Practices

1. **Initialize tracker once** at the start of your notebook/script
2. **Pass tracker to all upload functions** for comprehensive tracking
3. **Print summary regularly** to monitor progress
4. **Check for missing uploads** before finalizing pipeline
5. **Do not commit tracker JSON** to git (already in .gitignore)
6. **Backup tracker file** periodically if tracking important batch uploads

## Troubleshooting

### Tracker file not updating

- Check file permissions on `status/` directory
- Verify tracker is being passed to `upload_file_to_s3()`
- Check for exceptions during `_save_tracker()`

### Missing uploads not detected

- Verify expected combinations match actual pipeline structure
- Check that success_only=True filters are correct
- Review upload records with `get_uploads()` to confirm data

### Large tracker file

- Normal for complete pipeline runs (all cohorts × age bands × viz types)
- File is JSON and compresses well (add to .gitignore)
- Use `clear_tracker()` to reset if needed (backup first!)

## File Location

- Tracker JSON: `status/s3_upload_tracker.json` (gitignored)
- Module: `py_helpers/s3_upload_tracker.py`
- Integration: `py_helpers/checkpoint_utils.py`
- Examples: `examples/track_s3_uploads_example.py`

## Related Documentation

- [Checkpoint Utils](../py_helpers/checkpoint_utils.py) - S3 upload utilities
- [Dashboard Visuals](../4_dashboard_visuals.ipynb) - Visualization generation
- [Interactive Plotly Implementation](../docs/FPGROWTH_INTERACTIVE_IMPLEMENTATION.md)
