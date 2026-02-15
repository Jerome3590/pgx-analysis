# File Resolver - Universal File Resolution System

**Version:** 1.0  
**Last Updated:** February 15, 2026  
**Status:** ✅ Production Ready

## Overview

The File Resolver provides a centralized, consistent system for finding files across multiple locations in the pgx-analysis project. It eliminates manual path construction in notebooks and scripts, providing automatic fallback from local directories to data root (NVMe) to S3 with automatic download and caching.

### Key Benefits

- **Consistency**: Single source of truth for file resolution logic across all notebooks and scripts
- **Maintainability**: Centralized configuration - adding new file types or changing resolution logic affects all consumers automatically
- **Functionality**: Automatic S3 download with local caching, multi-location search, format detection
- **Developer Experience**: Simple API with pre-configured common file types and comprehensive error messages

## Quick Start

### Basic Usage

```python
from py_helpers.file_resolver import FileResolver

# Resolve a cohort feature importance file
resolver = FileResolver(
    file_type="cohort_feature_importance",
    project_root=Path.cwd(),
    cohort="opioid_ed",
    age_band="65-74"
)

# Get path (returns Path or None)
path = resolver.resolve()

# Or load directly (returns DataFrame)
df = resolver.load()
```

### Convenience Functions

For common file types, use the convenience functions:

```python
from py_helpers.file_resolver import (
    load_cohort_feature_importance,
    load_aggregated_feature_importance,
)

# Load cohort feature importance (Step 3b)
df = load_cohort_feature_importance("opioid_ed", "65-74", PROJECT_ROOT)

# Load aggregated feature importance (Step 3a)
df = load_aggregated_feature_importance("opioid_ed", "65-74", PROJECT_ROOT)
```

## Architecture

### Resolution Flow

The FileResolver checks locations in this priority order:

1. **Custom paths** (if provided)
2. **Environment variable overrides** (e.g., `PGX_FEATURE_IMPORTANCE_OUTPUTS`)
3. **Local project paths** (e.g., `3b_feature_importance_eda/outputs/`)
4. **Data root paths** (NVMe/local storage, e.g., `{DATA_ROOT}/gold/feature_importance/`)
5. **S3 bucket** (with automatic download and local caching)

### File Type Configurations

Each file type has a configuration defining:
- Filename pattern (with variable substitution)
- Local search paths
- Data root paths
- S3 key pattern
- Cache directory (for S3 downloads)
- Optional environment variable override

## Supported File Types

### 1. Feature Importance Files

#### `cohort_feature_importance`
Step 3b refined feature importance (leakage-filtered).

**Parameters:** `cohort`, `age_band`

**Locations checked:**
1. `3b_feature_importance_eda/outputs/{cohort}/{age_band_fname}/`
2. `3b_feature_importance_eda/outputs/{cohort}/{age_band}/`
3. `{DATA_ROOT}/gold/feature_importance/{cohort}/{age_band}/`
4. S3: `gold/feature_importance/{cohort}/{age_band}/`

**Example:**
```python
df = load_cohort_feature_importance("opioid_ed", "65-74", PROJECT_ROOT)
```

#### `aggregated_feature_importance`
Step 3a MC-CV aggregated feature importance.

**Parameters:** `cohort`, `age_band`

**Locations checked:**
1. `{PGX_FEATURE_IMPORTANCE_OUTPUTS}/{cohort}/`
2. `3a_feature_importance/outputs/{cohort}/`
3. `3a_feature_importance/outputs/{cohort}/{age_band}/`
4. `{DATA_ROOT}/gold/feature_importance/{cohort}/{age_band}/`
5. S3: `gold/feature_importance/{cohort}/{age_band}/`

**Example:**
```python
df = load_aggregated_feature_importance("opioid_ed", "65-74", PROJECT_ROOT)
```

#### `bupar_post_target_analysis`
BupaR post-target leakage analysis results.

**Parameters:** `cohort`, `age_band`

**Example:**
```python
resolver = FileResolver(
    file_type="bupar_post_target_analysis",
    project_root=PROJECT_ROOT,
    cohort="opioid_ed",
    age_band="65-74"
)
df = resolver.load()
```

### 2. Cohort and Model Data Files

#### `cohort_parquet`
Cohort data files (Step 2 output).

**Parameters:** `cohort`, `event_year`, `age_band`

**Locations checked:**
1. `data/gold/cohorts/cohort_name={cohort}/event_year={event_year}/age_band={age_band}/`
2. `{DATA_ROOT}/gold/cohorts/cohort_name={cohort}/event_year={event_year}/age_band={age_band}/`
3. S3: `gold/cohorts/cohort_name={cohort}/event_year={event_year}/age_band={age_band}/`

**Example:**
```python
resolver = FileResolver(
    file_type="cohort_parquet",
    project_root=PROJECT_ROOT,
    cohort="opioid_ed",
    age_band="65-74",
    event_year=2024
)
path = resolver.resolve()
```

#### `model_data`
Model events data (Step 4 output).

**Parameters:** `cohort`, `age_band`, `event_year`

**Example:**
```python
resolver = FileResolver(
    file_type="model_data",
    project_root=PROJECT_ROOT,
    cohort="opioid_ed",
    age_band="65-74",
    event_year=2024
)
df = resolver.load()
```

#### `final_model`
Trained model files (Step 6 output: XGBoost, CatBoost).

**Parameters:** `cohort`, `age_band`, `model_type` (e.g., "xgboost", "catboost"), `extension` (e.g., "json", "joblib")

**Example:**
```python
resolver = FileResolver(
    file_type="final_model",
    project_root=PROJECT_ROOT,
    cohort="opioid_ed",
    age_band="65-74",
    model_type="xgboost",
    extension="json"
)
model_path = resolver.resolve()
```

### 3. Configuration Files

#### `administrative_codes_lookup`
Administrative codes lookup (ICD, CPT, HCPCS).

**No parameters required.**

**Locations checked:**
1. `1b_apcd_event_filter/`
2. `4b_event_filter/`
3. `3b_feature_importance_eda/0_icd_cpt_check/`

**Example:**
```python
from py_helpers.file_resolver import load_administrative_codes

admin_codes = load_administrative_codes(PROJECT_ROOT)
# Returns: {'icd': [...], 'cpt': [...], 'hcpcs': [...]}
```

## Advanced Usage

### Custom Paths

Add custom search locations that are checked first:

```python
resolver = FileResolver(
    file_type="cohort_feature_importance",
    project_root=PROJECT_ROOT,
    cohort="opioid_ed",
    age_band="65-74",
    custom_paths=[
        "/custom/path/to/outputs",
        "/another/custom/path"
    ]
)
```

### Disable Auto-Download

Prevent automatic S3 downloads:

```python
# Disable for all resolve calls
resolver = FileResolver(
    file_type="cohort_feature_importance",
    project_root=PROJECT_ROOT,
    cohort="opioid_ed",
    age_band="65-74",
    auto_download=False
)

# Or disable for a single call
path = resolver.resolve(download_if_missing=False)
```

### Check Existence Without Loading

```python
resolver = FileResolver(...)
if resolver.exists():
    print("File found locally")
else:
    print("File not found")
```

### Multiple File Types with Fallback

Check multiple file types in order:

```python
def find_feature_importance(cohort, age_band, project_root):
    # Try cohort FI first (Step 3b refined)
    resolver = FileResolver(
        file_type="cohort_feature_importance",
        project_root=project_root,
        cohort=cohort,
        age_band=age_band
    )
    if resolver.exists():
        return resolver.load()
    
    # Fallback to aggregated FI (Step 3a)
    resolver = FileResolver(
        file_type="aggregated_feature_importance",
        project_root=project_root,
        cohort=cohort,
        age_band=age_band
    )
    return resolver.load()
```

## Notebook Integration

### Standard Notebook Setup

Add this cell to the top of notebooks after imports:

```python
import sys
from pathlib import Path

# Set project root
PROJECT_ROOT = Path(__file__).resolve().parent if "__file__" in dir() else Path.cwd()
if not (PROJECT_ROOT / "py_helpers").exists():
    PROJECT_ROOT = Path.cwd()
sys.path.insert(0, str(PROJECT_ROOT))

# Import file resolver functions
from py_helpers.file_resolver import (
    FileResolver,
    load_cohort_feature_importance,
    load_aggregated_feature_importance,
    resolve_cohort_fi_path,
    resolve_aggregated_fi_path,
)

print(f"Project root: {PROJECT_ROOT}")
```

### Replacing Manual Path Construction

**Before (manual path construction):**
```python
fi_path = Path(os.environ.get("PGX_FEATURE_IMPORTANCE_OUTPUTS", "3a_feature_importance/outputs"))
file = fi_path / cohort / f"{cohort}_{age_band.replace('-', '_')}_aggregated_feature_importance.csv"
if file.exists():
    df = pd.read_csv(file)
else:
    # Check other locations...
    # Try S3...
```

**After (file resolver):**
```python
df = load_aggregated_feature_importance(cohort, age_band, PROJECT_ROOT)
```

## Migration Guide

### Notebooks Status

| Notebook | Status | Action |
|----------|--------|--------|
| `2_feature_importance.ipynb` | ✅ Complete | Already uses `resolve_cohort_fi_path` |
| `3_model_train_shap_ffa.ipynb` | ✅ Complete | Updated to use FileResolver |
| `4_dashboard_visuals.ipynb` | 🔄 Ready | Should use resolver for FI files |
| `5_build_and_deploy.ipynb` | 🔄 Ready | Should use resolver instead of FI_ROOT paths |
| Interactive EDA notebooks | 🔄 Ready | Can migrate to resolver pattern |

### Scripts Ready to Migrate

- `3a_feature_importance/` scripts - Can use resolver
- `3b_feature_importance_eda/` scripts - Can use resolver
- `4_model_data/` scripts - Can use resolver for cohort files
- `6_final_model/` scripts - Can use resolver for model files

### Migration Steps

1. **Replace manual path construction** with `load_*()` convenience functions
2. **Replace file existence checks** with `resolver.exists()`
3. **Replace file loading** with `resolver.load()`
4. **Remove custom S3 download logic** - FileResolver handles it
5. **Test with local files first**, then test S3 fallback

## Error Handling

The resolver provides clear, actionable error messages:

```python
try:
    df = load_cohort_feature_importance("opioid_ed", "65-74", PROJECT_ROOT)
except FileNotFoundError as e:
    print(f"File not found: {e}")
    # Error message includes locations checked and next steps
    # Example: "Could not find cohort_feature_importance for opioid_ed/65-74.
    #           Run Step 3b for this cohort/age_band to produce the file."
except ValueError as e:
    print(f"File is empty or invalid: {e}")
    # Handle empty/invalid file
```

## Environment Variables

The resolver respects these environment variables:

- `PGX_FEATURE_IMPORTANCE_OUTPUTS`: Override for Step 3a outputs
- `DATA_ROOT`: Data root directory (NVMe/local storage)
- `LOCAL_DATA_PATH`: Alternative to DATA_ROOT
- `PGX_S3_BUCKET`: S3 bucket name (default: "pgxdatalake")
- `AWS_PROFILE`: AWS profile for S3 access

## Adding New File Types

To add a new file type, edit `py_helpers/file_resolver.py` and add to `FILE_TYPE_CONFIGS`:

```python
FILE_TYPE_CONFIGS = {
    # ... existing types ...
    
    "new_file_type": {
        "filename_pattern": "{cohort}_{age_band_fname}_newfile.csv",
        "local_paths": [
            "some_dir/outputs/{cohort}/{age_band}",
        ],
        "data_root_paths": [
            "gold/newfiles/{cohort}/{age_band}",
        ],
        "s3_key": "gold/newfiles/{cohort}/{age_band}/{filename}",
        "cache_dir": "some_dir/outputs/{cohort}",
    },
}
```

### Available Template Variables

- `{cohort}` - Cohort name (e.g., "opioid_ed")
- `{age_band}` - Age band with hyphens (e.g., "65-74")
- `{age_band_fname}` - Age band with underscores (e.g., "65_74")
- `{event_year}` - Event year (e.g., 2024)
- `{model_type}` - Model type (e.g., "xgboost")
- `{extension}` - File extension (e.g., "json")
- `{filename}` - Complete filename (generated from filename_pattern)

## Implementation Details

### File Format Detection

The resolver automatically detects and loads different file formats:

- **CSV**: `pd.read_csv()`
- **Parquet**: DuckDB (faster) with pandas fallback
- **JSON**: `json.load()`
- **Joblib**: `joblib.load()`
- **Other**: Returns path for manual handling

### S3 Caching Strategy

When a file is downloaded from S3:
1. File is retrieved using boto3
2. Saved to configured cache directory (usually local project outputs)
3. Subsequent calls use the cached version
4. Cache location is returned so it's used for all future calls

### Age Band Handling

The resolver handles both age band formats:
- **Hyphenated**: `65-74` (used in S3 keys and some local paths)
- **Underscored**: `65_74` (used in filenames and some local paths)

Function `age_band_to_fname()` converts hyphens to underscores for filename construction.

## Real-World Examples

### Example 1: Feature Importance in Step 3c

```python
# In 2_feature_importance.ipynb, Step 3c cell
from py_helpers.file_resolver import resolve_cohort_fi_path

for cohort, bands in REQUIRED_COHORTS.items():
    for age_band in bands:
        # Resolve cohort feature importance
        fi_path = resolve_cohort_fi_path(cohort, age_band, PROJECT_ROOT)
        
        if fi_path is None:
            print(f"⚠️  No FI found for {cohort}/{age_band}")
            continue
        
        # Load and process
        df = pd.read_csv(fi_path)
        # ... filter and process features ...
```

### Example 2: Model Data Creation

```python
# In 4_model_data/create_model_data.py
from py_helpers.file_resolver import FileResolver

# Load cohort data
cohort_resolver = FileResolver(
    file_type="cohort_parquet",
    project_root=PROJECT_ROOT,
    cohort=cohort,
    age_band=age_band,
    event_year=event_year
)
cohort_path = cohort_resolver.resolve()
if not cohort_path:
    raise FileNotFoundError(f"Cohort data not found for {cohort}/{age_band}/{event_year}")

# Load feature importance to determine which features to include
fi_df = load_cohort_feature_importance(cohort, age_band, PROJECT_ROOT)
feature_list = fi_df['feature'].tolist()

# Build model data using only these features
# ...
```

### Example 3: Model Training

```python
# In 6_final_model/train_model.py
from py_helpers.file_resolver import FileResolver

# Load model data
model_data_resolver = FileResolver(
    file_type="model_data",
    project_root=PROJECT_ROOT,
    cohort=cohort,
    age_band=age_band,
    event_year=event_year
)
model_data = model_data_resolver.load()

# Train model
# ...

# Save model using standard paths
output_dir = PROJECT_ROOT / "6_final_model" / "outputs" / cohort / age_band.replace("-", "_")
output_dir.mkdir(parents=True, exist_ok=True)
joblib.dump(model, output_dir / "xgboost.joblib")
```

## Troubleshooting

### File Not Found

If you get `FileNotFoundError`:
1. Check that the prerequisite step has completed (e.g., Step 3b for cohort FI)
2. Verify the file exists in at least one expected location
3. Check S3 credentials if auto_download is enabled
4. Use `resolver.resolve()` instead of `resolver.load()` to see if path can be resolved

### Empty or Invalid File

If you get `ValueError`:
1. Check that the source step completed successfully
2. Verify the file is not corrupted (check file size)
3. Re-run the source step to regenerate the file

### S3 Download Fails

If S3 download fails:
1. Check AWS credentials are configured (`AWS_PROFILE` or default credentials)
2. Verify you have read access to the S3 bucket
3. Check network connectivity
4. Use `auto_download=False` to skip S3 and use local files only

### Wrong File Loaded

If the wrong file is loaded:
1. Check that files in all locations are in sync
2. Use `resolve()` to see which path is being used
3. Consider clearing cached files in project outputs directories
4. Verify age_band format (hyphen vs underscore)

## Testing

### Validate File Resolution

```python
from py_helpers.file_resolver import FileResolver

cohorts = ["opioid_ed", "non_opioid_ed"]
age_bands = ["0-12", "13-24", "25-44", "45-54", "55-64", "65-74", "75-84", "85-114"]

print("Checking cohort feature importance files:")
for cohort in cohorts:
    for age_band in age_bands:
        resolver = FileResolver(
            file_type="cohort_feature_importance",
            project_root=PROJECT_ROOT,
            cohort=cohort,
            age_band=age_band,
            auto_download=False
        )
        path = resolver.resolve()
        status = "✓" if path else "✗"
        print(f"  {status} {cohort}/{age_band}: {path or 'NOT FOUND'}")
```

### Test S3 Fallback

```python
# Remove local file to force S3 download
import shutil
from py_helpers.file_resolver import FileResolver

# Remove cached file
cache_dir = PROJECT_ROOT / "3b_feature_importance_eda" / "outputs" / "opioid_ed"
test_file = cache_dir / "opioid_ed_65_74_cohort_feature_importance.csv"
if test_file.exists():
    shutil.move(test_file, test_file.with_suffix('.csv.backup'))

# Should download from S3
resolver = FileResolver(
    file_type="cohort_feature_importance",
    project_root=PROJECT_ROOT,
    cohort="opioid_ed",
    age_band="65-74"
)
path = resolver.resolve()
print(f"Downloaded from S3: {path}")

# Restore backup
if test_file.with_suffix('.csv.backup').exists():
    shutil.move(test_file.with_suffix('.csv.backup'), test_file)
```

## Performance Considerations

### Caching
- First call may be slow if downloading from S3
- Subsequent calls are fast (uses cached local file)
- Cache persists across Python sessions

### DuckDB for Parquet
- FileResolver uses DuckDB for faster parquet loading
- Fallback to pandas if DuckDB fails
- Significantly faster for large parquet files

### Local First
- Always checks local paths before S3
- S3 downloads only when necessary
- Set `auto_download=False` to skip S3 entirely

## Related Documentation

- [NOTEBOOK_FEATURE_IMPORTANCE_PATTERN.md](NOTEBOOK_FEATURE_IMPORTANCE_PATTERN.md) - Feature importance specific patterns
- [FEATURE_CREATION_FOR_MODEL.md](FEATURE_CREATION_FOR_MODEL.md) - Feature creation workflow
- `py_helpers/file_resolver.py` - Source code with inline documentation
- `py_helpers/feature_importance_eda_utils.py` - Feature importance utilities using FileResolver

## Version History

### Version 1.0 (February 15, 2026)
- Initial implementation with 7 pre-configured file types
- Multi-location search with S3 fallback
- Automatic format detection and loading
- Integration with existing utilities
- Comprehensive documentation

## Support

For issues or questions:
1. Check this documentation first
2. Review error messages - they contain actionable guidance
3. Verify prerequisite steps have completed
4. Check file exists in expected locations
5. Test with `auto_download=False` to isolate S3 issues
