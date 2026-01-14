# Feature Importance Visualization Guide

**Cross-Platform Visualization for Feature Importance Analysis**

This guide covers visualization generation for feature importance results, including cross-platform compatibility (Linux EC2 and Windows) and multiple usage patterns.

---

## Table of Contents

1. [Overview](#overview)
2. [Cross-Platform Compatibility](#cross-platform-compatibility)
3. [Usage Methods](#usage-methods)
4. [Path Resolution](#path-resolution)
5. [Troubleshooting](#troubleshooting)
6. [Integration Examples](#integration-examples)

---

## Overview

The visualization system uses Python (matplotlib/seaborn) for consistency with the rest of the analysis workflow. The main script is:

- **`py_helpers/create_feature_importance_visualizations.py`** - Primary visualization tool
- **`3_feature_importance/create_plots.py`** - Convenience wrapper script

### Generated Plots

For each cohort/age-band combination, four plots are generated:

1. **Top 50 Features Bar Chart** - `{cohort}_{age_band}_{year}_top50_features.png`
2. **Top 50 with Recall Confidence** - `{cohort}_{age_band}_{year}_top50_with_recall.png`
3. **Normalized vs Scaled Comparison** - `{cohort}_{age_band}_{year}_normalized_vs_scaled.png`
4. **Category Distribution** - `{cohort}_{age_band}_{year}_category_distribution.png`

All plots are saved to `{output_dir}/plots/` and optionally uploaded to S3.

---

## Cross-Platform Compatibility

### Linux EC2 (Headless)

**Automatic Configuration:**
- **Matplotlib Backend**: Automatically uses `Agg` backend when `DISPLAY` is not set
- **Python Path**: Uses `sys.executable` (kernel Python from Jupyter)
- **AWS CLI**: Detected via `shutil.which()` and common Linux paths (`/usr/local/bin/aws`, `/usr/bin/aws`, `/home/ec2-user/.local/bin/aws`)

**Example Usage:**
```bash
# From EC2 terminal (using kernel Python)
python py_helpers/create_feature_importance_visualizations.py \
    outputs/opioid_ed_0_12_aggregated_feature_importance.csv \
    --output-dir outputs \
    --cohort-name opioid_ed \
    --age-band 0-12 \
    --event-year 2019
```

### Windows

**Automatic Configuration:**
- **Matplotlib Backend**: Uses default backend (typically `TkAgg`)
- **Python Path**: Uses `sys.executable` (current Python interpreter)
- **AWS CLI**: Detected via `shutil.which()` (finds AWS CLI in PATH)

**Example Usage:**
```bash
# From Windows command prompt or PowerShell
python py_helpers\create_feature_importance_visualizations.py ^
    outputs\opioid_ed_0_12_aggregated_feature_importance.csv ^
    --output-dir outputs ^
    --cohort-name opioid_ed \
    --age-band 0-12 \
    --event-year 2019
```

### Platform Detection

The script automatically detects the platform and logs configuration:
```
INFO: Platform: Linux 5.4.0
INFO: Python executable: /usr/bin/python3
INFO: Matplotlib backend: Agg
```

---

## Usage Methods

### Method 1: Direct Script Execution

**Linux EC2:**
```bash
python py_helpers/create_feature_importance_visualizations.py \
    outputs/opioid_ed_0_12_aggregated_feature_importance.csv \
    --output-dir outputs \
    --cohort-name opioid_ed \
    --age-band 0-12 \
    --event-year 2019
```

**Windows:**
```cmd
python py_helpers\create_feature_importance_visualizations.py ^
    outputs\opioid_ed_0_12_aggregated_feature_importance.csv ^
    --output-dir outputs ^
    --cohort-name opioid_ed ^
    --age-band 0-12 ^
    --event-year 2019
```

**Using Wrapper Script:**
```bash
# Linux EC2
python 3_feature_importance/create_plots.py \
    outputs/opioid_ed_0_12_aggregated_feature_importance.csv

# Windows
python 3_feature_importance\create_plots.py ^
    outputs\opioid_ed_0_12_aggregated_feature_importance.csv
```

### Method 2: Notebook Import (Recommended)

**Jupyter Notebook Cell:**
```python
# Import the function
from py_helpers.create_feature_importance_visualizations import create_feature_importance_plots

# Call the function directly
plots = create_feature_importance_plots(
    aggregated_file='outputs/opioid_ed_0_12_aggregated_feature_importance.csv',
    output_dir='outputs',
    cohort_name='opioid_ed',
    age_band='0-12',
    event_year=2019,
    s3_upload=False  # Set to True to upload to S3
)

# Display results
print(f"Generated {len(plots)} plots:")
for name, path in plots.items():
    print(f"  - {name}: {path}")

# Optionally display plots inline
from IPython.display import Image, display
for name, path in plots.items():
    display(Image(path))
```

**Benefits:**
- ✅ Direct access to plot paths for inline display
- ✅ Can use plot data for further analysis
- ✅ Easy to iterate and experiment
- ✅ Works seamlessly with notebook workflow

### Method 3: Subprocess from Notebook

**Jupyter Notebook Bash Cell:**
```bash
# Linux EC2
!python py_helpers/create_feature_importance_visualizations.py \
    outputs/opioid_ed_0_12_aggregated_feature_importance.csv \
    --output-dir outputs \
    --cohort-name opioid_ed \
    --age-band 0-12 \
    --event-year 2019
```

**Python Subprocess Cell:**
```python
import subprocess
import sys

result = subprocess.run(
    [
        sys.executable,
        'py_helpers/create_feature_importance_visualizations.py',
        'outputs/opioid_ed_0_12_aggregated_feature_importance.csv',
        '--output-dir', 'outputs',
        '--cohort-name', 'opioid_ed',
        '--age-band', '0-12',
        '--event-year', '2019',
        '--no-s3-upload'
    ],
    cwd='.',
    capture_output=True,
    text=True
)

print(result.stdout)
if result.stderr:
    print("Errors:", result.stderr)
```

---

## Path Resolution

The script handles both **relative** and **absolute** paths:

```python
# Relative path (resolved from current working directory)
create_feature_importance_plots('outputs/file.csv')

# Absolute path
create_feature_importance_plots('/home/user/pgx-analysis/outputs/file.csv')

# Windows absolute path
create_feature_importance_plots(r'C:\Projects\pgx-analysis\outputs\file.csv')
```

### Automatic Parameter Extraction

If you don't specify `cohort_name`, `age_band`, or `event_year`, the script extracts them from the filename:

```python
# Filename: opioid_ed_0_12_aggregated_feature_importance.csv
# Automatically extracts:
#   cohort_name = "opioid_ed"
#   age_band = "0-12"
#   event_year = 2019 (if found in filename, else defaults to 2019)

create_feature_importance_plots('outputs/opioid_ed_0_12_aggregated_feature_importance.csv')
```

---

## Troubleshooting

### Linux EC2 Issues

**Problem**: "No display name and no $DISPLAY environment variable"
- **Solution**: Script automatically handles this by using `Agg` backend
- **Verify**: Check logs for "Matplotlib backend: Agg"

**Problem**: AWS CLI not found
- **Solution**: Install AWS CLI or ensure it's in PATH
- **Check**: `which aws` or `shutil.which('aws')` in Python

**Problem**: Python path issues
- **Solution**: Use `sys.executable` (already implemented)
- **Verify**: Script logs Python executable path at startup

### Windows Issues

**Problem**: Matplotlib window opens unexpectedly
- **Solution**: This is normal on Windows (uses TkAgg backend)
- **Note**: Script still saves files correctly

**Problem**: Path separator issues
- **Solution**: Script uses `pathlib.Path` which handles both `/` and `\`
- **Note**: You can use either forward or backslashes in arguments

**Problem**: AWS CLI not found
- **Solution**: Install AWS CLI for Windows or add to PATH
- **Check**: `where aws` in Command Prompt

### Import Errors

```python
# If import fails, ensure project root is in path
import sys
from pathlib import Path
project_root = Path.cwd()  # Or Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from py_helpers.create_feature_importance_visualizations import create_feature_importance_plots
```

### Path Issues

```python
# Use absolute paths if relative paths don't work
from pathlib import Path
csv_file = Path('outputs/file.csv').resolve()
create_feature_importance_plots(str(csv_file))
```

---

## Integration Examples

### Notebook Workflow

```python
# Cell 1: Run analysis
!python 3_feature_importance/run_cohort_1_0_12.py

# Cell 2: Generate visualizations
from py_helpers.create_feature_importance_visualizations import create_feature_importance_plots

plots = create_feature_importance_plots(
    '3_feature_importance/outputs/opioid_ed_0_12_aggregated_feature_importance.csv',
    output_dir='3_feature_importance/outputs',
    cohort_name='opioid_ed',
    age_band='0-12',
    event_year=2019,
    s3_upload=False
)

# Cell 3: Display plots inline
from IPython.display import Image, display
for name, path in plots.items():
    print(f"\n{name}:")
    display(Image(path))
```

### Automated Pipeline

```python
import subprocess
import sys
from pathlib import Path

def generate_visualizations(cohort_name, age_band, event_year=2019):
    """Generate visualizations for a cohort/age-band combination."""
    output_dir = Path(f"3_feature_importance/outputs")
    aggregated_file = output_dir / f"{cohort_name}_{age_band.replace('-', '_')}_aggregated_feature_importance.csv"
    
    if not aggregated_file.exists():
        print(f"Error: {aggregated_file} not found")
        return False
    
    result = subprocess.run(
        [
            sys.executable,
            'py_helpers/create_feature_importance_visualizations.py',
            str(aggregated_file),
            '--output-dir', str(output_dir),
            '--cohort-name', cohort_name,
            '--age-band', age_band,
            '--event-year', str(event_year),
            '--no-s3-upload'
        ],
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        print("Visualizations generated successfully")
        print(result.stdout)
        return True
    else:
        print("Error generating visualizations")
        print(result.stderr)
        return False

# Usage
generate_visualizations('opioid_ed', '0-12', 2019)
```

---

## Best Practices

1. **For Notebooks**: Use Method 2 (direct import) for better integration
2. **For Scripts**: Use Method 1 (command line) for automation
3. **For Debugging**: Use Method 3 (subprocess) to see full output
4. **Always specify parameters**: Don't rely on filename parsing if possible
5. **Use absolute paths**: More reliable across different execution contexts
6. **Test locally first**: Generate plots locally before uploading to S3

---

## Related Documentation

- [`docs/README_feature_importance.md`](README_feature_importance.md) - Main feature importance analysis guide
- [`3_feature_importance/README.md`](../3_feature_importance/README.md) - Step 3 analysis documentation
- [`docs/README_output_structure.md`](README_output_structure.md) - Standard output structure framework

