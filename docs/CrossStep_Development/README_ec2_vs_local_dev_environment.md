# Environment-Aware DuckDB Configuration

## Problem Solved

The pipeline now automatically adapts to different environments:
- **Production EC2**: 32 cores, 1TB RAM → Uses ~900GB memory, 30 threads
- **Testing Windows**: 4 cores, 16-32GB RAM → Uses ~14-28GB memory, 2 threads

Previously, scripts hardcoded `memory_limit='900GB'` which would fail on testing machines.

## How It Works

### Auto-Detection in `helpers/duckdb_utils.py`

```python
def get_duckdb_connection(logger, memory_limit=None, threads=None):
    """Create DuckDB connection with environment-aware optimization."""
    import psutil
    import os
    
    # Auto-detect memory limit if not provided
    if memory_limit is None:
        total_ram_gb = psutil.virtual_memory().total / (1024**3)
        # Use 90% of available RAM, capped at 900GB for safety
        memory_limit_gb = min(int(total_ram_gb * 0.9), 900)
        memory_limit = f'{memory_limit_gb}GB'
    
    # Auto-detect threads if not provided
    if threads is None:
        cpu_count = os.cpu_count() or 4
        # Leave 2 cores for OS, minimum 2 threads
        threads = max(cpu_count - 2, 2)
    
    # Apply settings...
```

### Usage in Scripts

**Before (Hardcoded):**
```python
conn = get_duckdb_connection(logger, memory_limit='900GB', threads=30)  # ❌ Fails on small machines
```

**After (Auto-Detect):**
```python
conn = get_duckdb_connection(logger, memory_limit=None, threads=None)  # ✅ Adapts to environment
```

**Override if Needed:**
```python
conn = get_duckdb_connection(logger, memory_limit='16GB', threads=4)  # ✅ Manual override
```

## Expected Behavior

### On EC2 (32 cores, 1TB RAM):
```
INFO - Creating DuckDB connection (memory: 900GB, threads: 30)
```

### On Windows Testing (4 cores, 32GB RAM):
```
INFO - Creating DuckDB connection (memory: 28GB, threads: 2)
```

## Configuration Layers

### 1. **Centralized Core Settings** (in `helpers/duckdb_utils.py`)
All core DuckDB settings managed centrally:
- ✅ `memory_limit` (auto-detected)
- ✅ `threads` (auto-detected)
- ✅ `checkpoint_threshold`
- ✅ `enable_object_cache`
- ✅ `disable_verification`
- ✅ `enable_optimizer`
- ✅ `disable_profiling`
- ✅ S3 credentials and extensions

### 2. **Job-Specific Settings** (in individual scripts)
Scripts ONLY add worker-specific settings:
- ✅ `temp_directory` (unique per worker to avoid conflicts)
- ✅ `s3_uploader_max_filesize` (for large file uploads)
- ✅ `s3_region` (if different from default)

## Files Updated

1. ✅ `helpers/duckdb_utils.py` - Added auto-detection logic
2. ✅ `1_apcd_input_data/clean_pharmacy.py` - Now uses auto-detection
3. ✅ `1_apcd_input_data/clean_medical.py` - Now uses centralized function with auto-detection
4. ✅ `requirements.txt` - Added `psutil>=5.9.0` for system resource detection

## Installation

```bash
pip install psutil>=5.9.0
# Or install all requirements:
pip install -r requirements.txt
```

## Testing

The configuration will automatically log what it detects:

```python
# Test on your local Windows machine
python 0_testing/test_pharmacy_duckdb_fixes.py
# Should see: "Creating DuckDB connection (memory: 28GB, threads: 2)" or similar

# Same code on EC2
python 1_apcd_input_data/clean_pharmacy.py ...
# Should see: "Creating DuckDB connection (memory: 900GB, threads: 30)"
```

## Benefits

1. ✅ **Single codebase** works on both testing and production
2. ✅ **No more crashes** from allocating too much RAM
3. ✅ **Optimal performance** on each environment
4. ✅ **Centralized configuration** - one place to maintain
5. ✅ **Override capability** for special cases
6. ✅ **Clear logging** shows what's being used

## Common Pitfall Fixed

**Before:** Scripts used invalid `PRAGMA` syntax that reset `memory_limit`:
```python
conn.sql("PRAGMA enable_object_cache=true")  # ❌ Invalid - resets memory_limit!
```

**After:** Correct syntax in centralized config:
```python
conn.sql("PRAGMA enable_object_cache")  # ✅ Valid - preserves all settings
```

## Architecture Summary

```
┌─────────────────────────────────────────┐
│   helpers/duckdb_utils.py               │
│   ─────────────────────────────         │
│   • Auto-detect system resources        │
│   • Apply all core DuckDB settings      │
│   • Use correct PRAGMA/SET syntax       │
│   • Install S3 extensions               │
└────────────────┬────────────────────────┘
                 │ Returns configured connection
                 ▼
┌─────────────────────────────────────────┐
│   Individual Scripts                    │
│   (clean_pharmacy.py, clean_medical.py) │
│   ─────────────────────────────────────│
│   • Get connection (auto-configured)    │
│   • Add ONLY worker-specific settings:  │
│     - temp_directory (unique per worker)│
│     - S3 uploader settings              │
└─────────────────────────────────────────┘
```

This ensures consistency, correctness, and environment adaptability! 🎯

