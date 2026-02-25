# %%
# OS Detection and Initial Setup
import sys
import os
import platform
import glob
import shutil
from pathlib import Path

# Set UTF-8 encoding for Windows console to handle emojis
if sys.platform == 'win32':
    import io
    if hasattr(sys.stdout, 'buffer'):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    if hasattr(sys.stderr, 'buffer'):
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Detect operating system
IS_WINDOWS = platform.system() == 'Windows'
IS_LINUX = platform.system() == 'Linux'

print(f"🖥️  Detected OS: {platform.system()}")

# Set project root based on OS
if IS_WINDOWS:
    # Windows: Use current workspace directory
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    print(f"   Using Windows workspace path")
elif IS_LINUX:
    # Linux/EC2: Use EC2 path
    PROJECT_ROOT = Path('/home/pgx3874/pgx-analysis')
    print(f"   Using Linux/EC2 path")
else:
    # Fallback: Use current file's parent directory
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    print(f"   Using fallback path (OS: {platform.system()})")

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Use centralized workflow Python (EC2 jupyter-env or PGX_PYTHON or sys.executable)
from py_helpers.env_utils import get_workflow_python_bin
PYTHON_BIN = get_workflow_python_bin()
print(f"   Using workflow Python: {PYTHON_BIN}")

# Set Rscript path based on OS
if IS_WINDOWS:
    # Windows: Check R_HOME first, then PATH, then common locations
    RSCRIPT_BIN = None
    rscript_from_r_home = None
    
    # First, try R_HOME environment variable (most reliable on Windows)
    r_home = os.environ.get('R_HOME')
    if r_home:
        rscript_from_r_home = Path(r_home) / 'bin' / 'Rscript.exe'
        if rscript_from_r_home.exists():
            RSCRIPT_BIN = rscript_from_r_home
            print(f"   Found Rscript via R_HOME: {RSCRIPT_BIN}")
    
    # If not found via R_HOME, try PATH
    if not RSCRIPT_BIN:
        rscript_path = shutil.which("Rscript")
        if rscript_path:
            RSCRIPT_BIN = Path(rscript_path)
            print(f"   Found Rscript in PATH: {RSCRIPT_BIN}")
    
    # If still not found, try common Windows installation locations
    if not RSCRIPT_BIN:
        common_windows_patterns = [
            'C:/Program Files/R/R-*/bin/Rscript.exe',
            'C:/Program Files (x86)/R/R-*/bin/Rscript.exe',
        ]
        for pattern in common_windows_patterns:
            matches = glob.glob(pattern)
            if matches:
                RSCRIPT_BIN = Path(matches[0])
                print(f"   Found Rscript at: {RSCRIPT_BIN}")
                break
    
    if not RSCRIPT_BIN:
        print(f"⚠️  Rscript not found on Windows, will use auto-detection")
        if r_home and rscript_from_r_home:
            print(f"   Note: R_HOME is set to {r_home} but Rscript.exe not found at {rscript_from_r_home}")
elif IS_LINUX:
    # Linux/EC2: Use EC2 default location
    RSCRIPT_BIN = Path('/usr/local/bin/Rscript')
    if not RSCRIPT_BIN.exists():
        # Try to find in PATH
        rscript_path = shutil.which("Rscript")
        if rscript_path:
            RSCRIPT_BIN = Path(rscript_path)
            print(f"⚠️  EC2 Rscript not found, using PATH: {RSCRIPT_BIN}")
        else:
            RSCRIPT_BIN = None
            print(f"⚠️  Rscript not found, will use auto-detection")
    else:
        print(f"   Using Linux/EC2 Rscript: {RSCRIPT_BIN}")
else:
    # Fallback: Try to find in PATH
    rscript_path = shutil.which("Rscript")
    if rscript_path:
        RSCRIPT_BIN = Path(rscript_path)
        print(f"   Found Rscript: {RSCRIPT_BIN}")
    else:
        RSCRIPT_BIN = None
        print(f"⚠️  Rscript not found, will use auto-detection")

print(f"✅ OS detection and path setup complete\n")

# Ensure PROJECT_ROOT is set and in sys.path before importing
if 'PROJECT_ROOT' not in globals():
    # Fallback: try to set PROJECT_ROOT from current file location or working directory
    import platform
    IS_WINDOWS = platform.system() == 'Windows'
    try:
        # Try to get from __file__ if available (normal script execution)
        if '__file__' in globals():
            PROJECT_ROOT = Path(__file__).resolve().parent.parent
        else:
            # Fallback: use current working directory (Jupyter/IPython)
            PROJECT_ROOT = Path.cwd()
            # If we're in a subdirectory, go up to project root
            if PROJECT_ROOT.name in ['3b_feature_importance_eda', '3a_feature_importance']:
                PROJECT_ROOT = PROJECT_ROOT.parent
    except:
        # Final fallback based on OS
        if IS_WINDOWS:
            PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == '3b_feature_importance_eda' else Path.cwd()
        else:
            PROJECT_ROOT = Path('/home/pgx3874/pgx-analysis')
    
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    print(f"⚠️  PROJECT_ROOT was not set, using fallback: {PROJECT_ROOT}")

# Import project utilities for configuration (PROJECT_ROOT already added to sys.path above)
try:
    from py_helpers.constants import age_band_to_fname
except ModuleNotFoundError as e:
    print(f"❌ Error importing py_helpers.constants: {e}")
    print(f"   PROJECT_ROOT: {PROJECT_ROOT}")
    print(f"   sys.path contains PROJECT_ROOT: {str(PROJECT_ROOT) in sys.path}")
    print(f"   Please ensure you've run the OS detection section at the top of this file.")
    raise

# Configuration: Cohort and Age Band
# Can be set via:
# 1. Command-line arguments: --cohort and --age-band
# 2. Environment variables: FEATURE_IMPORTANCE_EDA_COHORT and FEATURE_IMPORTANCE_EDA_AGE_BAND
# 3. Manual override: Change values below

# Check for command-line arguments
# POLYPHARMACY COHORT: cohort_name in data partitions is "non_opioid_ed"
# but we refer to this as "polypharmacy cohort" throughout
COHORT = "non_opioid_ed"  # Data partition name (must match S3/parquet partitions)
AGE_BAND = "85-114"
# User-facing target name (no F1120 reference for polypharmacy)
TARGET_LABEL = "F1120" if COHORT == "opioid_ed" else "target"

# First, try command-line arguments
# Only parse if running as script (not in Jupyter/interactive mode)
try:
    # Check if we're in Jupyter/IPython
    from IPython import get_ipython  # type: ignore
    # We're in Jupyter - skip argparse
    IN_JUPYTER = get_ipython() is not None
except (NameError, ImportError):
    IN_JUPYTER = False
    
if not IN_JUPYTER and len(sys.argv) > 1:
    import argparse
    parser = argparse.ArgumentParser(description="Feature Importance EDA Workflow")
    parser.add_argument("--cohort", type=str, default=None,
                       help="Cohort name (e.g., 'opioid_ed' or 'non_opioid_ed')")
    parser.add_argument("--age-band", type=str, default=None,
                       help="Age band (e.g., '13-24')")
    # Use parse_known_args to avoid errors
    args, unknown = parser.parse_known_args()
    if args.cohort:
        COHORT = args.cohort
    if args.age_band:
        AGE_BAND = args.age_band

# Second, try environment variables (with backward compatibility)
if not COHORT:
    COHORT = os.environ.get("FEATURE_IMPORTANCE_EDA_COHORT") or os.environ.get("STEP3B_COHORT")
if not AGE_BAND:
    AGE_BAND = os.environ.get("FEATURE_IMPORTANCE_EDA_AGE_BAND") or os.environ.get("STEP3B_AGE_BAND")

# Third, use defaults or manual override
if not COHORT:
    COHORT = "opioid_ed"  # Change as needed: "opioid_ed" or "non_opioid_ed" (polypharmacy cohort)
if not AGE_BAND:
    AGE_BAND = "13-24"    # Change as needed

AGE_BAND_FNAME = age_band_to_fname(AGE_BAND)

print(f"📋 Configuration:")
print(f"   Cohort: {COHORT}")
print(f"   Age Band: {AGE_BAND} ({AGE_BAND_FNAME})")
print(f"   Output Directory: {PROJECT_ROOT / '3b_feature_importance_eda' / 'outputs' / COHORT / AGE_BAND_FNAME}")
print(f"\n💡 Tip: Set cohort/age_band via:")
print(f"   - Command-line: python feature_importance_eda_workflow.py --cohort opioid_ed --age-band 13-24")
print(f"   - Environment: export FEATURE_IMPORTANCE_EDA_COHORT=opioid_ed && export FEATURE_IMPORTANCE_EDA_AGE_BAND=13-24")
print(f"   - Manual: Edit COHORT and AGE_BAND variables above\n")

# %% [markdown]
# Feature Importance EDA: Interactive Feature Importance EDA and Refinement

# %% [markdown]
# ## Overview
# 
# This workflow reads aggregated feature importances for each cohort and runs additional analyses to identify features that should be filtered:
# 
# 1. **Load aggregated feature importances** from Step 3 for the specified cohort
# 2. **Administrative/Non-informative code filtering** → Remove non-informative ICD/CPT codes (from lookup table)
# 3. **BupaR post-target analysis** → Identifies pre/post target events (target leakage detection; F1120 for opioid_ed, ED visit for polypharmacy) with automated ratio-based detection
# 4. **Interactive review** → Validate and manually add/remove codes to filter
# 5. **Filter & refine** → Generate final `cohort_feature_importance.csv` with filtered features for Step 4a

# %% [markdown]
# ## Workflow
# 
# ```
# Step 3: Aggregated Feature Importances (by cohort)
#          ↓
#     [Load Aggregated FI] → Read cohort-specific feature importances
#          ↓
#     [Admin Code Filtering] → Remove non-informative ICD/CPT codes (from lookup table)
#          ↓
#     [BupaR Analysis] → Identify pre/post target events (target leakage) with automated detection
#          ↓
#     [Interactive Review] → Manually validate and update filtering codes ← YOU ARE HERE
#          ↓
#     [Filter & Refine] → Generate cohort_feature_importance.csv
#          ↓
#     Step 4a: Model Data Creation
# ```

# %% [markdown]
# ## Navigation
# 
# - **Section A**: Configuration and Setup
# - **Section B**: Administrative/Non-Informative Code Filtering
# - **Section C**: BupaR Post-Target Analysis (Pre/Post Target Events with Automated Leakage Detection)
# - **Section D**: Interactive Code Review and Filtering
# - **Section E**: Generate Final Refined Feature Importances

# %% [markdown]
# ## A. Configuration and Setup

# %%
# Import additional libraries for analysis
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display, Image, HTML
import warnings
import subprocess
import shutil
from datetime import datetime
warnings.filterwarnings('ignore')

# Note: COHORT, AGE_BAND, and AGE_BAND_FNAME are defined in the OS detection section at the top

# Output directories
OUTPUT_DIR = PROJECT_ROOT / "3b_feature_importance_eda" / "outputs" / COHORT / AGE_BAND_FNAME
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PLOTS_DIR = OUTPUT_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

print(f"✅ Configuration loaded")
print(f"   Project Root: {PROJECT_ROOT}")
print(f"   Cohort: {COHORT}")
print(f"   Age Band: {AGE_BAND} ({AGE_BAND_FNAME})")
print(f"   Python Binary: {PYTHON_BIN}")
if RSCRIPT_BIN:
    print(f"   Rscript Binary: {RSCRIPT_BIN}")
else:
    print(f"   Rscript Binary: Auto-detect (will be found by script)")
print(f"   Output Directory: {OUTPUT_DIR}")

# %% [markdown]
# ### 1. Load Aggregated Feature Importances from Step 3

# %%
# Load aggregated feature importance from Step 3a
# Step 3a writes to outputs/{cohort}/{filename} (no age_band subdir); also check from_s3 and S3.
possible_paths = [
    PROJECT_ROOT / "3a_feature_importance" / "outputs" / COHORT / f"{COHORT}_{AGE_BAND_FNAME}_aggregated_feature_importance.csv",
    PROJECT_ROOT / "3a_feature_importance" / "outputs" / COHORT / AGE_BAND / f"{COHORT}_{AGE_BAND_FNAME}_aggregated_feature_importance.csv",
    PROJECT_ROOT / "3a_feature_importance" / "from_s3" / "by_cohort" / COHORT / AGE_BAND / f"{COHORT}_{AGE_BAND_FNAME}_aggregated_feature_importance.csv",
]
# NVMe on EC2 (PGX_FEATURE_IMPORTANCE_OUTPUTS)
_env_3a = os.environ.get("PGX_FEATURE_IMPORTANCE_OUTPUTS")
if _env_3a:
    possible_paths.insert(0, Path(_env_3a) / COHORT / f"{COHORT}_{AGE_BAND_FNAME}_aggregated_feature_importance.csv")

aggregated_fi = None
for path in possible_paths:
    if path.exists():
        aggregated_fi = pd.read_csv(path)
        print(f"✅ Loaded aggregated feature importance from: {path}")
        print(f"   Total features: {len(aggregated_fi):,}")
        break

if aggregated_fi is None:
    # Try S3 (pgxdatalake gold/feature_importance)
    try:
        import io
        try:
            from py_helpers.common_imports import s3_client, S3_BUCKET
        except ImportError:
            import boto3
            s3_client = boto3.client("s3")
            S3_BUCKET = "pgxdatalake"
        _key = f"gold/feature_importance/{COHORT}/{AGE_BAND}/{COHORT}_{AGE_BAND_FNAME}_aggregated_feature_importance.csv"
        _obj = s3_client.get_object(Bucket=S3_BUCKET, Key=_key)
        aggregated_fi = pd.read_csv(io.BytesIO(_obj["Body"].read()))
        print(f"✅ Loaded aggregated feature importance from S3: s3://{S3_BUCKET}/{_key}")
        print(f"   Total features: {len(aggregated_fi):,}")
        _local = possible_paths[0]
        _local.parent.mkdir(parents=True, exist_ok=True)
        aggregated_fi.to_csv(_local, index=False)
        print(f"   Saved locally: {_local}")
    except Exception:
        pass

if aggregated_fi is None:
    print(f"❌ Could not find aggregated feature importance file")
    print(f"   Checked paths:")
    for path in possible_paths:
        print(f"     - {path}")
    print(f"   Also tried S3: gold/feature_importance/{{cohort}}/{{age_band}}/...")
    raise SystemExit(
        "Aggregated feature importance is required for Step 3b. "
        "Run Step 3a (2_feature_importance.ipynb) for this cohort/age_band first. "
        "Do not continue—feature importances define the feature set and include potential target leakage."
    )
else:
    # Display summary
    print(f"\n📊 Feature Importance Summary:")
    print(f"   Columns: {list(aggregated_fi.columns)}")
    print(f"\n   Top 10 features:")
    display(aggregated_fi.head(10))

# %% [markdown]
# ## B. Administrative/Non-Informative Code Filtering
# 
# Load and review administrative/non-informative ICD/CPT codes from the lookup table. These codes (administrative, scheduling, protocol codes) don't add predictive value and should be filtered before other analyses.

# %% [markdown]
# ### 1. Load Administrative Codes Lookup Table

# %%
# Load administrative codes lookup table (try multiple locations)
_administrative_lookup_candidates = [
    PROJECT_ROOT / "4b_event_filter" / "administrative_codes_lookup.json",
    PROJECT_ROOT / "1b_apcd_event_filter" / "administrative_codes_lookup.json",
    PROJECT_ROOT / "3b_feature_importance_eda" / "0_icd_cpt_check" / "administrative_codes_lookup.json",
]
administrative_lookup_path = None
for _p in _administrative_lookup_candidates:
    if _p.exists():
        administrative_lookup_path = _p
        break
if administrative_lookup_path is None:
    administrative_lookup_path = _administrative_lookup_candidates[0]  # for error message

if administrative_lookup_path.exists():
    try:
        with open(administrative_lookup_path, 'r') as f:
            admin_lookup = json.load(f)
        
        # Load administrative ICD and CPT codes
        admin_codes = {}
        if 'administrative_codes' in admin_lookup:
            admin_codes = admin_lookup['administrative_codes']
            
            print(f"✅ Loaded administrative codes lookup table")
            print(f"   Administrative ICD codes: {len(admin_codes.get('icd', []))}")
            print(f"   Administrative CPT codes: {len(admin_codes.get('cpt', []))}")
            if 'hcpcs' in admin_codes:
                print(f"   Administrative HCPCS codes: {len(admin_codes.get('hcpcs', []))}")
            
            total_admin_codes = (
                len(admin_codes.get('icd', [])) + 
                len(admin_codes.get('cpt', [])) + 
                len(admin_codes.get('hcpcs', []))
            )
            print(f"   Total administrative codes: {total_admin_codes}")
            
            # Store for later use
            ADMINISTRATIVE_CODES = {
                'icd': set(admin_codes.get('icd', [])),
                'cpt': set(admin_codes.get('cpt', [])),
                'hcpcs': set(admin_codes.get('hcpcs', []))
            }
            
            # Display sample codes
            if len(admin_codes.get('icd', [])) > 0:
                print(f"\n   Sample administrative ICD codes (first 10):")
                for code in list(admin_codes.get('icd', []))[:10]:
                    print(f"     - {code}")
                if len(admin_codes.get('icd', [])) > 10:
                    print(f"     ... and {len(admin_codes.get('icd', [])) - 10} more")
            
            if len(admin_codes.get('cpt', [])) > 0:
                print(f"\n   Sample administrative CPT codes (first 10):")
                for code in list(admin_codes.get('cpt', []))[:10]:
                    print(f"     - {code}")
                if len(admin_codes.get('cpt', [])) > 10:
                    print(f"     ... and {len(admin_codes.get('cpt', [])) - 10} more")
        else:
            print(f"⚠️  No 'administrative_codes' key found in lookup table")
            ADMINISTRATIVE_CODES = {'icd': set(), 'cpt': set(), 'hcpcs': set()}
    except Exception as e:
        print(f"⚠️  Could not load administrative codes lookup table: {e}")
        print(f"   Path checked: {administrative_lookup_path}")
        ADMINISTRATIVE_CODES = {'icd': set(), 'cpt': set(), 'hcpcs': set()}
else:
    print(f"ℹ️  Administrative codes lookup table not found. Checked paths:")
    for _p in _administrative_lookup_candidates:
        print(f"     - {_p}")
    print(f"   Will proceed without pre-identified administrative codes")
    ADMINISTRATIVE_CODES = {'icd': set(), 'cpt': set(), 'hcpcs': set()}

print(f"\n✅ Administrative code filtering ready")
print(f"   These codes will be filtered in the 'Filter and Refine' step")

# %% [markdown]
# ## C. BupaR Post-Target Analysis (Pre/Post Target Events)
# 
# BupaR analysis identifies pre vs post-target events. Codes that appear primarily after the target event are post-target leakage and should be filtered.
# 
# **Polypharmacy (non_opioid_ed):** Model events are built with drug events only up to the first ED visit (HCG Setting) within 21 days of drug event, so post-target leakage is expected to be zero by construction.

# %% [markdown]
# ### 1. Verify Rscript is Available
# 
# **Note:** BupaR analysis uses R scripts, so Rscript must be installed and available in PATH. The Python script will automatically find Rscript, but you can verify it's available here.

# %%
# Verify Rscript is available
# Note: shutil already imported in Section A

# Check configured path first
if RSCRIPT_BIN and RSCRIPT_BIN.exists():
    print(f"✅ Rscript found at configured path: {RSCRIPT_BIN}")
    rscript_path = str(RSCRIPT_BIN)
else:
    # Try to find in PATH
    rscript_path = shutil.which("Rscript")
    if rscript_path:
        print(f"✅ Rscript found in PATH: {rscript_path}")
    else:
        # Try common EC2 locations
        common_paths = [
            Path('/usr/local/bin/Rscript'),  # EC2 default
            Path('/usr/bin/Rscript'),
        ]
        found = False
        for path in common_paths:
            if path.exists():
                print(f"✅ Rscript found at: {path}")
                rscript_path = str(path)
                found = True
                break
        if not found:
            print(f"⚠️  Rscript not found")
            print(f"   The Python script will try to find it automatically")
            rscript_path = None

# Check version if found
if rscript_path:
    try:
        result = subprocess.run([rscript_path, "--version"], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            version_line = result.stdout.strip().split('\n')[0] if result.stdout else "Unknown"
            print(f"   Version: {version_line}")
    except Exception as e:
        print(f"   Could not check version: {e}")

print("\n" + "="*80)

# %% [markdown]
# ### 2. Update cohort data, then build BupaR input (cohort + 3a FI + target), then run BupaR
#
# Sync gold/cohorts, gold/medical, gold/pharmacy from S3 so we use the latest data between 3a and 3b. Then build model data (gold cohort filtered by 3a FI + admin removed). R only consumes the parquet.

# %%
# Note: subprocess and datetime already imported in Section A

# Check if configuration variables are defined
if 'COHORT' not in globals():
    raise NameError("COHORT is not defined. Please run the 'Configuration and Setup' section first.")
if 'AGE_BAND' not in globals():
    raise NameError("AGE_BAND is not defined. Please run the 'Configuration and Setup' section first.")

# Update cohort data from S3 so 3b uses latest data (seamless between 3a and 3b)
try:
    from py_helpers.env_utils import get_data_root
    from py_helpers.workflow_sync_checkpoint import sync_s3_to_local
    _s3_bucket = os.environ.get("PGX_S3_BUCKET", "pgxdatalake")
    _data_root = get_data_root()
    for _name, _prefix, _local in [
        ("cohorts", f"s3://{_s3_bucket}/gold/cohorts/", _data_root / "gold" / "cohorts"),
        ("medical", f"s3://{_s3_bucket}/gold/medical/", _data_root / "gold" / "medical"),
        ("pharmacy", f"s3://{_s3_bucket}/gold/pharmacy/", _data_root / "gold" / "pharmacy"),
    ]:
        sync_s3_to_local(_prefix, _local)
    print("Cohort data synced from S3.")
except Exception as _e:
    print(f"Note: Could not sync cohort data from S3: {_e}")

# Build model data (gold cohort filtered by 3a aggregated FI with admin codes removed) before R runs.
print("Building BupaR input from cohort data + 3a aggregated feature importance + target (admin codes removed)...")
build_result = subprocess.run(
    [str(PYTHON_BIN), str(PROJECT_ROOT / "3b_feature_importance_eda" / "create_bupar_input_from_cohort.py"), "--cohort", COHORT, "--age-band", AGE_BAND],
    cwd=str(PROJECT_ROOT),
    capture_output=True,
    text=True,
)
if build_result.stdout:
    print(build_result.stdout)
if build_result.stderr:
    print("STDERR:", build_result.stderr)
if build_result.returncode != 0:
    print(f"BupaR input build failed (exit {build_result.returncode}). Check gold cohort and 3a aggregated FI paths.")
else:
    print("BupaR input built successfully.")

# Run BupaR Post-Target Analysis (uses 3b-built parquet or Step 4 if present)
print("\nRunning BupaR Post-Target Analysis...")
print(f"Started at: {datetime.now()}")
print("Note: This will call R scripts using Rscript")

cmd = [
    str(PYTHON_BIN),
    str(PROJECT_ROOT / "3b_feature_importance_eda" / "1_bupaR" / "run_bupar_post_target_analysis.py"),
    "--cohort", COHORT,
    "--age-band", AGE_BAND
]

result = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True)

if result.stdout:
    print(result.stdout)
if result.stderr:
    print("STDERR:", result.stderr)

if result.returncode == 0:
    print(f"\nBupaR analysis completed successfully")
else:
    print(f"\nBupaR analysis failed with return code {result.returncode}")
stderr_text = result.stderr or ""
stdout_text = result.stdout or ""
if "Rscript not found" in stderr_text or "Rscript not found" in stdout_text:
    print("\nTip: Make sure R is installed and Rscript is in your PATH")
    print(f"   Current RSCRIPT_BIN: {RSCRIPT_BIN if RSCRIPT_BIN else 'Not found (will use auto-detection)'}")
    print("   Rscript detection is configured in the OS detection section at the top of this file")

# %% [markdown]
# ### 3. Load and Review BupaR Results

# %%
# Load BupaR results
bupar_results_path = OUTPUT_DIR / f"{COHORT}_{AGE_BAND_FNAME}_bupar_post_target_analysis.csv"

# Check if file exists in wrong location (features/ subdirectory) and move it
wrong_location = OUTPUT_DIR / "features" / f"{COHORT}_{AGE_BAND_FNAME}_bupar_post_target_analysis.csv"
if wrong_location.exists() and not bupar_results_path.exists():
    print(f"⚠️  Found BupaR results in wrong location: {wrong_location}")
    print(f"   Moving to correct location: {bupar_results_path}")
    wrong_location.rename(bupar_results_path)
elif wrong_location.exists() and bupar_results_path.exists():
    # Both exist - remove the one in wrong location
    print(f"⚠️  BupaR results exist in both locations. Removing wrong location: {wrong_location}")
    wrong_location.unlink()

if bupar_results_path.exists():
    bupar_results = pd.read_csv(bupar_results_path)
    print(f"✅ Loaded BupaR results: {len(bupar_results)} features analyzed")
    
    # Show post-target leakage features
    post_target_leakage = bupar_results[bupar_results.get('is_post_target_leakage', pd.Series([0]*len(bupar_results))) == 1]
    
    print(f"\n📊 BupaR Analysis Summary:")
    print(f"   Total features analyzed: {len(bupar_results)}")
    print(f"   Post-target leakage features: {len(post_target_leakage)}")
    
    # Check for critical finding: no pre-target events
    if 'pre_count' in bupar_results.columns and 'post_count' in bupar_results.columns:
        total_pre = bupar_results['pre_count'].sum()
        total_post = bupar_results['post_count'].sum()
        
        if total_pre == 0 and total_post > 0:
            print(f"\n   ⚠️  CRITICAL FINDING: No pre-{TARGET_LABEL} events found!")
            print(f"   All {total_post:,} events occur AFTER the target event.")
            print(f"   This means ALL features are post-target leakage and should be filtered.")
            print(f"   Consider checking:")
            print(f"     - Data filtering in Step 4a (model_events.parquet)")
            print(f"     - Whether events before the target were filtered out")
            print(f"     - Cohort definition and target event identification")
    
    if len(post_target_leakage) > 0:
        print(f"\n   ⚠️  Post-target leakage features identified:")
        display(post_target_leakage[['feature', 'is_post_target_leakage']].head(20))
    else:
        print(f"\n   ✅ No post-target leakage features identified")
        if COHORT == "non_opioid_ed":
            print(f"   ℹ️  (Expected for polypharmacy: model_events are built with drug events only up to the first ED visit [HCG] within 21d of drug, so no post-target leakage by construction.)")
    
    # Display full results
    print(f"\n   Full BupaR results:")
    display(bupar_results.head(20))
else:
    print(f"❌ BupaR results not found: {bupar_results_path}")
    bupar_results = pd.DataFrame()

# %% [markdown]
# ### 4. Visualize Post-Target Leakage Candidates
#
# Visualize the post-target leakage analysis results to identify features that occur primarily after the target event.

# %%
# Create visualizations for post-target leakage candidates (ratio column: post_target_ratio or post_f1120_ratio)
post_ratio_col = 'post_target_ratio' if 'post_target_ratio' in bupar_results.columns else ('post_f1120_ratio' if 'post_f1120_ratio' in bupar_results.columns else None)
pre_ratio_col = 'pre_target_ratio' if 'pre_target_ratio' in bupar_results.columns else ('pre_f1120_ratio' if 'pre_f1120_ratio' in bupar_results.columns else None)
if not bupar_results.empty and post_ratio_col:
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Filter out features with no ratio data (from BupaR fallback)
    bupar_results_viz = bupar_results[bupar_results[post_ratio_col] > 0].copy()
    
    if len(bupar_results_viz) > 0:
        # Figure 1: Distribution of Post-Target Ratios
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Post-{TARGET_LABEL.capitalize()} Leakage Analysis: {COHORT} / {AGE_BAND}', fontsize=16, fontweight='bold')
        
        # 1. Histogram of post-target ratios
        ax1 = axes[0, 0]
        ax1.hist(bupar_results_viz[post_ratio_col], bins=50, edgecolor='black', alpha=0.7)
        ax1.axvline(x=0.8, color='r', linestyle='--', linewidth=2, label='Threshold (80%)')
        ax1.set_xlabel(f'Post-{TARGET_LABEL.capitalize()} Ratio', fontsize=12)
        ax1.set_ylabel('Number of Features', fontsize=12)
        ax1.set_title(f'Distribution of Post-{TARGET_LABEL.capitalize()} Ratios', fontsize=13, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Pre vs Post ratio comparison (scatter plot)
        ax2 = axes[0, 1]
        if pre_ratio_col and pre_ratio_col in bupar_results_viz.columns:
            # Scatter plot: Pre-target ratio vs Post-target ratio
            ax2.scatter(
                bupar_results_viz[pre_ratio_col],
                bupar_results_viz[post_ratio_col],
                alpha=0.6,
                s=50,
                c=['red' if row.get('is_post_target_leakage', 0) == 1 else 
                   'green' if row.get('is_pre_target_predictive', 0) == 1 else 
                   'gray' 
                   for _, row in bupar_results_viz.iterrows()]
            )
            ax2.axhline(y=0.8, color='r', linestyle='--', linewidth=2, label='Post Leakage Threshold (80%)')
            ax2.axvline(x=0.8, color='g', linestyle='--', linewidth=2, label='Pre Predictive Threshold (80%)')
            ax2.set_xlabel(f'Pre-{TARGET_LABEL.capitalize()} Ratio (Predictive)', fontsize=12)
            ax2.set_ylabel(f'Post-{TARGET_LABEL.capitalize()} Ratio (Leakage)', fontsize=12)
            ax2.set_title(f'Pre vs Post-{TARGET_LABEL.capitalize()} Ratios', fontsize=13, fontweight='bold')
            ax2.legend(fontsize=9)
            ax2.grid(True, alpha=0.3)
            ax2.set_xlim(-0.05, 1.05)
            ax2.set_ylim(-0.05, 1.05)
        else:
            # Fallback: Top leakage candidates if pre ratio not available
            leakage_candidates = bupar_results_viz.nlargest(20, post_ratio_col)
            if len(leakage_candidates) > 0:
                y_pos = range(len(leakage_candidates))
                colors = ['red' if ratio >= 0.8 else 'orange' if ratio >= 0.5 else 'yellow' 
                         for ratio in leakage_candidates[post_ratio_col]]
                ax2.barh(y_pos, leakage_candidates[post_ratio_col], color=colors, alpha=0.7)
                ax2.set_yticks(y_pos)
                feature_labels = [f[:40] + '...' if len(f) > 40 else f for f in leakage_candidates['feature']]
                ax2.set_yticklabels(feature_labels, fontsize=9)
                ax2.set_xlabel(f'Post-{TARGET_LABEL.capitalize()} Ratio', fontsize=12)
                ax2.set_title(f'Top 20 Features by Post-{TARGET_LABEL.capitalize()} Ratio', fontsize=13, fontweight='bold')
                ax2.axvline(x=0.8, color='r', linestyle='--', linewidth=2, label='Threshold (80%)')
                ax2.legend()
                ax2.grid(True, alpha=0.3, axis='x')
        
        # 3. Pre vs Post event counts for leakage features
        ax3 = axes[1, 0]
        leakage_features = bupar_results_viz[bupar_results_viz['is_post_target_leakage'] == 1]
        if len(leakage_features) > 0 and 'pre_count' in leakage_features.columns and 'post_count' in leakage_features.columns:
            # Sample up to 20 features for clarity
            sample_size = min(20, len(leakage_features))
            leakage_sample = leakage_features.nlargest(sample_size, post_ratio_col)
            
            x_pos = range(len(leakage_sample))
            width = 0.35
            
            pre_counts = leakage_sample['pre_count'].values
            post_counts = leakage_sample['post_count'].values
            
            ax3.bar([x - width/2 for x in x_pos], pre_counts, width, label=f'Pre-{TARGET_LABEL.capitalize()}', alpha=0.7, color='blue')
            ax3.bar([x + width/2 for x in x_pos], post_counts, width, label=f'Post-{TARGET_LABEL.capitalize()}', alpha=0.7, color='red')
            
            ax3.set_xlabel('Feature Index', fontsize=12)
            ax3.set_ylabel('Event Count', fontsize=12)
            ax3.set_title(f'Pre vs Post-{TARGET_LABEL.capitalize()} Event Counts (Top {sample_size} Leakage Features)', 
                         fontsize=13, fontweight='bold')
            ax3.legend()
            ax3.grid(True, alpha=0.3, axis='y')
            ax3.set_xticks(x_pos)
            ax3.set_xticklabels([f"F{i+1}" for i in range(len(leakage_sample))], fontsize=8)
        else:
            ax3.text(0.5, 0.5, 'No leakage features with event count data', 
                    ha='center', va='center', transform=ax3.transAxes, fontsize=12)
            ax3.set_title(f'Pre vs Post-{TARGET_LABEL.capitalize()} Event Counts', fontsize=13, fontweight='bold')
        
        # 4. Summary statistics
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        pre_col = bupar_results_viz.get(pre_ratio_col, pd.Series([0]*len(bupar_results_viz)))
        summary_text = f"""
        Summary Statistics
        
        Total Features Analyzed: {len(bupar_results_viz):,}
        
        Post-Target Leakage (≥80%): {len(bupar_results_viz[bupar_results_viz[post_ratio_col] >= 0.8]):,}
        Pre-Target Predictive (≥80%): {len(bupar_results_viz[pre_col >= 0.8]):,}
        High Risk Post (50-80%): {len(bupar_results_viz[(bupar_results_viz[post_ratio_col] >= 0.5) & 
                                                      (bupar_results_viz[post_ratio_col] < 0.8)]):,}
        Low Risk (<50% post): {len(bupar_results_viz[bupar_results_viz[post_ratio_col] < 0.5]):,}
        
        Mean Pre-{TARGET_LABEL.capitalize()} Ratio: {pre_col.mean():.2%}
        Mean Post-{TARGET_LABEL.capitalize()} Ratio: {bupar_results_viz[post_ratio_col].mean():.2%}
        Median Post-{TARGET_LABEL.capitalize()} Ratio: {bupar_results_viz[post_ratio_col].median():.2%}
        
        Total Events (All Features):
        """
        
        if 'total_count' in bupar_results_viz.columns:
            summary_text += f"""
        Pre-{TARGET_LABEL.capitalize()}: {bupar_results_viz.get('pre_count', pd.Series([0]*len(bupar_results_viz))).sum():,}
        Post-{TARGET_LABEL.capitalize()}: {bupar_results_viz.get('post_count', pd.Series([0]*len(bupar_results_viz))).sum():,}
        Total: {bupar_results_viz['total_count'].sum():,}
        """
        
        ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, 
                fontsize=11, verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        # Save the plot
        plot_path = PLOTS_DIR / f"{COHORT}_{AGE_BAND_FNAME}_post_target_leakage_analysis.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"✅ Saved leakage analysis plot: {plot_path}")
        
        # Display in notebook
        display(fig)
        plt.close()
        
        # Create a detailed table of top leakage candidates
        if len(leakage_features) > 0:
            print(f"\n📋 Top 20 Post-{TARGET_LABEL.capitalize()} Leakage Candidates:")
            print(f"{'='*100}")
            top_leakage = leakage_features.nlargest(20, post_ratio_col)
            
            cols_to_show = ['feature', post_ratio_col]
            if pre_ratio_col and pre_ratio_col in top_leakage.columns:
                cols_to_show.insert(1, pre_ratio_col)
            cols_to_show.extend(['pre_count', 'post_count', 'total_count'])
            
            display_df = top_leakage[[c for c in cols_to_show if c in top_leakage.columns]].copy()
            display_df[post_ratio_col] = display_df[post_ratio_col].apply(lambda x: f"{x:.1%}")
            if pre_ratio_col and pre_ratio_col in display_df.columns:
                display_df[pre_ratio_col] = display_df[pre_ratio_col].apply(lambda x: f"{x:.1%}")
            
            col_names = ['Feature', f'Post-{TARGET_LABEL.capitalize()} Ratio']
            if pre_ratio_col and pre_ratio_col in display_df.columns:
                col_names.insert(1, f'Pre-{TARGET_LABEL.capitalize()} Ratio')
            col_names.extend(['Pre Count', 'Post Count', 'Total Count'])
            display_df.columns = col_names
            
            display(display_df)
            
            # Save detailed leakage candidates to CSV
            leakage_csv = OUTPUT_DIR / f"{COHORT}_{AGE_BAND_FNAME}_post_target_leakage_candidates.csv"
            leakage_features_sorted = leakage_features.sort_values(post_ratio_col, ascending=False)
            leakage_features_sorted.to_csv(leakage_csv, index=False)
            print(f"\n💾 Saved detailed leakage candidates to: {leakage_csv}")
        
        # Also show pre-target predictive features
        if 'is_pre_target_predictive' in bupar_results_viz.columns and pre_ratio_col and pre_ratio_col in bupar_results_viz.columns:
            predictive_features = bupar_results_viz[bupar_results_viz['is_pre_target_predictive'] == 1]
            if len(predictive_features) > 0:
                print(f"\n✅ Top 20 Pre-{TARGET_LABEL.capitalize()} Predictive Features (safe to use):")
                print(f"{'='*100}")
                top_predictive = predictive_features.nlargest(20, pre_ratio_col)
                
                cols_to_show = ['feature', pre_ratio_col, post_ratio_col, 'pre_count', 'post_count', 'total_count']
                display_df = top_predictive[[c for c in cols_to_show if c in top_predictive.columns]].copy()
                display_df[pre_ratio_col] = display_df[pre_ratio_col].apply(lambda x: f"{x:.1%}")
                display_df[post_ratio_col] = display_df[post_ratio_col].apply(lambda x: f"{x:.1%}")
                display_df.columns = ['Feature', f'Pre-{TARGET_LABEL.capitalize()} Ratio', f'Post-{TARGET_LABEL.capitalize()} Ratio', 'Pre Count', 'Post Count', 'Total Count']
                
                display(display_df)
    else:
        print(f"⚠️  No features with post-{TARGET_LABEL} ratio data available for visualization")
elif not bupar_results.empty:
    print("ℹ️  Post-target ratio data not available (using BupaR fallback mode)")
    print("   Run with event-level data to get detailed ratio visualizations")
else:
    print("⚠️  No BupaR results available for visualization")

# %% [markdown]
# ### 5. View BupaR Visualizations

# %%
# Display BupaR visualizations (F1120 pre/post for opioid_ed; polypharmacy uses first ED within 21d of drug)
bupar_plots = [
    f"{COHORT}_{AGE_BAND_FNAME}_overall_activity_frequency.png",
    f"{COHORT}_{AGE_BAND_FNAME}_activity_milestones_gantt.png",
    f"{COHORT}_{AGE_BAND_FNAME}_activity_sequence_top.png",
]
if COHORT == "opioid_ed":
    bupar_plots.extend([
        f"{COHORT}_{AGE_BAND_FNAME}_pre_f1120_activity_frequency.png",
        f"{COHORT}_{AGE_BAND_FNAME}_post_f1120_activity_frequency.png",
    ])

# Check if plots directory exists and list available plots
if PLOTS_DIR.exists():
    available_plots = list(PLOTS_DIR.glob("*.png"))
    print(f"📁 Plots directory: {PLOTS_DIR}")
    print(f"   Found {len(available_plots)} PNG files")
    if available_plots:
        print(f"   Available plots:")
        for p in sorted(available_plots):
            print(f"     - {p.name}")
else:
    print(f"⚠️  Plots directory does not exist: {PLOTS_DIR}")

print(f"\n🔍 Looking for BupaR plots...")
for plot_name in bupar_plots:
    plot_path = PLOTS_DIR / plot_name
    if plot_path.exists():
        print(f"✅ Displaying: {plot_name}")
        display(Image(str(plot_path)))
    else:
        print(f"⚠️  Plot not found: {plot_path}")
        # Try alternative path (in case plots are in features/ subdirectory)
        alt_path = OUTPUT_DIR / "features" / plot_name
        if alt_path.exists():
            print(f"   Found in alternative location: {alt_path}")
            display(Image(str(alt_path)))
        else:
            # Try without cohort prefix (in case R script saved without it)
            simple_name = plot_name.replace(f"{COHORT}_{AGE_BAND_FNAME}_", "")
            simple_path = PLOTS_DIR / simple_name
            if simple_path.exists():
                print(f"   Found with simple name: {simple_path}")
                display(Image(str(simple_path)))

# %% [markdown]
# ## D. Interactive Code Review and Filtering
# 
# Review the analysis results and manually add/remove codes that should be filtered before Step 4a.

# %% [markdown]
# ### 1. Review Codes to Filter
# 
# Based on the BupaR post-target analysis and administrative code filtering, review codes that should be filtered:

# %%
# Combine filtering recommendations
filtering_recommendations = {
    'administrative_codes': set(),  # Administrative/non-informative codes from lookup table
    'bupar_post_target': set(),     # Post-target leakage features
    'manual_additional': set()       # Add codes manually here
}

# Add administrative codes from Section B (already loaded)
# Use ADMINISTRATIVE_CODES variable that was loaded in Section B
if 'ADMINISTRATIVE_CODES' in globals():
    admin_codes_dict = ADMINISTRATIVE_CODES
    
    # Add ICD codes to administrative codes
    if 'icd' in admin_codes_dict and len(admin_codes_dict['icd']) > 0:
        filtering_recommendations['administrative_codes'].update(admin_codes_dict['icd'])
        print(f"✅ Added {len(admin_codes_dict['icd'])} administrative ICD codes (from Section B)")
    
    # Add CPT codes to administrative codes
    if 'cpt' in admin_codes_dict and len(admin_codes_dict['cpt']) > 0:
        filtering_recommendations['administrative_codes'].update(admin_codes_dict['cpt'])
        print(f"✅ Added {len(admin_codes_dict['cpt'])} administrative CPT codes (from Section B)")
    
    # Add HCPCS codes if present
    if 'hcpcs' in admin_codes_dict and len(admin_codes_dict['hcpcs']) > 0:
        filtering_recommendations['administrative_codes'].update(admin_codes_dict['hcpcs'])
        print(f"✅ Added {len(admin_codes_dict['hcpcs'])} administrative HCPCS codes (from Section B)")
    
    total_admin_codes = (
        len(admin_codes_dict.get('icd', set())) + 
        len(admin_codes_dict.get('cpt', set())) + 
        len(admin_codes_dict.get('hcpcs', set()))
    )
    if total_admin_codes > 0:
        print(f"   Total administrative codes added: {total_admin_codes}")
else:
    print(f"ℹ️  ADMINISTRATIVE_CODES not found (Section B may not have been run)")
    print(f"   Will proceed without pre-identified administrative codes")

# Check for pre-existing filtering config JSON file (from previous runs)
existing_filtering_config_path = OUTPUT_DIR / f"{COHORT}_{AGE_BAND_FNAME}_manual_filtering_config.json"
if existing_filtering_config_path.exists():
    try:
        with open(existing_filtering_config_path, 'r') as f:
            existing_config = json.load(f)
        
        # Load previously identified codes
        if 'codes_to_filter' in existing_config:
            existing_codes = set(existing_config['codes_to_filter'])
            
            # Load codes into appropriate categories based on stored counts
            codes_list = sorted(list(existing_codes))
            admin_count = existing_config.get('administrative_codes_count', 0)
            bupar_count = existing_config.get('bupar_post_target_count', 0)
            
            if admin_count > 0:
                filtering_recommendations['administrative_codes'] = set(codes_list[:admin_count])
            if bupar_count > 0:
                start_idx = admin_count
                end_idx = admin_count + bupar_count
                filtering_recommendations['bupar_post_target'] = set(codes_list[start_idx:end_idx])
            if len(codes_list) > (admin_count + bupar_count):
                start_idx = admin_count + bupar_count
                filtering_recommendations['manual_additional'] = set(codes_list[start_idx:])
            
            print(f"✅ Loaded pre-existing filtering config from: {existing_filtering_config_path}")
            print(f"   Pre-existing codes to filter: {len(existing_codes)}")
            print(f"     - Administrative codes: {admin_count}")
            print(f"     - BupaR post-target codes: {bupar_count}")
            print(f"     - Manual codes: {len(existing_codes) - admin_count - bupar_count}")
            if 'codes_to_keep' in existing_config and len(existing_config['codes_to_keep']) > 0:
                print(f"   Codes to keep: {len(existing_config['codes_to_keep'])}")
    except Exception as e:
        print(f"⚠️  Could not load pre-existing filtering config: {e}")
        print(f"   Will proceed with fresh analysis")

# Add BupaR recommendations
# Merge with new analysis results (don't replace existing codes)
if 'bupar_results' in locals() and not bupar_results.empty:
    bupar_filtered = bupar_results[bupar_results.get('is_post_target_leakage', pd.Series([0]*len(bupar_results))) == 1]
    filtering_recommendations['bupar_post_target'].update(set(bupar_filtered['feature'].tolist()))

# Display summary
print("📋 Filtering Recommendations Summary:")
print(f"   Administrative/non-informative codes: {len(filtering_recommendations['administrative_codes'])}")
print(f"   BupaR post-target leakage codes: {len(filtering_recommendations['bupar_post_target'])}")
print(f"   Manual additional codes: {len(filtering_recommendations['manual_additional'])}")

# Show administrative codes separately
if len(filtering_recommendations['administrative_codes']) > 0:
    admin_codes_list = sorted(list(filtering_recommendations['administrative_codes']))
    print(f"\n   📋 Administrative Codes to Remove ({len(admin_codes_list)} codes):")
    print(f"   {'='*80}")
    for i, code in enumerate(admin_codes_list, 1):
        print(f"   {i:4d}. {code}")
    print(f"   {'='*80}")

# Show codes to filter
all_codes_to_filter = (
    filtering_recommendations['administrative_codes'] |
    filtering_recommendations['bupar_post_target'] |
    filtering_recommendations['manual_additional']
)

print(f"\n   Total unique codes to filter: {len(all_codes_to_filter)}")

if len(all_codes_to_filter) > 0:
    print(f"\n   All codes recommended for filtering (Administrative + BupaR + Manual):")
    codes_list = sorted(list(all_codes_to_filter))
    for i, code in enumerate(codes_list[:50], 1):  # Show first 50
        print(f"     {i}. {code}")
    if len(codes_list) > 50:
        print(f"     ... and {len(codes_list) - 50} more")

# %% [markdown]
# ### 2. Manually Add/Remove Codes to Filter
# 
# **Instructions:**
# 1. Review the visualizations and analysis results above
# 2. Add codes to filter in the cell below (one per line)
# 3. Remove codes from the filtering list if they should be kept
# 4. Run the cell to update the filtering list

# %%
# ============================================
# MANUAL CODE FILTERING
# ============================================
# Add codes here that you want to filter based on your review
# Format: one code per line as a string

MANUAL_CODES_TO_FILTER = [
    # Example: "Z00.00",  # Administrative code
    # Example: "V70.0",   # Routine exam
    # Add your codes here:
]

# Remove codes from filtering if they should be kept
CODES_TO_KEEP = [
    # Example: "F11.20",  # Keep this code even if flagged
    # Add codes to keep here:
]

# Update filtering recommendations
filtering_recommendations['manual_additional'] = set(MANUAL_CODES_TO_FILTER)

# Remove codes that should be kept
for code in CODES_TO_KEEP:
    filtering_recommendations['administrative_codes'].discard(code)
    filtering_recommendations['bupar_post_target'].discard(code)
    filtering_recommendations['manual_additional'].discard(code)

# Final list of codes to filter
final_codes_to_filter = (
    filtering_recommendations['administrative_codes'] |
    filtering_recommendations['bupar_post_target'] |
    filtering_recommendations['manual_additional']
)

print(f"✅ Updated filtering list")
print(f"   Total codes to filter: {len(final_codes_to_filter)}")
print(f"\n   Codes to filter:")
for code in sorted(final_codes_to_filter):
    print(f"     - {code}")

# Save filtering list to JSON for use in next step
filtering_config = {
    'codes_to_filter': sorted(list(final_codes_to_filter)),
    'codes_to_keep': CODES_TO_KEEP,
    'administrative_codes_count': len(filtering_recommendations['administrative_codes']),
    'bupar_post_target_count': len(filtering_recommendations['bupar_post_target']),
    'manual_additional_count': len(filtering_recommendations['manual_additional'])
}

filtering_config_path = OUTPUT_DIR / f"{COHORT}_{AGE_BAND_FNAME}_manual_filtering_config.json"
with open(filtering_config_path, 'w') as f:
    json.dump(filtering_config, f, indent=2)

print(f"\n   💾 Saved filtering config to: {filtering_config_path}")

# %% [markdown]
# ## E. Generate Final Refined Feature Importances

# %% [markdown]
# ### 1. Update Filtering Scripts (if needed)
# 
# If you've added manual codes, you may need to update the filtering scripts to include them. Otherwise, proceed to run the filter and refine step.

# %% [markdown]
# ### 2. Run Filter and Refine

# %%
# Note: subprocess and datetime already imported in Section A

# Check if configuration variables are defined
if 'COHORT' not in globals():
    raise NameError("COHORT is not defined. Please run the 'Configuration and Setup' section first.")
if 'AGE_BAND' not in globals():
    raise NameError("AGE_BAND is not defined. Please run the 'Configuration and Setup' section first.")

print("🚀 Filtering and Refining Features...")
print(f"Started at: {datetime.now()}")

cmd = [
    str(PYTHON_BIN),
    str(PROJECT_ROOT / "3b_feature_importance_eda" / "2_filtering" / "filter_and_refine_features.py"),
    "--cohort", COHORT,
    "--age-band", AGE_BAND
]

result = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True)

if result.stdout:
    print(result.stdout)
if result.stderr:
    print("STDERR:", result.stderr)

if result.returncode == 0:
    print(f"\n✅ Filter and refine completed successfully")
else:
    print(f"\n❌ Filter and refine failed with return code {result.returncode}")

# %% [markdown]
# ### 3. Review Final Refined Feature Importances

# %%
# Load final refined feature importance
refined_fi_path = OUTPUT_DIR / f"{COHORT}_{AGE_BAND_FNAME}_cohort_feature_importance.csv"

if refined_fi_path.exists():
    refined_fi = pd.read_csv(refined_fi_path)
    print(f"✅ Loaded refined feature importance: {len(refined_fi)} features")
    
    # Load filtering summary
    summary_path = OUTPUT_DIR / f"{COHORT}_{AGE_BAND_FNAME}_feature_filtering_summary.json"
    if summary_path.exists():
        with open(summary_path, 'r') as f:
            filtering_summary = json.load(f)
        
        print(f"\n📊 Filtering Summary:")
        print(f"   Original features: {filtering_summary.get('original_count', 'N/A')}")
        print(f"   Filtered by post-target: {filtering_summary.get('filtered_by_post_target', 0)}")
        print(f"   Filtered by non-value-added: {filtering_summary.get('filtered_by_non_value_added', 0)}")
        print(f"   Filtered by threshold: {filtering_summary.get('filtered_by_threshold', 0)}")
        print(f"   Final features: {filtering_summary.get('final_count', 'N/A')}")
    
    print(f"\n   Top 20 refined features:")
    display(refined_fi.head(20))
    
    print(f"\n   ✅ File ready for Step 4a: {refined_fi_path}")
else:
    print(f"❌ Refined feature importance not found: {refined_fi_path}")

# %% [markdown]
# ### 4. Verify S3 Upload
# 
# Check that the refined feature importance file was uploaded to S3 for Step 4a consumption.

# %%
import boto3

s3_client = boto3.client('s3')
s3_bucket = 'pgxdatalake'
s3_key = f"gold/feature_importance/{COHORT}/{AGE_BAND}/{COHORT}_{AGE_BAND_FNAME}_cohort_feature_importance.csv"

try:
    s3_client.head_object(Bucket=s3_bucket, Key=s3_key)
    print(f"✅ File exists in S3: s3://{s3_bucket}/{s3_key}")
    
    # Get file size
    response = s3_client.head_object(Bucket=s3_bucket, Key=s3_key)
    size_mb = response['ContentLength'] / (1024 * 1024)
    print(f"   File size: {size_mb:.2f} MB")
    print(f"   Last modified: {response['LastModified']}")
except s3_client.exceptions.ClientError as e:
    if e.response['Error']['Code'] == '404':
        print(f"❌ File not found in S3: s3://{s3_bucket}/{s3_key}")
    else:
        print(f"❌ Error checking S3: {e}")

# %% [markdown]
# ## Summary
# 
# ✅ **Feature Importance EDA Interactive Analysis Complete**
# 
# **Outputs Generated:**
# - ✅ Administrative code filtering (from lookup table)
# - ✅ BupaR post-target analysis results with automated leakage detection
# - ✅ Post-target leakage visualizations
# - ✅ Refined `cohort_feature_importance.csv` for Step 4a
# - ✅ Filtering summary JSON
# 
# **Next Steps:**
# - Proceed to **Step 4a: Model Data Creation** using the refined feature importances
# - The `cohort_feature_importance.csv` file is available locally and in S3
