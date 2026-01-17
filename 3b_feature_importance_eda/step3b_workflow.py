# %%
# OS Detection and Initial Setup
import sys
import os
import platform
import glob
import shutil
from pathlib import Path

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

# Set Python binary path based on OS
if IS_WINDOWS:
    # Windows: Check PYTHON_HOME or PYTHON env vars first, then use sys.executable
    PYTHON_BIN = None
    
    # First, try PYTHON_HOME environment variable (if set)
    python_home = os.environ.get('PYTHON_HOME')
    if python_home:
        # PYTHON_HOME typically points to the Python installation directory
        python_from_home = Path(python_home) / 'python.exe'
        if python_from_home.exists():
            PYTHON_BIN = python_from_home
            print(f"   Found Python via PYTHON_HOME: {PYTHON_BIN}")
        else:
            # Try pythonw.exe as alternative
            pythonw_from_home = Path(python_home) / 'pythonw.exe'
            if pythonw_from_home.exists():
                PYTHON_BIN = pythonw_from_home
                print(f"   Found Python via PYTHON_HOME (pythonw): {PYTHON_BIN}")
    
    # If not found via PYTHON_HOME, try PYTHON environment variable (direct path)
    if not PYTHON_BIN:
        python_env = os.environ.get('PYTHON')
        if python_env:
            python_from_env = Path(python_env)
            if python_from_env.exists():
                PYTHON_BIN = python_from_env
                print(f"   Found Python via PYTHON env var: {PYTHON_BIN}")
    
    # Fallback to sys.executable (most reliable - uses current Python)
    if not PYTHON_BIN:
        PYTHON_BIN = Path(sys.executable)
        print(f"   Using current Python interpreter: {PYTHON_BIN}")
elif IS_LINUX:
    # Linux/EC2: Try EC2 Jupyter environment first, fallback to sys.executable
    PYTHON_BIN = Path('/home/pgx3874/jupyter-env/bin/python3.11')
    if not PYTHON_BIN.exists():
        PYTHON_BIN = Path(sys.executable)
        print(f"⚠️  EC2 Python path not found, using: {PYTHON_BIN}")
    else:
        print(f"   Using Linux/EC2 Python: {PYTHON_BIN}")
else:
    # Fallback: Use sys.executable
    PYTHON_BIN = Path(sys.executable)
    print(f"   Using fallback Python: {PYTHON_BIN}")

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

# Import project utilities for configuration (PROJECT_ROOT already added to sys.path above)
from py_helpers.constants import age_band_to_fname

# Configuration: Cohort and Age Band
COHORT = "opioid_ed"  # Change as needed: "opioid_ed" or "non_opioid_ed"
AGE_BAND = "13-24"    # Change as needed
AGE_BAND_FNAME = age_band_to_fname(AGE_BAND)

print(f"📋 Configuration:")
print(f"   Cohort: {COHORT}")
print(f"   Age Band: {AGE_BAND} ({AGE_BAND_FNAME})\n")

# %% [markdown]
# Step 3b: Interactive Feature Importance EDA and Refinement

# %% [markdown]
# ## Overview
# 
# This workflow reads aggregated feature importances for each cohort and runs additional analyses to identify features that should be filtered:
# 
# 1. **Load aggregated feature importances** from Step 3 for the specified cohort
# 2. **DTW trajectory analysis** → Identifies administrative/non-informative ICD/CPT codes (protocols, scheduling codes, etc.)
# 3. **BupaR post-target analysis** → Identifies features with target leakage (codes that appear primarily after the target event)
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
#     [DTW Analysis] → Filter administrative/non-informative ICD/CPT codes
#          ↓
#     [BupaR Analysis] → Filter post-target leakage features
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
# - **Section B**: DTW Trajectory Analysis (Non-Value-Added Codes)
# - **Section C**: BupaR Post-Target Analysis (Leakage Detection)
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
# Load aggregated feature importance from Step 3
possible_paths = [
    PROJECT_ROOT / "3_feature_importance" / "outputs" / COHORT / AGE_BAND / f"{COHORT}_{AGE_BAND_FNAME}_aggregated_feature_importance.csv",
    PROJECT_ROOT / "3_feature_importance" / "from_s3" / "by_cohort" / COHORT / AGE_BAND / f"{COHORT}_{AGE_BAND_FNAME}_aggregated_feature_importance.csv",
]

aggregated_fi = None
for path in possible_paths:
    if path.exists():
        aggregated_fi = pd.read_csv(path)
        print(f"✅ Loaded aggregated feature importance from: {path}")
        print(f"   Total features: {len(aggregated_fi):,}")
        break

if aggregated_fi is None:
    print(f"❌ Could not find aggregated feature importance file")
    print(f"   Checked paths:")
    for path in possible_paths:
        print(f"     - {path}")
else:
    # Display summary
    print(f"\n📊 Feature Importance Summary:")
    print(f"   Columns: {list(aggregated_fi.columns)}")
    print(f"\n   Top 10 features:")
    display(aggregated_fi.head(10))

# %% [markdown]
# ## B. DTW Trajectory Analysis (Administrative/Non-Informative Codes)
# 
# DTW analysis identifies administrative and non-informative ICD/CPT codes (e.g., protocols, scheduling codes, administrative codes) that don't add predictive value and should be filtered.

# %% [markdown]
# ### 1. Run DTW Trajectory Analysis

# %%
import subprocess
from datetime import datetime

# Check if configuration variables are defined
if 'COHORT' not in globals():
    raise NameError("COHORT is not defined. Please run the 'Configuration and Setup' section first.")
if 'AGE_BAND' not in globals():
    raise NameError("AGE_BAND is not defined. Please run the 'Configuration and Setup' section first.")

print("🚀 Running DTW Trajectory Analysis...")
print(f"Started at: {datetime.now()}")

cmd = [
    str(PYTHON_BIN),
    str(PROJECT_ROOT / "3b_feature_importance_eda" / "run_dtw_trajectory_analysis.py"),
    "--cohort", COHORT,
    "--age-band", AGE_BAND
]

result = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True)

print(result.stdout)
if result.stderr:
    print("STDERR:", result.stderr)

if result.returncode == 0:
    print(f"\n✅ DTW analysis completed successfully")
else:
    print(f"\n❌ DTW analysis failed with return code {result.returncode}")

# %% [markdown]
# ### 2. Load and Review DTW Results

# %%
# Load DTW results
dtw_results_path = OUTPUT_DIR / f"{COHORT}_{AGE_BAND_FNAME}_dtw_trajectory_analysis.csv"

if dtw_results_path.exists():
    dtw_results = pd.read_csv(dtw_results_path)
    print(f"✅ Loaded DTW results: {len(dtw_results)} features analyzed")
    
    # Show non-value-added features
    non_value_added = dtw_results[dtw_results.get('is_non_value_added', pd.Series([0]*len(dtw_results))) == 1]
    
    print(f"\n📊 DTW Analysis Summary:")
    print(f"   Total features analyzed: {len(dtw_results)}")
    print(f"   Non-value-added features: {len(non_value_added)}")
    
    if len(non_value_added) > 0:
        print(f"\n   ⚠️  Non-value-added features identified:")
        display(non_value_added[['feature', 'is_non_value_added']].head(20))
    else:
        print(f"\n   ✅ No non-value-added features identified")
    
    # Display full results
    print(f"\n   Full DTW results:")
    display(dtw_results.head(20))
else:
    print(f"❌ DTW results not found: {dtw_results_path}")
    dtw_results = pd.DataFrame()

# %% [markdown]
# ### 3. View DTW Visualizations

# %%
# Display DTW visualizations
dtw_plots = [
    f"dtw_trajectory_analysis_{COHORT}_{AGE_BAND_FNAME}.png",
    f"dtw_sample_trajectories_{COHORT}_{AGE_BAND_FNAME}.png"
]

for plot_name in dtw_plots:
    plot_path = PLOTS_DIR / plot_name
    if plot_path.exists():
        print(f"✅ Displaying: {plot_name}")
        display(Image(str(plot_path)))
    else:
        print(f"⚠️  Plot not found: {plot_path}")

# %% [markdown]
# ## C. BupaR Post-Target Analysis (Target Leakage Detection)
# 
# BupaR analysis identifies features with target leakage - ICD/CPT codes that appear primarily after the target event occurred, which would not be available for prediction and should be filtered.

# %% [markdown]
# ### 1. Verify Rscript is Available
# 
# **Note:** BupaR analysis uses R scripts, so Rscript must be installed and available in PATH. The Python script will automatically find Rscript, but you can verify it's available here.

# %%
# Verify Rscript is available
import shutil

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
# ### 2. Run BupaR Post-Target Analysis
# 
# **Note:** This Python script calls R scripts (`create_bupar_outputs_*.R`) using Rscript. The script automatically finds Rscript, so no manual configuration is needed.

# %%
import subprocess
from datetime import datetime

# Check if configuration variables are defined
if 'COHORT' not in globals():
    raise NameError("COHORT is not defined. Please run the 'Configuration and Setup' section first.")
if 'AGE_BAND' not in globals():
    raise NameError("AGE_BAND is not defined. Please run the 'Configuration and Setup' section first.")

print("🚀 Running BupaR Post-Target Analysis...")
print(f"Started at: {datetime.now()}")
print(f"Note: This will call R scripts using Rscript")

cmd = [
    str(PYTHON_BIN),
    str(PROJECT_ROOT / "3b_feature_importance_eda" / "run_bupar_post_target_analysis.py"),
    "--cohort", COHORT,
    "--age-band", AGE_BAND
]

result = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True)

print(result.stdout)
if result.stderr:
    print("STDERR:", result.stderr)

if result.returncode == 0:
    print(f"\n✅ BupaR analysis completed successfully")
else:
    print(f"\n❌ BupaR analysis failed with return code {result.returncode}")
    if "Rscript not found" in result.stderr or "Rscript not found" in result.stdout:
        print("\n💡 Tip: Make sure R is installed and Rscript is in your PATH")
        print(f"   Current RSCRIPT_BIN: {RSCRIPT_BIN if RSCRIPT_BIN else 'Not found (will use auto-detection)'}")
        print("   Rscript detection is configured in the OS detection section at the top of this file")

# %% [markdown]
# ### 3. Load and Review BupaR Results

# %%
# Load BupaR results
bupar_results_path = OUTPUT_DIR / f"{COHORT}_{AGE_BAND_FNAME}_bupar_post_target_analysis.csv"

if bupar_results_path.exists():
    bupar_results = pd.read_csv(bupar_results_path)
    print(f"✅ Loaded BupaR results: {len(bupar_results)} features analyzed")
    
    # Show post-target leakage features
    post_target_leakage = bupar_results[bupar_results.get('is_post_target_leakage', pd.Series([0]*len(bupar_results))) == 1]
    
    print(f"\n📊 BupaR Analysis Summary:")
    print(f"   Total features analyzed: {len(bupar_results)}")
    print(f"   Post-target leakage features: {len(post_target_leakage)}")
    
    if len(post_target_leakage) > 0:
        print(f"\n   ⚠️  Post-target leakage features identified:")
        display(post_target_leakage[['feature', 'is_post_target_leakage']].head(20))
    else:
        print(f"\n   ✅ No post-target leakage features identified")
    
    # Display full results
    print(f"\n   Full BupaR results:")
    display(bupar_results.head(20))
else:
    print(f"❌ BupaR results not found: {bupar_results_path}")
    bupar_results = pd.DataFrame()

# %% [markdown]
# ### 4. View BupaR Visualizations

# %%
# Display BupaR visualizations
bupar_plots = [
    f"{COHORT}_{AGE_BAND_FNAME}_overall_activity_frequency.png",
    f"{COHORT}_{AGE_BAND_FNAME}_activity_milestones_gantt.png",
    f"{COHORT}_{AGE_BAND_FNAME}_activity_sequence_top.png",
    f"{COHORT}_{AGE_BAND_FNAME}_pre_f1120_activity_frequency.png",
    f"{COHORT}_{AGE_BAND_FNAME}_post_f1120_activity_frequency.png",
]

for plot_name in bupar_plots:
    plot_path = PLOTS_DIR / plot_name
    if plot_path.exists():
        print(f"✅ Displaying: {plot_name}")
        display(Image(str(plot_path)))
    else:
        print(f"⚠️  Plot not found: {plot_path}")

# %% [markdown]
# ## D. Interactive Code Review and Filtering
# 
# Review the analysis results and manually add/remove codes that should be filtered before Step 4a.

# %% [markdown]
# ### 1. Review Codes to Filter
# 
# Based on the DTW and BupaR analyses, review codes that should be filtered:

# %%
# Combine filtering recommendations
filtering_recommendations = {
    'dtw_non_value_added': set(),
    'bupar_post_target': set(),
    'manual_additional': set()  # Add codes manually here
}

# Add DTW recommendations
if 'dtw_results' in locals() and not dtw_results.empty:
    dtw_filtered = dtw_results[dtw_results.get('is_non_value_added', pd.Series([0]*len(dtw_results))) == 1]
    filtering_recommendations['dtw_non_value_added'] = set(dtw_filtered['feature'].tolist())

# Add BupaR recommendations
if 'bupar_results' in locals() and not bupar_results.empty:
    bupar_filtered = bupar_results[bupar_results.get('is_post_target_leakage', pd.Series([0]*len(bupar_results))) == 1]
    filtering_recommendations['bupar_post_target'] = set(bupar_filtered['feature'].tolist())

# Display summary
print("📋 Filtering Recommendations Summary:")
print(f"   DTW non-value-added codes: {len(filtering_recommendations['dtw_non_value_added'])}")
print(f"   BupaR post-target leakage codes: {len(filtering_recommendations['bupar_post_target'])}")
print(f"   Manual additional codes: {len(filtering_recommendations['manual_additional'])}")

# Show codes to filter
all_codes_to_filter = (
    filtering_recommendations['dtw_non_value_added'] |
    filtering_recommendations['bupar_post_target'] |
    filtering_recommendations['manual_additional']
)

print(f"\n   Total unique codes to filter: {len(all_codes_to_filter)}")

if len(all_codes_to_filter) > 0:
    print(f"\n   Codes recommended for filtering:")
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
    filtering_recommendations['dtw_non_value_added'].discard(code)
    filtering_recommendations['bupar_post_target'].discard(code)
    filtering_recommendations['manual_additional'].discard(code)

# Final list of codes to filter
final_codes_to_filter = (
    filtering_recommendations['dtw_non_value_added'] |
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
    'dtw_non_value_added_count': len(filtering_recommendations['dtw_non_value_added']),
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
import subprocess
from datetime import datetime

# Check if configuration variables are defined
if 'COHORT' not in globals():
    raise NameError("COHORT is not defined. Please run the 'Configuration and Setup' section first.")
if 'AGE_BAND' not in globals():
    raise NameError("AGE_BAND is not defined. Please run the 'Configuration and Setup' section first.")

print("🚀 Filtering and Refining Features...")
print(f"Started at: {datetime.now()}")

cmd = [
    str(PYTHON_BIN),
    str(PROJECT_ROOT / "3b_feature_importance_eda" / "filter_and_refine_features.py"),
    "--cohort", COHORT,
    "--age-band", AGE_BAND
]

result = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True)

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
# ✅ **Step 3b Interactive Analysis Complete**
# 
# **Outputs Generated:**
# - ✅ DTW trajectory analysis results
# - ✅ BupaR post-target analysis results  
# - ✅ Refined `cohort_feature_importance.csv` for Step 4a
# - ✅ Filtering summary JSON
# - ✅ Visualizations (DTW and BupaR)
# 
# **Next Steps:**
# - Proceed to **Step 4a: Model Data Creation** using the refined feature importances
# - The `cohort_feature_importance.csv` file is available locally and in S3
