# %%
# ICD and CPT Code Validation Workflow
# This script validates which ICD and CPT codes are informative vs administrative

import sys
import os
import platform
from pathlib import Path

# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    import io
    if hasattr(sys.stdout, 'buffer'):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Detect operating system
IS_WINDOWS = platform.system() == 'Windows'
IS_LINUX = platform.system() == 'Linux'

print(f"Detected OS: {platform.system()}")

# Set project root based on OS
if IS_WINDOWS:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
elif IS_LINUX:
    PROJECT_ROOT = Path('/home/pgx3874/pgx-analysis')
else:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

print(f"Project Root: {PROJECT_ROOT}")
print(f"OS detection and path setup complete\n")

# Import project utilities
from py_helpers.constants import age_band_to_fname

# Configuration
COHORT = "opioid_ed"  # Change as needed: "opioid_ed" or "non_opioid_ed"
AGE_BAND = "13-24"    # Change as needed
AGE_BAND_FNAME = age_band_to_fname(AGE_BAND)

print(f"Configuration:")
print(f"   Cohort: {COHORT}")
print(f"   Age Band: {AGE_BAND} ({AGE_BAND_FNAME})\n")

# %% [markdown]
# ## ICD and CPT Code Validation
# 
# This workflow validates which ICD and CPT codes in feature importance data are **informative** (should be kept) vs **administrative/non-informative** (should be filtered).

# %% [markdown]
# ### Step 1: Run Code Group Analysis

# %%
import subprocess
from datetime import datetime

print("Running Code Group Analysis...")
print(f"Started at: {datetime.now()}")

# Import and run the analysis function
sys.path.insert(0, str(Path(__file__).parent))
from analyze_code_groups import analyze_code_groups

icd_summary, cpt_summary = analyze_code_groups(COHORT, AGE_BAND, PROJECT_ROOT)

print(f"\nAnalysis completed at: {datetime.now()}")

# %% [markdown]
# ### Step 2: Review Administrative Codes Lookup

# %%
import json

# Load administrative codes lookup (check local copy first, then original location)
local_lookup_path = Path(__file__).parent / "administrative_codes_lookup.json"
admin_lookup_path = PROJECT_ROOT / "4b_dtw_filter" / "administrative_codes_lookup.json"

# Use local copy if available, otherwise use original
lookup_path = local_lookup_path if local_lookup_path.exists() else admin_lookup_path

if lookup_path.exists():
    with open(lookup_path, 'r') as f:
        admin_lookup = json.load(f)
    
    print("Administrative Codes Lookup Table:")
    print("="*80)
    print(f"Description: {admin_lookup.get('description', 'N/A')}")
    print(f"Version: {admin_lookup.get('version', 'N/A')}")
    print(f"Last Updated: {admin_lookup.get('last_updated', 'N/A')}")
    
    admin_codes = admin_lookup.get('administrative_codes', {})
    print(f"\nAdministrative ICD codes: {len(admin_codes.get('icd', []))}")
    for code in admin_codes.get('icd', []):
        print(f"  - {code}")
    
    print(f"\nAdministrative CPT codes: {len(admin_codes.get('cpt', []))}")
    for code in admin_codes.get('cpt', []):
        print(f"  - {code}")
else:
    print(f"Administrative codes lookup not found at:")
    print(f"  Local: {local_lookup_path}")
    print(f"  Original: {admin_lookup_path}")

# %% [markdown]
# ### Step 3: Summary and Recommendations

# %%
print("\n" + "="*80)
print("VALIDATION SUMMARY")
print("="*80)

if icd_summary is not None and not icd_summary.empty:
    total_icd = icd_summary['total_codes'].sum()
    admin_icd = icd_summary['administrative_codes'].sum()
    informative_icd = icd_summary['informative_codes'].sum()
    
    print(f"\nICD Codes:")
    print(f"  Total: {total_icd}")
    print(f"  Administrative: {admin_icd} ({100*admin_icd/total_icd:.2f}%)")
    print(f"  Informative: {informative_icd} ({100*informative_icd/total_icd:.2f}%)")
    
    mixed_icd = icd_summary[icd_summary['classification'] == 'Mixed']
    if not mixed_icd.empty:
        print(f"\n  Mixed chapters (contain both administrative and informative):")
        for _, row in mixed_icd.iterrows():
            print(f"    {row['letter']} - {row['chapter']}: {row['administrative_codes']} admin, {row['informative_codes']} informative")

if cpt_summary is not None and not cpt_summary.empty:
    total_cpt = cpt_summary['total_codes'].sum()
    admin_cpt = cpt_summary['administrative_codes'].sum()
    informative_cpt = cpt_summary['informative_codes'].sum()
    
    print(f"\nCPT Codes:")
    print(f"  Total: {total_cpt}")
    print(f"  Administrative: {admin_cpt} ({100*admin_cpt/total_cpt:.2f}%)")
    print(f"  Informative: {informative_cpt} ({100*informative_cpt/total_cpt:.2f}%)")
    
    mixed_cpt = cpt_summary[cpt_summary['classification'] == 'Mixed']
    if not mixed_cpt.empty:
        print(f"\n  Mixed ranges (contain both administrative and informative):")
        for _, row in mixed_cpt.iterrows():
            print(f"    {row['range']} - {row['description']}: {row['administrative_codes']} admin, {row['informative_codes']} informative")

print("\n" + "="*80)
print("RECOMMENDATIONS")
print("="*80)
print("1. Review administrative codes in lookup table")
print("2. Verify all administrative codes are being filtered in Feature Importance EDA")
print("3. Check if any additional codes should be added to administrative list")
print("4. Document findings in 3_feature_importance/README.md")
print("="*80)
