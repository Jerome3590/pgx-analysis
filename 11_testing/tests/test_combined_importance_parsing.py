#!/usr/bin/env python3
"""
Diagnostic script to test combined_importance.csv parsing and code extraction.
Run this on EC2 to verify the CSV files can be read and codes extracted.
"""
import sys
from pathlib import Path

# Add repo root to path (script lives in 11_testing/tests/)
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from py_helpers.shap_ffa_fpgrowth_utils import (
    _load_combined_importance_from_dashboard,
    _parse_feature_name,
    get_shap_ffa_important_codes,
    get_shap_ffa_allowed_codes_combined,
)

# Test one cohort/age_band
cohort = "opioid_ed"
age_band = "25-44"
age_band_fname = age_band.replace("-", "_")

print("=" * 80)
print(f"Testing combined_importance.csv parsing for {cohort}/{age_band}")
print("=" * 80)

# Check if file exists
csv_path = REPO_ROOT / "10_risk_dashboard" / "outputs" / cohort / age_band_fname / "combined_importance.csv"
print(f"\n1. Checking file existence:")
print(f"   Path: {csv_path}")
print(f"   Exists: {csv_path.exists()}")

if not csv_path.exists():
    print("\n❌ File does not exist! Cannot proceed.")
    sys.exit(1)

# Try loading with the function
print(f"\n2. Loading CSV with _load_combined_importance_from_dashboard:")
df = _load_combined_importance_from_dashboard(cohort, age_band, project_root=REPO_ROOT)
print(f"   Rows: {len(df)}")
print(f"   Columns: {list(df.columns) if not df.empty else 'N/A'}")

if df.empty:
    print("\n❌ DataFrame is empty! Function failed to load/parse CSV.")
    print("\n   Trying direct pandas read...")
    import pandas as pd
    try:
        df_raw = pd.read_csv(csv_path)
        print(f"   Direct read successful: {len(df_raw)} rows, columns: {list(df_raw.columns)}")
        print(f"\n   First 5 rows:")
        print(df_raw.head())
    except Exception as e:
        print(f"   Direct read failed: {e}")
    sys.exit(1)

# Show top features
print(f"\n3. Top 10 features (by importance):")
print(df.head(10).to_string(index=False))

# Test feature parsing
print(f"\n4. Testing feature name parsing on top 10 features:")
for feat in df.head(10)["feature"]:
    code_type, code = _parse_feature_name(feat)
    print(f"   {feat:40s} → type={code_type:8s}, code={code}")

# Test code extraction for each item type
print(f"\n5. Extracting codes by type (top 500):")
drug_codes = get_shap_ffa_important_codes(
    cohort, age_band, "drug_name", top_n=500,
    project_root=REPO_ROOT, data_root=None,
    use_shap=True, use_ffa=True
)
icd_codes = get_shap_ffa_important_codes(
    cohort, age_band, "icd_code", top_n=500,
    project_root=REPO_ROOT, data_root=None,
    use_shap=True, use_ffa=True
)
cpt_codes = get_shap_ffa_important_codes(
    cohort, age_band, "cpt_code", top_n=500,
    project_root=REPO_ROOT, data_root=None,
    use_shap=True, use_ffa=True
)

print(f"   Drug codes: {len(drug_codes)}")
if drug_codes:
    print(f"     Examples: {sorted(list(drug_codes))[:5]}")
print(f"   ICD codes: {len(icd_codes)}")
if icd_codes:
    print(f"     Examples: {sorted(list(icd_codes))[:5]}")
print(f"   CPT codes: {len(cpt_codes)}")
if cpt_codes:
    print(f"     Examples: {sorted(list(cpt_codes))[:5]}")

# Test combined extraction
print(f"\n6. Testing combined code extraction:")
all_codes = get_shap_ffa_allowed_codes_combined(
    cohort, age_band, top_n=500,
    project_root=REPO_ROOT, data_root=None,
    use_shap=True, use_ffa=True
)
print(f"   Total codes: {len(all_codes)}")
if all_codes:
    print(f"   Examples: {sorted(list(all_codes))[:10]}")
    print("\n✅ SUCCESS! Codes extracted successfully.")
else:
    print("\n❌ FAILED! No codes extracted.")
    sys.exit(1)

print("\n" + "=" * 80)
print("Diagnostic complete.")
print("=" * 80)
