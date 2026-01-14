"""
Quick test to verify local ICD lookup is working.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from research_icd_cpt_codes import lookup_icd_code_local, load_icd10_lookup, load_icd9_to_icd10_mapping

# Test codes
test_codes = [
    "A000",  # ICD-10
    "F1120",  # ICD-10 (should format to F11.20)
    "250.00",  # ICD-9
    "042",  # ICD-9 (short format)
    "Z00129",  # ICD-10
]

print("Testing local ICD lookup...")
print("=" * 80)

# Preload
print("\nLoading lookup files...")
load_icd10_lookup()
load_icd9_to_icd10_mapping()
print()

# Test lookups
print("Testing code lookups:")
print("-" * 80)
for code in test_codes:
    result = lookup_icd_code_local(code)
    if result:
        description, source = result
        print(f"[OK] {code:10s} -> {description[:60]}... (source: {source})")
    else:
        print(f"[NOT FOUND] {code:10s}")

print("\n" + "=" * 80)
print("Test complete!")
