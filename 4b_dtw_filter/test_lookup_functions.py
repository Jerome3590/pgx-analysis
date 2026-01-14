#!/usr/bin/env python3
"""Test the lookup functions directly."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from research_icd_cpt_codes import lookup_cpt_code_web, lookup_icd_code_web

# Test CPT lookup
print("Testing CPT Code 99213:")
print("="*80)
desc, resp = lookup_cpt_code_web('99213')
print(f"Description: {desc[:150] if desc else '(none)'}")
print(f"Response: {resp[:200]}")

print("\n" + "="*80)
print("Testing ICD Code F11.20:")
print("="*80)
desc2, resp2 = lookup_icd_code_web('F1120', 'F11.20')
print(f"Description: {desc2[:150] if desc2 else '(none)'}")
print(f"Response: {resp2[:200]}")
