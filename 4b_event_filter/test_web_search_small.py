#!/usr/bin/env python3
"""Test web search with just a few codes."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from research_icd_cpt_codes import lookup_code_meanings

# Test with just a few codes
test_codes = {
    'icd': {'F1120', 'Z00129'},
    'cpt': {'99213', '99284'},
    'hcpcs': set()
}

print("Testing web search with a few codes...")
print("="*80)

df = lookup_code_meanings(test_codes, use_web_search=True)

print("\n" + "="*80)
print("Results:")
print("="*80)
for _, row in df.iterrows():
    print(f"\nCode: {row['code']} ({row['code_type']})")
    print(f"  Description: {row['description'][:100] if row['description'] else '(none)'}")
    print(f"  Classification: {row['classification']}")
    print(f"  Source: {row['lookup_source']}")
    print(f"  Response: {row['lookup_response'][:150]}")
