#!/usr/bin/env python3
"""Check the status of the FFA analysis test."""

from pathlib import Path
import time
import os

log_file = Path('logs/ffa_analysis_20260110_200450.log')
causal_file = Path('outputs/opioid_ed/13_24/xgboost/causal_importance.parquet')

print("=" * 80)
print("FFA Analysis Test Status")
print("=" * 80)

# Check log file
if log_file.exists():
    size = log_file.stat().st_size
    mtime = log_file.stat().st_mtime
    age = time.time() - mtime
    print(f"\nLog file: {log_file}")
    print(f"  Size: {size:,} bytes")
    print(f"  Last modified: {age/60:.1f} minutes ago")
    
    # Read last 20 lines
    try:
        with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
            print(f"  Total lines: {len(lines)}")
            if lines:
                print(f"\n  Last 10 lines:")
                for line in lines[-10:]:
                    print(f"    {line.rstrip()}")
    except Exception as e:
        print(f"  Error reading log: {e}")
else:
    print(f"\nLog file not found: {log_file}")

# Check causal file
print(f"\nCausal importance file: {causal_file}")
if causal_file.exists():
    size = causal_file.stat().st_size
    mtime = causal_file.stat().st_mtime
    age = time.time() - mtime
    print(f"  ✓ EXISTS")
    print(f"  Size: {size/1024:.1f} KB")
    print(f"  Last modified: {age/60:.1f} minutes ago")
    
    # Try to read it
    try:
        import pandas as pd
        df = pd.read_parquet(causal_file)
        print(f"  Features analyzed: {len(df)}")
        binary_features = df[df.get('is_binary', pd.Series([False]*len(df))) == True] if 'is_binary' in df.columns else pd.DataFrame()
        if len(binary_features) > 0:
            binary_with_causal = binary_features[binary_features['causal_importance'] > 0]
            print(f"  Binary features: {len(binary_features)}")
            print(f"  Binary with causal > 0: {len(binary_with_causal)}")
    except Exception as e:
        print(f"  Error reading file: {e}")
else:
    print(f"  NOT FOUND (analysis may still be running or failed)")

print("\n" + "=" * 80)
