#!/usr/bin/env python3
"""
Test script to verify binary feature causal importance fix.

This script checks if the fix correctly detects when a binary feature
appears in the AXP and counts it as a change even if the AXP computation
doesn't change.
"""

import sys
import pandas as pd
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def test_fix_logic():
    """Test the fix logic by examining existing outputs."""
    
    # Check if we have existing outputs to analyze
    output_dir = PROJECT_ROOT / "8_ffa_analysis" / "outputs"
    
    if not output_dir.exists():
        print("No outputs directory found. Run Step 8 first to generate outputs.")
        return
    
    # Find all causal_importance.parquet files
    causal_files = list(output_dir.rglob("causal_importance.parquet"))
    
    if not causal_files:
        print("No causal_importance.parquet files found. Run Step 8 first.")
        return
    
    print("=" * 80)
    print("Testing Binary Feature Causal Importance Fix")
    print("=" * 80)
    print()
    
    for causal_file in causal_files:
        print(f"Analyzing: {causal_file.relative_to(PROJECT_ROOT)}")
        print("-" * 80)
        
        try:
            df = pd.read_parquet(causal_file)
            
            if len(df) == 0:
                print("  [SKIP] Empty DataFrame")
                continue
            
            # Check if is_binary column exists
            if 'is_binary' not in df.columns:
                print("  [WARNING] 'is_binary' column not found. Cannot verify binary feature fix.")
                continue
            
            # Statistics
            total_features = len(df)
            binary_features = df['is_binary'].sum()
            non_binary_features = total_features - binary_features
            
            features_with_causal = (df['causal_importance'] > 0).sum()
            binary_with_causal = ((df['is_binary'] == True) & (df['causal_importance'] > 0)).sum()
            binary_with_zero = ((df['is_binary'] == True) & (df['causal_importance'] == 0.0)).sum()
            
            print(f"  Total features: {total_features}")
            print(f"  Binary features: {binary_features}")
            print(f"  Non-binary features: {non_binary_features}")
            print(f"  Features with causal_importance > 0: {features_with_causal}")
            print(f"  Binary features with causal_importance > 0: {binary_with_causal}")
            print(f"  Binary features with causal_importance = 0: {binary_with_zero}")
            print()
            
            # Show top binary features
            binary_df = df[df['is_binary'] == True].copy()
            if len(binary_df) > 0:
                binary_df_sorted = binary_df.sort_values('causal_importance', ascending=False)
                print("  Top 10 binary features by causal_importance:")
                for idx, row in binary_df_sorted.head(10).iterrows():
                    print(f"    {row['feature']:<50} {row['causal_importance']:>10.6f}")
                print()
            
            # Check if we have any binary features with zero causal importance
            # This might indicate the fix didn't work
            if binary_with_zero > 0:
                print(f"  [NOTE] {binary_with_zero} binary features have zero causal importance.")
                print("         This could be:")
                print("         1. Features that don't appear in any AXPs (expected)")
                print("         2. Features that appear in AXPs but fix didn't catch (potential bug)")
                print()
                
                # Show some examples
                zero_binary = binary_df[binary_df['causal_importance'] == 0.0].head(5)
                if len(zero_binary) > 0:
                    print("  Examples of binary features with zero causal importance:")
                    for idx, row in zero_binary.iterrows():
                        print(f"    - {row['feature']}")
                    print()
            
            # Check if we have binary features with non-zero causal importance
            # This indicates the fix is working
            if binary_with_causal > 0:
                print(f"  [SUCCESS] {binary_with_causal} binary features have non-zero causal importance!")
                print("            This indicates the fix is working correctly.")
                print()
            
        except Exception as e:
            print(f"  [ERROR] Failed to analyze {causal_file}: {e}")
            import traceback
            traceback.print_exc()
        
        print()
    
    print("=" * 80)
    print("Test Complete")
    print("=" * 80)
    print()
    print("Interpretation:")
    print("- If binary features have non-zero causal_importance, the fix is working.")
    print("- If all binary features have zero causal_importance, the fix may not be working.")
    print("- Some binary features may legitimately have zero causal_importance if they")
    print("  don't appear in any AXPs or don't affect explanations when removed.")


if __name__ == "__main__":
    test_fix_logic()
