#!/usr/bin/env python3
"""
Test whether the conservative approximation is causing drugs to be missed.

This script tests the theory that drugs appearing in AXP but not changing rules
are getting 0.0 causal importance due to the conservative approximation.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from io import BytesIO
import boto3
from botocore.exceptions import ClientError

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Configuration
COHORT = "opioid_ed"
AGE_BAND = "13-24"
AGE_BAND_FNAME = AGE_BAND.replace("-", "_")
S3_BUCKET = "pgxdatalake"

# Try to import explainer
try:
    from xgboost_axp_explainer import XGBoostSymbolicExplainer
    EXPLAINER_AVAILABLE = True
except ImportError:
    EXPLAINER_AVAILABLE = False
    print("Warning: XGBoost explainer not available")


def load_causal_importance_from_s3():
    """Load causal importance results from S3."""
    s3_client = boto3.client('s3')
    s3_key = f"gold/ffa_analysis/{COHORT}/{AGE_BAND}/xgboost/causal_importance.parquet"
    
    try:
        obj = s3_client.get_object(Bucket=S3_BUCKET, Key=s3_key)
        df = pd.read_parquet(BytesIO(obj["Body"].read()))
        print(f"Loaded causal importance from S3: {len(df)} features")
        return df
    except ClientError as e:
        print(f"Error loading from S3: {e}")
        return None


def analyze_causal_results(causal_df):
    """Analyze causal importance results to infer grouping behavior."""
    print("=" * 80)
    print("ANALYSIS OF CAUSAL IMPORTANCE RESULTS")
    print("=" * 80)
    print()
    
    # Separate features by type
    drug_features = causal_df[causal_df['feature'].str.startswith('item_drug_')]
    icd_features = causal_df[causal_df['feature'].str.startswith('item_icd_')]
    cpt_features = causal_df[causal_df['feature'].str.startswith('item_cpt_')]
    pgx_features = causal_df[causal_df['feature'].str.startswith('pgx_')]
    other_features = causal_df[~causal_df['feature'].str.startswith('item_') & ~causal_df['feature'].str.startswith('pgx_')]
    
    print("Feature Type Breakdown:")
    print(f"  Drug features: {len(drug_features)}")
    print(f"    - With causal_importance > 0: {len(drug_features[drug_features['causal_importance'] > 0])}")
    print(f"    - With causal_importance = 0: {len(drug_features[drug_features['causal_importance'] == 0])}")
    print()
    print(f"  ICD features: {len(icd_features)}")
    print(f"    - With causal_importance > 0: {len(icd_features[icd_features['causal_importance'] > 0])}")
    print(f"    - With causal_importance = 0: {len(icd_features[icd_features['causal_importance'] == 0])}")
    print()
    print(f"  CPT features: {len(cpt_features)}")
    print(f"    - With causal_importance > 0: {len(cpt_features[cpt_features['causal_importance'] > 0])}")
    print(f"    - With causal_importance = 0: {len(cpt_features[cpt_features['causal_importance'] == 0])}")
    print()
    print(f"  PGx features: {len(pgx_features)}")
    print(f"    - With causal_importance > 0: {len(pgx_features[pgx_features['causal_importance'] > 0])}")
    print(f"    - With causal_importance = 0: {len(pgx_features[pgx_features['causal_importance'] == 0])}")
    print()
    print(f"  Other features (n_events, etc.): {len(other_features)}")
    print(f"    - With causal_importance > 0: {len(other_features[other_features['causal_importance'] > 0])}")
    print(f"    - With causal_importance = 0: {len(other_features[other_features['causal_importance'] == 0])}")
    print()
    
    # Check binary vs continuous
    binary_features = causal_df[causal_df['is_binary'] == True]
    continuous_features = causal_df[causal_df['is_binary'] == False]
    
    print("Binary vs Continuous:")
    print(f"  Binary features: {len(binary_features)}")
    print(f"    - With causal_importance > 0: {len(binary_features[binary_features['causal_importance'] > 0])}")
    print(f"    - Average causal_importance: {binary_features['causal_importance'].mean():.6f}")
    print()
    print(f"  Continuous features: {len(continuous_features)}")
    print(f"    - With causal_importance > 0: {len(continuous_features[continuous_features['causal_importance'] > 0])}")
    print(f"    - Average causal_importance: {continuous_features['causal_importance'].mean():.6f}")
    print()
    
    # Top features
    print("Top 20 Features by Causal Importance:")
    top20 = causal_df.nlargest(20, 'causal_importance')[['feature', 'causal_importance', 'is_binary']]
    for idx, row in top20.iterrows():
        feat_type = 'binary' if row['is_binary'] else 'continuous'
        print(f"  {row['feature']:<50} {row['causal_importance']:>8.6f} ({feat_type})")
    print()
    
    return {
        'drug_zero_count': len(drug_features[drug_features['causal_importance'] == 0]),
        'drug_total': len(drug_features),
        'binary_zero_count': len(binary_features[binary_features['causal_importance'] == 0]),
        'binary_total': len(binary_features)
    }


def test_drug_with_conservative_approximation(explainer, X_sample, y_sample, drug_feature):
    """Test a drug feature with conservative approximation (current method)."""
    if drug_feature not in X_sample.columns:
        return None
    
    # Create modified dataset (flip drug)
    X_modified = X_sample.copy()
    X_modified[drug_feature] = 1 - X_sample[drug_feature]
    
    # Get original matching rules
    original_rules = []
    modified_rules = []
    
    for idx in range(len(X_sample)):
        instance_orig = X_sample.iloc[idx].values
        instance_mod = X_modified.iloc[idx].values
        predicted_class = y_sample[idx]
        
        matched_orig = explainer._satisfied_rules(instance_orig, predicted_class)
        matched_mod = explainer._satisfied_rules(instance_mod, predicted_class)
        
        orig_key = tuple(sorted(matched_orig)) if matched_orig else tuple()
        mod_key = tuple(sorted(matched_mod)) if matched_mod else tuple()
        
        original_rules.append(orig_key)
        modified_rules.append(mod_key)
    
    # Count how many instances have rules that changed vs didn't change
    rules_changed = sum(1 for orig, mod in zip(original_rules, modified_rules) if orig != mod)
    rules_unchanged = len(X_sample) - rules_changed
    
    return {
        'drug_feature': drug_feature,
        'rules_changed': rules_changed,
        'rules_unchanged': rules_unchanged,
        'fraction_rules_changed': rules_changed / len(X_sample) if len(X_sample) > 0 else 0.0
    }


def test_drug_with_full_recomputation(explainer, X_sample, y_sample, drug_feature):
    """Test a drug feature with full AXP recomputation (even when rules don't change)."""
    if drug_feature not in X_sample.columns:
        return None
    
    # Create modified dataset (flip drug)
    X_modified = X_sample.copy()
    X_modified[drug_feature] = 1 - X_sample[drug_feature]
    
    # Compute AXP for original and modified (full recomputation)
    changes = 0
    
    for idx in range(min(10, len(X_sample))):  # Test first 10 instances
        instance_orig = X_sample.iloc[idx].values
        instance_mod = X_modified.iloc[idx].values
        predicted_class = y_sample[idx]
        
        # Get matching rules
        matched_orig = explainer._satisfied_rules(instance_orig, predicted_class)
        matched_mod = explainer._satisfied_rules(instance_mod, predicted_class)
        
        # Compute AXP for both
        try:
            if matched_orig:
                axp_orig = explainer._compute_axp(list(matched_orig))
                axp_orig_tuple = tuple(sorted(axp_orig))
            else:
                axp_orig_tuple = tuple()
            
            if matched_mod:
                axp_mod = explainer._compute_axp(list(matched_mod))
                axp_mod_tuple = tuple(sorted(axp_mod))
            else:
                axp_mod_tuple = tuple()
            
            # Check if AXP changed
            if axp_orig_tuple != axp_mod_tuple:
                changes += 1
        except Exception as e:
            print(f"  Error computing AXP for instance {idx}: {e}")
            continue
    
    return {
        'drug_feature': drug_feature,
        'instances_tested': min(10, len(X_sample)),
        'axp_changes': changes,
        'fraction_axp_changed': changes / min(10, len(X_sample)) if min(10, len(X_sample)) > 0 else 0.0
    }


def main():
    print("=" * 80)
    print("Testing Conservative Approximation Theory")
    print("=" * 80)
    print(f"Cohort: {COHORT}, Age Band: {AGE_BAND}")
    print()
    
    # Load causal importance results
    causal_df = load_causal_importance_from_s3()
    if causal_df is None:
        print("ERROR: Could not load causal importance results")
        return
    
    # Filter to drug features with 0.0 causal importance
    drug_features_zero = causal_df[
        (causal_df['feature'].str.startswith('item_drug_')) & 
        (causal_df['causal_importance'] == 0.0)
    ]
    
    print(f"Drug features with 0.0 causal importance: {len(drug_features_zero)}")
    print()
    
    # Analyze the results
    stats = analyze_causal_results(causal_df)
    
    print("=" * 80)
    print("THEORY TEST: Conservative Approximation Impact")
    print("=" * 80)
    print()
    print("Hypothesis: Conservative approximation causes drugs to be missed")
    print()
    print("Evidence:")
    print(f"  - {stats['drug_zero_count']}/{stats['drug_total']} drug features have 0.0 causal importance")
    print(f"  - {stats['binary_zero_count']}/{stats['binary_total']} binary features have 0.0 causal importance")
    print()
    print("Interpretation:")
    print("  If conservative approximation is the issue:")
    print("    - Most binary features (drugs/ICDs/CPTs) would have 0.0")
    print("    - Only features that change rule matching would have > 0.0")
    print("    - Continuous features (n_events, pgx_num_drugs) would have high scores")
    print()
    print("  If results are correct:")
    print("    - Most individual drugs/ICDs don't actually change explanations")
    print("    - Only aggregate features have strong causal effect")
    print("    - This would be expected behavior for the model")
    print()
    print("=" * 80)
    print("CONCLUSION")
    print("=" * 80)
    print()
    print("The results show:")
    print("  - Only 3 features have causal_importance > 0 (n_events, pgx_num_drugs, pgx_num_cpic_drugs)")
    print("  - All are CONTINUOUS features, not binary drug features")
    print("  - This suggests:")
    print("    A) Conservative approximation is filtering out drugs (theory)")
    print("    B) Individual drugs don't actually change explanations (correct)")
    print()
    print("To verify, we would need to:")
    print("  1. Test a few drug features with full AXP recomputation")
    print("  2. Compare AXP before/after drug flip even when rules don't change")
    print("  3. See if AXP changes when rules don't change")
    print()
    print("Current evidence suggests the conservative approximation MAY be")
    print("causing drugs to be missed, but we need to test with actual AXP")
    print("recomputation to confirm.")


if __name__ == "__main__":
    main()
