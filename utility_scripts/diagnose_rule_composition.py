#!/usr/bin/env python3
"""
Diagnostic script to analyze rule composition and feature coverage.

This script helps understand:
1. What features appear in rules
2. Which rules contain features with SHAP > 0
3. Whether rules without SHAP > 0 features are noise
4. Coverage of features with SHAP > 0 in the filtered rule set
"""

import sys
import json
from pathlib import Path
from typing import Dict, List, Set
from collections import defaultdict, Counter
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import boto3
    from botocore.exceptions import ClientError
except ImportError:
    print("ERROR: boto3 not installed. Install with: pip install boto3")
    sys.exit(1)

# S3 client
s3_client = boto3.client('s3')
OUTPUT_BUCKET = "pgxdatalake"

# Define cohorts
COHORTS = {
    "opioid_ed": ["13-24", "25-44", "45-54", "55-64"],
    "non_opioid_ed": ["65-74", "75-84", "85-94"],
}


def load_shap_importance(cohort: str, age_band: str) -> Dict[str, float]:
    """Load SHAP importance map from S3."""
    age_band_fname = age_band.replace("-", "_")
    
    # Try XGBoost first
    s3_key = f"gold/shap_analysis/{cohort}/{age_band}/{cohort}_{age_band_fname}_shap_global_importance_xgboost.csv"
    
    try:
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
            tmp_path = tmp_file.name
        
        s3_client.download_file(OUTPUT_BUCKET, s3_key, tmp_path)
        df = pd.read_csv(tmp_path)
        
        # Create SHAP map (feature -> mean_abs_shap)
        shap_map = {}
        if 'feature' in df.columns and 'mean_abs_shap' in df.columns:
            shap_map = dict(zip(df['feature'], df['mean_abs_shap']))
        elif 'feature_name' in df.columns and 'importance' in df.columns:
            shap_map = dict(zip(df['feature_name'], df['importance']))
        
        Path(tmp_path).unlink()
        return shap_map
    except Exception as e:
        print(f"  Error loading SHAP: {e}")
        return {}


def load_model_json(cohort: str, age_band: str) -> Dict:
    """Load model JSON from S3 or local."""
    age_band_fname = age_band.replace("-", "_")
    
    # Try S3 first
    s3_key = f"gold/final_model/{cohort}/{age_band_fname}/*_best_xgboost_model.json"
    
    # For now, try local
    local_path = PROJECT_ROOT / "6_final_model" / "outputs" / cohort / age_band_fname / "final_model_json" / f"{cohort}_{age_band_fname}_best_xgboost_model.json"
    
    if local_path.exists():
        with open(local_path, 'r') as f:
            return json.load(f)
    
    return None


def extract_rules_from_model(model_json: Dict) -> List[Dict]:
    """Extract rules from XGBoost model JSON."""
    if model_json is None or 'trees' not in model_json:
        return []
    
    rules = []
    feature_names = model_json.get('feature_names', [])
    
    for tree_idx, tree_dump in enumerate(model_json.get('trees', [])):
        # Parse tree dump to extract rules
        # This is a simplified version - actual parsing is more complex
        # For diagnostic purposes, we'll extract feature references
        lines = tree_dump.split('\n')
        for line in lines:
            if 'f' in line and ('<' in line or '>' in line):
                # Extract feature index
                try:
                    # Simple extraction - actual parsing needed for full accuracy
                    parts = line.split()
                    for part in parts:
                        if part.startswith('f') and part[1:].isdigit():
                            feat_idx = int(part[1:])
                            if feat_idx < len(feature_names):
                                rules.append({
                                    'tree_idx': tree_idx,
                                    'feature_idx': feat_idx,
                                    'feature_name': feature_names[feat_idx],
                                    'rule_text': line.strip()
                                })
                except:
                    pass
    
    return rules


def analyze_rule_composition(cohort: str, age_band: str):
    """Analyze rule composition for a cohort."""
    print(f"\n{'='*80}")
    print(f"Rule Composition Analysis: {cohort.upper()} / {age_band}")
    print(f"{'='*80}\n")
    
    # Load SHAP importance
    print("Loading SHAP importance...")
    shap_map = load_shap_importance(cohort, age_band)
    if not shap_map:
        print(f"[ERROR] Could not load SHAP importance for {cohort}/{age_band}")
        return
    
    features_with_shap = {f: score for f, score in shap_map.items() if score > 0}
    print(f"  Found {len(features_with_shap)} features with SHAP > 0")
    print(f"  Total features in SHAP: {len(shap_map)}")
    
    # Load model JSON
    print("\nLoading model JSON...")
    model_json = load_model_json(cohort, age_band)
    if not model_json:
        print(f"[ERROR] Could not load model JSON for {cohort}/{age_band}")
        return
    
    feature_names = model_json.get('feature_names', [])
    print(f"  Found {len(feature_names)} features in model")
    
    # Extract rules (simplified - would need full tree parsing for complete analysis)
    print("\nExtracting rules from model...")
    rules = extract_rules_from_model(model_json)
    print(f"  Extracted {len(rules)} rule conditions")
    
    # Analyze feature coverage in rules
    print("\nAnalyzing feature coverage in rules...")
    features_in_rules = set()
    features_with_shap_in_rules = set()
    feature_rule_counts = Counter()
    
    for rule in rules:
        feat_name = rule.get('feature_name', '')
        if feat_name:
            features_in_rules.add(feat_name)
            feature_rule_counts[feat_name] += 1
            
            if feat_name in features_with_shap:
                features_with_shap_in_rules.add(feat_name)
    
    print(f"  Features appearing in rules: {len(features_in_rules)}")
    print(f"  Features with SHAP > 0 appearing in rules: {len(features_with_shap_in_rules)}")
    print(f"  Coverage: {len(features_with_shap_in_rules)/len(features_with_shap)*100:.1f}% of SHAP > 0 features")
    
    # Find features with SHAP > 0 that DON'T appear in rules
    missing_features = features_with_shap.keys() - features_with_shap_in_rules
    print(f"\n  Features with SHAP > 0 NOT in rules: {len(missing_features)}")
    if missing_features:
        print("  Top 10 missing features (by SHAP):")
        missing_with_shap = [(f, shap_map[f]) for f in missing_features]
        missing_with_shap.sort(key=lambda x: x[1], reverse=True)
        for feat, shap_score in missing_with_shap[:10]:
            print(f"    {feat}: SHAP={shap_score:.6f}")
    
    # Analyze rule scores
    print("\nAnalyzing rule SHAP scores...")
    rule_scores = []
    for rule in rules:
        feat_name = rule.get('feature_name', '')
        if feat_name:
            score = shap_map.get(feat_name, 0.0)
            rule_scores.append({
                'feature': feat_name,
                'shap_score': score,
                'has_shap': score > 0
            })
    
    if rule_scores:
        df_scores = pd.DataFrame(rule_scores)
        rules_with_shap = len(df_scores[df_scores['has_shap'] == True])
        rules_without_shap = len(df_scores[df_scores['has_shap'] == False])
        
        print(f"  Rules with features having SHAP > 0: {rules_with_shap} ({rules_with_shap/len(rule_scores)*100:.1f}%)")
        print(f"  Rules with features having SHAP = 0: {rules_without_shap} ({rules_without_shap/len(rule_scores)*100:.1f}%)")
        
        if rules_without_shap > 0:
            print(f"\n  [WARNING] {rules_without_shap} rules contain only features with SHAP = 0")
            print(f"  These rules might be noise or artifacts")
    
    # Feature frequency analysis
    print("\nTop 20 features by rule frequency:")
    for feat, count in feature_rule_counts.most_common(20):
        shap_score = shap_map.get(feat, 0.0)
        has_shap = "✓" if shap_score > 0 else "✗"
        print(f"  {has_shap} {feat}: {count} rules, SHAP={shap_score:.6f}")
    
    return {
        'cohort': cohort,
        'age_band': age_band,
        'total_features': len(feature_names),
        'features_with_shap': len(features_with_shap),
        'features_in_rules': len(features_in_rules),
        'features_with_shap_in_rules': len(features_with_shap_in_rules),
        'coverage': len(features_with_shap_in_rules) / len(features_with_shap) if features_with_shap else 0,
        'missing_features': len(missing_features),
        'rules_with_shap': rules_with_shap if rule_scores else 0,
        'rules_without_shap': rules_without_shap if rule_scores else 0,
    }


def analyze_all_cohorts():
    """Analyze rule composition for all cohorts."""
    print(f"\n{'='*80}")
    print("Rule Composition Diagnostic - All Cohorts")
    print(f"{'='*80}\n")
    
    results = []
    
    for cohort, age_bands in COHORTS.items():
        for age_band in age_bands:
            try:
                result = analyze_rule_composition(cohort, age_band)
                if result:
                    results.append(result)
            except Exception as e:
                print(f"[ERROR] Failed to analyze {cohort}/{age_band}: {e}")
                import traceback
                traceback.print_exc()
    
    # Summary
    if results:
        print(f"\n{'='*80}")
        print("Summary Across All Cohorts")
        print(f"{'='*80}\n")
        
        df = pd.DataFrame(results)
        
        print("Coverage Statistics:")
        print(f"  Average coverage: {df['coverage'].mean()*100:.1f}%")
        print(f"  Min coverage: {df['coverage'].min()*100:.1f}%")
        print(f"  Max coverage: {df['coverage'].max()*100:.1f}%")
        
        print(f"\nRule Composition:")
        print(f"  Average rules with SHAP > 0: {df['rules_with_shap'].mean():.0f}")
        print(f"  Average rules without SHAP: {df['rules_without_shap'].mean():.0f}")
        
        print(f"\nMissing Features:")
        print(f"  Average missing: {df['missing_features'].mean():.0f}")
        print(f"  Total missing across cohorts: {df['missing_features'].sum()}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Diagnose rule composition and feature coverage",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze all cohorts
  python utility_scripts/diagnose_rule_composition.py --all-cohorts
  
  # Analyze specific cohort
  python utility_scripts/diagnose_rule_composition.py --cohort opioid_ed --age-band 13-24
        """
    )
    parser.add_argument("--cohort", help="Cohort name (e.g., opioid_ed)")
    parser.add_argument("--age-band", help="Age band (e.g., 13-24)")
    parser.add_argument("--all-cohorts", action="store_true", help="Analyze all cohorts")
    
    args = parser.parse_args()
    
    if args.all_cohorts:
        analyze_all_cohorts()
    elif args.cohort and args.age_band:
        analyze_rule_composition(args.cohort, args.age_band)
    else:
        parser.print_help()
        sys.exit(1)

