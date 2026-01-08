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
    
    # Try S3 first (check both possible path formats)
    s3_prefixes = [
        f"gold/final_model/{cohort}/{age_band_fname}/",
        f"gold/final_model/{cohort}/{age_band}/",  # Age band with dashes
    ]
    
    for s3_prefix in s3_prefixes:
        try:
            response = s3_client.list_objects_v2(
                Bucket=OUTPUT_BUCKET,
                Prefix=s3_prefix,
                MaxKeys=100
            )
            if 'Contents' in response:
                for obj in response['Contents']:
                    if obj['Key'].endswith('_best_xgboost_model.json'):
                        print(f"  Found model JSON in S3: s3://{OUTPUT_BUCKET}/{obj['Key']}")
                        import tempfile
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.json', mode='w') as tmp_file:
                            tmp_path = tmp_file.name
                        s3_client.download_file(OUTPUT_BUCKET, obj['Key'], tmp_path)
                        with open(tmp_path, 'r') as f:
                            result = json.load(f)
                        Path(tmp_path).unlink()
                        return result
        except Exception as e:
            print(f"  Error checking S3 prefix {s3_prefix}: {e}")
            continue
    
    # Try local paths (multiple possible locations)
    local_paths = [
        PROJECT_ROOT / "6_final_model" / "outputs" / cohort / age_band_fname / "final_model_json" / f"{cohort}_{age_band_fname}_best_xgboost_model.json",
        PROJECT_ROOT / "6_final_model" / "model_outputs" / cohort / age_band_fname / f"{cohort}_{age_band_fname}_best_xgboost_model.json",
    ]
    
    for local_path in local_paths:
        if local_path.exists():
            try:
                print(f"  Found model JSON locally: {local_path}")
                with open(local_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"  Error reading {local_path}: {e}")
                continue
    
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
    """Analyze rule composition for a cohort using actual explainer initialization."""
    print(f"\n{'='*80}")
    print(f"Rule Composition Analysis: {cohort.upper()} / {age_band}")
    print(f"{'='*80}\n")
    
    # Load SHAP importance
    print("Loading SHAP importance...")
    shap_map = load_shap_importance(cohort, age_band)
    if not shap_map:
        print(f"[ERROR] Could not load SHAP importance for {cohort}/{age_band}")
        return None
    
    features_with_shap = {f: score for f, score in shap_map.items() if score > 0}
    print(f"  Found {len(features_with_shap)} features with SHAP > 0")
    print(f"  Total features in SHAP: {len(shap_map)}")
    
    # Try to initialize explainer (like FFA does)
    print("\nInitializing explainer to extract rules...")
    try:
        from pathlib import Path as PathLib
        import sys as sys_module
        
        # Add FFA analysis to path
        ffa_path = PROJECT_ROOT / "8_ffa_analysis"
        if str(ffa_path) not in sys_module.path:
            sys_module.path.insert(0, str(ffa_path))
        
        from run_full_ffa_analysis import (
            load_model_json as ffa_load_model_json,
            extract_feature_mappings,
            load_shap_importance as ffa_load_shap,
            initialize_explainer,
            load_data
        )
        
        age_band_fname = age_band.replace("-", "_")
        
        # Load model JSON (try S3 first, then local)
        model_json = load_model_json(cohort, age_band)
        if not model_json:
            print(f"  [WARNING] Model JSON not found locally or in S3")
            print(f"  Skipping rule extraction (would need model JSON to analyze)")
            return {
                'cohort': cohort,
                'age_band': age_band,
                'features_with_shap': len(features_with_shap),
                'status': 'model_json_not_found'
            }
        
        # Create a temporary path for the explainer (it expects a Path object)
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as tmp_file:
            json.dump(model_json, tmp_file)
            model_json_path = Path(tmp_file.name)
        feature_mappings = extract_feature_mappings(model_json)
        
        # Load SHAP (using FFA function) - includes both importance map and values DataFrame
        shap_map_ffa, shap_values_df = ffa_load_shap(cohort, age_band, "xgboost")
        
        if shap_values_df is None or len(shap_values_df) == 0:
            print(f"  [WARNING] SHAP values DataFrame not available, loading sample...")
            # Try to load a small sample for diagnostic purposes
            try:
                from py_helpers.shap_parquet_loader import load_shap_parquet
                age_band_fname = age_band.replace("-", "_")
                s3_shap_path = f"gold/shap_analysis/{cohort}/{age_band_fname}/shap_sample_values_xgboost.parquet"
                shap_values_df = load_shap_parquet(OUTPUT_BUCKET, s3_shap_path, max_rows=1000)
                print(f"  Loaded {len(shap_values_df)} SHAP value rows for diagnostic")
            except Exception as e:
                print(f"  [ERROR] Could not load SHAP values: {e}")
                return None
        
        # Initialize explainer
        print("  Initializing explainer (this may take a moment)...")
        explainer = initialize_explainer(
            model_json_path=model_json_path,
            model_json=model_json,
            feature_mappings=feature_mappings,
            feature_names=None,
            shap_importance_map=shap_map_ffa,
            shap_values_df=shap_values_df
        )
        
        if explainer is None:
            print(f"  [ERROR] Could not initialize explainer")
            return None
        
        # Analyze rules from explainer
        print(f"\nAnalyzing {len(explainer.rule_clauses)} rules from explainer...")
        
        # Extract features from all rules
        features_in_rules = set()
        features_with_shap_in_rules = set()
        feature_rule_counts = Counter()
        rules_with_shap_features = 0
        rules_without_shap_features = 0
        
        for rule_idx, clause in enumerate(explainer.rule_clauses):
            rule_features = set()
            has_shap_feature = False
            
            for lit in clause:
                feat_idx, _, _ = explainer.id_condition_map[lit]
                feat_name = explainer.feature_names.get(feat_idx, f"f{feat_idx}")
                rule_features.add(feat_name)
                feature_rule_counts[feat_name] += 1
                
                if shap_map_ffa.get(feat_name, 0) > 0:
                    has_shap_feature = True
            
            features_in_rules.update(rule_features)
            features_with_shap_in_rules.update(
                f for f in rule_features if shap_map_ffa.get(f, 0) > 0
            )
            
            if has_shap_feature:
                rules_with_shap_features += 1
            else:
                rules_without_shap_features += 1
        
        print(f"  Features appearing in rules: {len(features_in_rules)}")
        print(f"  Features with SHAP > 0 appearing in rules: {len(features_with_shap_in_rules)}")
        if features_with_shap:
            coverage = len(features_with_shap_in_rules) / len(features_with_shap) * 100
            print(f"  Coverage: {coverage:.1f}% of SHAP > 0 features")
        
        # Find missing features
        missing_features = set(features_with_shap.keys()) - features_with_shap_in_rules
        print(f"\n  Features with SHAP > 0 NOT in rules: {len(missing_features)}")
        if missing_features:
            print("  Top 10 missing features (by SHAP):")
            missing_with_shap = [(f, shap_map_ffa[f]) for f in missing_features]
            missing_with_shap.sort(key=lambda x: x[1], reverse=True)
            for feat, shap_score in missing_with_shap[:10]:
                print(f"    {feat}: SHAP={shap_score:.6f}")
        
        # Rule composition analysis
        print(f"\nRule Composition:")
        print(f"  Rules with SHAP > 0 features: {rules_with_shap_features} ({rules_with_shap_features/len(explainer.rule_clauses)*100:.1f}%)")
        print(f"  Rules WITHOUT SHAP > 0 features: {rules_without_shap_features} ({rules_without_shap_features/len(explainer.rule_clauses)*100:.1f}%)")
        
        if rules_without_shap_features > 0:
            print(f"\n  [WARNING] {rules_without_shap_features} rules contain ONLY features with SHAP = 0")
            print(f"  These rules might be noise or artifacts")
        
        # Feature frequency analysis
        print("\nTop 20 features by rule frequency:")
        for feat, count in feature_rule_counts.most_common(20):
            shap_score = shap_map_ffa.get(feat, 0.0)
            has_shap = "[SHAP>0]" if shap_score > 0 else "[SHAP=0]"
            print(f"  {has_shap} {feat}: {count} rules, SHAP={shap_score:.6f}")
        
        return {
            'cohort': cohort,
            'age_band': age_band,
            'total_rules': len(explainer.rule_clauses),
            'features_with_shap': len(features_with_shap),
            'features_in_rules': len(features_in_rules),
            'features_with_shap_in_rules': len(features_with_shap_in_rules),
            'coverage': len(features_with_shap_in_rules) / len(features_with_shap) if features_with_shap else 0,
            'missing_features': len(missing_features),
            'rules_with_shap': rules_with_shap_features,
            'rules_without_shap': rules_without_shap_features,
        }
        
    except Exception as e:
        print(f"  [ERROR] Could not initialize explainer: {e}")
        import traceback
        traceback.print_exc()
        return None


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

