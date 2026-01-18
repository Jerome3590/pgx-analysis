#!/usr/bin/env python3
"""
Analyze post-F1120 leakage features to understand why they represent target leakage.

This script:
1. Loads the bupar_post_target_analysis.csv
2. Categorizes leakage features by type (ICD, CPT, drug)
3. Identifies patterns and common themes
4. Provides examples with explanations
"""

import pandas as pd
from pathlib import Path
import sys
from collections import Counter

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from py_helpers.constants import age_band_to_fname


def categorize_feature(feature: str) -> tuple:
    """Categorize a feature by type and extract the code."""
    if feature.startswith('item_icd_'):
        code = feature.replace('item_icd_', '')
        return ('ICD', code)
    elif feature.startswith('item_cpt_'):
        code = feature.replace('item_cpt_', '')
        return ('CPT', code)
    elif feature.startswith('item_drug_'):
        drug = feature.replace('item_drug_', '')
        return ('Drug', drug)
    else:
        return ('Unknown', feature)


def analyze_leakage_features(cohort: str, age_band: str):
    """Analyze leakage features in detail."""
    age_band_fname = age_band_to_fname(age_band)
    analysis_path = Path(__file__).parent / "outputs" / cohort / age_band_fname / f"{cohort}_{age_band_fname}_bupar_post_target_analysis.csv"
    
    if not analysis_path.exists():
        print(f"[ERROR] Analysis file not found: {analysis_path}")
        return
    
    print(f"\n{'='*80}")
    print(f"Analyzing Post-F1120 Leakage Features")
    print(f"Cohort: {cohort} / Age Band: {age_band}")
    print(f"{'='*80}\n")
    
    df = pd.read_csv(analysis_path)
    
    # Filter leakage features
    leakage = df[df['is_post_target_leakage'] == 1].copy()
    
    print(f"Total leakage features: {len(leakage)}")
    print(f"Total features analyzed: {len(df)}\n")
    
    # Categorize features
    leakage['feature_type'] = leakage['feature'].apply(lambda x: categorize_feature(x)[0])
    leakage['feature_code'] = leakage['feature'].apply(lambda x: categorize_feature(x)[1])
    
    # Summary by type
    print("="*80)
    print("LEAKAGE FEATURES BY TYPE")
    print("="*80)
    type_counts = leakage['feature_type'].value_counts()
    for ftype, count in type_counts.items():
        pct = (count / len(leakage)) * 100
        print(f"  {ftype:10s}: {count:4d} ({pct:5.1f}%)")
    
    # Analyze by type
    for ftype in ['ICD', 'CPT', 'Drug']:
        type_features = leakage[leakage['feature_type'] == ftype].sort_values('post_f1120_ratio', ascending=False)
        if len(type_features) > 0:
            print(f"\n{'='*80}")
            print(f"TOP {ftype} LEAKAGE FEATURES (sorted by post-F1120 ratio)")
            print(f"{'='*80}")
            print(f"{'Feature':<50s} | Post% | Pre% | Pre | Post | Total")
            print("-" * 80)
            
            for idx, row in type_features.head(30).iterrows():
                feature_name = row['feature_code'][:48]  # Truncate if too long
                post_pct = row['post_f1120_ratio'] * 100
                pre_pct = row['pre_f1120_ratio'] * 100
                pre_count = int(row['pre_count']) if pd.notna(row['pre_count']) else 0
                post_count = int(row['post_count']) if pd.notna(row['post_count']) else 0
                total_count = int(row['total_count']) if pd.notna(row['total_count']) else 0
                
                print(f"{feature_name:<50s} | {post_pct:5.1f}% | {pre_pct:5.1f}% | {pre_count:4d} | {post_count:4d} | {total_count:5d}")
            
            if len(type_features) > 30:
                print(f"\n  ... and {len(type_features) - 30} more {ftype} features")
    
    # Look for patterns in ICD codes
    print(f"\n{'='*80}")
    print("ICD CODE PATTERNS (potential categories)")
    print(f"{'='*80}")
    
    icd_features = leakage[leakage['feature_type'] == 'ICD'].copy()
    if len(icd_features) > 0:
        # Extract ICD code prefixes (first letter + first digit)
        icd_features['icd_prefix'] = icd_features['feature_code'].str[:2]
        prefix_counts = icd_features['icd_prefix'].value_counts()
        
        print("\nICD Code Prefixes (category indicators):")
        for prefix, count in prefix_counts.head(15).items():
            print(f"  {prefix}: {count} codes")
        
        # Look for specific patterns
        print("\nNotable ICD patterns:")
        if 'F11' in icd_features['feature_code'].values:
            f11_codes = icd_features[icd_features['feature_code'].str.startswith('F11')]
            print(f"  F11* (Opioid-related): {len(f11_codes)} codes")
            for code in f11_codes['feature_code'].head(10):
                print(f"    - {code}")
        
        if 'T81' in icd_features['feature_code'].values:
            t81_codes = icd_features[icd_features['feature_code'].str.startswith('T81')]
            print(f"\n  T81* (Complications of procedures): {len(t81_codes)} codes")
            for code in t81_codes['feature_code'].head(5):
                print(f"    - {code}")
        
        if 'N90' in icd_features['feature_code'].values:
            n90_codes = icd_features[icd_features['feature_code'].str.startswith('N90')]
            print(f"\n  N90* (Complications): {len(n90_codes)} codes")
            for code in n90_codes['feature_code'].head(5):
                print(f"    - {code}")
    
    # Look for patterns in CPT codes
    print(f"\n{'='*80}")
    print("CPT CODE PATTERNS")
    print(f"{'='*80}")
    
    cpt_features = leakage[leakage['feature_type'] == 'CPT']
    if len(cpt_features) > 0:
        print(f"\nTotal CPT leakage codes: {len(cpt_features)}")
        
        # Check for drug screening codes (80307 was mentioned earlier)
        drug_screening = cpt_features[cpt_features['feature_code'].str.contains('80307|80348|80349|80350|80351|80352|80353|80354|80355|80356|80357|80358|80359|80360|80361|80362|80363|80364|80365|80366|80367|80368|80369|80370|80371|80372|80373|80374|80375|80376|80377|80378|80379|80380|80381|80382|80383|80384|80385|80386|80387|80388|80389|80390|80391|80392|80393|80394|80395|80396|80397|80398|80399', na=False)]
        if len(drug_screening) > 0:
            print(f"\nDrug screening CPT codes (80307-80399): {len(drug_screening)}")
            for idx, row in drug_screening.head(10).iterrows():
                print(f"  {row['feature_code']}: {int(row['post_count'])} post-F1120 events")
        
        # Check for other common post-diagnosis procedures
        print("\nOther notable CPT codes:")
        for idx, row in cpt_features.head(15).iterrows():
            print(f"  {row['feature_code']}: {int(row['post_count'])} post-F1120 events, {row['post_f1120_ratio']*100:.1f}% post")
    
    # Look for patterns in drugs
    print(f"\n{'='*80}")
    print("DRUG PATTERNS")
    print(f"{'='*80}")
    
    drug_features = leakage[leakage['feature_type'] == 'Drug']
    if len(drug_features) > 0:
        print(f"\nTotal drug leakage codes: {len(drug_features)}")
        
        # Look for treatment-related drugs
        treatment_keywords = ['BUPRENORPHINE', 'NALTREXONE', 'METHADONE', 'SUBOXONE', 'VIVITROL', 
                             'NARCAN', 'NALOXONE', 'SUBUTEX', 'ZUB SOLV']
        treatment_drugs = drug_features[drug_features['feature_code'].str.contains('|'.join(treatment_keywords), case=False, na=False)]
        if len(treatment_drugs) > 0:
            print(f"\nOpioid treatment medications: {len(treatment_drugs)}")
            for idx, row in treatment_drugs.iterrows():
                print(f"  {row['feature_code']}: {int(row['post_count'])} post-F1120 events")
        
        # Look for other common post-diagnosis medications
        print("\nOther notable drugs (top 20 by post-count):")
        for idx, row in drug_features.sort_values('post_count', ascending=False).head(20).iterrows():
            drug_name = row['feature_code'][:50]
            print(f"  {drug_name:<50s}: {int(row['post_count']):4d} post-F1120 events")
    
    # Summary insights
    print(f"\n{'='*80}")
    print("SUMMARY INSIGHTS")
    print(f"{'='*80}")
    print("\nWhy these features represent target leakage:")
    print("  1. They occur AFTER the F1120 (opioid dependence) diagnosis")
    print("  2. They represent:")
    print("     - Treatment interventions (medications, procedures)")
    print("     - Monitoring/screening (drug tests, follow-ups)")
    print("     - Complications (adverse events, infections)")
    print("     - Care coordination (referrals, case management)")
    print("  3. These are CONSEQUENCES of the diagnosis, not predictors")
    print("  4. Using them for prediction would be cheating - they contain")
    print("     information that wouldn't be available at prediction time")
    print(f"\nRecommendation: Filter out all {len(leakage)} leakage features")
    print("                 before model training to prevent target leakage.")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze post-F1120 leakage features")
    parser.add_argument("--cohort", default="opioid_ed", help="Cohort name")
    parser.add_argument("--age-band", default="13-24", help="Age band")
    
    args = parser.parse_args()
    analyze_leakage_features(args.cohort, args.age_band)
