#!/usr/bin/env python3
"""
Create JSON file for pre-target predictive features to KEEP.

This script:
1. Loads bupar_post_target_analysis.csv
2. Extracts features with pre-F1120 ratio >= 80% (truly predictive)
3. Organizes them by type (ICD, CPT, Drug) and category
4. Creates a JSON file with features to KEEP (positive list approach)
5. For controls, use the same set of features
"""

import argparse
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Set
import pandas as pd

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


def categorize_predictive_feature(feature_code: str, feature_type: str, pre_count: int) -> str:
    """Categorize a predictive feature into a specific category."""
    
    if feature_type == 'Drug':
        # Pain medications (often precursors to opioid dependence)
        pain_keywords = ['CODEINE', 'HYDROCODONE', 'OXYCODONE', 'TRAMADOL', 'MORPHINE']
        if any(kw in feature_code.upper() for kw in pain_keywords):
            return 'pain_medications'
        
        # Mental health medications (risk factors)
        mental_health_keywords = ['SERTRALINE', 'FLUOXETINE', 'CITALOPRAM', 'ESCITALOPRAM', 
                                 'BUPROPION', 'VENLAFAXINE', 'DULOXETINE', 'TRAZODONE']
        if any(kw in feature_code.upper() for kw in mental_health_keywords):
            return 'mental_health_medications'
        
        # Substance use related
        substance_keywords = ['NICOTINE', 'ALCOHOL', 'TOBACCO']
        if any(kw in feature_code.upper() for kw in substance_keywords):
            return 'substance_use_medications'
        
        return 'other_predictive_medications'
    
    elif feature_type == 'CPT':
        # Emergency/urgent care visits (risk indicators)
        if feature_code.startswith('9928') or feature_code.startswith('9929'):
            return 'emergency_visits'
        
        # Mental health services
        if feature_code.startswith('908'):
            return 'mental_health_services'
        
        # Lab tests (screening, monitoring)
        if feature_code.startswith('8') or feature_code.startswith('364'):
            return 'lab_tests'
        
        # Office visits
        if feature_code.startswith('992'):
            return 'office_visits'
        
        return 'other_predictive_procedures'
    
    elif feature_type == 'ICD':
        # Mental health diagnoses (risk factors)
        if feature_code.startswith('F3') or feature_code.startswith('F4') or feature_code.startswith('F2'):
            return 'mental_health_diagnoses'
        
        # Pain-related diagnoses
        if feature_code.startswith('M') or feature_code.startswith('G89'):
            return 'pain_related_diagnoses'
        
        # Substance use (precursor conditions)
        if feature_code.startswith('F1') and feature_code != 'F1120':
            return 'substance_use_diagnoses'
        
        # Injury/trauma (risk factors)
        if feature_code.startswith('S') or feature_code.startswith('T'):
            return 'injury_trauma_diagnoses'
        
        return 'other_predictive_diagnoses'
    
    return 'uncategorized'


def create_pre_target_predictive_features_json(
    cohort: str, 
    age_band: str, 
    pre_f1120_threshold: float = 0.8,
    min_events: int = 5
):
    """Create JSON file for pre-target predictive features to KEEP."""
    age_band_fname = age_band_to_fname(age_band)
    
    # Load BupaR analysis results
    analysis_path = Path(__file__).parent / "outputs" / cohort / age_band_fname / f"{cohort}_{age_band_fname}_bupar_post_target_analysis.csv"
    
    if not analysis_path.exists():
        print(f"[ERROR] Analysis file not found: {analysis_path}")
        print(f"       Run create_bupar_post_target_analysis.py first")
        return None
    
    print(f"\n{'='*80}")
    print(f"Creating Pre-Target Predictive Features JSON (Features to KEEP)")
    print(f"Cohort: {cohort} / Age Band: {age_band}")
    print(f"Threshold: pre-F1120 ratio >= {pre_f1120_threshold:.0%}")
    print(f"Minimum events: {min_events}")
    print(f"{'='*80}\n")
    
    df = pd.read_csv(analysis_path)
    
    # Filter for predictive features (pre-F1120 ratio >= threshold)
    # Also ensure minimum event count for statistical significance
    predictive = df[
        (df['pre_f1120_ratio'] >= pre_f1120_threshold) &
        (df['total_count'] >= min_events)
    ].copy()
    
    # IMPORTANT: Always include F1120 for target creation
    f1120_feature = 'item_icd_F1120'
    if f1120_feature not in predictive['feature'].values:
        # Add F1120 if it's not already in the predictive list
        f1120_row = df[df['feature'] == f1120_feature]
        if len(f1120_row) > 0:
            predictive = pd.concat([predictive, f1120_row], ignore_index=True)
            print(f"[INFO] Added {f1120_feature} to predictive features (needed for target creation)")
    
    print(f"Total predictive features to KEEP: {len(predictive)}")
    print(f"Total features analyzed: {len(df)}\n")
    
    # Categorize features
    predictive['feature_type'] = predictive['feature'].apply(lambda x: categorize_feature(x)[0])
    predictive['feature_code'] = predictive['feature'].apply(lambda x: categorize_feature(x)[1])
    predictive['category'] = predictive.apply(
        lambda row: categorize_predictive_feature(row['feature_code'], row['feature_type'], row.get('pre_count', 0)),
        axis=1
    )
    
    # Organize by type and category
    organized = {
        'ICD': {},
        'CPT': {},
        'Drug': {}
    }
    
    category_counts = {}
    
    for _, row in predictive.iterrows():
        ftype = row['feature_type']
        category = row['category']
        feature = row['feature']
        code = row['feature_code']
        pre_ratio = row['pre_f1120_ratio']
        post_ratio = row.get('post_f1120_ratio', 0)
        pre_count = int(row.get('pre_count', 0)) if pd.notna(row.get('pre_count', 0)) else 0
        post_count = int(row.get('post_count', 0)) if pd.notna(row.get('post_count', 0)) else 0
        
        if ftype not in organized:
            organized[ftype] = {}
        
        if category not in organized[ftype]:
            organized[ftype][category] = []
            category_counts[category] = 0
        
        organized[ftype][category].append({
            'feature': feature,
            'code': code,
            'pre_f1120_ratio': float(pre_ratio),
            'post_f1120_ratio': float(post_ratio),
            'pre_count': pre_count,
            'post_count': post_count,
            'total_count': pre_count + post_count
        })
        category_counts[category] += 1
    
    # Create JSON structure
    features_json = {
        "description": f"Pre-target predictive features to KEEP for model training. These features have >= {pre_f1120_threshold:.0%} of their events occurring BEFORE the F1120 (opioid dependence) diagnosis, making them truly predictive. This is a POSITIVE LIST approach - only these features should be used for both cases and controls.",
        "version": "1.0",
        "created_date": datetime.now().strftime("%Y-%m-%d"),
        "cohort": cohort,
        "age_band": age_band,
        "pre_f1120_threshold": pre_f1120_threshold,
        "min_events": min_events,
        "total_predictive_features": len(predictive),
        "total_features_analyzed": len(df),
        "approach": "positive_list",
        "usage": {
            "cases": "Use ONLY these features for target cases (patients with F1120)",
            "controls": "Use the SAME features for control patients (same feature set, same count)",
            "rationale": "This ensures both cases and controls are evaluated on the same predictive features, preventing bias from different feature sets"
        },
        "why_these_are_predictive": {
            "summary": "These features represent events that occur BEFORE the target diagnosis (F1120), making them available at prediction time. They are true predictors, not consequences of the diagnosis.",
            "categories": {
                "pain_medications": "Pain medications that may precede opioid dependence",
                "mental_health_medications": "Mental health medications indicating risk factors",
                "substance_use_medications": "Substance use related medications",
                "mental_health_diagnoses": "Mental health conditions that are risk factors",
                "pain_related_diagnoses": "Pain conditions that may lead to opioid use",
                "substance_use_diagnoses": "Substance use conditions (precursors to opioid dependence)",
                "injury_trauma_diagnoses": "Injuries/trauma that may lead to pain management",
                "emergency_visits": "Emergency visits indicating health crises",
                "mental_health_services": "Mental health services indicating risk factors",
                "lab_tests": "Lab tests for screening/monitoring",
                "office_visits": "Office visits indicating healthcare utilization",
                "other_predictive_medications": "Other medications that appear before F1120",
                "other_predictive_procedures": "Other procedures that appear before F1120",
                "other_predictive_diagnoses": "Other diagnoses that appear before F1120"
            }
        },
        "predictive_features_to_keep": {
            "ICD": {},
            "CPT": {},
            "Drug": {}
        },
        "summary_by_category": {}
    }
    
    # Add organized features
    for ftype in ['ICD', 'CPT', 'Drug']:
        if ftype in organized:
            for category, features in organized[ftype].items():
                # Sort by pre_count descending
                features_sorted = sorted(features, key=lambda x: x['pre_count'], reverse=True)
                features_json["predictive_features_to_keep"][ftype][category] = features_sorted
    
    # Add summary by category
    for category, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
        features_json["summary_by_category"][category] = {
            "count": count,
            "description": features_json["why_these_are_predictive"]["categories"].get(category, "No description available")
        }
    
    # Add flat list for easy filtering
    features_json["all_features_to_keep"] = sorted(predictive['feature'].tolist())
    
    # Add by type for easy access
    features_json["features_by_type"] = {
        "ICD": sorted(predictive[predictive['feature_type'] == 'ICD']['feature'].tolist()),
        "CPT": sorted(predictive[predictive['feature_type'] == 'CPT']['feature'].tolist()),
        "Drug": sorted(predictive[predictive['feature_type'] == 'Drug']['feature'].tolist())
    }
    
    # Add note about F1120
    features_json["special_features"] = {
        "item_icd_F1120": {
            "reason": "F1120 is included for target creation in step 4a. It should be used to identify target cases but may be excluded from final model features depending on the modeling approach.",
            "note": "F1120 appears in both pre- and post-diagnosis contexts, but is essential for identifying target cases."
        }
    }
    
    # Save JSON file
    output_dir = Path(__file__).parent / "outputs" / cohort / age_band_fname
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / f"{cohort}_{age_band_fname}_pre_target_predictive_features.json"
    
    with open(output_path, 'w') as f:
        json.dump(features_json, f, indent=2)
    
    print(f"[OK] Saved predictive features JSON to: {output_path}")
    print(f"\nSummary:")
    print(f"  Total predictive features to KEEP: {len(predictive)}")
    print(f"  By type:")
    for ftype in ['ICD', 'CPT', 'Drug']:
        count = len(features_json["features_by_type"][ftype])
        if count > 0:
            print(f"    {ftype}: {count}")
    
    print(f"\n  By category (top 10):")
    for category, info in list(features_json["summary_by_category"].items())[:10]:
        print(f"    {category}: {info['count']} features")
    
    print(f"\n  Top predictive features by pre-count:")
    top_features = predictive.nlargest(10, 'pre_count')[['feature', 'pre_count', 'pre_f1120_ratio']]
    for _, row in top_features.iterrows():
        print(f"    {row['feature']}: {int(row['pre_count'])} pre-F1120 events ({row['pre_f1120_ratio']*100:.1f}% pre)")
    
    print(f"\n[INFO] This is a POSITIVE LIST approach:")
    print(f"  - Use ONLY these {len(predictive)} features for cases")
    print(f"  - Use the SAME {len(predictive)} features for controls")
    print(f"  - This ensures both groups are evaluated on the same feature set")
    
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create JSON file for pre-target predictive features to KEEP")
    parser.add_argument("--cohort", default="opioid_ed", help="Cohort name")
    parser.add_argument("--age-band", default="13-24", help="Age band")
    parser.add_argument(
        "--pre-f1120-threshold",
        type=float,
        default=0.8,
        help="Threshold for pre-F1120 ratio to flag as predictive (0.0-1.0, default: 0.8 = 80%%)"
    )
    parser.add_argument(
        "--min-events",
        type=int,
        default=5,
        help="Minimum number of events required (default: 5)"
    )
    
    args = parser.parse_args()
    create_pre_target_predictive_features_json(
        args.cohort, 
        args.age_band, 
        args.pre_f1120_threshold,
        args.min_events
    )
