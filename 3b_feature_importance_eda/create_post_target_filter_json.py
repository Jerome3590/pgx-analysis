#!/usr/bin/env python3
"""
Create JSON filter file for post-target leakage features.

This script:
1. Loads bupar_post_target_analysis.csv
2. Extracts features with post-F1120 ratio >= 80%
3. Organizes them by type (ICD, CPT, Drug) and category
4. Creates a JSON filter file with clear organization showing these are post-target features
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


def categorize_leakage_feature(feature_code: str, feature_type: str, post_count: int) -> str:
    """Categorize a leakage feature into a specific category."""
    
    if feature_type == 'Drug':
        # Treatment medications
        treatment_keywords = ['BUPRENORPHINE', 'SUBOXONE', 'ZUBSOLV', 'VIVITROL', 'NARCAN', 
                             'NALTREXONE', 'METHADONE', 'SUBUTEX']
        if any(kw in feature_code.upper() for kw in treatment_keywords):
            return 'treatment_medications'
        
        # Withdrawal/treatment support
        support_keywords = ['BUPROPION', 'CLONIDINE', 'BACLOFEN', 'NALOXONE']
        if any(kw in feature_code.upper() for kw in support_keywords):
            return 'treatment_support_medications'
        
        return 'other_post_diagnosis_medications'
    
    elif feature_type == 'CPT':
        # Drug screening
        if feature_code.startswith('803') or feature_code in ['80307', '80348', '80362']:
            return 'drug_screening_tests'
        
        # Substance abuse services (H codes)
        if feature_code.startswith('H'):
            return 'substance_abuse_services'
        
        # Therapy/mental health services
        if feature_code.startswith('908') or feature_code in ['90853', '90868']:
            return 'therapy_services'
        
        # Office visits (follow-up)
        if feature_code.startswith('992') or feature_code in ['99213', '99214', '99335', '99334']:
            return 'follow_up_visits'
        
        # Other services
        if feature_code.startswith('G') or feature_code.startswith('S') or feature_code.startswith('J'):
            return 'other_healthcare_services'
        
        return 'other_procedures'
    
    elif feature_type == 'ICD':
        # Target diagnosis itself
        if feature_code == 'F1120' or feature_code.startswith('F111'):
            return 'target_diagnosis_recurrence'
        
        # Mental health comorbidities
        if feature_code.startswith('F3') or feature_code.startswith('F2') or feature_code.startswith('F4'):
            return 'mental_health_comorbidities'
        
        # Complications
        if feature_code.startswith('T') or feature_code.startswith('N90') or feature_code.startswith('S8'):
            return 'complications_adverse_events'
        
        # Aftercare/administrative
        if feature_code.startswith('Z'):
            return 'aftercare_administrative_codes'
        
        # Other diagnoses
        return 'other_post_diagnosis_conditions'
    
    return 'uncategorized'


def create_post_target_filter_json(cohort: str, age_band: str, post_f1120_threshold: float = 0.8):
    """Create JSON filter file for post-target leakage features."""
    age_band_fname = age_band_to_fname(age_band)
    
    # Load BupaR analysis results
    analysis_path = Path(__file__).parent / "outputs" / cohort / age_band_fname / f"{cohort}_{age_band_fname}_bupar_post_target_analysis.csv"
    
    if not analysis_path.exists():
        print(f"[ERROR] Analysis file not found: {analysis_path}")
        print(f"       Run create_bupar_post_target_analysis.py first")
        return None
    
    print(f"\n{'='*80}")
    print(f"Creating Post-Target Leakage Filter JSON")
    print(f"Cohort: {cohort} / Age Band: {age_band}")
    print(f"Threshold: post-F1120 ratio >= {post_f1120_threshold:.0%}")
    print(f"{'='*80}\n")
    
    df = pd.read_csv(analysis_path)
    
    # Filter leakage features
    leakage = df[df['post_f1120_ratio'] >= post_f1120_threshold].copy()
    
    # IMPORTANT: Exclude F1120 itself from leakage filter
    # F1120 is needed to create the target column in step 4a
    # We filter out other post-F1120 leakage features, but keep F1120
    f1120_feature = 'item_icd_F1120'
    f1120_in_leakage = f1120_feature in leakage['feature'].values
    
    if f1120_in_leakage:
        leakage = leakage[leakage['feature'] != f1120_feature].copy()
        print(f"[INFO] Excluding {f1120_feature} from leakage filter (needed for target creation)")
    
    print(f"Total leakage features (excluding F1120): {len(leakage)}")
    print(f"Total features analyzed: {len(df)}\n")
    
    # Categorize features
    leakage['feature_type'] = leakage['feature'].apply(lambda x: categorize_feature(x)[0])
    leakage['feature_code'] = leakage['feature'].apply(lambda x: categorize_feature(x)[1])
    leakage['category'] = leakage.apply(
        lambda row: categorize_leakage_feature(row['feature_code'], row['feature_type'], row.get('post_count', 0)),
        axis=1
    )
    
    # Organize by type and category
    organized = {
        'ICD': {},
        'CPT': {},
        'Drug': {}
    }
    
    category_counts = {}
    
    for _, row in leakage.iterrows():
        ftype = row['feature_type']
        category = row['category']
        feature = row['feature']
        code = row['feature_code']
        post_ratio = row['post_f1120_ratio']
        post_count = int(row.get('post_count', 0)) if pd.notna(row.get('post_count', 0)) else 0
        pre_count = int(row.get('pre_count', 0)) if pd.notna(row.get('pre_count', 0)) else 0
        
        if ftype not in organized:
            organized[ftype] = {}
        
        if category not in organized[ftype]:
            organized[ftype][category] = []
            category_counts[category] = 0
        
        organized[ftype][category].append({
            'feature': feature,
            'code': code,
            'post_f1120_ratio': float(post_ratio),
            'post_count': post_count,
            'pre_count': pre_count,
            'total_count': post_count + pre_count
        })
        category_counts[category] += 1
    
    # Create JSON structure
    filter_json = {
        "description": "Post-target leakage features identified from BupaR analysis. These features have >= 80% of their events occurring AFTER the F1120 (opioid dependence) diagnosis, representing target leakage. NOTE: item_icd_F1120 is EXCLUDED from this filter as it is needed to create the target column in step 4a.",
        "version": "1.0",
        "created_date": datetime.now().strftime("%Y-%m-%d"),
        "cohort": cohort,
        "age_band": age_band,
        "post_f1120_threshold": post_f1120_threshold,
        "total_leakage_features": len(leakage),
        "total_features_analyzed": len(df),
        "why_these_are_leakage": {
            "summary": "These features represent events that occur AFTER the target diagnosis (F1120), which would not be available at prediction time. They are consequences of the diagnosis, not predictors.",
            "categories": {
                "treatment_medications": "Medications prescribed after diagnosis as part of treatment (e.g., SUBOXONE, BUPRENORPHINE)",
                "treatment_support_medications": "Medications used to support treatment (e.g., BUPROPION for depression, CLONIDINE for withdrawal)",
                "drug_screening_tests": "Drug screening tests ordered to monitor treatment compliance (e.g., CPT 80307)",
                "substance_abuse_services": "Healthcare services provided after diagnosis (e.g., H0020 - Alcohol and/or drug services)",
                "therapy_services": "Mental health therapy services provided after diagnosis",
                "follow_up_visits": "Follow-up office visits after diagnosis",
                "target_diagnosis_recurrence": "The target diagnosis itself appearing again after initial diagnosis",
                "mental_health_comorbidities": "Mental health diagnoses that may be comorbidities or complications",
                "complications_adverse_events": "Complications, injuries, or adverse events occurring during/after treatment",
                "aftercare_administrative_codes": "Administrative codes marking aftercare and follow-up activities",
                "other_post_diagnosis_medications": "Other medications prescribed after diagnosis",
                "other_healthcare_services": "Other healthcare services provided after diagnosis",
                "other_procedures": "Other procedures performed after diagnosis",
                "other_post_diagnosis_conditions": "Other conditions diagnosed after the target diagnosis"
            }
        },
        "post_target_leakage_features": {
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
                # Sort by post_count descending
                features_sorted = sorted(features, key=lambda x: x['post_count'], reverse=True)
                filter_json["post_target_leakage_features"][ftype][category] = features_sorted
    
    # Add summary by category
    for category, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
        filter_json["summary_by_category"][category] = {
            "count": count,
            "description": filter_json["why_these_are_leakage"]["categories"].get(category, "No description available")
        }
    
    # Add flat list for easy filtering
    filter_json["all_features_to_filter"] = sorted(leakage['feature'].tolist())
    
    # Add note about F1120 exclusion
    filter_json["excluded_features"] = {
        "item_icd_F1120": {
            "reason": "F1120 is needed to create the target column in step 4a model data preparation. It should NOT be filtered out as a leakage feature.",
            "note": "F1120 may appear in both pre- and post-diagnosis contexts, but it is essential for identifying target cases."
        }
    }
    
    # Add by type for easy access
    filter_json["features_by_type"] = {
        "ICD": sorted(leakage[leakage['feature_type'] == 'ICD']['feature'].tolist()),
        "CPT": sorted(leakage[leakage['feature_type'] == 'CPT']['feature'].tolist()),
        "Drug": sorted(leakage[leakage['feature_type'] == 'Drug']['feature'].tolist())
    }
    
    # Save JSON file
    output_dir = Path(__file__).parent / "outputs" / cohort / age_band_fname
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / f"{cohort}_{age_band_fname}_post_target_leakage_filter.json"
    
    with open(output_path, 'w') as f:
        json.dump(filter_json, f, indent=2)
    
    print(f"[OK] Saved post-target leakage filter to: {output_path}")
    print(f"\nSummary:")
    print(f"  Total leakage features: {len(leakage)}")
    print(f"  By type:")
    for ftype in ['ICD', 'CPT', 'Drug']:
        count = len(filter_json["features_by_type"][ftype])
        if count > 0:
            print(f"    {ftype}: {count}")
    
    print(f"\n  By category (top 10):")
    for category, info in list(filter_json["summary_by_category"].items())[:10]:
        print(f"    {category}: {info['count']} features")
    
    print(f"\n  Top leakage features by post-count:")
    top_features = leakage.nlargest(10, 'post_count')[['feature', 'post_count', 'post_f1120_ratio']]
    for _, row in top_features.iterrows():
        print(f"    {row['feature']}: {int(row['post_count'])} post-F1120 events ({row['post_f1120_ratio']*100:.1f}% post)")
    
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create JSON filter file for post-target leakage features")
    parser.add_argument("--cohort", default="opioid_ed", help="Cohort name")
    parser.add_argument("--age-band", default="13-24", help="Age band")
    parser.add_argument(
        "--post-f1120-threshold",
        type=float,
        default=0.8,
        help="Threshold for post-F1120 ratio to flag as leakage (0.0-1.0, default: 0.8 = 80%%)"
    )
    
    args = parser.parse_args()
    create_post_target_filter_json(args.cohort, args.age_band, args.post_f1120_threshold)
