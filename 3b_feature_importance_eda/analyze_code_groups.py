"""Analyze ICD and CPT codes by letter groups to identify administrative vs informative codes"""
import pandas as pd
import json
import re
from pathlib import Path
from collections import defaultdict

def analyze_code_groups(cohort="opioid_ed", age_band="13-24"):
    """Analyze ICD and CPT codes grouped by letter/range"""
    
    # Load feature importance data
    age_band_fname = age_band.replace("-", "_")
    fi_path = Path(f"3_feature_importance/outputs/{cohort}/{age_band}/{cohort}_{age_band_fname}_aggregated_feature_importance.csv")
    
    if not fi_path.exists():
        print(f"File not found: {fi_path}")
        return
    
    df = pd.read_csv(fi_path)
    
    # Load administrative codes lookup
    admin_lookup_path = Path("4b_dtw_filter/administrative_codes_lookup.json")
    admin_codes = {"icd": set(), "cpt": set()}
    if admin_lookup_path.exists():
        with open(admin_lookup_path, 'r') as f:
            admin_lookup = json.load(f)
            admin_codes["icd"] = {code.replace('.', '') for code in admin_lookup.get('administrative_codes', {}).get('icd', [])}
            admin_codes["cpt"] = set(admin_lookup.get('administrative_codes', {}).get('cpt', []))
    
    # Extract ICD codes (format: item_[A-Z][0-9]+)
    icd_pattern = re.compile(r'^item_([A-Z])(\d+)$')
    icd_data = []
    
    for _, row in df.iterrows():
        feature = str(row['feature'])
        match = icd_pattern.match(feature)
        if match:
            letter = match.group(1)
            code_num = match.group(2)
            code_clean = f"{letter}{code_num}"
            icd_data.append({
                'feature': feature,
                'code': code_clean,
                'letter': letter,
                'importance_scaled': row.get('importance_scaled', 0),
                'is_administrative': code_clean in admin_codes["icd"]
            })
    
    # Extract CPT codes (format: item_[0-9]{5})
    cpt_pattern = re.compile(r'^item_(\d{5})$')
    cpt_data = []
    
    for _, row in df.iterrows():
        feature = str(row['feature'])
        match = cpt_pattern.match(feature)
        if match:
            code = match.group(1)
            # CPT codes are grouped by first digit (0-9) and ranges
            first_digit = code[0]
            cpt_range = get_cpt_range(code)
            cpt_data.append({
                'feature': feature,
                'code': code,
                'range': cpt_range,
                'first_digit': first_digit,
                'importance_scaled': row.get('importance_scaled', 0),
                'is_administrative': code in admin_codes["cpt"]
            })
    
    # Analyze ICD codes by letter
    icd_df = pd.DataFrame(icd_data)
    icd_summary = []
    
    for letter in sorted(icd_df['letter'].unique()):
        letter_codes = icd_df[icd_df['letter'] == letter]
        admin_count = letter_codes['is_administrative'].sum()
        total_count = len(letter_codes)
        avg_importance = letter_codes['importance_scaled'].mean()
        max_importance = letter_codes['importance_scaled'].max()
        
        # Get ICD-10 chapter description
        chapter_desc = get_icd_chapter_description(letter)
        
        icd_summary.append({
            'letter': letter,
            'chapter': chapter_desc,
            'total_codes': total_count,
            'administrative_codes': admin_count,
            'informative_codes': total_count - admin_count,
            'avg_importance': avg_importance,
            'max_importance': max_importance,
            'classification': 'Administrative' if admin_count > 0 and admin_count == total_count else 
                            'Mixed' if admin_count > 0 else 'Informative'
        })
    
    # Analyze CPT codes by range
    cpt_df = pd.DataFrame(cpt_data)
    cpt_summary = []
    
    for cpt_range in sorted(cpt_df['range'].unique()):
        range_codes = cpt_df[cpt_df['range'] == cpt_range]
        admin_count = range_codes['is_administrative'].sum()
        total_count = len(range_codes)
        avg_importance = range_codes['importance_scaled'].mean()
        max_importance = range_codes['importance_scaled'].max()
        
        range_desc = get_cpt_range_description(cpt_range)
        
        cpt_summary.append({
            'range': cpt_range,
            'description': range_desc,
            'total_codes': total_count,
            'administrative_codes': admin_count,
            'informative_codes': total_count - admin_count,
            'avg_importance': avg_importance,
            'max_importance': max_importance,
            'classification': 'Administrative' if admin_count > 0 and admin_count == total_count else 
                            'Mixed' if admin_count > 0 else 'Informative'
        })
    
    # Print results
    print("="*80)
    print(f"CODE GROUP ANALYSIS: {cohort} - {age_band}")
    print("="*80)
    
    print("\nICD CODES BY LETTER (ICD-10 Chapters):")
    print("-"*80)
    icd_summary_df = pd.DataFrame(icd_summary)
    for _, row in icd_summary_df.iterrows():
        print(f"\n{row['letter']} - {row['chapter']}")
        print(f"  Total codes: {row['total_codes']}")
        print(f"  Administrative: {row['administrative_codes']}")
        print(f"  Informative: {row['informative_codes']}")
        print(f"  Avg importance: {row['avg_importance']:.6f}")
        print(f"  Max importance: {row['max_importance']:.6f}")
        print(f"  Classification: {row['classification']}")
    
    print("\n\nCPT CODES BY RANGE:")
    print("-"*80)
    cpt_summary_df = pd.DataFrame(cpt_summary)
    for _, row in cpt_summary_df.iterrows():
        print(f"\n{row['range']} - {row['description']}")
        print(f"  Total codes: {row['total_codes']}")
        print(f"  Administrative: {row['administrative_codes']}")
        print(f"  Informative: {row['informative_codes']}")
        print(f"  Avg importance: {row['avg_importance']:.6f}")
        print(f"  Max importance: {row['max_importance']:.6f}")
        print(f"  Classification: {row['classification']}")
    
    # Save to JSON for documentation
    output = {
        'cohort': cohort,
        'age_band': age_band,
        'icd_analysis': icd_summary_df.to_dict('records'),
        'cpt_analysis': cpt_summary_df.to_dict('records')
    }
    
    output_path = Path(f"3b_feature_importance_eda/outputs/{cohort}/{age_band_fname}/code_group_analysis.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n\nSaved analysis to: {output_path}")
    
    return icd_summary_df, cpt_summary_df

def get_icd_chapter_description(letter):
    """Get ICD-10 chapter description for a letter"""
    chapters = {
        'A': 'Certain infectious and parasitic diseases',
        'B': 'Certain infectious and parasitic diseases (continued)',
        'C': 'Neoplasms',
        'D': 'Diseases of blood and immune mechanism',
        'E': 'Endocrine, nutritional and metabolic diseases',
        'F': 'Mental, behavioral and neurodevelopmental disorders',
        'G': 'Diseases of the nervous system',
        'H': 'Diseases of the eye and adnexa',
        'I': 'Diseases of the circulatory system',
        'J': 'Diseases of the respiratory system',
        'K': 'Diseases of the digestive system',
        'L': 'Diseases of the skin and subcutaneous tissue',
        'M': 'Diseases of the musculoskeletal system and connective tissue',
        'N': 'Diseases of the genitourinary system',
        'O': 'Pregnancy, childbirth and the puerperium',
        'P': 'Certain conditions originating in the perinatal period',
        'Q': 'Congenital malformations, deformations and chromosomal abnormalities',
        'R': 'Symptoms, signs and abnormal clinical and laboratory findings',
        'S': 'Injury, poisoning and certain other consequences of external causes',
        'T': 'Injury, poisoning and certain other consequences of external causes (continued)',
        'U': 'Codes for special purposes',
        'V': 'External causes of morbidity',
        'W': 'External causes of morbidity (continued)',
        'X': 'External causes of morbidity (continued)',
        'Y': 'External causes of morbidity (continued)',
        'Z': 'Factors influencing health status and contact with health services'
    }
    return chapters.get(letter, 'Unknown')

def get_cpt_range(code):
    """Get CPT code range category"""
    code_int = int(code)
    if 0 <= code_int <= 999:
        return "00000-00999"
    elif 1000 <= code_int <= 1999:
        return "01000-01999"
    elif 10000 <= code_int <= 19999:
        return "10000-19999"
    elif 20000 <= code_int <= 29999:
        return "20000-29999"
    elif 30000 <= code_int <= 39999:
        return "30000-39999"
    elif 40000 <= code_int <= 49999:
        return "40000-49999"
    elif 50000 <= code_int <= 59999:
        return "50000-59999"
    elif 60000 <= code_int <= 69999:
        return "60000-69999"
    elif 70000 <= code_int <= 79999:
        return "70000-79999"
    elif 80000 <= code_int <= 89999:
        return "80000-89999"
    elif 90000 <= code_int <= 99999:
        return "90000-99999"
    else:
        return "Other"

def get_cpt_range_description(range_str):
    """Get CPT range description"""
    descriptions = {
        "00000-00999": "Anesthesia",
        "01000-01999": "Anesthesia (continued)",
        "10000-19999": "Surgery - Integumentary System",
        "20000-29999": "Surgery - Musculoskeletal System",
        "30000-39999": "Surgery - Respiratory, Cardiovascular, Hemic/Lymphatic",
        "40000-49999": "Surgery - Digestive System",
        "50000-59999": "Surgery - Urinary, Male Genital, Female Genital, Maternity",
        "60000-69999": "Surgery - Endocrine, Nervous System",
        "70000-79999": "Radiology",
        "80000-89999": "Pathology and Laboratory",
        "90000-99999": "Medicine, Evaluation and Management, Miscellaneous"
    }
    return descriptions.get(range_str, "Unknown")

if __name__ == '__main__':
    import sys
    cohort = sys.argv[1] if len(sys.argv) > 1 else "opioid_ed"
    age_band = sys.argv[2] if len(sys.argv) > 2 else "13-24"
    
    analyze_code_groups(cohort, age_band)
