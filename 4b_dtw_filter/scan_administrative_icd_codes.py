"""
Scan ICD code files to identify potentially administrative/non-medical codes.

ICD codes are diagnosis codes, but some Z codes represent encounters for
administrative purposes, routine examinations, or follow-ups rather than
medical diagnoses.
"""
import re
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).parent.parent
CODE_LOOKUP_DIR = PROJECT_ROOT / "4b_dtw_filter" / "code_lookup"

# Keywords that suggest administrative/non-medical encounters
ADMIN_KEYWORDS = [
    'routine', 'screening', 'check', 'examination', 'encounter for',
    'follow-up', 'followup', 'administrative', 'pre-employment',
    'insurance', 'driving license', 'sport', 'admission to',
    'recruitment', 'disability determination', 'medical certificate',
    'paternity', 'adoption', 'blood-alcohol', 'blood-drug test',
    'without abnormal findings', 'health examination',
    'preventive', 'wellness', 'annual', 'periodic'
]

# Z code categories that are typically administrative
Z_CODE_CATEGORIES = {
    'Z00': 'General health examinations (routine checkups)',
    'Z01': 'Special examinations (eye, dental, etc.)',
    'Z02': 'Administrative examinations (pre-employment, insurance, etc.)',
    'Z03': 'Medical observation for suspected conditions (ruled out)',
    'Z08': 'Follow-up examination after treatment for malignant neoplasm',
    'Z09': 'Follow-up examination after treatment for other conditions',
    'Z39': 'Encounter for maternal postpartum care and examination',
    'Z51': 'Encounters for other aftercare and medical care',
}

def scan_icd_file(file_path: Path) -> dict:
    """Scan an ICD code file for administrative codes."""
    results = {
        'z_codes': defaultdict(list),
        'admin_keywords': [],
        'total_codes': 0,
    }
    
    if not file_path.exists():
        return results
    
    print(f"Scanning {file_path.name}...")
    
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            results['total_codes'] += 1
            
            # Parse code and description
            parts = line.split(None, 1)
            if len(parts) < 2:
                continue
            
            code = parts[0].strip()
            description = parts[1].strip().lower()
            
            # Check for Z codes
            if code.startswith('Z'):
                category = code[:3]
                if category in Z_CODE_CATEGORIES:
                    results['z_codes'][category].append({
                        'code': code,
                        'description': parts[1].strip()
                    })
            
            # Check for administrative keywords
            for keyword in ADMIN_KEYWORDS:
                if keyword.lower() in description:
                    results['admin_keywords'].append({
                        'code': code,
                        'description': parts[1].strip(),
                        'keyword': keyword
                    })
                    break  # Only count once per code
    
    return results


def main():
    """Main function to scan ICD files."""
    print("=" * 80)
    print("Scanning ICD Code Files for Administrative/Non-Medical Codes")
    print("=" * 80)
    print()
    
    # Scan the most recent ICD-10-CM file
    icd_file = CODE_LOOKUP_DIR / "icd10cm_codes_2019.txt"
    
    if not icd_file.exists():
        print(f"ERROR: {icd_file} not found!")
        return
    
    results = scan_icd_file(icd_file)
    
    print(f"\nTotal codes scanned: {results['total_codes']:,}")
    print()
    
    # Report Z codes by category
    print("=" * 80)
    print("Z Codes (Factors influencing health status and contact with health services)")
    print("=" * 80)
    print()
    
    total_z_codes = 0
    for category in sorted(Z_CODE_CATEGORIES.keys()):
        if category in results['z_codes']:
            codes = results['z_codes'][category]
            total_z_codes += len(codes)
            print(f"\n{category} - {Z_CODE_CATEGORIES[category]}: {len(codes)} codes")
            print("-" * 80)
            # Show first 10 examples
            for item in codes[:10]:
                print(f"  {item['code']:10s} {item['description']}")
            if len(codes) > 10:
                print(f"  ... and {len(codes) - 10} more")
    
    print(f"\nTotal Z codes found: {total_z_codes:,}")
    
    # Report codes with administrative keywords
    print("\n" + "=" * 80)
    print("Codes with Administrative Keywords")
    print("=" * 80)
    print()
    
    # Group by keyword
    by_keyword = defaultdict(list)
    for item in results['admin_keywords']:
        by_keyword[item['keyword']].append(item)
    
    print(f"Total codes with administrative keywords: {len(results['admin_keywords']):,}")
    print()
    
    for keyword in sorted(by_keyword.keys()):
        items = by_keyword[keyword]
        print(f"\n'{keyword}': {len(items)} codes")
        print("-" * 80)
        # Show first 5 examples
        for item in items[:5]:
            print(f"  {item['code']:10s} {item['description'][:70]}")
        if len(items) > 5:
            print(f"  ... and {len(items) - 5} more")
    
    # Summary
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print()
    print("ICD codes are DIAGNOSIS codes, not service codes.")
    print("However, Z codes represent 'Factors influencing health status'")
    print("and include many administrative encounters and routine examinations.")
    print()
    print("Key findings:")
    print(f"  - Z codes found: {total_z_codes:,}")
    print(f"  - Codes with administrative keywords: {len(results['admin_keywords']):,}")
    print()
    print("Note: Services like ambulance transport or medical supplies")
    print("would typically be coded with CPT or HCPCS codes, not ICD codes.")
    print("ICD codes describe diagnoses, conditions, or reasons for encounters.")


if __name__ == "__main__":
    main()
