#!/usr/bin/env python3
"""
Research ICD and CPT codes from aggregated feature importances.

This script:
1. Extracts all unique ICD and CPT codes from aggregated feature importance CSVs
2. Looks up code meanings using medical coding references
3. Creates a research document with code classifications (administrative vs. clinical)
"""

import sys
import re
import json
import time
from pathlib import Path
from typing import Set, Dict, List, Optional, Tuple
import pandas as pd

try:
    import requests
    from bs4 import BeautifulSoup
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    print("Warning: requests/beautifulsoup4 not available. Install with: pip install requests beautifulsoup4")

PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from py_helpers.constants import COHORT_NAMES, AGE_BANDS

OUTPUT_DIR = PROJECT_ROOT / "4b_dtw_filter" / "outputs" / "code_research"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CODE_LOOKUP_DIR = PROJECT_ROOT / "4b_dtw_filter" / "code_lookup"

# Cache for ICD lookup files
_icd10_lookup_cache = None
_icd9_to_icd10_cache = None
_icd10_to_icd9_cache = None


def load_icd10_lookup() -> Dict[str, str]:
    """
    Load ICD-10-CM codes and descriptions from local files.
    
    Tries files in order: 2019, 2018, 2017
    
    Returns:
        Dictionary mapping ICD-10 code (without dots) to description
    """
    global _icd10_lookup_cache
    
    if _icd10_lookup_cache is not None:
        return _icd10_lookup_cache
    
    _icd10_lookup_cache = {}
    
    # Try files in order of recency
    for year in [2019, 2018, 2017]:
        file_path = CODE_LOOKUP_DIR / f"icd10cm_codes_{year}.txt"
        if file_path.exists():
            print(f"Loading ICD-10-CM codes from {file_path.name}...")
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        # Format: CODE    Description (tab or multiple spaces)
                        parts = line.split(None, 1)  # Split on whitespace, max 2 parts
                        if len(parts) >= 2:
                            code = parts[0].strip()
                            description = parts[1].strip()
                            # Store code without dots for easier lookup
                            code_no_dot = code.replace('.', '')
                            if code_no_dot not in _icd10_lookup_cache:
                                _icd10_lookup_cache[code_no_dot] = description
                print(f"  Loaded {len(_icd10_lookup_cache)} ICD-10-CM codes")
                break
            except Exception as e:
                print(f"  Warning: Could not load {file_path.name}: {e}")
                continue
    
    return _icd10_lookup_cache


def load_icd9_to_icd10_mapping() -> Dict[str, str]:
    """
    Load ICD-9 to ICD-10 mapping from I9gem file.
    
    Returns:
        Dictionary mapping ICD-9 code to ICD-10 code
    """
    global _icd9_to_icd10_cache
    
    if _icd9_to_icd10_cache is not None:
        return _icd9_to_icd10_cache
    
    _icd9_to_icd10_cache = {}
    
    file_path = CODE_LOOKUP_DIR / "2016_I9gem.txt"
    if file_path.exists():
        print(f"Loading ICD-9 to ICD-10 mapping from {file_path.name}...")
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    # Format: ICD9_CODE  ICD10_CODE  FLAGS
                    parts = line.split()
                    if len(parts) >= 2:
                        icd9 = parts[0].strip()
                        icd10 = parts[1].strip()
                        # Store without dots for easier lookup
                        icd9_no_dot = icd9.replace('.', '')
                        icd10_no_dot = icd10.replace('.', '')
                        if icd9_no_dot not in _icd9_to_icd10_cache:
                            _icd9_to_icd10_cache[icd9_no_dot] = icd10_no_dot
            print(f"  Loaded {len(_icd9_to_icd10_cache)} ICD-9 to ICD-10 mappings")
        except Exception as e:
            print(f"  Warning: Could not load {file_path.name}: {e}")
    
    return _icd9_to_icd10_cache


def lookup_icd_code_local(code: str) -> Optional[Tuple[str, str]]:
    """
    Look up ICD code description from local files.
    
    Handles both ICD-9 and ICD-10 codes.
    For ICD-9 codes, converts to ICD-10 first if mapping available.
    
    Returns:
        Tuple of (description, lookup_source) or None if not found
        lookup_source indicates where the description came from
    """
    code_clean = code.strip().replace('.', '')
    
    # Check if it's ICD-10
    if re.match(r'^[A-Z]\d{2,}', code_clean):
        # ICD-10 code
        icd10_lookup = load_icd10_lookup()
        if code_clean in icd10_lookup:
            return icd10_lookup[code_clean], 'local_icd10_file'
        
        # Try with dot format (e.g., F11.20)
        if len(code_clean) >= 5:
            # Try inserting dot: F1120 -> F11.20
            code_with_dot = f"{code_clean[0]}{code_clean[1:3]}.{code_clean[3:]}"
            code_with_dot_no_dot = code_with_dot.replace('.', '')
            if code_with_dot_no_dot in icd10_lookup:
                return icd10_lookup[code_with_dot_no_dot], 'local_icd10_file'
    
    # Check if it's ICD-9
    elif re.match(r'^\d{3}', code_clean):
        # ICD-9 code - try to convert to ICD-10
        icd9_to_icd10 = load_icd9_to_icd10_mapping()
        if code_clean in icd9_to_icd10:
            icd10_code = icd9_to_icd10[code_clean]
            icd10_lookup = load_icd10_lookup()
            if icd10_code in icd10_lookup:
                return icd10_lookup[icd10_code], 'local_icd10_file_via_icd9_mapping'
    
    return None


def is_icd_code(code: str) -> bool:
    """
    Determine if a code is an ICD code.
    
    ICD-10 codes:
    - Start with a letter (A-Z) followed by digits
    - Format: Letter + 2-3 digits + optional decimal + optional digits
    - Examples: F1120, F909, R509, Z00129, J069, J029
    
    ICD-9 codes (legacy):
    - 3 digits, may have decimal
    - Examples: 250.00, 401.9
    """
    code = code.strip()
    
    # ICD-10: Letter followed by digits
    if re.match(r'^[A-Z]\d{2,}', code):
        return True
    
    # ICD-9: 3 digits (may have decimal)
    if re.match(r'^\d{3}(\.\d+)?$', code):
        return True
    
    return False


def is_cpt_code(code: str) -> bool:
    """
    Determine if a code is a CPT code.
    
    CPT codes:
    - Typically 5 digits (numeric)
    - May have modifiers (letters/numbers after)
    - Examples: 99284, 80305, 87880, 90791, 99213
    """
    code = code.strip()
    
    # CPT: 5 digits (numeric)
    if re.match(r'^\d{5}$', code):
        return True
    
    # CPT with modifiers (e.g., 99213-25)
    if re.match(r'^\d{5}[-A-Z0-9]+$', code):
        return True
    
    return False


def is_hcpcs_code(code: str) -> bool:
    """
    Determine if a code is an HCPCS code.
    
    HCPCS Level II codes:
    - Start with a letter followed by 4 digits
    - Examples: H0004, H0005, H0020, G0483, G0480, S0109
    """
    code = code.strip()
    
    # HCPCS: Letter + 4 digits
    if re.match(r'^[A-Z]\d{4}$', code):
        return True
    
    return False


def extract_codes_from_feature_importance_files() -> Dict[str, Set[str]]:
    """
    Extract all unique ICD, CPT, and HCPCS codes from aggregated feature importance files.
    
    Returns:
        Dictionary with keys 'icd', 'cpt', 'hcpcs' containing sets of codes
    """
    codes = {
        'icd': set(),
        'cpt': set(),
        'hcpcs': set(),
    }
    
    fi_output_dir = PROJECT_ROOT / "3_feature_importance" / "outputs"
    
    total_files = sum(
        1 for cohort_name in COHORT_NAMES
        for age_band in AGE_BANDS
        if (fi_output_dir / cohort_name / age_band / 
            f"{cohort_name}_{age_band.replace('-', '_')}_aggregated_feature_importance.csv").exists()
    )
    
    processed = 0
    
    for cohort_name in COHORT_NAMES:
        for age_band in AGE_BANDS:
            age_band_fname = age_band.replace("-", "_")
            csv_path = (
                fi_output_dir / cohort_name / age_band /
                f"{cohort_name}_{age_band_fname}_aggregated_feature_importance.csv"
            )
            
            if not csv_path.exists():
                continue
            
            processed += 1
            print(f"  [{processed}/{total_files}] Processing {cohort_name}/{age_band}...", end='\r')
            
            try:
                df = pd.read_csv(csv_path)
                
                codes_before = {
                    'icd': len(codes['icd']),
                    'cpt': len(codes['cpt']),
                    'hcpcs': len(codes['hcpcs']),
                }
                
                # Check for 'feature' or 'feature_name' column
                feature_col = None
                if 'feature' in df.columns:
                    feature_col = 'feature'
                elif 'feature_name' in df.columns:
                    feature_col = 'feature_name'
                else:
                    print(f"\n    [WARN] No 'feature' or 'feature_name' column in {csv_path.name}")
                    continue
                
                # Extract codes (remove 'item_' prefix)
                features = df[feature_col].astype(str).str.replace('^item_', '', regex=True)
                
                for feature in features:
                    feature = feature.strip()
                    
                    # Skip drug names (uppercase, may contain spaces)
                    if feature.isupper() and ' ' in feature:
                        continue
                    
                    # Skip if it's clearly a drug name pattern
                    if re.match(r'^[A-Z]+(?:\s+[A-Z/]+)*$', feature) and len(feature) > 8:
                        continue
                    
                    # Check code types
                    if is_icd_code(feature):
                        codes['icd'].add(feature)
                    elif is_cpt_code(feature):
                        codes['cpt'].add(feature)
                    elif is_hcpcs_code(feature):
                        codes['hcpcs'].add(feature)
                
                codes_after = {
                    'icd': len(codes['icd']),
                    'cpt': len(codes['cpt']),
                    'hcpcs': len(codes['hcpcs']),
                }
                
                new_codes = {
                    'icd': codes_after['icd'] - codes_before['icd'],
                    'cpt': codes_after['cpt'] - codes_before['cpt'],
                    'hcpcs': codes_after['hcpcs'] - codes_before['hcpcs'],
                }
                
                if sum(new_codes.values()) > 0:
                    print(f"\n    -> Added: {new_codes['icd']} ICD, {new_codes['cpt']} CPT, {new_codes['hcpcs']} HCPCS")
            
            except Exception as e:
                print(f"\nError processing {csv_path}: {e}")
                continue
    
    print()  # New line after progress
    return codes


def lookup_icd_code(code: str) -> Optional[Dict[str, str]]:
    """
    Look up ICD code using web search.
    
    Returns:
        Dictionary with 'description' and 'classification' or None if not found
    """
    # Use web search to look up ICD codes
    # Format ICD-10 codes properly (e.g., F1120 -> F11.20, Z00129 -> Z00.129)
    formatted_code = code
    if re.match(r'^[A-Z]\d{5}$', code):  # 5 digits after letter
        formatted_code = f"{code[0]}{code[1:3]}.{code[3:]}"
    elif re.match(r'^[A-Z]\d{4}$', code):  # 4 digits after letter
        formatted_code = f"{code[0]}{code[1:3]}.{code[3:]}"
    elif re.match(r'^[A-Z]\d{3}$', code):  # 3 digits after letter
        formatted_code = f"{code[0]}{code[1:3]}.{code[3:]}"
    
    search_term = f"ICD-10 {formatted_code} code description"
    return None  # Will be populated by web search


def lookup_cpt_code(code: str) -> Optional[Dict[str, str]]:
    """
    Look up CPT code using web search.
    
    Returns:
        Dictionary with 'description' and 'classification' or None if not found
    """
    base_code = code.split('-')[0].split('_')[0]
    search_term = f"CPT {base_code} code description"
    return None  # Will be populated by web search


def lookup_hcpcs_code(code: str) -> Optional[Dict[str, str]]:
    """
    Look up HCPCS code using web search.
    
    Returns:
        Dictionary with 'description' and 'classification' or None if not found
    """
    search_term = f"HCPCS {code} code description"
    return None  # Will be populated by web search


def classify_icd_code(code: str, description: str = '') -> str:
    """
    Classify an ICD-10 code as administrative or medical based on heuristics.
    
    Heuristics based on ICD-10-CM Chapter 21 (Z00-Z99) patterns:
    - Z02.* codes are clearly administrative (administrative examinations)
    - Z00-Z04 can be administrative if for third-party requirements
    - Other Z-codes (Z11-Z13 screening, Z55-Z65 SDoH) are medical
    - All non-Z codes (A00-Y99) are medical diagnoses
    
    References:
    - https://www.bcbsri.com/providers/update/icd-10-administrative-examination-diagnosis-codes
    - https://www.wolterskluwer.com/en/expert-insights/guide-to-icd-10-cm-z-codes
    """
    code_upper = code.upper().strip()
    
    # Z02.* codes are clearly administrative (administrative examinations)
    if code_upper.startswith('Z02'):
        return 'administrative'
    
    # Z00-Z04 can be administrative if for third-party requirements
    # For now, we'll classify them as potentially administrative
    # (can be refined with encounter metadata if available)
    if re.match(r'^Z0[0-4]', code_upper):
        # Check description for third-party indicators
        desc_lower = description.lower()
        third_party_keywords = [
            'pre-employment', 'employment', 'school', 'insurance', 'legal',
            'administrative examination', 'third-party', 'requirement'
        ]
        if any(kw in desc_lower for kw in third_party_keywords):
            return 'administrative'
        # Default: treat as medical (preventive care)
        return 'medical'
    
    # Z11-Z13: Screening codes (medical - preventive care)
    if re.match(r'^Z1[1-3]', code_upper):
        return 'medical'
    
    # Z55-Z65, Z59: Social determinants of health (medical - clinical context)
    if re.match(r'^Z5[5-9]', code_upper) or code_upper.startswith('Z59'):
        return 'medical'
    
    # All other codes (A00-Y99, other Z-codes) are medical
    return 'medical'


def classify_cpt_code(code: str, description: str = '') -> str:
    """
    Classify a CPT code as administrative or medical based on heuristics.
    
    Heuristics:
    - E/M codes (9920x, 9921x) can be administrative if for third-party exams
    - Most Category I codes (Surgery, Radiology, Pathology, Medicine) are medical
    - HCPCS codes (G, H, S prefixes) need case-by-case review
    
    References:
    - https://www.ama-assn.org/topics/cpt-codes
    - https://www.aapc.com/resources/what-are-e-m-codes
    """
    code_clean = code.split('-')[0].split('_')[0].strip()
    
    # E/M codes (9920x, 9921x range) - check context
    if re.match(r'^992[01]', code_clean):
        desc_lower = description.lower()
        # Third-party exam indicators
        third_party_keywords = [
            'pre-employment', 'employment', 'independent medical examination',
            'disability evaluation', 'legal', 'administrative', 'form completion'
        ]
        if any(kw in desc_lower for kw in third_party_keywords):
            return 'administrative'
        # Default: medical (problem-oriented, preventive, management visits)
        return 'medical'
    
    # Emergency department E/M codes (9928x, 9929x) - typically medical
    if re.match(r'^992[89]', code_clean):
        return 'medical'
    
    # Critical care (99291, 99292) - medical
    if code_clean in ['99291', '99292']:
        return 'medical'
    
    # Most Category I CPT codes are medical:
    # - Surgery (10021-69990)
    # - Radiology (70000-79999)
    # - Pathology & Laboratory (80000-89999)
    # - Medicine (90000-99999)
    if re.match(r'^[1-9]\d{4}$', code_clean):
        return 'medical'
    
    # Default: medical (most CPT codes are medical procedures)
    return 'medical'


def classify_hcpcs_code(code: str, description: str = '') -> str:
    """
    Classify an HCPCS code as administrative or medical.
    
    HCPCS codes need case-by-case review, but some patterns:
    - G codes: Often administrative (care management, behavioral health)
    - H codes: Behavioral health services (can be medical or administrative)
    - S codes: Temporary codes (varies)
    """
    code_upper = code.upper().strip()
    
    # G codes: Often administrative services (care management, behavioral health)
    if code_upper.startswith('G'):
        desc_lower = description.lower()
        # Care management codes can be administrative
        if 'care management' in desc_lower or 'behavioral health' in desc_lower:
            # Review case-by-case, but many G codes are administrative
            return 'administrative'
        return 'medical'
    
    # H codes: Behavioral health services
    if code_upper.startswith('H'):
        # Most H codes are medical (behavioral health services)
        return 'medical'
    
    # S codes: Temporary codes (varies by code)
    if code_upper.startswith('S'):
        # Review case-by-case
        return 'TO_BE_DETERMINED'
    
    # Default: medical
    return 'medical'


def classify_code_as_administrative(code: str, code_type: str, description: str = '') -> str:
    """
    Classify a code as administrative or medical based on code type and heuristics.
    
    This function routes to type-specific classifiers.
    """
    if code_type.upper() in ['ICD-10', 'ICD-9']:
        return classify_icd_code(code, description)
    elif code_type.upper() == 'CPT':
        return classify_cpt_code(code, description)
    elif code_type.upper() == 'HCPCS':
        return classify_hcpcs_code(code, description)
    else:
        return 'TO_BE_DETERMINED'


def format_icd_code_for_lookup(code: str) -> str:
    """
    Format ICD-10 code for lookup (add decimal point).
    Examples: F1120 -> F11.20, Z00129 -> Z00.129, F909 -> F90.9
    """
    if re.match(r'^[A-Z]\d{5}$', code):  # 5 digits: Letter + 3 digits + 2 digits
        return f"{code[0]}{code[1:3]}.{code[3:]}"
    elif re.match(r'^[A-Z]\d{4}$', code):  # 4 digits: Letter + 3 digits + 1 digit
        return f"{code[0]}{code[1:3]}.{code[3:]}"
    elif re.match(r'^[A-Z]\d{3}$', code):  # 3 digits: Letter + 2 digits + 1 digit
        return f"{code[0]}{code[1:3]}.{code[3:]}"
    elif '.' in code:  # Already formatted
        return code
    else:
        return code  # Return as-is if pattern doesn't match


def google_search_code_description(search_query: str, code_type: str) -> Tuple[str, str]:
    """
    Use Google search to find code descriptions.
    
    Returns:
        Tuple of (description, lookup_response)
    """
    if not REQUESTS_AVAILABLE:
        return '', 'Web search not available (requests library not installed)'
    
    try:
        # URL encode the search query
        import urllib.parse
        encoded_query = urllib.parse.quote_plus(search_query)
        search_url = f"https://www.google.com/search?q={encoded_query}"
        
        response = requests.get(search_url, timeout=10, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Google search results can be in various formats
            # Try multiple selectors for result snippets
            
            # Method 1: Look for span elements with class containing "st" (snippet text)
            snippets = soup.find_all('span', class_=lambda x: x and 'st' in str(x).lower())
            for snippet in snippets[:5]:
                text = snippet.get_text(strip=True)
                if len(text) > 30 and len(text) < 500:
                    # Check if it looks like a code description
                    if any(keyword in text.lower() for keyword in ['code', 'description', 'diagnosis', 'procedure', 'medical', 'icd', 'cpt']):
                        return text[:200], f"Found via Google search: {search_url}"
            
            # Method 2: Look for div elements with specific classes
            result_divs = soup.find_all('div', class_=lambda x: x and any(
                cls in str(x).lower() for cls in ['VwiC3b', 's3v9rd', 'AP7Wnd', 'BNeawe', 's3v9rd']
            ))
            for div in result_divs[:5]:
                text = div.get_text(strip=True)
                if len(text) > 30 and len(text) < 500:
                    if any(keyword in text.lower() for keyword in ['code', 'description', 'diagnosis', 'procedure', 'medical']):
                        return text[:200], f"Found via Google search: {search_url}"
            
            # Method 3: Look for any text blocks that might contain descriptions
            # Extract text from the main content area
            body_text = soup.get_text()
            # Look for patterns that suggest code descriptions
            lines = body_text.split('\n')
            for line in lines:
                line = line.strip()
                if len(line) > 40 and len(line) < 300:
                    if any(keyword in line.lower() for keyword in ['code', 'description', 'diagnosis', 'procedure']):
                        # Check if it's not navigation or UI text
                        if not any(skip in line.lower() for skip in ['sign in', 'menu', 'search', 'google', 'images', 'videos']):
                            return line[:200], f"Found via Google search: {search_url}"
        
        return '', f"Google search attempted but description not extracted: {search_url}"
        
    except requests.exceptions.RequestException as e:
        return '', f"Google search request failed: {str(e)}"
    except Exception as e:
        return '', f"Google search error: {str(e)}"


def lookup_icd_code_web(code: str, formatted_code: str) -> Tuple[str, str]:
    """
    Look up ICD code description from local files first, then web sources.
    
    Note: CDC ICD-10-CM tool is JavaScript-rendered and cannot be easily scraped.
    We'll try local files first, then alternative sources and Google search.
    
    Returns:
        Tuple of (description, lookup_response)
    """
    # First, try local files
    local_result = lookup_icd_code_local(code)
    if local_result:
        description, source = local_result
        return description, f"Found via {source}: {CODE_LOOKUP_DIR}"
    
    if not REQUESTS_AVAILABLE:
        return '', 'Local lookup not found. Web search not available (requests library not installed)'
    
    try:
        # Try alternative ICD lookup sources first
        # Option 1: Try icd10data.com (if available)
        url = f"https://www.icd10data.com/ICD10CM/{formatted_code}"
        response = requests.get(url, timeout=10, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            # Look for description in title or meta
            title = soup.find('title')
            if title:
                title_text = title.get_text()
                if formatted_code in title_text or code in title_text:
                    # Extract description from title
                    desc = title_text.replace('ICD-10-CM', '').replace('Code', '').strip()
                    if len(desc) > 10:
                        return desc, f"Found via icd10data.com: {url}"
            
            # Look for description in page content
            body_text = soup.get_text()
            if formatted_code in body_text:
                # Try to find description near the code
                idx = body_text.find(formatted_code)
                if idx != -1:
                    context = body_text[idx:idx+200].strip()
                    if len(context) > 20:
                        return context[:150], f"Found via icd10data.com: {url}"
        
        # Fallback to Google search
        search_query = f"ICD-10 {formatted_code} code description"
        desc, resp = google_search_code_description(search_query, 'ICD-10')
        if desc:
            return desc, resp
        
        # Note: CDC tool is JavaScript-rendered, so we can't easily extract
        cdc_url = f"https://icd10cmtool.cdc.gov/?fy=FY2026&q={formatted_code}"
        return '', f"CDC ICD-10-CM tool is JavaScript-rendered (requires browser). Google search attempted. Reference: {cdc_url}"
        
    except requests.exceptions.RequestException as e:
        return '', f"Web request failed: {str(e)}"
    except Exception as e:
        return '', f"Web search error: {str(e)}"


def lookup_cpt_code_web(code: str) -> Tuple[str, str]:
    """
    Look up CPT code description from AAPC or Google search.
    
    Returns:
        Tuple of (description, lookup_response)
    """
    if not REQUESTS_AVAILABLE:
        return '', 'Web search not available (requests library not installed)'
    
    try:
        # Try AAPC CPT lookup first (works well based on testing)
        url = f"https://www.aapc.com/codes/cpt-codes/{code}"
        response = requests.get(url, timeout=10, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Try meta description first (usually most reliable)
            meta_desc = soup.find('meta', attrs={'name': 'description'})
            if meta_desc and meta_desc.get('content'):
                desc = meta_desc['content']
                if code in desc and len(desc) > 20:
                    # Clean up the description
                    desc = desc.replace('Codify by AAPC', '').strip()
                    return desc, f"Found via AAPC CPT lookup: {url}"
            
            # Try H1 tag
            h1 = soup.find('h1')
            if h1:
                h1_text = h1.get_text()
                if code in h1_text:
                    # Look for description in nearby elements
                    parent = h1.parent
                    if parent:
                        desc_text = parent.get_text(strip=True)
                        if len(desc_text) > 50:
                            # Extract meaningful portion
                            idx = desc_text.find(code)
                            if idx != -1:
                                context = desc_text[idx:idx+300].strip()
                                if len(context) > 50:
                                    return context[:200], f"Found via AAPC CPT lookup: {url}"
        
        # Fallback to Google search
        search_query = f"CPT {code} code description"
        desc, resp = google_search_code_description(search_query, 'CPT')
        if desc:
            return desc, resp
        
        # If AAPC lookup failed, note it
        if response.status_code != 200:
            return '', f"CPT lookup failed. Status: {response.status_code}. Google search attempted."
        
        return '', f"AAPC CPT lookup accessed but description not extracted. Google search attempted. Code: {code}. Reference: {url}"
        
    except requests.exceptions.RequestException as e:
        # Try Google search as fallback even on error
        search_query = f"CPT {code} code description"
        desc, resp = google_search_code_description(search_query, 'CPT')
        if desc:
            return desc, resp
        return '', f"Web request failed: {str(e)}. Google search attempted."
    except Exception as e:
        return '', f"Web search error: {str(e)}"


def lookup_hcpcs_code_web(code: str) -> Tuple[str, str]:
    """
    Look up HCPCS code description from official CMS sources or Google search.
    
    Returns:
        Tuple of (description, lookup_response)
    """
    if not REQUESTS_AVAILABLE:
        return '', 'Web search not available (requests library not installed)'
    
    try:
        # Try CMS HCPCS lookup first
        url = f"https://www.cms.gov/medicare/physician-fee-schedule/search?Y=0&T=0&HT=0&CT=2&H1={code}"
        response = requests.get(url, timeout=10, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            # Look for code description
            desc_elements = soup.find_all(['td', 'div', 'span'], string=re.compile(code, re.I))
            for elem in desc_elements:
                text = elem.get_text(strip=True)
                if code in text and len(text) > 10:
                    return text[:200], f"Found via CMS HCPCS lookup: {url}"
        
        # Fallback to Google search
        search_query = f"HCPCS {code} code description"
        desc, resp = google_search_code_description(search_query, 'HCPCS')
        if desc:
            return desc, resp
        
        return '', f"HCPCS lookup attempted. Google search attempted. Reference: {url}"
        
    except requests.exceptions.RequestException as e:
        # Try Google search as fallback
        search_query = f"HCPCS {code} code description"
        desc, resp = google_search_code_description(search_query, 'HCPCS')
        if desc:
            return desc, resp
        return '', f"Web request failed: {str(e)}. Google search attempted."
    except Exception as e:
        return '', f"Web search error: {str(e)}"


def attempt_web_search(search_term: str, code_type: str, code: str = '', formatted_code: str = '') -> Tuple[str, str]:
    """
    Attempt to look up code meaning using web search.
    
    This function queries official medical coding websites to get code descriptions.
    
    Returns:
        Tuple of (description, lookup_response)
        - description: Code description if found, empty string otherwise
        - lookup_response: Search term and any results/notes
    """
    print(f"\n    [WEB SEARCH REQUEST]")
    print(f"      Type: {code_type}")
    print(f"      Search Term: {search_term}")
    print(f"      Code: {code}")
    
    description = ''
    lookup_response = ''
    
    if not REQUESTS_AVAILABLE:
        lookup_response = 'Web search not available (requests/beautifulsoup4 not installed)'
        print(f"    [WEB SEARCH RESPONSE]")
        print(f"      Description: (not available - library not installed)")
        print(f"      Response: {lookup_response}")
        return description, lookup_response
    
    try:
        # Route to appropriate lookup function based on code type
        if code_type.upper() in ['ICD-10', 'ICD-9']:
            if formatted_code:
                description, lookup_response = lookup_icd_code_web(code, formatted_code)
            else:
                lookup_response = f"ICD code lookup attempted but formatted_code not provided"
        elif code_type.upper() == 'CPT':
            description, lookup_response = lookup_cpt_code_web(code)
        elif code_type.upper() == 'HCPCS':
            description, lookup_response = lookup_hcpcs_code_web(code)
        else:
            lookup_response = f"Web search attempted for {code_type} code: {code}"
        
        # Add a small delay to be respectful to servers (rate limiting)
        time.sleep(0.2)
        
    except Exception as e:
        lookup_response = f"Web search error: {str(e)}"
    
    print(f"    [WEB SEARCH RESPONSE]")
    print(f"      Description: {description if description else '(not available - using heuristic)'}")
    print(f"      Response: {lookup_response}")
    
    return description, lookup_response


def lookup_code_meanings(codes: Dict[str, Set[str]], use_web_search: bool = True) -> pd.DataFrame:
    """
    Look up code meanings and create a research dataframe.
    
    Uses web search to look up codes with fallback to heuristics if lookup fails.
    Includes lookup responses for validation.
    
    Returns:
        DataFrame with code, type, description, classification, lookup_source, and reference URLs
    """
    records = []
    total_codes = len(codes['icd']) + len(codes['cpt']) + len(codes['hcpcs'])
    processed = 0
    
    print(f"\nStep 2: Classifying {total_codes} codes (using web search with heuristic fallback)...")
    print(f"  Processing {len(codes['icd'])} ICD codes, {len(codes['cpt'])} CPT codes, {len(codes['hcpcs'])} HCPCS codes...")
    
    # ICD codes
    print(f"\n  Processing ICD codes...")
    for code in sorted(codes['icd']):
        processed += 1
        code_type = 'ICD-10' if re.match(r'^[A-Z]', code) else 'ICD-9'
        
        # Format code for lookup (ICD-10 format: Letter + 2 digits + decimal + remaining)
        formatted_code = format_icd_code_for_lookup(code)
        
        print(f"\n  [{processed}/{len(codes['icd'])}] Processing ICD code: {code} (formatted: {formatted_code})")
        
        description = ''
        classification = ''
        lookup_source = 'heuristic'
        lookup_response = ''
        
        if use_web_search:
            try:
                search_term = f"ICD-10 {formatted_code} code description meaning"
                ws_description, ws_response = attempt_web_search(search_term, code_type, code=code, formatted_code=formatted_code)
                if ws_description:
                    description = ws_description
                    lookup_source = 'web_search'
                    lookup_response = ws_response
                else:
                    lookup_source = 'web_search_with_heuristic_fallback'
                    lookup_response = ws_response
            except Exception as e:
                print(f"    [ERROR] Web search failed: {e}")
                lookup_response = f"Web search failed: {e}"
                lookup_source = 'heuristic_fallback'
        
        # Apply heuristic classification (always used, even if web search succeeds)
        # Heuristics provide reliable classification even without web search
        heuristic_classification = classify_code_as_administrative(code, code_type, description)
        
        # Use heuristic classification (web search may provide description but not classification)
        classification = heuristic_classification
        
        # If web search didn't provide description, note that heuristics were used
        if not description:
            if lookup_source == 'heuristic':
                lookup_response = f"Heuristic classification based on code pattern: {code} -> {classification}"
            elif lookup_source == 'web_search_with_heuristic_fallback':
                lookup_response += f" | Heuristic classification: {classification}"
        
        print(f"    [RESULT] Classification: {classification} | Source: {lookup_source}")
        
        records.append({
            'code': code,
            'formatted_code': formatted_code,
            'code_type': code_type,
            'description': description,
            'classification': classification,
            'lookup_source': lookup_source,
            'lookup_response': lookup_response,
            'reference_url': f'https://icd10cmtool.cdc.gov/?fy=FY2026&q={formatted_code}',
            'notes': '',
        })
    
    # CPT codes
    print(f"\n  Processing CPT codes...")
    cpt_start = processed
    for code in sorted(codes['cpt']):
        processed += 1
        base_code = code.split('-')[0].split('_')[0]
        
        print(f"\n  [{processed - cpt_start}/{len(codes['cpt'])}] Processing CPT code: {code} (base: {base_code})")
        
        description = ''
        classification = ''
        lookup_source = 'heuristic'
        lookup_response = ''
        
        if use_web_search:
            try:
                search_term = f"CPT {base_code} code description meaning"
                ws_description, ws_response = attempt_web_search(search_term, 'CPT', code=code, formatted_code=base_code)
                if ws_description:
                    description = ws_description
                    lookup_source = 'web_search'
                    lookup_response = ws_response
                else:
                    lookup_source = 'web_search_with_heuristic_fallback'
                    lookup_response = ws_response
            except Exception as e:
                print(f"    [ERROR] Web search failed: {e}")
                lookup_response = f"Web search failed: {e}"
                lookup_source = 'heuristic_fallback'
        
        # Apply heuristic classification
        heuristic_classification = classify_code_as_administrative(code, 'CPT', description)
        classification = heuristic_classification
        
        # If web search didn't provide description, note that heuristics were used
        if not description:
            if lookup_source == 'heuristic':
                lookup_response = f"Heuristic classification based on code pattern: {code} -> {classification}"
            elif lookup_source == 'web_search_with_heuristic_fallback':
                lookup_response += f" | Heuristic classification: {classification}"
        
        print(f"    [RESULT] Classification: {classification} | Source: {lookup_source}")
        
        records.append({
            'code': code,
            'formatted_code': base_code,
            'code_type': 'CPT',
            'description': description,
            'classification': classification,
            'lookup_source': lookup_source,
            'lookup_response': lookup_response,
            'reference_url': f'https://www.ama-assn.org/topics/cpt-codes',
            'notes': '',
        })
    
    # HCPCS codes
    print(f"\n  Processing HCPCS codes...")
    hcpcs_start = processed
    for code in sorted(codes['hcpcs']):
        processed += 1
        
        print(f"\n  [{processed - hcpcs_start}/{len(codes['hcpcs'])}] Processing HCPCS code: {code}")
        
        description = ''
        classification = ''
        lookup_source = 'heuristic'
        lookup_response = ''
        
        if use_web_search:
            try:
                search_term = f"HCPCS {code} code description meaning"
                ws_description, ws_response = attempt_web_search(search_term, 'HCPCS', code=code, formatted_code=code)
                if ws_description:
                    description = ws_description
                    lookup_source = 'web_search'
                    lookup_response = ws_response
                else:
                    lookup_source = 'web_search_with_heuristic_fallback'
                    lookup_response = ws_response
            except Exception as e:
                print(f"    [ERROR] Web search failed: {e}")
                lookup_response = f"Web search failed: {e}"
                lookup_source = 'heuristic_fallback'
        
        # Apply heuristic classification
        heuristic_classification = classify_code_as_administrative(code, 'HCPCS', description)
        classification = heuristic_classification
        
        # If web search didn't provide description, note that heuristics were used
        if not description:
            if lookup_source == 'heuristic':
                lookup_response = f"Heuristic classification based on code pattern: {code} -> {classification}"
            elif lookup_source == 'web_search_with_heuristic_fallback':
                lookup_response += f" | Heuristic classification: {classification}"
        
        print(f"    [RESULT] Classification: {classification} | Source: {lookup_source}")
        
        records.append({
            'code': code,
            'formatted_code': code,
            'code_type': 'HCPCS',
            'description': description,
            'classification': classification,
            'lookup_source': lookup_source,
            'lookup_response': lookup_response,
            'reference_url': f'https://www.cms.gov/medicare/physician-fee-schedule/search?Y=0&T=0&HT=0&CT=2&H1={code}',
            'notes': '',
        })
    
    print()  # New line after progress
    
    # Summary statistics
    df = pd.DataFrame(records)
    admin_count = len(df[df['classification'] == 'administrative'])
    medical_count = len(df[df['classification'] == 'medical'])
    unknown_count = len(df[df['classification'] == 'TO_BE_DETERMINED'])
    
    print(f"\n  Classification Summary:")
    print(f"    Administrative: {admin_count}")
    print(f"    Medical: {medical_count}")
    print(f"    To be determined: {unknown_count}")
    print(f"    Total: {len(df)}")
    
    return df


def create_lookup_table(df: pd.DataFrame) -> Dict:
    """
    Create a lookup table JSON structure for administrative codes.
    
    Returns:
        Dictionary structure matching administrative_codes_lookup.json format
    """
    admin_codes = {
        'icd': [],
        'cpt': [],
        'hcpcs': [],
    }
    
    for _, row in df.iterrows():
        if row['classification'] == 'administrative':
            code_type = row['code_type'].lower()
            if code_type == 'icd-10' or code_type == 'icd-9':
                admin_codes['icd'].append(row['code'])
            elif code_type == 'cpt':
                admin_codes['cpt'].append(row['code'])
            elif code_type == 'hcpcs':
                admin_codes['hcpcs'].append(row['code'])
    
    return {
        'description': 'Lookup table for classifying codes as administrative vs. medical/pharmacy. Codes listed here will be filtered out during event filtering (Step 4b).',
        'version': '1.0',
        'last_updated': pd.Timestamp.now().strftime('%Y-%m-%d'),
        'administrative_codes': admin_codes,
        'notes': {
            'icd': 'ICD codes for administrative/billing purposes (e.g., routine health checks, administrative visits)',
            'cpt': 'CPT codes for administrative procedures (e.g., post-operative follow-up, administrative services)',
            'hcpcs': 'HCPCS codes for administrative services (e.g., behavioral health services, care management)',
        },
        'research_guidelines': {
            'how_to_identify_administrative_codes': [
                '1. Review code_analysis_protocol_vs_clinical CSV from research outputs',
                '2. Identify codes with high protocol_pct (>80%) that appear in protocol-like sequences',
                '3. Manually review codes to determine if they are administrative vs. clinical',
                '4. Add confirmed administrative codes to this lookup table',
                '5. Codes should be administrative/billing/scheduling, not clinical diagnoses or treatments',
            ],
            'post_event_leakage': 'Events occurring on or after target event date are automatically classified as administrative (leakage) and do not need to be in this lookup table.',
        },
    }


def main():
    """Main function to extract and research codes."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Research ICD, CPT, and HCPCS codes from feature importances')
    parser.add_argument('--skip-web-search', action='store_true', 
                       help='Skip web search lookups (faster, but codes will need manual classification)')
    args = parser.parse_args()
    
    print("=" * 80)
    print("ICD/CPT/HCPCS Code Research and Classification")
    print("=" * 80)
    
    # Preload ICD lookup files if available
    if CODE_LOOKUP_DIR.exists():
        print("\nPreloading ICD lookup files from local directory...")
        load_icd10_lookup()
        load_icd9_to_icd10_mapping()
        print()
    
    print("\n" + "=" * 80)
    print("Step 1: Extracting codes from aggregated feature importances...")
    print("=" * 80)
    
    codes = extract_codes_from_feature_importance_files()
    
    print(f"\n[OK] Step 1 Complete: Found codes:")
    print(f"  ICD codes: {len(codes['icd'])}")
    print(f"  CPT codes: {len(codes['cpt'])}")
    print(f"  HCPCS codes: {len(codes['hcpcs'])}")
    print(f"  Total: {len(codes['icd']) + len(codes['cpt']) + len(codes['hcpcs'])}")
    
    if len(codes['icd']) + len(codes['cpt']) + len(codes['hcpcs']) == 0:
        print("\nERROR: No codes found! Check that feature importance files exist.")
        return
    
    # Create research dataframe
    print("\n" + "=" * 80)
    print("Step 2: Looking up code meanings and classifying...")
    print("=" * 80)
    if args.skip_web_search:
        print("  (Skipping web search - codes will need manual classification)")
        df = lookup_code_meanings(codes, use_web_search=False)
    else:
        df = lookup_code_meanings(codes, use_web_search=True)
    
    print(f"\n[OK] Step 2 Complete: Classified {len(df)} codes")
    
    # Save to CSV
    print("\n" + "=" * 80)
    print("Step 3: Saving research outputs...")
    print("=" * 80)
    
    output_csv = OUTPUT_DIR / "icd_cpt_hcpcs_codes_research.csv"
    df.to_csv(output_csv, index=False)
    print(f"  [OK] Saved research CSV: {output_csv}")
    print(f"    Rows: {len(df)}")
    
    # Create and save lookup table
    lookup_table = create_lookup_table(df)
    lookup_json = OUTPUT_DIR / "administrative_codes_lookup.json"
    with open(lookup_json, 'w') as f:
        json.dump(lookup_table, f, indent=2)
    print(f"  [OK] Saved lookup table: {lookup_json}")
    print(f"    Administrative codes: {len(lookup_table['administrative_codes']['icd']) + len(lookup_table['administrative_codes']['cpt']) + len(lookup_table['administrative_codes']['hcpcs'])}")
    
    # Summary statistics
    admin_count = len(df[df['classification'] == 'administrative'])
    medical_count = len(df[df['classification'] == 'medical'])
    unknown_count = len(df[df['classification'] == 'TO_BE_DETERMINED'])
    
    print(f"\n[OK] Step 3 Complete")
    
    # Save code lists separately
    print("\n" + "=" * 80)
    print("Step 4: Saving code lists...")
    print("=" * 80)
    
    icd_file = OUTPUT_DIR / "icd_codes_list.txt"
    with open(icd_file, 'w') as f:
        f.write('\n'.join(sorted(codes['icd'])))
    print(f"  [OK] Saved ICD codes list: {icd_file} ({len(codes['icd'])} codes)")
    
    cpt_file = OUTPUT_DIR / "cpt_codes_list.txt"
    with open(cpt_file, 'w') as f:
        f.write('\n'.join(sorted(codes['cpt'])))
    print(f"  [OK] Saved CPT codes list: {cpt_file} ({len(codes['cpt'])} codes)")
    
    hcpcs_file = OUTPUT_DIR / "hcpcs_codes_list.txt"
    with open(hcpcs_file, 'w') as f:
        f.write('\n'.join(sorted(codes['hcpcs'])))
    print(f"  [OK] Saved HCPCS codes list: {hcpcs_file} ({len(codes['hcpcs'])} codes)")
    
    print(f"\n[OK] Step 4 Complete")
    
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    print(f"Total codes processed: {len(df)}")
    print(f"  Administrative codes: {admin_count}")
    print(f"  Medical codes: {medical_count}")
    print(f"  To be determined: {unknown_count}")
    print(f"\nOutput files:")
    print(f"  Research CSV: {output_csv}")
    print(f"  Lookup table: {lookup_json}")
    print(f"  Code lists: {icd_file.name}, {cpt_file.name}, {hcpcs_file.name}")
    print(f"\n" + "=" * 80)
    print("Next Steps:")
    print("=" * 80)
    print(f"  1. Review the research CSV: {output_csv}")
    print(f"  2. Review the lookup table: {lookup_json}")
    print(f"  3. Manually verify classifications using:")
    print(f"     - ICD codes: https://icd10cmtool.cdc.gov/?fy=FY2026")
    print(f"     - CPT codes: https://www.ama-assn.org/topics/cpt-codes")
    print(f"     - HCPCS codes: https://www.cms.gov/medicare/physician-fee-schedule/search")
    print(f"  4. Update classifications in {output_csv} if needed")
    print(f"  5. Copy {lookup_json} to 4b_event_filter/administrative_codes_lookup.json when ready")
    print("=" * 80)
    lookup_json = OUTPUT_DIR / "administrative_codes_lookup.json"
    with open(lookup_json, 'w') as f:
        json.dump(lookup_table, f, indent=2)
    print(f"Step 4: Saved lookup table to: {lookup_json}")
    
    # Summary statistics
    admin_count = len(df[df['classification'] == 'administrative'])
    medical_count = len(df[df['classification'] == 'medical'])
    unknown_count = len(df[df['classification'] == 'TO_BE_DETERMINED'])
    
    print(f"\n" + "=" * 80)
    print("Classification Summary:")
    print("=" * 80)
    print(f"  Administrative codes: {admin_count}")
    print(f"  Medical codes: {medical_count}")
    print(f"  To be determined: {unknown_count}")
    print(f"  Total codes: {len(df)}")
    
    # Save code lists separately
    icd_file = OUTPUT_DIR / "icd_codes_list.txt"
    with open(icd_file, 'w') as f:
        f.write('\n'.join(sorted(codes['icd'])))
    
    cpt_file = OUTPUT_DIR / "cpt_codes_list.txt"
    with open(cpt_file, 'w') as f:
        f.write('\n'.join(sorted(codes['cpt'])))
    
    hcpcs_file = OUTPUT_DIR / "hcpcs_codes_list.txt"
    with open(hcpcs_file, 'w') as f:
        f.write('\n'.join(sorted(codes['hcpcs'])))
    
    print(f"\n" + "=" * 80)
    print("Next Steps:")
    print("=" * 80)
    print(f"  1. Review the research CSV: {output_csv}")
    print(f"  2. Review the lookup table: {lookup_json}")
    print(f"  3. Manually verify classifications using:")
    print(f"     - ICD codes: https://icd10cmtool.cdc.gov/?fy=FY2026")
    print(f"     - CPT codes: https://www.cms.gov/medicare/physician-fee-schedule/search")
    print(f"     - HCPCS codes: https://www.cms.gov/medicare/physician-fee-schedule/search")
    print(f"  4. Update classifications in {output_csv} if needed")
    print(f"  5. Copy {lookup_json} to 4b_event_filter/administrative_codes_lookup.json when ready")
    
    return codes, df, lookup_table


if __name__ == "__main__":
    codes, df, lookup_table = main()
