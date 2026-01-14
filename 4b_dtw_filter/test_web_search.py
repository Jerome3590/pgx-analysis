#!/usr/bin/env python3
"""
Test script to examine website structures and test web search functionality.
"""

import requests
from bs4 import BeautifulSoup
import re
import json

def test_icd_lookup(code: str, formatted_code: str):
    """Test ICD code lookup."""
    print(f"\n{'='*80}")
    print(f"Testing ICD Code: {code} (formatted: {formatted_code})")
    print(f"{'='*80}")
    
    url = f"https://icd10cmtool.cdc.gov/?fy=FY2026&q={formatted_code}"
    print(f"\n[REQUEST] URL: {url}")
    
    try:
        response = requests.get(url, timeout=10, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        print(f"[RESPONSE] Status: {response.status_code}")
        print(f"[RESPONSE] Content length: {len(response.content)} bytes")
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Check if it's a JavaScript app
            scripts = soup.find_all('script')
            print(f"[ANALYSIS] Found {len(scripts)} script tags")
            
            # Look for any text containing the code
            body_text = soup.get_text()
            if formatted_code in body_text or code in body_text:
                print(f"[ANALYSIS] Code found in page text")
                # Find the context around the code
                idx = body_text.find(formatted_code)
                if idx != -1:
                    context = body_text[max(0, idx-50):min(len(body_text), idx+200)]
                    print(f"[ANALYSIS] Context: ...{context}...")
            else:
                print(f"[ANALYSIS] Code not found in page text (likely JavaScript-rendered)")
            
            # Save HTML for inspection
            with open('test_icd_page.html', 'w', encoding='utf-8') as f:
                f.write(soup.prettify())
            print(f"[SAVED] HTML saved to test_icd_page.html")
            
            # Look for common description patterns
            desc_patterns = [
                soup.find_all('div', class_=re.compile(r'description|title|code|result', re.I)),
                soup.find_all('span', class_=re.compile(r'description|title|code|result', re.I)),
                soup.find_all('p', class_=re.compile(r'description|title|code|result', re.I)),
            ]
            
            for pattern_list in desc_patterns:
                if pattern_list:
                    print(f"[ANALYSIS] Found {len(pattern_list)} potential description elements")
                    for elem in pattern_list[:3]:  # Show first 3
                        text = elem.get_text(strip=True)
                        if text and len(text) > 10:
                            print(f"  - {text[:100]}")
        
    except Exception as e:
        print(f"[ERROR] {e}")


def test_cpt_lookup(code: str):
    """Test CPT code lookup."""
    print(f"\n{'='*80}")
    print(f"Testing CPT Code: {code}")
    print(f"{'='*80}")
    
    # Try AAPC CPT lookup site
    url = f"https://www.aapc.com/codes/cpt-codes/{code}"
    print(f"\n[REQUEST] URL: {url}")
    try:
        response = requests.get(url, timeout=10, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        print(f"[RESPONSE] Status: {response.status_code}")
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            print(f"[RESPONSE] Content length: {len(response.content)} bytes")
            
            # Try to find the description
            title = soup.find('title')
            if title:
                print(f"[EXTRACTED] Title: {title.get_text()}")
            
            h1 = soup.find('h1')
            if h1:
                print(f"[EXTRACTED] H1: {h1.get_text()}")
            
            # Look for description in various elements
            desc_candidates = []
            
            # Try meta description
            meta_desc = soup.find('meta', attrs={'name': 'description'})
            if meta_desc and meta_desc.get('content'):
                desc_candidates.append(('meta', meta_desc['content']))
            
            # Try to find description paragraphs or divs
            for tag in ['p', 'div', 'span']:
                elements = soup.find_all(tag, string=re.compile(code, re.I))
                for elem in elements[:3]:
                    parent = elem.parent
                    if parent:
                        text = parent.get_text(strip=True)
                        if len(text) > 20 and len(text) < 500:
                            desc_candidates.append((tag, text))
            
            # Look for specific class patterns
            desc_divs = soup.find_all(['div', 'p'], class_=lambda x: x and any(
                kw in x.lower() for kw in ['description', 'detail', 'code', 'info', 'content']
            ))
            for div in desc_divs[:3]:
                text = div.get_text(strip=True)
                if code in text and len(text) > 20:
                    desc_candidates.append(('div_class', text[:300]))
            
            if desc_candidates:
                print(f"[EXTRACTED] Found {len(desc_candidates)} potential descriptions:")
                for i, (source, text) in enumerate(desc_candidates[:3], 1):
                    print(f"  {i}. [{source}] {text[:200]}...")
            else:
                print(f"[ANALYSIS] No clear description found, but code appears in page")
                
    except Exception as e:
        print(f"[ERROR] {e}")


if __name__ == "__main__":
    # Test with a few sample codes
    print("Testing Web Search Functionality")
    print("="*80)
    
    # Test ICD codes
    test_icd_lookup("F1120", "F11.20")
    test_icd_lookup("Z00129", "Z00.129")
    
    # Test CPT codes
    test_cpt_lookup("99213")
    test_cpt_lookup("99284")
