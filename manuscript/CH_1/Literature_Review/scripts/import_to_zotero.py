"""
Script to automatically import all publications from CSV files to Zotero
Uses PMC IDs and titles to fetch complete citations and add to Zotero collection
"""

import os
import sys
import time
import pandas as pd
from pathlib import Path
from pyzotero import zotero
import requests
from xml.etree import ElementTree as ET
import re

# Fix Windows encoding issues
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

# Zotero API credentials
ZOTERO_USER_ID = '6037399'
ZOTERO_API_KEY = 'xxjsStqHkKgaSNnzb8FmG3Zb'
ZOTERO_COLLECTION_ID = 'LS75EWXU'  # Optional: specific collection

# Initialize Zotero client
zot = zotero.Zotero(ZOTERO_USER_ID, 'user', ZOTERO_API_KEY)

def fetch_pubmed_citation(pmcid_or_title):
    """Fetch complete citation from PubMed using PMC ID or title"""
    try:
        # Try PMC ID first
        if 'PMC' in str(pmcid_or_title):
            pmc_clean = str(pmcid_or_title).replace('PMC', '')
            # Use PubMed API to search by PMC ID
            url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
            params = {
                'db': 'pubmed',
                'term': f'{pmc_clean}[PMCID]',
                'retmode': 'json'
            }
            response = requests.get(url, params=params)
            data = response.json()
            
            if 'esearchresult' in data and 'idlist' in data['esearchresult']:
                pmids = data['esearchresult']['idlist']
                if pmids:
                    pmid = pmids[0]
                else:
                    return None
            else:
                return None
        else:
            # Search by title (simplified - would need more complex query)
            return None
        
        # Fetch full record
        fetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        fetch_params = {
            'db': 'pubmed',
            'id': pmid,
            'retmode': 'xml'
        }
        fetch_response = requests.get(fetch_url, params=fetch_params)
        
        # Parse XML
        root = ET.fromstring(fetch_response.content)
        
        # Extract citation information
        article = root.find('.//PubmedArticle')
        if article is None:
            return None
        
        # Title
        title_elem = article.find('.//ArticleTitle')
        title = title_elem.text if title_elem is not None else None
        
        # Authors
        authors = []
        for author in article.findall('.//Author'):
            last_name = author.find('LastName')
            first_name = author.find('ForeName')
            initials = author.find('Initials')
            
            if last_name is not None:
                last = last_name.text
                if first_name is not None:
                    first = first_name.text
                    authors.append(f"{last}, {first}")
                elif initials is not None:
                    authors.append(f"{last}, {initials.text}")
                else:
                    authors.append(last)
        
        authors_str = ", ".join(authors[:6])
        if len(authors) > 6:
            authors_str += ", et al."
        
        # Journal
        journal_elem = article.find('.//Journal/Title')
        journal = journal_elem.text if journal_elem is not None else None
        
        # Year
        year_elem = article.find('.//PubDate/Year')
        year = year_elem.text if year_elem is not None else None
        
        # Volume, Issue, Pages
        volume_elem = article.find('.//Volume')
        volume = volume_elem.text if volume_elem is not None else None
        
        issue_elem = article.find('.//Issue')
        issue = issue_elem.text if issue_elem is not None else None
        
        pages_elem = article.find('.//Pagination/MedlinePgn')
        pages = pages_elem.text if pages_elem is not None else None
        
        # DOI
        doi = None
        for article_id in article.findall('.//ArticleId'):
            if article_id.get('IdType') == 'doi':
                doi = article_id.text
                break
        
        # PMC ID
        pmc_id = None
        for article_id in article.findall('.//ArticleId'):
            if article_id.get('IdType') == 'pmc':
                pmc_id = f"PMC{article_id.text}"
                break
        
        return {
            'title': title,
            'authors': authors_str,
            'journal': journal,
            'year': year,
            'volume': volume,
            'issue': issue,
            'pages': pages,
            'doi': doi,
            'pmc_id': pmc_id
        }
    except Exception as e:
        print(f"  Error fetching citation: {e}")
        return None

def get_collection_and_tags(csv_path):
    """Extract collection name and tags from CSV file path"""
    path_parts = Path(csv_path).parts
    
    # Find chapter1 in path
    try:
        chapter1_idx = path_parts.index('chapter1')
        section_parts = path_parts[chapter1_idx + 1:]
    except ValueError:
        # Not in chapter1, check for other_chapters
        try:
            other_idx = path_parts.index('other_chapters')
            section_parts = path_parts[other_idx + 1:]
        except ValueError:
            return None, []
    
    # Extract section (1.1, 1.2, etc.)
    section = None
    topic = None
    tags = []
    
    if len(section_parts) > 0:
        section = section_parts[0]  # e.g., "1.1_introduction"
        tags.append(section.replace('_', ' ').title())
        
        if len(section_parts) > 1:
            topic = section_parts[1]  # e.g., "blackbox_cds"
            tags.append(topic.replace('_', ' ').title())
            
            # Add more specific tags for nested topics
            if len(section_parts) > 2:
                subtopic = section_parts[2]  # e.g., "fpgrowth"
                tags.append(subtopic.replace('_', ' ').title())
    
    # Create collection name
    collection_name = None
    if section:
        if topic:
            collection_name = f"{section} - {topic}".replace('_', ' ').title()
        else:
            collection_name = section.replace('_', ' ').title()
    
    return collection_name, tags

def get_or_create_collection(collection_name, parent_collection_id=None):
    """Get existing collection or create new one"""
    try:
        # Get all collections
        collections = zot.collections()
        
        # Search for existing collection
        for coll in collections:
            if coll['data']['name'] == collection_name:
                return coll['key']
        
        # Create new collection
        collection_data = {
            'name': collection_name,
            'parentCollection': parent_collection_id
        }
        created = zot.create_collections([collection_data])
        if created:
            # Get the newly created collection
            collections = zot.collections()
            for coll in collections:
                if coll['data']['name'] == collection_name:
                    return coll['key']
        
        return None
    except Exception as e:
        print(f"  Warning: Could not create collection '{collection_name}': {e}")
        return None

def create_zotero_item(citation_data, tags=None):
    """Create a Zotero item from citation data"""
    # Parse authors
    creators = []
    if citation_data.get('authors'):
        author_list = citation_data['authors'].split(', ')
        for author in author_list[:10]:  # Limit to 10 authors
            if ' et al.' in author:
                break
            parts = author.split(', ')
            if len(parts) >= 2:
                creators.append({
                    'creatorType': 'author',
                    'firstName': parts[1],
                    'lastName': parts[0]
                })
            else:
                creators.append({
                    'creatorType': 'author',
                    'lastName': parts[0]
                })
    
    # Build item
    item = {
        'itemType': 'journalArticle',
        'title': citation_data.get('title', ''),
        'creators': creators,
        'date': citation_data.get('year', ''),
        'publicationTitle': citation_data.get('journal', ''),
        'volume': citation_data.get('volume', ''),
        'issue': citation_data.get('issue', ''),
        'pages': citation_data.get('pages', ''),
        'DOI': citation_data.get('doi', ''),
        'extra': f"PMC ID: {citation_data.get('pmc_id', 'N/A')}"
    }
    
    # Add tags if provided
    if tags:
        item['tags'] = [{'tag': tag} for tag in tags]
    
    # Add URL if PMC ID available
    if citation_data.get('pmc_id'):
        pmc_clean = citation_data['pmc_id'].replace('PMC', '')
        item['url'] = f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmc_clean}/"
    
    return item

def check_item_exists(title):
    """Check if item already exists in Zotero (by title)"""
    try:
        # Get all items in collection
        items = zot.collection_items(ZOTERO_COLLECTION_ID)
        for item in items:
            if item['data'].get('title', '').lower() == title.lower():
                return True
        return False
    except:
        return False

def import_csv_to_zotero(csv_path, collection_id=None, limit=None, test_mode=False, organize=True):
    """Import articles from a CSV file to Zotero"""
    print(f"\n=== Processing: {csv_path} ===")
    
    # Get organization info from path
    collection_name = None
    tags = []
    target_collection_id = collection_id
    
    if organize:
        collection_name, tags = get_collection_and_tags(csv_path)
        if collection_name:
            print(f"Collection: {collection_name}")
            print(f"Tags: {', '.join(tags)}")
            # Get or create collection
            target_collection_id = get_or_create_collection(collection_name, parent_collection_id=collection_id)
            if not target_collection_id:
                print(f"  Warning: Using default collection")
                target_collection_id = collection_id
        else:
            print(f"  Using default collection")
    
    # Read CSV
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return None
    
    if len(df) == 0:
        print("No articles found")
        return None
    
    # Limit if specified
    if limit:
        df = df.head(limit)
    
    print(f"Found {len(df)} articles")
    if test_mode:
        print("TEST MODE: Only processing first 3 articles")
        df = df.head(3)
    
    print("Importing to Zotero...\n")
    
    success_count = 0
    error_count = 0
    skipped_count = 0
    
    for idx, row in df.iterrows():
        print(f"[{idx+1}/{len(df)}] ", end="")
        
        # Get identifier
        pmc_id = row.get('pmc_id', '')
        title = row.get('title', '')
        
        if pd.isna(pmc_id) or pmc_id == 'NA' or pmc_id == '':
            identifier = title
        else:
            identifier = pmc_id
        
        # Check if already exists (skip check for now to speed up - can add back later)
        # if check_item_exists(title):
        #     print(f"Skipped (already exists): {title[:50]}...")
        #     skipped_count += 1
        #     continue
        
        # Fetch complete citation
        citation = fetch_pubmed_citation(identifier)
        
        if citation is None:
            # Use basic data from CSV
            citation = {
                'title': title,
                'authors': row.get('authors', ''),
                'year': str(row.get('pubdate', '')),
                'journal': None,
                'volume': None,
                'issue': None,
                'pages': None,
                'doi': None,
                'pmc_id': pmc_id if not pd.isna(pmc_id) else None
            }
            print(f"Using basic data: {title[:50]}...")
        else:
            print(f"Fetched: {citation['title'][:50] if citation.get('title') else title[:50]}...")
        
        # Create Zotero item with tags
        zotero_item = create_zotero_item(citation, tags=tags)
        
        # Add to Zotero
        try:
            if target_collection_id:
                zot.create_items([zotero_item], target_collection_id)
            else:
                zot.create_items([zotero_item])
            success_count += 1
            print("  [OK] Added successfully")
        except Exception as e:
            error_count += 1
            print(f"  [ERROR] Failed: {str(e)[:100]}")
        
        # Rate limiting (Zotero allows 1 request per second for free tier)
        time.sleep(1.1)
        
        # Also rate limit PubMed API
        if citation and (idx + 1) % 3 == 0:
            time.sleep(0.35)
    
    print(f"\n=== Summary ===")
    print(f"Successfully added: {success_count}")
    print(f"Errors: {error_count}")
    print(f"Skipped: {skipped_count}\n")
    
    return {
        'success': success_count,
        'errors': error_count,
        'skipped': skipped_count
    }

def main():
    """Main execution"""
    print("=== Zotero Import Script ===")
    print(f"User ID: {ZOTERO_USER_ID}")
    print(f"Collection ID: {ZOTERO_COLLECTION_ID}\n")
    
    # Find all CSV files in Chapter 1
    base_dir = Path("data/chapter1")
    csv_files = list(base_dir.rglob("*.csv"))
    csv_files = [f for f in csv_files if "_with_citations.csv" not in str(f)]
    
    print(f"Found {len(csv_files)} CSV files")
    
    # Ask for confirmation
    response = input("\nDo you want to import all articles? (yes/no/test): ").lower()
    
    if response == 'no':
        print("Import cancelled.")
        return
    elif response == 'test':
        test_mode = True
        limit = 3
        print("Running in TEST MODE (3 articles per file)")
    else:
        test_mode = False
        limit = None
        print("Starting full import...")
    
    print("\nStarting import...\n")
    
    total_success = 0
    total_errors = 0
    total_skipped = 0
    
    for csv_file in csv_files:
        try:
            result = import_csv_to_zotero(
                str(csv_file),
                collection_id=ZOTERO_COLLECTION_ID,  # Parent collection
                limit=limit,
                test_mode=test_mode,
                organize=True  # Enable organization by folder structure
            )
            
            if result:
                total_success += result['success']
                total_errors += result['errors']
                total_skipped += result['skipped']
        except Exception as e:
            print(f"Error processing {csv_file}: {e}\n")
    
    print("\n=== Final Summary ===")
    print(f"Total successfully added: {total_success}")
    print(f"Total errors: {total_errors}")
    print(f"Total skipped: {total_skipped}")
    print("\nImport complete! Check your Zotero library.")

if __name__ == "__main__":
    main()
