# Zotero Import Script

## Overview

The `import_to_zotero.py` script automatically imports all publications from your CSV files to your Zotero account. It:

1. **Reads all CSV files** from `data/chapter1/`
2. **Fetches complete citations** from PubMed using PMC IDs or titles
3. **Adds items to Zotero** with full citation information
4. **Handles duplicates** by checking if items already exist
5. **Respects rate limits** for both PubMed and Zotero APIs

## Prerequisites

```bash
pip install pyzotero pandas requests
```

## Usage

### Test Mode (Recommended First)

```bash
python scripts/import_to_zotero.py
# When prompted, type "test"
```

This will import 3 articles from each CSV file to verify everything works.

### Full Import

```bash
python scripts/import_to_zotero.py
# When prompted, type "yes"
```

**Warning**: This will import ~23,542 articles and take several hours due to rate limiting.

### Import Specific File

You can modify the script to import from a specific CSV:

```python
import_csv_to_zotero(
    "data/chapter1/1.1_introduction/blackbox_cds/blackbox_cds_articles.csv",
    collection_id=ZOTERO_COLLECTION_ID,
    limit=10  # Import first 10 articles
)
```

## Features

### Automatic Citation Fetching
- Uses PMC IDs to fetch complete citations from PubMed
- Falls back to basic CSV data if PMC ID unavailable
- Extracts: title, authors, journal, year, volume, issue, pages, DOI

### Duplicate Detection
- Checks if article already exists in Zotero by title
- Skips duplicates automatically

### Rate Limiting
- Zotero API: 1 request per second (free tier limit)
- PubMed API: ~3 requests per second
- Automatic delays between requests

### Error Handling
- Continues processing even if individual articles fail
- Reports errors at the end
- Provides summary statistics

## Output

The script will:
- Add items to your Zotero collection (`LS75EWXU`)
- Include PMC ID in the "Extra" field
- Add PMC URL if available
- Include full citation information

## Collection Organization

Items are added to collection ID `LS75EWXU`. You can:
- Create separate collections for each topic
- Use Zotero tags to organize by topic
- Use Zotero's search to filter articles

## Troubleshooting

### "ModuleNotFoundError: No module named 'pyzotero'"
```bash
pip install pyzotero
```

### "Rate limit exceeded"
- The script includes rate limiting, but if you see this error:
- Increase delays in the script
- Run in smaller batches
- Wait and retry later

### "Item already exists"
- This is normal - the script skips duplicates
- Check your Zotero library to verify

### "Error fetching citation"
- Some articles may not have PMC IDs
- The script will use basic CSV data as fallback
- Check the summary for error count

## Estimated Time

- **Test mode (3 articles/file)**: ~5-10 minutes
- **Full import (~23,542 articles)**: ~6-8 hours

The time is primarily due to:
- 1 second delay per Zotero API call
- PubMed API rate limiting
- Network latency

## Next Steps

After importing:
1. **Review in Zotero**: Check that citations look correct
2. **Add tags**: Tag articles by topic (e.g., "Chapter1", "APCD", "SHAP")
3. **Download PDFs**: Use Zotero's "Find Available PDF" feature
4. **Export citations**: Export to BibTeX, RIS, or other formats as needed
