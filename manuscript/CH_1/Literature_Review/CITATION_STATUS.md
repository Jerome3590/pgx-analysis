# Citation and Reference Status

## Current Status

### What We Have

Our CSV files contain **basic metadata** for all downloaded publications:

- ✅ **Title**: Full article title
- ✅ **Authors**: Last names only (comma-separated)
- ✅ **Publication Year**: Year of publication
- ✅ **PMC ID**: PubMed Central ID (when available)

**Example from current CSV:**
```csv
title,pubdate,pmc_id,authors
"A Meta-Learning-Based Ensemble Model for Explainable Alzheimer's Disease Diagnosis.",2025,PMC12248535,"Al-Bakri, Bejuri, Al-Andoli, Ikram, Khor, Tahir, The Alzheimer's Disease Neuroimaging Initiative"
```

### What We're Missing

For **complete citations**, we need:

- ❌ **Full Author Names**: First names/initials (currently only last names)
- ❌ **Journal Name**: Journal title and abbreviation
- ❌ **Volume/Issue**: Volume and issue numbers
- ❌ **Pages**: Page numbers or article numbers
- ❌ **DOI**: Digital Object Identifier
- ❌ **PubMed ID (PMID)**: PubMed identifier (different from PMC ID)
- ❌ **Formatted Citations**: APA, MLA, BibTeX formats

## Statistics

- **Total CSV files**: 44 files across Chapter 1 topics
- **Total articles**: ~23,542 articles (across all CSV files)
- **Articles with PMC IDs**: Varies by topic (some have PMC IDs, some don't)

## Solution: Generate Complete Citations

A script has been created (`scripts/generate_citations.R`) that will:

1. **Fetch complete citation data** from PubMed using:
   - PMC IDs (when available)
   - Article titles (as fallback)

2. **Extract full citation information**:
   - Full author names with initials
   - Journal name and abbreviation
   - Volume, issue, pages
   - DOI and PMID
   - Publication date (full date)

3. **Generate formatted citations**:
   - APA format
   - BibTeX format (for LaTeX/BibTeX)
   - Structured data for other formats

4. **Save enhanced CSV files**:
   - Original columns preserved
   - New citation columns added
   - Files saved as `*_with_citations.csv`

## Usage

### Generate Citations for All Articles

```r
# Run the citation generation script
source("scripts/generate_citations.R")
```

**Note**: This will take several hours due to:
- Rate limiting (3 requests/second to PubMed API)
- ~23,542 articles to process
- Estimated time: ~2-3 hours

### Generate Citations for Specific Topic

```r
# Process a specific CSV file
source("scripts/generate_citations.R")
add_citations_to_csv("data/chapter1/1.1_introduction/blackbox_cds/blackbox_cds_articles.csv")
```

## Output Format

After running the script, you'll get CSV files with additional columns:

```csv
title,pubdate,pmc_id,authors,authors_full,journal,year,doi,pmid,apa_citation,bibtex_key
"Title...",2025,PMC12345,"Last1, Last2","Last1, F., Last2, M.",Journal Name,2025,10.1234/doi,12345678,"Last1, F., Last2, M. (2025). Title... Journal Name, 10(1), 123-456. https://doi.org/10.1234/doi","last12025tit"
```

## Alternative: Export to Reference Manager

### Zotero Integration

You already have Zotero integration scripts in `testing/`. You can:

1. **Import CSV to Zotero**:
   - Use Zotero's CSV import feature
   - Map columns: title, authors, year, DOI

2. **Use PMC IDs to fetch citations**:
   - Zotero can automatically fetch citations using PMC IDs
   - Use the "Add Item by Identifier" feature

### BibTeX Export

The citation script can be modified to export BibTeX format:

```r
# Generate BibTeX file
write_bibtex(citations_df, "references.bib")
```

## Recommendations

1. **For immediate use**: Current CSV files have enough information for basic references
2. **For complete citations**: Run `generate_citations.R` script (takes 2-3 hours)
3. **For reference management**: Import to Zotero using PMC IDs
4. **For LaTeX/BibTeX**: Generate BibTeX format from complete citations

## Next Steps

1. ✅ Script created: `scripts/generate_citations.R`
2. ⏳ Run citation generation (when ready)
3. ⏳ Review sample citations for accuracy
4. ⏳ Export to desired format (BibTeX, Zotero, etc.)
