# Zotero Organization Structure

## Current Organization

The import script now organizes items based on your project's directory structure:

### Collection Hierarchy

All items are organized into collections matching your Chapter 1 structure:

```
Literature Review (LS75EWXU) [Parent Collection]
├── 1.1 Introduction
│   ├── 1.1 Introduction - Blackbox Cds
│   └── 1.1 Introduction - Interpretability
├── 1.2 Clinical Background
│   ├── 1.2 Clinical Background - Opioid Disorder
│   ├── 1.2 Clinical Background - Polypharmacy
│   ├── 1.2 Clinical Background - Drug Interactions
│   └── 1.2 Clinical Background - Pharmacovigilance
├── 1.3 Methodological
│   ├── 1.3 Methodological - Apcd Analysis
│   ├── 1.3 Methodological - Pattern Mining
│   │   ├── 1.3 Methodological - Pattern Mining - Fpgrowth
│   │   ├── 1.3 Methodological - Pattern Mining - Process Mining
│   │   └── 1.3 Methodological - Pattern Mining - Dtw
│   ├── 1.3 Methodological - Temporal Causality
│   └── 1.3 Methodological - Target Leakage
└── 1.4 Technical
    ├── 1.4 Technical - Catboost Xgboost
    └── 1.4 Technical - Duckdb Olap
```

### Tags

Each item is automatically tagged with:
- **Section tag**: e.g., "1.1 Introduction", "1.2 Clinical Background"
- **Topic tag**: e.g., "Blackbox Cds", "Interpretability"
- **Sub-topic tag** (if applicable): e.g., "Fpgrowth", "Process Mining"

### Benefits

1. **Easy Navigation**: Find articles by Chapter 1 section
2. **Topic Filtering**: Use tags to filter by specific topics
3. **Hierarchical Organization**: Collections match your paper structure
4. **Search Flexibility**: Search across collections or filter by tags

## Usage in Zotero

### Viewing Collections
- Open Zotero → Collections panel
- Expand "Literature Review" to see all sections
- Click a collection to see articles in that topic

### Using Tags
- Click "Tags" panel to see all tags
- Click a tag to filter items across all collections
- Use multiple tags for advanced filtering

### Searching
- Search within a collection (right-click → Search)
- Search across all collections (main search bar)
- Combine tags and search terms

## Customization

To change organization:

1. **Modify collection names**: Edit `get_collection_and_tags()` function
2. **Add custom tags**: Modify the tags list in the function
3. **Change parent collection**: Update `ZOTERO_COLLECTION_ID` in script
4. **Disable organization**: Set `organize=False` in `import_csv_to_zotero()` call

## Example: Finding Articles

### Find all APCD articles:
- Go to collection: "1.3 Methodological - Apcd Analysis"
- Or search tag: "Apcd Analysis"

### Find all interpretability articles:
- Go to collection: "1.1 Introduction - Interpretability"
- Or search tag: "Interpretability"

### Find all Chapter 1.2 articles:
- Search tag: "1.2 Clinical Background"
- Or browse collections under "1.2 Clinical Background"
