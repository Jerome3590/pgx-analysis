# Manual Review — Folder Guide

## Purpose

This folder tracks articles that require manual PDF retrieval (not available via automated PMC/OA/VCU proxy downloads).

## Workflow

```
1. Identify missing articles  →  TO_DOWNLOAD.md / TO_DOWNLOAD.csv
       ↓
2. Retrieve PDF manually (VCU EZProxy browser, Interlibrary Loan, etc.)
       ↓
3a. Attach PDF to matching Zotero item (drag-and-drop onto item in Zotero)
       OR
3b. Drop PDF into  infrastructure_setup/manual_review/  and run _parse_pdf_titles.py
       ↓
4. File is now in Zotero directory:
       C:\Users\jerom\Zotero\storage\<8-char-hash>\Author et al. - Year - Title.pdf
       ↓
5. Run pipeline step 3b to pull any new PDFs into scholar_json/:
       python scripts/_run_fulltext_pipeline.py --step 3b
```

## Zotero Storage — Verified ✅

Zotero storage is **active** at:

```
C:\Users\jerom\Zotero\storage\
```

Each attached PDF lands in a unique 8-character subfolder named by Zotero
(e.g., `22UGGEJE/Petrovitch et al. - 2024 - State program enables...pdf`).

> Files in Zotero storage ARE the canonical copies — do NOT delete or move them.
> The pipeline reads these via `_import_zotero_pdfs.py` or step 3b.

## Key Files

| File | Description |
|---|---|
| `TO_DOWNLOAD.md` | Human-readable checklist of articles still needing PDFs |
| `TO_DOWNLOAD.csv` | Machine-readable version (used by pipeline) |
| `infrastructure_setup/manual_review/article_review_checklist.csv` | Full scored checklist (gitignored — not in repo) |
| `infrastructure_setup/manual_review/REVIEW_GUIDE.md` | Scoring thresholds and classification guide (gitignored) |

## Notes

- `infrastructure_setup/manual_review/` is **gitignored** (contains large generated files).
- This `CH_1/manual_review/` directory IS tracked in git — keep only lightweight tracking files here.
- Zotero user ID: `6037399` · API key: generate at zotero.org/settings/security when ready to sync.
