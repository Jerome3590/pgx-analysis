# Clinical and Translational Science (CTS) — Submission Guide

**Publisher:** Wiley on behalf of ASCPT  
**ISSN:** 1752-8054 (online)  
**Chapters:** CH_1 (`ch01_cts.qmd`), CH_3 (`ch03_cts.qmd`)  
**Guide to Authors:** https://ascpt.onlinelibrary.wiley.com/page/journal/17528062/guidetoauthors

---

## Required Manuscript Structure (in order)

1. **Title Page** (page 1 — standalone, no abstract or body text)
2. **Abstract** (page 2 — structured, ≤ 250 words)
3. **Study Highlights** (What is known / What question / What does this add / How might this change practice)
4. **Introduction**
5. **Methods**
6. **Results**
7. **Discussion**
8. **Limitations**
9. **Conclusions**
10. **Acknowledgements**
11. **Author Contributions** (CRediT format)
12. **Funding**
13. **Conflicts of Interest**
14. **Data Availability**
15. **References**
16. **Tables** (each on a separate page, with caption above)
17. **Figure Legends**
18. **Figures** (each on a separate page)
19. **Supplementary Materials note** (inline at end of manuscript)

> No TOC page. CTS does not request a table-of-contents page in the manuscript file.

---

## Title Page (page 1) — Required Elements

```
Complete manuscript title

Author Names (with superscript affiliation numbers)
e.g.: R. Jerome Dixon,¹² Elvin T. Price¹³

Affiliations:
¹ Department …
² Ph.D. Program …
³ ConvergenceLabs …

Corresponding Author:
  Full name, degree/title
  Department, Institution
  Mailing address with Box number
  City, State ZIP
  Phone: (xxx) xxx-xxxx
  Email: …
  ORCID: …

Running Title: (≤ 50 characters)

Figures: N · Tables: N

Keywords: term1; term2; term3; … (3–7 terms, semicolons)
```

**What does NOT go on the title page:**  
Abstract, Study Highlights, Author Contributions, Funding, Conflicts of Interest, Data Availability.  
All of those belong at the end of the manuscript body (items 10–14 above).

---

## Format Limits

| Item | Limit |
|:-----|:------|
| Abstract | ≤ 250 words (structured) |
| Main text | ≤ 4,000 words (original article) / ≤ 8,000 words (review) — excl. abstract, refs, tables, figs |
| References | ≤ 50 (original) / ≤ 100 (review) |
| Figures | ≤ 5 (original) / ≤ 8 (review) |
| Tables | ≤ 5 (original) |
| Keywords | 3–7 terms |
| Running title | ≤ 50 characters |
| Supplementary files | ≤ 8 files (recommend combining into one PDF) |

---

## DOCX Build Notes

Add `docx: toc: false` to the chapter YAML (overrides global `_quarto.yml` which sets `toc: true`):

```yaml
format:
  wiley-njd-pdf:
    …
    toc: false
  docx:
    toc: false
```

The `move_titlepage.py` post-processor (called by `build.ps1 -Docx`) moves the
`{.content-visible when-format="docx"}` block to immediately before the abstract,
inserting a page break. This creates the correct page-1 / page-2 structure.

---

## Reporting Standards

| Article type | Required checklist |
|:-------------|:-------------------|
| Systematic review | PRISMA 2020 (upload as supplementary file) |
| Prediction model | TRIPOD (upload as supplementary file) |
| All | CRediT author contributions |

---

## Figure Format and Upload

CTS requires **both** embedded figures in the manuscript DOCX **and** separate figure file uploads in Manuscript Central.

| Item | CTS requirement | Build output |
|:-----|:----------------|:-------------|
| Main figures in DOCX | Embedded (each on a separate page) | `insert_docx_images.py` inserts PNGs at render time |
| Separate figure uploads | TIFF, ≥ 300 DPI, RGB (CMYK not required) | `export_figures_psp.py` → `output/submission/cts/chNN/figures/` |
| Supplementary figures | PNG or TIFF acceptable | `make_supp_tables.py` → `output/submission/cts/chNN/supp/` as PNG |

**Manuscript Central upload order:**
1. DOCX file → **Article File** (with embedded figures for reviewer convenience)
2. `figures/Figure_1.tiff` … `Figure_N.tiff` → **Figure 1** … **Figure N** (one file per figure)
3. `supp/Figure_S*.png` and `supp/File_S*.docx/.csv` → **Supporting Information**

> **Contrast with PSP/CPT**: those journals require figures NOT embedded in the DOCX.
> CTS allows embedded for review but still expects separate high-res files in the portal.

**Rebuild command:**
```powershell
.\build.ps1 -Submit -Chapter 1   # or -Chapter 3
```
This runs DOCX build (with embedded figures) + TIFF export in one step.

---

## Supplementary Files

- Upload separately in the submission system (not embedded in the manuscript DOCX)
- CTS accepts: DOCX, PDF, CSV, XLSX, MP4
- Name files: `File_S1`, `File_S2`, … with descriptive legends in manuscript

---

## Submission System

https://mc.manuscriptcentral.com/cts-ascpt

Cover letter should address:
- How the work advances translational science
- Statement that manuscript is not under review elsewhere
- Suggested reviewers (optional but recommended)
