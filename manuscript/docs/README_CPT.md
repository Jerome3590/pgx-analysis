# Clinical Pharmacology & Therapeutics (CPT) — Submission Guide

**Publisher:** Wiley on behalf of ASCPT  
**ISSN:** 1532-6535 (online)  
**Chapters:** CH_5 (`ch05_cpt.qmd`)  
**Guide to Authors:** https://ascpt.onlinelibrary.wiley.com/page/journal/15326535/guidetoauthors

---

## Required Manuscript Structure (in order)

1. **Title Page** (page 1 — standalone)
2. **Abstract** (page 2 — structured, ≤ 150 words for Articles/Clinical Trials; some types have no abstract)
3. **Introduction**
4. **Methods**
5. **Results**
6. **Discussion**
7. **Acknowledgments**
8. **Author Contributions** (CRediT format)
9. **Funding**
10. **Conflicts of Interest**
11. **Data Availability**
12. **References**
13. **Tables** (each on a separate page, caption above)
14. **Figure Legends**
15. **Figures** (each on a separate page)
16. **Supplementary Materials note** (inline at end of manuscript)

> No TOC page. Any TOC graphic/text is entered in the submission system as a separate item,
> not as a page in the main manuscript file.

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

Keywords: term1; term2; term3; … (3–8 terms, semicolons)
```

**What does NOT go on the title page:**  
Abstract, Introduction, Author Contributions, Funding, Conflicts of Interest, Data Availability.  
Those belong at items 7–11 above (end of manuscript body).

---

## Format Limits

| Item | Limit |
|:-----|:------|
| Abstract | ≤ 150 words structured (Article); none for some article types |
| Main text | ≤ 4,000 words (Article — excl. title, abstract, refs, tables, figs) |
| References | ≤ 50 |
| Figures | ≤ 5; TIFF or EPS, 300 DPI minimum |
| Tables | ≤ 5 (combined figures + tables ≤ 7) |
| Keywords | 3–8 terms |
| Running title | ≤ 50 characters |
| Supplementary files | ≤ 8 files |

---

## Article Types

| Type | Abstract | Word limit | Notes |
|:-----|:---------|:-----------|:------|
| Article | 150 words structured | 4,000 | Standard research paper |
| Research Letter | None / brief intro | 1,200 | Rapid communication |
| Review | 250 words structured | 6,000 | Invited or proposal-based |
| Perspective | Unstructured | 2,500 | Opinion / commentary |
| Tutorial | Unstructured | 4,000 | Methods education |

CH_5 is a standard **Article** (clinical decision support / pharmacogenomics deployment).

---

## DOCX Build Notes

Add `docx: toc: false` to the chapter YAML (overrides global `_quarto.yml`):

```yaml
format:
  wiley-njd-pdf:
    …
    toc: false
  docx:
    toc: false
```

The `move_titlepage.py` post-processor moves the `{.content-visible when-format="docx"}`
block before the abstract with a page break, producing the correct page-1 / page-2 layout.

---

## Reporting Standards

| Article type | Required checklist |
|:-------------|:-------------------|
| Prediction model | TRIPOD (upload as supplementary file) |
| Software / CDS tool | Describe deployment architecture, latency benchmarks |
| All | CRediT author contributions |

CH_5 uses the **TRIPOD** checklist (prediction model with CDS deployment).

---

## IRB / Ethics / Privacy

CH_5 uses synthetic inputs in the deployed dashboard (no PHI transmitted).  
The underlying training cohort used Virginia APCD data under VCHI DUA.

> "No personally identifiable information is processed by the deployed dashboard.
> The underlying prediction models were trained on Virginia APCD data under a
> data use agreement with the Virginia Center for Health Innovation (VCHI).
> IRB waiver HM20022300 was granted by Virginia Commonwealth University."

---

## Supplementary Files

- Upload separately in the submission system
- CPT accepts: DOCX, PDF, CSV, XLSX, MP4
- Name files: `File_S1`, `File_S2`, …
- Include Lambda latency benchmarks and CloudWatch screenshots as supplementary figures

---

## Submission System

https://mc.manuscriptcentral.com/cpt-ascpt

Cover letter should address:
- Translational relevance (bench-to-bedside or model-to-clinic pathway)
- Open-source / reproducibility statement
- Statement that manuscript is not under review elsewhere
- Note any companion manuscripts under review at ASCPT journals (CPT:PSP)
