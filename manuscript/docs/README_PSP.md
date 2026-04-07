# CPT: Pharmacometrics & Systems Pharmacology (PSP) — Submission Guide

**Publisher:** Wiley on behalf of ASCPT  
**ISSN:** 2163-8306 (online)  
**Chapters:** CH_2 (`ch02_psp.qmd`), CH_4 (`ch04_psp.qmd`)  
**Guide to Authors:** https://ascpt.onlinelibrary.wiley.com/page/journal/21638306/guidetoauthors

---

## Required Manuscript Structure (in order)

1. **Title Page** (page 1 — standalone)
2. **Abstract** (page 2 — structured or unstructured per article type, ≤ 250 words)
3. **Introduction**
4. **Methods**
5. **Results**
6. **Discussion**
7. **Study Highlights** (PSP places these after Discussion)
8. **Acknowledgments**
9. **Author Contributions** (CRediT format)
10. **Funding**
11. **Conflicts of Interest**
12. **Data Availability**
13. **References**
14. **Tables** (each on a separate page, caption above)
15. **Figure Legends**
16. **Figures** (each on a separate page)
17. **Supplementary Materials note** (inline at end of manuscript)

> No TOC page. Any TOC graphic is entered in the submission system, not as a page in the manuscript.

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

Keywords: term1; term2; term3; … (5 terms, semicolons)
```

**What does NOT go on the title page:**  
Abstract, Introduction, Author Contributions, Funding, Conflicts of Interest, Data Availability.  
Those belong at items 8–12 above (after Discussion / Study Highlights).

---

## Format Limits

| Item | Limit |
|:-----|:------|
| Abstract | ≤ 250 words (unstructured for Research Article) |
| Main text | ≤ 5,000 words (Research Article — excl. abstract, refs, legends) |
| References | ≤ 50 |
| Figures | ≤ 5; TIFF or EPS, 300 DPI minimum |
| Tables | ≤ 5 (combined figures + tables ≤ 7) |
| Keywords | 5 terms |
| Running title | ≤ 50 characters |
| Supplementary files | ≤ 8 files |

---

## Article Types

| Type | Abstract style | Notes |
|:-----|:--------------|:------|
| Research Article | Unstructured | Standard empirical paper |
| Tutorial | Unstructured | Educational methodology |
| Database | Unstructured | Data resource description |
| Review | Structured | Must justify review scope |

CH_2 and CH_4 are **Research Articles**.

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
block before the abstract with a page break.

---

## Reporting Standards

| Article type | Required checklist |
|:-------------|:-------------------|
| Prediction model | TRIPOD (upload as supplementary file) |
| Clinical trial | CONSORT |
| All | CRediT author contributions |

CH_2 and CH_4 use the **TRIPOD** checklist (prediction model studies).

---

## IRB / Ethics

CH_2 and CH_4 use Virginia APCD data under VCHI DUA.  
IRB waiver: **HM20022300** (non-human-subjects research, VCU).  
Include in the Ethics / Data Availability section:

> "Data were obtained from Virginia's All-Payer Claims Database under a data use
> agreement with the Virginia Center for Health Innovation (VCHI). IRB waiver
> HM20022300 was granted by Virginia Commonwealth University."

---

## Supplementary Files

- Upload separately in the submission system
- PSP accepts: DOCX, PDF, CSV, XLSX
- Name files: `File_S1`, `File_S2`, …

---

## Submission System

https://mc.manuscriptcentral.com/psp

Cover letter should address:
- Pharmacometric or systems pharmacology relevance
- Data availability / reproducibility
- Statement that manuscript is not under review elsewhere
