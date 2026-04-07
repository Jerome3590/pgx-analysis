# Manuscript Next Steps

_Last updated: 2026-04-07 (session 3) — DOCX title page / abstract structure fixed for all journals (CTS, PSP, CPT); Study Highlights placement corrected; four post-processor bugs resolved; all CH_1–5 packages rebuilt._

---

## ✅ Completed 2026-04-07

| Item | File(s) |
|:-----|:--------|
| Project structure reorganized: `cloudwatch/`, `lambda_local/`, `scripts/` → `infrastructure_setup/`; `edits/` → `output/edits/`; `bmc/` → `docs/bmc/`; JSON files → `data/` | Multiple |
| `output/final_submission/` + `output/submission/` merged → single `output/submission/` | `build.ps1`, `export_figures_psp.py`, `make_supp_tables.py`, `.gitignore` |
| Added `-Submit` flag to `build.ps1` (DOCX + TIFFs + supp in one command) | `build.ps1` |
| Added `templates/export_figures_psp.py`, `make_supp_tables.py` | `templates/` |
| CH_4 Table S1 automated: 115 DDI pairs from S3 via `extract_ffa_table_s1.py` | `data/ffa_synergy_pairs.json` |
| All CH_1–4 submission packages built and verified | `output/submission/` |
| CH_5 confirmed accepted — CPT #2026-0568, no revision needed | — |
| CRediT author contribution statements added (formal taxonomy) | `CH_1–4/*.qmd` |
| FP-Growth target-class rule (omeprazole+naproxen→HCTZ, lift=49) validated via CPIC Tier A + PharmGKB VIP; inserted into CH_3 Results + Discussion and CH_5 FP-Growth section | `CH_3/ch03_cts.qmd`, `CH_5/ch05_cpt.qmd` |
| `@Dunnenberger2015` added to `discipline.bib` (preemptive CPIC testing reference) | `refs/discipline.bib` |
| Standalone chapter policy: all `Chapter N` intra-series refs removed from CH_1–5; CH_5 lines 415/419 fixed → `(Dixon and Price, manuscripts under review)` | `CH_5/ch05_cpt.qmd` |
| CH_3 + CH_5 rebuilt with no warnings; packages updated in `output/submission/` | `output/submission/` |
| **CTS figure export added** — `export_figures_psp.py` extended to CH_1/CH_3; generates TIFF RGB 300 dpi to `output/submission/cts/chNN/figures/` alongside embedded DOCX | `templates/export_figures_psp.py` |
| **`-Journal` flag** added to `build.ps1` + `export_figures_psp.py` — filter builds/exports by `cts \| psp \| cpt \| bmc`; journal→chapter mapping; validation | `build.ps1`, `templates/export_figures_psp.py` |
| `docs/README_CTS.md` updated with Figure Format section: dual-upload requirement documented (embedded DOCX + separate TIFF per figure in Manuscript Central) | `docs/README_CTS.md` |
| All changes committed and pushed (main → 0b97281) | GitHub |

## ✅ Completed 2026-04-07 (session 3) — DOCX front matter restructuring

| Item | File(s) |
|:-----|:--------|
| **Bug: Introduction heading deleted** — removal range `(tp_end, ...)` included first H1; fixed to `(tp_end - 1, ...)` | `templates/move_titlepage.py` |
| **Bug: "TITLE PAGE" label visible** in submitted DOCX — excluded from `tp_elems` so it is removed but never re-inserted | `templates/move_titlepage.py` |
| **Bug: title page elements inserted in reverse order** — `reversed(tp_elems)` + `addprevious` stacks backwards; changed to forward iteration | `templates/move_titlepage.py` |
| **Bug: Study Highlights paragraphs bled into title page section** — `reversed(sh_elems)` placed H1 last inside the TITLE PAGE boundary; `move_titlepage.py` swept up the paragraphs; changed to forward iteration | `templates/format_psp_manuscript.py` |
| **CTS "Abstract" heading** — Quarto renders `AbstractTitle` from `abstract:` YAML; script detects it, inserts title page block before it, skips duplicate | `templates/move_titlepage.py` |
| **PSP/CPT "Abstract" heading** — when no Quarto heading present (PSP template path), script creates `Heading1 "Abstract"` | `templates/move_titlepage.py` |
| **Study Highlights relocated** for CH_2, CH_4, CH_5 — moved from near-end to just before Introduction per PSP/CPT requirement | `templates/format_psp_manuscript.py` |
| All CH_1–5 rebuilt and verified; committed dcc60b1 | GitHub |

---

## 🚀 Immediate Action Required

### Upload to journal portals — all packages are ready

| Chapter | Journal | Package | What to upload |
|:--------|:--------|:--------|:---------------|
| CH_1 | CTS (Wiley) | `output/submission/cts/ch01/` | DOCX → Manuscript; `figures/*.tiff` → Figure 1–N; `supp/File_S*` → Supplementary |
| CH_2 | CPT:PSP (Wiley) | `output/submission/cpt_psp/ch02/` | DOCX → Manuscript; `figures/*.tiff` → Figure 1–N; `supp/` → Supplementary |
| CH_3 | CTS (Wiley) | `output/submission/cts/ch03/` | DOCX → Manuscript; `figures/*.tiff` → Figure 1–N; `supp/Figure_S*.png` → Supplementary |
| CH_4 | CPT:PSP (Wiley) | `output/submission/cpt_psp/ch04/` | DOCX → Manuscript; `figures/*.tiff` → Figure 1–N; `supp/Table_S*`, `Figure_S*` → Supplementary |
| CH_5 | CPT (Wiley) | `output/submission/cpt/ch05/` | DOCX → Manuscript; `figures/*.tiff` → Figure 1–N |

Portal links: `manuscript_status.txt`

---

## ⏳ Build Commands (current)

```powershell
cd C:\Projects\pgx-analysis\manuscript

# All chapters — DOCX + TIFFs + supp → output/submission/
.\build.ps1 -Submit

# By journal (cts | psp | cpt | bmc)
.\build.ps1 -Submit -Journal cts     # CH_1 + CH_3
.\build.ps1 -Submit -Journal psp     # CH_2 + CH_4

# Single chapter
.\build.ps1 -Submit -Chapter 3 -Journal cts

# TIFF export only (all journals)
.\build.ps1 -ExportFigures
.\build.ps1 -ExportFigures -Journal cts   # CTS only

# Advisor review DOCX only → output/edits/
.\build.ps1 -Docx -Chapter 1

# Journal PDFs only
.\build.ps1
```

_Last full build: 2026-04-07 (session 3) — all CH_1–5 packages rebuilt with corrected DOCX front matter; verified in `output/submission/`._

---

## 🔲 Still Pending

### Portal upload ← BLOCKING

All five packages are structurally correct and ready. Upload each per the table above.


---

## 📁 Generated Data Files (manuscript/)

| File | Contents | Used In |
|:-----|:---------|:--------|
| `data/brier_ici_results.json` | Brier + ICI per cohort/band | CH_3, CH_4 |
| `data/ffa_ie_ci.json` | IE scores + 95% CI (top 5 DDI pairs) | CH_4 |
| `data/ffa_manuscript_data.json` | FFA rules, IR scores, top drugs | CH_4 |
| `data/shap_top_features.json` | SHAP top-10 per cohort/band/bin | CH_3 |
| `data/visual_manuscript_data.json` | FP-Growth + DTW + SHAP per cohort/band/bin | reference |
| `data/pgx_coverage.json` | PGx feature coverage % per cohort/band | CH_5 |
| `infrastructure_setup/cloudwatch/LAST_RUN.txt` + optional `*.json` / `*.log.txt` | Dated CloudWatch CLI snapshot for CH_5 benchmark table | CH_5 (`{#tbl-benchmarks}`) |

> **CloudWatch maintenance**: Only re-run after a new Lambda image is deployed (`prepare_models.py` + ECR push). Re-pull CLI metrics, update `{#tbl-benchmarks-cw}` if aggregates shift, refresh `benchmark_snapshot.json` + `LAST_RUN.txt`. _Last snapshot: 2026-03-31T16:46:25Z._

---

## � Journal Format Lessons Learned

### How Quarto + python-docx post-processing interacts with each journal

#### General — `move_titlepage.py` (all journals)

The `{.content-visible when-format="docx"}` div in the QMD provides a custom **TITLE PAGE** block containing authors+markers, affiliations, corresponding author, running title, figures/tables count, and keywords.  The script moves this block to the correct position in the rendered DOCX.

Key findings from debugging (session 3):

- **`addprevious(ref)` + forward iteration = correct order.**  
  Each call inserts the new element immediately before `ref`.  Forward iteration over `tp_elems` preserves QMD source order.  `reversed()` produces inverted output — the original code was always wrong but went unnoticed.

- **The `tp_end` boundary is the first Heading 1 after the TITLE PAGE marker.**  
  Do NOT include `tp_end` in the removal range: it is a boundary sentinel only, not part of the title page block.  The original off-by-one (`range(tp_end, ...)`) silently deleted the Introduction heading from every chapter.

- **The "TITLE PAGE" label paragraph must be excluded from `tp_elems`.**  
  It is removed from the document as part of the normal block removal but must never be re-inserted.

- **`addprevious` inserts at the XML sibling level**, so when `move_study_highlights` places the Study Highlights H1 + paragraphs before Introduction using forward iteration, the H1 lands first — which is what `move_titlepage.py` then correctly treats as the `tp_end` boundary, leaving Study Highlights content outside the swept range.

#### CTS (CH_1, CH_3) — uses `insert_docx_images.py` + `move_titlepage.py`

- Quarto renders the `abstract:` YAML key as **`AbstractTitle` + `Abstract` style paragraphs** — an "Abstract" heading is already present before the post-processor runs.
- `move_titlepage.py` detects the `AbstractTitle` paragraph by text == `"Abstract"`, sets `abstract_heading_exists = True`, inserts the title page block *before* that heading, and skips creating a second one.
- Figures are **embedded** in the DOCX (via `insert_docx_images.py`) and also exported as separate RGB TIFF 300 dpi files for Manuscript Central upload.
- No Study Highlights section required for CTS.

#### PSP (CH_2, CH_4) — uses `suppress_images_psp.lua` + `format_psp_manuscript.py` + `move_titlepage.py`

- Quarto **does not** render a visible "Abstract" heading with the PSP Wiley template; the abstract text starts directly with `**Background:**`.
- `move_titlepage.py` falls through to the `"Background"` branch, inserts the title page block before that paragraph, then **creates a new `Heading1 "Abstract"`** paragraph.
- `format_psp_manuscript.py` moves **Study Highlights** from its near-end QMD position to just before Introduction (PSP submission requirement).
- Figures are suppressed in DOCX (callout placeholders only) and exported as CMYK TIFF 300 dpi.

#### CPT (CH_5) — same post-processing pipeline as PSP

- Quarto renders an `AbstractTitle` "Abstract" heading (same as CTS), so `move_titlepage.py` inserts before it and skips the duplicate heading.
- Study Highlights relocated identically to PSP.
- Figures suppressed + CMYK TIFF 300 dpi export.
- CPT does not require Study Highlights to appear after the abstract per se, but placing them there (consistent with PSP) is acceptable and logical.

#### Canonical DOCX structure produced (all journals)

```
[Title]              ← YAML-rendered, Title style
[Author ×N]          ← YAML-rendered, Author style
[Authors+markers]    ← title page block (BodyText)
[Affiliations]
[^1^, ^2^, ^3^]
[Corresponding Author]
[Running Title]
[Figures · Tables]
[Keywords]           ← end of title page block
[page break]
[Abstract]           ← AbstractTitle (CTS/CPT: Quarto's) or Heading1 (PSP: added by script)
[abstract text]      ← Abstract style or BodyText
[YAML keywords]      ← Quarto-rendered
[Study Highlights H1]  ← PSP/CPT only; moved by format_psp_manuscript.py
[Study Highlights paragraphs]
[Introduction H1]    ← preserved; was incorrectly deleted before the tp_end fix
[main text ...]
```

---

## �🚀 Future: FDA SaMD Commercial Deployment

> **Scope:** Transitioning the `pgx-analysis` dashboard from a **research prototype** to a
> regulatory-ready **Software as a Medical Device (SaMD)** requires the following phases.
> None of these are in scope for the dissertation; document here for post-defense roadmap.

### Regulatory & Quality Assurance
- Conduct formal FDA regulatory classification analysis under **21 CFR Part 820**.
- Establish comprehensive **Quality Management System (QMS)** documentation.
- Reference: CH_5 §Discussion already flags SaMD oversight risk — cite FDA Digital Health
  Center of Excellence guidance.

### Clinical-Grade Data Parsing
- Replace consumer-grade 23andMe input with parsers for:
  - **VCF v4.3** — standard clinical genomics variant call format
  - **HL7 FHIR R4 Genomics** profiles — EHR-interoperable genomic data exchange

### Automated Guideline Updates
- Implement container-start version check comparing bundled CPIC DB snapshot hash
  against live CPIC API; issue warning if offline data is stale.
- Current CPIC snapshot: March 2026 (573 gene-drug pairs, Level A/B).

### Live PDMP Integration
- Integrate real-time **Prescription Drug Monitoring Program (PDMP)** data directly
  into the opioid risk scoring pipeline to supplement retrospective claims-based features.

### Prospective Clinical Pilot
- Move beyond retrospective holdout validation:
  - Formal prospective trial in an ED or opioid treatment program
  - **$\ge$ 200 eligible encounters** with **6-month follow-up**
  - Measure clinician acceptance, time-to-decision, and prescribing behavior impact

### Frontend & Scaling Enhancements
- **Mobile-responsive frontend** for tablet use at point of care
- **Multi-lingual card generation** for high-LEP populations
- **Federated learning framework** — multi-state model weight updates without
  pooling patient data (architecture reference: Joshi et al. 2022)
