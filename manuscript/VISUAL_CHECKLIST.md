# Manuscript visual checklist

Inventory of **figures** and **tables** per chapter: source type (static PDF in `figures/`, embedded code, or inline Markdown), and **status** against the current `manuscript/figures/` tree (checked 2026-03-29).

**Legend**

| Status | Meaning |
|--------|---------|
| OK | Asset present or defined inline / generated in-document |
| Missing file | `![](../figures/...)` path in QMD but no matching PDF in repo |
| Build-time | Produced when rendering (TikZ/Mermaid/R chunk); not a static PDF |
| Conditional | R/other code includes graphics only if paths exist |

---

## Chapter 1 — `CH_1/ch01_bmic.qmd` (SQLR / BMIC–JPM)

| ID | Type | Source | Status | Notes |
|----|------|--------|--------|--------|
| @fig-ontology | Figure | Embedded `{tikz}` block | Build-time | OODA pipeline diagram; not a file under `figures/ch01/` |
| @fig-ooda-wordcloud | Figure | `figures/ch01/fig_ooda_wordcloud.pdf` | OK | |
| @fig-prisma | Figure | `figures/ch01/fig_prisma.pdf` | OK | |
| @fig-ml-methods | Figure | `figures/ch01/fig_ml_methods.pdf` (+ `.png` copy) | OK | |
| @fig-op-metrics | Figure | `figures/ch01/fig_op_metrics.pdf` (+ `.png` copy) | OK | |
| @tbl-study-chars | Table | Inline pipe table | OK | |
| @tbl-topic-trends | Table | Inline pipe table | OK | |
| @tbl-gaps | Table | Inline pipe table | OK | |

**Gaps / actions**

- None for static PDFs; ensure Quarto/TikZ toolchain renders @fig-ontology on CI/local PDF builds.

---

## Chapter 1 — `CH_1/Literature_Review/lit_review.qmd` (appendix-style review)

| ID | Type | Source | Status | Notes |
|----|------|--------|--------|--------|
| @fig-prisma | Figure | `{mermaid}` flowchart + `prisma_counts.rds` | Build-time | Distinct from CH1 PDF PRISMA; label overlaps CH1 naming—confirm intended |
| (unnamed chunks) | Figure | `knitr::include_graphics(here("data","wordclouds",...))` | Conditional / **missing** | `manuscript/data/wordclouds/` not present in workspace—figures skip unless generated |
| — | Image | `images/zotero.png` | **Missing file** (as of scan) | Referenced in `lit_review.qmd`; add `CH_1/Literature_Review/images/zotero.png` or remove/replace |

**Gaps / actions**

- Run `scripts/generate_wordclouds.py` (or supply `data/wordclouds/`) before rendering if word-cloud figures should appear.
- Resolve duplicate conceptual role of PRISMA: static PDF in CH1 vs Mermaid in lit review.

---

## Chapter 2 — `CH_2/ch02_psp.qmd` (architecture / CPT:PSP)

| ID | Type | Source | Status | Notes |
|----|------|--------|--------|--------|
| @fig-framework | Figure | `figures/ch02/pgx_architecture1.pdf` | OK | |
| @fig-architecture | Figure | `figures/ch02/pgx_architecture2.pdf` | OK | |
| @fig-dashboard | Figure | `figures/ch02/pgx_dashboard.pdf` | OK | |
| @fig-insights | Figure | `figures/ch02/pgx_architecture3.pdf` | OK | |
| @fig-attrition | Figure | `figures/ch02/fig_attrition.pdf` | OK | TikZ source: `figures/ch02/fig_attrition.tikz` |
| @fig-consensus | Figure | `figures/ch02/fig_consensus.pdf` | Build | TikZ: model tournament → SHAP∩FFA; compile `fig_consensus_standalone.tex` with `pdflatex -jobname=fig_consensus` in `figures/ch02/` |
| @tbl-throughput | Table | Inline pipe table | OK | |
| @tbl-validation | Table | Inline pipe table | OK | |

**Gaps / actions**

1. **@fig-consensus** — compile `fig_consensus_standalone.tex` → `fig_consensus.pdf` (SHAP ∩ FFA diagram).

---

## Chapter 3 — `CH_3/ch03_cts.qmd` (opioid ED / CTS)

| ID | Type | Source | Status | Notes |
|----|------|--------|--------|--------|
| @fig-attrition | Figure | `figures/ch03/fig_attrition.pdf` | OK | |
| @fig-curves | Figure | `figures/ch03/fig_curves.pdf` | OK | |
| @fig-shap | Figure | `figures/ch03/fig_shap.pdf` | OK | |
| @fig-trajectories | Figure | `figures/ch03/fig_trajectories.pdf` | OK | |
| @tbl-cohort-chars | Table | Inline pipe table | OK | |
| @tbl-performance | Table | Inline pipe table | OK | |

**Gaps / actions**

- None for file-based figures.

---

## Chapter 4 — `CH_4/ch04_psp.qmd` (polypharmacy / FFA calculator)

| ID | Type | Source | Status | Notes |
|----|------|--------|--------|--------|
| @fig-network | Figure | `figures/ch04/fig_network.pdf` | OK | |
| @fig-ir | Figure | `figures/ch04/fig_ir.pdf` | OK | |
| @fig-zcode | Figure | `figures/ch04/fig_zcode.pdf` | OK | |
| @tbl-cohort-chars | Table | Inline pipe table | OK | |
| @tbl-ch4-perf | Table | Inline pipe table | OK | |
| @tbl-ddi | Table | Inline pipe table | OK | Text references Supplementary Table S1 for full pair list |
| @tbl-ir | Table | Inline pipe table | OK | |

**Gaps / actions**

- Optional: confirm supplementary material file for “115 pairs” if journal requires upload beyond in-text pointer.

---

## Chapter 5 — `CH_5/ch05_bmic.qmd` (dashboard / BMIC–JPM)

| ID | Type | Source | Status | Notes |
|----|------|--------|--------|--------|
| @fig-architecture | Figure | `figures/ch05/fig_architecture.pdf` | OK | |
| @fig-imputation | Figure | `figures/ch05/fig_imputation.pdf` | OK | |
| @fig-dashboard | Figure | `figures/ch05/fig_dashboard.pdf` | OK | |
| @fig-latency | Figure | `figures/ch05/fig_latency.pdf` | OK | |
| @tbl-architecture | Table | Inline pipe table | OK | |
| @tbl-sizing | Table | Inline pipe table | OK | |
| @tbl-pgx-card | Table | Inline pipe table | OK | |
| @tbl-benchmarks | Table | Inline pipe table | OK | |

**Gaps / actions**

- None for file-based figures.

---

## Chapter 6 — `CH_6/ch06_conclusion.qmd`

| ID | Type | Source | Status | Notes |
|----|------|--------|--------|--------|
| @tbl-chapter-summary | Table | Inline pipe table | OK | |
| @tbl-performance | Table | Inline pipe table | OK | Cross-chapter performance summary; referenced from prose |

**Gaps / actions**

- **Dashboard figure plate:** canonical Risk Assessment screenshot is **CH5 only** (`@fig-dashboard` → `figures/ch05/fig_dashboard.pdf`). CH6 references Chapter 5 textually (OODA / tabs / PGx outcomes); no duplicate figure.
- OODA / research-question narrative is prose-first; pipeline viz bundles live under `10_risk_dashboard/visualizations/` and are documented in `10_risk_dashboard/docs/RESEARCH_QUESTIONS_ARTIFACTS.md`.

---

## Full dissertation — `full_dissertation/full_dissertation.qmd`

| ID | Type | Source | Status | Notes |
|----|------|--------|--------|--------|
| (dynamic) | Figure | R helper maps child chunk labels to `_files/figure-pdf/` | Build-time | Depends on included chapters and Knitr figure labels |

**Gaps / actions**

- Run full render to verify cross-chapter figure paths and duplicate labels across included QMDs.

---

## Summary: items needing attention

| Priority | Item | Action |
|----------|------|--------|
| **High** | CH2 @fig-consensus | Run `pdflatex -jobname=fig_consensus fig_consensus_standalone.tex` in `figures/ch02/` before full PDF builds |
| **Medium** | lit review word clouds | Populate `data/wordclouds/` or accept empty optional chunks |
| **Medium** | lit review Zotero screenshot | Add `CH_1/Literature_Review/images/zotero.png` or fix link |
| **Low** | PRISMA duplication | Align CH1 static figure vs lit review Mermaid diagram for submission rules |

---

## Quick path reference (expected PDFs)

```
figures/ch01/fig_ooda_wordcloud.pdf, fig_prisma.pdf, fig_ml_methods.pdf, fig_op_metrics.pdf
figures/ch02/pgx_architecture{1,2,3}.pdf, pgx_dashboard.pdf, fig_attrition.pdf, fig_consensus.pdf  ← last missing
figures/ch03/fig_{attrition,curves,shap,trajectories}.pdf
figures/ch04/fig_{network,ir,zcode}.pdf
figures/ch05/fig_{architecture,imputation,dashboard,latency}.pdf
```
