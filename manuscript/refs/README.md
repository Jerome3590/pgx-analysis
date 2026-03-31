# References — `.bib` files, Zotero workflow, and Quarto/BibTeX

This folder holds **BibTeX** sources for the dissertation and journal chapter builds. Quarto YAML lists which files each chapter uses; the **Wiley / MDPI LaTeX templates** also declare `\bibliography{...}` so BibTeX can write `\bibdata` into the `.aux` file (see [Template and BibTeX behavior](#template-and-bibtex-behavior) below).

---

## Files

| File | Role |
|:-----|:-----|
| `discipline.bib` | **Core cross-chapter** references (opioid epidemiology, PGx/CPIC, SHAP/ML, APCD, shared methods). Curated and edited in-repo; citation keys follow `AuthorYYYY` (see header comment in the file). |
| `cts.bib` | Extra references for **CH_3** (Clinical and Translational Science)—often merged from Zotero or filled when CH_3-only cites are added. |
| `cpt-psp.bib` | References for **CH_2** and **CH_4** (*CPT: Pharmacometrics & Systems Pharmacology*). |
| `bmic-jpm.bib` | References for **CH_1** and **CH_5** (*Journal of Personalized Medicine* / MDPI-style chapters). |
| `zotero_utilization_density_related.bib` | Supplemental export; not wired into every chapter YAML by default—merge into `discipline.bib` or a chapter `.bib` when those items are cited. |

Chapter YAML `bibliography:` blocks (paths relative to each `CH_*` folder):

| Chapter | `bibliography` in QMD |
|:--------|:----------------------|
| 1 | `discipline.bib`, `bmic-jpm.bib` |
| 2 | `discipline.bib`, `cpt-psp.bib` |
| 3 | `discipline.bib`, `cts.bib` |
| 4 | `discipline.bib`, `cpt-psp.bib` |
| 5 | `discipline.bib`, `bmic-jpm.bib` |
| 6 | `discipline.bib`, `bmic-jpm.bib`, `cpt-psp.bib` |
| Full dissertation | `discipline.bib`, `bmic-jpm.bib`, `cpt-psp.bib`, `cts.bib` |

When you add a **new** cited work, put the BibTeX entry in the `.bib` file that chapter already loads (or in `discipline.bib` if it is shared), and keep keys stable across chapters.

---

## Zotero workflow

1. **Library** — Keep project references in Zotero collections aligned with outputs (e.g. `PGx Dissertation / CH1 JPM` for Chapter 1 work). Use consistent tagging for topic and chapter.

2. **Better BibTeX (BBT)** — Use the [Better BibTeX](https://retorque.re/zotero-better-bibtex/) plugin so citation keys are stable (`AuthorYYYY`, pin key when needed) and you can **auto-export** a `.bib` file when the library changes.

3. **Export into this repo** — Point BBT auto-export at the appropriate file under `manuscript/refs/` (e.g. `bmic-jpm.bib`, or a staging file you then merge). Alternatively export manually after large edits.

4. **Literature review / bulk import (Chapter 1)** — For PubMed-driven bulk steps and API import details, see:
   - [`CH_1/Literature_Review/scripts/README_ZOTERO_IMPORT.md`](../CH_1/Literature_Review/scripts/README_ZOTERO_IMPORT.md) (`import_to_zotero.py`, rate limits, collection IDs).
   - [`CH_1/Literature_Review/README.md`](../CH_1/Literature_Review/README.md) — PRISMA pipeline, manual PDF → Zotero, and “Step 7 — Export to Zotero → bib”.

5. **Optional R citation pass** — For some CH_1 workflows, `source("scripts/generate_citations.R")` is used to tighten metadata (see Literature Review README). Not required for every chapter build.

6. **After editing `.bib` files** — Rebuild the affected chapter(s) or `.\build.ps1 -Full` so BibTeX and LaTeX pick up new entries.

---

## Template and BibTeX behavior

### Wiley CTS (Chapter 3) fix — `\bibdata` and missing keys

**Symptom:** `bibtex` reported `I found no \bibdata command` in `ch03_cts.aux` and then **Warning--I didn't find a database entry for "..."** for every citation—even when those keys existed in `discipline.bib` and `cts.bib`.

**Cause:** `manuscript/templates/cts_template.tex` originally ended with only `$body$` and assumed references would appear inside the body. With **`cite-method: natbib`**, the Wiley flow matches **Chapter 4’s** CPT template: the template must include an explicit **`\bibliography{...}`** after `$body$` so the `.aux` file gets `\bibdata` and BibTeX can read the databases.

**What we changed:**

- **`templates/cts_template.tex`** — After `$body$`, add:
  - `\bibliography{../refs/discipline,../refs/cts}`  
  Paths are relative to the rendered `.tex` location under `CH_3/`. Do **not** duplicate `\bibliographystyle` in the template; `WileyNJDv5.cls` sets `WileyNJD-AMA` when using AMA class options.

- **`CH_3/ch03_cts.qmd`** — Under `format: wiley-njd-pdf:`, set **`pdf-engine: xelatex`** and **`cite-method: natbib`** so CH_3 matches other Wiley chapter builds and uses natbib consistently.

### Other templates (already correct)

- **`cpt_psp_template.tex`** — `\bibliography{../refs/discipline,../refs/cpt-psp}`
- **`bmic_jpm_template.tex`** — `\bibliography{../refs/discipline,../refs/bmic-jpm}`

**Rule of thumb:** The **comma-separated list in `\bibliography{...}`** must cover the same `.bib` files listed in that chapter’s YAML `bibliography:` (Quarto uses YAML for citeproc metadata; natbib/BibTeX still needs matching `\bibdata` from the template).

---

## Build verification

From `manuscript/`:

```powershell
.\build.ps1 -Chapter 3   # CTS: BibTeX should list Database file #1 and #2 with no missing-key warnings
```

A healthy BibTeX pass shows both database files and **no** `Warning--I didn't find a database entry for`.

---

## See also

- [`../README.md`](../README.md) — manuscript build commands, chapter → journal map.
- [`../FIGURES.md`](../FIGURES.md) — figures (separate from references).
