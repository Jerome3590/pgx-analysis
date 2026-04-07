# Manuscript Next Steps

_Last updated: 2026-04-07 (session 2) — CRediT, FP-Growth/CPIC/VIP narrative, standalone policy clean; `-Journal` flag added to build system; CTS figure export (separate TIFF uploads) implemented; all packages rebuilt._

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

---

## 🚀 Immediate Action Required

### Step 1 — Final rebuild (generates CTS `figures/` TIFFs for CH_1/CH_3)

```powershell
.\ build.ps1 -Submit -Journal cts   # CH_1 + CH_3: DOCX (embedded) + figures/*.tiff + supp/
.\ build.ps1 -Submit -Journal psp   # CH_2 + CH_4: already built; skip if no changes
```

Or individually:
```powershell
.\build.ps1 -Submit -Chapter 1 -Journal cts
.\build.ps1 -Submit -Chapter 3 -Journal cts
```

### Step 2 — Upload to journal portals

| Chapter | Journal | Package | What to upload |
|:--------|:--------|:--------|:---------------|
| CH_1 | CTS (Wiley) | `output/submission/cts/ch01/` | DOCX → Manuscript; `figures/*.tiff` → Figure 1–N; `supp/File_S*` → Supplementary |
| CH_2 | CPT:PSP (Wiley) | `output/submission/cpt_psp/ch02/` | DOCX → Manuscript; `figures/*.tiff` → Figure 1–N; `supp/` → Supplementary |
| CH_3 | CTS (Wiley) | `output/submission/cts/ch03/` | DOCX → Manuscript; `figures/*.tiff` → Figure 1–N; `supp/Figure_S*.png` → Supplementary |
| CH_4 | CPT:PSP (Wiley) | `output/submission/cpt_psp/ch04/` | DOCX → Manuscript; `figures/*.tiff` → Figure 1–N; `supp/Table_S*`, `Figure_S*` → Supplementary |

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

_Last full build: 2026-04-07 — all CH_1–5 packages verified in `output/submission/`._

---

## 🔲 Still Pending

### Portal upload — follow Steps 1 + 2 above ← BLOCKING

Run `.\build.ps1 -Submit -Journal cts` first (generates missing CTS `figures/` TIFFs for CH_1/CH_3), then upload all four packages per the table in Step 2.


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

## 🚀 Future: FDA SaMD Commercial Deployment

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
