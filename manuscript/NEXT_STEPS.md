# Manuscript Next Steps

_Last updated: 2026-04-07 (session 2) — CRediT added CH_1–4; FP-Growth CPIC/VIP validated finding inserted in CH_3 + CH_5; all standalone chapter policy violations resolved; CH_3 + CH_5 rebuilt clean._

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
| All changes committed and pushed (main → 78241a3) | GitHub |

---

## 🚀 Immediate Action Required

### Upload revision packages to journal portals (CH_1–4)

| Chapter | Journal | Portal | Package location |
|:--------|:--------|:-------|:----------------|
| CH_1 | CTS (Wiley) | Link in `manuscript_status.txt` | `output/submission/cts/ch01/` |
| CH_2 | CPT:PSP (Wiley) | Link in `manuscript_status.txt` | `output/submission/cpt_psp/ch02/` |
| CH_3 | CTS (Wiley) | Link in `manuscript_status.txt` | `output/submission/cts/ch03/` |
| CH_4 | CPT:PSP (Wiley) | Link in `manuscript_status.txt` | `output/submission/cpt_psp/ch04/` |

Upload: DOCX as **Manuscript**, files in `supp/` as **Supplementary Material**, TIFFs in `figures/` as individual **Figure** files.

---

## ⏳ Build Commands (current)

```powershell
cd C:\Projects\pgx-analysis\manuscript

# Full submission package (all chapters) — DOCX + TIFFs + supp → output/submission/
.\build.ps1 -Submit

# Single chapter
.\build.ps1 -Submit -Chapter 4

# Advisor review DOCX only → output/edits/
.\build.ps1 -Docx -Chapter 1

# Journal PDFs only → output/<journal>/
.\build.ps1
```

_Last full build: 2026-04-07 — all CH_1–5 packages verified in `output/submission/`._

---

## 🔲 Still Pending

### 1. Upload revision packages to journal portals ← BLOCKING

All packages are built and ready. Upload before any other work.

| Chapter | Journal | Package |
|:--------|:--------|:--------|
| CH_1 | CTS (Wiley) | `output/submission/cts/ch01/` |
| CH_2 | CPT:PSP (Wiley) | `output/submission/cpt_psp/ch02/` |
| CH_3 | CTS (Wiley) | `output/submission/cts/ch03/` — **rebuilt today with FP-Growth + CRediT** |
| CH_4 | CPT:PSP (Wiley) | `output/submission/cpt_psp/ch04/` |

See portal links in `manuscript_status.txt`. Upload DOCX as Manuscript, `supp/` files as Supplementary Material, `figures/` TIFFs as Figure files.

### 2. CloudWatch — next refresh only (after Lambda redeploy)
When `prepare_models.py` + new Lambda image deployed: re-pull CloudWatch CLI, update `{#tbl-benchmarks-cw}`, refresh `infrastructure_setup/cloudwatch/benchmark_snapshot.json` + `LAST_RUN.txt`.
_Last snapshot: 2026-03-31T16:46:25Z — post-deploy, CW means unchanged from prior pull._

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
| `infrastructure_setup/cloudwatch/LAST_RUN.txt` + optional `*.json` / `*.log.txt` | Dated CloudWatch CLI snapshot for CH_5 benchmark table; keep until next redeploy | CH_5 (`{#tbl-benchmarks}`) |

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
