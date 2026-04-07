# Manuscript Next Steps

_Last updated: 2026-04-07 — All CH_1–4 submission packages built to `output/submission/`; CH_5 already accepted (CPT #2026-0568); project structure reorganized; CH_4 Table S1 (115 DDI pairs) automated from S3._

---

## ✅ Completed 2026-04-07

| Item | File(s) |
|:-----|:--------|
| Project structure reorganized: `cloudwatch/`, `lambda_local/`, `scripts/` → `infrastructure_setup/`; `edits/` → `output/edits/`; `bmc/` → `docs/bmc/`; JSON files → `data/` | Multiple |
| `output/final_submission/` + `output/submission/` merged → single `output/submission/` | `build.ps1`, `export_figures_psp.py`, `make_supp_tables.py`, `.gitignore` |
| Added `-Submit` flag to `build.ps1` (DOCX + TIFFs + supp in one command) | `build.ps1` |
| Fixed `[switch]$Submit` vs `$SubmitDir` variable collision | `build.ps1` line 46 |
| Added `templates/export_figures_psp.py` — PNG→TIFF (300 dpi CMYK, journal widths) | `templates/export_figures_psp.py` |
| Added `templates/make_supp_tables.py` — CH_1 S1–S5, CH_3 supp PNGs, CH_4 Table S1/S2 | `templates/make_supp_tables.py` |
| CH_4 Table S1 automated: `extract_ffa_table_s1.py` pulls 115 DDI pairs from S3 | `infrastructure_setup/scripts/extract_ffa_table_s1.py`, `data/ffa_synergy_pairs.json` |
| CH_3 supplementary figures automated (copy PNG → `output/submission/cts/ch03/supp/`) | `templates/make_supp_tables.py` |
| CH_1 standalone chapter policy enforced (`Chapter N` refs removed) | `CH_1/ch01_cts.qmd` |
| All CH_1–4 submission packages built and verified | `output/submission/` |
| CH_5 confirmed accepted — CPT #2026-0568, no revision needed | — |
| Added `SUBMISSION_BUILD.md`, `docs/` per-journal guides, `manuscript_status.txt` | `docs/`, root |
| Git commit + push (main → fb284dc → 481cd7f) | GitHub |

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

### CH_3 FP-Growth — rules exist, need clinical review before citing

`opioid_ed/25-44` data in `visual_manuscript_data.json`:

| Bin | Rules | Source |
|:----|------:|:-------|
| low | 0 | — |
| medium | 8 | `drug_name_rules.json` — **all transactions** (not target-specific) |
| high | 110 | `drug_name_rules_target_only.json` — **ED-positive class only** ✓ |
| extreme | 110 | same |

**Medium bin** (all-transaction rules): benzonatate → azithromycin, amoxicillin → ibuprofen — respiratory/antibiotic patterns, population-level noise, not ED-outcome-specific.

**High/extreme bin top rule (target-class only, lift=49):**
- Baclofen → Prednisone + Lamotrigine (and reverse)
- Omeprazole + Naproxen → Hydrochlorothiazide
- Cephalexin + Lamotrigine → Oxcarbazepine

**Clinical review — CPIC + PharmGKB VIP (dual validation):**

| Drug | CPIC guideline | CPIC tier | PharmGKB VIP gene |
|:-----|:--------------|:---------:|:------------------|
| Omeprazole | [CYP2C19 + PPIs](https://cpicpgx.org/guidelines/cpic-guideline-for-proton-pump-inhibitors-and-cyp2c19/) | **A** | [CYP2C19 VIP](https://www.pharmgkb.org/vip/PA166170264) |
| Naproxen | [CYP2C9 + NSAIDs](https://cpicpgx.org/guidelines/guideline-for-nonsteroidal-anti-inflammatory-drugs-and-cyp2c9/) | **A** | [CYP2C9 VIP](https://www.pharmgkb.org/vip/PA166170263) |
| Oxcarbazepine | [HLA-B + carbamazepine/related](https://cpicpgx.org/guidelines/cpic-guideline-for-carbamazepine-and-hla-b/) | **A** | [HLA-B VIP](https://www.pharmgkb.org/vip/PA166170267) |
| Lamotrigine | No CPIC guideline yet | — | [UGT1A4 VIP](https://www.pharmgkb.org/vip/PA166170273) — glucuronidation; clinical significance emerging |
| Baclofen | No CPIC guideline | — | No VIP summary |
| Prednisone | No CPIC guideline | — | [CYP3A4 VIP](https://www.pharmgkb.org/vip/PA166170265) — substrate |

_Action_: **Omeprazole + Naproxen → Hydrochlorothiazide (lift=49)** has the strongest dual validation: both antecedents are CPIC Tier A with PharmGKB VIP gene summaries (CYP2C19 + CYP2C9). Their co-occurrence in the ED-positive class (25–44, high-density bin) supports a pharmacogenomically actionable polypharmacy narrative in CH_3. Cite alongside the FFA pair (gabapentin ⊕ alprazolam). Confirm VIP links are current before drafting.

### CloudWatch — next refresh only (after Lambda redeploy)
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
