# Manuscript Next Steps

_Last updated: 2026-03-31 — `generate_figures.py` + PDF builds for CH_3/4/5 completed; CH_5 build required fixing `\(\approx\)` + bold in `{#tbl-benchmarks-cw}` paragraph (now Unicode ≈)._

---

## ✅ Completed This Session

| Item | File(s) |
|:-----|:--------|
| CH_3 ICI range corrected (0.163 → 0.164) | `CH_3/ch03_cts.qmd` lines 427, 513 |
| CH_3 abstract SHAP: `pgx_num_drugs` rank #1 / 1.22 | `CH_3/ch03_cts.qmd` line 66 |
| CH_3 Consensus-Causal section SHAP values updated | `CH_3/ch03_cts.qmd` lines 556, 576–586 |
| CH_3 oxycodone stale mean\|SHAP\| removed | `CH_3/ch03_cts.qmd` line 606 |
| CH_5 PGx coverage table added (`{#tbl-pgx-coverage}`) | `CH_5/ch05_bmic.qmd` lines 703–728 |
| METRICS.md model table: post-retrain + Brier/ICI columns | `METRICS.md` |
| METRICS.md checklist: 7 of 11 items confirmed ✅ | `METRICS.md` |
| **CH_5 Lambda benchmarks** — Synthetic targets `{#tbl-benchmarks}`; **operational CloudWatch snapshot** `{#tbl-benchmarks-cw}` + `benchmark_snapshot.json` — **verified post-deploy `2026-03-31T16:46:25Z`** (`lambda_timing*.py`; CW stats unchanged vs prior pull). | `CH_5/ch05_bmic.qmd`, `cloudwatch/*` |
| PROSPERO registration ID in CH_1 | `CH_1/ch01_bmic.qmd` — `CRD420261354089` |
| Figures + PDFs (`generate_figures.py`; `build.ps1` CH 3/4/5) | `manuscript/output/ch03_cts.pdf`, `ch04_psp.pdf`, `ch05_bmic_jpm.pdf` |

---

## ⏳ Run Now (PowerShell)

`.venv/` is **gitignored** (not in the repo clone). Create it locally at the project root if needed (see project `.cursorrules`), then activate—or call `.\.venv\Scripts\python.exe` directly so activation is optional.

```powershell
# From project root
cd C:\Projects\pgx-analysis
& .\.venv\Scripts\Activate.ps1   # omit if you use system Python or .venv\Scripts\python.exe

# Step 1 — regenerate figures for CH_3, CH_4, CH_5
# (UTF-8 avoids UnicodeEncodeError on Windows consoles when the script prints ✓)
$env:PYTHONIOENCODING = "utf-8"
python manuscript/scripts/generate_figures/generate_figures.py

# Step 2 — rebuild PDFs (changed chapters only)
cd manuscript
.\build.ps1 -Chapter 3
.\build.ps1 -Chapter 4
.\build.ps1 -Chapter 5
```

Output PDFs land in `manuscript/output/`.

_Last run: 2026-03-31 — `ch03_cts.pdf`, `ch04_psp.pdf`, `ch05_bmic_jpm.pdf` rebuilt._

---

## 🔲 Still Pending

### CloudWatch — next refresh only (after Lambda redeploy)

When you run **`prepare_models.py`** and deploy a **new** Lambda image, repeat the CLI CloudWatch pull, update **`{#tbl-benchmarks-cw}`** text/table if aggregates move, update **`cloudwatch/LAST_RUN.txt`** (new ISO time), and refresh **`benchmark_snapshot.json`** (see `cloudwatch/README.md`). **2026-03-31:** Post-deploy pull recorded in `LAST_RUN.txt`; rolling CW/Logs *n* and means matched the prior snapshot—only ECR push time was new.

### After next pipeline run (FP-Growth)
- **CH_3 FP-Growth top rule** — `opioid_ed/25-44/low` returned 0 rules this run.
  Medium bin rules are respiratory (benzonatate/azithromycin), not opioid-relevant.
  High bin target-only rules show baclofen+prednisone+lamotrigine (lift=49) but need
  clinical narrative review before inserting.
  _Action_: re-run FP-Growth with lower support threshold OR accept that
  this band's FP-Growth adds no manuscript-level rule, and rely on the
  FFA pair (Feature 4: gabapentin ⊕ alprazolam) instead.

### Author metadata (manual, all chapters)
- **CRediT author contributions** — MDPI JPM and Wiley both require this field.
  Add to each chapter's YAML front-matter.
  Template:
  ```
  R.J.D.: Conceptualization, Methodology, Software, Formal Analysis,
  Writing – Original Draft. E.T.P.: Supervision, Writing – Review & Editing.
  ```

---

## 📁 Generated Data Files (manuscript/)

| File | Contents | Used In |
|:-----|:---------|:--------|
| `data/brier_ici_results.json` | Brier + ICI per cohort/band | CH_3, CH_4 |
| `data/ffa_ie_ci.json` | IE scores + 95% CI (top 5 DDI pairs) | CH_4 |
| `data/ffa_manuscript_data.json` | FFA rules, IR scores, top drugs | CH_4 |
| `shap_top_features.json` | SHAP top-10 per cohort/band/bin | CH_3 |
| `visual_manuscript_data.json` | FP-Growth + DTW + SHAP per cohort/band/bin | reference |
| `pgx_coverage.json` | PGx feature coverage % per cohort/band | CH_5 |
| `cloudwatch/LAST_RUN.txt` + optional `*.json` / `*.log.txt` | Dated CloudWatch CLI snapshot for CH_5 benchmark table; keep until next redeploy | CH_5 (`{#tbl-benchmarks}`) |

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
