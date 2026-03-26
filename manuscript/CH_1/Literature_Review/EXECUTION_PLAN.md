# Literature Review Execution Plan
# Searches → PRISMA Chart → Ontology Organisation

> **Goal:** Run all 18 PubMed searches, build the PRISMA 2020 mermaid chart,
> and organise returned JSON articles by keyword ontology.
>
> **R path:** `C:\Program Files\R\R-4.5.2\bin\Rscript.exe`  
> **Working file:** `lit_review.qmd`  
> **Source of truth (RQs):** `docs/CrossStep_Workflow/README_research_questions_mapping.md`

---

## 📊 Live Execution Status

> Last updated: 2026-03-26 03:39 · Phase 1 ✅ · Phase 1b ✅ · Phase 2 ✅ · Phase 3 ✅ · Phase 4 ✅ · Phase 5 ✅ · Phase 6 ✅ · Phase 7 ✅ (human review pending)

| Phase | Description | Status | Started | Completed | Notes |
|-------|-------------|--------|---------|-----------|-------|
| 1 | PubMed Searches (18 topics) | ✅ DONE | 2026-03-25 20:25 | 2026-03-25 20:29 | 6,034 raw records; see table below |
| 1b | Fallback Searches (4 topics) | ✅ DONE | 2026-03-25 21:22 | 2026-03-25 21:23 | +2,477 records; DDI+2336, Leakage+122, CPT+9, Temporal+10 |
| 2 | PMC JSON Downloads (Python) | ✅ DONE | 2026-03-25 21:30 | 2026-03-26 ~00:00 | 5,372 success · 24 errors · log: `scripts/download_log.csv` |
| 3 | OA PDF Discovery (Unpaywall) | ✅ DONE | 2026-03-25 21:32 | 2026-03-26 ~01:00 | 40 OA PDFs · 3,303 no-OA (manual needed) · log: `scripts/unpaywall_log.csv` |
| 4 | PRISMA Stage Counts | ✅ DONE (refreshed) | 2026-03-25 21:24 | 2026-03-25 21:24 | 9,420 screened; 708 full-text JSONs |
| 5 | Render PRISMA Chart | ✅ DONE | 2026-03-25 21:30 | 2026-03-25 21:30 | `output/CH_1/Literature_Review/lit_review.html` |
| 6 | Ontology Organisation | ✅ DONE (refreshed) | 2026-03-25 21:24 | 2026-03-25 21:24 | 9,454 tagged; 14 nodes; 701 JSONs |
| 7 | Article Screening | ⚠️ HUMAN REVIEW | 2026-03-25 21:30 | 2026-03-26 03:39 | 2,298 recommended include (24.3%, full-text); open `articles_screened.csv` |
| 8 | Zotero → bib Export | ⏳ PENDING | | | Manual — Zotero Better BibTeX |

### Search Progress (Phase 1 detail)

> Per-search status written to `scripts/search_status_log.csv` during run.

| # | Topic | OODA Phase | Status | Articles | Flag |
|---|-------|------------|--------|----------|------|
| 1 | Black-Box ML + CDS | Orient | ✅ | 26 | |
| 2 | APCD Analysis | Observe | ✅ | 626 | |
| 3 | Pharmacovigilance | Observe | ✅ | 124 | |
| 4 | Interpretability / SHAP | Orient | ✅ | 97 | |
| 5 | FP-Growth / Assoc. Rules | Orient | ✅ | 899 | |
| 6 | Process Mining / BupaR | Orient | ✅ | 621 | |
| 7 | Opioid Use Disorder | Act | ✅ | 1,607 | |
| 8 | Polypharmacy | Act | ✅ | 102 | |
| 9 | Drug-Drug Interactions | Act | ✅ | 4 | ⚠️ query too narrow — broaden ||
| 10 | CatBoost / XGBoost | Decide | ✅ | 15 | |
| 11 | Dynamic Time Warping | Orient | ✅ | 37 | |
| 12 | Temporal Causality | Orient | ✅ | 1 | ⚠️ query too narrow — broaden |
| 13 | Target Leakage Prevention | Orient | ✅ | 2 | ⚠️ query too narrow — broaden |
| 14 | DuckDB / OLAP Analytics | Observe | ✅ | 1,978 | |
| 15 | CPT Codes + Opioid Risk *(new)* | Observe | ✅ | 2 | ⚠️ query too narrow — broaden |
| 16 | Opioid ED Prediction *(new)* | Act | ✅ | 6 | |
| 17 | Polypharmacy ED / Drug Combos *(new)* | Act | ✅ | 7 | |
| 18 | Routine vs. Non-Routine Care *(new)* | Orient | ✅ | 17 | |

> ⚠️ **4 searches returned < 5 articles** (9, 12, 13, 15). Broader fallback queries recommended — see **Phase 1b** below.

### OODA Phase Article Distribution (Phase 6 — refreshed after Phase 1b)

| OODA Phase | Articles Tagged | Top Node |
|------------|----------------|----------|
| Observe | 2,749 | DuckDB/OLAP (1,939), APCD (669), Pharmacovigilance (359) |
| Orient | 1,643 | Association Rules (898), Process Mining (613), Target Leakage (124) |
| Decide | 172 | Explainable AI (161), Gradient Boosting (45) |
| Act | 4,217 | Polypharmacy ED (2,633 ↑), Opioid ED (1,674) |
| Unclassified | 673 | — |

---

## Phase 1 — PubMed Searches (R · `lit_review.qmd`)

**Prerequisite:** Open `lit_review.qmd` in RStudio or VS Code. Run chunks in this exact order.

### Step 1.1 — Run helper functions first

```r
# In lit_review.qmd: run the chunk labelled 'search-function' (~line 742)
# This defines search_pubmed_all(), download_pmc_article(), find_missing_articles()
# MUST run before any search chunk below.
```

### Step 1.2 — Run all 18 search chunks

Run each chunk in order. Each saves a CSV to `data/chapter1/.../` or `data/other_chapters/...`.

| Order | Chunk label | Topic | Aim / RQ |
|-------|-------------|-------|----------|
| 1 | `blackbox-cds-search` | Black-Box ML + CDS | N5 |
| 2 | `apcd-analysis-search` | APCD Analysis | RQ1, RQ2 |
| 3 | `pharmacovigilance-search` | Pharmacovigilance | RQ1, RQ2 |
| 4 | `interpretability-search` | SHAP / Interpretability | N5 |
| 5 | `fpgrowth-search` | FP-Growth / Association Rules | N4 |
| 6 | `process-mining-search` | Process Mining (BupaR) | N2, N3 |
| 7 | `opioid-disorder-search` | Opioid Use Disorder | RQ2 |
| 8 | `polypharmacy-search` | Polypharmacy | RQ1 |
| 9 | `ddi-search` | Drug-Drug Interactions | RQ1, N6 |
| 10 | `catboost-search` | CatBoost / XGBoost | RQ1, RQ2 |
| 11 | `dtw-search` | Dynamic Time Warping | N1 |
| 12 | `temporal-causality-search` | Temporal Causality | RQ1 |
| 13 | `target-leakage-search` | Target Leakage Prevention | RQ1, RQ2 |
| 14 | `duckdb-search` | DuckDB / OLAP Analytics | RQ1, RQ2 |
| 15 | `cpt-opioid-search` | CPT Codes + Opioid *(new)* | RQ2 / Aim 1 |
| 16 | `opioid-ed-search` | Opioid ED Visit Prediction *(new)* | RQ2 / Aim 1+2 |
| 17 | `polypharmacy-ed-search` | Drug Combinations → Polypharmacy ED *(new)* | N6 / Aim 3 |
| 18 | `routine-care-search` | Routine vs. Non-Routine Care *(new)* | N1 / Aim 4 |

> **Rate limit note:** `search_pubmed_all()` uses `Sys.sleep()` between batches.
> Expect ~2–5 min per search topic. All 18 searches ≈ 45–90 min total.

### Step 1.3 — Also run other-chapter searches (optional)

Located at the bottom of `lit_review.qmd`:
- PGx Classification Models (`pgx-search` chunk)
- Risk Models with EHR/CDS
- Risk Models with FHIR

---

## Phase 1b — Fallback Searches for Low-Yield Topics ⚠️

> Run these broader queries for the 4 searches that returned < 5 articles.
> They append to the same CSVs (or write separate `_broad` CSVs for manual merge).
> Command: `Rscript.exe scripts/run_fallback_searches.R`

| # | Original query (yield) | Broadened query | File |
|---|------------------------|-----------------|------|
| 9 | drug-drug interactions DDI synergistic (4) | `drug-drug interaction adverse drug event` | `drug_interactions_articles.csv` |
| 12 | temporal causality healthcare claims temporal windows (1) | `temporal analysis healthcare claims longitudinal` | `temporal_causality_articles.csv` |
| 13 | target leakage data leakage ML healthcare prevention (2) | `data leakage machine learning clinical prediction` | `target_leakage_articles.csv` |
| 15 | CPT procedure codes opioid risk prediction claims (2) | `opioid risk prediction administrative claims` | `cpt_opioid_articles.csv` |

```r
# Run from manuscript/CH_1/Literature_Review/
Rscript.exe scripts/run_fallback_searches.R
# Then re-run Phase 4 + 6 to refresh counts
```

---

## Phase 2 — Download Open-Access Full Text (Python · `lit_review.qmd`)

Run the Python download chunks immediately after each search group, or batch after all searches complete.

```python
# Each download chunk calls download_pmc_article() for every PMC ID in results
# JSONs saved to: data/chapter1/<topic>/pubmed_json_files/<pmc_id>.json
# Articles without PMC IDs get HSH stub — handled in Phase 3
```

> **Expected:** ~30–60% of articles will have PMC IDs with open-access JSON.
> The rest require manual Zotero download (Phase 3).

---

## Phase 3 — Manual Download for Non-OA Articles (Zotero)

```r
# Run in R console or as standalone script:
source("scripts/prisma_tracker.R")
# Generates: scripts/missing_articles_combined.csv
```

1. Open `missing_articles_combined.csv`
2. Search each title in PubMed / Google Scholar / Unpaywall
3. Download PDF → save to local folder
4. Upload PDFs to S3: `pgx-repository/projects/Lit_Review/pdf_files/`
5. Run AWS Textract chunk in `lit_review.qmd` → JSON saved alongside OA files

---

## Phase 4 — PRISMA Stage Counts

```r
# Generates PRISMA counts from all CSVs + JSON file counts
source("scripts/prisma_tracker.R")

# Outputs:
#   scripts/prisma_counts.rds      ← loaded by lit_review.qmd PRISMA chunk
#   scripts/prisma_counts.csv      ← human-readable stage table
#   scripts/prisma_rq_counts.csv   ← per-RQ article counts
#   scripts/missing_articles_combined.csv
```

Verify outputs:

```r
pc <- readRDS(here("scripts", "prisma_counts.rds"))
print(pc)
```

### Current PRISMA Counts (refreshed 2026-03-25 21:24 after Phase 1b)

| PRISMA Stage | N |
|---|---|
| Identified (18+fallback searches + 3 other-chapter) | 9,571 |
| Duplicates removed | 151 |
| **Screened (after dedup)** | **9,420** |
| Full-text retrieved (JSON) | 708 |
| Full-text not retrieved | 5,259 |
| Excluded at screen (no text) | 3,453 |
| Excluded (full-text review) | *pending Phase 7* |
| **Included in synthesis** | *pending Phase 7* |

### Per-RQ Article Counts (after dedup)

| RQ | Search Topic | N |
|----|-------------|---|
| N1 | Dynamic Time Warping | 36 |
| N2, N3 | Process Mining (BupaR) | 610 |
| N4 | FP-Growth / Association Rules | 894 |
| N5 | Black-Box ML + CDS | 26 |
| N5 | Interpretability / SHAP | 96 |
| RQ1 | Polypharmacy | 102 |
| RQ1 | Temporal Causality | 1 ⚠️ |
| RQ1, N6 | Drug-Drug Interactions | 2,340 ↑ |
| RQ1, RQ2 | APCD Analysis | 626 |
| RQ1, RQ2 | CatBoost / XGBoost | 14 |
| RQ1, RQ2 | DuckDB / OLAP Analytics | 1,933 |
| RQ1, RQ2 | PGx Classification Models | 441 |
| RQ1, RQ2 | Pharmacovigilance | 123 |
| RQ1, RQ2 | Risk Models with EHR/CDS | 372 |
| RQ1, RQ2 | Risk Models with FHIR | 130 |
| RQ1, RQ2 | Target Leakage Prevention | 124 ↑ |
| RQ2 | Opioid Use Disorder | 1,595 |

> ✅ Phase 1b complete. All fallback counts updated.

---

## Phase 5 — Render PRISMA Mermaid Chart ✅

**Status: DONE** — `output/CH_1/Literature_Review/lit_review.html`

```bash
# Re-render any time (no-execute = no API calls, layout only)
quarto render lit_review.qmd --to html --no-execute

# To re-render WITH live data (re-runs all R/Python chunks):
quarto render lit_review.qmd --to html

# Copy approved PRISMA figure:
#   figures/ch01/fig_prisma.pdf  ← export from browser after manual review
```

The chart autopopulates from `prisma_counts.rds`. If counts are stale,
re-run Phase 4 first.

---

## Phase 6 — Keyword Ontology Organisation

```r
# Tags every article with ontology nodes from keyword_ontology.yaml
# Copies JSON files into data/ontology/<domain>/<node>/ directories
source("scripts/organize_by_ontology.R")

# Outputs:
#   data/ontology/articles_tagged.csv      ← all articles + ontology columns
#   data/ontology/ontology_summary.csv     ← article count per node
#   data/ontology/ontology_index.json      ← machine-readable index
#   data/ontology/<domain>/<node>/*.json   ← organised JSON files
```

### Ontology Structure

```
data/ontology/
├── clinical_outcomes/
│   ├── opioid_ed/               ← RQ2 · Aims 1,2
│   ├── polypharmacy_ed/         ← RQ1 · Aims 2,3
│   └── pharmacovigilance/       ← RQ1,RQ2 · Aim 3
├── analytical_methods/
│   ├── gradient_boosting/       ← RQ1,RQ2 · Aims 1,2
│   ├── explainable_ai/          ← N5 · Aims 1,4
│   ├── association_rules/       ← N4 · Aims 3,4
│   ├── process_mining/          ← N2,N3 · Aim 4
│   ├── temporal_analysis/       ← RQ1,N1,N2,N3 · Aims 2,4
│   └── target_leakage/          ← RQ1,RQ2 · Aims 1,2,3
├── data_sources/
│   ├── claims_apcd/             ← RQ1,RQ2 · Aims 1,2,3
│   ├── cpt_icd_codes/           ← RQ2 · Aim 1
│   └── ehr_fhir/                ← RQ1,RQ2 · Aim 4
└── technical_infrastructure/
    ├── scalable_analytics/      ← RQ1,RQ2 · Aim 4
    └── routine_care_utilization/ ← N1 · Aim 4
```

---

## Phase 7 — Screen Articles for Inclusion

**Pre-screening is automated.** `scripts/screen_articles.py` scored all 9,454 articles
using spaCy + pytextrank + RQ keyword taxonomy → `data/ontology/articles_screened.csv`.

### Pre-screening results (2026-03-26 03:39 — threshold=0.05, with full-text PMC JSONs)

| Recommendation | N | Notes |
|---|---|---|
| include | 2,298 | composite score ≥ 0.05 (24.3%) |
| exclude | 7,156 | score < threshold or hard-exclude pattern |

> Full PMC JSON text now incorporated (+650 articles vs title-only run).

### Human review required ⚠️

Open `data/ontology/articles_screened.csv` and fill the `human_decision` column:
- `include` — meets inclusion criteria
- `exclude` — out of scope
- `maybe` — borderline; flag for second review

**Inclusion criteria:**
- Published 2021–2026 · English · peer-reviewed journal
- Addresses at least one of RQ1, RQ2, N1–N6
- Human/clinical data (exclude animal models, in vitro, pure engineering)

**Quick filter by RQ in R (start here):**
```r
screened <- read_csv(here("data/ontology", "articles_screened.csv"))

# Start with pre-screened include recommendations
screened %>% filter(include_recommended == "include") %>%
  arrange(desc(composite_score)) %>%
  select(title, ooda_phase_primary, composite_score, rq1_score, rq2_score)

# Also check high-score excludes (borderline)
screened %>% filter(include_recommended == "exclude", composite_score > 0.08)
```

### After human review → refresh PRISMA

```r
# Add Selected column to articles_screened.csv (TRUE/FALSE from human_decision)
# Then:
screened <- read_csv(here("data/ontology", "articles_screened.csv")) %>%
  mutate(Selected = human_decision == "include")
write_csv(screened, here("data/ontology", "articles_screened.csv"))
source("scripts/prisma_tracker.R")
```

---

## Phase 8 — Export to Zotero → bib

> ⚠️ **REQUIRES HUMAN** — Zotero desktop app + Better BibTeX plugin.

### Step 8.1 — Bulk-import included articles via Zotero API

```python
# After Phase 7 human review, run to push included articles to Zotero:
# python scripts/zotero_import.py   (to be built — see note below)
```

Zotero has an HTTP API (https://api.zotero.org) that can bulk-import items.
This is automatable IF you have a Zotero API key + library ID.
Set env vars and the script will import all `human_decision == "include"` rows.

### Step 8.2 — Enrich metadata (human, ~15 min)

1. Open Zotero → select imported items → right-click → **Retrieve Metadata**
   (auto-fetches DOI, journal, volume, pages via CrossRef)
2. Review any unresolved items manually

### Step 8.3 — Better BibTeX export (human, ~2 min)

1. Zotero → File → **Export Library** → Better BibTeX → `refs/bmic-jpm.bib`
2. Better BibTeX auto-updates the file on every Zotero sync

### Step 8.4 — Verify bib integrity

```r
# Check .bib loads without errors:
biblio <- RefManageR::ReadBib(here("../../refs/bmic-jpm.bib"))
cat("Entries:", length(biblio), "\n")
```

---

---

## Automation Assessment

### ✅ Fully automated

| Script | Phase | What it does |
|--------|-------|--------------|
| `scripts/run_all_searches.R` | 1 | 18 PubMed searches → CSVs |
| `scripts/run_fallback_searches.R` | 1b | 4 broader fallback queries |
| `scripts/download_pmc_articles.py` | 2 | PMC BioC JSON download for all 5,990 OA articles |
| `scripts/unpaywall_lookup.py` | 3 | CrossRef→DOI + Unpaywall OA PDF + Textract/pdfminer |
| `scripts/prisma_tracker.R` | 4 | PRISMA counts + per-RQ table |
| `scripts/organize_by_ontology.R` | 6 | Ontology tagging + OODA distribution |
| `scripts/screen_articles.py` | 7 | pytextrank + RQ keyword pre-screening |
| `quarto render --no-execute` | 5 | PRISMA chart HTML |
| `scripts/generate_wordclouds.py` | viz | 11 word cloud PNGs + PDFs in `data/wordclouds/` |
| `scripts/zotero_import.py` | 8 | Bulk Zotero API import of included articles |
| `scripts/refresh_after_download.ps1` | post-2 | Chain Phase 4+6+7+5 refresh after downloads |

### ⚠️ Requires human (cannot automate)

| Task | Why |
|------|-----|
| **Phase 3 paywalled PDFs** | Articles with `status=no_oa_pdf` in `unpaywall_log.csv` have no open-access version; retrieve via [VCU library](https://library.vcu.edu) institutional access |
| **Phase 7 final screening** | Include/exclude decisions must be human-verified for PRISMA/systematic review validity; pre-screened CSV provides a prioritised list |
| **Phase 8 Zotero metadata** | Zotero "Retrieve Metadata" for DOI/journal/volume enrichment; no API equivalent |
| **Phase 8 BibTeX export** | Better BibTeX export requires Zotero desktop GUI |
| **Figure export (PRISMA PDF)** | Export Mermaid render from browser → `figures/ch01/fig_prisma.pdf` |

### Puppeteer use cases (optional)

Puppeteer is available for:
1. **Automating Zotero "Add by Identifier"** — but Zotero API (`zotero_import.py`) is cleaner
2. **Bulk PDF download** from publisher sites using institutional VPN — feasible but fragile (dynamic JS, CAPTCHAs, ToS risk)
3. **Not recommended** for NCBI (already handled by `rentrez` + BioC API)

---

## Full Run Order (Updated)

```
Phase 1   ✅  run_all_searches.R          → data/chapter1/*/  (18 CSVs)
Phase 1b  ✅  run_fallback_searches.R     → +2,477 records
Phase 2   🔄  download_pmc_articles.py   → {topic}/pubmed_json_files/*.json
              (running in background; ~5,400 remaining @ 0.35s/req ≈ 31 min)
Phase 3        unpaywall_lookup.py        → HSH-stub OA PDFs via CrossRef+Unpaywall
               ⚠️  no_oa_pdf rows → manual via VCU library
Phase 4   ✅  prisma_tracker.R            → scripts/prisma_counts.csv
Phase 5   ✅  quarto render --no-execute  → output/CH_1/Literature_Review/lit_review.html
Phase 6   ✅  organize_by_ontology.R      → data/ontology/ (9,454 tagged)
Phase 7        screen_articles.py         → data/ontology/articles_screened.csv
               ⚠️  fill human_decision column (298 pre-screened include)
               python scripts/screen_articles.py --threshold 0.06  ← re-run after Phase 2
Phase 8        ⚠️  Zotero desktop: import → enrich metadata → BibTeX export

```

---

## Full Run Order (Quick Reference — Legacy)

```
Phase 8  →  Import to Zotero → export refs/bmic-jpm.bib
```

---

## Estimated Time

| Phase | Task | Time |
|-------|------|------|
| 1 | 18 PubMed searches | 45–90 min |
| 2 | PMC JSON downloads | 30–60 min |
| 3 | Zotero manual download | 2–4 hours |
| 4 | PRISMA tracker | < 5 min |
| 5 | Render + review chart | 10 min |
| 6 | Ontology organisation | < 5 min |
| 7 | Article screening | 3–6 hours |
| 8 | Zotero + bib export | 30 min |
| | **Total** | **~8–12 hours** |
