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

> Last updated: 2026-03-26 18:00 · Phase 1 ✅ · Phase 1b ✅ · Phase 2 ✅ · Phase 3 ✅ · Phase 3b ✅ · **Phase 3c ✅ 119/119 on disk** · **Phase 3d ✅ 2,735 full-text JSONs** · Phase 4 ✅ · Phase 5 ✅ · Phase 6 ✅ · **Phase 7 ✅ pytextrank scored (4,847 include)** · **Phase 7b 🔄 Google Sheets review (728 pending)** · Phase 8 ⏳

| Phase | Description | Status | Started | Completed | Notes |
|-------|-------------|--------|---------|-----------|-------|
| 1 | PubMed Searches (18 topics) | ✅ DONE | 2026-03-25 20:25 | 2026-03-25 20:29 | 6,034 raw records; see table below |
| 1b | Fallback Searches (4 topics) | ✅ DONE | 2026-03-25 21:22 | 2026-03-25 21:23 | +2,477 records; DDI+2336, Leakage+122, CPT+9, Temporal+10 |
| 2 | PMC JSON Downloads (Python) | ✅ DONE | 2026-03-25 21:30 | 2026-03-26 ~00:00 | 5,372 success · 24 errors · log: `scripts/download_log.csv` |
| 3 | OA PDF Discovery (Unpaywall) | ✅ DONE | 2026-03-25 21:32 | 2026-03-26 ~01:00 | 40 OA PDFs · 3,303 no-OA (manual needed) · log: `scripts/unpaywall_log.csv` |
| 3b | VCU Proxy PDF Download (Puppeteer) | ✅ DONE | 2026-03-26 05:00 | 2026-03-26 09:00 | 43 auto-downloaded (36.8%) · log: `scripts/vcu_download_log.csv` |
| 3c | Manual PDF Downloads (Zotero + manual_review/) | ✅ DONE | 2026-03-26 09:35 | 2026-03-26 18:00 | **119/119 on disk** (117 original + 10 user-added) · `data/scholar_pdfs/` · doi_map: `scripts/screened_doi_map.csv` |
| 3d | Full-Text JSON Extraction | ✅ DONE | 2026-03-26 18:00 | 2026-03-26 18:00 | **2,735 JSONs** in `data/scholar_json/` · 119 PDF-extracted + 2,616 PMC BioC · script: `scripts/_build_full_json.py` |
| 4 | PRISMA Stage Counts | ✅ DONE (refreshed) | 2026-03-25 21:24 | 2026-03-25 21:24 | 9,454 screened; 2,735 full-text JSONs |
| 5 | Render PRISMA Chart | ✅ DONE | 2026-03-25 21:30 | 2026-03-25 21:30 | `output/CH_1/Literature_Review/lit_review.html` |
| 6 | Ontology Organisation | ✅ DONE (refreshed) | 2026-03-25 21:24 | 2026-03-25 21:24 | 9,454 tagged; 14 nodes |
| 7 | Article Screening (pytextrank) | ✅ DONE | 2026-03-26 17:00 | 2026-03-26 17:30 | threshold 0.20 · **4,847 include / 4,607 exclude** · `data/ontology/articles_screened.csv` |
| 7b | Google Sheets Manual Review | 🔄 IN PROGRESS | 2026-03-26 18:00 | — | **4,119 pre-selected (Y)** · **728 need review** · checklist: `infrastructure_setup/manual_review/article_review_checklist.csv` |
| 8 | Zotero → bib Export | ⏳ PENDING | | | Requires Zotero API key · run `scripts/zotero_import.py --screened` after Phase 7b |

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

## File Storage & Review Strategy

> **Canonical locations — single source of truth for each artifact type.**

```
CH_1/Literature_Review/
│
├── data/
│   ├── scholar_pdfs/          # 119 PDFs named {hsh_id}.pdf  ← DO NOT RENAME
│   ├── scholar_json/          # 2,735 full-text JSONs (unified schema)  ← PRIMARY ANALYSIS INPUT
│   ├── chapter1/              # PMC BioC source JSONs (read-only, used by _build_full_json.py)
│   ├── ontology/              # PMC BioC + articles_screened.csv
│   └── other_chapters/        # PMC BioC (other dissertation chapters)
│
├── infrastructure_setup/
│   └── manual_review/         # REVIEW HUB
│       ├── article_review_checklist.csv  ← Google Sheets sync point (4,847 rows)
│       ├── screened_doi_map.csv          ← copy of canonical (scripts/ is master)
│       └── REVIEW_GUIDE.md
│
└── scripts/
    ├── screened_doi_map.csv   # CANONICAL doi_map (119 articles with hsh_ids + DOIs)
    └── vcu_download_log.csv   # Download audit trail
```

### PRISMA Flow (current)

| Stage | N |
|-------|---|
| Records identified (18 searches + fallback) | 9,571 |
| Duplicates removed | 151 |
| **Screened** | **9,454** |
| Excluded at title/abstract (pytextrank < 0.20) | 4,607 |
| **Passed screening (human_decision=include)** | **4,847** |
| Pre-selected Y (top 85% combined score) | 4,119 |
| Pending manual review (bottom 15%) | 728 |
| **Full-text retrieved (PDFs on disk)** | **119** |
| Full-text JSON available (PDF + PMC) | 2,735 |

### Review Workflow

1. Upload `infrastructure_setup/manual_review/article_review_checklist.csv` → Google Sheets
2. Filter `selected` = blank → review 728 rows, set Y or N
3. Export CSV → replace `article_review_checklist.csv`
4. `python scripts/_apply_checklist_decisions.py` → updates `articles_screened.csv`
5. `python scripts/zotero_import.py --screened` → bulk Zotero import (needs API key)

---

## Phase 2 — Download Open-Access Full Text (Python · `lit_review.qmd`)

```python
# Each download chunk calls download_pmc_article() for every PMC ID in results
# JSONs saved to: data/chapter1/<topic>/pubmed_json_files/<pmc_id>.json
```

> **Result:** 5,372 PMC BioC JSONs downloaded successfully.

---

## Phase 3 — PDF Acquisition Pipeline

> All extraction is done locally. S3 / AWS Textract approach replaced.

| Step | Tool | Output |
|------|------|--------|
| 3a | Unpaywall API (`_download_oa_urls.py`) | 40 OA PDFs |
| 3b | VCU EZProxy + Puppeteer (`scholar_lookup.py`) | 43 proxy PDFs |
| 3c | Zotero manual assign + `_import_zotero_pdfs.py` | 36 Zotero PDFs |
| 3c+ | User-added papers via `_add_new_papers.py` | 10 additional |
| **Total** | | **119 PDFs** in `data/scholar_pdfs/` |

---

## Phase 3d — Full-Text JSON Extraction + Programmatic VCU Download

> **Single idempotent master command** — safe to re-run at any time.  
> Each step checks existing files/logs before doing any work.

```bash
# Full automated run (steps 1 → 5, skips 3e which requires Duo):
python scripts/_run_fulltext_pipeline.py

# Resume from any step:
python scripts/_run_fulltext_pipeline.py --step 3c    # start at DOI lookup
python scripts/_run_fulltext_pipeline.py --step 3e    # VCU proxy only (needs Duo)
python scripts/_run_fulltext_pipeline.py --step 4     # rescore only
```

### Pipeline Steps

| Step | Script | Description | Auth? | Idempotent? |
|------|--------|-------------|-------|-------------|
| 1  | `_build_full_json.py --pdfs` | Extract text from `scholar_pdfs/` → `scholar_json/` | — | ✅ skip-existing |
| 2  | `_build_full_json.py --pmc`  | Parse local PMC BioC JSONs → `scholar_json/` | — | ✅ skip-existing |
| 3  | `_fetch_missing_fulltext.py` | Fetch real PMC IDs from PMC OA API | — | ✅ log-based |
| 3c | `_build_vcu_doi_map.py` | NCBI ESummary + CrossRef DOI lookup → `vcu_queue_with_dois.csv` | — | ✅ append-only |
| 3d | `scholar_lookup.py --vcu-queue` | Free OA: EuropePMC / CORE / Semantic Scholar | — | ✅ log-based |
| 3e | `vcu_download.js --input vcu_queue_with_dois.csv` | VCU EZProxy Puppeteer download → `scholar_pdfs/` | Duo 2FA | ✅ log-based |
| 3f | `_build_full_json.py --pdfs` | Extract PDFs added by step 3e → `scholar_json/` | — | ✅ skip-existing |
| 3b | `_import_vcu_pdfs.py` | Import any PDFs manually placed in `vcu_downloads/` | — | ✅ skip-existing |
| 4  | `_phase7_review.py --write` | Re-score all articles with pytextrank (full text) | — | ✅ rewrites decisions |
| 5  | `_build_review_checklist.py` | Rebuild Google Sheets checklist | — | ✅ rebuilds fresh |

### Programmatic VCU Download (steps 3c–3e)

**Why significant:** `vcu_download.js` already has full VCU CAS+Duo 2FA auth, EZProxy subdomain routing, and publisher PDF detection (Wiley, Elsevier, BMJ, etc.). We just needed to feed it the extended queue with resolved DOIs.

```bash
# Step 3c — Resolve DOIs for all 2,229 missing articles (~17 min, no auth)
# NCBI ESummary for 457 real PMC IDs + CrossRef/EPMC for 1,772 title-only
python scripts/_build_vcu_doi_map.py
#    → scripts/vcu_queue_with_dois.csv

# Step 3d — Free OA pass first (no authentication needed, ~hours)
python scripts/scholar_lookup.py --vcu-queue --source epmc
python scripts/scholar_lookup.py --vcu-queue --source core
python scripts/scholar_lookup.py --vcu-queue --source ss
#    → data/scholar_json/{id}.json  (directly, no PDF step)

# Step 3e — VCU proxy for remaining paywalled articles
# Requires: node + secrets/secrets.txt (username/password) + Duo push
node scripts/vcu_download.js --input scripts/vcu_queue_with_dois.csv
#    → data/scholar_pdfs/{pmc_id}.pdf

# Step 3f — Extract text from new PDFs
python scripts/_build_full_json.py --pdfs --skip-existing
```

### VCU Queue Breakdown (2,229 included articles as of 2026-03-26)

| Route | Count | Notes |
|-------|-------|-------|
| NCBI DOI lookup (PMC IDs) | 457 | ~100% resolution rate confirmed |
| CrossRef/EPMC DOI lookup (title) | 1,772 | ~80–90% resolution expected |
| Free OA (step 3d catches before proxy) | est. 20–30% | EuropePMC, CORE, SemanticScholar |
| VCU proxy required (paywalled) | est. 70–80% | `vcu_download.js` handles |

### Current Full-Text Coverage (2026-03-26)

| Stage | N |
|-------|---|
| Total screened | 9,454 |
| `scholar_json/` files | 2,861 |
| Coverage | 29.0% |
| Included with full text | 2,745 |
| Included missing full text (VCU queue) | 2,229 |
| DOIs resolved so far (`vcu_queue_with_dois.csv`) | in progress |

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

## Phase 3b — VCU Library Proxy PDF Downloads

> Scripts: `manuscript/infrastructure_setup/web_scraping/` (gitignored — local only)  
> Input: `scripts/screened_doi_map.csv` (117 articles with DOIs)  
> Output: `data/scholar_pdfs/{hsh_id}.pdf`  
> Log: `scripts/vcu_download_log.csv`

### Authentication flow (what worked)

- **VCU CAS login** at `login.vcu.edu` (selectors: `#username`, `#password`, `#submitBtn`)
- **Duo Universal Prompt** (v4) — push notification OR `--duo-passcode=XXXXXX` CLI arg
- **Cookies saved** via CDP `Network.getAllCookies` → `secrets/session_cookies.json` — no repeat Duo after first login
- **EZProxy direct URL** format: `https://publisher-host.proxy.library.vcu.edu/path` (NOT `proxy.../login?url=`)
- Requires `--ignore-certificate-errors` in Puppeteer (EZProxy wildcard cert)

### Download results by publisher (pass 5 final, 2026-03-26)

#### ✅ Paywall publishers — working via VCU proxy

| Publisher | Downloaded | no_pdf | error | Notes |
|-----------|-----------|--------|-------|-------|
| Taylor & Francis | **15** | 0 | 2 | 88% · `/doi/full/` → `/doi/pdf/` |
| Wiley (all subdomains) | **11** | 0 | 2 | 85% · `/doi/` → `/doi/pdfdirect/` |
| **SAGE** | **5** | 3 | 0 | Fixed pass 5: stealth plugin + `domcontentloaded` |
| J-STAGE | **4** | 0 | 0 | 100% · open access |
| Wolters Kluwer (LWW) | **2** | 4 | 1 | Some articles HTML-only |
| Springer Nature | **2** | 0 | 0 | 100% |
| AHA Journals | **1** | 0 | 0 | 100% |
| Health Affairs | **1** | 0 | 0 | 100% |
| De Gruyter | **1** | 0 | 0 | 100% |
| The Psychiatrist | **1** | 0 | 0 | 100% |

#### ❌ Paywall publishers — PDF blocked / requires manual download or ILL

| Publisher | ok | no_pdf | Reason | Action |
|-----------|----|--------|--------|--------|
| **Elsevier (ScienceDirect)** | 0 | **~43** | JS-gated PDF renderer; `pdfft` URL blocked via proxy | Manual / ILL |
| Oxford (OUP) | 0 | **~7** | Article page loads; PDF behind JS paywall button | Manual / ILL |
| APA PsycNET | 0 | **~3** | doiLanding redirect; no direct PDF link | Manual / ILL |
| IEEE Xplore | 0 | **2** | Login wall persists through proxy | Manual / ILL |
| Journal of Opioid Mgmt | 0 | **2** | OJS — no direct PDF URL | Manual / ILL |
| Project MUSE | 0 | **1** | Verify redirect loop | Manual / ILL |
| Thieme | 0 | **1** | Abstract-only page | Manual / ILL |

#### 📊 Pass 5 Summary

```
Total articles with DOI : 117
Downloaded auto (ok)    : 43  (36.8%)  ← VCU proxy (passes 1–5)
Downloaded free (OA)    :  2  (+1.7%)  ← Unpaywall / Europe PMC
no_pdf / blocked        : ~64          ← Elsevier ~43, OUP ~7, APA ~3, others

Key fixes across passes:
  Pass 1: resolveDoi() + --ignore-certificate-errors + linkinghub rewrite
  Pass 2: isLoginPage() VCU-only + miss_pmc_id unique per DOI
  Pass 3: guessPdfUrl regex [.-] for proxied hostnames
  Pass 4: 45s → 90s timeout + bioRxiv .full.pdf pattern
  Pass 5: puppeteer-extra-plugin-stealth + waitUntil:domcontentloaded (SAGE fixed)
```

---

## Phase 3c — Manual PDF Downloads

> **Current status: 100/117 on disk · 17 remaining**  
> Drop zone: `C:\Projects\pgx-analysis\manuscript\infrastructure_setup\manual_review\`  
> Checklist: `manual_review/TO_DOWNLOAD.md` (updated — sorted by publisher)  
> Scripts: `_import_zotero_pdfs.py`, `_parse_pdf_titles.py`, `_reconcile_pdfs.py`

### How 100/117 was reached

| Method | PDFs | Script |
|--------|------|--------|
| VCU Puppeteer proxy (passes 1–5) | 43 | `vcu_download.js` |
| Zotero "Find Available PDF" (OA) | 4 | `_import_zotero_pdfs.py` |
| Direct OA URL download (Unpaywall/Europe PMC) | 4 | `_download_oa_urls.py` |
| Zotero storage reconcile (DOI + title match) | 46 | `_reconcile_pdfs.py` |
| PDF title parsing from manual_review/ | 3 | `_parse_pdf_titles.py` |
| **Total** | **100** | |

> **Key insight:** Zotero's "Find Available PDF" silently downloaded 50+ PDFs into  
> `C:\Users\jerom\Zotero\storage\` while the proxy was auto-associating publishers.  
> `_reconcile_pdfs.py` harvested these by DOI + fuzzy title matching against the doi map.

### Workflow A — Assign PDF in Zotero desktop (recommended for remaining 17)

1. Open Zotero → collection **"PGx - Needs PDF"** (key `GW8MHKW2`)
2. Open `manual_review/TO_DOWNLOAD.md` — each entry has a **Proxy URL**
3. Click Proxy URL → PDF opens via VCU EZProxy (authenticate with Duo once)
4. Save PDF anywhere on disk
5. Drag the saved PDF onto the matching Zotero item to attach it
6. Repeat for all 17 articles
7. When done — close Zotero, then run:

```powershell
python scripts/_import_zotero_pdfs.py
# Harvests new Zotero attachments → data/scholar_pdfs/{hsh_id}.pdf
# Updates vcu_download_log.csv with status=zotero
```

### Workflow B — Drop in manual_review/ and auto-match by PDF title

```powershell
# Save PDFs (any filename) to:
# C:\Projects\pgx-analysis\manuscript\infrastructure_setup\manual_review\

# Run title-based matcher (uses pdfminer + pytextrank for key-phrase extraction):
python scripts/_parse_pdf_titles.py

# OR run full reconcile (DOI match + fuzzy title + brute-force filename scan):
python scripts/_reconcile_pdfs.py

# Regenerate checklist after import:
python scripts/_gen_checklist_by_publisher.py
```

### Matching logic

| Priority | Method | Script | Notes |
|----------|--------|--------|-------|
| 1 | DOI exact match | `_reconcile_pdfs.py` | Most reliable |
| 2 | Zotero item title fuzzy (≥4 tokens) | `_reconcile_pdfs.py` | Handles renamed files |
| 3 | PDF metadata `/Title` field | `_parse_pdf_titles.py` | Fast, works for publisher PDFs |
| 4 | First-page text + pytextrank phrases | `_parse_pdf_titles.py` | Fallback for OCR-only PDFs |
| 5 | Brute-force filename scan | `_reconcile_pdfs.py` | Last resort |

### Remaining 17 articles (as of 2026-03-26 14:00)

See `manual_review/TO_DOWNLOAD.md` for full list with proxy URLs.

| Publisher | Count | Note |
|-----------|-------|------|
| Elsevier (ScienceDirect) | 8 | Log into ScienceDirect via proxy |
| Lippincott (LWW) | 5 | Log into journals.lww.com via proxy |
| OUP | 1 | academic.oup.com via proxy |
| Taylor & Francis | 1 | tandfonline.com via proxy |
| Other | 2 | Various |

### Status tracker

```powershell
# Count PDFs on disk
(Get-ChildItem data\scholar_pdfs\*.pdf).Count

# Full status by source
python -c "
import csv
from collections import Counter
rows = list(csv.DictReader(open('scripts/vcu_download_log.csv')))
latest = {}
for r in rows:
    if r['hsh_id'] not in latest or r['timestamp'] > latest[r['hsh_id']]['timestamp']:
        latest[r['hsh_id']] = r
print(Counter(r['status'] for r in latest.values()))
"
```

---

## Full Run Order (Updated)

```
Phase 1   ✅  run_all_searches.R          → data/chapter1/*/  (18 CSVs)
Phase 1b  ✅  run_fallback_searches.R     → +2,477 records
Phase 2   ✅  download_pmc_articles.py   → {topic}/pubmed_json_files/*.json
Phase 3        unpaywall_lookup.py        → HSH-stub OA PDFs via CrossRef+Unpaywall
               ⚠️  no_oa_pdf rows → manual via VCU library
Phase 4   ✅  prisma_tracker.R            → scripts/prisma_counts.csv
Phase 5   ✅  quarto render --no-execute  → output/CH_1/Literature_Review/lit_review.html
Phase 6   ✅  organize_by_ontology.R      → data/ontology/ (9,454 tagged)
Phase 7        screen_articles.py         → data/ontology/articles_screened.csv
               ⚠️  fill human_decision column (298 pre-screened include)
               python scripts/screen_articles.py --threshold 0.06  ← re-run after Phase 2
Phase 8        ⚠️  zotero_import.py --screened  → Zotero Web API → BibTeX export
               Zotero data: C:\Users\jerom\Zotero  (997 journal articles already)
               Needs: ZOTERO_API_KEY + ZOTERO_USER_ID  (see Phase 8 section below)

```

---

---

## Phase 8 — Zotero Import & BibTeX Export

> Zotero data dir : `C:\Users\jerom\Zotero`  
> Existing library : **997 journal articles** · 5,425 total items  
> Script : `scripts/zotero_import.py`  
> Prerequisite : human review of `articles_screened.csv` (Phase 7) — currently pending

### Option A — Web API (recommended, Zotero can stay open)

**One-time setup:**
1. Go to [zotero.org/settings/security](https://www.zotero.org/settings/security)
2. Create a new private key → Personal Library → Read/Write
3. Your numeric user ID: **6037399** (already confirmed)

```powershell
# Credentials are read automatically from secrets/secrets.txt
# Make sure zotero_api_key=YOUR_KEY is set there first, then:

# Dry run (validates without posting)
python scripts/zotero_import.py --screened --dry-run

# Full import (only articles where human_decision == include)
python scripts/zotero_import.py --screened
```

**After import in Zotero desktop:**
1. Select all newly imported items
2. Right-click → **Retrieve Metadata** (fetches DOI/journal/volume/ISSN from CrossRef)
3. File → Export Library → **Better BibTeX** → `refs/bmic-jpm.bib`

### Option B — Add downloaded PDFs directly (no API key needed)

For the 43+ PDFs already in `data/scholar_pdfs/`, drag them into Zotero desktop manually
or use Zotero's "Add by Identifier" with DOIs from `scripts/screened_doi_map.csv`.

### Credentials in secrets/secrets.txt (already partially set)

```
# Zotero Web API
zotero_api_key=YOUR_KEY_HERE    ← replace with real key from zotero.org/settings/security
zotero_user_id=6037399          ← already set
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
