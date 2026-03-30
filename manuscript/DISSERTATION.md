# Dissertation: Outline, Workflow & Manuscript Guide

**Author:** R. Jerome Dixon — dixonrj@vcu.edu — ORCID: 0000-0001-8622-0597  
**Program:** PhD in Health Related Sciences (Translational Health Research)  
**Institution:** Virginia Commonwealth University, Richmond, VA 23284  
**Data:** Virginia All-Payer Claims Database (APCD) + FDA FAERS | DUA: VCHI / VHI

---

## Chapter → Journal Map

| # | Chapter | Focus | Target Journal | Publisher | Template / Format |
|:--|:--------|:------|:--------------|:----------|:-----------------|
| 1 | `CH_1/ch01_bmic.qmd` | SQLR: XAI, PGx & Opioid Risk | *Journal of Personalized Medicine* | MDPI | `bmic_jpm_template.tex` |
| 2 | `CH_2/ch02_psp.qmd` | Partition-First Architecture & Consensus Filter | *CPT: Pharmacometrics & Systems Pharmacology* | Wiley / ASCPT | `wiley-njd-pdf` extension |
| 3 | `CH_3/ch03_cts.qmd` | Opioid ED Prediction & Trajectories (ages 13–64) | *Clinical and Translational Science* | Wiley / ASCPT | `wiley-njd-pdf` extension |
| 4 | `CH_4/ch04_psp.qmd` | Polypharmacy DDI Causal Calculator (ages 65–94) | *CPT: Pharmacometrics & Systems Pharmacology* | Wiley / ASCPT | `wiley-njd-pdf` extension |
| 5 | `CH_5/ch05_bmic.qmd` | PGx Risk Dashboard (serverless, privacy-first) | *Journal of Personalized Medicine* | MDPI | `bmic_jpm_template.tex` |
| 6 | `CH_6/ch06_conclusion.qmd` | Dissertation Synthesis | *(dissertation only)* | — | plain `article` |

**Journal rationale:**
- **BMIC / JPM** (Ch1, Ch5): PGx + AI precision-medicine framing; strong interest in XAI-enabled personalized dosing and clinical decision support tools.
- **CPT:PSP** (Ch2, Ch4): Infrastructure/methodology paper for precision therapeutics; FFA/DDI systems-pharmacology framing aligns with current ML/DDI work.
- **CTS** (Ch3): Translational clinical focus; opioid ED prediction with trajectory mapping fits "from bench to bedside" scope. Backup: *CPT* if leaning on PGx/opioid management angle.

**Backup journals:**
- Ch1 backup: *Clinical Pharmacology & Therapeutics (CPT)* — strong PGx + AI interest
- Ch5 backup: *CTS* — if emphasizing implementation science / clinical workflow integration

---

## Dissertation Outline

### Phase 1 — Foundation & Methodological Architecture

#### Chapter 1: Introduction & Systematic Quantitative Literature Review (SQLR)
- **Clinical challenge:** OUD in youth and polypharmacy in older adults; ED utilization burden
- **Technical challenge:** APCD/EHR data complexity, "warped time," administrative noise; need for XAI (SHAP + FFA)
- **SQLR methodology:** PRISMA 2020 flowchart, five-database search (PubMed, Embase, Web of Science, Cochrane, IEEE), inclusion/exclusion, PROBAST quality assessment
- **Key finding:** No study combines XAI + PGx guidelines + prospective causal validation simultaneously
- **Data sources:** APCD, FDA FAERS, EHR cohorts
- **Word target:** 6,000–8,000 (JPM Review) | Figures: ≤8 | Tables: ≤6

#### Chapter 2: Methodology & Data Engineering Architecture
- **Partition-First Architecture:** Age Band × Year partitions, DuckDB workers, S3 checkpoints, 15× throughput
- **Cohort design:** Opioid ED (F11.xx) vs. Non-Opioid ED; 5:1 controls; strict temporal split (train: 2016–2018 | holdout: 2019 | excluded: 2020)
- **Leakage prevention:** DTW, BupaR, FP-Growth for *visualization only*; event filtering before cohort creation
- **Consensus Filter:** SHAP ∩ FFA — feature designated causal only when confirmed by both SHAP (distributional) and FFA (structural/Boolean)
- **Monte Carlo Cross-Validation:** 50+ random splits; CatBoost, XGBoost, XGBoost-RF ensemble
- **Word target:** 4,500–5,500 (CPT:PSP Research Article) | Figures: ≤6 | Tables: ≤4

##### Project Workflow Mapping (Steps 1–3)
| Step | Description | Repo Dir |
|:-----|:------------|:---------|
| 1a | APCD Input Processing (text → Parquet, cleaning, code mapping) | `1a_apcd_input_data/` |
| 1b | Event Filtering (ICD/admin codes, leakage removal) | `1b_apcd_event_filter/` |
| 2  | Cohort Creation (target/control, age bands, QA) | `2_create_cohort/` |
| 3a | Feature Importance Screening (MC-CV, model aggregation) | `3a_feature_importance/` |
| 3b | Feature Refinement (BupaR, clinical validation) | `3b_feature_importance_eda/` |
| 3c | Final Feature Update (leakage removal, QA) | `4_model_data/` |

---

### Phase 2 — Core Research Chapters

#### Chapter 3: Predicting Opioid-Related ED Visits & Trajectory Mapping
- **Cohort:** Opioid ED (F11.xx); ages 13–64; four age bands; Virginia APCD 2016–2019
- **Feature space:** ICD/CPT codes, drug names/counts, CPIC PGx interaction scores
- **Exploratory:** BupaR process mining, DTW clustering (visualization only — leakage prevention)
- **Modeling:** Ensemble CatBoost + XGBoost + XGBoost-RF; per-density-bin stratification
- **Causal attribution:** Consensus Filter (SHAP + FFA + counterfactuals)
- **Trajectories:** Rapid-Onset (~4 months) vs. Chronic-Escalation (~22 months) archetypes
- **Word target:** 4,000–5,000 (CTS Original Article) | Figures: ≤5 | Tables: ≤4

#### Chapter 4: Polypharmacy, Drug Interactions & Causal Rules
- **Cohort:** Non-Opioid ED (HCG O11/P51, excl. F11.xx/T40.xx); ages 65–94; 30-day causality window
- **Feature space:** Drug names/counts, CPIC PGx scores (30-day window only)
- **Causal calculator:** FFA multi-feature interaction testing — pairwise and triplet synergy/antagonism
- **IR scores:** Intervention Rate — expected risk reduction per drug removal; deprescribing priority ranking
- **Z-code analysis:** Managed polypharmacy (high Z-code proportion) is protective even at high drug counts
- **Word target:** 4,500–5,500 (CPT:PSP Research Article) | Figures: ≤5 | Tables: ≤4

##### Project Workflow Mapping (Steps 4–8)
| Step | Description | Repo Dir |
|:-----|:------------|:---------|
| 4 | Model Data Preparation (event extraction, leakage removal) | `4_model_data/` |
| 5 | PGx Feature Engineering (gene-drug interactions, CPIC) | `5_pgx_analysis/` |
| 6 | Final Model Training (XGBoost, CatBoost, Ensemble, per-bin) | `6_final_model/` |
| 7 | SHAP Analysis (global/local feature importance) | `7_shap_analysis/` |
| 8 | FFA Analysis (symbolic Boolean rules) | `8_ffa_analysis/` |

#### Chapter 5: Translation to Practice — The PGx Risk Dashboard
- **Architecture:** Hybrid serverless — S3 static frontend + AWS Lambda Docker container (ECR); 1.4 GB image with all 21 models
- **Inference:** Performance-weighted ensemble; Imputation of Normality for partial inputs
- **PGx Card:** Stateless CPIC lookup (573 gene-drug pairs); ephemeral — zero PII storage; 23andMe input format
- **What-If tab:** FFA counterfactual analysis grounded in Ch4 IR scores
- **Latency targets:** Cold-start < 500 ms; warm inference < 100 ms ✓
- **Word target:** 5,500–7,000 (JPM Article) | Figures: ≤7 | Tables: ≤5

##### Project Workflow Mapping (Steps 9–10)
| Step | Description | Repo Dir |
|:-----|:------------|:---------|
| 9  | Dashboard Visuals (BupaR, DTW, FP-Growth — context only) | `9_dashboard_visuals/` |
| 10 | Build & Deploy (Lambda, Docker/ECR, S3, API Gateway) | `10_risk_dashboard/` |

---

### Phase 3 — Synthesis

#### Chapter 6: Conclusion
- Integrates findings from all manuscript chapters
- Contributions: APCD scalability, causal interpretability, PGx-enriched risk prediction, clinical deployment
- Limitations: single-state Virginia APCD, imputed PGx features, cash-pay blind spot, 2020 exclusion
- Future work: external validation, prospective clinical pilot, federated learning, measured genotypes

---

## Figure & Data Checklist

### Chapter 1 (SQLR)
**Data to generate:**
- Final SQLR results table + CSV of included studies (ID, year, design, outcome, XAI/PGx flags)
- PRISMA counts (records identified/screened/included) in JSON/CSV for offline flow diagram
- Evidence-map matrix (study × method) for heatmaps

**Figures needed** (`figures/ch01/`):
- `fig_prisma.pdf` — PRISMA 2020 flow diagram (`prisma2020` R package or prisma-statement.org)
- `fig_ml_methods.pdf` — ML model distribution bar chart
- `fig_evidence_map.pdf` — study × method heatmap (optional)

---

### Chapter 2 (Architecture)
**Data to generate:**
- Schema snapshot (tables, columns, row counts per pipeline stage)
- Cohort attrition counts per step (initial → filtered → final train/holdout) as CSV
- Temporal distributions (counts by year/month) for train/test/excluded 2020

**Figures needed** (`figures/ch02/`):
- `fig_architecture.pdf` — end-to-end pipeline diagram (draw.io / Lucidchart)
- `fig_attrition.pdf` — CONSORT-style cohort attrition flowchart
- `fig_consensus.pdf` — Consensus Filter (SHAP ∩ FFA → Causal Features; TikZ `fig_consensus_standalone.tex`)

---

### Chapter 3 (Opioid ED)
**Data to generate:**
- Per-band metrics table: AUROC, PR-AUC, Brier, ICI per age band (from `6_final_model/outputs/`)
- SHAP outputs: global importance values + saved per-patient SHAP vectors (`7_shap_analysis/outputs/`)
- Trajectory summaries: DTW cluster assignments + compressed event sequences (`8_ffa_analysis/`)

**Figures needed** (`figures/ch03/`):
- `fig_attrition.pdf` — cohort attrition flowchart
- `fig_curves.pdf` — PR curves + calibration diagrams + ensemble weights by age band
- `fig_shap.pdf` — SHAP beeswarm plot (top 20 Consensus-Causal features, 25–44 band)
- `fig_trajectories.pdf` — DTW distance heatmap + representative event sequences

---

### Chapter 4 (Polypharmacy)
**Data to generate:**
- Rule tables from FFA: condition → outcome, support, confidence, IR score CSV
- Drug network edges from FP-Growth: (drug A, drug B, support, lift) CSV
- Synergistic pair/triplet bootstrap results CSV

**Figures needed** (`figures/ch04/`):
- `fig_network.pdf` — FP-Growth drug co-occurrence + FFA synergy overlay (Cytoscape 3.10)
- `fig_ir.pdf` — Intervention Rate rankings top-15 drugs
- `fig_zcode.pdf` — Z-code proportion vs. ADE risk (violin + OR plot)

---

### Chapter 5 (PGx Dashboard)
**Data to generate:**
- Lambda benchmark JSON: latency measurements (cold-start, warm inference, PGx card) × 1,000 requests
- Risk score breakdowns: per-component PR-AUC and ensemble weights CSV
- PGx mapping table: CPIC genotype → phenotype → dose adjustments for key genes

**Figures needed** (`figures/ch05/`):
- `fig_architecture.pdf` — system architecture (Lambda, Docker, ECR, S3, API Gateway)
- `fig_dashboard.pdf` — Tab 2 risk score display screenshot (synthetic patient)
- `fig_imputation.pdf` — Imputation of Normality sensitivity analysis
- `fig_latency.pdf` — cold-start + warm inference latency histograms

---

## Journal Submission Checklists

### Ch1 & Ch5 — MDPI Journal of Personalized Medicine
- [ ] Abstract: structured (Background/Methods/Results/Conclusions), 200–350 words
- [ ] Keywords: 5–8 terms, semicolon-separated
- [ ] Word limit: 8,000 (review), 7,000 (article) excl. references/supplementary
- [ ] Figures: ≤ 8 (review), ≤ 7 (article); all PDF/EPS at ≥ 300 DPI
- [ ] Reference style: numbered, MDPI style (author, title, journal, year, vol, pages, doi)
- [ ] Cover letter: positioned for precision medicine + clinical decision support readership
- [ ] ORCID: 0000-0001-8622-0597 — verify in submission system
- [ ] Author contributions statement: CRediT taxonomy (`R.J.D.` initials)
- [ ] Data availability statement present
- [ ] Ethics/IRB statement present (or "Not applicable" for Ch1)

### Ch2 & Ch4 — CPT: Pharmacometrics & Systems Pharmacology (Wiley)
- [ ] Abstract: unstructured, ≤ 250 words
- [ ] Keywords: 5 terms, semicolon-separated
- [ ] Word limit: 5,000 (excl. abstract, references, figure legends)
- [ ] Figures: ≤ 5; 2-column format (STIX/AMA); TIFF/EPS 300 DPI for submission
- [ ] Reference style: AMA (numbered superscript); `wileyNJD-AMA.bst` ✓
- [ ] Article type: "Research Article"
- [ ] Data availability statement + code availability
- [ ] Conflict of interest + funding statement

### Ch3 — Clinical and Translational Science (Wiley)
- [ ] Abstract: structured (Background/Methods/Results/Conclusions), ≤ 250 words
- [ ] Keywords: 3–6 terms
- [ ] Word limit: 4,000 (excl. abstract, references)
- [ ] Figures: ≤ 5; 1-column STIX format for submission
- [ ] Reference style: AMA (numbered superscript)
- [ ] Article type: "Original Article"
- [ ] TRIPOD reporting checklist in supplement
- [ ] Ethics statement / DUA statement (VCHI/VHI) present
- [ ] Lay summary (1–2 sentences for press office) — optional but recommended

---

## Repository Structure

```
pgx-analysis/                        ← git root
│
├── 1a_apcd_input_data/              ← APCD text → Parquet
├── 1b_apcd_event_filter/            ← ICD/admin event filtering
├── 2_create_cohort/                 ← cohort construction
├── 3a_feature_importance/           ← MC-CV feature screening
├── 3b_feature_importance_eda/       ← clinical feature refinement
├── 4_model_data/                    ← model-ready features
├── 5_pgx_analysis/                  ← PGx CPIC enrichment
├── 6_final_model/                   ← model training (CatBoost/XGBoost)
├── 7_shap_analysis/                 ← SHAP global/local
├── 8_ffa_analysis/                  ← FFA Boolean rules
├── 9_dashboard_visuals/             ← BupaR, DTW, FP-Growth plots
├── 10_risk_dashboard/               ← Lambda, Docker, S3 deployment
├── 11_testing/                      ← integration/smoke tests
├── py_helpers/                      ← Python utilities
├── r_helpers/                       ← R utilities
│
└── manuscript/                      ← THIS DIRECTORY
    ├── CH_1/ch01_bmic.qmd           ← SQLR → JPM
    ├── CH_2/ch02_psp.qmd            ← Architecture → CPT:PSP
    ├── CH_3/ch03_cts.qmd            ← Opioid ED → CTS
    ├── CH_4/ch04_psp.qmd            ← Polypharmacy → CPT:PSP
    ├── CH_5/ch05_bmic.qmd           ← Dashboard → JPM
    ├── CH_6/ch06_conclusion.qmd     ← Synthesis (dissertation)
    ├── templates/
    │   ├── Definitions/             ← mdpi.cls bundle (MDPI)
    │   ├── bmic_jpm_template.tex    ← Pandoc template (MDPI/JPM)
    │   ├── cpt_psp_template.tex     ← (reference copy; ch02/ch04 use wiley-njd-pdf)
    │   └── cts_template.tex         ← (reference copy; ch03 uses wiley-njd-pdf)
    ├── _extensions/ramiromagno/wiley-njd/  ← Wiley NJDv5 cls (auto-installed)
    ├── refs/
    │   ├── discipline.bib           ← core cross-chapter refs
    │   ├── bmic_jpm.bib             ← JPM-specific refs
    │   ├── cpt_psp.bib              ← CPT:PSP-specific refs
    │   └── cts.bib                  ← CTS-specific refs
    ├── figures/ch01/ … ch05/        ← PDF/PNG figures by chapter
    ├── tables/ch01/ … ch05/         ← CSV/LaTeX table sources
    ├── output/                      ← compiled PDFs
    ├── DISSERTATION.md              ← ← THIS FILE
    ├── README.md                    ← quick-start build instructions
    ├── _quarto.yml                  ← shared Quarto project config
    ├── Makefile                     ← Linux/macOS build
    └── build.ps1                    ← Windows PowerShell build
```

---

## Key Design Decisions

| Decision | Rationale |
|:---------|:----------|
| Partition-First Architecture | Linear scalability; isolates age-band/year strata for parallel compute |
| S3 checkpoints per partition | Fault-tolerant; enables mid-run resume without reprocessing |
| Per-density-bin models (`n_event_bin`) | Prevents high-utilization patients from biasing predictions for average risk patients |
| Consensus Filter (SHAP ∩ FFA) | Dual-confirmation reduces false-positive causal features vs. single-method |
| Visualization-only BupaR/FP-Growth | Prevents target leakage from trajectory/association mining into predictive features |
| Temporal validation (train 2016–2018 / hold 2019) | Mirrors real-world deployment; prevents optimistic CV-only estimates |
| Exclude 2020 entirely | COVID-19 caused non-representative utilization patterns |
| 5:1 case-control matching | Sufficient power; avoids class-weight hyperparameter sensitivity |
| Stateless Lambda + ephemeral PGx card | HIPAA-compliant CDS without dedicated PHI infrastructure |
