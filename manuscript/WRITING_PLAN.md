# Train-Trip Writing Plan

Dissertation writing sessions scheduled around Amtrak legs, Mar 24–29.  
See **[DISSERTATION.md](DISSERTATION.md)** for full chapter outlines, journal targets, and submission checklists.

---

## Tue Mar 24 — HAR → PHL → WAS
*Leg 1 (Carolinian 79): HAR dep. 5:45 AM → PHL arr.*  
*Leg 2 (Regional): PHL dep. 8:58 AM → WAS arr.*

**Focus: Chapter 1 — Establishing the Gap (SQLR)**

- [ ] Draft **3–4 Key Messages** for *Journal of Personalized Medicine* (BMIC framing)
- [ ] Draft **Clinical Challenge** section (OUD in youth, polypharmacy in older adults, ED utilization)
- [ ] Draft **Technical Challenge** section (APCD/EHR complexity, warped time, XAI need)
- [ ] Write **SQLR methodology** outline (PRISMA, 5-database search, inclusion/exclusion criteria)
- [ ] Sketch **SQLR tables/figures** needed (PRISMA diagram, evidence map matrix)
- [ ] Draft **knowledge gap analysis** (absence of combined XAI + PGx models for dosing)

**Focus: Chapter 2 — Architecture Scaffolding**

- [ ] Outline **Partition-First architecture** (10-step pipeline, DuckDB, S3 checkpoints)
- [ ] Bullet **cohort design & temporal validation** (dual targets, 5:1 controls, 2016–2019 split)
- [ ] Note **target leakage prevention** rules (DTW/BupaR/FP-Growth banned from feature engineering)
- [ ] Draft **"Target Leakage Prevention"** prose for CPT:PSP submission
- [ ] Review and tune language for *CPT: Pharmacometrics & Systems Pharmacology* framing
- [ ] Create rough **figure description** for full workflow diagram

---

## Thu Mar 26 – Fri Mar 27 — RGH → CLE (overnight, ~17.5 hrs)
*Train 40 (Capitol Limited): RGH dep. 9:17 AM → CLE arr. next day*

**Focus: Chapter 2 — Full Methods Draft**

- [ ] Write **APCD input processing** section (text → Parquet, cleaning, code mapping)
- [ ] Write **event filtering & leakage removal** section (ICD/admin filters, timing rules)
- [ ] Write **cohort creation** section (eligibility criteria, QA, age bands)
- [ ] Draft **Monte Carlo CV & model stack** (CatBoost, XGBoost, XGBoost-RF; 50+ splits)
- [ ] Turn **Steps 1a–3c** into cohesive prose tied to repo directory structure
- [ ] Detail **temporal validation rules** (train 2016–2018, holdout 2019, exclude 2020)

**Focus: Chapter 3 — Opioid ED Prediction**

- [ ] Draft **cohort definition** (F11.xx, ED visit criteria, age 13–64, exclusions)
- [ ] Describe **feature space** (ICD/CPT, drug names/counts, CPIC counts, n\_event\_bin)
- [ ] Outline **per-bin modeling** strategy (low/medium/high/extreme density — 4 separate models)
- [ ] Describe planned **BupaR/DTW exploratory analyses** (visualizations only; what plots, what they show)
- [ ] Bullet **Consensus Filter** components (CatBoost SHAP ∩ FFA rules + counterfactuals)
- [ ] Tune clinical framing for *Clinical and Translational Science* (translational, TRIPOD)

**Focus: Chapter 4 — Polypharmacy & Causal Rules**

- [ ] Draft **Non-Opioid ED cohort** description (ages 65–94, HCG O11/P51, 30-day window)
- [ ] Describe **causal calculator feature space** constraint (drug names/counts + CPIC only)
- [ ] Write **FFA pipeline** narrative (pairs/triplets, Boolean logic rules, no top-K limit)
- [ ] Explain **synergy vs. antagonism** and IR score interpretation
- [ ] Note candidate **clinical vignettes** illustrating high-risk drug combinations
- [ ] Tune framing for *CPT: PSP* (systems pharmacology angle)

---

## Sun Mar 29 — CLE → WAS → RVM
*Leg 1 (Train 41): CLE dep. 1:54 AM → WAS arr.*  
*Leg 2 (Regional 99): WAS → RVM*

**Focus: Chapter 3 — Results & Interpretation**

- [ ] Summarize **model performance by bin** (metrics to report: AUROC, PR-AUC, Brier, ICI)
- [ ] Draft **key SHAP findings** for opioid cohort (top Consensus-Causal features, clinical meaning)
- [ ] Describe **trajectory archetypes** (Rapid-Onset ~4 months vs Chronic-Escalation ~22 months)
- [ ] Write **clinical interpretation** section (how trajectory-type changes intervention strategy)

**Focus: Chapter 5 — PGx Risk Dashboard**

- [ ] Outline **system architecture** (Lambda Docker/ECR, API Gateway, S3 static frontend)
- [ ] Define **ensemble risk scoring** prose (PR-AUC weighting, probability averaging, partial inputs)
- [ ] Bullet **PGx Patient Card** content (573 gene-drug pairs, CPIC matching, dosing recs, stateless)
- [ ] List **dashboard visuals** to describe (BupaR flows, DTW plots, FP-Growth networks — context only)
- [ ] Write **clinical workflow story** (step-by-step: clinician opens dashboard → risk score → PGx card)
- [ ] Add **implementation & reproducibility** details (env setup, data flow, cold-start latency)
- [ ] Draft **evaluation & deployment** section (benchmarks, monitoring, governance, HIPAA framing)
- [ ] Tune for *Journal of Personalized Medicine* (translational implementation paper framing)

**Focus: Chapter 6 — Synthesis**

- [ ] Bullet **key contributions** across all 5 manuscripts
- [ ] Synthesize findings against overarching research questions (XAI + PGx + causal = CDS)
- [ ] List **limitations** (single-state APCD, imputed PGx, cash-pay blind spot, 2020 exclusion)
- [ ] List **future work** (external validation, prospective pilot, federated learning, genotyping)
- [ ] Draft **Limitations & Future Research** section prose
- [ ] Compile **master revision checklist** across Ch1–5 (placeholders still to fill)

---

## Master Placeholder Tracker

Fill these in as pipeline outputs become available:

| Placeholder | Chapter(s) | Source |
|:------------|:-----------|:-------|
| `[CRD-XXXXXX]` | Ch1 | PROSPERO registration |
| `[IRB-XXXX]` | Ch2–Ch5 | VCU IRB waiver |
| `[N = XX,XXX]` | Ch2–Ch5 | `2_create_cohort/outputs/` |
| `[0.XX ± 0.XX]` AUROC/PR-AUC | Ch3–Ch4 | `6_final_model/outputs/` |
| `[top drug/code]` SHAP results | Ch3–Ch4 | `7_shap_analysis/outputs/` |
| `[XX synergistic pairs]` | Ch4 | `8_ffa_analysis/outputs/` |
| `< 500 ms` cold-start verified | Ch5 | Lambda CloudWatch logs |
| `../figures/chXX/fig_*.pdf` | All | `9_dashboard_visuals/` |
| `[Funding statement]` | All | Grant number + funder |
| `[Department Name]` | Ch1 affiliation | VCU department name (e.g., "Department of Health Behavior and Policy") |
| Author contributions (CRediT) | Ch1–Ch5 | Required by MDPI + Wiley — conceptualization, methodology, writing, etc. |
| Data availability statement | Ch1–Ch5 | Required by MDPI + Wiley — VCHI DUA restricts public deposit; cite DUA |
