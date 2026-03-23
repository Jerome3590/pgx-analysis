<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# yes

Here’s a markdown checklist you can paste into Obsidian or your repo and tick off during the trip.

```markdown
# Train-Leg Writing Checklist

## Tue Mar 24 – PHL → RGH (Carolinian 79)

### Chapter 1 – Introduction & SQLR
- [ ] Draft **Clinical Challenge** (OUD youth, polypharmacy older adults, ED utilization).
- [ ] Draft **Technical Challenge** (APCD/EHR complexity, warped time, XAI need).
- [ ] Write **SQLR methodology** (PRISMA, inclusion/exclusion, databases, search terms).
- [ ] Sketch **SQLR tables/figures** (PRISMA diagram, evidence map).
- [ ] Draft **knowledge gap analysis** (lack of XAI + PGx models for dosing).

### Chapter 2 – Architecture Scaffolding
- [ ] Outline **Partition-First architecture** and 10-step pipeline.
- [ ] Bullet **cohort design & temporal validation** (dual targets, 5:1 controls, 2016–2019 split).
- [ ] Note **target leakage prevention** rules (what’s banned before cohort creation).
- [ ] Create a rough **figure description** for the full workflow diagram.

---

## Thu Mar 26–Fri Mar 27 – RGH → CLE (Train 40, overnight)

### Chapter 2 – Full Methods Draft
- [ ] Write **APCD input processing** section (text→Parquet, cleaning, mapping).
- [ ] Write **event filtering & leakage removal** section (ICD/admin filters, timing rules).
- [ ] Write **cohort creation** section (eligibility, QA, age bands).
- [ ] Draft **Monte Carlo CV & model stack** section (CatBoost, XGBoost, RF).
- [ ] Turn **Steps 1a–3c** into prose tied to repo directories.

### Chapter 3 – Predicting Opioid ED Visits
- [ ] Draft **cohort definition** (F11.20, ED visit criteria, exclusions).
- [ ] Describe **feature space** (ICD/CPT, drug names/counts, CPIC counts).
- [ ] Outline **per-bin modeling** strategy (low/med/high/extreme density).
- [ ] Describe planned **BupaR/DTW exploratory analyses** (what plots, what they show).
- [ ] Bullet **Consensus Filter** components (CatBoost SHAP, rule models, counterfactuals).

---

## Sun Mar 29 – CLE → WAS (Train 41)

### Chapter 3 – Results & Interpretation
- [ ] Summarize **model performance by bin** (metrics to report, expected patterns).
- [ ] Draft **key SHAP findings** for opioid cohort (top drivers, clinical meaning).
- [ ] Describe **trajectory patterns** from DTW/BupaR (typical vs risky paths).
- [ ] Write **clinical interpretation** section (how this changes ED risk understanding).

### Chapter 4 – Polypharmacy & Causal Rules
- [ ] Draft **Non-Opioid ED cohort** description (geriatric/high-cost criteria).
- [ ] Describe **causal calculator feature space** (drug counts, CPIC counts only).
- [ ] Write **FFA pipeline** narrative (pairs/triplets, Boolean rules).
- [ ] Explain **synergy vs antagonism** and how FP-Growth networks are interpreted.
- [ ] Note candidate **clinical vignettes** illustrating risky combinations.

### Chapter 5 – PGx Risk Dashboard
- [ ] Outline **system architecture** (Lambda, Docker, API Gateway, S3 frontend).
- [ ] Define **ensemble risk scoring** text (PR-AUC weighting, probability averaging).
- [ ] Bullet **PGx Patient Card** content (genetic markers, CPIC mapping, dosing recs).
- [ ] List **dashboard visuals** to describe (BupaR flows, DTW plots, FP-Growth graphs).

---

## Sun Mar 29 – WAS → RVM (Regional 99)

### Chapter 5 – Narrative & Integration
- [ ] Write **clinical workflow story** (how a clinician uses the dashboard step-by-step).
- [ ] Add **implementation details** important for reproducibility (env, data flow).
- [ ] Draft **evaluation & deployment considerations** (latency, monitoring, governance).

### Phase 3 – Conclusion Prep (Chapter 6)
- [ ] Bullet **key contributions** (XAI, APCD processing, PGx integration).
- [ ] List **limitations** (data, generalizability, pipeline constraints).
- [ ] Note **future work directions** (new cohorts, additional PGx, prospective trials).
- [ ] Compile a **master revision checklist** across Chapters 1–5.

```

Do you want a second version of this that’s even more compressed (e.g., 1–2 tasks per leg) for days when you know you’ll be tired?

