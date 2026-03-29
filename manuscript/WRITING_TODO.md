# Manuscript Writing TODO — Section Expansion Plan

**Generated:** 2026-03-29  
**Status:** All chapters short of journal word-count targets. Word counts exclude tables/figures/captions.

---

## Word Count Summary

| Chapter | Current | Target | Gap | Independent of Retrain? |
|:--------|--------:|:------:|----:|:---:|
| CH_1 (SQLR) | 3,361 | 5,500–7,000 | ~2,139 | ✅ Yes |
| CH_2 (Architecture) | 2,206 | 4,500–5,500 | ~2,294 | ✅ Yes |
| CH_3 (Opioid) | 1,532 | 4,000–5,000 | ~2,468 | 🔄 Partial |
| CH_4 (Polypharmacy) | 1,875 | 4,500–5,500 | ~2,625 | 🔄 Partial |
| CH_5 (Dashboard) | 2,248 | 5,500–7,000 | ~3,252 | 🔄 Partial |
| CH_6 (Conclusion) | 1,479 | 4,000–6,000 | ~2,521 | 🔄 Partial |

> ✅ = Can expand now (no numeric values depend on retrain)  
> 🔄 = Structure/prose can be written now; numeric values fill in post-retrain

---

## CH_1 — SQLR (Target: +2,139 words minimum)

### High Priority (expand now — no retrain dependency)

**Protocol and Registration** (29 words → target 100)
- Add PROSPERO registration number `[CRD-XXXXXX]` placeholder
- Explain non-human subjects research / IRB waiver

**Research Aims and Questions** (80 words → target 200)
- Expand RQ1 and RQ2 with specific PICO framework components
- Explain how OODA maps to the two RQs
- State primary and secondary objectives explicitly

**Eligibility Criteria** (130 words → target 300)
- Add full inclusion/exclusion criteria table or prose list
- Specify language restrictions (English only), date range (2013–2026), study designs included
- Explain why grey literature was excluded

**Study Selection** (42 words → target 200)
- Describe Covidence screening workflow: title/abstract → full text → consensus
- Report Cohen's κ at both stages with actual values (or placeholder if not yet computed)
- Describe third-reviewer arbitration criteria

**Data Extraction** (66 words → target 250)
- List all extracted data fields (sample size, outcome definition, ML algorithm, XAI method, PGx feature presence, validation strategy, leakage prevention, PROBAST domain ratings)
- Describe pilot extraction on 10% random sample for calibration

**Quality Assessment** (44 words → target 200)
- Describe PROBAST domain mapping to this review's questions
- State that all 4 domains were assessed: Participants, Predictors, Outcome, Analysis
- Note which domain drove highest risk-of-bias (Analysis: 62%)

**Evidence Synthesis** (62 words → target 200)
- Describe meta-analytic vs. narrative synthesis decision (narrative, given heterogeneous outcomes)
- Explain how frequency counts and heatmaps were generated (PubMed API + Python NLP pipeline)
- Note citation network analysis approach

**Study Characteristics** (133 words → target 400)
- Expand prose description of tbl-study-chars data: geographic distribution (USA 68%), design types (EHR 47%, claims 40%)
- Add paragraph on outcome definitions: heterogeneous (OUD diagnosis, ED visit, overdose)
- Add paragraph on algorithm variety: XGBoost/RF (58%), LSTM/RNN (21%), logistic regression (18%)
- Add sentence on follow-up periods

**NIH AI Checklist Domain Coverage** (130 words → target 300)
- Add specific domain count details (Performance Metrics n=, Explainability n=)
- Add year-over-year trend sentence (improving since 2022)
- Note which domains improved most post-2020 (Reproducibility, Fairness/Bias)

**PGx Feature Integration** (72 words → target 250)
- Expand on which PGx resources were cited (CPIC, PharmGKB, DPWG)
- Describe the proxy-genotype problem: claims-derived CYP2D6 inference vs. measured genotype
- Add sentence on CPIC evidence level distribution (Level A: n=, Level B: n=)

**Temporal Validation and Leakage** (73 words → target 250)
- Describe leakage patterns found in literature: target encoding, future-data inclusion, time-series overlap
- Distinguish prospective vs. retrospective temporal validation
- Note that only 12% of reviewed studies documented explicit leakage prevention

**Limitations of This Review** (40 words → target 300)
- Single-language restriction (English only)
- PubMed-primary search may miss engineering conferences (IEEE, ACM)
- NLP-based eligibility screening has false-negative risk (~2% estimated)
- Rapidly evolving literature: search frozen at [DATE]
- PROSPERO pre-registration constraint: protocol fixes search terms

**Operational Performance Metrics** (147 words → target 300)
- Add paragraph explaining the two operational categories: deployment architecture and clinical integration
- Discuss why Hospital/System Capacity is near-absent (methodological gap, not clinical gap)
- Mention relationship to this dissertation: Chapters 3–5 address operational deployment gaps

---

## CH_2 — Architecture (Target: +2,294 words minimum)

### High Priority (expand now)

**Study Objectives** (62 words → target 200)
- Expand each of 4 objectives with 1–2 sentences of rationale
- Add a fifth objective: demonstrate that visualization-only process mining is sufficient for clinical insight

**Data Source** (54 words → target 300)
- Describe Virginia APCD data structure: medical + pharmacy + dental claims, eligibility files
- Specify file format (fixed-width text, ~1.8 TB raw), VCHI data-use agreement, IRB waiver
- Describe Bronze zone staging: raw file sizes per year, record counts by type
- Add table or list of data elements extracted

**Cohort Construction** (139 words → target 400)
- Expand DTW Protocol Filtering methodology: what constitutes an "admin noise" template, threshold derivation
- Expand Extreme-Density Split: describe top-5% threshold derivation per age band, how these patients differ (dialysis, nursing home)
- Add sentence on exclusion of 2020 and rationale (COVID ED utilization collapse)
- Describe matching algorithm: 5:1 nearest-neighbor without replacement, matching variables

**Ensemble Modeling** (131 words → target 400)
- Expand MCCV design: 50 splits, 80/20 train/test within partition, stratified by outcome
- Add CatBoost hyperparameter ranges explored by Optuna (iterations, depth, learning rate, l2_leaf_reg)
- Add XGBoost hyperparameter ranges (n_estimators, max_depth, eta, subsample, colsample_bytree)
- Describe early stopping criterion and maximum evaluation rounds
- Explain why XGBoost-RF differs from standard XGBoost in this context (bagging without sequential boosting bias)

**Deployed Risk Dashboard** (34 words → target 200)
- Describe the four-tab interface and what each does (briefly — full detail in CH_5)
- Explain how ensemble weights are serialized and loaded at inference time
- Mention imputation of normality as the key production feature

**Cohort Characteristics** (62 words → target 250)
- Add sentence on sex distribution (cases vs. controls), comorbidity burden (Elixhauser score)
- Add opioid prescription patterns: mean prescription count, common drug classes
- Add sentence on geriatric cohort demographics for polypharmacy group

**Architecture Performance** (60 words → target 300)
- Expand sub-linear scaling explanation: S3 checkpoint I/O overhead quantified (~50 ms per commit)
- Add sentence on total pipeline wall-clock time: 3 complete runs, average runtime per partition
- Add worker failure simulation result: checkpoint recovery in < 60 seconds

**Feature Selection** (42 words → target 250)
- Add sentence on bottom-quartile features that were dropped: what types dominated (administrative codes, rare procedure codes)
- Add sentence on PGx enrichment coverage: how many drugs had CPIC Level A/B evidence in cohort
- Describe final feature counts per cohort: 498 opioid / 89 polypharmacy — explain the 5.6× difference

**Consensus Filter Results** (71 words → target 300)
- Add specific Brier score values: Consensus < SHAP-only < FFA-only < All-features (reference @fig-consensus)
- Describe false-positive categories eliminated by FFA confirmation (administrative artifacts, rare codes)
- Describe FFA-only features dropped (low-prevalence drug combinations)
- Add sentence on inter-partition stability: Consensus feature overlap across age bands

**Temporal Validation Summary** (101 words → target 300)
- Add per-cohort commentary: opioid ED outperforms polypharmacy in lower age bands; polypharmacy outperforms in geriatric bands
- Explain why LogLoss is higher for younger opioid bands (lower case prevalence)
- Note 2019 holdout is strictly temporal (no shuffling, no data from 2019 used in training)

**Discussion** (112 words → target 500)
- Add paragraph comparing to distributed alternatives: Apache Spark (shuffle bottleneck), AWS EMR (straggler risk), Dask (Python-only constraint)
- Add paragraph on generalizability: partition-first architecture is data-source agnostic; applicable to Medicare, Medicaid, any state APCD
- Add paragraph on Consensus Filter vs. existing feature selection: compare to LASSO, recursive feature elimination, SHAP-only approaches

**Limitations** (45 words → target 200)
- Expand single-state limitation (one sentence per limitation)
- Add: no cost-effectiveness analysis of infrastructure investment
- Add: DuckDB version pinning creates upgrade dependency risk
- Add: Optuna hyperparameter search not fully reproducible without fixed random seed

---

## CH_3 — Opioid ED (Target: +2,468 words minimum)

### High Priority (expand now — methods/discussion are retrain-independent)

**Study Design and Data Source** (48 words → target 200)
- State this is a retrospective cohort study with temporal validation
- Specify IRB waiver protocol number (HM20022300)
- Describe Virginia APCD years, claim types, and record volumes relevant to this cohort
- State primary exposure window: 12-month lookback from index ED visit

**Methods — Cohort Construction** (147 words → target 400)
- Expand opioid ED case definition: list specific ICD-10-CM codes (F11.xx) used across all 10 diagnosis columns
- Describe matching protocol fully: 5:1 nearest-neighbor, matching on age band, sex, year; without replacement
- Add exclusion criteria: patients <13 years, patients with no pharmacy claims, patients appearing in both cohorts
- State how index date was defined for controls (random date within matching year)

**Methods — Feature Engineering** (119 words → target 350)
- Describe item_ binary feature generation: NDC → RxNorm drug code → binary count ≥1
- Describe ICD-10 diagnosis code binarization (one-hot per 3-character code)
- Describe CPT procedure code binarization
- State that drug count window is 365 days before index date
- Describe n_event_bin_ordinal derivation: quartile thresholds per cohort/age band from training data

**Methods — Ensemble and Consensus Filter** (45 words → target 200)
- Cross-reference CH_2 Section 4 (Methods) for full details
- Specify which model was used for FFA rule extraction (XGBoost)
- State Consensus threshold: SHAP ≥ 75th pct AND FFA support ≥ 0.05

**Methods — Trajectory Analysis** (84 words → target 300)
- Describe DTW distance metric (Sakoe-Chiba band width = 3 time steps)
- Describe event sequence encoding: drug events as drug class codes, diagnoses as ICD chapter codes, procedures as CPT category codes; each as time-indexed vector
- Describe elbow criterion implementation: range k=2 to k=8; within-cluster sum of DTW distances
- State that clustering used k-medoids (PAM algorithm) rather than k-means (DTW not Euclidean)

**Methods — Statistical Analysis** (59 words → target 200)
- State significance level (α = 0.05)
- Describe bootstrap CI computation for SHAP values (1,000 resamples)
- Describe DTW cluster stability: silhouette score computed for k=2 through k=8
- State software stack: Python 3.11, XGBoost 2.0, CatBoost 1.2, dtaidistance 2.3, SHAP 0.43

**Results — Cohort Characteristics** (72 words → target 300)
- Expand with sex distribution, comorbidity burden, mean drug count by age band
- Add sentence on most common comorbidities in cases (chronic pain, anxiety, depression)
- Add sentence on most common opioid classes (oxycodone, hydrocodone, tramadol) in cases vs. controls

**Results — Consensus-Causal Features** (115 words → target 350)
- Expand top feature discussion beyond top 5 — mention top opioid drug classes, comorbidity codes, CPT procedure codes
- Add sentence on PGx features: CYP2D6 interaction score rank among top-10
- Add sentence on what SHAP-only features were removed by FFA confirmation (administrative artifacts)
- Mention n_event_bin_ordinal rank and interpretation (density stratum as proxy for healthcare utilization severity)

**Results — Trajectory Analysis** (135 words → target 350)
- Expand Rapid-Onset pathway: specific drug sequences, injury ICD codes present
- Expand Chronic-Escalation pathway: number of non-opioid intervention windows, what interventions appeared
- Add silhouette score value confirming k=2 separation
- Add sentence on DTW cluster validation: internal (silhouette) and external (case rate by cluster)

**Discussion — Modifiable Causal Drivers** (83 words → target 300)
- Expand opioid prescribing + chronic pain + no PT finding: cite prior literature on non-pharmacological pain management efficacy
- Discuss CYP2D6 finding: what it means clinically (poor metabolizers at higher risk of dose escalation)
- Compare to prior opioid prediction models in literature (CH_1 finding: most lacked PGx features)

**Discussion — Trajectory-Informed Intervention** (47 words → target 250)
- Expand on the 4.2-month Rapid-Onset window: what clinical actions are feasible in this window
- Describe the 3+ intervention windows in Chronic-Escalation: map to specific care transition points
- Connect to the dashboard "What-If" feature in CH_5

**Limitations** (60 words → target 250)
- Cash-pay/illicit opioid: explain why this biases toward sicker insured patients
- Missing lab data: explain why renal function matters for opioid dosing decisions
- ICD-10 F11.xx sensitivity/specificity: known underdetection of OUD in claims
- No prospective validation: temporal holdout ≠ prospective clinical study

---

## CH_4 — Polypharmacy (Target: +2,625 words minimum)

### High Priority (expand now — methods/discussion are retrain-independent)

**Study Design and Data Source** (31 words → target 200)
- State retrospective cohort study design, IRB waiver (HM20022300)
- Specify age range (65–114), years (2016–2019), APCD source
- State this is a non-opioid ED cohort to distinguish from CH_3

**Methods — Cohort Construction** (134 words → target 400)
- Expand Milliman HCG code definitions (O11, P51): what these codes capture vs. opioid ED (F11.xx)
- Describe strict opioid exclusion: any F11.xx in any diagnosis column = excluded
- Describe 30-day outcome window for ADE definition
- Add Z-code case/control classification: cases have significantly lower Z-code proportions
- Describe matching procedure: same 5:1 nearest-neighbor within age band and year

**Methods — Ensemble Model** (39 words → target 200)
- Cross-reference CH_2 for full ensemble details
- Note that all three geriatric bands (65–74, 75–84, 85–114) are analyzed separately
- Describe which Consensus-Causal features from this cohort were retained (89 features vs. opioid cohort's 498)

**Methods — FFA Multi-Feature Interaction Testing** (143 words → target 400)
- Describe interaction effect (IE) formula: IE(A,B) = P(ED | A=1, B=1) − P(ED | A=1) − P(ED | B=1) + P(ED)
- Describe intervention rate (IR) formula: IR(drug) = E[Δp̂ | remove drug from patient profile]
- Describe bootstrap CI computation for both IE and IR (1,000 resamples of 2019 holdout)
- State threshold for synergistic pair designation: IE > 0, 95% CI lower bound > 0
- Describe triplet interaction extension: enumeration of all 3-drug combinations among top-20 causal drugs

**Methods — Statistical Analysis** (60 words → target 200)
- State Mann-Whitney U test for Z-code proportion comparison (non-normal distribution)
- Describe logistic regression model for adjusted ORs: covariates (age band, sex, 30-day drug count)
- State multiple comparison correction (Bonferroni for pairwise IE tests)
- State significance level α = 0.05

**Results — Cohort Characteristics** (91 words → target 300)
- Add sex distribution (geriatric cohorts tend female-predominant)
- Add comorbidity burden by band: 85–114 has highest Elixhauser scores
- Add top 3 drug classes by frequency in each geriatric band

**Results — Ensemble Performance** (55 words → target 200)
- Expand beyond table reference: explain why polypharmacy PR-AUC (0.984–0.997) dramatically exceeds opioid (0.835–0.916)
- Cite case prevalence difference as primary driver
- Note Brier scores are near-zero due to high precision at low recall threshold

**Results — Synergistic DDI** (143 words → target 400)
- Add sentence on IE score range across 115 pairs
- Describe the clinical rationale for levofloxacin + lorazepam (QT prolongation + CNS depression)
- Describe the clinical rationale for acetaminophen + levofloxacin (hepatic + renal load)
- Add sentence on 3+ drug combinations that appeared benign under pairwise screening

**Results — Triplet Interactions** (60 words → target 200)
- Expand furosemide + HCTZ + lisinopril mechanistic explanation (electrolyte depletion + ACE inhibition + potassium wasting)
- Describe at least one other high-risk triplet from 85–114 band
- State triplet-level IE scores and how they compare to pairwise predictions

**Discussion — FFA Causal Calculator** (107 words → target 350)
- Compare IE detection to standard DDI databases (Drugs.com, Micromedex): how many of the 115 pairs were already flagged?
- Discuss IR score as prioritization tool: how a pharmacist would use the ranked list
- Compare to Beers Criteria: what was captured by FFA that Beers Criteria misses (multi-drug emergent effects)

**Discussion — Managed vs. Unmanaged Polypharmacy** (51 words → target 250)
- Expand U-shaped finding: explain why Q4 (very high Z-code) reverts to high risk (disease severity confounding)
- Cite prior literature on Z-code protective effect
- Clinical implication: "coordinate monitoring" rather than "prescribe less"

**Limitations** (49 words → target 200)
- Cash-pay prescription gap (same as CH_3)
- IE/IR scores are observational, not experimentally validated — requires randomized deprescribing trial to confirm
- Milliman HCG codes may have variable sensitivity across payers
- FFA IE computation assumes conditional independence between drug pairs — may underestimate triplet effects

---

## CH_5 — Dashboard (Target: +3,252 words minimum)

### High Priority (expand now — architecture/design sections are retrain-independent)

**Design Philosophy** (82 words → target 300)
- Expand privacy-first rationale: HIPAA § 164.502(b) minimum necessary standard; why ephemeral compute satisfies this
- Describe design decision to reject browser-local processing (model size prohibitive)
- Describe design decision to reject server-side session state (PII retention risk)
- Connect to Chapter 1 finding: 0% of prior tools integrated XAI + PGx without server storage

**Hybrid Deployment Model** (52 words → target 300)
- Describe S3 static hosting specifics: bucket policy, CloudFront OAI, HTTPS enforcement
- Describe Lambda container specifics: ECR registry, Docker base image (python:3.11-slim), container size optimization
- Describe API Gateway configuration: Lambda proxy integration, CORS headers, rate limiting

**Partition-First Routing** (46 words → target 250)
- Expand routing logic: age → age band → n_event_bin (computed from submitted code count) → per-bin model
- Describe how Lambda determines n_event_bin at inference time (count submitted codes, apply threshold JSON)
- Add sentence on fallback behavior if age or bin model is missing

**Model Storage Feasibility** (45 words → target 250)
- Describe container size breakdown: per-bin models (~X MB each × bins × cohorts × bands = total), CPIC DB (~Y MB), Python runtime (~Z MB)
- Compare to Lambda 10 GB container limit
- Add sentence on cold-start vs. warm-start model loading strategy

**Ensemble Risk Scoring** (104 words → target 300)
- Expand imputation of normality technical detail: per-feature median computed from training partition, stored in feature_schema.json
- Describe how ensemble weights are applied at inference: JSON-serialized w_i loaded per bin/cohort/band combination
- Add sentence on probability calibration: Platt scaling applied post-ensemble-blend

**CI/CD Pipeline** (92 words → target 300)
- Describe GitHub Actions workflow: trigger (push to main), steps (pytest → Docker build → ECR push → Lambda update)
- Describe test coverage: Lambda handler unit tests, integration test with synthetic patient input
- Add sentence on container image versioning strategy (SHA-tagged, rollback via Lambda alias)

**Performance Benchmarks** (68 words → target 300)
- Expand cold-start analysis: explain container init sequence (Python runtime → model deserialize → CPIC DB load)
- Explain why warm inference is ~6 ms (in-memory model, no I/O after cold start)
- Describe provisioned concurrency configuration (if used)
- Add sentence on API Gateway overhead (~1–2 ms) and CloudFront CDN delivery (~8–15 ms globally)

**Discussion — Resolving the Last-Mile Problem** (176 words → target 450)
- Compare to prior clinical AI deployment efforts: Epic Sepsis Model, Epic deterioration index — closed-ecosystem, no transparency
- Discuss stateless PGx card as novel contribution: first tool integrating 573 CPIC gene-drug pairs with opioid risk scoring
- Address clinician trust: FFA causal rules ("what-if" counterfactuals) directly address black-box adoption barrier

**Limitations and Future Work** (91 words → target 350)
- Expand each limitation with 2 sentences: single-state models, no measured genotypes, synthetic latency data, no clinical pilot
- Add: container rebuild required for CPIC guideline updates (CI/CD partially addresses this)
- Add: model drift detection not yet implemented
- Add: dashboard currently English-only

---

## CH_6 — Conclusion (Target: +2,521 words minimum)

### High Priority (expand now — synthesis sections are retrain-independent)

**Overview of the Dissertation** (99 words → target 300)
- Add paragraph connecting the five chapters as a unified translational pipeline
- Describe how each chapter addresses a specific failure mode identified in Chapter 1's SQLR
- State the overarching thesis in 2–3 sentences explicitly

**Chapter summaries — each needs doubling** (~70–100 words each → target 200 each)
- **CH_1:** Add specific SQLR contribution: quantified the 0% overlap at XAI+PGx+causal intersection; established the OODA architecture framework
- **CH_2:** Add specific architecture contribution: 15× speedup quantified, Consensus Filter superiority over SHAP-only/FFA-only proven on 2019 holdout
- **CH_3:** Add specific prediction contribution: rapid-onset archetype's 4.2-month window, CYP2D6 in top-10 Consensus features
- **CH_4:** Add specific DDI contribution: 115 synergistic pairs beyond pairwise DDI databases, Z-code protective effect for managed polypharmacy
- **CH_5:** Add specific deployment contribution: first serverless PGx+risk integration, sub-100 ms warm inference, zero PII storage

**XAI–PGx–Causal Integration** (109 words → target 400)
- Expand step 2 (Consensus Filter): describe how the SHAP ∩ FFA intersection operationalizes the "explain and confirm" principle from causal inference literature
- Add paragraph on the mutual dependency between PGx enrichment and causal validation: without FFA confirmation, CYP2D6 signal is correlation not causation
- Add paragraph connecting to Pearl's causal hierarchy: association (SHAP) → intervention (FFA IR scores) → counterfactual (What-If dashboard)

**Common Methodology** (79 words → target 250)
- Expand on why methodological consistency enables cross-chapter comparison
- Add specific example: polypharmacy PR-AUC (0.984–0.997) vs. opioid (0.835–0.916) — attribute to case prevalence, not methodology
- Describe what would be different if cohorts used different feature engineering

**For Translational Informatics** (46 words → target 250)
- Add paragraph on the translational pipeline: APCD → Cohort → Features → Ensemble → Consensus → Rules → Dashboard → Clinician
- Discuss replicability: what another researcher needs to replicate this framework (data DUA, cloud infrastructure, CPIC access)
- Note open-source code availability (GitHub)

**For the Opioid Epidemic** (67 words → target 250)
- Expand on CYP2D6 poor metabolizer + opioid escalation pathway: how the PGx card enables pre-prescription genotype screening
- Add sentence on population-level impact: if 21% of opioid ED cases are Rapid-Onset, earlier identification represents substantial ED burden reduction

**For Geriatric Pharmacotherapy** (61 words → target 250)
- Expand Z-code finding clinical implication: current quality metrics reward prescribing volume reduction but not monitoring coordination
- Add sentence on IR scores as actionable tool: specifically for pharmacist-led deprescribing programs
- Discuss how the framework could inform geriatric polypharmacy guidelines

**Limitations — Data** (93 words → target 300)
- Expand each limitation with mechanistic reasoning and impact estimate
- Add quantification: cash-pay opioid underestimation estimated at 15–30% in 18–34 age group (cite prior literature)

**Limitations — Methodological** (71 words → target 250)
- Add FFA limitation: symbolic rules from XGBoost are path-dependent, not globally unique
- Add 5:1 matching limitation: discuss sensitivity analysis with 3:1 and 7:1 ratios

**Future Work — Near-Term** (86 words → target 300)
- Add prospective clinical pilot: specific venue (opioid treatment program or emergency department), outcome metric (prescriber acceptance rate), timeline

**Future Work — Long-Term** (101 words → target 300)
- Add federated learning architecture detail: how partition-first generalizes to multi-state federation
- Add drift detection mechanism: SHAP distribution shift as early warning signal

---

## Implementation Priority Order

### Do Now (no retrain needed, highest gap-per-hour value)

1. **CH_2 Results + Discussion** — All 6 result sections thin; Discussion only 112 words
2. **CH_6 Chapter summaries + Integration** — Pure synthesis, no numeric dependencies
3. **CH_1 Methods** — SQLR protocol, eligibility, data extraction sections all <100 words
4. **CH_5 Architecture sections** — Design philosophy, deployment model, CI/CD pipeline

### Do After Retrain

5. **CH_3 Results** — Cohort characteristics, causal features with actual SHAP values
6. **CH_4 Results** — DDI pairs/triplets with actual IE scores
7. **CH_5 Results** — Performance benchmarks with real latency values
8. **CH_6 Performance Summary** — Update table + 2 paragraphs of commentary
