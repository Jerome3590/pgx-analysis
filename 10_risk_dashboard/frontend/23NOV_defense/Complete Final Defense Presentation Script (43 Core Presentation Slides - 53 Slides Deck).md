# Complete Final Defense Presentation Script (43 Core Presentation Slides - 53 Slides Deck)

- **Candidate:** R. Jerome Dixon, MDA, MSc IT (`dixonrj@vcu.edu`)
- **Chair & Sponsor:** Elvin T. Price, Pharm.D., Ph.D., FAHA
- **Defense Date:** Monday, November 23, 2026
- **Deck Structure:** 53 Total Slides (43 Core Slides across Modules I–V + 10 Appendix Slides [Divider + A1–A9])
- **Live Deck:** https://pgx.jerome-dixon.io/23NOV_defense/

---

## Slide 01: Gradient-Boosted Trees and Consensus-Risk
- **Section:** Title Slide · VCU Integrative Life Sciences Ph.D. Defense · Nov 23, 2026
- **Live Anchor:** https://pgx.jerome-dixon.io/23NOV_defense/#/title-slide

"Good afternoon, members of the committee. *Brief pause*  
My name is Jerome Dixon. I am a student in the Integrative Life Sciences Doctoral Program, conducting my dissertation research within the School of Pharmacy at Virginia Commonwealth University.  
Welcome to my final dissertation defense: *Gradient-Boosted Trees and Consensus-Risk For Opioid and Polypharmacy Emergency Department Visits in Virginia’s APCD*.  
Across emergency medicine and outpatient pharmacotherapy, clinicians face an escalating dual crisis: the ongoing mortality of the opioid epidemic and the surging, silent morbidity of geriatric polypharmacy adverse drug events - ADEs. Today, I will present our computational framework addressing this challenge.  
This research presents an end-to-end architecture that connects population-scale predictive analytics with explainable clinical decision support.  
Using longitudinal statewide claims from the Virginia All-Payer Claims Database, the architecture integrates gradient-boosted tree ensembles across multiple topologies, rigorous temporal leakage audits, and a dual-model consensus attribution framework combining SHAP values with formal symbolic logic.  
*Lean in / Vocal emphasis*  
Our ultimate aim is to support actionable insights at the point of care while minimizing unnecessary alerts and clinician alert fatigue.  
With that, let us proceed on our journey."

---

## Slide 02: Supervisory Committee
- **Section:** Integrative Life Sciences Doctoral Program · School of Pharmacy · Department of Pharmacotherapy & Outcomes Science
- **Live Anchor:** https://pgx.jerome-dixon.io/23NOV_defense/#/slide-2

"Before we dive into the core modules, I want to express my sincere gratitude to:  
my chair and sponsor, Dr. Price, as well as Professor Bost, Dr. Gal, Dr. Wijesinghe, and Dr. DeShazo. *Brief pause*  
Your time, example, and mentorship has been invaluable throughout this journey. Your combined expertise across pharmacotherapy, systems engineering, informatics, and real-world evidence is vital to this work.  
*Lean in / Vocal emphasis*  
Thank you all for being here today for this final dissertation defense."

---

## Slide 03: Roadmap
- **Section:** Final Defense Agenda · 5 Core Modules
- **Live Anchor:** https://pgx.jerome-dixon.io/23NOV_defense/#/sec-roadmap

"Here on our Roadmap slide, we outline the final defense agenda across five core modules: *Brief pause*  
**Modules I through IV** establish our scientific problem formulation, three-tier governance and Virginia APCD data architecture, dual cohort derivation pipelines, and GBDT with `TreeSHAP ∩ FFA` consensus modeling and Step 3b leakage protection.  
*Lean in / Vocal emphasis*  
**Module V: Translation & Deployment** bridges our architecture into practice—starting with the Clinical OODA Loop and study boundaries, presenting our complete empirical results across **Data Engineering (Chapter 2)**, the **Opioid Cohort (Chapter 3)**, the **Polypharmacy Cohort (Chapter 4)**, and the **PGx Risk Dashboard (Chapter 5)**, and closing with Chapter 6's translational **Next Steps** spanning our 2020–2024 Wright Center APCD refresh, VCU CCTR Epic SMART on FHIR pilot, partner EHR linkage, and FDA SaMD governance."

---

## Slide 04: Initial Analysis of Problem
- **Section:** Module I · Infographic Part 1 of 4 · Downstream Public Health & Economic Toll
- **Live Anchor:** https://pgx.jerome-dixon.io/23NOV_defense/#/slide-1

"Turning to Slide 4, we begin Module I by examining the downstream clinical and economic effects of medication-related morbidity. *Brief pause*  
Preventable emergency department utilization in the United States is driven by two distinct medication-risk cohorts with verified national epidemiological benchmarks:  
**First**, in Cohort 1, CDC WONDER epidemiological surveillance documents that since 1999, prescription and illicit opioid overdoses have claimed over 500,000 lives in the United States, with Florence and colleagues' landmark CDC health-economics evaluation in *Medical Care* establishing an annual societal burden exceeding 78.5 billion dollars across direct healthcare ($26.1B), lost workplace productivity ($41.8B), and criminal justice expenditures ($7.6B).  
**Second**, in Cohort 2, Charlesworth et al. in the *Journals of Gerontology* and Kantor et al. in *JAMA* document that geriatric polypharmacy—defined as five or more concurrent prescription medications—affects over 40% of older adults aged 65 and older (with ~20% taking 10 or more), while CDC NEISS-CADES active surveillance published by Budnitz and Shehab in *JAMA* demonstrates that outpatient adverse drug events drive between 700,000 and 1.3 million emergency department visits and over 125,000 deaths annually."

---

## Slide 05: Literature Gap: Opioid vs. Polypharmacy Research
- **Section:** Module I · Infographic Part 2 of 4 · Systematic Review (n = 94 ML Studies)
- **Live Anchor:** https://pgx.jerome-dixon.io/23NOV_defense/#/slide-1-imbalance

"Slide 5 presents the second panel of our systematic review infographic: opioid research heavily overshadows polypharmacy research across 94 supervised machine learning studies. *Brief pause*  
**First**, 95% of published machine learning research clusters into just two buckets—46% predicting opioid use disorder diagnosis or dependence, and 49% predicting general acute care or overdose utilization—leaving only 5%, or just five studies, addressing polypharmacy adverse drug events and drug-drug interactions.  
**Second**, out of 1,473 published opioid use disorder literature records screened under PRISMA 2020, only 6 studies specifically evaluate machine learning models designed to predict opioid-related emergency department visits using administrative claims data."

---

## Slide 06: ML Modeling Gaps: PGx, XAI & Target Leakage
- **Section:** Module I · Infographic Part 3 of 4 · PROBAST Appraisal (n = 73 Models)
- **Live Anchor:** https://pgx.jerome-dixon.io/23NOV_defense/#/slide-1-empty

"Slide 6 examines the third panel of our infographic: the three methodological failures uncovered across 73 PROBAST-assessed prediction models. *Brief pause*  
**First**, in Failure 1, 30% of published models suffer from temporal data leakage—allowing post-index treatments such as buprenorphine or naloxone to leak into pre-outcome predictor windows, artificially inflating AUROC by 5 to 25 percentage points before collapsing in out-of-time deployment.  
**Second**, in Failure 2, 81% completely omit pharmacogenomic modifiers despite mature CPIC guidelines for *CYP2D6* and *OPRM1*.  
*Lean in / Vocal emphasis*  
**Third**, in Failure 3, across all 73 PROBAST-assessed models, exactly zero combine structured pharmacogenomic features, explainable AI, and out-of-time temporal holdout validation."

---

## Slide 07: Preliminary Data: 90% of ADEs Occur Within 21 Days
- **Section:** Module I · FAERS
- **Live Anchor:** https://pgx.jerome-dixon.io/23NOV_defense/#/slide-1-faers

"Before moving to the frontline clinical bottleneck, Slides 7 through 9 present our preliminary data that shaped our pipeline architecture. *Brief pause*  
**First**, on Slide 7, we asked a foundational pharmacoepidemiologic question using external FDA FAERS pharmacovigilance data: over what pre-visit lookback window do serious acute adverse drug events actually occur?  
**Second**, across 468,298 deduplicated FDA FAERS hospitalization and death reports, nearly half—49.5%—occur within the first 7 days, and 87.7% occur within 21 days of drug initiation.  
*Lean in / Vocal emphasis*  
**Third**, this external FDA FAERS distribution establishes our empirical prior for a proximal 1-to-21-day acute polypharmacy window, which we will validate directly against Virginia APCD claims side-by-side in Module III (Cohorts)."

---

## Slide 08: Preliminary Data: Opioid Cohort Feature Analysis (opioid_ed)
- **Section:** Module I · Step 3a · Panel A (8 Age Bands)
- **Live Anchor:** https://pgx.jerome-dixon.io/23NOV_defense/#/slide-1-fi-heatmap

"Slide 8 begins our Step 3a preliminary multi-model feature analysis with Panel A: the `opioid_ed` cohort across all eight age bands. *Brief pause*  
We pivoted event-level claims into patient-level numeric count matrices and aggregated Recall-scaled feature importances across CatBoost, XGBoost, and XGBoost-RF. Across adolescents and adults aged 13 to 64, true clinical drivers center on long-term opiate therapy (`Z79.891`), chronic pain (`G89.29`, `G89.4`), psychiatric care (`90837`), oxycodone, and gabapentin.  
*Lean in / Vocal emphasis*  
Critically, this raw heatmap also exposed a major methodological trap—marked with asterisks: unfiltered Step 3a models achieved an artificially inflated 0.99 AUROC because synchronous toxicology screens (`G0479`), ED evaluation CPTs (`99284`, `99285`), and post-event buprenorphine treatment leaked into the feature space, directly motivating our Step 3b leakage filter."

---

## Slide 09: Preliminary Data: Polypharmacy Feature Analysis (non_opioid_ed)
- **Section:** Module I · Step 3a · Panel B (8 Age Bands)
- **Live Anchor:** https://pgx.jerome-dixon.io/23NOV_defense/#/slide-1-fi-non-opioid

"Slide 9 shows Panel B of our Step 3a preliminary feature analysis for our geriatric polypharmacy cohort, `non_opioid_ed`, across all eight age bands. *Brief pause*  
Unlike `opioid_ed`, predictive importance concentrates sharply in older adults aged 65 to 84 around concurrent cardiometabolic, diuretic, beta-blocker, and statin regimens—including amlodipine, losartan, furosemide, metoprolol, metformin, gabapentin, omeprazole, and multiple statins.  
*Lean in / Vocal emphasis*  
Notably, many of these top geriatric predictors are CPIC-actionable pharmacogenomic substrates—directly motivating our Step 5 CPIC drug mapping on the following slides."

---

## Slide 10: Preliminary Data: Cohort Feature Target Split
- **Section:** Module I · Step 3a · Panel C (Cumulative Separation)
- **Live Anchor:** https://pgx.jerome-dixon.io/23NOV_defense/#/slide-1-fi-target-split

"Slide 10 summarizes Panel C—the cumulative target split directly comparing `opioid_ed` on the left against `non_opioid_ed` on the right. *Brief pause*  
Look at the sharp separation across the red dashed line: the top predictors bifurcate cleanly into two orthogonal clinical regimes, with gabapentin standing out as the key cross-cohort bridge.  
*Lean in / Vocal emphasis*  
This cumulative split empirically proves why these two populations cannot be pooled into one generic risk model—justifying our dual-cohort architecture and mandatory Step 3b pre-index leakage purge."

---

## Slide 11: Preliminary Data: CPIC Drug Density by Cohort
- **Section:** Module I · Step 5 · Panel A (8 Age Bands)
- **Live Anchor:** https://pgx.jerome-dixon.io/23NOV_defense/#/slide-1-cpic-fi

"Slide 11 presents Panel A of our Step 5 preliminary pharmacogenomic mapping: quantifying how many drug features surfaced in initial Feature Importance are CPIC-actionable medications across all eight age bands. *Brief pause*  
Between 24% and 35% of all drug features selected in initial feature importance map directly to Clinical Pharmacogenetics Implementation Consortium guidelines—spanning 187 distinct generic CPIC drugs in `opioid_ed` and 224 distinct generic CPIC drugs in `non_opioid_ed`.  
*Lean in / Vocal emphasis*  
At the patient level across 4.04 million Virginia APCD drug records, 76% to 92% of prescription-exposed `opioid_ed` patients and 61% to 78% of `non_opioid_ed` patients receive at least one CPIC-actionable medication—and our engineered exposure counts `pgx_num_drugs` and `pgx_num_cpic_drugs` rank among the top 3 predictors across nearly every final model."

---

## Slide 12: Preliminary Data: CPIC Gene–Drug Pairs by Cohort
- **Section:** Module I · Step 5 · Panel B (CPIC / PharmGKB Pathways)
- **Live Anchor:** https://pgx.jerome-dixon.io/23NOV_defense/#/slide-1-cpic-pairs

"Slide 12 shows Panel B: the dominant CPIC pharmacogene-drug pairs surfaced across each target cohort. *Brief pause*  
The specific CPIC drug classes separate cleanly by clinical mechanism: *CYP2D6* opioids and psychotropics dominate `opioid_ed`, whereas *CYP2D6* beta-blockers, *SLCO1B1* statins, *CYP2C19* proton-pump inhibitors, and *CYP2C9* ARBs and anticoagulants dominate `non_opioid_ed`.  
*Lean in / Vocal emphasis*  
We calibrate our scientific framing explicitly: administrative claims do not recover a patient's genetic diplotype; rather, they quantify cumulative CPIC-actionable medication exposure burden—pinpointing patients whose regimens warrant pharmacogenomic review and motivating addressing the frontline clinical bottleneck on the next slide."

---

## Slide 13: Frontline Clinical Bottleneck
- **Section:** Module I · Infographic Part 4 of 4 · Point-of-Care Translation Gap
- **Live Anchor:** https://pgx.jerome-dixon.io/23NOV_defense/#/slide-1-bottleneck

"Slide 13 isolates the fourth component of our problem formulation: The Frontline Clinical Bottleneck. *Brief pause*  
**First**, frontline clinicians and pharmacists must navigate complex, multi-year medication and claims histories during rapid clinical visits, while traditional pairwise drug-interaction checkers trigger 90% to 95% alert override fatigue and miss higher-order multi-drug synergies.  
**Second**, black-box machine learning models that output raw probability scores without verifiable clinical rules fail to earn clinician trust at the point of care.  
*Lean in / Vocal emphasis*  
**Third**, our dissertation bridges this bottleneck with a calibrated, defensible impact: identifying and prioritizing potentially modifiable medication-related risk factors via leakage-free gradient-boosted trees and Consensus-Risk Horn rules to establish a framework for pharmacist-led medication review."

---

## Slide 14: Specific Aim 1: Cohort 1 (opioid_ed)
- **Section:** Module I · Aim 1 · Cohort 1 (opioid_ed)
- **Live Anchor:** https://pgx.jerome-dixon.io/23NOV_defense/#/slide-1a

"Slide 14 presents Specific Aim 1 alongside our Cohort 1 target definition. *Brief pause*  
In Aim 1 (Chapter 3), we develop and temporally validate our tri-model gradient-boosted tree family—CatBoost, XGBoost, and XGBoost-RF—paired with `SHAP ∩ FFA` consensus attribution for Cohort 1 (`opioid_ed`, ages 13 to 64), evaluating pre-index leakage prevention, PR-AUC and AUROC discrimination, and calibration on an independent 2019 temporal holdout.  
*Lean in / Vocal emphasis*  
Below, Cohort 1 captures chronic opioid escalation over a 12-month pre-index lookback window, anchored directly to explicit ICD-10 `F11.20` opioid dependence and `T40.0X` through `T40.4X` poisoning codes scanned across all ten claim diagnostic positions."

---

## Slide 15: Specific Aim 2: Cohort 2 (ed_non_opioid)
- **Section:** Module I · Aim 2 · Cohort 2 (ed_non_opioid)
- **Live Anchor:** https://pgx.jerome-dixon.io/23NOV_defense/#/slide-1b

"Slide 15 presents Specific Aim 2 alongside our Cohort 2 target definition. *Brief pause*  
In Aim 2 (Chapter 4), we extend our tri-model gradient-boosted tree family and `SHAP ∩ FFA` symbolic Boolean Horn rules to Cohort 2 (`ed_non_opioid`, ages 65 to 114), quantifying candidate higher-order multi-drug interaction effects (`IE`) and consensus risk attenuation priorities (`IR`) on the untouched 2019 holdout.  
*Lean in / Vocal emphasis*  
Below, Cohort 2 addresses the 'Silent ADE' coding reality: because fewer than 5% of drug-induced geriatric ED visits are billed with explicit poisoning codes, we anchor to Milliman HCG emergency lines `P51b`, `O11`, and `P33` within a 21-day proximal prescription window and zero opioid codes—capturing downstream falls, fractures, syncope, delirium, GI bleeds, and acute kidney injury."

---

## Slide 16: Specific Aim 3: Decision Support
- **Section:** Module I · Aim 3 · Decision Support
- **Live Anchor:** https://pgx.jerome-dixon.io/23NOV_defense/#/slide-1c

"Slide 16 presents Specific Aim 3 alongside our core clinical decision-support design principles. *Brief pause*  
At the top, Aim 3 (Chapter 5) translates our validated Aim 1 and Aim 2 risk estimates and interpretable Boolean rules into a usable prototype decision-support tool for pharmacist-led Medication Therapy Management review prioritization—explicitly scoped to evaluating explanation fidelity, clinical interpretability, and workflow usability rather than claiming prospective clinical effectiveness.  
*Lean in / Vocal emphasis*  
Below, the interface is governed by Lean Value Information Flow: organizing clinical evidence in a natural high-to-low, left-to-right visual hierarchy—from top-level risk score down and across to causal drivers and counterfactual levers—while maintaining low cognitive overhead through progressive disclosure to prevent pharmacist alert fatigue."

---

## Slide 17: Governance & Analytical Frameworks, Study Boundaries
- **Section:** Module II · Phase 1 · CRISP-DM, TRIPOD+AI & Clinical OODA Loop
- **Live Anchor:** https://pgx.jerome-dixon.io/23NOV_defense/#/slide-2

"Directing your attention to our governance architecture, our engineering and validation pipeline is governed by three concentric architectural rings. *Brief pause*  
**First**, the outer ring executes the six-phase CRISP-DM systems engineering lifecycle, enforcing strict reproducibility across our ten-step codebase—from clinical problem formulation and DuckDB ingestion through Step 3b leakage filtering, tri-model GBDT fitting, and containerized deployment.  
**Second**, the middle ring enforces the 27-item TRIPOD+AI biomedical reporting standard for Type 2b and Type 3 independent temporal holdout validation. Models are trained and tuned exclusively on 2016 through 2018 Virginia APCD claims and evaluated once on an untouched 2019 calendar holdout. Crucially, we prohibit post-hoc recalibration on the 2019 holdout: adjusting probability curves after inspecting test-period outcomes is equivalent to grading your own exam after peeking at the answer key, which masks temporal calibration drift.  
*Lean in / Vocal emphasis*  
**Third**, the inner core operationalizes the Clinical OODA Loop—Observe, Orient, Decide, and Act. Our GBDT models and symbolic Horn rules execute Observe and Orient, while licensed clinical pharmacists retain 100% authority over Decide and Act within an asynchronous medication review queue."

---

## Slide 18: Study Boundaries
- **Section:** Module II · Phase 1 · 8 Interlocking Epidemiological & Governance Safeguards
- **Live Anchor:** https://pgx.jerome-dixon.io/23NOV_defense/#/slide-10

"Within Module II Governance, our dissertation is bounded by eight interlocking safeguards—four epidemiological cohort constraints on the left, and four clinical governance principles on the right:  
**On the left**, our epidemiological design enforces:  
**First**, the Absolute Opioid Cleanse across all ten diagnostic positions in our geriatric cohort;  
**Second**, High-Utilization Capping excluding patients with seven or more annual emergency visits;  
**Third**, the 21-Day Proximal Pharmacy Window—where external FDA FAERS onset kinetics directly validate our cohort derivation logic; and  
**Fourth**, the Zero-Day Discharge Guard systematically purging day-zero fills and post-index confounders to eliminate target leakage, while transparently noting claims limitations around cash-pay PDMP fills, VOAA community reversals, and undifferentiated `R41.82` Altered Mental Status coding.  
*Brief pause*  
**On the right**, our governance framework establishes:  
**Fifth**, Aim 3 is consultative pharmacist decision-support triage, not a patient outcomes trial—separating direct consumer self-use under FTC privacy from hospital deployment under our two-tier HIPAA BAA chain;  
**Sixth**, Consensus Attribution quantifies observational risk priority rather than causal treatment effects;  
**Seventh**, the untouched 2019 temporal holdout evaluates real-world model transportability while quarantining 2020 pandemic distortion; and  
**Eighth**, our features measure observable CPIC medication burden rather than genotypes, deliberately omit administrative race as a proxy for biogeographical genetic ancestry, and enforce a hard `CLIA/CAP` referral lockout on unphased consumer DNA arrays."

---

## Slide 19: Virginia APCD vs. Single-Facility EHR
- **Section:** Module III · Phase 1 · Statewide Generalizability & Data Architecture
- **Live Anchor:** https://pgx.jerome-dixon.io/23NOV_defense/#/slide-3

"Turning to Slide 19, we justify selecting the Virginia All-Payer Claims Database over Electronic Health Records or single-payer Medicare datasets. *Brief pause*  
Single-system EHRs suffer from critical outcome blindness in emergency prediction: when an individual overdoses or an older adult suffers a syncopal fall, emergency medical services transport them to the nearest hospital, not necessarily their home academic center.  
**First**, the Virginia APCD provides a multi-payer, statewide window encompassing commercial, Medicare Advantage, and Medicaid populations—eliminating out-of-network emergency blindness across competing health systems with zero linkage attrition via Virginia Health Information.  
**Second**, whereas EHRs record written e-prescriptions—missing the roughly 30% primary non-adherence drop-off abandoned at the pharmacy counter—APCD pharmacy claims capture paid, adjudicated dispenses that verify physical drug possession.  
*Lean in / Vocal emphasis*  
**Third**, while administrative claims lack raw DNA swabs and bedside vitals, we methodologically compensate in Step 5 (`5_pgx_analysis`) by mapping all dispensed APCD medications against the official CPIC directory (`drug_cpic_mapping_global.csv`) to determine whether each drug carries a CPIC guideline, computing patient-level counts of total drugs (`pgx_num_drugs`), CPIC-guideline drugs (`pgx_num_cpic_drugs`), and how many patients carry CPIC drugs—quantifying observable medication exposure burden directly without unscientific genotype imputation."

---

## Slide 20: Medical & Pharmacy Tables with Virginia APCD
- **Section:** Module III · Step 2
- **Live Anchor:** https://pgx.jerome-dixon.io/23NOV_defense/#/slide-4

"Before examining our two target cohorts on Slides 21 and 22, Slide 20 details how we validate, clean, and relationally join the Virginia APCD Gold medical and pharmacy tables into our 39-column unified event fact table. *Brief pause*  
To ensure administrative claims integrity prior to modeling, automated preprocessing scripts filter voided or reversed claim lines and harmonize schema fields across years.  
**First**, the Gold medical table contributes 25 standardized columns—capturing all ten ICD-10 diagnostic positions, nine ICD procedure and three CPT modifier fields, Milliman HCG emergency service lines `P51b`, `O11`, and `P33`, and imputed demographics—while the Gold pharmacy table captures paid prescription fills normalized from National Drug Codes to generic RxNorm `drug_name` and `therapeutic_class_1`.  
**Second**, in Phase 2, DuckDB links and unions these medical and pharmacy records at the patient level on `mi_person_key`, `event_year`, and `age_band`, backfilling demographic covariates across pharmacy rows and computing a global chronological `event_sequence` ordered by claim date relative to index date zero.  
*Lean in / Vocal emphasis*  
**Third**, from this strictly pre-index timeline—where claim service date precedes index date—we engineer our four core feature families: multi-position diagnosis and procedure indicators (`item_icd_*`, `item_cpt_*`), active generic drug exposures (`item_drug_*`), our Step 5 CPIC drug counts (`pgx_num_drugs`, `pgx_num_cpic_drugs`), and claims-density utilization tiers."

---

## Slide 21: Aim 1: Opioid ED Cohort
- **Section:** Module III · Step 2
- **Live Anchor:** https://pgx.jerome-dixon.io/23NOV_defense/#/slide-4a

"Slide 21 presents the architectural pipeline for Cohort 1, our 12-month chronic opioid emergency cohort, `opioid_ed`. *Brief pause*  
To capture true acute opioid toxicity and dependence crises across adolescents and adults aged 13 to 64, we anchor index encounters using validated ICD-10 diagnostic criteria.  
**First**, case inclusion requires an emergency claim carrying explicit ICD-10 `F11.20` opioid dependence or `T40.0X` through `T40.4X` acute opioid poisoning codes inspected across all ten diagnostic positions.  
**Second**, because opioid tolerance and dose escalation evolve longitudinally, we extract a 12-month pre-index lookback window—from day minus 365 through day minus 1—paired with 1-to-5 incidence-density matched controls across eight age bands, sex, and calendar year, and screened via our Step 3b leakage purge.  
*Lean in / Vocal emphasis*  
**Third**, to process 1.8 Terabytes and 380 million claims without memory exhaustion, we engineered a partition-first out-of-core DuckDB pipeline on AWS Spot EC2 `x2iedn.8xlarge` workers. Partitioning by Age Band and Event Year eliminated shuffle skew—achieving a 15.1-fold speedup to 18.1 million records per minute, zero failures across 189 passes, and reducing compute cost from 23,000 dollars a month on EMR clusters to under 10 dollars per pass."

---

## Slide 22: Aim 2: Polypharmacy ADE Cohort
- **Section:** Module III · Step 2
- **Live Anchor:** https://pgx.jerome-dixon.io/23NOV_defense/#/slide-4b

"Turning to Slide 22, we examine the parallel pipeline architecture for our geriatric polypharmacy cohort, `ed_non_opioid`.  
*Brief pause*  
Focusing on older adults aged 65 to 114, we anchor index encounters using Milliman HCG emergency service lines—specifically `P51b`, `O11`, and `P33`. To isolate pure medication-related adverse events from opioid toxicity, we apply an absolute opioid cleanse: excluding any patient with even a single opioid ICD-10 code across all ten diagnostic positions during the study year.  
Next, we pass candidate records through a three-stage epidemiological funnel:  
**First**, a high-utilization cap excluding individuals with seven or more annual emergency visits, filtering out chronic crisis utilizers.  
**Second**, a 21-day proximal pharmacy window—from day minus 21 through day minus 1. As established on Slide 7, this lookback is empirically validated by our FAERS-to-APCD time-to-onset convergence demonstrating that over 90% of acute ADEs occur within 3 weeks of medication initiation or modification, providing external cohort validation for our polypharmacy pipeline—detailed in the Cohort Validation card and Figure 1B.  
*Lean in / Vocal emphasis*  
**Third**, we enforce a strict Zero-Day Discharge Guard where claim date precedes index date. This systematically purges any prescription filled on the calendar day of the emergency visit, guaranteeing that post-discharge medications are never mistaken for pre-visit etiology.  
*Brief pause*  
Cohort cases are matched 1-to-5 with controls across age band, sex, and calendar year. Finally, our verification script—`qa_index_date_uniqueness.py`—validates that every patient retains exactly one unique index date per calendar year."

---

## Slide 23: Cohort Validation: FAERS & Virginia APCD 21-Day Convergence
- **Section:** Module III · Cohort Validation · FAERS + Virginia APCD
- **Live Anchor:** https://pgx.jerome-dixon.io/23NOV_defense/#/slide-4c

"Bringing both distributions side-by-side here in Module III provides direct external-to-internal construct validation for our Cohort 2 polypharmacy pipeline. *Brief pause*  
On the left, external FDA FAERS pharmacovigilance across 468,298 serious acute ADE hospitalizations and deaths shows 87.7% occurring within 21 days.  
*Lean in / Vocal emphasis*  
On the right, when calibrated internally on 189,411 Virginia APCD emergency claims after enforcing our Zero-Day Discharge Guard, 92.4% of pre-ED fills—and 90.5% of all qualifying drug–ED pairs—fall within Days 1 to 21, with a median gap of 6 days. This tight FAERS-to-APCD convergence scientifically validates our 1-to-21-day proximal window for acute geriatric polypharmacy versus our 12-month trajectory window for chronic opioid escalation."

---

## Slide 24: Gradient-Boosted Trees & Model Selection
- **Section:** Module IV · Step 6
- **Live Anchor:** https://pgx.jerome-dixon.io/23NOV_defense/#/slide-5

"Turning to Slide 24, we present our Step 6 model selection protocol across three gradient-boosted tree architectures. *Brief pause*  
Standard logistic regression assumes log-odds linearity, failing under non-linear drug interactions, while deep neural networks risk severe overfitting on tabular claims with sparse, bursty features.  
**First**, we train three distinct tree geometries: CatBoost utilizes symmetric oblivious trees that evaluate identical feature splits across an entire depth level, providing mathematical immunity to target shift on high-cardinality NDC and ICD codes. XGBoost employs asymmetric depth-wise splits that isolate rare multi-drug interaction branches. And XGBoost Random Forest operates in parallel bagging mode to minimize prediction variance in sparse feature subsets.  
**Second**, models are trained across 25-split stratified Monte Carlo cross-validation. In imbalanced clinical prediction, AUROC is misleading because large true-negative counts inflate performance; therefore, Step 6 evaluates models strictly on Area Under the Precision-Recall Curve—PR-AUC—as the primary metric, with Recall at a 0.5 threshold as the tiebreaker.  
*Lean in / Vocal emphasis*  
**Third**, the top-performing model is selected for risk scoring, while the winning XGBoost variant and CatBoost are locked for Step 7 dual-model consensus attribution."

---

## Slide 25: Consensus Risk Attribution
- **Section:** Module IV · Consensus Risk Architecture
- **Live Anchor:** https://pgx.jerome-dixon.io/23NOV_defense/#/slide-6a

"Slide 25 details the core theoretical contribution of Aim 2: our Consensus-Risk attribution engine. *Brief pause*  
Standard impurity metrics like Gini gain fail under collinearity, and single explainable AI algorithms can split credit arbitrarily across correlated prescriptions. We resolve this with a three-stage dual-key security lock.  
**First**, we compute game-theoretic TreeSHAP attributions across both the winning CatBoost model and winning XGBoost variant, retaining only features that clear the 75th percentile in both architectures.  
**Second**, we compile the root-to-leaf paths of the winning XGBoost model into Conjunctive Normal Form clauses and apply the PySAT `Hitman` minimal hitting-set solver to extract irreducible Abductive Explanations—or AXPs—requiring rule support of at least 5% and confidence of at least 70%, combined with equal 50-50 weighting in combined synthesis. Think of TreeSHAP as an economist measuring marginal risk shifts across coalitions, and Formal Feature Attribution as a structural engineer proving the exact Boolean circuit that trips the breaker.  
*Lean in / Vocal emphasis*  
**Third**, to address Specific Aim 2 strictly via in-model consensus risk sensitivity contrasts ($\Delta\hat{p}$) from tree and AXP perturbations—expressed as consensus risk shifts rather than causal treatment effects—Intervention Rate measures single-feature predicted risk shift when set to its baseline median, while Interaction Effect uses second-order finite differences and Class-1 AXP co-occurrence lift above one to quantify super-additive model synergy."

---

## Slide 26: Target Leakage
- **Section:** Module IV · Step 3b
- **Live Anchor:** https://pgx.jerome-dixon.io/23NOV_defense/#/slide-7

"Slide 26 details our Step 3b temporal leakage audit—the engineering firewall that eliminates reverse causality. *Brief pause*  
Following an acute emergency overdose, patients receive immediate clinical rescue protocols—such as emergency naloxone administration, rapid buprenorphine induction, and point-of-care urine toxicology screens. In 30% of published studies, models inadvertently ingest these post-index treatments, inflating AUROC by 5 to 28 percentage points simply by detecting the emergency response after the patient arrives.  
**First**, Step 3b partitions every candidate code's timestamps around index date zero and computes its post-target ratio: the proportion of claims occurring on or after the emergency visit.  
**Second**, any feature with a post-target ratio of 80% or higher is systematically purged. Across 1,029 candidate features, Step 3b purged 348 post-index confounders—or 33.8%—including 164 medications such as Suboxone at 100% post-index, 84 CPT codes such as urine toxicology `80307` at 100% post-index, and 100 secondary diagnostic codes.  
*Lean in / Vocal emphasis*  
**Third**, the remaining 682 verified pre-index antecedents—where claim date strictly precedes index date—are locked in `safe_feature_filter.json` and applied symmetrically across cases and controls, cutting the movie frame strictly before the emergency room doors open."

---

## Slide 27: Aim 3: Actionable Intelligence & Workflow Integration
- **Section:** Module V · Step 10
- **Live Anchor:** https://pgx.jerome-dixon.io/23NOV_defense/#/slide-8

"Slide 27 presents Specific Aim 3: translating our validated modeling and attribution insights into a structured regimen-triage prototype built specifically for consultative clinical pharmacists. *Brief pause*  
Traditional EHR drug-interaction checkers fire synchronous pop-up interrupts during rapid prescribing encounters, suffering 90% to 95% clinician override rates. Rather than competing with the EHR, our microservice is architected for asynchronous pre-visit Medication Therapy Management, geriatric deprescribing consults, and emergency department post-overdose discharge handoffs via SMART on FHIR and CDS Hooks.  
**First**, our interface operationalizes the Clinical OODA Loop and a Lean value-stream layout—pairing partition-routed risk tiers with verifiable Boolean Horn rules while strictly separating population visualizations from patient-level scoring.  
**Second**, interactive 'What-If' profile comparisons display associational score shifts when candidate regimen features are toggled while preserving 100% human clinician authority, and our allele resolver enforces a hard prescribing lockout and CLIA/CAP laboratory referral whenever unphased direct-to-consumer genetic arrays lack chromosomal phase or structural variant resolution.  
*Lean in / Vocal emphasis*  
**Third**, we enforce a strict statutory demarcation between direct patient self-use under FTC consumer privacy—where no HIPAA BAA exists legally—and hospital clinical deployment under our two-tier BAA execution chain via AWS Artifact with KMS encryption, private VPC routing, and CloudWatch log scrubbing."

---

## Slide 28: Clinical OODA Loop
- **Section:** Module V · Translation & Deployment · Clinical OODA Loop Architecture · Chapters 1–5 End-to-End Integration
- **Live Anchor:** https://pgx.jerome-dixon.io/23NOV_defense/#/slide-ooda-loop

"Slide 28 synthesizes the entire dissertation architecture into the clinical OODA loop—Observe, Orient, Decide, and Act—emphasizing the iterative progression and feedback between phases. *Brief pause*  
**First**, under **Observe**, we bridge the triple gap in healthcare AI by scaling partition-first DuckDB ingestion across 1.8 terabytes of Virginia APCD claims—linking external CPIC pharmacogenomic reference data to quantify PGx burden and incorporating FDA FAERS safety kinetics to validate adverse event windows.  
**Second**, under **Orient**, we build our **Initial Model**—performing Step 3a exploratory feature screening across all 16 cohort and age strata, applying our dual-gated TreeSHAP and symbolic FFA consensus filter to isolate robust feature attributions, and enforcing the B1 balancing constraint to evaluate baseline transportability.  
**Third**, under **Decide**, we execute the critical iterative refinement step: confronting the initial model's vulnerability to target leakage, our Step 3b pre-index firewall purges reverse-causality rescue codes to construct the **Refined Model**. Here, we evaluate three distinct gradient-boosted tree geometries—CatBoost symmetric oblivious trees for categorical shift immunity, XGBoost asymmetric depth-wise trees for deep multi-drug interaction discovery, and XGBoost-RF bagging for variance stabilization—translating vetted patterns into clinical risk allocation and deprescribing priorities.  
**Fourth**, under **Act**, we deploy a stateless, HIPAA-compliant serverless decision-support dashboard delivering sub-100 millisecond inference to prioritize pharmacist medication therapy management reviews.  
*Lean in / Vocal emphasis*  
Finally, clinical interventions, pharmacist reviews, and longitudinal follow-up claims feed directly back into our continuous learning loop, retraining initial screening baselines and refining decision rules over time."

---

## Slide 29: Data Engineering Results: DuckDB & Cloud Economics
- **Section:** Module V · Results · Data Engineering (CH2) · J Big Data (Under Review)
- **Live Anchor:** https://pgx.jerome-dixon.io/23NOV_defense/#/slide-res-ch2

"Moving into our empirical results under Module V, we present findings across Chapters 2 through 5, beginning on Slide 29 with Chapter 2 Data Engineering. *Brief pause*  
Processing 1.8 terabytes and 380 million Virginia APCD claims on traditional distributed EMR and Apache Spark clusters triggered repeated out-of-memory failures (`exit code 137`) at a run rate of 23,000 dollars per month due to severe patient-level skew.  
**First**, our partition-first out-of-core DuckDB architecture on a single fat AWS `x2iedn.8xlarge` Spot EC2 instance achieved 18.1 million records per minute—a 15.1-fold speedup over monolithic execution—with zero partition failures across 189 consecutive passes.  
**Second**, idempotent S3 SHA-256 checkpoint manifests enabled sub-60-second Spot interruption recovery while collapsing compute cost by 99% to between 6 and 10 dollars per full statewide pass.  
*Lean in / Vocal emphasis*  
**Takeaway:** For skewed statewide healthcare claims, vectorized DuckDB on a single fat EC2 instance decisively beats distributed EMR clusters and Apache Spark on throughput, reliability, and cost."

---

## Slide 30: Data Engineering Results: Step 3b Leakage Purge & 5-Set CNF Filter
- **Section:** Module V · Results · Data Engineering (CH2) · Step 3b & Step 8
- **Live Anchor:** https://pgx.jerome-dixon.io/23NOV_defense/#/slide-res-ch2-leakage

"Slide 30 reports the empirical validation of our two core engineering gates in Chapter 2. *Brief pause*  
**First**, across 1,029 Step 3a candidate features, our Step 3b BupaR event-partitioning audit identified and purged 348 post-index confounders—33.8% of the candidate pool—including 164 rescue medications like Suboxone and Narcan occurring 100% on or after the emergency visit, 84 toxicology CPTs like `80307`, and 100 follow-up diagnostic codes. Locking the 682 verified pre-index antecedents in `safe_feature_filter.json` replaced artificially leaked 0.99 AUROC scores with honest out-of-time generalization.  
**Second**, our 5-Set candidate rule filter bounded the PySAT `Hitman` solver to achieve 100% patient explanation coverage across our matched opioid (`N = 207,350`) and polypharmacy (`N = 647,002`) cohorts."

---

## Slide 31: Opioid Cohort Results: 2019 Holdout & Two-Gate Mirror Ablation
- **Section:** Module V · Results · Opioid Cohort (CH3 · N = 207,350) · Published: Clin Transl Sci (10.1111/cts.70690)
- **Live Anchor:** https://pgx.jerome-dixon.io/23NOV_defense/#/slide-res-ch3-holdout

"Turning to our published Chapter 3 results in *Clinical and Translational Science* (`10.1111/cts.70690`) across 207,350 matched subjects in our chronic opioid cohort (`26,710` cases and `180,640` 1:5 controls). *Brief pause*  
**First**, evaluated strictly once on the untouched 2019 calendar holdout with zero post-hoc recalibration, our Step 6 winning architectures achieved out-of-time AUROCs from 0.800 in our primary `25–44` cohort up to 0.890 in `55–64`, with Precision-Recall lifts of 2.3-fold to 4.4-fold above baseline prevalence and Brier scores between 0.011 and 0.015.  
**Second**, our two-gate mirror ablation proved the operational value of intersecting TreeSHAP with symbolic FFA: retaining 20 out of 20 top clinical drivers while compressing feature review load by 61%—from 112 down to 44 features—and filtering boundary buprenorphine noise."

---

## Slide 32: Opioid Cohort Results: Consensus Drivers & 6.8-Month Window
- **Section:** Module V · Results · Opioid Cohort (CH3) · HOW & WHEN to Intervene
- **Live Anchor:** https://pgx.jerome-dixon.io/23NOV_defense/#/slide-res-ch3-drivers

"This slide answers both HOW and WHEN clinical pharmacists can intervene in the opioid cohort. *Brief pause*  
**First, HOW to intervene:** across adult strata, total drug burden (`pgx_num_drugs`, `100%` AXP support), `gabapentin` (`87%` support), benzodiazepines (`clonazepam`, `alprazolam`), and CPIC-actionable drug burden (`pgx_num_cpic_drugs`, present in 74% to 86% of cases) dominate consensus Horn rules—pinpointing synergistic CNS respiratory depression.  
**Second, WHEN to intervene:** exploratory Dynamic Time Warping across 59,813 patient trajectories uncovered a 6.8-month—or 27-week—escalation inflection window before emergency presentation, while `bupaR` process mining showed that initiating physical therapy prior to gabapentin escalation was associated with 58% lower progression to an opioid ED visit."

---

## Slide 33: Polypharmacy Cohort Results: Geriatric Holdout & Stress Test
- **Section:** Module V · Results · Polypharmacy Cohort (CH4 · N = 647,002) · Published: Clin Transl Sci (10.1111/cts.70718)
- **Live Anchor:** https://pgx.jerome-dixon.io/23NOV_defense/#/slide-res-ch4-holdout

"This slide presents our published Chapter 4 results (`10.1111/cts.70718`) evaluating acute 21-day geriatric polypharmacy ADEs across `N = 647,002` total patients, focusing on older adults aged 65 to 114. *Brief pause*  
**First**, on the untouched 2019 holdout, models achieved `0.877` AUROC (`3.9×` lift, Brier `0.008`, `89` consensus rules) in ages `65–74` and `0.858` AUROC (`4.2×` lift, `86` rules) in `75–84`, before attenuating to `0.686` (`1.6×` lift) in `85–114` due to unmeasured physiological frailty ceilings in claims data.  
**Second**, our `Table S5` utilization-free stress test delivered a striking methodological proof: removing all 17 utilization features collapsed univariate top-25 drug SHAP rankings to a Jaccard similarity of `0.00` in `65–74`, yet symbolic FFA Horn clauses preserved 60% (`3/5`) of core synergistic drug pairs—proving that Boolean minimal hitting sets anchor clinical stability when marginal Shapley credits shift."

---

## Slide 34: Polypharmacy Cohort Results: Multi-Drug Synergies & Z-Code Shield
- **Section:** Module V · Results · Polypharmacy Cohort (CH4) · IE, IR & Preventive Shield
- **Live Anchor:** https://pgx.jerome-dixon.io/23NOV_defense/#/slide-res-ch4-synergies

"This slide highlights three translational discoveries from Chapter 4. *Brief pause*  
**First**, beyond pairwise checkers, second-order interaction screening validated 115 synergistic pairs—led by fluoroquinolone-CNS combinations like `levofloxacin + lorazepam` (`IE = 11.9`)—and 312 Bonferroni-significant triplets out of 5,021 candidates, independently recovering the STOPP D4 'Triple Whammy' (`furosemide + HCTZ + lisinopril`) in 12.3% of oldest-old cases.  
**Second**, counterfactual in-model Intervention Rate ranked `simvastatin`, `furosemide`, and `alprazolam/lorazepam` as the top deprescribing review priorities—capturing high-prevalence metabolic and volume-depletion risks overlooked by static Beers anticholinergic lists.  
**Third**, we discovered a U-shaped Preventive Z-Code Shield: older adults in `Q2` (`1%–12%` ICD-10 `Z00` routine wellness exams, `Z23` preventive immunizations, and `Z79` long-term drug therapy documentation) experienced a 3.0% vs. 10.7% ED rate—an odds ratio of `0.25`, or 75% lower odds, before risk rises again in fragmented high-acuity `Q4` crisis coding.  
**Clinical & Q&A Clarification:** In claims data, `Z00/Z23/Z79` is not a billed patient navigator code itself—it is an observational proxy for **managed vs. unmanaged polypharmacy** (longitudinal PCP continuity and active medication reconciliation). Because observational engagement carries a healthy-user caveat (caregiver support, health literacy, care navigation), we frame this as an *association* rather than causal proof—where **patient navigators, care coordinators, and clinical pharmacists (MTM/CMR)** are the actionable translational workforce to move unengaged `Q0` patients into `Q2` structured stewardship."

---

## Slide 35: Consensus Attribution Results: SHAP ∩ FFA Sensitivity Analysis
- **Section:** Module V · Results · Clinical Alert Compression (CH3) · Methodological Invariance (CH4 Table S5)
- **Live Anchor:** https://pgx.jerome-dixon.io/23NOV_defense/#/slide-res-sensitivity

"Slide 35 synthesizes our peer-review sensitivity analyses across both cohorts. *Brief pause*  
**First**, in Chapter 3 (`25–44 low` bin, `181` whitelisted features), our two-gate mirror ablation addresses clinician alert fatigue: intersecting `TreeSHAP >= P75` with `PySAT` `FFA` (`Support >= 0.05, Confidence >= 0.70`) compresses review load by `61%`—from `112` down to `44` features—while preserving `100%` (`20/20`) of top clinical drivers (`pgx_num_drugs`, `gabapentin`, `clonazepam`, `pgx_num_cpic_drugs`), filtering `252` routine billing codes, rejecting sparse boundary buprenorphine correlates (`BUNAVAIL`, `BUTRANS`), and tightening cross-partition holdout Brier dispersion by `0.008` to `0.013`.  
*Lean in / Vocal emphasis*  
**Second**, in our Chapter 4 `Table S5` utilization-free stress test, stripping all `17` healthcare-utilization features and density strata in the `65–74` geriatric cohort collapses univariate top-25 drug `SHAP` rankings to `Jaccard = 0.00`, yet symbolic `PySAT Hitman` Horn rules preserve `60%` (`3/5`) of core synergistic drug pairs—visualized full-screen on the next slide."

---

## Slide 36: Consensus Attribution Results: Table S5 Stress Test Visual
- **Section:** Module V · Results · CH4 Table S5 · Utilization-Free Stress Test (2019 Holdout)
- **Live Anchor:** https://pgx.jerome-dixon.io/23NOV_defense/#/slide-res-sensitivity-fig

"Here on Figure 15, you can see the Table S5 utilization-free stress test side-by-side: even when stripping all 17 healthcare-utilization features causes univariate SHAP rankings to reshuffle completely (`Jaccard = 0.00`), our symbolic PySAT Hitman FFA clauses retain 60% (`3/5`) of the published multi-drug synergistic pairs—proving that Boolean minimal hitting sets anchor clinical stability when marginal Shapley credits shift."

---

## Slide 37: PGx Dashboard Results: Serverless Zero-PHI Architecture & Latency
- **Section:** Module V · Results · PGx Dashboard (CH5) · Published: Clin Transl Sci (10.1111/cts.70697)
- **Live Anchor:** https://pgx.jerome-dixon.io/23NOV_defense/#/slide-res-ch5-arch

"Turning to Slide 37, we present our published Chapter 5 results (`10.1111/cts.70697`) operationalizing Specific Aim 3 into a live serverless clinical decision-support prototype at `pgx.jerome-dixon.io`. *Brief pause*  
**First**, we packaged all 56 stratified ensembles—168 trained CatBoost, XGBoost, and XGBoost-RF base learners—alongside our CPIC lookup engine inside a single AWS Lambda container under 10 gigabytes, operating on a stateless `compute–respond–discard` lifecycle for under 5 dollars per month.  
**Second**, empirical benchmarking confirmed `6 millisecond` warm risk inference (`SD 1 ms`; `2,100 ms` cold start), `60 millisecond` Patient Card rendering with 100% rule-logic concordance across all 573 deterministic CPIC test cases, and bioinformatic safety lockouts on unphased consumer genetic arrays requiring CLIA/CAP clinical confirmation."

---

## Slide 38: PGx Dashboard Results: Imputation of Normality Sparsity Shield
- **Section:** Module V · Results · PGx Dashboard (CH5) · Bedside Input Sparsity Resilience
- **Live Anchor:** https://pgx.jerome-dixon.io/23NOV_defense/#/slide-res-ch5-sparsity

"Slide 38 validates how our population claims models remain stable when a consultative pharmacist enters only 3 to 5 active medications rather than 682 historical billing codes. *Brief pause*  
Under our 'Imputation of Normality' protocol, unentered features default cleanly so mean absolute probability drift remains below `0.06` across up to 70% input sparsity, with stable risk-band assignment through 60% missingness and 87% tier concordance at 50% sparsity—bridging population claims models directly to lean bedside entry."

---

## Slide 39: PGx Dashboard Results: Scenario Comparison & What-If Engine
- **Section:** Module V · Results · PGx Dashboard (CH5) · Live Two-Tab Scenario Architecture
- **Live Anchor:** https://pgx.jerome-dixon.io/23NOV_defense/#/slide-res-ch5-scenario

"Slide 39 shows a live capture from our deployed PGx Risk Dashboard at `pgx.jerome-dixon.io`, highlighting our decoupled two-tab scenario architecture. *Brief pause*  
On the left in Panel A—the **Risk Assessment** tab—pharmacists can swap codes directly via `Replace / swap a code`, save multiple candidate regimens including a one-click Care-Continuity What-If, and run side-by-side ensemble scoring via `POST /risk/comparison` alongside leave-one-out `Drug contributions`.  
*Lean in / Vocal emphasis*  
On the right in Panel B—the **Scenario Analysis (FFA/SHAP)** tab—we deliberately decouple attribution from risk recalculation: querying `GET /visualizations/scenario` renders three interactive Plotly views (`Top Interaction Factors (FFA)`, `SHAP Feature Importance`, and the per-feature `Effect on outcome` radar) with a second orange trace for any comma-separated What-If regimen—here comparing a 4-drug geriatric regimen (`furosemide`, `simvastatin`, `alprazolam`, `gabapentin`) against removing `alprazolam` and `gabapentin`—while preserving 100% human clinician authority over Decide and Act."

---

## Slide 40: Results: Executive Summary
- **Section:** Module V · Results · Synthesis Across Chapters 2–5 & Specific Aims 1–3 · 3 Published (Clin Transl Sci) · 1 Under Review (J Big Data)
- **Live Anchor:** https://pgx.jerome-dixon.io/23NOV_defense/#/slide-res-summary

"Slide 40 synthesizes our complete empirical results across Chapters 2 through 5 and Specific Aims 1 through 3 on a single executive matrix. *Brief pause*  
**First**, in Chapter 2 Data Engineering, vectorized DuckDB on a single fat EC2 instance beat distributed EMR clusters and Apache Spark—delivering a 15.1-fold speedup, zero shuffle-skew OOMs, and a 99% cloud cost reduction across 380 million claims—while purging 348 post-index leakage codes via Step 3b and bounding PySAT rule extraction via our 5-Set CNF filter.  
**Second**, in Chapter 3 (Specific Aim 1), leakage-free GBDTs achieved 0.800 to 0.890 AUROC and up to 4.4-fold Precision-Recall lift on the untouched 2019 opioid holdout, compressing pharmacist alert load by 61% while identifying modifiable CNS and CPIC burden 6.8 months before emergency presentation.  
**Third**, in Chapter 4 (Specific Aim 2), geriatric polypharmacy models achieved 0.877 and 0.858 AUROC in ages 65–84, proved 60% symbolic FFA pair invariance when univariate SHAP collapsed to zero Jaccard similarity, uncovered 312 higher-order drug triplets including the Triple Whammy, and quantified a 75% risk reduction under structured outpatient Z-code stewardship.  
*Lean in / Vocal emphasis*  
**Fourth**, in Chapter 5 (Specific Aim 3), our stateless AWS Lambda microservice operationalized all 168 trained base learners and 573 CPIC guidelines in 6 milliseconds of warm compute for under 5 dollars a month with 87% bedside sparsity stability—directly closing the Empty Intersection across all three Specific Aims."

---

## Slide 41: Next Steps
- **Section:** Module V · Next Steps · VCU Wright Center CCTR Endowment Pilot Grant (\$25,000 Direct Costs)
- **Live Anchor:** https://pgx.jerome-dixon.io/23NOV_defense/#/slide-next-steps

"Finally, Slide 41 outlines our immediate next steps through our VCU Wright Center CCTR Endowment Pilot Grant—25,000 dollars in direct costs—structured across three concrete aims. *Brief pause*  
**First**, in CCTR Aim 1, we extend prospective temporal validation onto updated 2020 through 2024 Virginia APCD holdings while partnering with VCU Health InfoSec, Massey Informatics, and OVPRI Counsel to complete a third-party AWS zero-PHI security audit and execute our two-tier AWS Artifact BAA and private VPC enclave.  
**Second**, in CCTR Aim 2, we conduct a structured human-factors usability study with 12 licensed clinical pharmacists executing MTM triage on `pgx.jerome-dixon.io`—targeting a mean System Usability Scale score above 80 and under 30 seconds per prioritized alert while advancing our asynchronous SMART on FHIR and CDS Hooks integration path for Epic Willow Ambulatory.  
*Lean in / Vocal emphasis*  
**Third**, in CCTR Aim 3, we package Aim 1 temporal validation, third-party security certification, and Aim 2 pharmacist usability outcomes with VCU Innovation Gateway to submit competitive extramural applications to the VIPC Commonwealth Commercialization Fund, NSF SBIR Phase I (275,000 dollars), and MTEC by Q2 2027."

---

## Slide 42: Thank You & Acknowledgements
- **Section:** Integrative Life Sciences Ph.D. Defense · Virginia Commonwealth University · November 23, 2026
- **Live Anchor:** https://pgx.jerome-dixon.io/23NOV_defense/#/slide-acknowledgements

"Thank you to my Dissertation Chair Dr. Elvin Price, my supervisory committee members Professor Franklin Bost, Dr. Tamas Gal, Dr. Dayanjan Wijesinghe, and Dr. Jonathan DeShazo, special thanks to Dr. Michael Neale—for first introducing me to the Integrative Life Sciences Ph.D. program—along with Dr. Hermine Maes and Dr. Tim York in Human Genetics and VIPBG, Dr. Amy Olex at the VCU Wright Center, Dr. Stephen Fong and the ILS leadership, our colleagues in the VCU School of Pharmacy and Price Lab, our data partners at Virginia Health Information and the Virginia Center for Health Innovation, and above all, my family, friends, work colleagues, and mentors for making this work possible."

---

## Slide 43: Questions & Discussion
- **Section:** Defense
- **Live Anchor:** https://pgx.jerome-dixon.io/23NOV_defense/#/slide-qa

"I welcome the committee's questions and discussion."

---

## Slide 44: Appendix: Technical Defenses & Supplemental Evidence
- **Section:** Appendix
- **Live Anchor:** https://pgx.jerome-dixon.io/23NOV_defense/#/sec-appendix

"This section serves as our interactive Technical Appendix index across Appendices A1 through A9—covering Formal Feature Attribution & PySAT AXP bounds (A1), Step 3b temporal event partitioning (A2), Step 5 CPIC drug mapping (A3), our reproducible notebook pipeline (A4), formal Aim 1–3 estimators (A5), master nomenclature (A6), two-tier HIPAA & FTC cloud architecture (A7), cohort-specific PharmGKB VIP gene networks (A8), and our 6-chapter Dissertation Portfolio & Publication Status (A9)."

---

## Slide 45: Appendix A1: Formal Feature Attribution (FFA) & PySAT AXP Bounds
- **Section:** Specific Aim 2 · Steps 6–8 & 10 · Mathematical Verification & PySAT AXP
- **Live Anchor:** https://pgx.jerome-dixon.io/23NOV_defense/#/sec-appendix-ffa

Slide A1 provides the mathematical and empirical defense for our Consensus-Risk architecture across Steps 6 through 8 and Step 10. Because CatBoost uses internal target-statistic hashing on categorical variables, Step 7 first computes exact TreeSHAP on both the best CatBoost model and the best XGBoost variant, enforcing their strict feature intersection as a cross-architecture gate. Step 8 then explodes the decision trees of the best XGBoost variant into CNF clauses and prioritizes the top SHAP-supported matched rules into the PySAT Hitman minimal hitting-set solver to extract exact Abductive Explanations in polynomial O(T times L) time. Importantly, our Consensus-Risk thresholds—75th percentile SHAP, support of 0.05, and confidence of 0.70—operate as a validated precision-volume dial for clinical review capacity, proven empirically in Module V Results: first, our Two-Gate Mirror Ablation on Slide 31 retained 20 out of 20 core clinical risk drivers while compressing alert volume by 61 percent and rejecting 252 routine billing codes; and second, our Table S5 utilization-free stress test on Slide 33 demonstrated 60 percent symbolic FFA drug-pair retention even when univariate SHAP Jaccard similarity collapsed to 0.00.

---

## Slide 46: Appendix A2: Step 3b Temporal Event Partitioning & Leakage Audit
- **Section:** Specific Aim 1 · Step 3b · Temporal Process Audit
- **Live Anchor:** https://pgx.jerome-dixon.io/23NOV_defense/#/sec-appendix-leakage

Slide A2 details our Step 3b temporal event-partitioning protocol. We classify post-index contamination into three clinical vectors: post-overdose OUD rescue and MAT induction, confirmatory ED and follow-up toxicology screens, and downstream administrative follow-up codes. Any candidate feature whose claims events occur 80% or more on or after the index timestamp t0 is automatically purged, locking an immutable pre-index antecedent whitelist in safe_feature_filter.json. As demonstrated empirically in Module V on Slide 25, this Step 3b audit purged 348 post-index confounders out of 1,029 candidate features and locked 682 verified pre-index antecedents—replacing artificially leaked 0.99 AUROC scores with honest 0.800 to 0.890 out-of-time 2019 holdout generalization.

---

## Slide 47: Appendix A3: Step 5 CPIC Drug Mapping & Patient-Level Count Features
- **Section:** Specific Aim 1 · Step 5 (5_pgx_analysis) · Claims Exposure vs. Genotype Defense
- **Live Anchor:** https://pgx.jerome-dixon.io/23NOV_defense/#/sec-appendix-pgx

Slide A3 details our Step 5 pharmacogenomic feature engineering pipeline in `5_pgx_analysis` alongside Chapter 6's bioinformatic phasing boundaries. First, `build_global_drug_cpic_mapping.py` maps our APCD drug vocabulary against the official CPIC directory (`573` gene-drug pairs) at a 95% threshold to build `drug_cpic_mapping_global.csv`. Second, `create_pgx_features_patient_level.py` computes two observable exposure counts per patient in DuckDB: `pgx_num_drugs` and `pgx_num_cpic_drugs`. Third, administrative race checkboxes are strictly excluded because pharmacogenomic allele prevalence tracks biogeographical genetic ancestry requiring direct genotyping. Finally, in our Aim 3 Patient Card, `cpic_allele_resolver.py` (`cpic-unphased-exploratory-v2`) never defaults unprobed loci to wild-type `*1/*1`: unphased consumer microarrays with compound heterozygotes ($2^{k-1}$ chromosomal phases) or unassessed `CYP2D6` structural variants trigger `INSUFFICIENT_GENOTYPE_RESOLUTION` and a mandatory CLIA/CAP laboratory test referral, whereas phased clinical alleles execute in 60 milliseconds with 100% concordance across 573 test cases.

---

## Slide 48: Appendix A4: Computational Pipeline & Notebook Architecture
- **Section:** Aim 1 · Aim 2 · Aim 3 · Full Traceability
- **Live Anchor:** https://pgx.jerome-dixon.io/23NOV_defense/#/sec-appendix-notebooks

Slide A4 details the computational reproducibility of the dissertation. All ten analytical steps across our three specific aims are operationalized into seven sequential Jupyter notebooks in our public GitHub repository. The workflow uses S3-backed state checkpoints for idempotency and runs seamlessly on both local workstations and AWS EC2 spot instances. Every dataset transformation, hyperparameter search, and model artifact is fully versioned and traceable back to the raw APCD extracts.

---

## Slide 49: Appendix A5: Specific Aims, Methodological Estimators & Evaluation Metrics
- **Section:** Specific Aim 1 · Specific Aim 2 · Specific Aim 3 · Formal Methodological & Mathematical Specification
- **Live Anchor:** https://pgx.jerome-dixon.io/23NOV_defense/#/sec-appendix-metrics

Slide A5 provides a single-slide formal mathematical specification across Specific Aims 1 through 3, documenting the pre-specified study design, data transformation, and formal estimator for every pipeline metric—and explicitly distinguishing supervised predictive and explanatory estimators (TreeSHAP, AXP, IR, IE) from exploratory visual analytics (DTW, FP-Growth, bupaR). All pre-specified estimators and post-execution sensitivity discoveries are evaluated empirically on the untouched 2019 calendar holdout in Module V Results (Slides 24 through 33).

---

## Slide 50: Appendix A6: Master Acronym & Technical Nomenclature Reference
- **Section:** Governance: CRISP-DM · TRIPOD+AI · OODA · APCD · HCG (Milliman) · CPIC PGx · GBDT · TreeSHAP · PySAT FFA/AXP · Serverless CDS & MTM
- **Live Anchor:** https://pgx.jerome-dixon.io/23NOV_defense/#/sec-appendix-acronyms

Appendix A6 provides a comprehensive quick-reference glossary of all governance, clinical, biostatistical, machine learning, formal logic, and cloud engineering acronyms used across the seven scientific modules of the dissertation—including CRISP-DM, TRIPOD+AI, the Clinical OODA Loop, PROBAST, Virginia APCD, Milliman Health Cost Guidelines (`HCG`: `hcg_setting`, `hcg_line`, `hcg_detail` emergency acuity codes `P51b`, `O11`, `P33`), CPIC pharmacogenomics, Silent ADEs, GBDT tree geometries, TreeSHAP, Formal Feature Attribution (FFA), Conjunctive Normal Form (CNF), Abductive Explanations (AXPs), Intervention Rate (IR), Interaction Effect (IE), and our zero-PHI Medication Therapy Management (MTM) Clinical Decision Support (CDS) prototype.

---

## Slide 51: Appendix A7: Regulatory Scope Demarcation & HIPAA Cloud Architecture
- **Section:** Specific Aim 3 · Clinical Deployment & Translation · Regulatory Scope: FTC Act vs. HIPAA BAA · AWS HIPAA-Eligible Enclave & Shared Responsibility
- **Live Anchor:** https://pgx.jerome-dixon.io/23NOV_defense/#/sec-appendix-hipaa

Slide A7 addresses the regulatory, legal, and cloud-security architecture formalized in Chapter 6.  
First, we enforce a strict statutory demarcation: when an individual patient enters their own medication list or uploads consumer genetic files for personal health literacy, 45 CFR Section 160.103 dictates that HIPAA does not apply—that relationship falls under Federal Trade Commission Act Section 5 and the Health Breach Notification Rule, where a HIPAA BAA is neither required nor legally recognized.  
Second, when deployed by a healthcare provider or hospital processing ePHI, compliance requires an unbroken two-tier legal BAA execution chain: the hospital signs BAA number 1 with our legal entity, and our legal entity executes BAA number 2 with AWS via AWS Artifact.  
Third, under the AWS Shared Responsibility Model, we harden HIPAA-eligible primitives into a compliant enclave via KMS encryption, Firecracker microVM zero `/tmp` persistence inside private VPC and PrivateLink subnets, strict CloudWatch log scrubbing, and multi-region CloudTrail auditing—while requiring separate signed BAAs for any external third-party AI APIs outside AWS Bedrock or SageMaker.  
Finally, Aim 2 of our VCU Wright Center CCTR 25,000-dollar Pilot Grant allocates 4,500 dollars to partner with VCU Health Information Security, Massey Cancer Center Informatics Governance, and OVPRI Legal Counsel to vet and execute all institutional BAA and DUA templates before clinical pilot testing.

---

## Slide 52: Appendix A8: Pharmacogenomic Gene Markers & Mechanistic Networks by Cohort
- **Section:** Specific Aim 1 · CPIC & PharmGKB VIP Mechanistic Evidence · Opioid Cohort: CYP2D6, OPRM1 · Polypharmacy Cohort: CYP2C19, CYP2C9, SLCO1B1, VKORC1
- **Live Anchor:** https://pgx.jerome-dixon.io/23NOV_defense/#/sec-appendix-gene-markers

Slide A8 provides the deep mechanistic and biological defense for the specific pharmacogenomic gene markers identified across our two clinical cohorts, directly synthesizing PharmGKB Very Important Pharmacogene (VIP) pathways and CPIC Level A guidelines.
In the Opioid Cohort, we focus on the pharmacokinetic anchor CYP2D6 in hepatic microsomes and the pharmacodynamic anchor OPRM1 in the central nervous system. CYP2D6 controls the rate-limiting phase I bioactivation of prodrug opioids—specifically O-demethylating codeine to morphine and tramadol to O-desmethyltramadol, which possesses over one-hundred-fold higher affinity for the mu-opioid receptor. Downstream, OPRM1 governs the Gi/Go protein-coupled receptor cascade across brainstem respiratory pacemakers like the pre-Bötzinger complex and mesolimbic reward pathways. The clinical phenotype is bidirectionally lethal: Ultra-rapid Metabolizers convert prodrugs at accelerated rates, causing a sudden toxic surge in active mu-agonists and fatal respiratory depression at standard doses; conversely, Poor Metabolizers carrying null alleles or patients with the OPRM1 118G variant experience profound analgesic failure, driving dose escalation, accidental toxicity, and ED encounters. Supporting shunt enzymes like CYP3A4 and glucuronidation clearance via UGT2B7 modulate systemic exposure.
In the Polypharmacy Cohort, adverse drug events in older adults are driven by a multi-target network across hepatic clearance, transport, and coagulation. First, CYP2C19 loss-of-function star alleles (*2 and *3) impair the bioactivation of clopidogrel into its active thiol metabolite, failing to inhibit platelet P2RY12 and precipitating stent thrombosis and major adverse cardiovascular events. Second, S-warfarin clearance is governed stereoselectively by hepatic CYP2C9, while its therapeutic target is VKORC1 in the endoplasmic reticulum; patients combining CYP2C9 reduced-function variants with the VKORC1 -1639A low-expression promoter variant experience severe clearance failure and extreme target sensitivity, driving supratherapeutic INR and catastrophic hemorrhage. Finally, SLCO1B1 encodes the OATP1B1 sinusoidal influx transporter; the star 5 variant halts hepatic statin uptake, trapping statins in systemic circulation where they reach toxic skeletal muscle concentrations, triggering severe statin-induced myopathy and rhabdomyolysis.

---

## Slide 53: Appendix A9: Dissertation Portfolio & Status
- **Section:** Appendix A9 · 6-Chapter Manuscript Portfolio & Publication Status
- **Live Anchor:** https://pgx.jerome-dixon.io/23NOV_defense/#/sec-appendix-portfolio

"Appendix A9 summarizes our cohesive six-chapter manuscript portfolio bridging peer-reviewed publications with our complete dissertation synthesis. *Brief pause*  
**First**, on the left, our three empirical chapters representing Specific Aims 1, 2, and 3 are published peer-reviewed articles in *Clinical and Translational Science*: Chapter 3 on prescription opioid ED prediction and trajectory mapping, Chapter 4 on our polypharmacy FFA interaction framework, and Chapter 5 on our serverless PGx Risk Dashboard prototype.  
**Second**, on the right, Chapter 1 is complete and in preparation for an invited submission to *Pharmacogenomics* (`PROSPERO CRD420261354089`), Chapter 2 is under consideration at the *Journal of Big Data* formalizing our DuckDB fat-node architecture and Consensus Attribution Filter, and Chapter 6 completes our dissertation synthesis—answering all ten research questions across three evidentiary tiers alongside Commonwealth surveillance alignment and regulatory governance.  
*Lean in / Vocal emphasis*  
**Third**, all six dissertation chapters and empirical 2019 holdout validations are complete for today's final dissertation defense."

---
