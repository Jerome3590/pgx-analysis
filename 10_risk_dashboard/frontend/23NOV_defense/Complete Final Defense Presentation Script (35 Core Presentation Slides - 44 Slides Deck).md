### Complete Final Defense Presentation Script (35 Core Presentation Slides / 44 Slides Deck) & Strategic Defense Guide: Gradient-Boosted Trees and Consensus-Risk For Opioid and Polypharmacy Emergency Department Visits in Virginia’s APCD

> **Duplex Podium Print Layout (November 23, 2026 Final Defense Edition):** 3 Slide Cards + 1 Notes Footer Per Page • **Print on Both Sides (Flip on Long Edge)** • **16 Pages Total = 8 Physical Sheets** (*Sheets 1–6 = 35 Core Presentation Slides + Pacing & Routing Card; Sheets 7–8 = Appendix Slides A1–A8 + Master Committee Q&A Playbook*)

#### Master Final Defense Execution Strategy & Pacing Architecture

* **Final Dissertation Defense Scope (November 23, 2026):** Presents the complete end-to-end 6-chapter dissertation across **7 Scientific Modules (`Slides 1–35`, ~42 minutes paced delivery)**—uniting clinical problem formulation & PRISMA/PROBAST literature synthesis (`Module I`), 3-ring governance & Virginia APCD architecture (`Module II`), dual-cohort engineering (`Module III`), tri-model GBDT & `TreeSHAP ∩ FFA` attribution with Step 3b leakage protection (`Module IV`), Clinical OODA Loop workflow & 8 study safeguards (`Module V`), **complete 2019 prospective holdout empirical results across Chapters 2–5 (`Module VI`, Slides 24–32)**, and **post-November 9 committee feedback integration, VCU CCTR pilot trial, 4-state APCD federation, and FDA SaMD governance (`Module VII`, Slides 33–35)**.
* **Technical Backup Bank (`Slides 36–44` / `Appendices A1–A8`):** Pre-staged on-demand mathematical, computational, regulatory/HIPAA, and molecular pharmacogenomic proofs linked directly via interactive buttons.
* **Single Source of Truth Synchronization:** Every spoken presenter script in this document is synchronized 1-to-1 with the `::: {.notes}` speaker blocks inside `final_defense_slides.qmd` and `final_defense_slides.html`.

---

## PAGE 1 OF 16 — SHEET 1 OF 8 (FRONT / RECTO (Odd Page)) • SLIDES 1–3 + NOTES

#### Slide 1: Title Slide — Gradient-Boosted Trees and Consensus-Risk in Virginia's APCD

* **Module Context:** Orientation — Doctoral Final Dissertation Defense (Nov 23, 2026)  
* **Target Pacing:** `0:45 (00:45)`

##### Spoken Presenter Script

"Good afternoon, members of the committee. *Brief pause*  
My name is Jerome Dixon. I am a student in the Integrative Life Sciences Doctoral Program, conducting my dissertation research within the School of Pharmacy at Virginia Commonwealth University.  
Welcome to my final dissertation defense: *Gradient-Boosted Trees and Consensus-Risk For Opioid and Polypharmacy Emergency Department Visits in Virginia’s APCD*.  
Across emergency medicine and outpatient pharmacotherapy, clinicians face an escalating dual crisis: the ongoing mortality of the opioid epidemic and the surging, silent morbidity of geriatric polypharmacy adverse drug events - ADEs. Today, I will present our computational framework addressing this challenge.  
This research presents an end-to-end architecture that connects population-scale predictive analytics with explainable clinical decision support.  
Using longitudinal statewide claims from the Virginia All-Payer Claims Database, the architecture integrates gradient-boosted tree ensembles across multiple topologies, rigorous temporal leakage audits, and a dual-model consensus attribution framework combining SHAP values with formal symbolic logic.  
*Lean in / Vocal emphasis*  
Our ultimate aim is to support actionable insights at the point of care while minimizing unnecessary alerts and clinician alert fatigue.  
With that, let us proceed on our journey."

##### Podium Delivery & Visual Callout

* Open calmly with authoritative poise; pause briefly after greeting the committee; emphasize actionable bedside decision support without alert fatigue.

##### Strategic Defense Tips & Committee Q&A Pivots

* **Core Thesis Anchor:** Connects 1.8 TB statewide Virginia APCD claims (`2016–2019`) with tri-model GBDTs, Step 3b temporal leakage protection, `TreeSHAP ∩ FFA` symbolic Horn rules, and a 6 ms serverless CDS prototype.
* **Three Published *CTS* Empirical Pillars:** Ch. 3 (`cts.70690`), Ch. 4 (`cts.70718`), and Ch. 5 (`cts.70697`).

##### Transition

*Advance to Slide 2 to acknowledge the Supervisory Committee and multidisciplinary alignment.*

#### Slide 2: Supervisory Committee & Multidisciplinary Research Alignment

* **Module Context:** Orientation — Faculty Mentorship & Cross-Disciplinary Governance  
* **Target Pacing:** `0:45 (01:30)`

##### Spoken Presenter Script

"Before we dive into the core modules, I want to express my sincere gratitude to:  
my chair and sponsor, Dr. Price, as well as Professor Bost, Dr. Gal, Dr. Wijesinghe, and Dr. DeShazo. *Brief pause*  
Your time, example, and mentorship has been invaluable throughout this journey. Your combined expertise across pharmacotherapy, systems engineering, informatics, and real-world evidence is vital to this work.  
*Lean in / Vocal emphasis*  
Thank you all for being here today for this final dissertation defense."

##### Podium Delivery & Visual Callout

* Make warm eye contact with each committee member by name; keep gratitude concise and sincere.

##### Strategic Defense Tips & Committee Q&A Pivots

* **Question Routing Map:** Route CPIC/clinical pharmacology to **Dr. Price & Dr. Wijesinghe**; DuckDB/leakage/biostatistics to **Dr. Gal**; symbolic CNF/AWS/SaMD to **Prof. Bost**; APCD RWE/EHR comparison to **Dr. DeShazo**.

##### Transition

*Move to Slide 3 to walk through the 7-module Final Defense Roadmap.*

#### Slide 3: Final Dissertation Defense Roadmap (Modules I–VII)

* **Module Context:** Orientation — 7-Module End-to-End Defense Agenda  
* **Target Pacing:** `1:00 (02:30)`

##### Spoken Presenter Script

"Here on our Roadmap slide, we outline the final defense agenda across seven core modules: *Brief pause*  
**Modules I through V** establish our scientific problem formulation, three-tier governance and Virginia APCD data architecture, dual cohort derivation pipelines, GBDT and `TreeSHAP ∩ FFA` consensus modeling with Step 3b leakage protection, and Clinical OODA Loop translation.  
**Module VI: Results** presents the complete empirical findings across **Data Engineering (Chapter 2)**, the **Opioid Cohort (Chapter 3)**, the **Polypharmacy Cohort (Chapter 4)**, and the **PGx Risk Dashboard (Chapter 5)**.  
*Lean in / Vocal emphasis*  
**Module VII: Next Steps** synthesizes pending committee feedback from our November 9 Candidacy presentation alongside our prospective VCU CCTR pilot trial, multi-state APCD federation, and FDA Software as a Medical Device governance roadmap."

##### Podium Delivery & Visual Callout

* Signpost the progression from architecture (Modules I–V) into empirical 2019 holdout results (Module VI) and post-9NOV synthesis (Module VII).

##### Strategic Defense Tips & Committee Q&A Pivots

* **45-Minute Time Budget:** Modules I–V (`~24 min`, Slides 1–23) &rarr; Module VI Empirical Results (`~16 min`, Slides 24–32) &rarr; Module VII Next Steps & Closeout (`~4 min`, Slides 33–35).

##### Transition

*Enter Module I on Slide 4 with the national clinical and economic burden of opioid and polypharmacy ED visits.*

#### Notes Section (Slides 1–3 — Sheet 1 FRONT / RECTO (Odd Page))

- [ ] **1. Pacing / Delivery Check (Slides 1–3):** _____________________________________________________________________
- [ ] **2. Committee Question / Feedback (Price / Gal / Bost / Wijesinghe / DeShazo):** ___________________________
- [ ] **3. ETD Read-Copy Action Item:** ___________________________________________________________________________

---

## PAGE 2 OF 16 — SHEET 1 OF 8 (BACK / VERSO (Even Page)) • SLIDES 4–6 + NOTES

#### Slide 4: Module I — Downstream Clinical & Economic Effects (#slide-1)

* **Module Context:** Module I (Ch. 1) — Dual Medication-Related ED Burden  
* **Target Pacing:** `1:15 (03:45)`

##### Spoken Presenter Script

"Turning to Slide 4, we begin Module I by examining the downstream clinical and economic effects of medication-related morbidity. *Brief pause*  
Preventable emergency department utilization in the United States is driven by two distinct medication-risk cohorts with verified national epidemiological benchmarks:  
**First**, in Cohort 1, CDC WONDER epidemiological surveillance documents that since 1999, prescription and illicit opioid overdoses have claimed over 500,000 lives in the United States, with Florence and colleagues' landmark CDC health-economics evaluation in *Medical Care* establishing an annual societal burden exceeding 78.5 billion dollars across direct healthcare ($26.1B), lost workplace productivity ($41.8B), and criminal justice expenditures ($7.6B).  
**Second**, in Cohort 2, Charlesworth et al. in the *Journals of Gerontology* and Kantor et al. in *JAMA* document that geriatric polypharmacy—defined as five or more concurrent prescription medications—affects over 40% of older adults aged 65 and older (with ~20% taking 10 or more), while CDC NEISS-CADES active surveillance published by Budnitz and Shehab in *JAMA* demonstrates that outpatient adverse drug events drive between 700,000 and 1.3 million emergency department visits and over 125,000 deaths annually."

##### Podium Delivery & Visual Callout

* Contrast the acute visibility of the opioid crisis against the pervasive, silent morbidity of geriatric polypharmacy.

##### Strategic Defense Tips & Committee Q&A Pivots

* **Literature Anchor:** Florence et al. (*Med Care*), Charlesworth et al. (*J Gerontol*), Kantor et al. (*JAMA*), Budnitz & Shehab (*JAMA* NEISS-CADES).
* **Why Two Cohorts:** Distinct age profiles, mechanisms, and onset kinetics prevent pooling.

##### Transition

*Advance to Slide 5 to show how machine learning literature disproportionately neglects polypharmacy.*

#### Slide 5: Module I — Opioid Research Overshadows Polypharmacy (#slide-1-imbalance)

* **Module Context:** Module I (Ch. 1) — PRISMA 2020 SQLR Literature Imbalance  
* **Target Pacing:** `1:00 (04:45)`

##### Spoken Presenter Script

"Slide 5 presents the second panel of our systematic review infographic: opioid research heavily overshadows polypharmacy research across 94 supervised machine learning studies. *Brief pause*  
**First**, 95% of published machine learning research clusters into just two buckets—46% predicting opioid use disorder diagnosis or dependence, and 49% predicting general acute care or overdose utilization—leaving only 5%, or just five studies, addressing polypharmacy adverse drug events and drug-drug interactions.  
**Second**, out of 1,473 published opioid use disorder literature records screened under PRISMA 2020, only 6 studies specifically evaluate machine learning models designed to predict opioid-related emergency department visits using administrative claims data."

##### Podium Delivery & Visual Callout

* Highlight the stark 95% vs. 5% imbalance across 94 supervised ML studies.

##### Strategic Defense Tips & Committee Q&A Pivots

* **Chapter 1 Provenance:** Invited systematic review at *Journal of Personalized Medicine* screening `9,571` PubMed records (`2015–March 2026`).

##### Transition

*Turn to Slide 6 to expose the three methodological failures across 73 PROBAST-assessed models.*

#### Slide 6: Module I — ML Gap: No PGx, No XAI, No Leakage Validation (#slide-1-empty)

* **Module Context:** Module I (Ch. 1) — PROBAST Audit & 'The Empty Intersection' (0 / 73)  
* **Target Pacing:** `1:00 (05:45)`

##### Spoken Presenter Script

"Slide 6 examines the third panel of our infographic: the three methodological failures uncovered across 73 PROBAST-assessed prediction models. *Brief pause*  
**First**, in Failure 1, 30% of published models suffer from temporal data leakage—allowing post-index treatments such as buprenorphine or naloxone to leak into pre-outcome predictor windows, artificially inflating AUROC by 5 to 25 percentage points before collapsing in out-of-time deployment.  
**Second**, in Failure 2, 81% completely omit pharmacogenomic modifiers despite mature CPIC guidelines for *CYP2D6* and *OPRM1*.  
*Lean in / Vocal emphasis*  
**Third**, in Failure 3, across all 73 PROBAST-assessed models, exactly zero combine structured pharmacogenomic features, explainable AI, and out-of-time temporal holdout validation."

##### Podium Delivery & Visual Callout

* Lean in on 'exactly zero out of 73'—this is Core Anchor #1 ('The Empty Intersection').

##### Strategic Defense Tips & Committee Q&A Pivots

* **Anchor 1 (Novelty Defense):** This dissertation is not 'just another risk score'—it is the first architecture uniting CPIC exposure counts, dual `TreeSHAP ∩ FFA` explainability, and untouched temporal holdout validation.

##### Transition

*Move to Slide 7 to examine our initial FAERS-to-APCD exploratory time-to-onset analysis.*

#### Notes Section (Slides 4–6 — Sheet 1 BACK / VERSO (Even Page))

- [ ] **1. Pacing / Delivery Check (Slides 4–6):** _____________________________________________________________________
- [ ] **2. Committee Question / Feedback (Price / Gal / Bost / Wijesinghe / DeShazo):** ___________________________
- [ ] **3. ETD Read-Copy Action Item:** ___________________________________________________________________________

---

## PAGE 3 OF 16 — SHEET 2 OF 8 (FRONT / RECTO (Odd Page)) • SLIDES 7–9 + NOTES

#### Slide 7: Module I — 90% of Acute ADEs Occur Within 21 Days (#slide-1-faers)

* **Module Context:** Module I (Step 1a) — External FAERS to Internal APCD Construct Validation  
* **Target Pacing:** `1:15 (07:00)`

##### Spoken Presenter Script

"Before moving to the frontline clinical bottleneck, Slides 7 through 9 present our initial exploratory analysis that shaped our pipeline architecture. *Brief pause*  
**First**, on Slide 7, we asked a foundational pharmacoepidemiologic question: over what pre-visit lookback window do acute adverse drug events actually occur?  
**Second**, across 468,298 deduplicated FDA FAERS hospitalization and death reports, 49.5% occur within the first 7 days and 87.7% occur within 21 days of drug initiation.  
*Lean in / Vocal emphasis*  
**Third**, when we validated this directly on Virginia APCD emergency claims—after excluding Day-0 discharge prescriptions—92.4% of pre-ED fills and 90.5% of all qualifying drug-ED pairs occurred within 1 to 21 days, with a median gap of 6 days. An unforeseen benefit of this FAERS-to-APCD time-to-onset convergence is that it provides empirical external-to-internal construct validation for our polypharmacy cohort pipeline—scientifically justifying our proximal 21-day window for acute polypharmacy versus our 12-month trajectory window for chronic opioid escalation."

##### Podium Delivery & Visual Callout

* Emphasize how external federal pharmacovigilance data converged with Virginia claims to justify the 21-day lookback.

##### Strategic Defense Tips & Committee Q&A Pivots

* **Why Not 12 Months for Polypharmacy?** In older adults taking &ge;5 drugs, a 365-day window dilutes acute triggering regimen changes with stale historical refills; 21 days captures >90% of true proximal ADE onset.

##### Transition

*Advance to Slide 8 for Step 3a multi-model exploratory feature screening across all 8 age bands.*

#### Slide 8: Module I — Initial Feature Analysis & Leakage Trap Discovery (#slide-1-fi-heatmap)

* **Module Context:** Module I (Step 3a) — 8-Band Feature Separation & 0.99 Leaked AUROC Discovery  
* **Target Pacing:** `1:15 (08:15)`

##### Spoken Presenter Script

"Slide 8 shows the second phase of our initial exploratory analysis: our Step 3a multi-model feature analysis across all 8 age bands from 0–12 through 85–114 and both target cohorts. *Brief pause*  
**First**, we pivoted event-level claims into patient-level numeric event-count matrices and evaluated all three gradient-boosted tree architectures—`CatBoost`, `XGBoost`, and `XGBoost-RF`—aggregating Recall-scaled feature importances across Monte Carlo cross-validation splits. Comparing Panel A (`opioid_ed`), Panel B (`non_opioid_ed`), and Panel C (target separation) demonstrates why these two cohorts cannot be pooled into a single model: `opioid_ed` features concentrate in young and middle-aged adults around chronic pain, psychiatric care, oxycodone, and gabapentin, whereas `non_opioid_ed` concentrates sharply in older adults aged 65 and above around cardiovascular, diuretic, metabolic, and statin polypharmacy.  
*Lean in / Vocal emphasis*  
**Second**, this multi-model feature analysis exposed a critical methodological trap: unfiltered Step 3a models achieved an artificially high 0.99 AUROC because synchronous toxicology screens (`G0479`), emergency visit CPTs (`99284`, `99285`), and post-event buprenorphine treatment leaked into the feature space—directly motivating our Step 3b BupaR process-mining leakage filter."

##### Podium Delivery & Visual Callout

* Point to Panel C target separation first, then deliver the 0.99 AUROC trap as the motivation for Step 3b.

##### Strategic Defense Tips & Committee Q&A Pivots

* **Methodological Honesty Anchor:** Catching our own 0.99 Step 3a leakage artifact directly motivated the automated Step 3b `bupaR` event-partitioning firewall.

##### Transition

*Turn to Slide 9 to map Step 3a drug predictors against official CPIC pharmacogenomic guidelines.*

#### Slide 9: Module I — CPIC Drugs Drive Initial Feature Importance (#slide-1-cpic-fi)

* **Module Context:** Module I (Step 5 Preview) — Population CPIC Exposure Burden vs. Genotyping  
* **Target Pacing:** `1:15 (09:30)`

##### Spoken Presenter Script

"Slide 9 presents the third finding from our initial exploratory analysis: mapping the drug features surfaced in our initial Feature Importance directly against Clinical Pharmacogenetics Implementation Consortium, or CPIC, gene-drug guidelines. *Brief pause*  
**First**, 24% to 35% of all drug features selected in initial feature importance are CPIC-actionable medications—spanning 187 distinct generic CPIC drugs in `opioid_ed` and 224 distinct generic CPIC drugs in `non_opioid_ed`, with 76% to 92% of prescription-exposed `opioid_ed` patients and 61% to 78% of `non_opioid_ed` patients taking at least one CPIC medication.  
**Second**, the specific CPIC drug classes separate by cohort: *CYP2D6* opioids and psychotropics dominate `opioid_ed`, whereas *CYP2D6* beta-blockers, *SLCO1B1* statins, *CYP2C19* proton-pump inhibitors, and *CYP2C9* ARBs/NSAIDs dominate `non_opioid_ed`.  
*Lean in / Vocal emphasis*  
**Third**, we calibrate our scientific framing explicitly: claims do not recover a patient's genetic diplotype; rather, they quantify cumulative CPIC-actionable medication exposure burden (`pgx_num_drugs` and `pgx_num_cpic_drugs`), which subsequently ranks among our top 3 predictors across nearly every age band and motivates addressing the frontline clinical bottleneck on the next slide."

##### Podium Delivery & Visual Callout

* State the claims boundary crisply: claims measure cumulative CPIC substrate exposure, never germline star alleles.

##### Strategic Defense Tips & Committee Q&A Pivots

* **Appendix Jump:** Pivot to **Appendix A3 (Slide 39)** for `build_global_drug_cpic_mapping.py` (95% fuzzy threshold) and **Appendix A8 (Slide 44)** for *CYP2D6 / OPRM1 / CYP2C19 / CYP2C9 / VKORC1 / SLCO1B1* mechanisms.

##### Transition

*Move to Slide 10 to frame the frontline clinical bottleneck at the point of care.*

#### Notes Section (Slides 7–9 — Sheet 2 FRONT / RECTO (Odd Page))

- [ ] **1. Pacing / Delivery Check (Slides 7–9):** _____________________________________________________________________
- [ ] **2. Committee Question / Feedback (Price / Gal / Bost / Wijesinghe / DeShazo):** ___________________________
- [ ] **3. ETD Read-Copy Action Item:** ___________________________________________________________________________

---

## PAGE 4 OF 16 — SHEET 2 OF 8 (BACK / VERSO (Even Page)) • SLIDES 10–12 + NOTES

#### Slide 10: Module I — The Frontline Clinical Bottleneck (#slide-1-bottleneck)

* **Module Context:** Module I — Alert Override Fatigue vs. Black-Box ML Distrust  
* **Target Pacing:** `1:00 (10:30)`

##### Spoken Presenter Script

"Slide 10 isolates the fourth component of our problem formulation: The Frontline Clinical Bottleneck. *Brief pause*  
**First**, frontline clinicians and pharmacists must navigate complex, multi-year medication and claims histories during rapid clinical visits, while traditional pairwise drug-interaction checkers trigger 90% to 95% alert override fatigue and miss higher-order multi-drug synergies.  
**Second**, black-box machine learning models that output raw probability scores without verifiable clinical rules fail to earn clinician trust at the point of care.  
*Lean in / Vocal emphasis*  
**Third**, our dissertation bridges this bottleneck with a calibrated, defensible impact: identifying and prioritizing potentially modifiable medication-related risk factors via leakage-free gradient-boosted trees and Consensus-Risk Horn rules to establish a framework for pharmacist-led medication review."

##### Podium Delivery & Visual Callout

* Keep brisk (~60s); frame pharmacist-led MTM prioritization as the translational target.

##### Strategic Defense Tips & Committee Q&A Pivots

* **Alert Fatigue Defense:** Combining `TreeSHAP >= P75` with `FFA (Support >= 0.05, Conf >= 0.70)` compresses clinician review lists by **61%** (`112 -> 44`) while keeping `20/20` core drivers (Slide 26).

##### Transition

*Advance to Slide 11 to formalize Specific Aims 1–3 alongside our two mutually exclusive cohorts.*

#### Slide 11: Module I — Specific Aims 1–3 & Dual Target Cohorts (#slide-1a)

* **Module Context:** Module I — Specific Aims Architecture (Chapters 3, 4 & 5)  
* **Target Pacing:** `1:30 (12:00)`

##### Spoken Presenter Script

"Slide 11 presents our three Specific Aims stacked on the left alongside our two mutually exclusive target cohorts on the right. *Brief pause*  
**First**, on the left, Aim 1 (Chapter 3) develops and temporally validates our tri-model gradient-boosted tree family with `SHAP ∩ FFA` consensus attribution for Cohort 1 (`opioid_ed`, 12-month lookback, ages 13 to 64) on the 2019 temporal holdout; Aim 2 (Chapter 4) extends this framework to Cohort 2 (`ed_non_opioid`, 21-day lookback, ages 65 to 114), incorporating symbolic Boolean Horn rules to quantify multi-drug interaction effects (`IE`) and consensus risk attenuation priorities (`IR`); and Aim 3 (Chapter 5) translates these validated findings into an interpretable, usable decision-support tool for pharmacist-led medication review prioritization—evaluating explanation fidelity and clinical usability rather than claiming clinical outcomes effectiveness.  
*Lean in / Vocal emphasis*  
**Second**, on the right, Cohort 1 models chronic opioid risk anchored to explicit ICD-10 `F11.20` and `T40.0X` through `T40.4X` codes across all ten diagnostic positions, whereas Cohort 2 captures acute geriatric polypharmacy and 'Silent ADEs' billed as downstream falls, fractures, syncope, delirium, bleeding, and acute kidney injury via Milliman HCG emergency lines `P51b`, `O11`, and `P33` with zero opioid codes."

##### Podium Delivery & Visual Callout

* Walk cleanly left-to-right: Aims 1–3 on the left, Cohort 1 vs. Cohort 2 Silent ADEs on the right.

##### Strategic Defense Tips & Committee Q&A Pivots

* **Silent ADE Definition:** Geriatric toxicity rarely carries `T36–T50` poisoning codes (<5%); >95% presents as syncope (`R55`), falls (`R29.6`), GI bleed (`K92.2`), delirium, or AKI (`N17`).

##### Transition

*Enter Module II on Slide 12 to establish our 3-ring governance framework.*

#### Slide 12: Module II — Governance and Frameworks (#slide-2)

* **Module Context:** Module II (Ch. 2) — CRISP-DM, TRIPOD+AI Type 2b/3 & Clinical OODA Loop  
* **Target Pacing:** `1:15 (13:15)`

##### Spoken Presenter Script

"Directing your attention to Slide 12, our engineering and validation pipeline is governed by three concentric architectural rings. *Brief pause*  
**First**, the outer ring executes the six-phase CRISP-DM systems engineering lifecycle, enforcing strict reproducibility across our ten-step codebase—from clinical problem formulation and DuckDB ingestion through Step 3b leakage filtering, tri-model GBDT fitting, and containerized deployment.  
**Second**, the middle ring enforces the 27-item TRIPOD+AI biomedical reporting standard for Type 2b and Type 3 independent temporal holdout validation. Models are trained and tuned exclusively on 2016 through 2018 Virginia APCD claims and evaluated once on an untouched 2019 calendar holdout. Crucially, we prohibit post-hoc recalibration on the 2019 holdout: adjusting probability curves after inspecting test-period outcomes is equivalent to grading your own exam after peeking at the answer key, which masks temporal calibration drift.  
*Lean in / Vocal emphasis*  
**Third**, the inner core operationalizes the Clinical OODA Loop—Observe, Orient, Decide, and Act. Our GBDT models and symbolic Horn rules execute Observe and Orient, while licensed clinical pharmacists retain 100% authority over Decide and Act within an asynchronous medication review queue."

##### Podium Delivery & Visual Callout

* Use the 'grading your own exam after peeking at the answer key' analogy for zero holdout recalibration.

##### Strategic Defense Tips & Committee Q&A Pivots

* **Why Quarantine 2020?** COVID-19 stay-at-home lockdowns distorted emergency utilization and prescribing patterns (*Ochalek, Gal et al., JAMA 2020*), making 2019 the cleanest pre-pandemic out-of-time test.

##### Transition

*Turn to Slide 13 to justify selecting statewide Virginia APCD claims over single-system EHRs.*

#### Notes Section (Slides 10–12 — Sheet 2 BACK / VERSO (Even Page))

- [ ] **1. Pacing / Delivery Check (Slides 10–12):** _____________________________________________________________________
- [ ] **2. Committee Question / Feedback (Price / Gal / Bost / Wijesinghe / DeShazo):** ___________________________
- [ ] **3. ETD Read-Copy Action Item:** ___________________________________________________________________________

---

## PAGE 5 OF 16 — SHEET 3 OF 8 (FRONT / RECTO (Odd Page)) • SLIDES 13–15 + NOTES

#### Slide 13: Module II — Why Virginia APCD Over EHR (#slide-3)

* **Module Context:** Module II (Ch. 2) — Statewide Outcome Capture & Adjudicated Pharmacy Fills  
* **Target Pacing:** `1:15 (14:30)`

##### Spoken Presenter Script

"Turning to Slide 13, we justify selecting the Virginia All-Payer Claims Database over Electronic Health Records or single-payer Medicare datasets. *Brief pause*  
Single-system EHRs suffer from critical outcome blindness in emergency prediction: when an individual overdoses or an older adult suffers a syncopal fall, emergency medical services transport them to the nearest hospital, not necessarily their home academic center.  
**First**, the Virginia APCD provides a multi-payer, statewide window encompassing commercial, Medicare Advantage, and Medicaid populations—eliminating out-of-network emergency blindness across competing health systems with zero linkage attrition via Virginia Health Information.  
**Second**, whereas EHRs record written e-prescriptions—missing the roughly 30% primary non-adherence drop-off abandoned at the pharmacy counter—APCD pharmacy claims capture paid, adjudicated dispenses that verify physical drug possession.  
*Lean in / Vocal emphasis*  
**Third**, while administrative claims lack raw DNA swabs and bedside vitals, we methodologically compensate in Step 5 (`5_pgx_analysis`) by mapping all dispensed APCD medications against the official CPIC directory (`drug_cpic_mapping_global.csv`) to determine whether each drug carries a CPIC guideline, computing patient-level counts of total drugs (`pgx_num_drugs`), CPIC-guideline drugs (`pgx_num_cpic_drugs`), and how many patients carry CPIC drugs—quantifying observable medication exposure burden directly without unscientific genotype imputation."

##### Podium Delivery & Visual Callout

* Use the EMS ambulance transport example—patients go to the nearest ED, not necessarily their home academic EHR.

##### Strategic Defense Tips & Committee Q&A Pivots

* **Data Provenance:** Authorized extract via Virginia Health Information (VHI) and Virginia Center for Health Innovation covering `380M` claim lines (`1.8 TB` Parquet).

##### Transition

*Enter Module III on Slide 14 to inspect the 39-column unified medical and pharmacy schema.*

#### Slide 14: Module III — Medical & Pharmacy Claims Schema (#slide-4)

* **Module Context:** Module III (Ch. 2) — 39-Column Gold Relational Join & Pre-Index Feature Families  
* **Target Pacing:** `1:15 (15:45)`

##### Spoken Presenter Script

"Before examining our two target cohorts on Slides 15 and 16, Slide 14 details how we validate, clean, and relationally join the Virginia APCD Gold medical and pharmacy tables into our 39-column unified event fact table. *Brief pause*  
To ensure administrative claims integrity prior to modeling, automated preprocessing scripts filter voided or reversed claim lines and harmonize schema fields across years.  
**First**, the Gold medical table contributes 25 standardized columns—capturing all ten ICD-10 diagnostic positions, nine ICD procedure and three CPT modifier fields, Milliman HCG emergency service lines `P51b`, `O11`, and `P33`, and imputed demographics—while the Gold pharmacy table captures paid prescription fills normalized from National Drug Codes to generic RxNorm `drug_name` and `therapeutic_class_1`.  
**Second**, in Phase 2, DuckDB links and unions these medical and pharmacy records at the patient level on `mi_person_key`, `event_year`, and `age_band`, backfilling demographic covariates across pharmacy rows and computing a global chronological `event_sequence` ordered by claim date relative to index date zero.  
*Lean in / Vocal emphasis*  
**Third**, from this strictly pre-index timeline—where claim service date precedes index date—we engineer our four core feature families: multi-position diagnosis and procedure indicators (`item_icd_*`, `item_cpt_*`), active generic drug exposures (`item_drug_*`), our Step 5 CPIC drug counts (`pgx_num_drugs`, `pgx_num_cpic_drugs`), and claims-density utilization tiers."

##### Podium Delivery & Visual Callout

* Emphasize void/reversal scrubbing and chronological `event_sequence` ordering strictly prior to $t_0$.

##### Strategic Defense Tips & Committee Q&A Pivots

* **Four Feature Families:** `item_icd_*` diagnoses, `item_cpt_*` procedures, `item_drug_*` generic exposures, and Step 5 CPIC exposure counts (`pgx_num_drugs`, `pgx_num_cpic_drugs`) + density strata.

##### Transition

*Advance to Slide 15 for Cohort 1 (`opioid_ed`) pipeline derivation.*

#### Slide 15: Module III — Cohort 1 Pipeline: Opioid-Related ED (`opioid_ed`) (#slide-4a)

* **Module Context:** Module III (Ch. 2–3) — 12-Month Chronic Opioid Escalation Funnel  
* **Target Pacing:** `1:15 (17:00)`

##### Spoken Presenter Script

"Slide 15 presents the architectural pipeline for Cohort 1, our 12-month chronic opioid emergency cohort, `opioid_ed`. *Brief pause*  
To capture true acute opioid toxicity and dependence crises across adolescents and adults aged 13 to 64, we anchor index encounters using validated ICD-10 diagnostic criteria.  
**First**, case inclusion requires an emergency claim carrying explicit ICD-10 `F11.20` opioid dependence or `T40.0X` through `T40.4X` acute opioid poisoning codes inspected across all ten diagnostic positions.  
**Second**, because opioid tolerance and dose escalation evolve longitudinally, we extract a 12-month pre-index lookback window—from day minus 365 through day minus 1—paired with 1-to-5 incidence-density matched controls across eight age bands, sex, and calendar year, and screened via our Step 3b leakage purge.  
*Lean in / Vocal emphasis*  
**Third**, to process 1.8 Terabytes and 380 million claims without memory exhaustion, we engineered a partition-first out-of-core DuckDB pipeline on AWS Spot EC2 `x2iedn.8xlarge` workers. Partitioning by Age Band and Event Year eliminated shuffle skew—achieving a 15.1-fold speedup to 18.1 million records per minute, zero failures across 189 passes, and reducing compute cost from 23,000 dollars a month on EMR clusters to under 10 dollars per pass."

##### Podium Delivery & Visual Callout

* Connect the 365-day lookback to chronic tolerance/escalation, and preview the DuckDB 15.1x engineering win.

##### Strategic Defense Tips & Committee Q&A Pivots

* **Incidence-Density Matching:** Exact 1:5 control matching stratified by `(age_band, sex, event_year)` prevents demographic confounding while preserving realistic class imbalance (~12.9%–16.7%).

##### Transition

*Turn to Slide 16 for the parallel geriatric polypharmacy pipeline (`ed_non_opioid`).*

#### Notes Section (Slides 13–15 — Sheet 3 FRONT / RECTO (Odd Page))

- [ ] **1. Pacing / Delivery Check (Slides 13–15):** _____________________________________________________________________
- [ ] **2. Committee Question / Feedback (Price / Gal / Bost / Wijesinghe / DeShazo):** ___________________________
- [ ] **3. ETD Read-Copy Action Item:** ___________________________________________________________________________

---

## PAGE 6 OF 16 — SHEET 3 OF 8 (BACK / VERSO (Even Page)) • SLIDES 16–18 + NOTES

#### Slide 16: Module III — Cohort 2 Pipeline: Polypharmacy ADEs (`ed_non_opioid`) (#slide-4b)

* **Module Context:** Module III (Ch. 2 & 4) — 3-Stage Geriatric Epidemiological Funnel & 0-Day Guard  
* **Target Pacing:** `1:30 (18:30)`

##### Spoken Presenter Script

"Turning to Slide 16, we examine the parallel pipeline architecture for our geriatric polypharmacy cohort, `ed_non_opioid`.  
*Brief pause*  
Focusing on older adults aged 65 to 114, we anchor index encounters using Milliman HCG emergency service lines—specifically `P51b`, `O11`, and `P33`. To isolate pure medication-related adverse events from opioid toxicity, we apply an absolute opioid cleanse: excluding any patient with even a single opioid ICD-10 code across all ten diagnostic positions during the study year.  
Next, we pass candidate records through a three-stage epidemiological funnel:  
**First**, a high-utilization cap excluding individuals with seven or more annual emergency visits, filtering out chronic crisis utilizers.  
**Second**, a 21-day proximal pharmacy window—from day minus 21 through day minus 1. As established on Slide 7, this lookback is empirically validated by our FAERS-to-APCD time-to-onset convergence demonstrating that over 90% of acute ADEs occur within 3 weeks of medication initiation or modification, providing external cohort validation for our polypharmacy pipeline—detailed in the Cohort Validation card and Figure 1B.  
*Lean in / Vocal emphasis*  
**Third**, we enforce a strict Zero-Day Discharge Guard where claim date precedes index date. This systematically purges any prescription filled on the calendar day of the emergency visit, guaranteeing that post-discharge medications are never mistaken for pre-visit etiology.  
*Brief pause*  
Cohort cases are matched 1-to-5 with controls across age band, sex, and calendar year. Finally, our verification script—`qa_index_date_uniqueness.py`—validates that every patient retains exactly one unique index date per calendar year."

##### Podium Delivery & Visual Callout

* Deliver the 3 funnel gates crisply; highlight the Zero-Day Discharge Guard (`t_claim < t_0`) and unique patient-year index check.

##### Strategic Defense Tips & Committee Q&A Pivots

* **Cross-Fold Patient Leakage Defense:** `qa_index_date_uniqueness.py` enforces 1 unique $t_0$ per `mi_person_key`, while `2016–2018` train vs. `2019` holdout blocks temporal split leakage.

##### Transition

*Enter Module IV on Slide 17 to examine our tri-model GBDT selection protocol.*

#### Slide 17: Module IV — Gradient-Boosted Trees & Model Selection (#slide-5)

* **Module Context:** Module IV (Step 6) — CatBoost, XGBoost & XGBoost-RF Topologies  
* **Target Pacing:** `1:15 (19:45)`

##### Spoken Presenter Script

"Turning to Slide 17, we present our Step 6 model selection protocol across three gradient-boosted tree architectures. *Brief pause*  
Standard logistic regression assumes log-odds linearity, failing under non-linear drug interactions, while deep neural networks risk severe overfitting on tabular claims with sparse, bursty features.  
**First**, we train three distinct tree geometries: CatBoost utilizes symmetric oblivious trees that evaluate identical feature splits across an entire depth level, providing mathematical immunity to target shift on high-cardinality NDC and ICD codes. XGBoost employs asymmetric depth-wise splits that isolate rare multi-drug interaction branches. And XGBoost Random Forest operates in parallel bagging mode to minimize prediction variance in sparse feature subsets.  
**Second**, models are trained across 25-split stratified Monte Carlo cross-validation. In imbalanced clinical prediction, AUROC is misleading because large true-negative counts inflate performance; therefore, Step 6 evaluates models strictly on Area Under the Precision-Recall Curve—PR-AUC—as the primary metric, with Recall at a 0.5 threshold as the tiebreaker.  
*Lean in / Vocal emphasis*  
**Third**, the top-performing model is selected for risk scoring, while the winning XGBoost variant and CatBoost are locked for Step 7 dual-model consensus attribution."

##### Podium Delivery & Visual Callout

* Explain why PR-AUC beats AUROC under 1:5 class imbalance (true negatives don't inflate precision-recall).

##### Strategic Defense Tips & Committee Q&A Pivots

* **Why Not Deep Learning?** On sparse, skewed tabular healthcare claims, GBDTs outperform deep neural nets while compiling directly into exact polynomial-time $O(T \cdot L)$ Boolean CNF formulas.

##### Transition

*Advance to Slide 18 for the core methodological contribution of Aim 2: Consensus-Risk Attribution.*

#### Slide 18: Module IV — Consensus Risk Attribution (`TreeSHAP ∩ FFA`) (#slide-6a)

* **Module Context:** Module IV (Steps 7–8) — Dual-Model SHAP + Symbolic PySAT Hitman AXPs  
* **Target Pacing:** `1:30 (21:15)`

##### Spoken Presenter Script

"Slide 18 details the core theoretical contribution of Aim 2: our Consensus-Risk attribution engine. *Brief pause*  
Standard impurity metrics like Gini gain fail under collinearity, and single explainable AI algorithms can split credit arbitrarily across correlated prescriptions. We resolve this with a three-stage dual-key security lock.  
**First**, we compute game-theoretic TreeSHAP attributions across both the winning CatBoost model and winning XGBoost variant, retaining only features that clear the 75th percentile in both architectures.  
**Second**, we compile the root-to-leaf paths of the winning XGBoost model into Conjunctive Normal Form clauses and apply the PySAT `Hitman` minimal hitting-set solver to extract irreducible Abductive Explanations—or AXPs—requiring rule support of at least 5% and confidence of at least 70%, combined with equal 50-50 weighting in combined synthesis. Think of TreeSHAP as an economist measuring marginal risk shifts across coalitions, and Formal Feature Attribution as a structural engineer proving the exact Boolean circuit that trips the breaker.  
*Lean in / Vocal emphasis*  
**Third**, to address Specific Aim 2 strictly via in-model consensus risk sensitivity contrasts ($\Delta\hat{p}$) from tree and AXP perturbations—expressed as consensus risk shifts rather than causal treatment effects—Intervention Rate measures single-feature predicted risk shift when set to its baseline median, while Interaction Effect uses second-order finite differences and Class-1 AXP co-occurrence lift above one to quantify super-additive model synergy."

##### Podium Delivery & Visual Callout

* Use the memorable analogy: TreeSHAP is the economist pricing marginal coalitions; FFA is the structural engineer proving the Boolean circuit.

##### Strategic Defense Tips & Committee Q&A Pivots

* **Appendix Jump:** Jump to **Appendix A1 (Slide 37)** for CNF clause math & 5-Set rule filter, and **Appendix A5 (Slide 41)** for exact $\text{IR}_{\Delta\hat{p}}$, $\Delta_{11}$, and $\text{IE}$ equations.

##### Transition

*Turn to Slide 19 to detail our Step 3b temporal leakage firewall.*

#### Notes Section (Slides 16–18 — Sheet 3 BACK / VERSO (Even Page))

- [ ] **1. Pacing / Delivery Check (Slides 16–18):** _____________________________________________________________________
- [ ] **2. Committee Question / Feedback (Price / Gal / Bost / Wijesinghe / DeShazo):** ___________________________
- [ ] **3. ETD Read-Copy Action Item:** ___________________________________________________________________________

---

## PAGE 7 OF 16 — SHEET 4 OF 8 (FRONT / RECTO (Odd Page)) • SLIDES 19–21 + NOTES

#### Slide 19: Module IV — Target Leakage Firewall (`Step 3b`) (#slide-7)

* **Module Context:** Module IV (Step 3b) — Temporal Event-Partitioning & `safe_feature_filter.json`  
* **Target Pacing:** `1:15 (22:30)`

##### Spoken Presenter Script

"Slide 19 details our Step 3b temporal leakage audit—the engineering firewall that eliminates reverse causality. *Brief pause*  
Following an acute emergency overdose, patients receive immediate clinical rescue protocols—such as emergency naloxone administration, rapid buprenorphine induction, and point-of-care urine toxicology screens. In 30% of published studies, models inadvertently ingest these post-index treatments, inflating AUROC by 5 to 28 percentage points simply by detecting the emergency response after the patient arrives.  
**First**, Step 3b partitions every candidate code's timestamps around index date zero and computes its post-target ratio: the proportion of claims occurring on or after the emergency visit.  
**Second**, any feature with a post-target ratio of 80% or higher is systematically purged. Across 1,029 candidate features, Step 3b purged 348 post-index confounders—or 33.8%—including 164 medications such as Suboxone at 100% post-index, 84 CPT codes such as urine toxicology `80307` at 100% post-index, and 100 secondary diagnostic codes.  
*Lean in / Vocal emphasis*  
**Third**, the remaining 682 verified pre-index antecedents—where claim date strictly precedes index date—are locked in `safe_feature_filter.json` and applied symmetrically across cases and controls, cutting the movie frame strictly before the emergency room doors open."

##### Podium Delivery & Visual Callout

* Deliver 'cutting the movie frame strictly before the emergency room doors open' with crisp vocal emphasis.

##### Strategic Defense Tips & Committee Q&A Pivots

* **Clinical Alignment (Dr. Gal):** Confirms statewide in Virginia APCD what Dr. Gal's 2023 VCU ED OUD study demonstrated clinically regarding post-event administrative code contamination (**Appendix A2, Slide 38**).

##### Transition

*Enter Module V on Slide 20 to translate validated rules into our clinical decision-support workflow.*

#### Slide 20: Module V — Actionable Intelligence & Workflow Integration (#slide-8)

* **Module Context:** Module V (Aim 3 / Ch. 5) — OODA Interface, Lean Value Stream & HIPAA/FTC Scope  
* **Target Pacing:** `1:30 (24:00)`

##### Spoken Presenter Script

"Slide 20 presents Specific Aim 3: translating our validated modeling and attribution insights into an actionable clinical decision-support prototype. *Brief pause*  
Traditional EHR drug-interaction checkers fire synchronous pop-up interrupts during rapid clinical encounters, suffering 90% to 95% clinician override rates. Furthermore, because administrative claims carry billing latency, claims-based consensus models are designed to support proactive Medication Therapy Management rather than acute bedside emergency triage.  
**First**, our interface operationalizes the Clinical OODA Loop—Observe, Orient, Decide, and Act—delivering information structured for rapid clinical review while preserving 100% human clinician authority.  
**Second**, we implement a Lean value stream framework, ranking and laying out information from high to low and left to right by clinical value—pairing patient event-density tiers with verifiable Boolean Horn rules rather than unexplained probability scores.  
*Lean in / Vocal emphasis*  
**Third**, an interactive 'What-If' scenario exploration view displays model-predicted risk shifts when candidate regimen features are toggled, supporting pharmacist medication review and evaluating explanation fidelity and clinical usability rather than claiming clinical outcomes effectiveness.  
**Fourth [Preempting Committee — Patient vs. Hospital System Use]**: We establish a strict regulatory scope demarcation: when an individual patient directly inputs their own personal health data (e.g., consumer DNA files), HIPAA does not govern that relationship and no BAA is required (regulated by FTC Act / HBNR consumer privacy). Conversely, when a hospital system or healthcare organization deploys the tool for patient care, we execute the formal two-tier BAA chain—Hospital to our entity, and our entity to AWS via AWS Artifact—enforcing KMS encryption at rest, private VPC isolation, and CloudWatch log scrubbing."

##### Podium Delivery & Visual Callout

* Explicitly state the Fourth point preempting committee HIPAA/BAA questions: patient self-use (FTC) vs. hospital ePHI two-tier BAA.

##### Strategic Defense Tips & Committee Q&A Pivots

* **Appendix Jump:** Pivot directly to **Appendix A7 (Slide 43)** for the full statutory matrix (`45 CFR § 160.103`, Two-Tier BAA #1/#2, Firecracker `/tmp` zero persistence, CloudWatch log scrubbing, CCTR legal review).

##### Transition

*Advance to Slide 21 to review the 6-chapter dissertation manuscript portfolio.*

#### Slide 21: Module V — Dissertation Portfolio & Publication Status (#slide-9)

* **Module Context:** Module V — 6-Chapter Cohesive Portfolio (3 Published in *CTS*)  
* **Target Pacing:** `1:00 (25:00)`

##### Spoken Presenter Script

"Turning to Slide 21, this dissertation is structured as a cohesive six-chapter manuscript portfolio representing our complete published and peer-reviewed scientific contributions. *Brief pause*  
**First**, on the left, our three core empirical chapters—representing Specific Aims 1, 2, and 3—are fully published in *Clinical and Translational Science*: Chapter 3 on temporal drivers of opioid emergencies, Chapter 4 on our dual-model SHAP and symbolic FFA consensus engine, and Chapter 5 on our serverless clinical decision support prototype.  
**Second**, on the right, our foundational and synthesis chapters complete the dissertation: Chapter 1 is an invited systematic review at the *Journal of Personalized Medicine* establishing The Empty Intersection, Chapter 2 is under review at the *Journal of Big Data* formalizing our three-tier governance and out-of-core claims architecture, and Chapter 6 synthesizes all five chapters into our final dissertation manuscript.  
*Lean in / Vocal emphasis*  
**Third**, all six dissertation chapters, empirical holdout validations, and companion manuscripts are complete—and remain open to the committee's final guidance prior to VCU Scholars Compass ETD deposit."

##### Podium Delivery & Visual Callout

* Highlight that all three empirical Specific Aims (Chapters 3, 4, and 5) have passed external peer review and are published in *CTS*.

##### Strategic Defense Tips & Committee Q&A Pivots

* **Codebase Reproducibility:** All 10 pipeline steps across 7 sequential Jupyter notebooks are publicly versioned at `github.com/Jerome3590/pgx-analysis` (**Appendix A4, Slide 40**).

##### Transition

*Turn to Slide 22 to summarize the 8 interlocking epidemiological and governance safeguards.*

#### Notes Section (Slides 19–21 — Sheet 4 FRONT / RECTO (Odd Page))

- [ ] **1. Pacing / Delivery Check (Slides 19–21):** _____________________________________________________________________
- [ ] **2. Committee Question / Feedback (Price / Gal / Bost / Wijesinghe / DeShazo):** ___________________________
- [ ] **3. ETD Read-Copy Action Item:** ___________________________________________________________________________

---

## PAGE 8 OF 16 — SHEET 4 OF 8 (BACK / VERSO (Even Page)) • SLIDES 22–24 + NOTES

#### Slide 22: Module V — Study Boundaries: 8 Interlocking Safeguards (#slide-10)

* **Module Context:** Module V — 4 Epidemiological Constraints + 4 Clinical Governance Fences  
* **Target Pacing:** `1:15 (26:15)`

##### Spoken Presenter Script

"Turning to Slide 22, before synthesizing our Clinical OODA Loop and entering our empirical results, our dissertation is bounded by eight interlocking safeguards—four epidemiological cohort constraints on the left, and four clinical governance principles on the right:  
**On the left**, our epidemiological design enforces:  
**First**, the Absolute Opioid Cleanse across all ten diagnostic positions in our geriatric cohort;  
**Second**, High-Utilization Capping excluding patients with seven or more annual emergency visits;  
**Third**, the 21-Day Proximal Pharmacy Window—where external FDA FAERS onset kinetics directly validate our cohort derivation logic; and  
**Fourth**, the Zero-Day Discharge Guard systematically purging day-zero fills and post-index confounders to eliminate target leakage.  
*Brief pause*  
**On the right**, our governance framework establishes:  
**Fifth**, Aim 3 is decision-support triage evaluating explanation fidelity, not a patient outcomes trial—with a strict regulatory boundary separating direct patient self-use under FTC privacy from enterprise hospital deployment under our two-tier HIPAA BAA chain;  
**Sixth**, Consensus Attribution quantifies predictive consensus risk for clinical review rather than causal treatment effects;  
**Seventh**, the untouched 2019 temporal holdout evaluates real-world model transportability with zero recalibration; and  
**Eighth**, our pharmacogenomic features evaluate claims-derived CPIC medication exposure rather than inferred germline genotypes.  
*Lean in / Vocal emphasis*  
By grounding our cohort logic in convergent FAERS surveillance kinetics and enforcing strict clinical governance, this dissertation advances both health data science methodology and public health pharmacovigilance."

##### Podium Delivery & Visual Callout

* Sweep cleanly down the left 4 epidemiological fences, pause briefly, then sweep down the right 4 governance fences.

##### Strategic Defense Tips & Committee Q&A Pivots

* **Causality Fence (#6):** Observational claims + `SHAP ∩ FFA` identify modifiable risk associations for pharmacist review; causal treatment effects require target-trial emulation or our proposed CCTR cluster RCT.

##### Transition

*Advance to Slide 23 to synthesize Chapters 1–5 inside the Clinical OODA Loop before entering Module VI Results.*

#### Slide 23: Module V — The Clinical OODA Loop Synthesis (#slide-ooda-loop)

* **Module Context:** Module V (Ch. 6 Bridge) — Observe, Orient, Decide & Act Closed-Loop Architecture  
* **Target Pacing:** `1:15 (27:30)`

##### Spoken Presenter Script

"Slide 23 synthesizes the entire dissertation architecture into the clinical OODA loop—Observe, Orient, Decide, and Act—emphasizing the iterative progression and feedback between phases. *Brief pause*  
**First**, under **Observe**, we bridge the triple gap in healthcare AI by scaling partition-first DuckDB ingestion across 1.8 terabytes of Virginia APCD claims—linking external CPIC pharmacogenomic reference data to quantify PGx burden and incorporating FDA FAERS safety kinetics to validate adverse event windows.  
**Second**, under **Orient**, we build our **Initial Model**—performing Step 3a exploratory feature screening across all 16 cohort and age strata, applying our dual-gated TreeSHAP and symbolic FFA consensus filter to isolate robust feature attributions, and enforcing the B1 balancing constraint to evaluate baseline transportability.  
**Third**, under **Decide**, we execute the critical iterative refinement step: confronting the initial model's vulnerability to target leakage, our Step 3b pre-index firewall purges reverse-causality rescue codes to construct the **Refined Model**. Here, we evaluate three distinct gradient-boosted tree geometries—CatBoost symmetric oblivious trees for categorical shift immunity, XGBoost asymmetric depth-wise trees for deep multi-drug interaction discovery, and XGBoost-RF bagging for variance stabilization—translating vetted patterns into clinical risk allocation and deprescribing priorities.  
**Fourth**, under **Act**, we deploy a stateless, HIPAA-compliant serverless decision-support dashboard delivering sub-100 millisecond inference to prioritize pharmacist medication therapy management reviews.  
*Lean in / Vocal emphasis*  
Finally, clinical interventions, pharmacist reviews, and longitudinal follow-up claims feed directly back into our continuous learning loop, retraining initial screening baselines and refining decision rules over time."

##### Podium Delivery & Visual Callout

* Show how Observe -> Orient -> Decide -> Act unifies the entire pipeline and sets the stage for Module VI empirical results.

##### Strategic Defense Tips & Committee Q&A Pivots

* **Iterative Feedback:** Clinical pharmacist overrides, longitudinal follow-up claims, and quarterly SHAP rank-drift monitoring feed back into continuous baseline recalibration.

##### Transition

*Now entering Module VI on Slide 24, let us examine our empirical results across Chapters 2 through 5.*

#### Slide 24: Module VI Results (Ch. 2) — Out-of-Core DuckDB & Cloud Economics (#slide-res-ch2)

* **Module Context:** Module VI Empirical Results — Chapter 2 Lakehouse Throughput & Cost Collapse  
* **Target Pacing:** `1:15 (28:45)`

##### Spoken Presenter Script

"Entering Module VI, we present our empirical results across Chapters 2 through 5, beginning on Slide 24 with Chapter 2 Data Engineering. *Brief pause*  
Processing 1.8 terabytes and 380 million Virginia APCD claims on traditional distributed EMR and Apache Spark clusters triggered repeated out-of-memory failures (`exit code 137`) at a run rate of 23,000 dollars per month due to severe patient-level skew.  
**First**, our partition-first out-of-core DuckDB architecture on a single fat AWS `x2iedn.8xlarge` Spot EC2 instance achieved 18.1 million records per minute—a 15.1-fold speedup over monolithic execution—with zero partition failures across 189 consecutive passes.  
**Second**, idempotent S3 SHA-256 checkpoint manifests enabled sub-60-second Spot interruption recovery while collapsing compute cost by 99% to between 6 and 10 dollars per full statewide pass.  
*Lean in / Vocal emphasis*  
**Takeaway:** For skewed statewide healthcare claims, vectorized DuckDB on a single fat EC2 instance decisively beats distributed EMR clusters and Apache Spark on throughput, reliability, and cost."

##### Podium Delivery & Visual Callout

* Punch the contrast: $23,000/month crashing Spark clusters vs. $6–$10/pass with zero failures on single-node vectorized DuckDB.

##### Strategic Defense Tips & Committee Q&A Pivots

* **Why DuckDB Beat Spark:** Partitioning by `Age Band × Year` (`63` slices) eliminated network shuffle skew and JVM `toPandas` serialization, keeping `CatBoost`, `TreeSHAP`, and `PySAT` co-located in 1 TB RAM + NVMe.

##### Transition

*Advance to Slide 25 for the empirical audit of Step 3b leakage pruning and the 5-Set CNF solver.*

#### Notes Section (Slides 22–24 — Sheet 4 BACK / VERSO (Even Page))

- [ ] **1. Pacing / Delivery Check (Slides 22–24):** _____________________________________________________________________
- [ ] **2. Committee Question / Feedback (Price / Gal / Bost / Wijesinghe / DeShazo):** ___________________________
- [ ] **3. ETD Read-Copy Action Item:** ___________________________________________________________________________

---

## PAGE 9 OF 16 — SHEET 5 OF 8 (FRONT / RECTO (Odd Page)) • SLIDES 25–27 + NOTES

#### Slide 25: Module VI Results (Ch. 2) — Step 3b Purge & 5-Set CNF Filter (#slide-res-ch2-leakage)

* **Module Context:** Module VI Empirical Results — Chapter 2 Leakage Elimination & 100% AXP Coverage  
* **Target Pacing:** `1:15 (30:00)`

##### Spoken Presenter Script

"Slide 25 reports the empirical validation of our two core engineering gates in Chapter 2. *Brief pause*  
**First**, across 1,029 Step 3a candidate features, our Step 3b BupaR event-partitioning audit identified and purged 348 post-index confounders—33.8% of the candidate pool—including 164 rescue medications like Suboxone and Narcan occurring 100% on or after the emergency visit, 84 toxicology CPTs like `80307`, and 100 follow-up diagnostic codes. Locking the 682 verified pre-index antecedents in `safe_feature_filter.json` replaced artificially leaked 0.99 AUROC scores with honest out-of-time generalization.  
**Second**, our 5-Set candidate rule filter bounded the PySAT `Hitman` solver to achieve 100% patient explanation coverage across our matched opioid (`N = 207,350`) and polypharmacy (`N = 647,002`) cohorts."

##### Podium Delivery & Visual Callout

* Highlight both gates: Gate 1 guarantees temporal validity (`682` antecedents); Gate 2 guarantees tractable 100% symbolic rule coverage.

##### Strategic Defense Tips & Committee Q&A Pivots

* **5-Set Non-Circularity Defense:** Sets 1 & 2 sample boosting/random trees independently of SHAP; Set 4 (`SHAP = 0` fallback) implements *Huang & Marques-Silva (2024)* so abductive gates are never dropped.

##### Transition

*Turn to Slide 26 for our published Chapter 3 Opioid Cohort results on the untouched 2019 holdout.*

#### Slide 26: Module VI Results (Ch. 3) — Opioid 2019 Holdout & Mirror Ablation (#slide-res-ch3-holdout)

* **Module Context:** Module VI Empirical Results — Specific Aim 1 (`Clin Transl Sci` `10.1111/cts.70690`)  
* **Target Pacing:** `1:30 (31:30)`

##### Spoken Presenter Script

"Turning to Slide 26, we present our published Chapter 3 results in *Clinical and Translational Science* (`10.1111/cts.70690`) across 207,350 matched subjects in our chronic opioid cohort (`26,710` cases and `180,640` 1:5 controls). *Brief pause*  
**First**, evaluated strictly once on the untouched 2019 calendar holdout with zero post-hoc recalibration, our Step 6 winning architectures achieved out-of-time AUROCs from 0.800 in our primary `25–44` cohort up to 0.890 in `55–64`, with Precision-Recall lifts of 2.3-fold to 4.4-fold above baseline prevalence and Brier scores between 0.011 and 0.015.  
**Second**, our two-gate mirror ablation proved the operational value of intersecting TreeSHAP with symbolic FFA: retaining 20 out of 20 top clinical drivers while compressing feature review load by 61%—from 112 down to 44 features—and filtering boundary buprenorphine noise."

##### Podium Delivery & Visual Callout

* Direct eyes to primary `25–44` (`0.800 AUROC`, `3.3x lift`) and the `61%` clinician review-load compression in Reviewer #2's gate ablation.

##### Strategic Defense Tips & Committee Q&A Pivots

* **Why Mirror Ablation Matters:** Pure SHAP (`112` features) admits routine E/M billing noise, while pure FFA admits rare boundary buprenorphine rules; `SHAP ∩ FFA` (`44` rules) keeps 100% of core clinical signal.

##### Transition

*Advance to Slide 27 to show HOW and WHEN clinical pharmacists intervene in the opioid cohort.*

#### Slide 27: Module VI Results (Ch. 3) — Consensus Drivers & 6.8-Month Window (#slide-res-ch3-drivers)

* **Module Context:** Module VI Empirical Results — Specific Aim 1 Actionable 'HOW' & 'WHEN' Horizon  
* **Target Pacing:** `1:15 (32:45)`

##### Spoken Presenter Script

"Slide 27 answers both HOW and WHEN clinical pharmacists can intervene in the opioid cohort. *Brief pause*  
**First, HOW to intervene:** across adult strata, total drug burden (`pgx_num_drugs`, `100%` AXP support), `gabapentin` (`87%` support), benzodiazepines (`clonazepam`, `alprazolam`), and CPIC-actionable drug burden (`pgx_num_cpic_drugs`, present in 74% to 86% of cases) dominate consensus Horn rules—pinpointing synergistic CNS respiratory depression.  
**Second, WHEN to intervene:** exploratory Dynamic Time Warping across 59,813 patient trajectories uncovered a 6.8-month—or 27-week—escalation inflection window before emergency presentation, while `bupaR` process mining showed that initiating physical therapy prior to gabapentin escalation was associated with 58% lower progression to an opioid ED visit."

##### Podium Delivery & Visual Callout

* Contrast HOW (`gabapentin` + CNS + CPIC burden) with WHEN (`6.8-month` DTW escalation lead time & `-58%` physical therapy pathway).

##### Strategic Defense Tips & Committee Q&A Pivots

* **Pharmacologic Mechanism:** Gabapentinoids potentiate opioid-induced respiratory depression by bypassing mu-opioid tolerance in brainstem respiratory centers—an actionable outpatient deprescribing target 27 weeks pre-ED.

##### Transition

*Move to Slide 28 for our published Chapter 4 Geriatric Polypharmacy results.*

#### Notes Section (Slides 25–27 — Sheet 5 FRONT / RECTO (Odd Page))

- [ ] **1. Pacing / Delivery Check (Slides 25–27):** _____________________________________________________________________
- [ ] **2. Committee Question / Feedback (Price / Gal / Bost / Wijesinghe / DeShazo):** ___________________________
- [ ] **3. ETD Read-Copy Action Item:** ___________________________________________________________________________

---

## PAGE 10 OF 16 — SHEET 5 OF 8 (BACK / VERSO (Even Page)) • SLIDES 28–30 + NOTES

#### Slide 28: Module VI Results (Ch. 4) — Geriatric Holdout & Table S5 Stress Test (#slide-res-ch4-holdout)

* **Module Context:** Module VI Empirical Results — Specific Aim 2 (`Clin Transl Sci` `10.1111/cts.70718`)  
* **Target Pacing:** `1:30 (34:15)`

##### Spoken Presenter Script

"Slide 28 presents our published Chapter 4 results (`10.1111/cts.70718`) evaluating acute 21-day geriatric polypharmacy ADEs across `N = 647,002` total patients, focusing on older adults aged 65 to 114. *Brief pause*  
**First**, on the untouched 2019 holdout, models achieved `0.877` AUROC (`3.9×` lift, Brier `0.008`, `89` consensus rules) in ages `65–74` and `0.858` AUROC (`4.2×` lift, `86` rules) in `75–84`, before attenuating to `0.686` (`1.6×` lift) in `85–114` due to unmeasured physiological frailty ceilings in claims data.  
**Second**, our `Table S5` utilization-free stress test delivered a striking methodological proof: removing all 17 utilization features collapsed univariate top-25 drug SHAP rankings to a Jaccard similarity of `0.00` in `65–74`, yet symbolic FFA Horn clauses preserved 60% (`3/5`) of core synergistic drug pairs—proving that Boolean minimal hitting sets anchor clinical stability when marginal Shapley credits shift."

##### Podium Delivery & Visual Callout

* Lean in on Table S5 (Anchor #3): when 17 utilization features are removed, univariate SHAP collapses (`Jaccard = 0.00`), yet symbolic FFA preserves `60%` of drug pairs!

##### Strategic Defense Tips & Committee Q&A Pivots

* **Why 0.686 in 85–114?** Honest biological boundary: administrative claims lack gait speed, grip strength, glomerular filtration labs, and nursing-home ADL frailty driving oldest-old ED transfers.

##### Transition

*Advance to Slide 29 for Chapter 4's three clinical discoveries: multi-drug triplets, IR priorities, and the Z-code shield.*

#### Slide 29: Module VI Results (Ch. 4) — Multi-Drug Synergies & Z-Code Shield (#slide-res-ch4-synergies)

* **Module Context:** Module VI Empirical Results — Specific Aim 2 Triplets, `IR` & Preventive Stewardship  
* **Target Pacing:** `1:30 (35:45)`

##### Spoken Presenter Script

"Slide 29 highlights three translational discoveries from Chapter 4. *Brief pause*  
**First**, beyond pairwise checkers, second-order interaction screening validated 115 synergistic pairs—led by fluoroquinolone-CNS combinations like `levofloxacin + lorazepam` (`IE = 11.9`)—and 312 Bonferroni-significant triplets out of 5,021 candidates, independently recovering the STOPP D4 'Triple Whammy' (`furosemide + HCTZ + lisinopril`) in 12.3% of oldest-old cases.  
**Second**, counterfactual in-model Intervention Rate ranked `simvastatin`, `furosemide`, and `alprazolam/lorazepam` as the top deprescribing review priorities—capturing high-prevalence metabolic and volume-depletion risks overlooked by static Beers anticholinergic lists.  
**Third**, we discovered a U-shaped Preventive Z-Code Shield: older adults with moderate routine wellness and long-term medication reconciliation encounters (`Q2`, 1%–12% Z-codes) experienced a 3.0% vs. 10.7% ED rate—an odds ratio of `0.25`, or 75% lower risk."

##### Podium Delivery & Visual Callout

* Deliver the three numbered breakthroughs rhythmically; emphasize that Q2 structured wellness/medication reconciliation cuts 21-day ADEs by 75% (`OR = 0.25`).

##### Strategic Defense Tips & Committee Q&A Pivots

* **Triple Whammy Physiology:** Loop/thiazide volume depletion + ACEi efferent arteriolar dilation collapses glomerular filtration pressure, causing acute kidney injury (`N17.9`) and orthostatic syncope within 21 days.

##### Transition

*Turn to Slide 30 for our published Chapter 5 Serverless PGx Clinical Decision Support results.*

#### Slide 30: Module VI Results (Ch. 5) — Serverless Zero-PHI Architecture & Latency (#slide-res-ch5-arch)

* **Module Context:** Module VI Empirical Results — Specific Aim 3 (`Clin Transl Sci` `10.1111/cts.70697`)  
* **Target Pacing:** `1:00 (36:45)`

##### Spoken Presenter Script

"Turning to Slide 30, we present our published Chapter 5 results (`10.1111/cts.70697`) operationalizing Specific Aim 3 into a live serverless clinical decision-support prototype at `pgx.jerome-dixon.io`. *Brief pause*  
**First**, we packaged all 56 stratified ensembles—168 trained CatBoost, XGBoost, and XGBoost-RF models—alongside 573 official CPIC gene-drug pairs inside a single AWS Lambda container under 10 gigabytes, operating on a stateless `compute–respond–discard` lifecycle for under 5 dollars per month.  
**Second**, empirical load testing confirmed `6 millisecond` warm risk inference (`SD 1 ms`) and `60 millisecond` CPIC pharmacogenomic card generation—well beneath the 100-millisecond threshold required for fluid interactive clinical workflows."

##### Podium Delivery & Visual Callout

* Emphasize real-world engineering feasibility: 168 models + 573 CPIC rules running in 6 milliseconds for under $5/month.

##### Strategic Defense Tips & Committee Q&A Pivots

* **Enterprise HIPAA Transition:** Native public demo uses stateless zero-storage RAM; hospital EHR deployment activates our Two-Tier AWS Artifact BAA + VPC/KMS/CloudTrail controls (**Appendix A7, Slide 43**).

##### Transition

*Advance to Slide 31 to validate bedside accuracy under severe point-of-care input sparsity.*

#### Notes Section (Slides 28–30 — Sheet 5 BACK / VERSO (Even Page))

- [ ] **1. Pacing / Delivery Check (Slides 28–30):** _____________________________________________________________________
- [ ] **2. Committee Question / Feedback (Price / Gal / Bost / Wijesinghe / DeShazo):** ___________________________
- [ ] **3. ETD Read-Copy Action Item:** ___________________________________________________________________________

---

## PAGE 11 OF 16 — SHEET 6 OF 8 (FRONT / RECTO (Odd Page)) • SLIDES 31–33 + NOTES

#### Slide 31: Module VI Results (Ch. 5) — Sparsity Shield & What-If Engine (#slide-res-ch5-sparsity)

* **Module Context:** Module VI Empirical Results — Specific Aim 3 'Imputation of Normality' Validation  
* **Target Pacing:** `1:00 (37:45)`

##### Spoken Presenter Script

"Slide 31 validates how our population claims models remain accurate when a bedside pharmacist enters only 3 to 5 active medications rather than 682 historical billing codes. *Brief pause*  
**First**, under our 'Imputation of Normality' protocol, unentered historical covariates default to age-stratified training medians—maintaining a mean absolute probability drift under `0.06` across up to 70% feature masking and 87% risk-tier concordance at 50% sparsity.  
**Second**, Tab 3's interactive What-If Scenario Engine computes real-time consensus risk deltas (`Δp̂`) as candidate medications are toggled, surfacing `IR > 0.10` synergy alerts alongside CPIC pharmacogene networks."

##### Podium Delivery & Visual Callout

* Anchor #4: explain clearly why population claims models fail at the bedside without Imputation of Normality.

##### Strategic Defense Tips & Committee Q&A Pivots

* **Scientific Hypothesis of Ch. 5:** Proves Core Proposition 3—that cohort-median background anchoring resolves the point-of-care claims sparsity gap without miscalibrating risk tiers.

##### Transition

*Turn to Slide 32 for the single-slide Executive Summary across Chapters 2 through 5.*

#### Slide 32: Module VI Results — Executive Empirical Summary Matrix (#slide-res-summary)

* **Module Context:** Module VI Synthesis — Complete Empirical Closure Across Chapters 2–5 & Aims 1–3  
* **Target Pacing:** `1:30 (39:15)`

##### Spoken Presenter Script

"Slide 32 synthesizes our complete empirical results across Chapters 2 through 5 and Specific Aims 1 through 3 on a single executive matrix. *Brief pause*  
**First**, in Chapter 2 Data Engineering, vectorized DuckDB on a single fat EC2 instance beat distributed EMR clusters and Apache Spark—delivering a 15.1-fold speedup, zero shuffle-skew OOMs, and a 99% cloud cost reduction across 380 million claims—while purging 348 post-index leakage codes via Step 3b and bounding PySAT rule extraction via our 5-Set CNF filter.  
**Second**, in Chapter 3 (Specific Aim 1), leakage-free GBDTs achieved 0.800 to 0.890 AUROC and up to 4.4-fold Precision-Recall lift on the untouched 2019 opioid holdout, compressing pharmacist alert load by 61% while identifying modifiable CNS and CPIC burden 6.8 months before emergency presentation.  
**Third**, in Chapter 4 (Specific Aim 2), geriatric polypharmacy models achieved 0.877 and 0.858 AUROC in ages 65–84, proved 60% symbolic FFA pair invariance when univariate SHAP collapsed to zero Jaccard similarity, uncovered 312 higher-order drug triplets including the Triple Whammy, and quantified a 75% risk reduction under structured outpatient Z-code stewardship.  
*Lean in / Vocal emphasis*  
**Fourth**, in Chapter 5 (Specific Aim 3), our stateless AWS Lambda microservice operationalized all 168 trained base learners and 573 CPIC guidelines in 6 milliseconds of warm compute for under 5 dollars a month with 87% bedside sparsity stability—directly closing the Empty Intersection across all three Specific Aims."

##### Podium Delivery & Visual Callout

* Deliver the four chapter summaries with commanding cadence—this is the empirical crescendo of the defense.

##### Strategic Defense Tips & Committee Q&A Pivots

* **Closing the Empty Intersection:** Directly demonstrates that all three gaps from Slide 6 (`0/73` PROBAST models) are empirically resolved, peer-reviewed, and deployed.

##### Transition

*Enter Module VII on Slide 33 to review Next Steps, 9NOV committee feedback integration, and our translational horizon.*

#### Slide 33: Module VII — Next Steps (9NOV Feedback, CCTR Pilot, Federation & SaMD) (#slide-next-steps)

* **Module Context:** Module VII (Ch. 6) — Post-9NOV Synthesis, Prospective Trial & Regulatory Roadmap  
* **Target Pacing:** `1:30 (40:45)`

##### Spoken Presenter Script

"Finally, Slide 33 outlines Module VII: Next Steps—explicitly structured to incorporate pending committee feedback from our November 9 presentation alongside our translational research horizon. *Brief pause*  
**First**, immediately following our November 9 oral examination, all committee refinements and methodological recommendations are integrated directly into the final Chapters 1 through 6 manuscript, alongside our institutional VCU Health Compliance and Counsel review of our ephemeral cloud architecture and two-tier HIPAA BAA chain.  
**Second**, through our VCU Wright Center CCTR 25,000-dollar Pilot Grant, Phase 1 conducts formative think-aloud usability testing with ambulatory MTM pharmacists (`SUS >= 75`), laying the groundwork for a Phase 2 `2×2` factorial cluster randomized trial measuring real-world deprescribing action and 90-day ED outcomes.  
**Third**, we extend external validity via 4-state Appalachian APCD federation (`VA, MD, NC, WV`) and sequenced biobank linkage (`All of Us`, `BioVU`) to unite claims CPIC exposure with true germline star-allele diplotypes.  
*Lean in / Vocal emphasis*  
**Fourth**, long-term clinical translation aligns with HL7 FHIR R4 CDS Hooks and FDA Software as a Medical Device Predetermined Change Control Plans to govern post-2020 secular prescribing and fentanyl drift."

##### Podium Delivery & Visual Callout

* Acknowledge the committee's November 9 guidance directly first, then project the funded CCTR pilot, multi-state federation, and FDA SaMD PCCP.

##### Strategic Defense Tips & Committee Q&A Pivots

* **Secular Fentanyl / Post-2020 Drift Defense:** Our FDA SaMD Predetermined Change Control Plan (`PCCP`) triggers automated retraining whenever quarterly top-10 `|SHAP|` Spearman correlation drifts >15%.

##### Transition

*Advance to Slide 34 to open the floor for Questions & Discussion, with Slide 35 Acknowledgements ready.*

#### Notes Section (Slides 31–33 — Sheet 6 FRONT / RECTO (Odd Page))

- [ ] **1. Pacing / Delivery Check (Slides 31–33):** _____________________________________________________________________
- [ ] **2. Committee Question / Feedback (Price / Gal / Bost / Wijesinghe / DeShazo):** ___________________________
- [ ] **3. ETD Read-Copy Action Item:** ___________________________________________________________________________

---

## PAGE 12 OF 16 — SHEET 6 OF 8 (BACK / VERSO (Even Page)) • SLIDES 34–35 + DEFENSE GUIDE + NOTES

#### Slide 34: Module VII — Questions & Discussion (#slide-qa)

* **Module Context:** Module VII — Supervisory Committee Oral Defense Examination Floor  
* **Target Pacing:** `0:15 (41:00)`

##### Spoken Presenter Script

"I welcome the committee's questions and discussion."

##### Podium Delivery & Visual Callout

* Pause, smile, make eye contact with Chair Dr. Elvin Price, and invite the committee's questions.

##### Strategic Defense Tips & Committee Q&A Pivots

* **Podium Discipline:** Listen to the full question, pause 2 seconds, state the direct answer in the first sentence, and click the matching `Appendix A1–A8` jump button only if visual math/tables help.

##### Transition

*After or before opening Q&A (per Chair's preference), present Slide 35 Thank You & Acknowledgements.*

#### Slide 35: Module VII — Thank You & Acknowledgements (#slide-acknowledgements)

* **Module Context:** Module VII — Mentors, Faculty, Data Partners & Family Gratitude  
* **Target Pacing:** `1:00 (42:00)`

##### Spoken Presenter Script

"Thank you to my Dissertation Chair Dr. Elvin Price, my supervisory committee members Professor Franklin Bost, Dr. Tamas Gal, Dr. Dayanjan Wijesinghe, and Dr. Jonathan DeShazo, special thanks to Dr. Michael Neale—for first introducing me to the Integrative Life Sciences Ph.D. program—along with Dr. Hermine Maes and Dr. Tim York in Human Genetics and VIPBG, Dr. Amy Olex at the VCU Wright Center, Dr. Stephen Fong and the ILS leadership, our colleagues in the VCU School of Pharmacy and Price Lab, our data partners at Virginia Health Information and the Virginia Center for Health Innovation, and above all, my family, friends, work colleagues, and mentors for making this work possible."

##### Podium Delivery & Visual Callout

* Deliver warmly and unhurriedly; specifically recognize Dr. Michael Neale for introducing ILS, Drs. Maes & York, Dr. Olex, Dr. Fong, VHI, and family.

##### Strategic Defense Tips & Committee Q&A Pivots

* **Total Core Presentation Elapsed Time:** **~42 minutes** across Slides 1–35—leaving a comfortable 3–8 minute buffer inside the 45–50 minute defense window!

##### Transition

*Transition to Committee Q&A using Slides 36–44 (Appendices A1–A8) as on-demand technical backup.*

#### Defense Guide (Slot 36): 45-Minute Module Pacing Checkpoints & Committee Routing

* **Pacing Checkpoints:** Orientation (`S1–3`: `02:30`) • Module I (`S4–11`: `12:00`) • Module II (`S12–13`: `14:30`) • Module III (`S14–16`: `18:30`) • Module IV (`S17–19`: `22:30`) • Module V (`S20–23`: `27:30`) • Module VI Empirical Results (`S24–32`: `39:15`) • Module VII Next Steps & Acknowledgements (`S33–35`: `42:00`).
* **4 Core Defense Anchors:** (1) `0/73` PROBAST Empty Intersection • (2) Step 3b Purge (`348` post-index codes removed -> honest `0.800–0.890` holdout AUROC) • (3) Table S5 `60%` (`3/5`) symbolic FFA pair retention under `SHAP Jaccard = 0.00` collapse • (4) `6 ms` Zero-PHI AWS Lambda + Imputation of Normality (`|Δp̂| < 0.06` at `70%` sparsity).

#### Notes Section (Slides 34–35 + Defense Guide — Sheet 6 BACK / VERSO (Even Page))

- [ ] **1. Pacing / Delivery Check (Slides 34–35 + Defense Guide):** _____________________________________________________________________
- [ ] **2. Committee Question / Feedback (Price / Gal / Bost / Wijesinghe / DeShazo):** ___________________________
- [ ] **3. ETD Read-Copy Action Item:** ___________________________________________________________________________

---

## PAGE 13 OF 16 — SHEET 7 OF 8 (FRONT / RECTO (Odd Page)) • APPENDIX SLIDES 36–38 + NOTES

#### SLIDE 36 (APPX): Appendix Section Divider — Technical Defenses & Supplemental Evidence (#sec-appendix)

* **Module Context:** Technical Backup Bank — Interactive Jump Index to Appendices A1–A8  
* **Target Pacing:** `On-Demand Q&A`

##### Spoken Presenter Script

"Slide 36 serves as our interactive Technical Appendix index across Appendices A1 through A8—providing on-demand mathematical proofs for Conjunctive Normal Form and Abductive Explanations (A1), Step 3b event partitioning (A2), Step 5 CPIC drug mapping (A3), our seven reproducible Jupyter notebooks (A4), formal Aim 1–3 estimators (A5), master nomenclature (A6), two-tier HIPAA and FTC cloud governance (A7), and cohort-specific PharmGKB and CPIC Level A molecular pathways (A8)."

##### Podium Delivery & Visual Callout

* Hold in reserve during Q&A; use keyboard hash links or return buttons for instant non-linear navigation.

##### Strategic Defense Tips & Committee Q&A Pivots

* **Faculty Mapping:** `A1, A4, A5` -> Prof. Bost & Dr. Gal; `A2` -> Dr. Gal; `A3, A8` -> Dr. Price & Dr. Wijesinghe; `A6` -> General Reference; `A7` -> Dr. Gal, Dr. Price & Dr. DeShazo.

##### Return / Jump Navigation

*Select the specific Appendix slide (A1–A8) matching the committee member's question.*

#### SLIDE 37 (A1): Appendix A1 — Formal Feature Attribution (FFA): CNF & AXP Bounds (#sec-appendix-ffa)

* **Module Context:** Appendix A1 — Steps 6–8 & 10 Mathematical Proof & Sensitivity Framework  
* **Target Pacing:** `On-Demand Q&A`

##### Spoken Presenter Script

Slide A1 provides the mathematical and empirical defense for our Consensus-Risk architecture across Steps 6 through 8 and Step 10. Because CatBoost uses internal target-statistic hashing on categorical variables, Step 7 first computes exact TreeSHAP on both the best CatBoost model and the best XGBoost variant, enforcing their strict feature intersection as a cross-architecture gate. Step 8 then explodes the decision trees of the best XGBoost variant into CNF clauses and prioritizes the top SHAP-supported matched rules into the PySAT Hitman minimal hitting-set solver to extract exact Abductive Explanations in polynomial O(T times L) time. Importantly, our Consensus-Risk thresholds—75th percentile SHAP, support of 0.05, and confidence of 0.70—operate as a validated precision-volume dial for clinical review capacity, proven empirically in Module VI: first, our Two-Gate Mirror Ablation on Slide 26 retained 20 out of 20 core clinical risk drivers while compressing alert volume by 61 percent and rejecting 252 routine billing codes; and second, our Table S5 utilization-free stress test on Slide 28 demonstrated 60 percent symbolic FFA drug-pair retention even when univariate SHAP Jaccard similarity collapsed to 0.00.

##### Podium Delivery & Visual Callout

* Emphasize why `Q75 / 0.05 / 0.70` is an operational precision–volume dial for pharmacist review capacity rather than an arbitrary cutoff.

##### Strategic Defense Tips & Committee Q&A Pivots

* **Why Not Explode CatBoost Directly into CNF?** CatBoost uses online target-statistic hashing on categorical combinations; intersecting CatBoost TreeSHAP with XGBoost CNF AXPs gets both shift immunity and exact Boolean rules.

##### Return / Jump Navigation

*Click 'Return to Consensus Attribution (Slide 18)' or jump to CH3/CH4 sensitivity results.*

#### SLIDE 38 (A2): Appendix A2 — Step 3b Target Leakage: Event-Partitioning Protocol (#sec-appendix-leakage)

* **Module Context:** Appendix A2 — 3 Contamination Vectors & Symmetric Pre-Index Whitelist  
* **Target Pacing:** `On-Demand Q&A`

##### Spoken Presenter Script

Slide A2 details our Step 3b temporal event-partitioning protocol. We classify post-index contamination into three clinical vectors: post-overdose OUD rescue and MAT induction, confirmatory ED and follow-up toxicology screens, and downstream administrative follow-up codes. Any candidate feature whose claims events occur 80% or more on or after the index timestamp t0 is automatically purged, locking an immutable pre-index antecedent whitelist in safe_feature_filter.json. As demonstrated empirically in Module VI on Slide 25, this Step 3b audit purged 348 post-index confounders out of 1,029 candidate features and locked 682 verified pre-index antecedents—replacing artificially leaked 0.99 AUROC scores with honest 0.800 to 0.890 out-of-time 2019 holdout generalization.

##### Podium Delivery & Visual Callout

* Walk through the 3 contamination rows, then link directly to Slide 25's empirical purge (`0.99` leaked -> `0.800–0.890` honest holdout).

##### Strategic Defense Tips & Committee Q&A Pivots

* **Symmetric Application:** `safe_feature_filter.json` is applied identically to cases and 1:5 controls so feature filtering never introduces differential selection bias.

##### Return / Jump Navigation

*Click 'Return to Target Leakage (Slide 19)' or 'Return to CH2 Empirical Audit (Slide 25)'.*

---

## PAGE 14 OF 16 — SHEET 7 OF 8 (BACK / VERSO (Even Page)) • APPENDIX SLIDES 39–41 + NOTES

#### SLIDE 39 (A3): Appendix A3 — Step 5 CPIC Drug Mapping & Patient Count Features (#sec-appendix-pgx)

* **Module Context:** Appendix A3 — `5_pgx_analysis` Fuzzy Matching & Claims-vs.-Genotype Boundary  
* **Target Pacing:** `On-Demand Q&A`

##### Spoken Presenter Script

Slide A3 details our exact Step 5 pharmacogenomic feature engineering pipeline in 5_pgx_analysis. First, build_global_drug_cpic_mapping.py scans our APCD drug vocabulary against the official CPIC gene-drug directory using a 95 percent fuzzy-matching threshold to determine whether or not each dispensed medication carries an official CPIC guideline, persisting the global lookup table drug_cpic_mapping_global.csv. Second, create_pgx_features_patient_level.py executes a single DuckDB aggregation over pre-index claims to compute two integer features per patient—pgx_num_drugs (total distinct drugs) and pgx_num_cpic_drugs (distinct CPIC-guideline drugs)—as well as cohort-level counts of how many patients carry CPIC drugs. Crucially, no allele or race-based genotype imputation is performed in the modeling pipeline; our models strictly evaluate observable CPIC medication exposure burden directly from claims data, while interactive patient-level genotype decision-support cards execute in 60 milliseconds at the point of care in Tab 5 of our serverless dashboard (Slide 30).

##### Podium Delivery & Visual Callout

* Highlight the scientific ethics of refusing race-based allele imputation in claims data.

##### Strategic Defense Tips & Committee Q&A Pivots

* **Point-of-Care Layering:** Population models score observable CPIC drug burden; when a patient has real genetic test results (`VCF` / star alleles), Tab 5 overlays CPIC guideline dosing cards in `60 ms`.

##### Return / Jump Navigation

*Click 'Return to Claims Schema (Slide 14)' or jump to Appendix A8 (Slide 44) for gene networks.*

#### SLIDE 40 (A4): Appendix A4 — Computational Pipeline & 7-Notebook Architecture (#sec-appendix-notebooks)

* **Module Context:** Appendix A4 — Full Idempotent S3/NVMe Reproducibility (`pgx-analysis`)  
* **Target Pacing:** `On-Demand Q&A`

##### Spoken Presenter Script

Slide A4 details the computational reproducibility of the dissertation. All ten analytical steps across our three specific aims are operationalized into seven sequential Jupyter notebooks in our public GitHub repository. The workflow uses S3-backed state checkpoints for idempotency and runs seamlessly on both local workstations and AWS EC2 spot instances. Every dataset transformation, hyperparameter search, and model artifact is fully versioned and traceable back to the raw APCD extracts.

##### Podium Delivery & Visual Callout

* Show complete end-to-end lineage from raw Virginia APCD Parquet to Dockerized AWS Lambda container.

##### Strategic Defense Tips & Committee Q&A Pivots

* **Dual Environment Parity:** Runs identically on Windows 11 local dev (DuckDB 1.4+, Python 3.11, R 4.3 `bupaR`) and Amazon Linux 2023 Spot EC2.

##### Return / Jump Navigation

*Click 'Return to Dissertation Portfolio (Slide 21)'.*

#### SLIDE 41 (A5): Appendix A5 — Specific Aims, Estimators & Evaluation Metrics (#sec-appendix-metrics)

* **Module Context:** Appendix A5 — Formal Mathematical Specification & Predictive-vs.-Visual Wall  
* **Target Pacing:** `On-Demand Q&A`

##### Spoken Presenter Script

Slide A5 provides a single-slide formal mathematical specification across Specific Aims 1 through 3, documenting the pre-specified study design, data transformation, and formal estimator for every pipeline metric—and explicitly distinguishing supervised predictive and explanatory estimators (TreeSHAP, AXP, IR, IE) from exploratory visual analytics (DTW, FP-Growth, bupaR). All pre-specified estimators and post-execution sensitivity discoveries are evaluated empirically on the untouched 2019 calendar holdout in Module VI (Slides 24 through 32).

##### Podium Delivery & Visual Callout

* Point out the strict firewall keeping exploratory visual analytics (`DTW`, `FP-Growth`, `bupaR`) out of the supervised predictor matrix `X`.

##### Strategic Defense Tips & Committee Q&A Pivots

* **Why Exclude FP-Growth from IR?** Association rules measure marginal co-dispensing frequency, whereas `IR` perturbations measure in-model predicted risk attenuation holding covariates fixed.

##### Return / Jump Navigation

*Click 'Return to Consensus Attribution (Slide 18)' or 'Module VI Results (Slide 24)'.*

---

## PAGE 15 OF 16 — SHEET 8 OF 8 (FRONT / RECTO (Odd Page)) • APPENDIX SLIDES 42–44 + NOTES

#### SLIDE 42 (A6): Appendix A6 — Master Acronym & Technical Nomenclature Reference (#sec-appendix-acronyms)

* **Module Context:** Appendix A6 — Multidisciplinary Glossary Across Modules I–VII  
* **Target Pacing:** `On-Demand Q&A`

##### Spoken Presenter Script

Appendix A6 provides a comprehensive quick-reference glossary of all governance, clinical, biostatistical, machine learning, formal logic, and cloud engineering acronyms used across the seven scientific modules of the dissertation—including CRISP-DM, TRIPOD+AI, the Clinical OODA Loop, PROBAST, Virginia APCD, Milliman Health Cost Guidelines (`HCG`: `hcg_setting`, `hcg_line`, `hcg_detail` emergency acuity codes `P51b`, `O11`, `P33`), CPIC pharmacogenomics, Silent ADEs, GBDT tree geometries, TreeSHAP, Formal Feature Attribution (FFA), Conjunctive Normal Form (CNF), Abductive Explanations (AXPs), Intervention Rate (IR), Interaction Effect (IE), and our zero-PHI Medication Therapy Management (MTM) Clinical Decision Support (CDS) prototype.

##### Podium Delivery & Visual Callout

* Keep available as a shared vocabulary bridge across pharmacy, biostatistics, and systems engineering.

##### Strategic Defense Tips & Committee Q&A Pivots

* **Milliman HCG Codes:** `P51b` (ED Visits), `O11` (Observation Care), and `P33` (Urgent Care) capture acute unscheduled outpatient crises uniformly across commercial, Medicaid, and Medicare Advantage plans.

##### Return / Jump Navigation

*Return to active discussion slide via Reveal.js hash navigation.*

#### SLIDE 43 (A7): Appendix A7 — Regulatory Scope Demarcation & HIPAA Cloud Architecture (#sec-appendix-hipaa)

* **Module Context:** Appendix A7 — FTC Consumer Self-Use vs. Hospital Two-Tier HIPAA BAA & AWS Controls  
* **Target Pacing:** `On-Demand Q&A`

##### Spoken Presenter Script

Slide A7 addresses the critical governance, regulatory, and legal questions regarding our clinical decision-support architecture.  
First, we establish a strict statutory demarcation: when an individual patient directly enters their own personal health data or uploads consumer genetic files for personal use, federal statute dictates that HIPAA does not govern that relationship. It falls under Federal Trade Commission Section 5 and the FTC Health Breach Notification Rule, meaning a BAA is neither required nor legally recognized.  
Second, when deployed within a healthcare provider or hospital system processing ePHI under 45 CFR Section 160.103, compliance is governed by a two-tier legal BAA execution chain: the covered entity signs BAA number 1 with our legal entity, and our legal entity executes BAA number 2 directly with AWS through AWS Artifact covering the underlying cloud infrastructure.  
Third, on AWS, services are 'HIPAA-eligible,' not automatically compliant. We implement the full technical safeguard suite: mandatory KMS envelope encryption on S3, Firecracker microVM ephemeral compute with zero persistence in slash-tmp, strict CloudWatch log scrubbing to prevent ePHI from entering cloud telemetry, and CloudTrail audit logging under 45 CFR Section 164.312(b). Furthermore, any external AI APIs outside of AWS Bedrock would require their own signed BAA.  
Finally, in response to Dr. Gal's and Dr. Price's guidance, our VCU Wright Center CCTR Pilot Grant formally funds institutional compliance and university legal review to vet our ephemeral compute posture and execute the required hospital BAA and DUA framework prior to prospective clinical pilot testing.

##### Podium Delivery & Visual Callout

* Deliver the 4-part legal & technical architecture crisply—this directly answers Dr. Gal's and Dr. Price's November 9 cloud compliance question!

##### Strategic Defense Tips & Committee Q&A Pivots

* **Institutional Next Step:** Our VCU Wright Center CCTR $25,000 Pilot Grant explicitly budgets VCU Health Compliance and university legal counsel review prior to any prospective clinical EHR integration.

##### Return / Jump Navigation

*Click 'Return to Aim 3 Decision Support (Slide 20)' or 'Study Boundaries (Slide 22)'.*

#### SLIDE 44 (A8): Appendix A8 — PGx Gene Markers & Mechanistic Networks by Cohort (#sec-appendix-gene-markers)

* **Module Context:** Appendix A8 — PharmGKB VIP & CPIC Level A PK/PD Pathways  
* **Target Pacing:** `On-Demand Q&A`

##### Spoken Presenter Script

Slide A8 provides the deep mechanistic and biological defense for the specific pharmacogenomic gene markers identified across our two clinical cohorts, directly synthesizing PharmGKB Very Important Pharmacogene (VIP) pathways and CPIC Level A guidelines.
In the Opioid Cohort, we focus on the pharmacokinetic anchor CYP2D6 in hepatic microsomes and the pharmacodynamic anchor OPRM1 in the central nervous system. CYP2D6 controls the rate-limiting phase I bioactivation of prodrug opioids—specifically O-demethylating codeine to morphine and tramadol to O-desmethyltramadol, which possesses over one-hundred-fold higher affinity for the mu-opioid receptor. Downstream, OPRM1 governs the Gi/Go protein-coupled receptor cascade across brainstem respiratory pacemakers like the pre-Bötzinger complex and mesolimbic reward pathways. The clinical phenotype is bidirectionally lethal: Ultra-rapid Metabolizers convert prodrugs at accelerated rates, causing a sudden toxic surge in active mu-agonists and fatal respiratory depression at standard doses; conversely, Poor Metabolizers carrying null alleles or patients with the OPRM1 118G variant experience profound analgesic failure, driving dose escalation, accidental toxicity, and ED encounters. Supporting shunt enzymes like CYP3A4 and glucuronidation clearance via UGT2B7 modulate systemic exposure.
In the Polypharmacy Cohort, adverse drug events in older adults are driven by a multi-target network across hepatic clearance, transport, and coagulation. First, CYP2C19 loss-of-function star alleles (*2 and *3) impair the bioactivation of clopidogrel into its active thiol metabolite, failing to inhibit platelet P2RY12 and precipitating stent thrombosis and major adverse cardiovascular events. Second, S-warfarin clearance is governed stereoselectively by hepatic CYP2C9, while its therapeutic target is VKORC1 in the endoplasmic reticulum; patients combining CYP2C9 reduced-function variants with the VKORC1 -1639A low-expression promoter variant experience severe clearance failure and extreme target sensitivity, driving supratherapeutic INR and catastrophic hemorrhage. Finally, SLCO1B1 encodes the OATP1B1 sinusoidal influx transporter; the star 5 variant halts hepatic statin uptake, trapping statins in systemic circulation where they reach toxic skeletal muscle concentrations, triggering severe statin-induced myopathy and rhabdomyolysis.

##### Podium Delivery & Visual Callout

* Walk cohort-by-cohort: bidirectional lethality in Opioid (`CYP2D6` UM vs. PM + `OPRM1`), then hepatic/transporter cascade in Geriatric Polypharmacy.

##### Strategic Defense Tips & Committee Q&A Pivots

* **Direct Tie to Empirical Ch. 4 Top IR Drivers:** Notice `simvastatin` was our #1 `IR` deprescribing target on Slide 29—directly governed by `SLCO1B1*5` (`c.521T>C`) OATP1B1 hepatic uptake impairment!

##### Return / Jump Navigation

*Click 'Return to CPIC FI (Slide 9)' or 'Appendix A3 (Slide 39)'.*

---

## PAGE 16 OF 16 — SHEET 8 OF 8 (BACK / VERSO (Even Page)) • MASTER COMMITTEE Q&A PLAYBOOK + ETD CHECKLIST

* **Q1 (DuckDB vs. Spark/EMR):** Partition-first DuckDB across 63 `Age × Year` slices on Spot `x2iedn.8xlarge` eliminated geriatric shuffle `exit 137` OOMs, achieving `18.1M rec/min` (`15.1x`), `0/189` failures, and `$6–$10/pass` (`-99%` vs. `$23k/mo` EMR).
* **Q2 (5-Set CNF Non-Circularity):** Sets 1–2 sample boosting/random rules independently of SHAP; Set 4 (`SHAP=0` fallback) implements *Huang & Marques-Silva (2024)*; Table S5 proves `60%` (`3/5`) FFA pair retention when drug SHAP drops to `Jaccard = 0.00`.
* **Q3 (Claims PGx vs. Genotyping):** Claims quantify cumulative CPIC drug exposure (`pgx_num_drugs` #1, `pgx_num_cpic_drugs` #4) without race-based allele imputation; patient genotypes layer at the bedside in Tab 5 (`60 ms`).
* **Q4 (Triple Whammy & U-Shaped Z-Code Shield):** `furosemide + HCTZ + lisinopril` (`12.3%` in `85–114`) collapses glomerular pressure; `Q2` Z-code wellness/med-rec stewardship cuts 21-day ADEs by `75%` (`OR = 0.25`, `3.0%` vs. `10.7%`).
* **Q5 (Zero-PHI vs. Two-Tier HIPAA BAA):** Direct patient self-use falls under FTC Act § 5 / HBNR (no BAA); hospital ePHI deployment executes Two-Tier BAA (#1 Hospital↔Entity, #2 Entity↔AWS Artifact) + KMS, zero `/tmp` persistence, CloudWatch scrubbing, and VCU CCTR legal audit (`Appendix A7`).
* **Q6 (2016–2019 Secular Drift & Sparsity):** 2020 quarantined per *Ochalek, Gal et al. (JAMA 2020)*; FDA SaMD PCCP triggers retraining on `>15%` quarterly SHAP rank drift; Imputation of Normality holds `|Δp̂| < 0.06` up to `70%` bedside missingness.
