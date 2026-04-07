# INFORMS Healthcare 2026
**Conference:** INFORMS Healthcare Conference 2026
**Submission deadline:** TBD — check [meetings.informs.org/wordpress/healthcare](https://meetings.informs.org/wordpress/healthcare/submit/)
**Format:** 250-word abstract · In-person presentation · English

---

## Talk Title
*Formal Feature Attribution as a Causal Calculator for Drug-Drug Interaction Risk in Geriatric Polypharmacy: Evidence from a State-Level All-Payer Claims Database*

**Speaker:** R. Jerome Dixon, Ph.D. Candidate
**Co-author:** Elvin T. Price, Pharm.D., Ph.D., FAHA
**Based on:** Dixon & Price, manuscript submitted to *CPT: Pharmacometrics & Systems Pharmacology* (Wiley)

**Topical areas:** Personalized Medicine · AI/Machine Learning · Bioinformatics · Data Analytics · Geriatrics

---

## Abstract (250 words)

**Background:** Polypharmacy (≥5 concurrent medications) affects more than 40% of adults aged ≥65 years and drives adverse drug events (ADEs) responsible for an estimated 100,000 hospitalizations annually. Pairwise DDI databases flag co-prescribed pairs in isolation, missing synergistic multi-drug effects where each constituent pair appears benign.

**Methods:** Using Virginia's All-Payer Claims Database (APCD; 6,929,576 patients; 2016–2019), we constructed a non-opioid ED cohort (n = 1,182 geriatric cases; 65–114 years; 5:1 matched controls). A 30-day causality window isolated proximal drug exposures. A per-event-density ensemble (CatBoost/XGBoost/XGBoost-RF; Optuna-tuned; training: 2016–2018; holdout: 2019) fed Formal Feature Attribution (FFA) extended to multi-feature interaction testing of drug pairs and triplets, with Intervention Rate (IR) scores quantifying expected risk reduction per deprescribing action. FP-Growth network visualization contextualized causal findings within drug co-occurrence topology.

**Results:** All 8 age bands achieved PR-AUC ≥ 0.908 (peak: 0.997 at 75–84 band). FFA identified 115 synergistic drug pairs and 5,021 high-risk triplets invisible to pairwise DDI databases. Levofloxacin acted as a pharmacological hub (CYP1A2 inhibition), appearing in 4 of the top 5 synergistic pairs: acetaminophen + levofloxacin (IE = 16.3) and levofloxacin + lorazepam (IE = 11.9). Top IR-ranked deprescribing targets: simvastatin (IR 7.0×10⁻⁴), furosemide (IR 2.0×10⁻⁴), alprazolam (IR 1.0×10⁻⁴). A U-shaped Z-code monitoring effect identified a fragmented-care phenotype (Q4; OR ≈ 0.94 vs. unmonitored) distinct from protectively monitored patients (Q2; OR = 0.25).

**Conclusions:** FFA multi-feature interaction analysis constitutes a causal calculator for polypharmacy ADE risk, detecting synergistic combinations invisible to pairwise DDI databases and yielding IR-ranked deprescribing priorities actionable from standard claims data without prospective genotyping.

---

## Slide Outline (~20 min)

### Slide 1 — Motivation (2 min)
- Polypharmacy ≥5 drugs: >40% of adults ≥65 yrs; 100,000 hospitalizations/yr
- Pairwise DDI databases (Drugs.com, Micromedex): flag pairs in isolation → miss synergistic multi-drug risk
- Gap: no claims-based causal calculator that quantifies multi-drug synergy AND ranks deprescribing priorities

### Slide 2 — FFA as a Causal Calculator (3 min)
- FFA extracts Boolean rules from XGBoost decision trees → symbolic, interpretable causal hypotheses
- Extended to **multi-feature interaction testing**: pairs (IE score) and triplets
- IE = lift-based interaction effect; IE > 1.0 = synergistic; 95% CI bootstrapped (n=1,000)
- Intervention Rate (IR) = expected Δp̂ if drug removed; ranks deprescribing priority over raw frequency

### Slide 3 — The Levofloxacin Hub (4 min)
- FFA identified 115 synergistic drug pairs and 5,021 high-risk triplets across geriatric bands
- Levofloxacin in 4 of top 5 pairs — CYP1A2 hub: frequently prescribed for CAP without medication review
  - Acetaminophen + Levofloxacin: IE = 16.3 (CYP1A2 reactive metabolite production)
  - Levofloxacin + Lorazepam: IE = 11.9 (QT prolongation + CNS depression)
  - Carvedilol + Levofloxacin: IE = 10.5 (beta-blocker plasma level elevation)
- **Figure:** `fig_network.png` — FP-Growth co-occurrence network overlaid with FFA synergistic pairs

### Slide 4 — Triplet Interactions: Beyond Pairwise Alerts (3 min)
- 312 triplets exceeded synergistic IE threshold (IE > 1, 95% CI > 0)
- **Triple Whammy** (85–114 band): furosemide + hydrochlorothiazide + lisinopril — present in 12.3% of cases; STOPP D4 criterion; convergent renal clearance burden
- **Digoxin + furosemide + amiodarone** (75–84 band; IE = 8.7): each pairwise flag = "major" → alert fatigue; triplet reveals convergent pharmacokinetic + pharmacodynamic synergy
- Pairwise review alone = insufficient; triplet-level FFA catches what individual DDI alerts miss

### Slide 5 — IR-Ranked Deprescribing Priorities (3 min)
- **Figure:** `fig_ir.png` — top 15 drugs by IR score across three geriatric bands
- Top 3 targets (consistent across 65–74, 75–84, 85–114):
  1. **Simvastatin** — IR 7.0×10⁻⁴; Beers (CYP3A4); interaction with amlodipine/diltiazem
  2. **Furosemide** — IR 2.0×10⁻⁴; triple-whammy + digoxin toxicity via hypokalemia
  3. **Alprazolam** — IR 1.0×10⁻⁴; Beers CNS/falls; IE amplified by levofloxacin co-prescription
- IR rank correlation ρ = 0.53–0.68 across bands → same deprescribing protocol generalizes to 65–114

### Slide 6 — The Z-Code Paradox: Managed vs. Unmanaged Polypharmacy (3 min)
- U-shaped monitoring–risk relationship (adjusted for age band, sex, drug count)
- **Q2** (1–12% Z-code claims): OR = 0.25 (95% CI 0.18–0.34) — protective; driven by Z71.89 medication counseling
- **Q4** (≥12% Z-code claims): OR ≈ 0.94 — equivalent risk to unmonitored patients
- Q4 = fragmented-care phenotype: preventive screenings without coordinated medication reconciliation
- **Figure:** `fig_zcode.png` — violin plots + adjusted OR by quartile
- Clinical implication: monitoring *type* matters more than monitoring *volume*

### Slide 7 — Clinical So-What (2 min)
- FFA causal calculator goes beyond "flag and forget": IR scores give a continuous deprescribing priority
- Levofloxacin is *substitutable* (amoxicillin/clavulanate) → highest-yield intervention for CAP in geriatric patients
- Medication *counseling* (Z71.89), not volume of visits, is the protective monitoring activity
- Deployable from standard claims data — no prospective genotyping infrastructure required
- Based on: Dixon & Price, manuscript under review, *CPT: Pharmacometrics & Systems Pharmacology*

---

## Source Chapter
`CH_4/ch04_psp.qmd` — submitted to *CPT: Pharmacometrics & Systems Pharmacology* (Wiley, PSP)

## Key Figures (from submission package)
| Figure | File | Used in Talk |
|--------|------|-------------|
| Figure 2 — Drug network + FFA overlay | `output/submission/cpt_psp/ch04/figures/Figure_2.tiff` | Slide 3 |
| Figure 3 — IR scores (top 15 drugs) | `output/submission/cpt_psp/ch04/figures/Figure_3.tiff` | Slide 5 |
| Figure 4 — Z-code U-shape | `output/submission/cpt_psp/ch04/figures/Figure_4.tiff` | Slide 6 |

## Ethics / Dual-Submission Note
Presenting journal-submitted work at INFORMS Healthcare is permitted under ICMJE norms — conference presentations are a distinct channel from duplicate publication. Slides should note: *"Based on manuscript submitted to CPT: Pharmacometrics & Systems Pharmacology (Dixon & Price, under review)."* No INFORMS proceedings paper is required or requested (abstract only).
