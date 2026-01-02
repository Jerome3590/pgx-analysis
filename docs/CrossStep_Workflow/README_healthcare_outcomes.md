## Healthcare Outcomes Rationale for Cohort Design

This document explains how the cohort structure in this project lines up with
established healthcare outcomes research, and why we focus our **heaviest**
Monte Carlo (MC‑CV) feature importance work on specific cohort groups.

The key idea is to concentrate high‑fidelity, health‑grade modeling on
subpopulations that:

- Are **clinically coherent**, and  
- Are **well‑known in the literature** as high risk for the outcomes we study
  (opioid‑related ED visits, polypharmacy‑related adverse events).

---

## Cohort Groups in This Project

We conceptually organize analysis into two main cohort groups:

- **Cohort Group 1 – Opioid ED (`opioid_ed`)**
  - Focus: opioid use disorder / opioid‑related ED events.
  - Age bands: “cohorts 1–5” in our `AGE_BANDS` definition  
    (you can think of these as the **younger and mid‑age adults**, e.g. 0–12,
    13–24, 25–44, 45–54, 55–64).
  - Analysis intensity:
    - Full 3‑model ensemble (CatBoost, XGBoost, XGBoost RF mode).
    - 50 MC‑CV splits with temporal validation (train 2016–2018, test 2019).
    - Permutation‑based feature importance on the entire 2019 holdout.

- **Cohort Group 2 – Polypharmacy ED visits (`non_opioid_ed`)**
  - Focus: polypharmacy‑related ED visits, especially in older adults with
    multiple chronic conditions.
  - Age bands: “cohorts 6–8” in `AGE_BANDS`  
    (older adults, roughly 65–74, 75–84, 85–94).
  - Analysis intensity:
    - Same full 3‑model ensemble and MC‑CV configuration as Opioid ED.
    - Emphasis on **polypharmacy burden** and medication complexity as drivers
      of ED utilization.

Other `(cohort, age_band)` combinations can still be run with lighter
configurations (fewer splits, fewer models, or restricted feature sets), but
our **primary, publication‑grade inference** is anchored on these two groups.

---

## Alignment With Opioid Outcomes Literature

Multiple national datasets and studies (e.g., **CDC Vital Signs / NCHS reports**,
and the **SAMHSA National Survey on Drug Use and Health (NSDUH)**) show a
consistent pattern for opioid misuse and opioid use disorder (OUD):

- **Misuse and OUD prevalence:**
  - Highest rates of **non‑medical opioid use and OUD** are typically seen in
    **late adolescents and young adults (roughly 18–25)**, with substantial
    burden carried into the **26–34** and **35–44** age groups.
  - Middle‑aged adults (often 35–54) have historically shown some of the
    highest **overdose mortality rates**, reflecting cumulative exposure and
    comorbidity.

- **ED and acute‑care utilization:**
  - Young and mid‑age adults with OUD frequently present to EDs with
    overdose, withdrawal, or complications of injection drug use.
  - These age groups account for a disproportionate share of **opioid‑related
    ED visits and hospitalizations** relative to their population size.

Our decision to focus the **opioid_ed** feature importance work on cohorts 1–5
(spanning childhood through mid‑adulthood) is therefore:

- Consistent with observed **age distributions of misuse and OUD**, and  
- Clinically meaningful for identifying early and mid‑life risk patterns that
  can inform prevention and intervention.

---

## Alignment With Polypharmacy Outcomes Literature

Polypharmacy (commonly defined as **≥5 chronic medications**) and “excessive”
polypharmacy (e.g., **≥10 medications**) are consistently associated with
adverse outcomes in **older adults**, including ED visits, hospitalizations, and
mortality.

Several recent studies illustrate this:

- **Continuous polypharmacy and adverse outcomes (older adults 65–84)**  
  - A large cohort of >6 million older adults found that continuous
    polypharmacy (≥5 drugs for ≥90 days) was associated with higher rates of:
    - All‑cause hospitalization
    - ED visits
    - All‑cause mortality  
  - (Example: PMID **39144630**.)

- **Inappropriate polypharmacy and ED visits in older cancer patients**  
  - A national Korean study of older adults receiving anti‑neoplastic therapy
    reported that ~85% had **inappropriate polypharmacy** (potentially
    inappropriate medications, drug–drug interactions), which significantly
    increased the odds of ED visits during treatment.  
  - (Example: PMID **33037903**.)

- **Polypharmacy, frailty, and mortality in ED patients ≥70 years**  
  - A prospective ED‑based cohort (patients ≥70) found:
    - 43% had polypharmacy (≥5 meds),
    - 18% had excessive polypharmacy (≥10 meds), and
    - Polypharmacy was independently associated with increased mortality,
      with effects modified by frailty and comorbidity.  
  - (Example: PMID **35723840**.)

Across these and related studies, **high‑risk age cohorts for polypharmacy and
ED utilization** are consistently:

- Adults **≥65 years**, especially  
- Those **≥75 years** and/or with multiple chronic conditions and frailty.

That is exactly the age range covered by our **cohort group 2** focus
(`non_opioid_ed`, cohorts 6–8), so concentrating heavier feature importance
analysis there is directly supported by the literature.

---

## Why Deep MC‑CV Only on Priority Cohorts Is Reasonable

From a healthcare outcomes perspective, there is a standard tradeoff between:

- **Depth** (number of splits, models, and robustness checks per cohort), and  
- **Breadth** (how many cohorts / age bands are studied at that depth).

Common practice in high‑quality observational studies is to:

- **Pre‑specify primary cohorts / subgroups** (e.g., opioid ED vs polypharmacy
  ED; younger vs older adults), and  
- Invest the most intensive modeling and validation effort there, while
  treating other cuts as **secondary or exploratory**.

Our design mirrors this:

- Opioid‑related risk is deeply characterized in **younger and mid‑age adults**
  (cohort group 1).  
- Polypharmacy‑related risk is deeply characterized in **older adults**
  (cohort group 2).  
- Both groups use the **same high‑rigor MC‑CV protocol** (temporal validation,
  50 splits, 3‑model ensemble, permutation importance), making results
  comparable and suitable for publication‑grade interpretation.

This approach is therefore **supported by and consistent with** modern
healthcare outcomes research norms.


