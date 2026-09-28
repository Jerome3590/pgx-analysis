# VCU School of Pharmacy Research Day 2026 — Abstract

Event: 10 November 2026 · Submit by 26 October 2026 · Poster + laptop demo  
Presenting author: R. Jerome Dixon · Mentor: Elvin T. Price  
Trim if the call is shorter than ~250 words (body below is 275 words excluding title/authors).

---

**Title**  
Translating Ensemble Models of Opioid and Polypharmacy Emergency Department Risk into a Serverless Pharmacogenomic Dashboard

**Authors**  
R. Jerome Dixon<sup>1,2</sup>, Elvin T. Price<sup>1,3</sup>

<sup>1</sup>Department of Pharmacotherapy and Outcomes Science, School of Pharmacy, Virginia Commonwealth University  
<sup>2</sup>Ph.D. Program in Integrative Life Sciences, School of Life Sciences and Sustainability, Virginia Commonwealth University  
<sup>3</sup>ConvergenceLabs@VCU—Health Outcomes, Virginia Commonwealth University

**Background**  
Prescription opioid–related and polypharmacy emergency department (ED) visits are common, yet most published risk models stay in research notebooks. Pharmacists need a point-of-review tool that scores a regimen, shows model-based drivers, and attaches CPIC gene–drug context without uploading a medical record.

**Objective**  
To implement published, leakage-corrected 2019 holdout ensembles as a public, privacy-first dashboard for opioid ED (ICD-10 F11.xx) and polypharmacy ED risk.

**Methods**  
Virginia All-Payer Claims Database models were trained and tuned on 2016–2018 only and evaluated on an untouched 2019 temporal holdout. Density-routed CatBoost, XGBoost, and XGBoost-RF learners were ensembled by age band. Selected-model discrimination is as reported in *Clinical and Translational Science* (opioid ED Table 2; polypharmacy Table 2). The dashboard serves those ensembles from a serverless container (static frontend plus Lambda), with training-median imputation for sparse codes, FFA/SHAP scenario views that do not rescore the ensemble, and stateless CPIC lookups (573 gene–drug pairs) that store no personal identifiers. Unphased consumer DNA or VCF input is labeled indeterminate when a unique star allele cannot be resolved.

**Results**  
On the 2019 holdout, selected opioid ED models achieved AUROC 0.800–0.890, PR-AUC 0.375–0.741, and PR lift 2.3×–4.4× (low-density partitions except age-band aggregate at 75–84 and 85–114; ages 0–12 excluded). Selected polypharmacy models achieved AUROC 0.686–0.877, PR-AUC 0.101–0.335, and PR lift 1.6×–4.2×. Deployed inference met the sub-100 ms warm-path target (mean 6 ms; cold start mean 2.1 s). Sparse input (≤5 features) changed predicted risk by mean |Δp̂| = 0.10 versus a full-feature baseline. The live tool is https://pgx.jerome-dixon.io/.

**Conclusions**  
Leakage-corrected holdout models can be delivered as a no-PHI pharmacist review dashboard. Outputs are observational associations for risk stratification and PGx review, not treatment-effect estimates.

**References (if the form allows)**  
Dixon RJ, Price ET. *Clin Transl Sci.* doi:10.1111/cts.70690; doi:10.1111/cts.70718; doi:10.1111/cts.70697.

**Keywords**  
pharmacogenomics; opioid risk; polypharmacy; clinical decision support; ensemble learning
