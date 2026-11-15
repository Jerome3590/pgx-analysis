# Appendix Deck Presentation Script & Technical Defense Bank (Slides 37–45: Appendices A1–A8)

- **Candidate:** R. Jerome Dixon, MDA, MSc IT
- **Program:** Integrative Life Sciences Doctoral Program & VCU School of Pharmacy (Dept. of Pharmacotherapy & Outcomes Science)
- **Final Dissertation Defense Date:** November 23, 2026
- **Standalone Interactive Appendix Deck (9 Slides):** https://pgx.jerome-dixon.io/23NOV_defense/final_defense_appendix_slides.html
- **Standalone 16:9 Appendix PDF:** https://pgx.jerome-dixon.io/23NOV_defense/final_defense_appendix_slides.pdf
- **Complete 45-Slide Defense Deck:** https://pgx.jerome-dixon.io/23NOV_defense/

---

## Quick Committee Q&A Routing Matrix (Appendices A1–A8)

| Appendix Slide | Core Defense Topic | Primary Committee Domain | Linked Core Slides |
| :--- | :--- | :--- | :--- |
| **Slide 37 (Index)** | Technical Defenses & Supplemental Evidence Overview | All Committee Members | Slide 35 (Q&A), Slide 36 |
| **Slide 38 (A1)** | Formal Feature Attribution (FFA): O(T · L) CNF, PySAT Hitman AXPs, Mirror Ablation (-61%) & Table S5 (60% FFA vs. SHAP Jaccard = 0.00) | Prof. Bost & Dr. Gal | Slide 18, Slide 26, Slide 28, Slide 30 |
| **Slide 39 (A2)** | Step 3b Target Leakage: 3 Contamination Vectors & τ_post ≥ 0.80 Whitelist (348 Purged / 682 Locked) | Dr. Gal & Dr. DeShazo | Slide 19, Slide 25 |
| **Slide 40 (A3)** | Step 5 CPIC Drug Mapping (≥ 95% Fuzzy Match, 573 Pairs), Biogeographical Ancestry & DTC Phasing Guard | Dr. Price & Dr. Wijesinghe | Slide 14, Slide 17, Slide 31 |
| **Slide 41 (A4)** | Computational Pipeline: 7 Reproducible Notebooks, S3 SHA-256 Checkpoints & Fat-Spot EC2 (15.1×) | Prof. Bost & Dr. Gal | Slide 21, Slide 24 |
| **Slide 42 (A5)** | Formal Methodological Estimators & Predictive-vs.-Visual Wall (IR_Δp̂, Δ₁₁/IE, DTW, FPG) | All Committee Members | Slide 18, Slide 22, Slides 24–33 |
| **Slide 43 (A6)** | Master Acronym & Technical Nomenclature Reference (CRISP-DM, TRIPOD+AI, HCG P51b/O11/P33, AXP) | General Reference | Slide 12, Slide 16, Slide 22 |
| **Slide 44 (A7)** | Regulatory Scope Demarcation (FTC Act § 5 Consumer vs. 45 CFR § 160.103 Two-Tier Hospital HIPAA BAA & CCTR Aim 2) | Dr. Gal, Dr. Price & Dr. DeShazo | Slide 20, Slide 22, Slide 34 |
| **Slide 45 (A8)** | Pharmacogenomic Gene Markers & Mechanistic Networks (CYP2D6/OPRM1 vs. CYP2C19, CYP2C9/VKORC1, SLCO1B1) | Dr. Price & Dr. Wijesinghe | Slide 09, Slide 29, Slide 40 |

---

## Appendix Deck Slide 01 (Full Deck Slide 37): Appendix Section Divider — Technical Defenses & Supplemental Evidence (#sec-appendix)

- **Module & Context:** Technical Backup Bank — Interactive Jump Index to Appendices A1–A8
- **Visual Callout Anchors:** **8 Technical Backup Slides:** `A1` FFA CNF/AXP Math • `A2` Step 3b Leakage • `A3` Step 5 CPIC Mapping • `A4` 7 Jupyter Notebooks • `A5` Estimators & Metrics • `A6` Master Acronyms • `A7` HIPAA/FTC Cloud Architecture • `A8` PGx Gene Networks

### 1. Verbatim Spoken Defense Script (`final_defense_appendix_slides.qmd`)

"Slide 37 serves as our interactive Technical Appendix index across Appendices A1 through A8—providing on-demand mathematical proofs for Conjunctive Normal Form and Abductive Explanations (A1), Step 3b event partitioning (A2), Step 5 CPIC drug mapping (A3), our seven reproducible Jupyter notebooks (A4), formal Aim 1–3 estimators (A5), master nomenclature (A6), two-tier HIPAA and FTC cloud governance (A7), and cohort-specific PharmGKB and CPIC Level A molecular pathways (A8)."

### 2. Detailed Technical Walkthrough & Panel Notes

Transitioning now to our backup section, this slide introduces the Appendix: Technical Defenses & Supplemental Evidence, functioning as our Committee Q&A Backup Bank.
Purpose & Contents: This section houses the granular methodological, algebraic, and systems-level documentation supporting our presentation. It provides immediate access to complete mathematical derivations, exhaustive feature audit catalogs, pharmacogenomic scoring matrices, two-tier HIPAA and FTC cloud governance architectures (A7), cohort-specific PharmGKB and CPIC Level A molecular gene networks (A8), and reproducible 7-notebook computational pipelines (A4) to address specific committee inquiries during the discussion.
Resource Access: As linked directly in the button below the subtitle, both the complete 45-slide 16:9 defense deck (final_defense_slides.pdf) and the standalone 9-slide Technical Appendix deck (final_defense_appendix_slides.pdf & final_defense_appendix_slides.html) are compiled and synced for direct review.
We are fully prepared to dive into Appendices A1 through A8 to examine any specific equation, code implementation, or validation table.

### 3. Committee Q&A Delivery Cues & Pivots

- **Delivery Cue:** Hold in reserve during Q&A; use keyboard hash links or return buttons for instant non-linear navigation.
- **Faculty Mapping:** `A1, A4, A5` -> Prof. Bost & Dr. Gal; `A2` -> Dr. Gal; `A3, A8` -> Dr. Price & Dr. Wijesinghe; `A6` -> General Reference; `A7` -> Dr. Gal, Dr. Price & Dr. DeShazo.
- **Navigation Return / Pivot:** *Select the specific Appendix slide (A1–A8) matching the committee member's question.*

---

## Appendix Deck Slide 02 (Full Deck Slide 38): Appendix A1 — Formal Feature Attribution (FFA): CNF & AXP Bounds (#sec-appendix-ffa)

- **Module & Context:** Appendix A1 — Steps 6–8 & 10 Mathematical Proof & Sensitivity Framework
- **Visual Callout Anchors:** **Cross-Architecture Gate:** Exact TreeSHAP on `CatBoost ∩ Best XGB` (≥ Q_75) • **Symbolic Gate:** O(T·L) CNF compilation + PySAT `Hitman` minimal hitting-set AXPs (`Supp ≥ 0.05, Conf ≥ 0.70`) • Validated via Mirror Ablation & Table S5

### 1. Verbatim Spoken Defense Script (`final_defense_appendix_slides.qmd`)

Slide A1 provides the mathematical and empirical defense for our Consensus-Risk architecture across Steps 6 through 8 and Step 10. Because CatBoost uses internal target-statistic hashing on categorical variables, Step 7 first computes exact TreeSHAP on both the best CatBoost model and the best XGBoost variant, enforcing their strict feature intersection as a cross-architecture gate. Step 8 then explodes the decision trees of the best XGBoost variant into CNF clauses and prioritizes the top SHAP-supported matched rules into the PySAT Hitman minimal hitting-set solver to extract exact Abductive Explanations in polynomial O(T times L) time. Importantly, our Consensus-Risk thresholds—75th percentile SHAP, support of 0.05, and confidence of 0.70—operate as a validated precision-volume dial for clinical review capacity, proven empirically in Module VI: first, our Two-Gate Mirror Ablation on Slide 26 retained 20 out of 20 core clinical risk drivers while compressing alert volume by 61 percent and rejecting 252 routine billing codes; and second, our Table S5 utilization-free stress test on Slide 28 demonstrated 60 percent symbolic FFA drug-pair retention even when univariate SHAP Jaccard similarity collapsed to 0.00.

### 2. Detailed Technical Walkthrough & Panel Notes

Turning to Appendix A1, this slide establishes our mathematical verification pipeline for Formal Feature Attribution (FFA), detailing how we compile our best-performing tree ensembles into Conjunctive Normal Form (CNF) and extract Abductive Explanations (AXP).
**The methodology bridges heuristic attribution with exact formal methods across three stages:**
**Step 7 Dual-Model Gate & Step 8 CNF Compilation (Left Panel):**
Dual-Model TreeSHAP Gate: Before any symbolic extraction occurs, we enforce cross-architecture agreement between CatBoost and XGBoost (F_dual = supp(SHAP_CatBoost) ∩ supp(SHAP_BestXGB)). Because CatBoost CTR feature hashing precludes direct tree traversal, we compile the best explicit tree ensemble (selected by PR-AUC, then Recall) into symbolic format.
CNF Encoding: We parse JSON tree dumps, unpack all root-to-leaf paths into disjoint hyper-rectangle tabular intervals via _explode_tree_to_dataframe(), and compile them directly into formal CNF clauses (rule_clauses).
**SHAP-Guided Minimal Hitting-Set AXP & Complexity (Right Panel):**
PySAT Hitman Solver: Rather than brute-forcing tens of thousands of raw matched tree paths per patient, we feed top SHAP-supported matched rules into the PySAT minimal hitting-set solver. This extracts exact, irreducible Abductive Explanations (AXPs / Horn clauses) bound by clinical relevance gates: support Supp(R) ≥ 0.05 and confidence Conf(R) ≥ 0.70.
Synthesized Responsibility Metric: We balance frequency and continuous magnitude by formalizing attribution responsibility (AttrResp) and computing our final combined rank: Score_combined(f) = 0.5 · SHAP_norm(f) + 0.5 · FFA_norm(f).
Tractable Complexity: Tree CNF extraction scales polynomially (O(T · L)), avoiding the co-NP-complete computational bottlenecks inherent in ReLU neural network verification.
**Operational Threshold Validation Protocol (Bottom Callout):**
The operational thresholds (SHAP ≥ P75 ∩ Supp ≥ 0.05, Conf ≥ 0.70) function as a tunable precision-volume dial calibrated to clinical review capacity.
Empirically validated across two Module VI experiments synthesized on Slide 30: (1) Two-Gate Mirror Ablation (Slides 26 & 30) — retains 20/20 (100%) primary clinical risk drivers while compressing pharmacist review load by -61% (112 → 44 features) and rejecting 252 routine E/M billing artifacts; and (2) Table S5 Utilization-Free Specification Stress Test (Slides 28 & 30) — proves 60% (3/5) symbolic FFA drug-pair invariance even when univariate TreeSHAP rankings collapse to Jaccard = 0.00.
Interactive navigation links jump directly back to Consensus Attribution (Slide 18) or Consensus Sensitivity Analysis (Slide 30).

### 3. Committee Q&A Delivery Cues & Pivots

- **Delivery Cue:** Emphasize why `Q75 / 0.05 / 0.70` is an operational precision–volume dial for pharmacist review capacity rather than an arbitrary cutoff.
- **Why Not Explode CatBoost Directly into CNF?** CatBoost uses online target-statistic hashing on categorical combinations; intersecting CatBoost TreeSHAP with XGBoost CNF AXPs gets both shift immunity and exact Boolean rules.
- **Navigation Return / Pivot:** *Click 'Consensus Attribution (Slide 18)' or 'Sensitivity Analysis (Slide 30)'.*

---

## Appendix Deck Slide 03 (Full Deck Slide 39): Appendix A2 — Step 3b Target Leakage: Event-Partitioning Protocol (#sec-appendix-leakage)

- **Module & Context:** Appendix A2 — 3 Contamination Vectors & Symmetric Pre-Index Whitelist
- **Visual Callout Anchors:** **3 Contamination Vectors:** (1) Post-OD MAT rescue (Suboxone, Naloxone) • (2) ED/MAT toxicology (`CPT 80307, G0480`) • (3) Admin follow-up • τ_post(f) ≥ 0.80 → Purge (**348/1,029 purged**; **682 locked** in `safe_feature_filter.json`)

### 1. Verbatim Spoken Defense Script (`final_defense_appendix_slides.qmd`)

Slide A2 details our Step 3b temporal event-partitioning protocol. We classify post-index contamination into three clinical vectors: post-overdose OUD rescue and MAT induction, confirmatory ED and follow-up toxicology screens, and downstream administrative follow-up codes. Any candidate feature whose claims events occur 80% or more on or after the index timestamp t0 is automatically purged, locking an immutable pre-index antecedent whitelist in safe_feature_filter.json. As demonstrated empirically in Module VI on Slide 25, this Step 3b audit purged 348 post-index confounders out of 1,029 candidate features and locked 682 verified pre-index antecedents—replacing artificially leaked 0.99 AUROC scores with honest 0.800 to 0.890 out-of-time 2019 holdout generalization.

### 2. Detailed Technical Walkthrough & Panel Notes

Here in Appendix A2, we walk through our Step 3b Target Leakage — Temporal Event-Partitioning Protocol, which serves as our formal audit to excise post-event artifacts before training our tree ensembles.
The core issue this protocol resolves is the pervasive risk of models learning from consequences rather than causes:
Leakage Vectors and Clinical Mechanisms (Main Table): On the left, we identify three distinct categories of claims that occur at or after the index emergency department event (t ≥ t₀):
Post-Event OUD Rescue & MAT Induction: Medications such as buprenorphine, suboxone, naloxone, and extended-release MAT initiated during emergency department triage or bridge clinics as direct treatment for acute overdose.
Confirmatory Toxicology & Monitoring: Presumptive and definitive urine drug screens ordered in the ED or as post-discharge compliance checks.
Secondary Administrative Follow-Up: Downstream billing codes and long-term therapy encounters generated in response to the index event.
Step 3b Formal Event-Partitioning Rule & Symmetric Whitelist (Lower Panel): To mathematically ensure temporal hygiene, we enforce two strict filters:
Post-Target Ratio Filter: Any feature demonstrating a post-index event concentration ratio τ_post(f) ≥ 0.80 is purged from safe_feature_filter.json.
Symmetric Pre-Index Enforcement: All retained predictors are restricted strictly to antecedent claims (t_claim < t₀), bounded symmetrically across cases and 1:5 matched controls within [t₀ − 365, t₀ − 1] for opioid analyses and [t₀ − 21, t₀ − 1] for polypharmacy analyses.
Context and Scope Demarcation (Right Panel): Without this partition, models fall victim to the classic PROBAST risk-of-bias trap: greedy tree ensembles split heavily on post-event rescue procedures, artificially inflating apparent discrimination by 5 to 20 AUROC points. Empirically validated in Module VI (Slide 25), Step 3b purged 348 / 1,029 (33.8%) post-index confounders and locked 682 verified pre-index antecedents in safe_feature_filter.json—replacing artificially leaked 0.99 AUROC scores with honest 0.800–0.890 untouched 2019 holdout generalization.
The navigation link at the bottom right leads back to our main Target Leakage Defense slide.

### 3. Committee Q&A Delivery Cues & Pivots

- **Delivery Cue:** Walk through the 3 contamination rows, then link directly to Slide 25's empirical purge (`0.99` leaked -> `0.800–0.890` honest holdout).
- **Symmetric Application:** `safe_feature_filter.json` is applied identically to cases and 1:5 controls so feature filtering never introduces differential selection bias.
- **Navigation Return / Pivot:** *Click 'Return to Target Leakage (Slide 19)' or 'Return to CH2 Empirical Audit (Slide 25)'.*

---

## Appendix Deck Slide 04 (Full Deck Slide 40): Appendix A3 — Step 5 CPIC Drug Mapping & Patient Count Features (#sec-appendix-pgx)

- **Module & Context:** Appendix A3 — `5_pgx_analysis` Fuzzy Matching, Ancestry & DTC Phasing Guard
- **Visual Callout Anchors:** **Step 5 & Phasing Guard:** `build_global_drug_cpic_mapping.py` (573 CPIC pairs, 300 drugs, 121 genes; ≥95% fuzzy) → `pgx_num_drugs` & `pgx_num_cpic_drugs` • Biogeographical ancestry (no admin race) • `cpic-unphased-exploratory-v2` (2^k-1 phasing / `*5` CNV lockout)

### 1. Verbatim Spoken Defense Script (`final_defense_appendix_slides.qmd`)

Slide A3 details our Step 5 pharmacogenomic feature engineering pipeline in `5_pgx_analysis` alongside Chapter 6's bioinformatic phasing boundaries. First, `build_global_drug_cpic_mapping.py` maps our APCD drug vocabulary against the official CPIC directory (`573` gene-drug pairs) at a 95% threshold to build `drug_cpic_mapping_global.csv`. Second, `create_pgx_features_patient_level.py` computes two observable exposure counts per patient in DuckDB: `pgx_num_drugs` and `pgx_num_cpic_drugs`. Third, administrative race checkboxes are strictly excluded because pharmacogenomic allele prevalence tracks biogeographical genetic ancestry requiring direct genotyping. Finally, in our Aim 3 Patient Card, `cpic_allele_resolver.py` (`cpic-unphased-exploratory-v2`) never defaults unprobed loci to wild-type `*1/*1`: unphased consumer microarrays with compound heterozygotes (2^(k−1) chromosomal phases) or unassessed `CYP2D6` structural variants trigger `INSUFFICIENT_GENOTYPE_RESOLUTION` and a mandatory CLIA/CAP laboratory test referral, whereas phased clinical alleles execute in 60 milliseconds with 100% concordance across 573 test cases.

### 2. Detailed Technical Walkthrough & Panel Notes

In Appendix A3, we detail our Step 5 CPIC Drug Mapping and Patient-Level Count Features, which establishes our pharmacogenomic feature engineering and formal defense of claims exposure versus unmeasured genotype imputation.
**The slide is structured into three key operational components:**
**1. Global APCD Drug—CPIC Guideline Mapping (build_global_drug_cpic_mapping.py):**
Ingests the official CPIC reference corpus—spanning 573 gene-drug pairs, 300 unique drugs, and 121 pharmacogenes across CPIC Levels A through D.
Extracts unique APCD generic drug vectors (DRUG_*, item_drug_*) from Step 3 outputs and matches them against CPIC drug nomenclature using a ≥ 95% fuzzy-matching threshold to construct drug_cpic_mapping_global.csv.
Generates binary guideline indicators (d ∈ C_CPIC) flagging whether each dispensed medication possesses an actionable clinical CPIC guideline.
**2. DuckDB Patient-Level Drug & CPIC Counts (create_pgx_features_patient_level.py):**
Computes deterministic count features per patient identifier (mi_person_key): total distinct pre-index drug exposures (pgx_num_drugs) and distinct CPIC guideline-bound drugs (pgx_num_cpic_drugs).
Aggregates cohort-level exposure distributions in DuckDB, contrasting patients with baseline pharmacotherapy against those carrying actionable CPIC-guideline regimens.
**3. Claims-Derived Exposure vs. Genotype Imputation Defense:**
Methodological Boundary: Our supervised models evaluate observable dispensed prescription claims directly, strictly avoiding the conflation of administrative billing codes with molecular DNA sequencing or unmeasured germline biology.
Biogeographical Genetic Ancestry vs. Administrative Race: Administrative race categories are rigorously excluded in favor of verified biogeographical genetic ancestry, preventing social demographic proxies from substituting for germline CYP metabolizer phenotypes.
- Consumer-Array Phasing Lockout (cpic_allele_resolver.py, cpic-unphased-exploratory-v2): Unprobed consumer-array loci never default to wild-type *1/*1; compound heterozygotes (k ≥ 2 SNPs ⇒ 2^(k−1) phases) or unprobed CYP2D6*5 CNVs trigger INSUFFICIENT_GENOTYPE_RESOLUTION and refer for confirmatory clinical CLIA/CAP testing.
Translational Continuity (Aim 1 → Aim 3): Supervised population models (Steps 5–6) evaluate observable claims CPIC exposure directly, while interactive point-of-care patient genotype and diplotype actionability cards execute in 60 milliseconds within Tab 5 of our stateless Aim 3 AWS Lambda microservice (Slide 31).
Footer links return directly to our Claims Ingestion & Schema or downstream Model Selection discussions.

### 3. Committee Q&A Delivery Cues & Pivots

- **Delivery Cue:** Highlight refusing administrative race proxies AND locking out unphased compound-heterozygote DTC consumer arrays (`CLIA/CAP` referral).
- **Bioinformatic Phasing Safety:** Unprobed array loci never default to wild-type `*1/*1`; k ≥ 2 heterozygous SNPs (2^(k−1) phases) or unprobed `CYP2D6*5` CNVs return `INSUFFICIENT_GENOTYPE_RESOLUTION`.
- **Navigation Return / Pivot:** *Click 'Return to Claims Schema (Slide 14)' or jump to Appendix A8 (Slide 45) for gene networks.*

---

## Appendix Deck Slide 05 (Full Deck Slide 41): Appendix A4 — Computational Pipeline & 7-Notebook Architecture (#sec-appendix-notebooks)

- **Module & Context:** Appendix A4 — Full Idempotent S3/NVMe Reproducibility (`pgx-analysis`)
- **Visual Callout Anchors:** **7 Sequential Notebooks (Steps 0–10):** `0_config` → `1_cohort_workflow` → `2_feature_importance` → `3_model_train_shap_ffa` → `3_model_sensitivity` → `4_dashboard_visuals` → `5_build_and_deploy` • SHA-256 S3 checkpoints

### 1. Verbatim Spoken Defense Script (`final_defense_appendix_slides.qmd`)

Slide A4 details the computational reproducibility of the dissertation. All ten analytical steps across our three specific aims are operationalized into seven sequential Jupyter notebooks in our public GitHub repository. The workflow uses S3-backed state checkpoints for idempotency and runs seamlessly on both local workstations and AWS EC2 spot instances. Every dataset transformation, hyperparameter search, and model artifact is fully versioned and traceable back to the raw APCD extracts.

### 2. Detailed Technical Walkthrough & Panel Notes

Appendix A4 lays out the computational pipeline, repository architecture, and reproducibility engineering supporting all three Specific Aims.
The architecture is divided into the sequenced computational steps on the left and our execution and reproducibility infrastructure on the right:
Jupyter Notebook Pipeline Directory (pgx-analysis): The left table tracks the end-to-end analytical workflow across modular notebooks:
Setup (0_config_and_pipeline.ipynb): Initializes environment configuration, synchronizes S3 data to NVMe, and establishes the DuckDB data dictionary.
Steps 1a, 1b, 2 (1_cohort_workflow.ipynb): Executes FAERS and APCD 21-day ADE window calibration, dual-cohort extraction, 1:5 incidence-density matching, and temporal guardrails.
Steps 3a, 3b, 3c (2_feature_importance.ipynb): Handles feature screening, the Step 3b target-leakage audit, and antecedent whitelisting.
Steps 4, 5, 6, 7, 8 (3_model_train_shap_ffa.ipynb): Manages claim-density stratification, CPIC drug mappings, best-model selection, TreeSHAP attributions, and PySAT formal feature attribution (FFA) rules.
Sensitivity Stage (3_model_sensitivity.ipynb): Evaluates utilization-free sensitivity and claims-density sub-cohort checks.
Step 9 (4_dashboard_visuals.ipynb): Generates visual representations of clinical pathways, Dynamic Time Warping (DTW) trajectory clusters, and FP-Growth itemsets.
Step 10 (5_build_and_deploy.ipynb): Compiles multi-stage Docker artifacts and deploys the serverless inference engine via AWS Lambda and API Gateway.
Execution Model, Checkpoints & Reproducibility: The right panel outlines the operational rigor guaranteeing reproducibility across stages:
S3 Checkpoint Idempotency: Each pipeline stage synchronizes inputs from s3://pgx-analysis-data/ to local NVMe, utilizing SHA-256 state markers to eliminate redundant compute on verified steps.
Dual Compute Environments: Combines interactive prototyping on a Windows 11 workstation (DuckDB 1.4+, Python 3.11/3.12) with scalable batch processing on Amazon Linux 2023 Fat-Spot EC2 instances (x2iedn.8xlarge; 15.1× DuckDB speedup, 0/189 OOMs, Slide 24).
Source Control & Auditing: Version-controlled and tracked under the monorepo at github.com/Jerome3590/pgx-analysis.
The bottom link navigates directly back into the primary dissertation portfolio.

### 3. Committee Q&A Delivery Cues & Pivots

- **Delivery Cue:** Show complete end-to-end lineage from raw Virginia APCD Parquet to Dockerized AWS Lambda container.
- **Dual Environment Parity:** Runs identically on Windows 11 local dev (DuckDB 1.4+, Python 3.11, R 4.3 `bupaR`) and Amazon Linux 2023 Spot EC2.
- **Navigation Return / Pivot:** *Click 'Return to Dissertation Portfolio (Slide 21)'.*

---

## Appendix Deck Slide 06 (Full Deck Slide 42): Appendix A5 — Specific Aims, Estimators & Evaluation Metrics (#sec-appendix-metrics)

- **Module & Context:** Appendix A5 — Formal Mathematical Specification & Predictive-vs.-Visual Wall
- **Visual Callout Anchors:** **Formal Estimators:** Burstiness B in [-1,1] • PR-AUC weighted ensemble w_m • Consensus |φ_j| ≥ Q_75 ∩ Supp ≥ 0.05 ∩ Conf ≥ 0.70 • IR_Δp & Δ_11 / IE > 1.0 • `DTW / FP-Growth / bupaR` excluded from X

### 1. Verbatim Spoken Defense Script (`final_defense_appendix_slides.qmd`)

Slide A5 provides a single-slide formal mathematical specification across Specific Aims 1 through 3, documenting the pre-specified study design, data transformation, and formal estimator for every pipeline metric—and explicitly distinguishing supervised predictive and explanatory estimators (TreeSHAP, AXP, IR, IE) from exploratory visual analytics (DTW, FP-Growth, bupaR). All pre-specified estimators and post-execution sensitivity discoveries are evaluated empirically on the untouched 2019 calendar holdout in Module VI (Slides 24 through 33).

### 2. Detailed Technical Walkthrough & Panel Notes

This slide returns us to Appendix A5: Specific Aims, Methodological Estimators & Evaluation Metrics, which serves as our formal methodological and mathematical specification and governance contract.
**Here is how each component breaks down for the committee:**
**Aim 1 Pipeline Stages (Left Column): This structures our cohort data engineering and modeling standards:**
Steps 1–2 (Matching & Day Guard): Formulates the 1:5 incidence-density matching window criteria and pre-index observation baselines.
Step 3b (Leakage Ratio τ_post): Enforces our target leakage exclusion threshold (τ_post ≥ 0.80), purging claims heavily skewed post-index.
Step 3c (Steady-State & Burstiness B): Defines temporal claim distribution, distinguishing acute utilization bursts from baseline chronic regimens.
Step 5 (CPIC Drug Mapping & Counts): Maps statewide APCD drug records directly to CPIC guidelines across level A/B actionability tiers.
Step 6 (MCCV Ensembles & Holdout): Specifies 25-fold Monte Carlo cross-validation, Brier score optimization, and integrated calibration index (ICI) limits.
Aims 2 & 3 Attribution & Clinical Architecture (Right Column): This locks the mathematical machinery translating raw ensemble outputs into actionable clinical logic:
Steps 7–8 (Consensus Gate ΔR): Sets the dual-model alignment criteria across our tree ensembles, requiring consensus before triggering alerts.
Step 8 (Intervention Effect IR_Δp̂ & Interaction Effect Δ_IE): Establishes formal estimators for individual counterfactual risk shifts and multi-drug interaction impacts.
Step 9 (Visual Attribution vs. Demarcation): Formulates the strict mathematical fence separating predictive features from post-hoc visual representations (TreeSHAP vs. PySAT FFA/AXP).
Step 10 (OODA Scenario Loss L_i): Formalizes the clinical penalty function balancing false positive alert fatigue against the risk of unflagged adverse events.
Empirical Closure Across Module VI (Bottom Callout): These formal mathematical estimators directly govern the untouched 2019 holdout evaluations, Two-Gate Mirror Ablation (-61% alert load, Slides 26 & 30), Table S5 utilization-free stress test (60% FFA rule retention vs. SHAP Jaccard = 0.00, Slides 28 & 30), and super-additive drug triplet / Z-code findings (Slide 29).
Interactive footer links provide direct navigation back to Aim 2 Consensus Attribution, the Methodological Fences, or the Appendix A6 Acronym Reference.

### 3. Committee Q&A Delivery Cues & Pivots

- **Delivery Cue:** Point out the strict firewall keeping exploratory visual analytics (`DTW`, `FP-Growth`, `bupaR`) out of the supervised predictor matrix `X`.
- **Why Exclude FP-Growth from IR?** Association rules measure marginal co-dispensing frequency, whereas `IR` perturbations measure in-model predicted risk attenuation holding covariates fixed.
- **Navigation Return / Pivot:** *Click 'Return to Consensus Attribution (Slide 18)' or 'Module VI Results (Slide 24)'.*

---

## Appendix Deck Slide 07 (Full Deck Slide 43): Appendix A6 — Master Acronym & Technical Nomenclature Reference (#sec-appendix-acronyms)

- **Module & Context:** Appendix A6 — Multidisciplinary Glossary Across Modules I–VII
- **Visual Callout Anchors:** **Cross-Domain Glossary:** Governance (`CRISP-DM, TRIPOD+AI, OODA, PROBAST, SaMD PCCP`) • Claims/Clinical (`APCD, HCG P51b/O11/P33, CPIC, Silent ADEs, MTM`) • ML/Logic (`GBDT, MCCV, TreeSHAP, FFA, CNF, AXP, IR, IE`)

### 1. Verbatim Spoken Defense Script (`final_defense_appendix_slides.qmd`)

Appendix A6 provides a comprehensive quick-reference glossary of all governance, clinical, biostatistical, machine learning, formal logic, and cloud engineering acronyms used across the seven scientific modules of the dissertation—including CRISP-DM, TRIPOD+AI, the Clinical OODA Loop, PROBAST, Virginia APCD, Milliman Health Cost Guidelines (`HCG`: `hcg_setting`, `hcg_line`, `hcg_detail` emergency acuity codes `P51b`, `O11`, `P33`), CPIC pharmacogenomics, Silent ADEs, GBDT tree geometries, TreeSHAP, Formal Feature Attribution (FFA), Conjunctive Normal Form (CNF), Abductive Explanations (AXPs), Intervention Rate (IR), Interaction Effect (IE), and our zero-PHI Medication Therapy Management (MTM) Clinical Decision Support (CDS) prototype.

### 2. Detailed Technical Walkthrough & Panel Notes

What we have here in Appendix A6 is our master technical reference and acronym crosswalk, built to anchor the terminology across the entire dissertation portfolio.
To keep the methodology rigorous and reproducible, the slide organizes our nomenclature across four operational domains:
Governance and Oversight: At the top left, we outline the foundational frameworks that structure our research lifecycle—governing the end-to-end engineering via CRISP-DM, enforcing clinical AI validation standards using TRIPOD+AI, and embedding the clinical OODA loop to maintain essential human-in-the-loop oversight. Alongside these are our systematic evaluation protocols, including PROBAST.
Data Infrastructure and Clinical Cohorts: Moving down the left side, we define the data ingestion and domain taxonomy. This links the Virginia APCD longitudinal claims records with Milliman Health Cost Guidelines (HCG Grouper: P51b, O11, P33) encounter acuity categories, integrates CPIC guidelines for pharmacogenomics, and structures surveillance for Adverse Drug Events (ADEs), Drug-Drug Interactions (DDIs), and longitudinal Opioid Use Disorder (OUD) pathways.
Machine Learning and Formal Explainability: On the right column, we move into the methodological engine. This spans our Gradient-Boosted Decision Tree (GBDT) ensembles, our cross-validation and hyperparameter tuning protocols (MCCV and TPE), and discrimination metrics like AUROC, PR-AUC, and ICI. For interpretability, it details the dual bridge between heuristic attribution—using TreeSHAP—and exact symbolic verification, utilizing PySAT for Formal Feature Attribution (FFA), Abductive Explanations (AXP), and Boolean Horn logic clauses.
Architecture and Clinical Delivery: Finally, at the bottom right, we map how these models translate into practice: deploying lightweight, zero-PHI serverless microservices for Clinical Decision Support (CDS) and Medication Therapy Management (MTM).
The interactive navigation links along the bottom allow us to jump straight back into the governance foundations, the polypharmacy cohort pipeline, or the core methodological fences as needed during our discussion.

### 3. Committee Q&A Delivery Cues & Pivots

- **Delivery Cue:** Keep available as a shared vocabulary bridge across pharmacy, biostatistics, and systems engineering.
- **Milliman HCG Codes:** `P51b` (ED Visits), `O11` (Observation Care), and `P33` (Urgent Care) capture acute unscheduled outpatient crises uniformly across commercial, Medicaid, and Medicare Advantage plans.
- **Navigation Return / Pivot:** *Return to active discussion slide via Reveal.js hash navigation.*

---

## Appendix Deck Slide 08 (Full Deck Slide 44): Appendix A7 — Regulatory Scope Demarcation & HIPAA Cloud Architecture (#sec-appendix-hipaa)

- **Module & Context:** Appendix A7 — FTC Consumer Self-Use vs. Hospital Two-Tier HIPAA BAA & CCTR Aim 2
- **Visual Callout Anchors:** **Statutory Boundary:** Consumer self-use = **FTC Act § 5 / HBNR** (no BAA) vs. Hospital ePHI (`45 CFR § 160.103`) = **Two-Tier BAA Chain** (#1 Hospital↔Entity + #2 Entity↔AWS Artifact) + PrivateLink VPC, KMS, `/tmp`=0 • CCTR Aim 2 ($4.5K: VCU InfoSec, Massey & OVPRI)

### 1. Verbatim Spoken Defense Script (`final_defense_appendix_slides.qmd`)

Slide A7 addresses the regulatory, legal, and cloud-security architecture formalized in Chapter 6.  
First, we enforce a strict statutory demarcation: when an individual patient enters their own medication list or uploads consumer genetic files for personal health literacy, 45 CFR Section 160.103 dictates that HIPAA does not apply—that relationship falls under Federal Trade Commission Act Section 5 and the Health Breach Notification Rule, where a HIPAA BAA is neither required nor legally recognized.  
Second, when deployed by a healthcare provider or hospital processing ePHI, compliance requires an unbroken two-tier legal BAA execution chain: the hospital signs BAA number 1 with our legal entity, and our legal entity executes BAA number 2 with AWS via AWS Artifact.  
Third, under the AWS Shared Responsibility Model, we harden HIPAA-eligible primitives into a compliant enclave via KMS encryption, Firecracker microVM zero `/tmp` persistence inside private VPC and PrivateLink subnets, strict CloudWatch log scrubbing, and multi-region CloudTrail auditing—while requiring separate signed BAAs for any external third-party AI APIs outside AWS Bedrock or SageMaker.  
Finally, Aim 2 of our VCU Wright Center CCTR 25,000-dollar Pilot Grant allocates 4,500 dollars to partner with VCU Health Information Security, Massey Cancer Center Informatics Governance, and OVPRI Legal Counsel to vet and execute all institutional BAA and DUA templates before clinical pilot testing.

### 2. Detailed Technical Walkthrough & Panel Notes

Appendix A7 addresses the critical governance, regulatory, and legal questions regarding our clinical decision-support architecture, establishing our two-tier compliance framework.

**Regulatory Scope Demarcation (Direct Patient vs. Hospital Deployment):**
1. Direct Patient Consumer Use (FTC Act § 5): When an individual patient directly enters their own personal health data or uploads consumer genetic files for personal use, federal statute dictates that HIPAA does not govern that relationship. It falls under Federal Trade Commission Section 5 and the FTC Health Breach Notification Rule, meaning a BAA is neither required nor legally recognized.
2. Enterprise Hospital Deployment (HIPAA 45 CFR § 160.103): When deployed within a healthcare provider or hospital system processing ePHI, compliance is governed by an executed two-tier legal BAA chain: the covered entity signs BAA #1 with our legal entity, and our entity executes BAA #2 directly with AWS through AWS Artifact covering the underlying cloud infrastructure.

**AWS HIPAA-Eligible Technical Safeguard Architecture:**
3. Mandatory Envelope Encryption: Enforces KMS envelope encryption (AES-256) on Amazon S3 with automated key rotation and TLS 1.3 in-transit encryption.
4. Stateless Ephemeral Compute: Operates on AWS Lambda using Firecracker microVMs with strict in-memory execution and zero persistent ePHI storage in /tmp.
5. CloudWatch Log Scrubbing: Custom log filters scrub all query strings, headers, and payloads to prevent ePHI from entering cloud telemetry.
6. Comprehensive Audit Logging: AWS CloudTrail records all management events and API calls under 45 CFR § 164.312(b).
7. External AI API Isolation: Strict protocol boundary; external AI APIs outside of AWS Bedrock are barred from processing ePHI without independent BAAs.

**Institutional Review & CCTR Grant Funding:**
8. VCU CCTR Aim 2 Governance ($4,500 Dedicated Allocation): Formally engages (a) VCU Health Information Security & Compliance, (b) Massey Comprehensive Cancer Center Informatics Governance, and (c) VCU OVPRI Legal Counsel to execute our Two-Tier AWS Artifact HIPAA BAA and PrivateLink VPC enclave prior to N ≥ 200 clinical encounters in Epic Willow Ambulatory and Compass Rose.

### 3. Committee Q&A Delivery Cues & Pivots

- **Delivery Cue:** Deliver the 4-part legal & technical architecture crisply—highlighting FTC vs. Two-Tier BAA and the 3 VCU CCTR Aim 2 governance bodies.
- **VCU CCTR Aim 2 ($4,500 Governance Gate):** Engages (1) VCU Health InfoSec & Compliance, (2) Massey Informatics Governance, and (3) VCU OVPRI Legal Counsel prior to `N >= 200` clinical encounters.
- **Navigation Return / Pivot:** *Click 'Return to Aim 3 Decision Support (Slide 20)' or 'Study Boundaries (Slide 22)'.*

---

## Appendix Deck Slide 09 (Full Deck Slide 45): Appendix A8 — PGx Gene Markers & Mechanistic Networks by Cohort (#sec-appendix-gene-markers)

- **Module & Context:** Appendix A8 — PharmGKB VIP & CPIC Level A PK/PD Pathways
- **Visual Callout Anchors:** **Opioid PK/PD:** `CYP2D6` (codeine→morphine, tramadol→M1; UM respiratory depression vs. PM analgesic failure) + `OPRM1 118G` + `CYP3A4/UGT2B7` • **Polypharmacy:** `CYP2C19*2/*3` (clopidogrel MACE), `CYP2C9 + VKORC1 -1639A` (warfarin bleed), `SLCO1B1*5` (statin myopathy)

### 1. Verbatim Spoken Defense Script (`final_defense_appendix_slides.qmd`)

Slide A8 provides the deep mechanistic and biological defense for the specific pharmacogenomic gene markers identified across our two clinical cohorts, directly synthesizing PharmGKB Very Important Pharmacogene (VIP) pathways and CPIC Level A guidelines.
In the Opioid Cohort, we focus on the pharmacokinetic anchor CYP2D6 in hepatic microsomes and the pharmacodynamic anchor OPRM1 in the central nervous system. CYP2D6 controls the rate-limiting phase I bioactivation of prodrug opioids—specifically O-demethylating codeine to morphine and tramadol to O-desmethyltramadol, which possesses over one-hundred-fold higher affinity for the mu-opioid receptor. Downstream, OPRM1 governs the Gi/Go protein-coupled receptor cascade across brainstem respiratory pacemakers like the pre-Bötzinger complex and mesolimbic reward pathways. The clinical phenotype is bidirectionally lethal: Ultra-rapid Metabolizers convert prodrugs at accelerated rates, causing a sudden toxic surge in active mu-agonists and fatal respiratory depression at standard doses; conversely, Poor Metabolizers carrying null alleles or patients with the OPRM1 118G variant experience profound analgesic failure, driving dose escalation, accidental toxicity, and ED encounters. Supporting shunt enzymes like CYP3A4 and glucuronidation clearance via UGT2B7 modulate systemic exposure.
In the Polypharmacy Cohort, adverse drug events in older adults are driven by a multi-target network across hepatic clearance, transport, and coagulation. First, CYP2C19 loss-of-function star alleles (*2 and *3) impair the bioactivation of clopidogrel into its active thiol metabolite, failing to inhibit platelet P2RY12 and precipitating stent thrombosis and major adverse cardiovascular events. Second, S-warfarin clearance is governed stereoselectively by hepatic CYP2C9, while its therapeutic target is VKORC1 in the endoplasmic reticulum; patients combining CYP2C9 reduced-function variants with the VKORC1 -1639A low-expression promoter variant experience severe clearance failure and extreme target sensitivity, driving supratherapeutic INR and catastrophic hemorrhage. Finally, SLCO1B1 encodes the OATP1B1 sinusoidal influx transporter; the star 5 variant halts hepatic statin uptake, trapping statins in systemic circulation where they reach toxic skeletal muscle concentrations, triggering severe statin-induced myopathy and rhabdomyolysis.

### 2. Detailed Technical Walkthrough & Panel Notes

Appendix A8 provides the deep mechanistic and biological defense for the specific pharmacogenomic gene markers identified across our two clinical cohorts, directly synthesizing PharmGKB Very Important Pharmacogene (VIP) pathways and CPIC Level A guidelines.

**Opioid Cohort: Pharmacokinetic Anchor (CYP2D6) & Pharmacodynamic Anchor (OPRM1):**
1. CYP2D6 Hepatic Bioactivation: Catalyzes the rate-limiting phase I bioactivation of prodrug opioids—specifically O-demethylating codeine to morphine and tramadol to O-desmethyltramadol, which possesses >100-fold higher affinity for the mu-opioid receptor.
2. OPRM1 CNS Signal Transduction: Governs the Gi/Go protein-coupled receptor cascade across brainstem respiratory pacemakers (pre-Bötzinger complex) and mesolimbic reward pathways.
3. Bidirectional Lethal Phenotypes: Ultra-rapid Metabolizers convert prodrugs at accelerated rates, causing a sudden toxic surge in active mu-agonists and fatal respiratory depression at standard doses; Poor Metabolizers carrying null alleles or patients with the OPRM1 118G variant experience profound analgesic failure, driving dose escalation, accidental toxicity, and ED encounters.
4. Supporting Enzymes: Shunt enzymes like CYP3A4 and glucuronidation clearance via UGT2B7 modulate systemic opioid exposure.

**Polypharmacy Cohort: Multi-Target Hepatic, Transport & Coagulation Network:**
5. CYP2C19 Bioactivation Failure: Loss-of-function star alleles (*2, *3) impair bioactivation of clopidogrel into its active thiol metabolite, failing to inhibit platelet P2RY12 and precipitating stent thrombosis and major adverse cardiovascular events.
6. CYP2C9 / VKORC1 Hemorrhagic Synergy: S-warfarin clearance is governed stereoselectively by hepatic CYP2C9, while its therapeutic target is VKORC1 in the endoplasmic reticulum; patients combining CYP2C9 reduced-function variants with the VKORC1 -1639A low-expression promoter variant experience severe clearance failure and extreme target sensitivity, driving supratherapeutic INR and catastrophic hemorrhage.
7. SLCO1B1 Sinusoidal Influx Failure: Encodes the OATP1B1 sinusoidal influx transporter; the *5 variant halts hepatic statin uptake, trapping statins in systemic circulation where they reach toxic skeletal muscle concentrations, triggering severe statin-induced myopathy and rhabdomyolysis.

### 3. Committee Q&A Delivery Cues & Pivots

- **Delivery Cue:** Walk cohort-by-cohort: bidirectional lethality in Opioid (`CYP2D6` UM vs. PM + `OPRM1`), then hepatic/transporter cascade in Geriatric Polypharmacy.
- **Direct Tie to Empirical Ch. 4 Top IR Drivers:** Notice `simvastatin` was our #1 `IR` deprescribing target on Slide 29—directly governed by `SLCO1B1*5` (`c.521T>C`) OATP1B1 hepatic uptake impairment!
- **Navigation Return / Pivot:** *Click 'Return to CPIC FI (Slide 9)' or 'Appendix A3 (Slide 40)'.*

---
