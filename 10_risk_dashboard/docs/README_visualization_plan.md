# PGx Risk Dashboard: Final Production Visualization Plan

**Single Source of Truth** for Dashboard Visualizations, Production Workflow, Research Questions, and Clinical Governance.

- **Reference Implementation:** [PGx Risk Calculator Live Deployment](https://jerome-dixon.io.s3.us-east-1.amazonaws.com/vcu/pgx-risk-calculator/index.html)
- **Clinical Governance & Phasing Architecture:** [docs/README_pgx_card.md](README_pgx_card.md)
- **Artifact Allowlist & Paths:** [docs/RESEARCH_QUESTIONS_ARTIFACTS.md](RESEARCH_QUESTIONS_ARTIFACTS.md)
- **Archived Outputs:** [docs/ARCHIVED_ARTIFACTS_NO_LONGER_USED.md](ARCHIVED_ARTIFACTS_NO_LONGER_USED.md)

---

## 1. Executive Summary & Design Principles

The PGx Risk Dashboard provides an end-to-end clinical and translational research platform bridging population-level health claims analytics with individual-level pharmacogenomic precision medicine.

### Core Principles
1. **Decoupled Feature Engineering & Visualization**: We do **not** use process mining (BupaR), Dynamic Time Warping (DTW), or association rule mining (FP-Growth) for predictive model feature engineering to prevent target leakage. We **do** use them with model-important features (SHAP/FFA allowed codes) for **causal discovery, temporal sequence exploration, and dashboard visualization**.
2. **Full-Dataset, Filter-to-Features**: Heavy pipeline transformations execute upstream on EC2 / Batch. The pipeline exports standardized JSON and precomputed visual assets; Lambda and the browser only filter and render.
3. **Artifact Economy**: We produce and retain **only** artifacts tied directly to validated research questions (**N1–N6, PGx1, PGx2**). All unmapped artifacts are archived.
4. **JSON-First Visualization**: Visuals prioritize structured JSON data payloads, rendering natively in the browser via Plotly.js and Chart.js. High-complexity network topologies (Cytoscape, Pyvis) are rendered in isolated sandboxed iframes.
5. **Bi-Directional PGx Architecture**: Pharmacogenomics is split into two complementary layers:
   - **Macro-Scale (PGx Cohort)**: Population gene–drug–phenotype network topology, PubMed literature citations, and claims actionability radar.
   - **Micro-Scale (PGx Card)**: Individual patient decision support featuring an Options Matrix, Gene–Drug Actionability Matrix with CPIC filter pills, native Plotly Clustered Dendrogram Heatmaps, and population figure pack integration.
6. **Strict Phasing & Consumer DNA Safety**: Direct-to-consumer genealogy data (AncestryDNA, 23andMe, MyHeritage, unphased VCF) is locked to exploratory status (`INSUFFICIENT_GENOTYPE_RESOLUTION`). The pipeline **never defaults unprobed sites to `*1`** and hard-locks automated prescribing changes.

---

## 2. Complete Dashboard Tab Layout (13 Tabs)

The dashboard organizes 13 tabs into two responsive rows: **Primary Clinical Workflow** (Row 1) and **Secondary Research Visualizations** (Row 2).

```
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ PRIMARY WORKFLOW: [User Guide] [Risk Assessment] [Drugs] [ICD Codes] [CPT Codes] [PGx Card]            │
├────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ VISUALIZATIONS:   [Feature Importance] [Scenario Analysis] [BupaR] [DTW] [FP-Growth] [Drug Networks]   │
│                   [PGx Cohort]                                                                         │
└────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

### Tab Registry & Specifications

| # | Tab Identifier | Name | Category | Primary Focus | Key Visuals & Formats |
|---|----------------|------|----------|---------------|-----------------------|
| 1 | `documentation` | **User Guide** | Governance | Clinical onboarding, SOPs, training | Architecture tables, training handoffs, unphased DNA warnings |
| 2 | `risk-assessment` | **Risk Assessment** | Clinical Decision | CatBoost/XGBoost ensemble predictions | Risk gauge (0–1), risk band badge, model agreement, scenario replacement |
| 3 | `drugs` | **Drugs** | Cohort Selection | APCD generic medications filter | Multi-select dropdown, removable active filter chips |
| 4 | `icd-codes` | **ICD Codes** | Cohort Selection | Diagnostic ICD-10 codes filter | Multi-select dropdown, removable active filter chips |
| 5 | `cpt-codes` | **CPT Codes** | Cohort Selection | Procedure CPT/HCPCS codes filter | Multi-select dropdown, removable active filter chips |
| 6 | `pgx-card` | **PGx Card** | Precision Medicine | Individual patient prescribing decision support | 3-Panel Options Matrix, Gene–Drug Actionability Matrix, Plotly Clustered Dendrogram Heatmaps, Population Figure Pack |
| 7 | `feature-importance-visualizations` | **Feature Importance** | Population Research | Population-level outcome drivers (N5) | Age-band SHAP/FFA importance heatmaps (`aggregated_fi_heatmap.json`/PNG) |
| 8 | `scenario-analysis` | **Scenario Analysis (FFA/SHAP)** | Causal Discovery | Feature interactions & what-if exploration (N5, N6) | FFA interaction bars, SHAP importance bars, polar outcome radar chart |
| 9 | `bupar-visualizations` | **BupaR Process Mining** | Temporal Discovery | Longitudinal event sequencing to target (N2, N6) | Activity frequency (pre/post target), aggregated trace explorer, Drug × Drug process matrix |
| 10 | `dtw-visualizations` | **DTW Trajectories** | Temporal Trajectory | Routine care vs. medical utilization (N1, N3) | Routine vs Utilization bar chart, high-risk trajectory quartiles, aligned time-between charts |
| 11 | `fpgrowth-visualizations` | **FP-Growth Patterns** | Pattern Mining | Multi-drug risk-predictive co-occurrence (N4) | Drug-only frequent itemset support distributions, co-occurrence network |
| 12 | `cytoscape-visualizations` | **Drug Networks** | Network Topology | FI-gated rule association graphs (N4) | Cytoscape.js interactive network topology HTML iframe |
| 13 | `cohort-pgx-visualizations` | **PGx Cohort** | Pharmacogenomics | Population-scale gene–drug network topology (PGx2) | Pyvis/NetworkX graph iframe, PubMed literature citations, Gene Actionability Radar |

---

## 3. Research Questions → Visuals Mapping

Each visual artifact produced and displayed is strictly aligned with a research question and evaluated through the **Clinical OODA Loop**:

```
           ┌──────────┐         ┌──────────┐
           │ Observe  │ ──────> │  Orient  │
           └──────────┘         └──────────┘
                ▲                     │
                │                     ▼
           ┌──────────┐         ┌──────────┐
           │   Act    │ <────── │  Decide  │
           └──────────┘         └──────────┘
```

| ID | Research Question | Target Tab(s) | Production Visual Artifacts | Clinical OODA Mapping |
|----|-------------------|---------------|-----------------------------|-----------------------|
| **N1** | Routine vs. utilization appointments → outcomes? (How do routine screenings reduce extreme cohorts?) | **DTW Trajectories** | `routine_comparison`, `routine_comparison_counts`, `routine_by_medical_utilization`, high-risk trajectory quartiles | **Observe & Orient:** Reveals that high routine screening density significantly blunts adverse acute events. |
| **N2** | What sequences lead to target outcomes? | **BupaR Process Mining** | Sequences to target (`*_activity_sequence_top.png`), pre-target activity frequencies (`*_pre_target_activity_frequency.json`), aggregated trace explorer | **Orient:** Identifies recurring clinical escalations preceding opioid overdose or polypharmacy crisis. |
| **N3** | What times between sequences lead to target outcomes? | **DTW Trajectories**, **BupaR** | DTW aligned sequence time-between (`times_between_sequences`), time-to-target (`time_to_target_sequences`) | **Orient & Decide:** Measures the acceleration velocity of visits. Alignment makes inter-visit intervals comparable across heterogeneous patients. |
| **N4** | Drug connections → target? (Risk-predictive co-occurrence) | **FP-Growth Patterns**, **Drug Networks** | Top drug itemsets (`.../data/drug_name_itemsets.json`), combined rules network (`*_combined_rules_network.html`), Cytoscape HTML | **Orient & Decide:** Highlights multi-drug prescribing cliques (e.g., opioid + benzodiazepine + muscle relaxant) driving acute risk. |
| **N5** | What features drive outcome and how do they relate? | **Scenario Analysis**, **Feature Importance** | `causal_data.json` (FFA interaction factors, SHAP importance, effect radar), `aggregated_fi_heatmap.json` | **Observe & Orient:** Transparently discloses non-linear model drivers and interaction strengths across age bands. |
| **N6** | What drug combinations drive polypharmacy ED? | **Scenario Analysis**, **BupaR** | Drug-focused causal interaction factors, Drug × Drug process transition matrix (`*_process_matrix_drug_drug.json`) | **Decide:** Points directly to specific medication pairs that precipitate emergency admissions. |
| **PGx1** | How do individual genetic variants guide precise prescribing and manage multi-gene complexity? | **PGx Card** | Gene–Drug Actionability Matrix, Plotly Clustered Dendrogram Heatmaps (Drug × Gene, Drug Profile, Gene Profile) | **Decide & Act:** Directs personalized dose titration, drug substitution, or clinical laboratory testing orders. |
| **PGx2** | What is the population pharmacogenomic landscape and evidence network for high-risk cohorts? | **PGx Cohort** | Cohort Gene Actionability Radar, Pyvis network topology, PubMed literature citations, Publication Figure Pack | **Orient:** Establishes guideline strength, VIP evidence tiers, and recent peer-reviewed citations for target genes. |

---

## 4. In-Depth Visualization Architecture

### A. PGx Card: Precision Prescribing Decision Support
The PGx Card serves as the primary actionable decision interface for pharmacogenomics:

1. **3-Panel Options Matrix (`.pgx-options-matrix`)**:
   - **⚡ Run & Generate**: Cohort Claims Profile loader, genetic variant/file submission, and sample profile injection.
   - **🩺 Clinical Reports**: Formulate CPIC guidance, generate physician summaries, and launch clinical test referrals.
   - **📑 Export Matrix**: High-resolution print, JSON clinical payload, CSV data export, and technical appendix downloads.
2. **Gene–Drug Actionability Matrix (`#pgx-action-matrix`)**:
   - Color-coded action grid: Avoid/Alternative Needed (`#b91c1c`), Dose Adjustment Needed (`#ea580c`), Enhanced Monitoring (`#d97706`), Standard Prescribing (`#10b981`), and Insufficient Genetic Resolution (`#7c3aed` cross-hatch).
   - Dynamic CPIC filter pills (`ALL`, `AVOID`, `DOSE`, `MONITOR`, `STD`, `INDET`) for instantaneous medication triage.
   - Interactive cell click inspection cards providing diplotype, phenotype, clinical rationale, and direct CPIC guideline URLs.
3. **Plotly Clustered Dendrogram Heatmaps (`#pgx-dendrogram-section`)**:
   - Native agglomerative hierarchical clustering (`hierarchicalCluster`) with Euclidean/Manhattan distance and UPGMA/Complete/Single linkage.
   - **Mode A: Drug × Gene Clustering**: Dual dendrogram layout aligning row dendrograms (drugs) and column dendrograms (genes) with heatmaps via numeric coordinate mapping (`tickmode: "array"`).
   - **Mode B: Drug Action Profile**: Clusters medications by recommendation severity distribution.
   - **Mode C: Gene Action Profile**: Clusters pharmacogenes by clinical actionability spread.
4. **Aggregated Population Visuals & Figure Pack**:
   - Pre-computed publication figures selectable via dropdown (`pgx_global_intervention_network`, `pgx_cohort_small_multiples`, `pgx_cluster_ego_networks`, `pgx_intervention_priority_heatmap`, etc.).
   - Embedded Cohort Gene Actionability Radar chart displaying multi-dimensional evidence scores (CPIC, VIP tier, literature count, model importance).

### B. PGx Cohort: Population Topology & Evidence Base
1. **Interactive Network Topology**:
   - 100+ node interactive graph linking cohort drugs, metabolic enzymes (CYP450s, transporters), and adverse clinical phenotypes.
   - Isolated iframe container (`#cohort-pgx-iframe`) preventing DOM degradation.
2. **NCBI PubMed Automated Literature Integration**:
   - Queries PubMed E-utilities for recent peer-reviewed citations tied to cohort-specific pharmacogenes.
   - Collapsible citation cards with direct PMID deep-links.

### C. Scenario Analysis (FFA/SHAP)
1. **Top Interaction Factors (FFA)**: Multi-trace bar charts comparing baseline feature importance against user what-if scenarios.
2. **SHAP Feature Importance**: Quantifies marginal contribution to the ensemble probability.
3. **Polar Outcome Radar Chart (`#scenario-radar-chart`)**: Multi-axial radar displaying normalized feature impact vectors for top 5–8 drivers.

### D. Temporal & Process Mining (BupaR & DTW)
1. **BupaR Activity Frequency**: Interactive Chart.js/Plotly bar charts reporting overall, pre-target, and post-target code frequencies across longitudinal patient timelines.
2. **DTW Routine vs. Utilization**: Grouped bar charts demonstrating adverse outcome probabilities stratified by administrative screening counts and overall health system utilization.
3. **Aligned Sequence Interval Metrics**: Boxplots and bar charts quantifying inter-event duration for patients aligned to clinical archetype trajectories.

---

## 5. Genetic Phasing & Consumer Genealogy Safety Architecture

Direct-to-consumer arrays (23andMe, AncestryDNA, MyHeritage) lack physical chromosome phasing and copy-number measurements. The dashboard implements five clinical guardrails:

```
[Consumer Array Upload] ──> [Zero Reference Imputation] ──> [Phase Collision Check]
                                                                     │
                                      ┌──────────────────────────────┴──────────────────────────────┐
                                      ▼                                                             ▼
                         [Multiple Heterozygous Hits]                                   [Single Detected SNP]
                                      │                                                             │
                                      ▼                                                             ▼
                       "Phasing required for diplotype"                            "CNV/Duplication not measured"
                                      │                                                             │
                                      └──────────────────────────────┬──────────────────────────────┘
                                                                     ▼
                                                   [INSUFFICIENT_GENOTYPE_RESOLUTION]
                                                                     │
                                      ┌──────────────────────────────┴──────────────────────────────┐
                                      ▼                                                             ▼
                         [Purple Cross-Hatch Matrix Cell]                             [Hard Prescribing Lockout]
                         [Exploratory Warning Banner]                                 [CLIA/CAP Lab Test Nudge]
```

1. **Strict Category Assignment**: Loci with unphased ambiguity map to `INSUFFICIENT_GENOTYPE_RESOLUTION`.
2. **Zero Reference Allele Imputation**: Unprobed loci are marked `Data Not Present in File`. The pipeline **never defaults unmeasured sites to `*1`**.
3. **Phase Collision Flag**: The backend engine ([`cpic_allele_resolver.py`](file:///c:/Projects/pgx-analysis/10_risk_dashboard/backend/cpic_allele_resolver.py)) flags genes with $\ge 2$ variant hits:  
   *`"Multiple {gene} variants found. Phasing required for diplotype call."`*
4. **Copy Number Disclaimers**: Explicitly states `cnvMeasured: False` and adds warnings for structurally variable genes like `CYP2D6`.
5. **Prescribing Lockout & Clinical Referral**: Blocks automated dosing changes and prompts ordering a CLIA/CAP-certified confirmatory laboratory panel.

---

## 6. Implementation & Automated Test Verification Matrix

All visualization components are integrated into the live codebase and validated by automated end-to-end test suites:

| Component | Target File | Verification Test Suite | Status |
|-----------|-------------|-------------------------|--------|
| **Options Matrix Grid** | `tabs/pgx-card.html` | `tab-pgx-card.test.js`, `recapture_uc07.js` | ✅ Fully Implemented & Verified |
| **Actionability Matrix** | `tabs/pgx-card.html`, `pgx-workflow.js` | `tab-pgx-card.test.js` | ✅ Fully Implemented & Verified |
| **Clustered Dendrograms** | `pgx-workflow.js`, `index.html` | Visual screenshots (`pgx_dendrogram_*_verified.png`) | ✅ Fully Implemented & Verified |
| **Figure Pack & Radar** | `tabs/pgx-card.html`, `index.html` | Screenshot (`pgx_aggregated_visuals_verified.png`) | ✅ Fully Implemented & Verified |
| **Scenario Polar Radar** | `tabs/scenario-analysis.html`, `index.html` | `recapture_uc01_to_uc06.js` | ✅ Fully Implemented & Verified |
| **Unphased Allele Matcher** | `backend/cpic_allele_resolver.py` | `test_cpic_allele_resolver.py` (12/12 passing) | ✅ Fully Implemented & Verified |
| **Standalone Documentation** | `docs/README_pgx_card.md` | Markdown link validation | ✅ Fully Implemented & Verified |

---

## 7. Production Workflow & Deployment Commands

```bash
# 1. Synchronize model assets and feature importance from S3
python 9_dashboard_visuals/sync_visualization_data_from_s3.py --no-sync

# 2. Generate BupaR, DTW, and FP-Growth visualizations
python 9_dashboard_visuals/run_dashboard_visuals.py --cohort opioid_ed --age-band 25-44

# 3. Execute backend unphased resolver unit test suite
pytest 11_testing/tests/test_cpic_allele_resolver.py

# 4. Run Puppeteer automated browser test suite
npm test -- 11_testing/puppeteer/tests/tabs/tab-pgx-card.test.js
```
