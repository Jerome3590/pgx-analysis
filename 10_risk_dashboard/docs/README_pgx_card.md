# PGx Card: Personalized Pharmacogenomic Clinical Decision Support

## Overview

The **PGx Card** tab transforms patient-specific genetic variant data and Virginia All-Payer Claims Database (APCD) medication regimens into point-of-care clinical prescribing actions aligned with official **Clinical Pharmacogenetics Implementation Consortium (CPIC®)** guidelines and **PharmGKB** evidence.

While the **PGx Cohort** tab serves epidemiological and population-scale network exploration, the **PGx Card** focuses squarely on the individual patient: cross-referencing prescribed or candidate medications against identified metabolizer phenotypes to alert clinicians to adverse drug reactions, therapeutic failures, and recommended dosing adjustments.

---

## Architecture & Two-Phase Workflow

```mermaid
flowchart TD
    subgraph PHASE1["Phase 1: Population Baseline & Identified Genes"]
        A["Select Cohort\n(Opioid ED / Polypharmacy)"] --> B["Select Age Band & Event Density"]
        B --> C["Load Cohort PGx Profile"]
        C --> D["Identified Cohort Genes\n(COMT, CYP2D6, OPRM1, etc.)"]
        C --> E["Cohort Actionability Radar\n(Literature, VIP Tier, Causal Signal)"]
    end

    subgraph PHASE2["Phase 2: Individual Variant Resolution & Card Generation"]
        F["Input Genotypes\n(Lab Call, rsID, 23andMe, VCF, Excel)"] --> G["Browser Privacy Guard\n(Raw DNA stays local/ephemeral)"]
        G --> H["POST /pgx/card\n(DuckDB / Parquet allele resolution)"]
        H --> I["Diplotype & Phenotype Calling\n(Official CPIC Diplotype-Phenotype Tables)"]
        I --> J["Cross-reference APCD Generic Medications"]
        J --> K["Generate Personalized PGx Card"]
    end

    PHASE1 -.-> PHASE2
    K --> L["⊞ Actionability Matrix"]
    K --> M["🌳 Clustered Dendrogram Heatmaps"]
    K --> N["📊 Aggregated Figure Pack"]
    K --> O["📑 Multi-Format Clinical Exports"]
```

### Phase 1: Population Baseline
1. **Cohort & Density Filter**: The clinician selects the target cohort (Opioid ED or Polypharmacy), patient age band, and claims event density bin.
2. **Identified Genes**: Sourced from claims feature importance (SHAP/FFA) and linked to PubMed citation frequency and PharmGKB Very Important Pharmacogene (VIP) tiers.
3. **Cohort Radar Baseline**: Pre-renders the multi-dimensional OODA evidence radar chart.

### Phase 2: Patient Genotype Translation
1. **Multi-Format Input**:
   - **Official Lab Genotypes**: Format `Gene,*allele1,*allele2` (e.g., `CYP2D6,*1,*2`, `CYP2C19,*1,*17`, `SLCO1B1,*5,*1`).
   - **Single Nucleotide Polymorphisms (rsIDs)**: Direct rsID entries (e.g., `rs1045642,TT`).
   - **Consumer Array Uploads**: AncestryDNA, 23andMe, MyHeritage text files, unphased VCFs, and Excel spreadsheets.
2. **Phenotype Translation**: Server-side CPIC reference tables translate diplotypes into standardized metabolizer categories (*Poor Metabolizer*, *Intermediate Metabolizer*, *Normal Metabolizer*, *Rapid Metabolizer*, *Ultrarapid Metabolizer*).
3. **APCD Formulary Cross-Reference**: Evaluates every approved APCD generic medication against the resolved diplotypes and assigns official CPIC recommendation levels (Level A, B, C, D).

---

## Prescribing Visualizations & Layouts

### 1. Options Matrix (`.pgx-options-matrix`)
The header controls are structured as a 3-panel matrix rather than an unorganized linear button row:

| Panel | Type | Description |
|---|---|---|
| **⚡ Run & Generate** | Primary Action | Gradient hero button (`#btnGenerateCard`) to execute analysis, plus session reset (`#btnResetCard`). |
| **🩺 Clinical Reports** | Structured Handoffs | Direct generation of **Doctor Summary**, **Pharmacy Handoff**, and **Technical Appendix**. |
| **📑 Export Matrix** | File & Data Exports | 2-column grid for **Print**, **PDF**, **PNG**, **CSV**, **JSON**, and **Copy to Clipboard**. |

### 2. Gene–Drug Actionability Matrix (`#pgx-action-matrix`)
The primary prescribing view cross-references approved generic medications against patient genes in a structured grid:
- **Format Switcher**: Segmented toggle allows instant switching between:
  - **⊞ Matrix Format**: High-density 2D cross-reference grid (default).
  - **☰ Row Format**: Card-by-card queue sorted by action priority.
- **CPIC Action Filter Pills**:
  - `All`: Total scoped medications.
  - `Avoid / Alt`: High-risk contraindicated medications requiring alternative therapy.
  - `Dose Adjust`: Dosage adjustments or titration requirements.
  - `Monitoring`: Enhanced therapeutic drug or toxicity monitoring.
  - `Standard`: Standard prescribing per clinical guidelines.
  - `Insufficient`: Genes with incomplete or unphased consumer array coverage.
- **On-Click Detail Inspection**: Clicking any cell (`.pgx-interactive-cell`) triggers the on-screen detail card (`#pgx-matrix-detail-card`) showing full Diplotype, Phenotype, clinical text, and CPIC reference links.

### 3. Clustered Dendrogram Heatmaps (`#pgx-dendrogram-plot`)
Agglomerative hierarchical clustering groups medications and genes based on functional prescribing similarity:
- **Distance Metrics**: *Euclidean* or *Manhattan*.
- **Linkage Criteria**: *Average (UPGMA)*, *Complete*, or *Single*.
- **Three Clustering Modes**:
  1. **Drug, Gene & Recommendation (Dual Dendrogram)**:
     - Rows = Drugs (with left-hand row dendrogram tree).
     - Columns = Genes (with top column dendrogram tree).
     - Heatmap Cells = CPIC action severity scale (Avoid=4, Dose Adjust=3, Monitor=2, Standard=1, Insufficient=0.5, None=0) with discrete color ramps.
  2. **Drug × Recommendation Profile**:
     - Clusters medications by their distribution across clinical recommendation categories, grouping drugs with identical clinical impact.
  3. **Gene × Recommendation Profile**:
     - Clusters PGx genes based on the number and distribution of affected medications in the patient's regimen.
- **Interactive Inspection**: Clicking heatmap cells updates the dedicated `#pgx-dendrogram-detail` card.

### 4. Aggregated Population Visuals & Figure Pack (`#pgx-aggregated-visuals-section`)
Provides clinical and population context directly within the individual patient report, grounded in the study's two target adverse event cohorts (**ED** and **Polypharmacy** across older adult strata `65–74` and `75–84`):
- **Publication Figure Pack Dropdown**:
  - *Intervention priority heatmap* (`pgx_intervention_priority_heatmap`) — Multi-evidence drug–gene prioritization scores across **ED 65–74**, **ED 75–84**, **Polypharmacy 65–74**, and **Polypharmacy 75–84**.
  - *Global intervention network* (`pgx_global_intervention_network`) — Top model-seeded drug–gene intervention network.
  - *Cohort small multiples* (`pgx_cohort_small_multiples`) — Side-by-side network topologies across **ED** and **Polypharmacy** age bands.
  - *Therapeutic cluster ego networks* (`pgx_cluster_ego_networks`) — Focused modules separating beta-blockers, statins, and diuretics.
  - *Pathway context panel* (`pgx_pathway_context_panel`) — Edge counts and distributions across dynamics, kinetics, allergic response, and underappreciated signaling.
  - *Medication lead-time panel* (`pgx_time_to_event_panel`) — Days before event for sentinel drugs (e.g. Furosemide) in **ED** and **Polypharmacy** cohorts.
- **Interactive Deep-Links**: Direct links to full interactive Plotly HTML and publication-ready PNGs.
- **Cohort Gene Actionability Radar**: Embedded radar plot displaying multi-dimensional OODA evidence (Literature citations, VIP Evidence tiers, Causal signals, CPIC Guideline presence).

---

## Comparison: PGx Card vs. PGx Cohort

To avoid cognitive overload and preserve clear clinical utility, visual artifacts are segregated between the two tabs:

| Visual / Feature | PGx Card Tab | PGx Cohort Tab | Rationale |
|---|:---:|:---:|---|
| **Gene–Drug Actionability Matrix** | ✅ **Primary** | ❌ | Point-of-care patient prescribing decision support. |
| **Clustered Dendrogram Heatmaps** | ✅ **Interactive** | ❌ | Clusters the individual patient's medications and genes by prescribing consequence. |
| **Action Queue & Format Switcher** | ✅ **Yes** | ❌ | Patient-specific dosing adjustments and alert cards. |
| **Clinical Handoffs (Doctor/Pharmacy)** | ✅ **Yes** | ❌ | Actionable exports for pharmacy and medical records. |
| **Gene Actionability Radar** | ✅ **Embedded** | ✅ **Full View** | Patient report includes baseline radar; cohort tab provides full deep-dive. |
| **Publication Figure Pack** | ✅ **Preview / Links** | ✅ **Full View** | Embedded compactly in card report for contextual reference without clutter. |
| **Gene–Drug–Phenotype Network Topology** | ❌ *Excluded* | ✅ **Primary** | 100+ node interactive graph is valuable for epidemiological research but overwhelms clinical patient care. |
| **PubMed Literature Explorer** | 🔗 *Badge Links* | ✅ *Full Text List* | Card displays citation counts/badges; full publication browser lives in cohort tab. |

---

## The Phase Problem & Consumer Genealogy Data: Biological Mechanism & Safety Architecture

A critical clinical safety vulnerability in direct-to-consumer (DTC) genetic interpretation is the **unphased genotype problem**. The PGx Card implements strict bioinformatic guardrails to prevent erroneous clinical claims from genealogy array data.

```
                    THE CIS vs. TRANS PHASING DILEMMA
                    
   Genotype Call: Locus A = Heterozygous (A/G), Locus B = Heterozygous (C/T)
   
        CIS CONFIGURATION                          TRANS CONFIGURATION
    (Both mutations on same chromosome)        (Mutations on opposite chromosomes)

   Maternal: ──[ G (Mut) ]──[ T (Mut) ]──     Maternal: ──[ G (Mut) ]──[ C (Ref) ]──
               Allele 1: Defective                        Allele 1: Defective
               
   Paternal: ──[ A (Ref) ]──[ C (Ref) ]──     Paternal: ──[ A (Ref) ]──[ T (Mut) ]──
               Allele 2: Fully Functional                 Allele 2: Defective
               
   Result: Intermediate Metabolizer (IM)      Result: Poor Metabolizer (PM)
           One functional enzyme copy                 Zero functional enzyme copies
```

### 1. The Biological Reality: Cis vs. Trans Ambiguity
- Consumer arrays (e.g., Illumina Global Screening Array used by 23andMe and AncestryDNA) measure hybridization intensity at isolated single nucleotide polymorphisms (SNPs).
- They report an unphased unordered genotype (e.g., `rs1065852 = AG`, `rs3892097 = CT`), indicating the presence of two variant alleles, but **cannot determine chromosomal phase**—whether the variants sit in *cis* (on the same chromosome inherited from one parent) or in *trans* (on opposite homologous chromosomes).
- **Clinical Consequence**: 
  - In *cis*, the patient carries one severely mutated allele and one completely normal wild-type (`*1`) allele, retaining ~50% functional enzyme capacity (**Intermediate Metabolizer**).
  - In *trans*, both copies of the gene carry an inactivating mutation, yielding 0% enzyme function (**Poor Metabolizer**).
  - Prescribing a prodrug like codeine or tamoxifen based on unphased data risks fatal toxicity or complete therapeutic failure.

### 2. The "Default to *1" Imputation Hazard
- Consumer array bead chips probe only ~500,000 to 700,000 markers across the entire human genome.
- Over **80% of rare or population-specific functional star-allele defining variants** (especially in high-risk loci like `DPYD`, `TPMT`, and `SLCO1B1`) are physically absent from DTC arrays.
- **The Dangerous Legacy Practice**: Older, naive genomic calculators assumed that if a specific variant probe was absent from an input file, the patient must carry the wild-type reference allele (`*1`).
  - *Example*: A patient with a lethal `DPYD*2A` splice variant whose array did not test `rs3918290` would be called `*1/*1` (Normal Metabolizer). Administering standard 5-fluorouracil chemotherapy to this patient causes life-threatening neutropenia, mucositis, and death.
- **The PGx Card Guardrail**: Any unprobed locus is explicitly designated **`Data Not Present in File`** (`NOT_TESTED`). The pipeline **never imputes reference `*1`** in the absence of active probe evidence.

### 3. Copy Number Variation (CNV) & Pseudogene Blindspots
- Key pharmacogenes such as `CYP2D6` exhibit extreme structural complexity, including whole-gene deletions (`CYP2D6*5`), duplications (`*1xN`, `*2xN`), and hybrid conversions with adjacent pseudogenes (`CYP2D7`, `CYP2D8`).
- Standard consumer arrays cannot measure gene copy number (which requires clinical MLPA, ddPCR, or quantitative depth analysis).
- An individual with a `CYP2D6*1/*1xN` duplication is an **Ultrarapid Metabolizer (URM)**, rapidly converting codeine into dangerous concentrations of morphine. A genealogy file would blind-call this patient as an ordinary normal metabolizer.

### 4. Software Safety Architecture & Clinical Governance
To address these biological limitations, the PGx Card enforces five hard guardrails:

1. **Strict Category Classification**:
   - Any consumer-derived call with unphased ambiguity is assigned the distinct CPIC category:  
     `INSUFFICIENT_GENOTYPE_RESOLUTION`.
   - In the Actionability Matrix, these cells are color-coded in purple with diagnostic cross-hatch patterning (`pgx-cell-indet`) to alert the clinician immediately.
2. **Hard Prescribing Lockout**:
   - For `INSUFFICIENT_GENOTYPE_RESOLUTION` results, the dashboard blocks automated dose adjustments. The guidance text explicitly states: *Clinical action cannot be determined from unphased consumer genetic data. Order clinical-grade confirmatory testing.*
3. **Mandatory Prominent Warning Banner**:
   - Every on-screen report, PDF, PNG, and printed handoff includes an unmissable clinical notice:
     > **Important medical and technical notice:**  
     > *This report is exploratory. A genealogy file cannot support a prescribing claim; it is unphased and does not measure copy number. It does not assign a metabolizer status or a dose. Do not start, stop, or change a medicine from these results.*
4. **Clinical Lab Referral Protocol**:
   - Identifies candidate genes that warrant formal CLIA-certified/CAP-accredited laboratory testing, translating consumer curiosity into legitimate clinical diagnostics.
5. **Lab-Grade Allele Separation**:
   - Definitive CPIC dosing recommendations are strictly reserved for official phased/lab-verified diplotypes (e.g., `CYP2C19,*1,*17`, `SLCO1B1,*5,*1`) passed via standardized laboratory notation.

### 5. Backend Resolution Engine & Unit Test Suite
The biological guardrails are hardcoded in the serverless backend resolution engine:

- **Source Code**: [`10_risk_dashboard/backend/cpic_allele_resolver.py`](file:///c:/Projects/pgx-analysis/10_risk_dashboard/backend/cpic_allele_resolver.py)
  - `CALLING_ALGORITHM = "cpic-unphased-exploratory-v2"`
  - **Zero Imputation**: Absent assay sites are flagged `MISSING_SITE_STATUS = "Data Not Present in File"`. Never defaulted to `*1`.
  - **Phase Collision Detection**: When $\ge 2$ variant sites or candidate alleles are detected for a gene:
    ```python
    if len(hits) >= 2 or len(candidate_names) >= 2:
        return f"Multiple {gene} variants found. Phasing required for diplotype call."
    ```
  - **Copy-Number Safeguards**: Every returned gene record explicitly appends:
    `"Copy-number variation was not measured by this array file."` (or for `CYP2D6`: `"CYP2D6 region evaluated for target SNPs. CNV/Duplication not measured."`) and sets `cnvMeasured: False`.
  - **Clinical Nudge Trigger**: Genes in `NUDGE_GENES = {"CYP2C19", "CYP2D6", "VKORC1", "SLCO1B1", "HLA-B"}` automatically route to clinical lab confirmatory testing rather than dosage modification.

- **Unit Test Suite**: [`11_testing/tests/test_cpic_allele_resolver.py`](file:///c:/Projects/pgx-analysis/11_testing/tests/test_cpic_allele_resolver.py)
  - `test_two_disjoint_hets_require_phasing()`: Validates that disjoint heterozygous mutations in `CYP2C19` (`rs4244285` AG + `rs12248560` CT) do not call a diplotype and output `"Phasing required"`.
  - `test_reference_at_defining_site_is_not_called_star1()`: Validates that homozygous reference genotypes at `SLCO1B1` `rs4149056` are reported as detected reference and never called `*1`.
  - `test_single_snp_hom_does_not_assign_diplotype()`: Verifies that homozygous variant calls (`CYP2D6` `rs3892097` AA) still leave `diplotype: None` and flag `"CNV/Duplication not measured"`.
  - `test_incomplete_multivariant_is_indeterminate()`: Verifies that partial defining variant haplotypes remain `EXPLORATORY` and indeterminate.

### 6. When Consumer Microarray Data CAN Be Used Deterministically Without Parental Phase Data (The 4 Phase-Free Resolution Rules & Clinical Lab Correlation Workflow)

While multi-SNP pharmacogenes with **two or more heterozygous sites** (`Compound Heterozygotes`) require parental phasing or clinical long-read/haplotype testing to distinguish *cis* vs. *trans*, **four mathematical classes of genetic markers in a consumer `AncestryDNA.txt` or `23andMe` file have zero *cis/trans* phase ambiguity** and can be resolved **100% deterministically without parental DNA**:

```
   CASE A: TWO HETEROZYGOUS SITES                CASE B: HOMOZYGOTE + HETEROZYGOTE
   (Requires Parental Phase Data)                (100% Deterministically Phased!)
   Site 1 = A/G (Het), Site 2 = C/T (Het)        SLCO1B1: rs2306283 = G/G (Hom)
                                                          rs4149056 = T/C (Het)

   Possibility 1 (CIS — Same Parent):            ONLY 1 Mathematical Possibility:
     Parent 1: ──[ G ]──[ T ]── (Mut+Mut)          Parent 1: ──[ G ]──[ T ]── = *1B (*37) Normal
     Parent 2: ──[ A ]──[ C ]── (Normal *1)        Parent 2: ──[ G ]──[ C ]── = *15 Decreased
     Result: 1 working copy (Intermediate)
                                                 Why? Because BOTH parents contributed [G]
   Possibility 2 (TRANS — Opposite Parents):     at rs2306283! Whichever parent gave the [C]
     Parent 1: ──[ G ]──[ C ]── (Defective)      at rs4149056 MUST have [G + C] on that same
     Parent 2: ──[ A ]──[ T ]── (Defective)      chromosome! Zero parental phase ambiguity!
     Result: 0 working copies (Poor)
```

#### Rule 1 — Homozygote + Heterozygote Deterministic Phasing (`0` *Cis/Trans* Ambiguity)
- **Mathematical Proof**: Phase ambiguity only exists when **$\ge 2$ sites in the same gene are simultaneously heterozygous ($2^{k-1}$ phase configurations for $k$ heterozygous sites)**. When $k = 0$ (all homozygous) or $k = 1$ (at most **one** site is heterozygous and all other defining sites are homozygous), the number of possible chromosomal configurations is $2^{1-1} = 1$.
- **Real Clinical Example (`SLCO1B1` Statin Transporter)**:
  - A naive single-SNP lookup table (such as legacy `pgx-genes.csv` extractors) sees `rs2306283 = G/G` (`*1B`) and `rs4149056 = T/C` (`*1/*5`) independently, outputting a contradictory 3-allele row (`SLCO1B1,*1B,*1,*5`) or flagging "missing phase data."
  - However, because `rs2306283` (`c.388A>G`) is **homozygous `G/G`**, **both** maternal and paternal chromosomes carry `G` at position 388.
  - Because `rs4149056` (`c.521T>C`) is **heterozygous `T/C`**, one chromosome is locked as **`[c.388G + c.521T]` (`SLCO1B1*1B` / `*37` — Normal Function)** and the other chromosome is locked as **`[c.388G + c.521C]` (`SLCO1B1*15` — Decreased Function)**.
- **Real Clinical Example (`APOE` Lipid & Cardiovascular Isoform)**:
  - `APOE` is defined by two SNPs (`rs429358` + `rs7412`). Whenever at least one of the two SNPs is homozygous (e.g., `rs429358 = T/T` + `rs7412 = C/C` $\rightarrow$ `APOE ε3/ε3`; or `rs429358 = C/T` + `rs7412 = C/C` $\rightarrow$ `APOE ε3/ε4`), the exact two-SNP isoform on each parental strand is 100% deterministic without parental DNA. (Only the rare double-heterozygote `rs429358 = C/T` + `rs7412 = C/T` has `ε2/ε4` vs. `ε1/ε3` phase ambiguity).

#### Rule 2 — Single-Causal-SNP Pharmacogenes (Phenotype Driven by a Single Functional Locus)
- Even when CPIC assigns different star-allele numbers to different chromosomal backbones, the clinical phenotype for several key pharmacogenes is governed by a **single causal functional variant**:
  - **`SLCO1B1 rs4149056 (c.521T>C, p.Val174Ala)`**: Whether `c.521C` sits on a `c.388A` backbone (`*5`) or a `c.388G` backbone (`*15`), CPIC classifies both `*5` and `*15` identically as **No/Decreased Function** alleles conferring **SLCO1B1 Decreased Function** (statin myopathy susceptibility with high-dose simvastatin/atorvastatin) when heterozygous (`T/C`).
  - **`VKORC1 rs9923231 (c.-1639G>A)`**: Warfarin target sensitivity is defined by this single regulatory promoter SNP (`C/C` = Normal sensitivity, `C/T` = Increased sensitivity, `T/T` = High sensitivity).
  - **`CYP3A5 rs776746 (*3 splicing defect)`**: `C/C` (`*3/*3`) deterministically establishes non-expression on both chromosomes unless rare `*6`/`*7` alleles are involved.

#### Rule 3 — Homozygous Loci (`A/A`, `G/G`, `C/C`, `T/T`)
- Whenever a consumer array reports a **homozygous** call at a locus, both maternal and paternal chromosomes carry the identical base:
  - **`LPA rs10455872 = A/A` & `rs3798220 = T/T`**: Deterministically **100% negative on both chromosomes** for the two primary European/Mendelian high-`Lp(a)` structural variants.
  - **`IL6 -174G>C (rs1800795 = G/G)`**: Deterministically **homozygous** on both chromosomes for the high-transcription `IL-6` promoter allele (which drives acute-phase hepatic `LPA` gene transcription during severe Obstructive Sleep Apnea intermittent hypoxia).

#### Rule 4 — Single-Locus Clinical & Disease-Modifier SNPs Correlated with Objective Labs
- Unlike multi-SNP cytochrome P450 star alleles, clinical susceptibility and physiological modifier SNPs are evaluated **per locus** (additive or dominant allele presence) and correlated directly with a patient's longitudinal blood work and diagnoses:

| Clinical Axis | Target Loci in `AncestryDNA.txt` / `23andMe` | Why Parental Phasing Is Not Required | Clinical & Lab Correlation |
| :--- | :--- | :--- | :--- |
| **1. Statin Transport & Myopathy** | `SLCO1B1 rs2306283` + `rs4149056`; `ABCG2 rs2231142` | `Hom + Het` (`G/G + T/C`) phases deterministically to `*1B/*15`; `c.521T>C` alone defines decreased OATP1B1 uptake | Explains statin muscle sensitivity; supports preferring rosuvastatin or low-dose therapy if needed |
| **2. `Lp(a)` & OSA–`IL-6` Inflammatory Axis** | `LPA rs10455872`, `rs3798220` + `IL6 rs1800795 (-174G>C)` | Homozygous calls (`A/A`, `T/T`, `G/G`) have zero phase ambiguity | Distinguishes primary Mendelian `LPA` mutation from secondary `IL-6`-driven `Lp(a)` elevation caused by untreated Severe OSA (`AHI 52.8`) |
| **3. Periodontal & Alveolar Bone Resorption** | `TNF rs1800629 (-308G>A)`, `IL6 rs1800795`, `VDR rs731236 (TaqI)` | Single-SNP promoter & receptor modifiers act by allele dosage | Explains accelerated periodontal ligament/alveolar bone loss when compounded by OSA mouth-breathing & bruxism |
| **4. Vitamin D Synthesis & Transport** | `CYP2R1 rs10741657`, `GC rs7041`, `GC rs4588`, `VDR rs731236` | Single-SNP enzyme/binding-protein modifiers | Correlates with serum `25-OH Vitamin D` insufficiency (`<30 ng/mL`) and guides `D3` supplementation |
| **5. Iron Regulation & Blood Donation Recovery** | `TMPRSS6 rs855791`, `HFE rs1799945 (H63D)`, `HFE rs1800562 (C282Y)` | Single-SNP hepcidin/iron modifiers | Guides post-donation iron recovery after frequent whole-blood or `Power Red` apheresis donations |
| **6. African-Diaspora Hematology & Renal Traits** | `ACKR1 rs2814778` (Duffy-null), `APOL1 rs73885319`/`rs60910145`/`rs71785313`, `HBB rs334` | Single-SNP promoter (`ACKR1 c.-67T>C`) or biallelic risk loci | Establishes personalized ANC neutrophil reference ranges (`ACKR1`) and kidney screening (`APOL1`) |

#### Reusable Family Member Tooling
- **CLI Scanner**: [`C:\Projects\action\family_reunion\ancestry_com_migration\scripts\analyze_family_member_dna.py`](file:///c:/Projects/action/family_reunion/ancestry_com_migration/scripts/analyze_family_member_dna.py) — Any family member can pass their `AncestryDNA.txt` or `23andMe.txt` file to generate a complete **2-Part Clinical & Pharmacogenomic Report** separating **(Part A) Deterministically Phased / Phase-Free Clinical Findings** from **(Part B) Multi-SNP Unphased Pharmacogenes Requiring a CLIA/CAP Lab Panel**.
- **Interactive Browser Tool**: Available in **Tab 6 (`🩺 Healthcare PGx`)** of the Dixon Family Reunion Portal (`https://dixon-family-reunion.jerome-dixon.io/`), documented in [`C:\Projects\action\family_reunion\ancestry_com_migration\README.md`](file:///c:/Projects/action/family_reunion/ancestry_com_migration/README.md).

---

## Automated Testing & Validation

The PGx Card implementation is covered by automated end-to-end Puppeteer and Jest test suites in `11_testing/puppeteer/`:

- `tests/tabs/tab-pgx-card.test.js`:
  - Two-phase cohort profile loading and variant submission.
  - Action card counts, matrix rows, and triplet limits.
  - Scoped vs. full-formulary exports (JSON, CSV, PDF, Print, Technical Appendix).
  - Chip filtering, search autocomplete, and session reset.
  - Multi-variant and unphased consumer array uploads.
- Verification checks:
  - Options Matrix 3-panel layout and button accessibility.
  - Format toggle (Matrix vs. Row) DOM visibility.
  - CPIC action filter pills dynamic filtering.
  - Interactive cell click inspection cards.
  - Plotly dual-dendrogram and single-dendrogram rendering across all 3 clustering modes.
