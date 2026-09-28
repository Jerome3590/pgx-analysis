# Dashboard use cases and workflows

**Purpose:** Canonical **persona → tab → numbered workflow** map so dashboard visuals stay aligned with **live product behavior** on [https://pgx.jerome-dixon.io/](https://pgx.jerome-dixon.io/).

Labels below match the live tab bar and buttons in `frontend/index.html` and `frontend/tabs/*.html`. This is not the older Step 9 four-tab plan.

**Related (do not duplicate here):**

- Research-question → artifact table: [RESEARCH_QUESTIONS_ARTIFACTS.md](RESEARCH_QUESTIONS_ARTIFACTS.md)
- How visuals are produced: [README_visualization_plan.md](README_visualization_plan.md)
- Static-first JSON (S3/CloudFront, Lambda fallback): [STATIC_FIRST_JSON.md](STATIC_FIRST_JSON.md)
- Training screenshots + README pack: [TRAINING.md](TRAINING.md). Public video / slides / audio: [Drive folder](https://drive.google.com/drive/folders/1cGbdoH-HDEooRyqWnhS6btWyhD4XOKQt?usp=drive_link)

---

## Live tab map

| Row | Live names |
|-----|------------|
| Cohort | **Opioid ED** · **Polypharmacy** |
| Primary | **User Guide** · **Risk Assessment** · **Drugs** · **ICD Codes** · **CPT Codes** · **PGx Card** |
| Visualizations | **Feature Importance** · **Scenario Analysis (FFA/SHAP)** · **BupaR Process Mining** · **DTW Trajectories** · **FP-Growth Patterns** · **Drug Networks** · **PGx Cohort** |

ICD Codes and CPT Codes are used for **Opioid ED** risk only. On **Polypharmacy** those tabs are hidden; risk uses drugs only.

Age on Risk Assessment selects the age band. Scoring requires **age 13–114** (band 0–12 is excluded). After **Calculate Risk Score**, an **Event Density** badge appears: **low / medium / high / extreme**. That bin is auto-synced onto visualization tabs that have an Event density control, and onto **PGx Card** → Event Density.

---

## Personas

**Clinician / pharmacist.** Score a claims-style profile (age + selected codes), replace one code at a time, compare two or more saved sets (or empty baseline vs current), and read leave-one-out drug contributions (Δp̂). After a score, open **View PGx Card →** for the cohort/claims radar, then optionally upload a consumer array file or type a lab `Gene,Allele` line and use **Clinician / pharmacist** view. Array files produce coverage and a clinical-test referral. Lab alleles produce action categories. Exports include **Pharmacy handoff** and **Download summary for your doctor**. **Send to pharmacy (coming soon)** is disabled — there is no live e-prescribe.

**Researcher.** Use **Feature Importance**, **Scenario Analysis (FFA/SHAP)**, **BupaR Process Mining**, **DTW Trajectories**, **FP-Growth Patterns**, **Drug Networks**, and **PGx Cohort** to inspect population drivers, sequences, and gene–drug topology. These tabs answer RQ1/RQ2 and N1–N6. The artifact table lives in [RESEARCH_QUESTIONS_ARTIFACTS.md](RESEARCH_QUESTIONS_ARTIFACTS.md) — this document only names the path.

**Patient / self-assessment.** Enter **Age**, add codes on **Drugs** (and ICD/CPT if Opioid ED), click **Calculate Risk Score**, then optionally upload AncestryDNA, 23andMe, MyHeritage, or VCF, or type a lab `Gene,Allele` line on **PGx Card**. Array files are parsed in the browser. The report lists detected variants and gene coverage and points to a clinical test. It does not assign a diplotype or a dose from those files. Switch **View** to **Patient** for plainer language. There is no account and no pharmacy transmission.

---

## Clinical OODA path

Clinical use is a short loop: score → inspect the same density bin → act on the card.

```mermaid
flowchart LR
    A["Observe<br/>Risk Assessment<br/>Calculate Risk Score"] --> B["Orient<br/>Event Density badge<br/>synced viz tabs"]
    B --> C["Decide<br/>Replace / Compare<br/>Drug contributions"]
    C --> D["Act<br/>PGx Card<br/>View PGx Card →"]
    D -.-> A
```

1. **Observe** — **Risk Assessment**: Age + codes → **Calculate Risk Score** → ensemble p̂, band, **Event Density** badge.
2. **Orient** — Open the density-synced research tabs (BupaR, DTW, FP-Growth, PGx Cohort) and/or the **PGx Card** radar for that cohort / age band / bin.
3. **Decide** — **Replace / swap a code**, **Save current selection** + **Compare Scenarios**, and **Drug contributions** (LOO Δp̂). **Scenario Analysis (FFA/SHAP)** explains drivers; it does **not** recompute ensemble risk.
4. **Act** — **PGx Card**: claims profile via **Load Cohort PGx Profile**, an exploratory array-file report (coverage and clinical-test referral), or a lab allele card. Lab-allele actions are categories plus guideline URLs. Array files do not assign a dose.

Researcher work sits beside this loop (same cohort / age / bin) and is mapped in [RESEARCH_QUESTIONS_ARTIFACTS.md](RESEARCH_QUESTIONS_ARTIFACTS.md).

---

## Researcher RQ path

Do not treat Scenario Analysis as a second risk calculator.

| Intent | Live tabs | Notes |
|--------|-----------|--------|
| Population drivers | **Feature Importance** | Heatmap by age band. Standalone — does not need a prior score. |
| Feature / interaction signals | **Scenario Analysis (FFA/SHAP)** | Same cohort, age, and selected codes as Risk Assessment. Optional what-if codes. **Does not recalculate ensemble p̂.** |
| Sequences and time | **BupaR Process Mining**, **DTW Trajectories** | Cohort × age × Event density (auto-synced after a score). Not filtered by selected codes. |
| Co-occurrence | **FP-Growth Patterns**, **Drug Networks** | FP-Growth has Event density. Drug Networks is cohort × age (FI-filtered rules, combined train window). Neither is filtered by selected codes. |
| Population PGx topology | **PGx Cohort** | Network + figure pack + radar. Distinct from the patient **PGx Card**. |

Full RQ → artifact mapping: [RESEARCH_QUESTIONS_ARTIFACTS.md](RESEARCH_QUESTIONS_ARTIFACTS.md). How those files are built: [README_visualization_plan.md](README_visualization_plan.md). How the browser loads them: [STATIC_FIRST_JSON.md](STATIC_FIRST_JSON.md).

---

## Use cases

### 1. Cohort risk assessment with replace-and-compare

Modeled ensemble risk (p̂) for a claims-style profile.

1. Choose **Opioid ED** or **Polypharmacy**.
2. On **Risk Assessment**, set **Age** (13–114).
3. Open **Drugs**, search, and select medications (Ctrl+click / Cmd+click). Use **Edit codes** to jump there from Risk Assessment.
4. For **Opioid ED** only, add codes on **ICD Codes** and **CPT Codes**.
5. Optional overrides: **# drugs**, **# CPIC drugs**, **# total events** (blank defaults to the low-density bin).
6. Click **Calculate Risk Score**. Read the score, band, model name, and **Event Density** badge (low / medium / high / extreme).
7. To swap one selected drug, ICD, or CPT: use **⇄** on a chip, or **Replace / swap a code** → **Replace**, then **Calculate Risk Score** again.
8. To compare modeled probabilities: name the set → **Save current selection**. Save **two or more** sets, then **Compare Scenarios**. With no saved sets, Compare is empty baseline vs current. Leave **Include empty baseline** checked to also show the 2019 population baseline.
9. Click **Drug contributions** for leave-one-out Δp̂ per selected drug (ICD/CPT stay in the vector). Positive Δp̂ means removing that drug lowers predicted risk.

**Compare Scenarios** scores every saved set you kept (two or more), not a single scenario. Deltas are versus the reference (empty baseline or the first saved set).

---

### 2. Scenario Analysis (FFA/SHAP) — explain drivers

This tab shows FFA interaction factors and SHAP importance. It does **not** recalculate the ensemble risk score.

1. Set cohort and **Age** on **Risk Assessment**, and select the codes you care about. A prior **Calculate Risk Score** is the intended context (Event density auto-syncs).
2. Open **Scenario Analysis (FFA/SHAP)**.
3. Optional: **What-if scenario** (comma-separated codes), **Show features** (Top 10 / Top 20 / All), **Event density**.
4. Click **Load Scenario Analysis**. Charts: **Top Interaction Factors (FFA)**, **SHAP Feature Importance**, **Effect on outcome (by feature)**.
5. **Clear filters** reloads with no code selection.

For modeled p̂, return to **Risk Assessment** and use **Calculate Risk Score**, **Replace**, or **Compare Scenarios**. This is the only visualization tab that filters charts by the codes you selected.

---

### 3. Post-score density-bin exploration

After **Calculate Risk Score**, the **Event Density** badge is the same stratum the models and most visuals were built on.

1. Note the badge (low / medium / high / extreme).
2. Open **BupaR Process Mining** → **Load BupaR Visualizations** (Event density already set).
3. Open **DTW Trajectories** → **Load DTW Visualizations**.
4. Open **FP-Growth Patterns** → **Load FP-Growth Visualizations**.
5. Open **PGx Cohort** → **Load PGx Cohort Network** (and optionally **Load Figure Pack Visual**).
6. Open **PGx Card**: Event Density is auto-set → **Load Cohort PGx Profile**.

These tabs load **cohort × age × bin** (or cohort × age) artifacts. They are **not** filtered by the individual codes you selected, except Scenario Analysis (use case 2).

---

### 4. Feature Importance (population drivers)

1. Open **Feature Importance**.
2. Set **View** to **Opioid ED**, **Polypharmacy**, or **Combined cohorts**.
3. Set **Show features** (Top 10 / Top 20 / All).
4. Click **Load Feature Importance Heatmap**.

Standalone population view (Step 3a Monte Carlo CV). It does not use the current patient’s selected codes.

---

### 5. Pattern and process tabs

| Tab | Load button | What you get |
|-----|-------------|--------------|
| **BupaR Process Mining** | **Load BupaR Visualizations** | Sequences, activity frequency, Gantt / trace explorer; Event density control. |
| **DTW Trajectories** | **Load DTW Visualizations** | Trajectory clusters, routine vs utilization, time-to-target; Event density control. |
| **FP-Growth Patterns** | **Load FP-Growth Visualizations** | Drug itemsets and association network; Event density control. |
| **Drug Networks** | **Load Drug Network** | Interactive Drug Networks (Cytoscape) graph of **FI-filtered** FP-Growth rules (not Plotly). Cohort × age only — no Event density dropdown. |

**Drug Networks** is a live third-row tab. It is the same FI-gated rule set as FP-Growth, rendered as Cytoscape HTML. It is not listed in older Step 9 four-tab docs.

None of these four tabs recompute ensemble p̂. None filter by the codes selected on Drugs / ICD / CPT.

---

### 6. Claims-only PGx Card

Population / claims evidence for the patient’s cohort, age band, and density bin — not a genotype call.

1. After a score, click **View PGx Card →**, or open **PGx Card** and set **Cohort**, **Age Band**, and **Event Density**.
2. Click **Load Cohort PGx Profile**.
3. Read **Gene Actionability Profile** (radar) and **Identified PGx Genes**. The radar is **cohort/claims association**, not a CPIC prescribing action.

When you later generate a personalized card (use case 7):

- **Drug scope:** **Active medications** (`ACTIVE`), **Selected drugs** (`SELECTED`), or **All matched drugs** (`ALL_MATCHED`).
- **Actionable only** hides standard / no-CPIC rows in the queue, matrix, and exports.
- **Polypharmacy triplet engine** enumerates **regimen-only** three-way matches (need **≥ 3** APCD generics). Triplets are never shown as CPIC recommendations. `ALL_MATCHED` does not explode every CPIC drug into triplets.

---

### 7. Personalized PGx Card from DNA

Optional refinement after (or instead of) the claims radar.

1. Under **1. Gene data**, leave **Session / patient pseudonym** blank for an anonymous session, or enter a label.
2. Enter **Gene / allele** lines (`CYP2D6,*1,*4`) **or** **Or upload file**: AncestryDNA, 23andMe, or MyHeritage (txt/csv/zip), Excel `.xlsx` (`Gene,Allele1,Allele2`), or unphased VCF.
3. Files are parsed **in this browser**. They are not uploaded to S3. The browser keeps official CPIC defining rsids, chromosome, and position, then `POST /pgx/card` runs the DuckDB Parquet pipeline.

   **Array files stay exploratory.** DuckDB writes the parsed rows to ephemeral Snappy Parquet (`Chromosome`, `Start`, `End = Start + 1`, `genotype`) and joins them to a CPIC target Parquet on chromosome and `Start BETWEEN gene_start AND gene_end`. The card then shows detected variants and **Gene Coverage** (percent of known sites present). A site missing from the file is **Data Not Present in File**. It is not filled in as `*1`. Copy number is not measured. The card does not assign a diplotype, a metabolizer status, or a dose from this path. High-value genes (`CYP2C19`, `CYP2D6`, `VKORC1`, `SLCO1B1`, `HLA-B`) with a detected variant open a clinical-test referral. **Download summary for your doctor** lists rsids and CPIC drug references for a clinician.

   **Lab lines are different.** `CYP2C19,*1,*2` skips the array pipeline and uses the official diplotype-to-phenotype table.

4. Under **2. Virginia APCD medications**, search generics (two characters), set **Drug scope** and **Actionable only**, and choose **Patient** or **Clinician / pharmacist**. Medication actions apply to lab allele lines. Array-file results do not become dose adjustments.
5. Click **Generate PGx results**. The request is **POST `/pgx/card`** with parsed rsid rows or lab alleles (plus selected APCD drugs and scope) — not the raw genome file.
6. For a lab allele line, review **Medication-first action queue**, **Gene–drug actionability matrix**, **Polypharmacy triplet engine**, and **Genes tested**. Unlisted allele pairs stay **indeterminate**, never “normal.” For an array file, review the exploratory finding, gene-coverage line, and clinical-test referral instead of a dose queue.
7. Export: **Print**, **Download JSON**, **Download CSV**, **Download PNG**, **Download PDF**, **Download summary for your doctor**, **Copy to clipboard**, **Technical appendix**, **Pharmacy handoff**.
8. **Send to pharmacy (coming soon)** stays disabled. There is no live e-prescribe. Ordering a clinical PGx panel is a referral, not an in-dashboard lab order.

---

### 8. PGx Cohort tab vs PGx Card

| | **PGx Cohort** | **PGx Card** |
|--|----------------|--------------|
| Role | Population topology | Patient / session card |
| Load | **Load PGx Cohort Network** | **Load Cohort PGx Profile** and/or **Generate PGx results** |
| Content | Gene–drug–phenotype network, figure pack, cohort radar | Claims radar (optional). Array upload: coverage and clinical-test referral. Lab alleles: action queue, triplets, exports |
| Question | How do PGx genes, drugs, and phenotypes connect in this cohort × age band? | What did this array file contain, or what does a lab diplotype imply for this session? |

Use **PGx Cohort** for research topology. Use **PGx Card** for a claims radar, an exploratory array-file report, or a lab allele review. Array files do not produce a dose list.

---

### 9. Documentation / model trust

1. Open **User Guide**. Use the **Training** table for video / slides / audio. Files are in the [public Drive folder](https://drive.google.com/drive/folders/1cGbdoH-HDEooRyqWnhS6btWyhD4XOKQt?usp=drive_link) as `{nn}_{use_case}.mp4` / `.pdf` / `.m4a`.
2. Read **How to use**, research-question coverage, and the compact use-case summary (this file is the long form).
3. Review **Model performance and at-risk identification** (2019 temporal holdout after leakage correction; selected model by cohort and age band from [Dixon & Price, *Clin Transl Sci*, doi:10.1111/cts.70690](https://doi.org/10.1111/cts.70690) Table 2 and [Dixon & Price, *Clin Transl Sci*, doi:10.1111/cts.70718](https://doi.org/10.1111/cts.70718) Table 2).
4. Review **Dashboard visual artifacts (from manifest)** (`visualizations/dashboard_visual_objects.json`).
5. Use **Feature importance sources for visuals** and **Event density bins** to interpret why BupaR/DTW vs FP-Growth can disagree, and why the badge matters for model routing.
6. Read **Unphased DNA files and star alleles** so AncestryDNA, 23andMe, MyHeritage, and VCF uploads are read as coverage and referral, not as a diplotype or a dose.

In-app User Guide is the trust surface for users who never open the repo. This markdown is the workflow source of truth for implementers.

---

## What this dashboard does not do

- A diplotype, metabolizer status, or dose from AncestryDNA, 23andMe, MyHeritage, or unphased VCF. Those files are unphased and do not measure copy number. The live path is DuckDB Parquet coverage plus a clinical-test referral. Missing sites stay **Data Not Present in File**.
- An in-dashboard order for a CLIA/CAP PGx panel. The card links to CPIC guidelines and NSGC counselor search and can download a physician summary.
- Invented CPIC guideline prose. Lab allele lines use action categories and guideline URLs. Array files say “CPIC guideline reference available” for an associated drug and do not recommend a dose. Unlisted lab allele pairs stay indeterminate.
- Live e-prescribe or **Send to pharmacy**.
- Filtering BupaR, DTW, FP-Growth, Drug Networks, or PGx Cohort by the codes selected on Drugs / ICD / CPT. **Scenario Analysis (FFA/SHAP)** is the exception.
- Treating **Compare Scenarios** as a single-scenario button. Live behavior is **two or more saved sets**, or empty baseline vs current when nothing is saved.
