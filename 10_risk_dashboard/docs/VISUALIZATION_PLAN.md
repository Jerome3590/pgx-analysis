# PGx Risk Dashboard: Visualization Plan

**Reference:** [PGx Risk Calculator](https://jerome-dixon.io.s3.us-east-1.amazonaws.com/vcu/pgx-risk-calculator/index.html)

This document maps **research questions** to **dashboard tabs** and **data visuals**, and recommends how to populate each tab. Use the **full-dataset, filter-to-features** pattern where possible (all processing done upfront; Lambda/frontend only filter).

---

## Tab Order and Purpose

| # | Tab | Primary role |
|---|-----|---------------|
| 1 | **Causal Analysis** | Features driving target outcome; relations; drug combinations → polypharmacy ED |
| 2 | **BupaR Process Mining** | Sequences to target outcomes; times between sequences |
| 3 | **DTW Trajectories** | Routine vs no routine (admin ICDs vs ICD event count) → outcomes |
| 4 | **FP-Growth Patterns** | Connections/relationships between ICD, CPT, Drugs → target outcome |

---

## Data Pattern

- **Preferred:** Full dataset prepared upfront; dashboard filters to selected features (cohort, age band, item type). Minimal Lambda processing.
- **SHAP/FFA-driven:** BupaR, DTW, FP-Growth, and Causal use **SHAP/FFA important features** (Step 7 / Step 8) to filter the original dataset or to define the feature set. Event logs, trajectories, itemsets, and causal/SHAP charts are restricted to model-important items, so **visuals stay aligned with what drives model results** (see `README_implementation_plan_tab_visualizations.md`).
- **Fallback:** Run analysis based on selected features only if full-dataset approach is not feasible with current data.

---

## Filterability by user selections (Drug / ICD / CPT)

Visuals are **not all** filterable by the user’s selected drug/ICD/CPT codes from the Drugs and ICD/CPT tabs. Current behavior:

| Tab | Filterable by selected codes? | Notes |
|-----|-------------------------------|--------|
| **Risk Assessment** | ✅ Yes | Risk score is computed from the selected drugs, ICDs, and CPTs. |
| **Causal Analysis** | ✅ Yes | Optional: pass selected drugs/ICD/CPT as query params; causal and SHAP charts show only features matching those codes. Frontend sends current selections when loading. |
| **BupaR** | ❌ No | Cohort and age band only. Sequence/trajectory outputs are cohort-level. |
| **DTW** | ❌ No | Cohort and age band only. |
| **FP-Growth** | ✅ By cohort, age band, item type | Filtered by **cohort**, **age band**, and **item type** (Drug / ICD / CPT). Not by the specific selected code list; optional future: filter or highlight by selected codes. |

See `README_implementation_plan_tab_visualizations.md` for API details (query params and response fields).

---

## Research Questions → Tabs and Recommended Visuals

### 1. Routine vs no routine appointments → outcomes

**Question:** Is there a difference in outcomes for patients that don’t have routine appointments vs those that do?

**Tab:** **DTW Trajectories**

**Recommended visuals:**
- **DTW visual:** Admin ICDs vs number of ICD events (or similar trajectory metric) to compare “routine” vs “non-routine” patterns and their association with target outcome.
- **Existing panels to use:** “Trajectory Analysis Overview”, “Sample Trajectories”, “Trajectory Metrics”.
- **Data:** Trajectory features from `create_dtw_features.py` (e.g. `trajectory_length`, `trajectory_diversity`, DTW distances); optionally stratify or annotate by admin/routine vs non-routine (e.g. admin ICD count or protocol-filtered events from `model_events_no_protocols.parquet`).
- **Suggested addition:** One plot comparing outcome (or risk) by “routine vs non-routine” (e.g. by admin ICD count or protocol flag) to directly answer this question.

---

### 2. Sequences to target outcomes

**Question:** What are the sequences that lead to target outcomes?

**Tab:** **BupaR Process Mining**

**Recommended visuals:**
- **BupaR sequence/trace visuals:** Top traces and activity sequences that precede (and optionally follow) the target event.
- **Existing panels:** “Activity Sequence (Top)”, “Overall Activity Frequency”, “Pre-Target Activity Frequency”, “Gantt Chart (Pre-Target)”.
- **Data:** BupaR event logs and trace tables from `create_bupar_outputs_*` (e.g. `*_traces_top_bupar.csv`, process matrices). Already in `gold/bupar/{cohort}/{age_band}/` and feature_importance plots.
- **Suggested emphasis:** Label “Activity Sequence (Top)” as “Sequences to target outcomes” and ensure it shows pre-target sequences that are most associated with the outcome.

---

### 3. Times between sequences to target outcomes

**Question:** What are the times in between sequences that lead to target outcomes?

**Tab:** **BupaR Process Mining**

**Recommended visuals:**
- **BupaR time / Gantt visuals:** Time-to-target and inter-activity times (e.g. time between key activities and target).
- **Existing panels:** “Gantt Chart (Pre-Target)”, “Gantt Chart (Post-Target)”, “Activity Milestones Gantt”.
- **Data:** Same BupaR outputs; time-to-target and per-trace timing from `create_bupar_outputs_*` and related feature CSVs.
- **Suggested addition:** If not already present, one summary chart of “time between key milestones” or “time from last drug/ICD/CPT to target” by sequence type (e.g. top vs rare) to answer this question directly.

---

### 4. ICD / CPT / Drug connections to target outcome

**Question:** What are the connections/relationships between ICD, CPT, and Drugs that lead to target outcome?

**Tab:** **FP-Growth Patterns**

**Recommended visuals:**
- **FP-Growth network visuals:** Co-occurrence and (where available) association-rule networks for drugs, ICD, and CPT.
- **Existing panels:** “Co-occurrence Network”, “Top Itemsets”, “Itemset Support Distribution”; item type selector (Drug Names, ICD Codes, CPT Codes, Medical Codes).
- **Data:** FP-Growth itemsets and rules from `10_risk_dashboard/visualizations/fpgrowth` outputs; network HTML and plots in `gold/fpgrowth/{cohort}/{age_band}/plots/`.
- **Suggested emphasis:** Use “Co-occurrence Network” as the main answer to this question; keep item type filter so users can switch between ICD, CPT, and Drugs. Optionally add a small note that these patterns are exploratory (visualization-only, not model features) per `README_visualization_only.md`.

---

### 5. Features driving target outcome and how they relate

**Question:** What features drive the target outcome and how do they relate to each other?

**Tab:** **Causal Analysis**

**Recommended visuals:**
- **Causal/FFA + SHAP:** Feature importance and causal importance; optionally a compact view of how features relate (e.g. interaction plot or small multi-dimensional view).
- **Existing panels:** “Top Causal Factors (FFA)”, “SHAP Feature Importance”, “Feature Interactions”.
- **Data:** FFA causal importance and SHAP from Lambda (`/causal/importance`, `load_causal_importance`, `load_shap_importance`); S3: `gold/ffa_analysis/`, `gold/shap_analysis/`.
- **Suggested addition:** **Radar chart** (as in plan): one radar chart of top 5–8 causal/SHAP features (normalized importance) to show “how they relate” in one view. Frontend can build this from existing causal_factors + shap_importance API response.

---

### 6. Drug combinations that drive polypharmacy ED visit

**Question:** What combination of drugs drives polypharmacy ED visit?

**Tab:** **Causal Analysis** (and optionally BupaR for sequence context)

**Recommended visuals:**
- **Causal + BupaR combo:** Bar or combo chart of “drug combinations” or “last drugs before target” from BupaR, plus causal importance of drug-related features from FFA/SHAP.
- **Existing:** Causal tab already shows top causal factors (often drug-related); BupaR “Pre-Target Activity Frequency” and “Activity Sequence (Top)” show drug sequences.
- **Data:** Causal importance (drug features); BupaR top sequences and pre-target activity frequencies (drugs).
- **Suggested addition:** One panel or chart that explicitly shows “Top drug combinations before polypharmacy ED” (from BupaR top sequences or FP-Growth drug itemsets) and/or “Drug features by causal importance” (from FFA). If a single tab is preferred, keep this in Causal Analysis and add a small “Drug combinations (from process mining)” subsection or link to BupaR tab.

---

## Summary: Which Visuals Populate Which Tab

| Tab | Panels / visuals | Research questions |
|-----|-------------------|--------------------|
| **Causal Analysis** | Top Causal Factors (FFA), SHAP Feature Importance, Feature Interactions; **+ Radar chart (recommended)**; **+ Drug-combo emphasis** | Q5 (features & relations), Q6 (drug combinations → polypharmacy ED) |
| **BupaR Process Mining** | Activity frequencies (overall, pre-, post-target), Gantt charts, Top sequences, Milestones; **+ Time-between summary (recommended)** | Q2 (sequences to target), Q3 (times between sequences) |
| **DTW Trajectories** | Trajectory overview, Sample trajectories, Trajectory metrics; **+ Routine vs non-routine comparison (recommended)** | Q1 (routine vs no routine → outcomes) |
| **FP-Growth Patterns** | Top itemsets, Support distribution, **Co-occurrence network** (main), item type = Drug/ICD/CPT | Q4 (ICD/CPT/Drug connections to target) |

---

## Coverage: Original + New Research Questions

**Source for original questions:** `docs/CrossStep_Workflow/README_research_questions_mapping.md` and `docs/Presentations/Pharmacy_Translational_Informatics_Presentation.md`.

### Original research questions (project)

| # | Cohort | Question / component | Dashboard tab(s) | Visual(s) | Covered? |
|---|--------|----------------------|-------------------|----------|-----------|
| **RQ1** | **ED_NON_OPIOID (Polypharmacy)** | Does drug window influence target outcome? | **Risk Assessment** + **Causal Analysis** | Risk score (ensemble); FFA + SHAP feature importance | ✅ Yes |
| | | Which drugs are involved? | **FP-Growth** (+ Causal) | Top itemsets, co-occurrence network (item type = Drug); causal factors (drug features) | ✅ Yes |
| | | Is there a temporal/ordering aspect? | **BupaR** | Sequences to target, pre-target Gantt, activity frequency | ✅ Yes |
| **RQ2** | **OPIOID_ED** | What CPT/ICD codes and drugs predict OPIOID_ED? | **FP-Growth** + **Causal Analysis** | Itemsets + network (ICD, CPT, Drug); FFA + SHAP | ✅ Yes |
| | | Predictive patterns (sequences)? | **BupaR** | Top sequences, pre-target frequency, Gantt | ✅ Yes |
| | | Feature importance? | **Causal Analysis** | Top Causal Factors, SHAP, Feature Relations (radar) | ✅ Yes |
| | | Can we identify high-risk trajectories? | **DTW Trajectories** | Trajectory overview, sample trajectories, metrics; **High-Risk vs Low-Risk Trajectories** (outcome rate by trajectory archetype quartiles) | ✅ Yes |
| | | Formal causality assessment? | **Causal Analysis** | FFA causal importance, SHAP, interactions | ✅ Yes |

\* **Partial:** Current DTW panels show trajectory patterns and metrics but do not yet explicitly stratify or label “high-risk” vs “low-risk” trajectory archetypes (e.g. by target status or risk band). Adding a routine-vs-non-routine or risk-stratified view would fully answer “high-risk trajectories.”

---

### New additions (plan)

| # | Question | Tab(s) | Visual(s) | Covered? |
|---|----------|--------|-----------|-----------|
| N1 | Routine vs no routine appointments → outcomes? (admin ICDs vs # ICD events) | **DTW** | Trajectory overview, sample trajectories, metrics; **Routine vs No Routine (Outcomes)** (outcome rate by trajectory intensity: Low/Medium/High) | ✅ Yes* |
| N2 | What sequences lead to target outcomes? | **BupaR** | “Sequences to Target Outcomes”, pre-target frequency, Gantt | ✅ Yes |
| N3 | What times between sequences lead to target outcomes? | **BupaR** | “Times Between Sequences (Pre-Target Gantt)”, milestones Gantt | ✅ Yes |
| N4 | What connections between ICD, CPT, Drugs → target outcome? | **FP-Growth** | Co-occurrence network, itemsets (filter by item type) | ✅ Yes |
| N5 | What features drive outcome and how do they relate? | **Causal** | FFA, SHAP, Feature Interactions, **Feature Relations (radar)** | ✅ Yes |
| N6 | What drug combinations drive polypharmacy ED visit? | **Causal** + **BupaR** | Causal factors (drugs), BupaR sequences / pre-target activity | ✅ Yes |

\*\* **Partial:** DTW tab answers trajectory *patterns*; a dedicated “routine vs non-routine” comparison (e.g. outcome by admin ICD count or protocol flag) is recommended but not yet implemented.

---

### Summary: Do the visualizations answer all research questions?

| Scope | Fully covered | Partial / gap |
|-------|----------------|----------------|
| **Original (RQ1 + RQ2)** | Which drugs/ICD/CPT involved; temporal/ordering; drug window influence; predictive patterns; feature importance; formal causality | **High-risk trajectories:** DTW shows trajectories but not yet an explicit “high-risk” stratification. |
| **New (N1–N6)** | Sequences to target (N2); times between sequences (N3); ICD/CPT/Drug connections (N4); features & relations (N5); drug combinations → polypharmacy ED (N6) | **Routine vs no routine (N1):** Needs a dedicated comparison plot (admin ICDs vs # events or protocol flag). |

**Conclusion:** The dashboard **answers all** original and new research questions. N1 is answered via outcome rate by trajectory intensity (Low/Medium/High) as a proxy for routine vs non-routine care; optional future enhancement: direct admin ICD or protocol-flag comparison if data is available. (Previously noted gaps—Routine vs no routine comparison and High-risk trajectories—are now implemented.)

1. **DTW: “Routine vs no routine” (N1)** – Add one comparison visual (e.g. outcome or risk by routine vs non-routine / admin ICD count) from existing DTW or protocol-filtered data.
2. **DTW: “High-risk trajectories” (RQ2)** – Optionally add stratification or labeling of trajectory archetypes by outcome/risk so “high-risk” trajectories are explicitly visible.

Once these two additions are in place, the visualizations will fully cover all original and new research questions.

---

## Implementation Notes

1. **Causal:** Add a radar chart in the frontend using existing `causal_factors` and/or `shap_importance` from `/visualizations/causal`. Add a short “Drug combinations” subsection or panel that surfaces drug-related causal factors and/or points to BupaR for sequences.
2. **BupaR:** Ensure “Activity Sequence (Top)” and Gantt panels are clearly labeled as “Sequences to target” and “Times between sequences”. Optionally add a time-between summary chart (backend or frontend from existing BupaR outputs).
3. **DTW:** Add a “Routine vs non-routine” comparison (e.g. by admin ICD count or protocol flag) using existing DTW metrics or protocol-filtered events; can be a new plot in the pipeline or a small frontend chart if summary stats are exposed.
4. **FP-Growth:** Keep item type selector; emphasize “Co-occurrence Network” as the main visual for ICD/CPT/Drug relationships. Ensure network HTML is loaded (iframe or fetched and injected) when available.

All new visuals should prefer the **full-dataset, filter-to-features** pattern: precompute at build/pipeline time; dashboard only filters by cohort, age_band, and item type.
