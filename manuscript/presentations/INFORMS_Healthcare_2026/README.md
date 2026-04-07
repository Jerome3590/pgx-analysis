# INFORMS Healthcare 2026
**Conference:** INFORMS Healthcare Conference 2026
**Submission deadline:** TBD — check [meetings.informs.org/wordpress/healthcare](https://meetings.informs.org/wordpress/healthcare/submit/)
**Format:** 250-word abstract · In-person presentation · English

---

## Talk Title
*Consensus-Causal Feature Selection for Opioid ED Risk: SHAP ∩ FFA Across Seven Age Bands in a State-Level All-Payer Claims Database*

**Speaker:** R. Jerome Dixon, Ph.D. Candidate
**Co-author:** Elvin T. Price, Pharm.D., Ph.D., FAHA
**Based on:** Dixon & Price, manuscript submitted to *Clinical and Translational Science* (Wiley)

**Topical areas:** Personalized Medicine · AI/Machine Learning · Bioinformatics · Data Analytics

---

## Abstract (250 words)

**Background:** Ensemble machine learning applied to all-payer claims data (APCD) can identify opioid use disorder–related ED (OUD-ED) risk, but most published models are cross-sectional and do not characterize modifiable causal drivers across age bands.

**Methods:** A partition-first DuckDB architecture processed Virginia APCD data (2016–2019; 6.9M patients) across seven age bands (13–114 years) using ensemble models (CatBoost/XGBoost/XGBoost-RF). A Consensus Filter required that features be confirmed by both SHAP (gradient-based) and Formal Feature Attribution (FFA; Boolean rule extraction), yielding features designated causal rather than merely associational. Dynamic time warping (DTW) identified pre-diagnosis trajectory archetypes.

**Results:** Ensemble models achieved PR-AUC 0.840–0.979 on a prospective 2019 holdout. The pharmacogenomic polydrug score (`pgx_num_cpic_drugs`; CYP2D6/2C19 burden) ranked as the highest-SHAP causal feature (mean |SHAP| = 1.22). Gabapentin co-prescription (41% cases vs. 18% controls) and long-term opioid ICD-10 Z79.891 (prevalence ratio 5.6 in 55–64 band) were top actionable features. Notably, 23–41 FFA rules per band contained NOT clauses (absence of physical therapy or preventive monitoring), capturing protective care gaps invisible to cross-sectional models. DTW clustering identified two archetypes: Rapid-Onset (21%; 4.2-month trajectory — shorter than 90-day PDMP review cycles) and Chronic-Escalation (79%; 22.1 months with ≥3 identifiable care-addition windows).

**Conclusions:** PGx-enriched Consensus-Causal features from standard claims data identify modifiable OUD-ED drivers up to 22 months before presentation, enabling trajectory-stratified precision intervention protocols without genotyping infrastructure.

---

## Slide Outline (~20 min)

### Slide 1 — Motivation
- Cross-sectional ML models predict OUD risk but cannot identify *modifiable* causal drivers
- Virginia APCD: 6.9M patients, 2016–2019; 7 age bands (13–114 yrs)
- Gap: no published framework simultaneously resolving scalability + causal validity + temporal leakage prevention

### Slide 2 — The Consensus Filter
- Two orthogonal attribution methods: SHAP (gradient-based) + FFA (Boolean rule extraction from XGB)
- Consensus threshold: SHAP ≥ 75th percentile **AND** FFA support ≥ 0.05
- Features passing both = "causal" (not merely associational)
- **Figure:** `fig_shap.png` — top 20 Consensus-Causal features, 25–44 band; ★ marks intersection

### Slide 3 — Top Causal Features
- `pgx_num_cpic_drugs` — rank #1 (mean |SHAP| = 1.22); compound CYP enzyme burden
- Gabapentin count — rank #2 in 25–44 band; 41% cases vs. 18% controls; FDA 2019 black-box warning
- Z79.891 (long-term opioid) — 67% of 55–64 cases vs. 12% controls (prevalence ratio 5.6)
- **Protective absence features:** physical therapy absence (CPT 97110/97530) elevates risk; 23–41 NOT-clause FFA rules per band — invisible to cross-sectional models

### Slide 4 — FFA Rules (verbatim examples)
```
[25–44] Oxycodone ≥ 2 AND M54.5 (low back pain) AND NOT PT → P(OUD-ED) ↑↑
        [support=0.12, confidence=0.83]

[45–54] Z79.891 AND Gabapentin ≥ 1 AND F41.1 (anxiety) → P(OUD-ED) ↑↑
        [support=0.09, confidence=0.78]

[55–64] Hydrocodone ≥ 3 AND G89.29 (chronic pain) AND NOT Z23 → P(OUD-ED) ↑↑
        [support=0.07, confidence=0.75]
```

### Slide 5 — Trajectory Archetypes
- DTW clustering → k=2 (elbow criterion)
- **Rapid-Onset** (21%; 4.2 mo) — shorter than 90-day PDMP review cycle → patients deteriorate *before* protocols detect them
- **Chronic-Escalation** (79%; 22.1 mo) — ≥3 identifiable intervention windows at 3, 9, 15 months
- **Figure:** `fig_trajectories.png`

### Slide 6 — Performance
- PR-AUC 0.840–0.979 across 7 age bands on 2019 prospective holdout
- 384–498 Consensus-Causal features per band
- Training: 2016–2018 | Holdout: 2019 | 2020 excluded (COVID disruption)

### Slide 7 — Clinical Implications
- **Rapid-Onset:** trigger at first high-dose opioid fill, not 90 days → ~1 in 5 OUD-ED cases interceptable
- **Chronic-Escalation:** non-opioid care *addition* at 3/9/15 months (not discontinuation)
- PGx polydrug score deployable from standard claims — no genotyping infrastructure required
- Based on: Dixon & Price, manuscript under review, *Clinical and Translational Science*

---

## Source Chapter
`CH_3/ch03_cts.qmd` — submitted to *Clinical and Translational Science* (Wiley, CTS)

## Key Figures (from submission package)
| Figure | File | Used in Talk |
|--------|------|-------------|
| Figure 3 — SHAP feature importance | `output/submission/cts/ch03/figures/Figure_3.tiff` | Slide 2, 3 |
| Figure 4 — DTW trajectory archetypes | `output/submission/cts/ch03/figures/Figure_4.tiff` | Slide 5 |

## Ethics / Dual-Submission Note
Presenting journal-submitted work at INFORMS Healthcare is permitted under ICMJE norms — conference presentations are a distinct channel from duplicate publication. Slides should note: *"Based on manuscript submitted to Clinical and Translational Science (Dixon & Price, under review)."* No INFORMS proceedings paper is required or requested (abstract only).
