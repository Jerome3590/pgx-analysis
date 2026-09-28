# UC08 — PGx Cohort tab vs PGx Card

**Live:** https://pgx.jerome-dixon.io/

**Persona:** Researcher (PGx Cohort). Clinician / pharmacist (PGx Card).

**Live tabs:** PGx Cohort · PGx Card

## Numbered steps

| | **PGx Cohort** | **PGx Card** |
|--|----------------|--------------|
| Role | Population topology | Patient / session card |
| Load | **Load PGx Cohort Network** | **Load Cohort PGx Profile** and/or **Generate PGx results** |
| Content | Gene–drug–phenotype network, figure pack, cohort radar | Claims radar. Array upload: coverage and clinical-test referral. Lab alleles: action queue and exports |
| Question | How do PGx genes, drugs, and phenotypes connect in this cohort × age band? | What did this array file contain, or what does a lab diplotype imply? |

Use **PGx Cohort** for research topology. Use **PGx Card** for a claims radar, an exploratory array-file report, or a lab allele review. The cohort network is a prebuilt visual. The array-file card is computed at request time with DuckDB and is not one of the visualization artifacts.


## Live button names

- **Load PGx Cohort Network**
- **Load Figure Pack Visual**
- **Load Cohort PGx Profile**
- **Generate PGx results**

## Screenshot index

PNGs in `screenshots/`. These are captures of the live dashboard. Do not replace them with generated mockups. Recaptured 2026-09-28 as **clipped sections**.

- `01-pgx-cohort-network.png` — **PGx Cohort**: **Load PGx Cohort Network**, gene–drug–phenotype iframe
- `02-pgx-card-patient.png` — **PGx Card** **Patient** view after **Generate PGx results**, including the further-testing card (**Further clinical test recommended** for SLCO1B1) and the clopidogrel action
- `03-roles-compared.png` — **PGx Card** **Load Cohort PGx Profile** (claims radar + identified genes). Same genes as the cohort tab, but this is a session/claims profile, not the research network.

Canonical workflow text: repo `10_risk_dashboard/docs/DASHBOARD_USE_CASES.md`.

## NotebookLM prompt (required)

When creating Audio Overview, Video Overview, Slide Deck, or Briefing Doc:

- Use only the uploaded screenshots for any visual depiction of the PGx dashboard.
- Do not generate dashboard images, synthetic UI, or illustrative mockups.
- Describe the real screenshots; if a slide needs a figure, embed/refer to the uploaded PNG.
- Training slides should show or cite the captured screenshots, not AI-drawn interfaces.

