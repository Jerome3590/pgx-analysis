# UC08 — PGx Cohort tab vs PGx Card

**Live:** https://pgx.jerome-dixon.io/

**Persona:** Researcher (PGx Cohort). Clinician / pharmacist (PGx Card).

**Live tabs:** PGx Cohort · PGx Card

## Numbered steps

| | **PGx Cohort** | **PGx Card** |
|--|----------------|--------------|
| Role | Population topology | Patient / session card |
| Load | **Load PGx Cohort Network** | **Load Cohort PGx Profile** and/or **Generate PGx results** |
| Content | Gene–drug–phenotype network, figure pack, cohort radar | Claims radar (optional) + gene-data card, action queue, triplets, exports |
| Question | How do PGx genes, drugs, and phenotypes connect in this cohort × age band? | What should this session do given claims context and optional alleles? |

Use **PGx Cohort** for research topology. Use **PGx Card** when a person (or a claims proxy) needs an action list.


## Live button names

- **Load PGx Cohort Network**
- **Load Figure Pack Visual**
- **Load Cohort PGx Profile**
- **Generate PGx results**

## Screenshot index

PNGs in `screenshots/`. These are captures of the live dashboard. Do not replace them with generated mockups. Recaptured 2026-09-26 as **clipped sections** (the first pass used full-page shots; `01` and `03` were nearly the same PGx Cohort viewport).

- `01-pgx-cohort-network.png` — **PGx Cohort**: **Load PGx Cohort Network**, gene–drug–phenotype iframe
- `02-pgx-card-patient.png` — **PGx Card** **Patient** view after **Generate PGx results** (session header + medication-first action queue)
- `03-roles-compared.png` — **PGx Card** **Load Cohort PGx Profile** (claims radar + identified genes). Same genes as the cohort tab, but this is a session/claims profile, not the research network.

Canonical workflow text: repo `10_risk_dashboard/docs/DASHBOARD_USE_CASES.md`.

## NotebookLM prompt (required)

When creating Audio Overview, Video Overview, Slide Deck, or Briefing Doc:

- Use only the uploaded screenshots for any visual depiction of the PGx dashboard.
- Do not generate dashboard images, synthetic UI, or illustrative mockups.
- Describe the real screenshots; if a slide needs a figure, embed/refer to the uploaded PNG.
- Training slides should show or cite the captured screenshots, not AI-drawn interfaces.

