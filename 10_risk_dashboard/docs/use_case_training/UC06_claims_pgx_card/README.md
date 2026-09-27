# UC06 — Claims-only PGx Card

**Live:** https://pgx.jerome-dixon.io/

**Persona:** Clinician / pharmacist. Patient may view the claims radar only.

**Live tabs:** PGx Card (after Risk Assessment score, or standalone cohort/age/bin)

## Numbered steps

Population / claims evidence for the patient’s cohort, age band, and density bin — not a genotype call.

1. After a score, click **View PGx Card →**, or open **PGx Card** and set **Cohort**, **Age Band**, and **Event Density**.
2. Click **Load Cohort PGx Profile**.
3. Read **Gene Actionability Profile** (radar) and **Identified PGx Genes**. The radar is **cohort/claims association**, not a CPIC prescribing action.

When you later generate a personalized card (use case 7):

- **Drug scope:** **Active medications** (`ACTIVE`), **Selected drugs** (`SELECTED`), or **All matched drugs** (`ALL_MATCHED`).
- **Actionable only** hides standard / no-CPIC rows in the queue, matrix, and exports.
- **Polypharmacy triplet engine** enumerates **regimen-only** three-way matches (need **≥ 3** APCD generics). Triplets are never shown as CPIC recommendations.


## Live button names

- **View PGx Card →**
- **Load Cohort PGx Profile**
- **Drug scope (Active medications / Selected drugs / All matched drugs)**
- **Actionable only**

## Screenshot index

PNGs in `screenshots/`. These are captures of the live dashboard. Do not replace them with generated mockups. Recaptured 2026-09-26 as **clipped sections** (`01` and `02` were identical full-page PGx Card dumps).

- `01-view-pgx-card.png` — **View PGx Card →** on Risk Assessment after a score
- `02-load-cohort-profile.png` — **Load Cohort PGx Profile** (cohort / age band / event density)
- `03-radar-and-genes.png` — Gene Actionability Profile radar + Identified PGx Genes

Canonical workflow text: repo `10_risk_dashboard/docs/DASHBOARD_USE_CASES.md`.

## NotebookLM prompt (required)

When creating Audio Overview, Video Overview, Slide Deck, or Briefing Doc:

- Use only the uploaded screenshots for any visual depiction of the PGx dashboard.
- Do not generate dashboard images, synthetic UI, or illustrative mockups.
- Describe the real screenshots; if a slide needs a figure, embed/refer to the uploaded PNG.
- Training slides should show or cite the captured screenshots, not AI-drawn interfaces.

