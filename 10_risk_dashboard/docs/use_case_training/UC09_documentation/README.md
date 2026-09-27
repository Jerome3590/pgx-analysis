# UC09 — Documentation / model trust

**Live:** https://pgx.jerome-dixon.io/

**Persona:** All users. In-app Documentation is the trust surface for people who never open the repo.

**Live tabs:** Documentation

## Numbered steps

1. Open **Documentation**.
2. Read **How to use**, research-question coverage, and the compact use-case summary.
3. Review **Model performance and at-risk identification** (Monte Carlo 2016–2018 / 2019 holdout metrics by cohort and age band).
4. Review **Dashboard visual artifacts (from manifest)**.
5. Use **Feature importance sources for visuals** and **Event density bins** to interpret why BupaR/DTW vs FP-Growth can disagree, and why the badge matters for model routing.
6. Read **Unphased DNA files and star alleles** so VCF / 23andMe / AncestryDNA uploads are not mistaken for full haplotype calling.


## Live button names

- **Documentation (primary tab)**

## Screenshot index

PNGs in `screenshots/`. These are captures of the live dashboard. Do not replace them with generated mockups. Recaptured 2026-09-26 as **clipped sections** (the first pass used full-page shots; `03` and `04` were identical).

- `01-documentation-how-to.png` — **Overview** + **Tabs**
- `02-model-performance.png` — **Model performance and at-risk identification**
- `03-visual-artifacts.png` — **Dashboard visual artifacts (from manifest)**
- `04-density-bins.png` — **Feature importance sources for visuals** + **Event density bins**
- `05-unphased-dna.png` — **Unphased DNA files and star alleles**

Canonical workflow text: repo `10_risk_dashboard/docs/DASHBOARD_USE_CASES.md`.

## NotebookLM prompt (required)

When creating Audio Overview, Video Overview, Slide Deck, or Briefing Doc:

- Use only the uploaded screenshots for any visual depiction of the PGx dashboard.
- Do not generate dashboard images, synthetic UI, or illustrative mockups.
- Describe the real screenshots; if a slide needs a figure, embed/refer to the uploaded PNG.
- Training slides should show or cite the captured screenshots, not AI-drawn interfaces.

