# UC03 — Post-score density-bin exploration

**Live:** https://pgx.jerome-dixon.io/

**Persona:** Researcher (primary). Clinician uses the same Event Density badge to orient after a score.

**Live tabs:** Risk Assessment · BupaR Process Mining · DTW Trajectories · FP-Growth Patterns · PGx Cohort · PGx Card

## Numbered steps

After **Calculate Risk Score**, the **Event Density** badge is the same stratum the models and most visuals were built on.

1. Note the badge (low / medium / high / extreme).
2. Open **BupaR Process Mining** → **Load BupaR Visualizations** (Event density already set).
3. Open **DTW Trajectories** → **Load DTW Visualizations**.
4. Open **FP-Growth Patterns** → **Load FP-Growth Visualizations**.
5. Open **PGx Cohort** → **Load PGx Cohort Network** (and optionally **Load Figure Pack Visual**).
6. Open **PGx Card**: Event Density is auto-set → **Load Cohort PGx Profile**.

These tabs load **cohort × age × bin** (or cohort × age) artifacts. They are **not** filtered by the individual codes you selected, except Scenario Analysis (use case 2).


## Live button names

- **Calculate Risk Score**
- **Load BupaR Visualizations**
- **Load DTW Visualizations**
- **Load FP-Growth Visualizations**
- **Load PGx Cohort Network**
- **Load Figure Pack Visual**
- **Load Cohort PGx Profile**

## Screenshot index

PNGs in `screenshots/`. These are captures of the live dashboard. Do not replace them with generated mockups. Recaptured 2026-09-26 as **clipped sections** (`01` was a full-page copy of the contributions page).

- `01-event-density-badge.png` — Event density bin table (opioid_ed / 45–54) + HIGH / LOW badge
- `02-bupar-loaded.png` — **Load BupaR Visualizations** + activity-frequency chart
- `03-dtw-loaded.png` — **Load DTW Visualizations** + trajectory clusters
- `04-fpgrowth-loaded.png` — **Load FP-Growth Visualizations** + itemset support
- `05-pgx-cohort-loaded.png` — **Load PGx Cohort Network** + topology iframe
- `06-pgx-card-cohort-profile.png` — **Load Cohort PGx Profile** radar + identified genes

Canonical workflow text: repo `10_risk_dashboard/docs/DASHBOARD_USE_CASES.md`.

## NotebookLM prompt (required)

When creating Audio Overview, Video Overview, Slide Deck, or Briefing Doc:

- Use only the uploaded screenshots for any visual depiction of the PGx dashboard.
- Do not generate dashboard images, synthetic UI, or illustrative mockups.
- Describe the real screenshots; if a slide needs a figure, embed/refer to the uploaded PNG.
- Training slides should show or cite the captured screenshots, not AI-drawn interfaces.

