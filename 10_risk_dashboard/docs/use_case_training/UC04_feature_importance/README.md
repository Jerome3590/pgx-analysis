# UC04 — Feature Importance (population drivers)

**Live:** https://pgx.jerome-dixon.io/

**Persona:** Researcher.

**Live tabs:** Feature Importance

## Numbered steps

1. Open **Feature Importance**.
2. Set **View** to **Opioid ED**, **Polypharmacy**, or **Combined cohorts**.
3. Set **Show features** (Top 10 / Top 20 / All).
4. Click **Load Feature Importance Heatmap**.

Standalone population view (Step 3a Monte Carlo CV). It does not use the current patient’s selected codes.


## Live button names

- **View (Opioid ED / Polypharmacy / Combined cohorts)**
- **Show features (Top 10 / Top 20 / All)**
- **Load Feature Importance Heatmap**

## Screenshot index

PNGs in `screenshots/`. These are captures of the live dashboard. Do not replace them with generated mockups. Recaptured 2026-09-26 as **clipped sections** (`01` and `02` were identical; `03` was an unreadable full-page All-features dump).

- `01-feature-importance-tab.png` — default **View: Opioid ED**, **Show features: All**
- `02-view-and-top-n.png` — **View: Combined cohorts**, **Show features: Top 20**
- `03-heatmap-loaded.png` — **Load Feature Importance Heatmap** (Opioid ED × Top 20, age-band columns)

Canonical workflow text: repo `10_risk_dashboard/docs/DASHBOARD_USE_CASES.md`.

## NotebookLM prompt (required)

When creating Audio Overview, Video Overview, Slide Deck, or Briefing Doc:

- Use only the uploaded screenshots for any visual depiction of the PGx dashboard.
- Do not generate dashboard images, synthetic UI, or illustrative mockups.
- Describe the real screenshots; if a slide needs a figure, embed/refer to the uploaded PNG.
- Training slides should show or cite the captured screenshots, not AI-drawn interfaces.

