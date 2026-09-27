# UC02 — Scenario Analysis (FFA/SHAP) — explain drivers

**Live:** https://pgx.jerome-dixon.io/

**Persona:** Researcher. Clinician may open this after a score to see drivers — it does not recalculate ensemble p̂.

**Live tabs:** Risk Assessment (context) · Scenario Analysis (FFA/SHAP)

## Numbered steps

This tab shows FFA interaction factors and SHAP importance. It does **not** recalculate the ensemble risk score.

1. Set cohort and **Age** on **Risk Assessment**, and select the codes you care about. A prior **Calculate Risk Score** is the intended context (Event density auto-syncs).
2. Open **Scenario Analysis (FFA/SHAP)**.
3. Optional: **What-if scenario** (comma-separated codes), **Show features** (Top 10 / Top 20 / All), **Event density**.
4. Click **Load Scenario Analysis**. Charts: **Top Interaction Factors (FFA)**, **SHAP Feature Importance**, **Effect on outcome (by feature)**.
5. **Clear filters** reloads with no code selection.

For modeled p̂, return to **Risk Assessment** and use **Calculate Risk Score**, **Replace**, or **Compare Scenarios**. This is the only visualization tab that filters charts by the codes you selected.


## Live button names

- **Load Scenario Analysis**
- **Clear filters**
- **What-if scenario**
- **Show features (Top 10 / Top 20 / All)**
- **Event density**

## Screenshot index

PNGs in `screenshots/`. These are captures of the live dashboard. Do not replace them with generated mockups. Recaptured 2026-09-26 as **clipped sections** (the first pass copied UC01 `07` as `01`, and `03`–`05` were the same full-page frame).

- `01-risk-context.png` — Risk Assessment score + Event Density badge after calculate (same clip as UC01 `04`)
- `02-scenario-tab.png` — **What-if scenario**, **Show features**, **Event density**, **Load Scenario Analysis**, **Clear filters**
- `03-load-scenario-analysis.png` — loaded FFA / SHAP / effect charts with a Current vs What-if trace
- `04-ffa-shap-charts.png` — the three chart panels
- `05-clear-filters.png` — controls after **Clear filters** (what-if emptied)

Canonical workflow text: repo `10_risk_dashboard/docs/DASHBOARD_USE_CASES.md`.

## NotebookLM prompt (required)

When creating Audio Overview, Video Overview, Slide Deck, or Briefing Doc:

- Use only the uploaded screenshots for any visual depiction of the PGx dashboard.
- Do not generate dashboard images, synthetic UI, or illustrative mockups.
- Describe the real screenshots; if a slide needs a figure, embed/refer to the uploaded PNG.
- Training slides should show or cite the captured screenshots, not AI-drawn interfaces.

