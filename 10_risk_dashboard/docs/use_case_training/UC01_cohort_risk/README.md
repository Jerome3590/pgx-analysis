# UC01 — Cohort risk assessment with replace-and-compare

**Live:** https://pgx.jerome-dixon.io/

**Persona:** Clinician / pharmacist (primary). Patient / self-assessment can stop after Calculate Risk Score.

**Live tabs:** Opioid ED or Polypharmacy · Risk Assessment · Drugs · ICD Codes · CPT Codes (Opioid ED only)

## Numbered steps

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


## Live button names

- **Calculate Risk Score**
- **Edit codes**
- **Replace / swap a code → Replace**
- **chip ⇄**
- **Save current selection**
- **Compare Scenarios**
- **Include empty baseline**
- **Drug contributions**
- **View PGx Card →**

## Screenshot index

PNGs in `screenshots/`. These are captures of the live dashboard. Do not replace them with generated mockups. Recaptured 2026-09-26 as **clipped sections** (the first pass used full-page shots; `04`–`07` were the same Risk Assessment page).

- `01-cohort-and-age.png` — Opioid ED, Age 45, **Edit codes**, **Calculate Risk Score**
- `02-drugs-select.png` — oxycodone, hydrocodone, gabapentin chips
- `03-icd-cpt-codes.png` — ICD (age-band list; F11 is not in 45–54) + CPT 99214
- `04-calculate-risk.png` — ensemble p̂, HIGH band, **Event Density: LOW**, **View PGx Card →**
- `05-replace-swap-code.png` — **Replace / swap a code** + **Save current selection** (Baseline set A)
- `06-save-and-compare.png` — **Compare Scenarios** cards (empty baseline vs saved sets)
- `07-drug-contributions.png` — leave-one-out Δp̂ table

Canonical workflow text: repo `10_risk_dashboard/docs/DASHBOARD_USE_CASES.md`.

## NotebookLM prompt (required)

When creating Audio Overview, Video Overview, Slide Deck, or Briefing Doc:

- Use only the uploaded screenshots for any visual depiction of the PGx dashboard.
- Do not generate dashboard images, synthetic UI, or illustrative mockups.
- Describe the real screenshots; if a slide needs a figure, embed/refer to the uploaded PNG.
- Training slides should show or cite the captured screenshots, not AI-drawn interfaces.

