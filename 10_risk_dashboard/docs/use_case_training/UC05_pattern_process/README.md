# UC05 — Pattern and process tabs including Drug Networks

**Live:** https://pgx.jerome-dixon.io/

**Persona:** Researcher.

**Live tabs:** BupaR Process Mining · DTW Trajectories · FP-Growth Patterns · Drug Networks

## Numbered steps

| Tab | Load button | What you get |
|-----|-------------|--------------|
| **BupaR Process Mining** | **Load BupaR Visualizations** | Sequences, activity frequency, Gantt / trace explorer; Event density control. |
| **DTW Trajectories** | **Load DTW Visualizations** | Trajectory clusters, routine vs utilization, time-to-target; Event density control. |
| **FP-Growth Patterns** | **Load FP-Growth Visualizations** | Drug itemsets and association network; Event density control. |
| **Drug Networks** | **Load Drug Network** | Interactive Drug Networks (Cytoscape) graph of **FI-filtered** FP-Growth rules (not Plotly). Cohort × age only — no Event density dropdown. |

None of these four tabs recompute ensemble p̂. None filter by the codes selected on Drugs / ICD / CPT.


## Live button names

- **Load BupaR Visualizations**
- **Load DTW Visualizations**
- **Load FP-Growth Visualizations**
- **Load Drug Network**

## Screenshot index

PNGs in `screenshots/`. These are captures of the live dashboard. Do not replace them with generated mockups. Recaptured 2026-09-26 as **clipped sections**.

- `01-bupar.png` — **Load BupaR Visualizations** (same live clip as UC03 `02`)
- `02-dtw.png` — **Load DTW Visualizations**
- `03-fpgrowth.png` — **Load FP-Growth Visualizations**
- `04-drug-networks.png` — **Load Drug Network** (FI-filtered Cytoscape, opioid_ed / 45–54)

Canonical workflow text: repo `10_risk_dashboard/docs/DASHBOARD_USE_CASES.md`.

## NotebookLM prompt (required)

When creating Audio Overview, Video Overview, Slide Deck, or Briefing Doc:

- Use only the uploaded screenshots for any visual depiction of the PGx dashboard.
- Do not generate dashboard images, synthetic UI, or illustrative mockups.
- Describe the real screenshots; if a slide needs a figure, embed/refer to the uploaded PNG.
- Training slides should show or cite the captured screenshots, not AI-drawn interfaces.

