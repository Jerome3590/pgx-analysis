# NotebookLM — Drive import (no local file picker)

Stay signed in as **dixonrj@vcu.edu**. Do **not** use `*canallc*` / CANA.

Do **not** use **Upload files** / the computer file picker. That path hung. Import from **Google Drive** instead.

## Where the folder lives

| Account | Windows letter | Drive path |
|---|---|---|
| **dixonrj@vcu.edu** (VCU) | **`I:\`** | `I:\My Drive\PGx_Dashboard_Use_Cases\` |
| jerome.dixon90@gmail.com (personal) | `G:\` | `G:\My Drive\PGx_Dashboard_Use_Cases\` |

**NotebookLM as dixonrj@vcu.edu:** **Add source → Drive** opens **that account’s My Drive**. Choose **`PGx_Dashboard_Use_Cases`** on **VCU My Drive (`I:\`)**. Do **not** pick CANA (`J:`) and do **not** expect the Gmail `G:\` tree in the VCU picker.

Drive root in the picker: `My Drive` → `PGx_Dashboard_Use_Cases`

If the Drive picker hangs: add `README.md` first, then **2–3 screenshots at a time**. Never dump all PNGs in one pick.

## Mandatory Studio prompt (every Audio / Video / Slides / Briefing)

Paste the full text of that UC’s `NOTEBOOKLM_PROMPTS.md` (same text in every folder). Hard rules:

- Use **only** the Drive screenshots as visual source of truth.
- Do **not** generate dashboard images, mock UIs, or fictional screens.
- Slides should embed or cite the uploaded PNGs.

## Per-UC clicks

For each row: **New notebook** → name it → **Add sources** → **Drive** → open that folder on **VCU My Drive (`I:\`)** → select `README.md` + the listed PNGs (batch 2–3 if needed) → Insert → wait until sources finish indexing → **Studio** → paste the prompt → generate **Audio Overview**, **Video Overview**, **Slide Deck** → download into **both** `notebooklm\` folders.

If the current tab is stuck on “uploading files”: close that modal (**X** / Esc). If the spinner stays, leave that untitled notebook and start a **new** notebook for UC01.

| Notebook name | Drive folder | Add from Drive |
|---|---|---|
| PGx UC01 — Cohort risk | `PGx_Dashboard_Use_Cases/UC01_cohort_risk` | `README.md` + `screenshots/01-cohort-and-age.png` … `07-drug-contributions.png` (7 PNGs) |
| PGx UC02 — Scenario analysis | `PGx_Dashboard_Use_Cases/UC02_scenario_analysis` | `README.md` + `screenshots/01-risk-context.png` … `05-clear-filters.png` (5 PNGs) |
| PGx UC03 — Density bin exploration | `PGx_Dashboard_Use_Cases/UC03_density_bin_exploration` | `README.md` + `screenshots/01-event-density-badge.png` … `06-pgx-card-cohort-profile.png` (6 PNGs) |
| PGx UC04 — Feature importance | `PGx_Dashboard_Use_Cases/UC04_feature_importance` | `README.md` + `screenshots/01-feature-importance-tab.png` … `03-heatmap-loaded.png` (3 PNGs) |
| PGx UC05 — Pattern and process | `PGx_Dashboard_Use_Cases/UC05_pattern_process` | `README.md` + `screenshots/01-bupar.png` … `04-drug-networks.png` (4 PNGs) |
| PGx UC06 — Claims PGx card | `PGx_Dashboard_Use_Cases/UC06_claims_pgx_card` | `README.md` + `screenshots/01-view-pgx-card.png` … `03-radar-and-genes.png` (3 PNGs) |
| PGx UC07 — Personalized PGx card | `PGx_Dashboard_Use_Cases/UC07_personalized_pgx_card` | `README.md` + `screenshots/01-gene-data-entry.png` … `05-exports.png` (5 PNGs) |
| PGx UC08 — Cohort vs card | `PGx_Dashboard_Use_Cases/UC08_cohort_vs_card` | `README.md` + `screenshots/01-pgx-cohort-network.png` … `03-roles-compared.png` (3 PNGs) |
| PGx UC09 — Documentation | `PGx_Dashboard_Use_Cases/UC09_documentation` | `README.md` + `screenshots/01-documentation-how-to.png` … `05-unphased-dna.png` (5 PNGs) |

Optional: also add that folder’s `NOTEBOOKLM_PROMPTS.md` as a Drive source. Still paste it into Studio.

## Downloads

Save Audio / Video / Slides into:

- `I:\My Drive\PGx_Dashboard_Use_Cases\<UC>\notebooklm\` (VCU — NotebookLM account)
- `G:\My Drive\PGx_Dashboard_Use_Cases\<UC>\notebooklm\` (personal Gmail mirror)
- `C:\Projects\pgx-analysis\10_risk_dashboard\docs\use_case_training\<UC>\notebooklm\`

Then tell the agent to resume so it can copy any missing files between Drive and the repo mirror.
