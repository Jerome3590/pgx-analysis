# NotebookLM — next click (IDE browser blocked)

The Cursor IDE browser could not attach a tab (`No browser tab available. Please navigate to a page first.`). No NotebookLM Studio outputs were generated. Do not invent audio/video/slides.

**Preferred account (already on this PC):** Chrome profile **Default** = Gmail (`jerome.dixon90` / `@gmail.com`).  
**Do not use:** Chrome **Profile 1** = CANA (`*canallc*`).  
Also acceptable if you must switch: **VCU** (Profile 4) or **mushinsolutions.com** (Profile 6).

A Chrome window was opened with `--profile-directory=Default` at https://notebooklm.google.com/

Sources (do not recapture screenshots):

- Drive: `G:\My Drive\PGx_Dashboard_Use_Cases\`
- Repo: `C:\Projects\pgx-analysis\10_risk_dashboard\docs\use_case_training\`

## Exact account picker clicks

1. In the Chrome window that just opened, look at the **top-right avatar**.
2. If you see a **Google account chooser** (list of emails):
   - **Do not click** any `*canallc*` / `jdixon@canallc.com` / CANA row.
   - **Click** `jerome.dixon90@gmail.com` (first choice). If that row is missing, click `jerome@mushinsolutions.com` or `dixonrj@vcu.edu`.
3. If you are already inside NotebookLM but the avatar shows **CANA / canallc**:
   - Click the **avatar** (top right).
   - Click **Switch account** (or the other listed account).
   - Click **`jerome.dixon90@gmail.com`**.
4. If you see **Sign in** instead of notebooks:
   - Click **Sign in**.
   - On the chooser, same rule: Gmail / mushinsolutions / VCU only — never canallc.
5. After the notebook list loads, continue below.

## Mandatory prompt (paste into every Studio generation)

Use only the uploaded screenshots as visual source of truth.  
Do NOT generate, invent, or restyle dashboard images, mock UIs, or fictional screens.  
Describe/cite the real PNGs; slides should embed or refer to uploaded screenshots.

Also paste the full text of `NOTEBOOKLM_PROMPTS.md` from that UC folder.

## Per-UC clicks (repeat 9 times)

For each row: **New notebook** → name it → **Add sources** (README + every PNG) → **Studio** → paste prompts → generate → download into **both** `notebooklm\` folders (Drive and repo).

| Notebook name | Folder | Upload these |
|---|---|---|
| PGx UC01 — Cohort risk | `UC01_cohort_risk` | `README.md` + `screenshots\01-cohort-and-age.png` … `07-drug-contributions.png` (7 PNGs) |
| PGx UC02 — Scenario analysis | `UC02_scenario_analysis` | `README.md` + `screenshots\01-risk-context.png` … `05-clear-filters.png` (5 PNGs) |
| PGx UC03 — Density bin exploration | `UC03_density_bin_exploration` | `README.md` + `screenshots\01-event-density-badge.png` … `06-pgx-card-cohort-profile.png` (6 PNGs) |
| PGx UC04 — Feature importance | `UC04_feature_importance` | `README.md` + `screenshots\01-feature-importance-tab.png` … `03-heatmap-loaded.png` (3 PNGs) |
| PGx UC05 — Pattern and process | `UC05_pattern_process` | `README.md` + `screenshots\01-bupar.png` … `04-drug-networks.png` (4 PNGs) |
| PGx UC06 — Claims PGx card | `UC06_claims_pgx_card` | `README.md` + `screenshots\01-view-pgx-card.png` … `03-radar-and-genes.png` (3 PNGs) |
| PGx UC07 — Personalized PGx card | `UC07_personalized_pgx_card` | `README.md` + `screenshots\01-gene-data-entry.png` … `05-exports.png` (5 PNGs) |
| PGx UC08 — Cohort vs card | `UC08_cohort_vs_card` | `README.md` + `screenshots\01-pgx-cohort-network.png` … `03-roles-compared.png` (3 PNGs) |
| PGx UC09 — Documentation | `UC09_documentation` | `README.md` + `screenshots\01-documentation-how-to.png` … `05-unphased-dna.png` (5 PNGs) |

Studio buttons (if present on that notebook): **Audio Overview**, **Video Overview**, **Slide Deck**, **Briefing Doc**. Paste the prompt **before** each generate.

Download destinations (create files, do not leave them only in the browser):

- `G:\My Drive\PGx_Dashboard_Use_Cases\<UC>\notebooklm\`
- `C:\Projects\pgx-analysis\10_risk_dashboard\docs\use_case_training\<UC>\notebooklm\`

Then tell the agent to resume: it will copy any missing files between Drive and the repo mirror.
