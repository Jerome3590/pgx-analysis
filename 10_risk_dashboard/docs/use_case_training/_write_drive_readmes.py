"""Write PGx Dashboard use-case README folders (Drive sync + repo mirror)."""
from pathlib import Path

ROOTS = [
    Path(r"G:\My Drive\PGx_Dashboard_Use_Cases"),
    Path(r"C:\Projects\pgx-analysis\10_risk_dashboard\docs\use_case_training"),
]

NOTEBOOKLM_RULES = """
## NotebookLM prompt (required)

When creating Audio Overview, Video Overview, Slide Deck, or Briefing Doc:

- Use only the uploaded screenshots for any visual depiction of the PGx dashboard.
- Do not generate dashboard images, synthetic UI, or illustrative mockups.
- Describe the real screenshots; if a slide needs a figure, embed/refer to the uploaded PNG.
- Training slides should show or cite the captured screenshots, not AI-drawn interfaces.
"""

INDEX = """# PGx Dashboard Use Cases — training pack

**Live:** https://pgx.jerome-dixon.io/

**Source of truth:** `10_risk_dashboard/docs/DASHBOARD_USE_CASES.md` in the pgx-analysis repo.

Each `UC##_*` folder has:

- `README.md` — title, persona, numbered live steps, button names, screenshot index
- `screenshots/` — captured PNGs from the live dashboard (not AI mockups)
- `notebooklm/` — Audio / Video / Slides downloads if NotebookLM generation succeeded

Do not treat Scenario Analysis as a second risk calculator. Do not invent CPIC prose.
""" + NOTEBOOKLM_RULES

UCS = {
    "UC01_cohort_risk": {
        "title": "UC01 — Cohort risk assessment with replace-and-compare",
        "persona": "Clinician / pharmacist (primary). Patient / self-assessment can stop after Calculate Risk Score.",
        "tabs": "Opioid ED or Polypharmacy · Risk Assessment · Drugs · ICD Codes · CPT Codes (Opioid ED only)",
        "buttons": [
            "Calculate Risk Score",
            "Edit codes",
            "Replace / swap a code → Replace",
            "chip ⇄",
            "Save current selection",
            "Compare Scenarios",
            "Include empty baseline",
            "Drug contributions",
            "View PGx Card →",
        ],
        "steps": """1. Choose **Opioid ED** or **Polypharmacy**.
2. On **Risk Assessment**, set **Age** (13–114).
3. Open **Drugs**, search, and select medications (Ctrl+click / Cmd+click). Use **Edit codes** to jump there from Risk Assessment.
4. For **Opioid ED** only, add codes on **ICD Codes** and **CPT Codes**.
5. Optional overrides: **# drugs**, **# CPIC drugs**, **# total events** (blank defaults to the low-density bin).
6. Click **Calculate Risk Score**. Read the score, band, model name, and **Event Density** badge (low / medium / high / extreme).
7. To swap one selected drug, ICD, or CPT: use **⇄** on a chip, or **Replace / swap a code** → **Replace**, then **Calculate Risk Score** again.
8. To compare modeled probabilities: name the set → **Save current selection**. Save **two or more** sets, then **Compare Scenarios**. With no saved sets, Compare is empty baseline vs current. Leave **Include empty baseline** checked to also show the 2019 population baseline.
9. Click **Drug contributions** for leave-one-out Δp̂ per selected drug (ICD/CPT stay in the vector). Positive Δp̂ means removing that drug lowers predicted risk.

**Compare Scenarios** scores every saved set you kept (two or more), not a single scenario. Deltas are versus the reference (empty baseline or the first saved set).
""",
        "shots": [
            "01-cohort-and-age.png",
            "02-drugs-select.png",
            "03-icd-cpt-codes.png",
            "04-calculate-risk.png",
            "05-replace-swap-code.png",
            "06-save-and-compare.png",
            "07-drug-contributions.png",
        ],
    },
    "UC02_scenario_analysis": {
        "title": "UC02 — Scenario Analysis (FFA/SHAP) — explain drivers",
        "persona": "Researcher. Clinician may open this after a score to see drivers — it does not recalculate ensemble p̂.",
        "tabs": "Risk Assessment (context) · Scenario Analysis (FFA/SHAP)",
        "buttons": [
            "Load Scenario Analysis",
            "Clear filters",
            "What-if scenario",
            "Show features (Top 10 / Top 20 / All)",
            "Event density",
        ],
        "steps": """This tab shows FFA interaction factors and SHAP importance. It does **not** recalculate the ensemble risk score.

1. Set cohort and **Age** on **Risk Assessment**, and select the codes you care about. A prior **Calculate Risk Score** is the intended context (Event density auto-syncs).
2. Open **Scenario Analysis (FFA/SHAP)**.
3. Optional: **What-if scenario** (comma-separated codes), **Show features** (Top 10 / Top 20 / All), **Event density**.
4. Click **Load Scenario Analysis**. Charts: **Top Interaction Factors (FFA)**, **SHAP Feature Importance**, **Effect on outcome (by feature)**.
5. **Clear filters** reloads with no code selection.

For modeled p̂, return to **Risk Assessment** and use **Calculate Risk Score**, **Replace**, or **Compare Scenarios**. This is the only visualization tab that filters charts by the codes you selected.
""",
        "shots": [
            "01-risk-context.png",
            "02-scenario-tab.png",
            "03-load-scenario-analysis.png",
            "04-ffa-shap-charts.png",
            "05-clear-filters.png",
        ],
    },
    "UC03_density_bin_exploration": {
        "title": "UC03 — Post-score density-bin exploration",
        "persona": "Researcher (primary). Clinician uses the same Event Density badge to orient after a score.",
        "tabs": "Risk Assessment · BupaR Process Mining · DTW Trajectories · FP-Growth Patterns · PGx Cohort · PGx Card",
        "buttons": [
            "Calculate Risk Score",
            "Load BupaR Visualizations",
            "Load DTW Visualizations",
            "Load FP-Growth Visualizations",
            "Load PGx Cohort Network",
            "Load Figure Pack Visual",
            "Load Cohort PGx Profile",
        ],
        "steps": """After **Calculate Risk Score**, the **Event Density** badge is the same stratum the models and most visuals were built on.

1. Note the badge (low / medium / high / extreme).
2. Open **BupaR Process Mining** → **Load BupaR Visualizations** (Event density already set).
3. Open **DTW Trajectories** → **Load DTW Visualizations**.
4. Open **FP-Growth Patterns** → **Load FP-Growth Visualizations**.
5. Open **PGx Cohort** → **Load PGx Cohort Network** (and optionally **Load Figure Pack Visual**).
6. Open **PGx Card**: Event Density is auto-set → **Load Cohort PGx Profile**.

These tabs load **cohort × age × bin** (or cohort × age) artifacts. They are **not** filtered by the individual codes you selected, except Scenario Analysis (use case 2).
""",
        "shots": [
            "01-event-density-badge.png",
            "02-bupar-loaded.png",
            "03-dtw-loaded.png",
            "04-fpgrowth-loaded.png",
            "05-pgx-cohort-loaded.png",
            "06-pgx-card-cohort-profile.png",
        ],
    },
    "UC04_feature_importance": {
        "title": "UC04 — Feature Importance (population drivers)",
        "persona": "Researcher.",
        "tabs": "Feature Importance",
        "buttons": [
            "View (Opioid ED / Polypharmacy / Combined cohorts)",
            "Show features (Top 10 / Top 20 / All)",
            "Load Feature Importance Heatmap",
        ],
        "steps": """1. Open **Feature Importance**.
2. Set **View** to **Opioid ED**, **Polypharmacy**, or **Combined cohorts**.
3. Set **Show features** (Top 10 / Top 20 / All).
4. Click **Load Feature Importance Heatmap**.

Standalone population view (Step 3a Monte Carlo CV). It does not use the current patient’s selected codes.
""",
        "shots": [
            "01-feature-importance-tab.png",
            "02-view-and-top-n.png",
            "03-heatmap-loaded.png",
        ],
    },
    "UC05_pattern_process": {
        "title": "UC05 — Pattern and process tabs including Drug Networks",
        "persona": "Researcher.",
        "tabs": "BupaR Process Mining · DTW Trajectories · FP-Growth Patterns · Drug Networks",
        "buttons": [
            "Load BupaR Visualizations",
            "Load DTW Visualizations",
            "Load FP-Growth Visualizations",
            "Load Drug Network",
        ],
        "steps": """| Tab | Load button | What you get |
|-----|-------------|--------------|
| **BupaR Process Mining** | **Load BupaR Visualizations** | Sequences, activity frequency, Gantt / trace explorer; Event density control. |
| **DTW Trajectories** | **Load DTW Visualizations** | Trajectory clusters, routine vs utilization, time-to-target; Event density control. |
| **FP-Growth Patterns** | **Load FP-Growth Visualizations** | Drug itemsets and association network; Event density control. |
| **Drug Networks** | **Load Drug Network** | Interactive Drug Networks (Cytoscape) graph of **FI-filtered** FP-Growth rules (not Plotly). Cohort × age only — no Event density dropdown. |

None of these four tabs recompute ensemble p̂. None filter by the codes selected on Drugs / ICD / CPT.
""",
        "shots": [
            "01-bupar.png",
            "02-dtw.png",
            "03-fpgrowth.png",
            "04-drug-networks.png",
        ],
    },
    "UC06_claims_pgx_card": {
        "title": "UC06 — Claims-only PGx Card",
        "persona": "Clinician / pharmacist. Patient may view the claims radar only.",
        "tabs": "PGx Card (after Risk Assessment score, or standalone cohort/age/bin)",
        "buttons": [
            "View PGx Card →",
            "Load Cohort PGx Profile",
            "Drug scope (Active medications / Selected drugs / All matched drugs)",
            "Actionable only",
        ],
        "steps": """Population / claims evidence for the patient’s cohort, age band, and density bin — not a genotype call.

1. After a score, click **View PGx Card →**, or open **PGx Card** and set **Cohort**, **Age Band**, and **Event Density**.
2. Click **Load Cohort PGx Profile**.
3. Read **Gene Actionability Profile** (radar) and **Identified PGx Genes**. The radar is **cohort/claims association**, not a CPIC prescribing action.

When you later generate a personalized card (use case 7):

- **Drug scope:** **Active medications** (`ACTIVE`), **Selected drugs** (`SELECTED`), or **All matched drugs** (`ALL_MATCHED`).
- **Actionable only** hides standard / no-CPIC rows in the queue, matrix, and exports.
- **Polypharmacy triplet engine** enumerates **regimen-only** three-way matches (need **≥ 3** APCD generics). Triplets are never shown as CPIC recommendations.
""",
        "shots": [
            "01-view-pgx-card.png",
            "02-load-cohort-profile.png",
            "03-radar-and-genes.png",
        ],
    },
    "UC07_personalized_pgx_card": {
        "title": "UC07 — Personalized PGx Card from DNA",
        "persona": "Clinician / pharmacist (Clinician / pharmacist view). Patient / self-assessment uses Patient view.",
        "tabs": "PGx Card",
        "buttons": [
            "Generate PGx results",
            "Or upload file",
            "Drug scope",
            "Actionable only",
            "View (Patient / Clinician / pharmacist)",
            "Print / Download JSON / CSV / PNG / PDF / Copy to clipboard / Technical appendix / Pharmacy handoff",
        ],
        "steps": """Optional refinement after (or instead of) the claims radar.

1. Under **1. Gene data**, leave **Session / patient pseudonym** blank for an anonymous session, or enter a label.
2. Enter **Gene / allele** lines (`CYP2D6,*1,*4`) **or** **Or upload file**: AncestryDNA or 23andMe (txt/csv/zip), Excel `.xlsx` (`Gene,Allele1,Allele2`), or VCF.
3. Files are parsed **in this browser**. They are not uploaded to S3. VCF is an **unphased rsid → star** map. It does **not** call full haplotypes.
4. Under **2. Virginia APCD medications**, search generics (two characters), set **Drug scope** and **Actionable only**, and choose **Patient** or **Clinician / pharmacist**.
5. Click **Generate PGx results**. The request is **POST `/pgx/card`** with **parsed variants** (plus selected APCD drugs and scope) — not the raw genome file.
6. Review **Medication-first action queue**, **Gene–drug actionability matrix**, **Polypharmacy triplet engine**, and **Genes tested**. Rows are **action categories** plus guideline URLs. Unlisted allele pairs stay **indeterminate**, never “normal.”
7. Export: **Print**, **Download JSON**, **Download CSV**, **Download PNG**, **Download PDF**, **Copy to clipboard**, **Technical appendix**, **Pharmacy handoff**.
8. **Send to pharmacy (coming soon)** stays disabled. There is no live e-prescribe.
""",
        "shots": [
            "01-gene-data-entry.png",
            "02-apcd-meds-and-scope.png",
            "03-generate-pgx-results.png",
            "04-action-queue-matrix.png",
            "05-exports.png",
        ],
    },
    "UC08_cohort_vs_card": {
        "title": "UC08 — PGx Cohort tab vs PGx Card",
        "persona": "Researcher (PGx Cohort). Clinician / pharmacist (PGx Card).",
        "tabs": "PGx Cohort · PGx Card",
        "buttons": [
            "Load PGx Cohort Network",
            "Load Figure Pack Visual",
            "Load Cohort PGx Profile",
            "Generate PGx results",
        ],
        "steps": """| | **PGx Cohort** | **PGx Card** |
|--|----------------|--------------|
| Role | Population topology | Patient / session card |
| Load | **Load PGx Cohort Network** | **Load Cohort PGx Profile** and/or **Generate PGx results** |
| Content | Gene–drug–phenotype network, figure pack, cohort radar | Claims radar (optional) + gene-data card, action queue, triplets, exports |
| Question | How do PGx genes, drugs, and phenotypes connect in this cohort × age band? | What should this session do given claims context and optional alleles? |

Use **PGx Cohort** for research topology. Use **PGx Card** when a person (or a claims proxy) needs an action list.
""",
        "shots": [
            "01-pgx-cohort-network.png",
            "02-pgx-card-patient.png",
            "03-roles-compared.png",
        ],
    },
    "UC09_documentation": {
        "title": "UC09 — Documentation / model trust",
        "persona": "All users. In-app Documentation is the trust surface for people who never open the repo.",
        "tabs": "Documentation",
        "buttons": ["Documentation (primary tab)"],
        "steps": """1. Open **Documentation**.
2. Read **How to use**, research-question coverage, and the compact use-case summary.
3. Review **Model performance and at-risk identification** (Monte Carlo 2016–2018 / 2019 holdout metrics by cohort and age band).
4. Review **Dashboard visual artifacts (from manifest)**.
5. Use **Feature importance sources for visuals** and **Event density bins** to interpret why BupaR/DTW vs FP-Growth can disagree, and why the badge matters for model routing.
""",
        "shots": [
            "01-documentation-how-to.png",
            "02-model-performance.png",
            "03-visual-artifacts.png",
            "04-density-bins.png",
        ],
    },
}


def render(uc_id: str, spec: dict) -> str:
    buttons = "\n".join(f"- **{b}**" for b in spec["buttons"])
    shots = "\n".join(f"- `{n}`" for n in spec["shots"])
    return f"""# {spec["title"]}

**Live:** https://pgx.jerome-dixon.io/

**Persona:** {spec["persona"]}

**Live tabs:** {spec["tabs"]}

## Numbered steps

{spec["steps"]}

## Live button names

{buttons}

## Screenshot index

PNGs in `screenshots/`. These are captures of the live dashboard. Do not replace them with generated mockups.

{shots}

Canonical workflow text: repo `10_risk_dashboard/docs/DASHBOARD_USE_CASES.md`.
{NOTEBOOKLM_RULES}
"""


def main() -> None:
    for root in ROOTS:
        root.mkdir(parents=True, exist_ok=True)
        (root / "README.md").write_text(INDEX, encoding="utf-8")
        for uc_id, spec in UCS.items():
            uc_dir = root / uc_id
            (uc_dir / "screenshots").mkdir(parents=True, exist_ok=True)
            (uc_dir / "notebooklm").mkdir(parents=True, exist_ok=True)
            (uc_dir / "README.md").write_text(render(uc_id, spec), encoding="utf-8")
        print(f"Wrote {root}")


if __name__ == "__main__":
    main()
