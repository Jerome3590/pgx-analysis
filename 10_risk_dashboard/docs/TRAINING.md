# Dashboard use-case training pack

Canonical workflows: [DASHBOARD_USE_CASES.md](DASHBOARD_USE_CASES.md). In-app user guide: **User Guide** tab on [https://pgx.jerome-dixon.io/](https://pgx.jerome-dixon.io/) (Training table + section **Unphased DNA files and star alleles**). That section describes the DuckDB Parquet coverage path for AncestryDNA, 23andMe, MyHeritage, and unphased VCF, and the clinical-test referral. Lab `Gene,Allele` lines are the path that still uses the phenotype table.

## Public artifacts folder

Publish video, slides, and audio (and any other finished training files) here:

https://drive.google.com/drive/folders/1cGbdoH-HDEooRyqWnhS6btWyhD4XOKQt?usp=drive_link

Formatted posting files live at the folder root as `{nn}_{use_case}.mp4` / `.pdf` / `.m4a`. The User Guide Training table links to those files.

## Google Drive layout (working pack)

Local Google Drive desktop sync path for screenshots and UC READMEs (this machine):

`G:\My Drive\PGx_Dashboard_Use_Cases\`

Repo mirror (same tree, for git): [use_case_training/](use_case_training/).

```
PGx_Dashboard_Use_Cases/
  README.md
  UC01_cohort_risk/README.md + screenshots/ + notebooklm/
  UC02_scenario_analysis/
  UC03_density_bin_exploration/
  UC04_feature_importance/
  UC05_pattern_process/
  UC06_claims_pgx_card/
  UC07_personalized_pgx_card/
  UC08_cohort_vs_card/
  UC09_documentation/
```

Each UC `README.md` is taken from DASHBOARD_USE_CASES.md (title, persona, numbered steps, live button names, screenshot index). Screenshots are captures of https://pgx.jerome-dixon.io/ — not generated mockups.

## NotebookLM

Studio outputs were **not** generated in this pass (Google login / IDE browser session blocked). Each UC `notebooklm/` folder is empty until you generate them.

Custom instruction for every notebook: [NOTEBOOKLM_PROMPTS.md](use_case_training/NOTEBOOKLM_PROMPTS.md) — use only the uploaded screenshots; do not generate dashboard images, synthetic UI, or illustrative mockups.

Next click: open https://notebooklm.google.com/ → **Create new notebook** → add that UC’s `README.md` + `screenshots/*.png` → paste the prompt → generate Audio / Video / Slides → download into the [public artifacts folder](https://drive.google.com/drive/folders/1cGbdoH-HDEooRyqWnhS6btWyhD4XOKQt?usp=drive_link) (and optionally `notebooklm/` in the working pack).
