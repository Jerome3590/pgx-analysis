# Analysis Workflow (Alias)

This file is a **short pointer** to the canonical workflow documentation.

- For the full end‑to‑end analysis workflow (Steps 1–9; five notebooks 1→2→3→4→5), including:
  - **Steps 3a–3c:** Monte Carlo feature importance, BupaR/code research, then final update to features (3c) → refined `cohort_feature_importance.csv`
  - **Step 4:** Model data (`4_model_data/` → `model_events.parquet`)
  - **Step 5:** PGx feature engineering (`5_pgx_analysis/`)
  - **Step 6:** Final model training and selection (`6_final_model/`); train/test uploaded to S3 (required for SHAP/FFA)
  - **Steps 7–8:** SHAP analysis, then FFA (XGBoost only; uses SHAP to prioritize rules)
  - **Step 9:** Risk dashboard deployment

  see: **`docs/README_analysis_workflow.md`** (top-level workflow doc).

- For research‑question–to‑method mappings and cohort‑specific examples, see:
  - `docs/CrossStep_Workflow/README_research_questions_mapping.md`
  - `docs/archived/README_cross_ageband_analysis.md` (optional cross-age-band analysis; archived) 

