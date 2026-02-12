# Structure and Workflow: 10_risk_dashboard vs 9_dashboard_visuals

## Yes: the risk dashboard needs the visualization artifacts

The **deployed** risk dashboard (frontend) loads BupaR, DTW, and FP-Growth assets from S3 (plots, chart_data.json, network HTML, etc.). Those artifacts are **produced by step 10** (dashboard visuals). So we **do** need the outputs of step 10 before the dashboard can show them.

**How we satisfy that:** We run **4_dashboard_visuals.ipynb (step 10)** **before** **5_build_and_deploy**. Step 10 uploads the visual artifacts to the dashboard bucket (e.g. `s3://.../vcu/pgx-risk-calculator/bupar/...`, `dtw/...`, `fpgrowth/...`). Then 5_build_and_deploy syncs the frontend; the frontend already points at those S3 URLs, so the deployed dashboard loads the prebuilt visuals. So the **execution order** is: generate visuals (step 10) → then build and deploy. The artifacts from data visualizations are therefore in place before the risk dashboard is deployed.

---

## Current layout

| Folder / concept | Role |
|------------------|------|
| **10_risk_dashboard/** | **Product**: Risk calculator + dashboard. Contains frontend, backend (Lambda), data_preparation (prepare_models, generate_metadata, CPIC), deployment, **and** the visualization **source code** (visualizations/bupar, dtw, fpgrowth). |
| **9_dashboard_visuals/** | **Pipeline step 10**: Orchestration for generating dashboard visuals. Contains README, test notebook, sync script, and `run_dashboard_visuals.py` that **calls into** 10_risk_dashboard/visualizations/ to run BupaR, DTW, and FP-Growth. |

So: **9** = the dashboard application and all its code (including visualization scripts). **10** = the pipeline phase that **runs** those visualization scripts and writes step checkpoints.

## Pipeline execution order (dependency-safe)

- **Step 9:** Risk dashboard **artifacts** — models, metadata, CPIC (scripts under 10_risk_dashboard/data_preparation). These feed the Lambda and frontend.
- **Step 10:** Dashboard **visuals** — BupaR, DTW, FP-Growth. Produces the plots and JSON that the frontend loads from S3. **Must run before deploy** so those URLs exist when the dashboard is deployed.

Notebook order: 3_model_train_shap_ffa (steps 4–8) → **4_dashboard_visuals (step 10)** → 5_build_and_deploy. So the visualization artifacts are created and uploaded **before** the frontend is synced to S3; the risk dashboard then has the visual assets it needs.

## Should we switch the order (9_dashboard_visuals ↔ 10_risk_dashboard)?

**Recommendation: no.** Keep **10_risk_dashboard** and **9_dashboard_visuals** as they are.

1. **10_risk_dashboard** is the product name used everywhere (README, deployment, docs). Renaming it to **10_risk_dashboard** would require a large, repo-wide change and would be confusing (“risk dashboard” is step 10 but the product is still the same).
2. **Execution order** is already correct: dashboard artifact prep (step 9) and visual generation (step 10) both feed build/deploy; step 10 is the last “content” step before 5_build_and_deploy.
3. **Conceptually:** 9 = the **module** that contains the app and the visualization code. 10 = the **pipeline step** that runs that code. Switching would make 10 the module and 9 the step, forcing 10_risk_dashboard → 10_risk_dashboard and 9_dashboard_visuals → 9_dashboard_visuals, with no real benefit and many broken references.

If you ever want **folder** order to match **step** order literally (9 = step 9, 10 = step 10), the current names already do: step 9 = risk dashboard artifacts (in 10_risk_dashboard), step 10 = dashboard visuals (in 9_dashboard_visuals).
