# Plan: Integrate Optuna into 3_model_train_shap_ffa (run_final_model.py)

This plan refactors the final-model pipeline so that **Optuna** selects both **model type** (CatBoost, XGBoost, XGBoost RF) and **hyperparameters** in a single study, optimizing **Recall and AUC-PR**. The notebook **3_model_train_shap_ffa** continues to drive the pipeline via Step 6 (`run_final_model.py`); no change to how the notebook invokes Step 6.

---

## Current state (reference)

- **Entry:** `6_final_model/run_final_model.py` is invoked per (cohort, age_band) from notebook 3 (Step 6 cell).
- **Flow:** Build features → `train_and_evaluate(df, cohort, age_band, n_runs)`:
  - **Data:** `train_and_evaluate` receives `df`; inside it builds `X`, `y`, `numeric_feature_cols`, `cat_feature_indices` (lines ~1460–1514).
  - **MCCV:** `n_runs` stratified 70/30 splits via `train_test_split(..., random_state=42 + run_idx)` (lines 1583–1587).
  - **Training:** For each split, train **XGBoost**, **XGBoost RF**, **CatBoost** with **fixed** params; collect AUC, PR-AUC, LogLoss, Recall (lines 1611–1755).
  - **Selection:** Best model by AUC-PR, tie-break Recall (lines 1911–1933).
  - **Final training:** Train selected model(s) on **full** `X`, `y` with fixed params; save artifacts (lines 2139–2488).
- **n_runs:** From `get_mc_cv_n_runs()` or `--n_runs` (e.g. 25).

---

## 1. Define the model search space (branching)

**Where:** New helper in `run_final_model.py` (or a small module used by it), used inside the Optuna objective.

**Design:**

- Single objective that first chooses model type:
  - `trial.suggest_categorical("model_type", ["xgb", "xgb_rf", "cat"])`.
- For each branch, suggest model-specific hyperparameters.

**Search spaces (concrete):**

| Branch    | Parameter           | Suggest call / notes |
|----------|----------------------|----------------------|
| **xgb**  | n_estimators         | `suggest_int("n_estimators", 200, 600)` |
|          | max_depth            | `suggest_int("max_depth", 4, 10)` |
|          | learning_rate        | `suggest_float("learning_rate", 0.02, 0.2, log=True)` |
|          | min_child_weight     | `suggest_int("min_child_weight", 1, 10)` |
|          | subsample            | `suggest_float("subsample", 0.6, 1.0)` |
|          | colsample_bytree     | `suggest_float("colsample_bytree", 0.6, 1.0)` |
|          | reg_lambda           | `suggest_float("reg_lambda", 1e-3, 10.0, log=True)` |
|          | gamma                | `suggest_float("gamma", 1e-8, 1.0, log=True)` |
| **xgb_rf** | Same as xgb where applicable; XGBRFClassifier uses `n_estimators`, `max_depth`, `subsample`, `colsample_bytree`. Optionally add `num_parallel_tree` (e.g. 2–8) if exposing RF-style parallelism. |
| **cat**  | iterations           | `suggest_int("iterations", 300, 800)` |
|          | learning_rate        | `suggest_float("learning_rate", 0.02, 0.2, log=True)` |
|          | depth                | `suggest_int("depth", 4, 10)` |
|          | l2_leaf_reg          | `suggest_float("l2_leaf_reg", 1.0, 10.0)` |
|          | border_count         | `suggest_int("border_count", 32, 255)` (optional) |
|          | bagging_temperature  | `suggest_float("bagging_temperature", 0.0, 1.0)` (optional) |

**Implementation:** Add a function `build_model_from_trial(trial, model_type, device, nthread, cat_feature_indices)` that, given `trial` and chosen `model_type`, reads the above params (with `trial.suggest_*` only for the active branch) and returns a configured XGBClassifier / XGBRFClassifier / CatBoostClassifier. Use a fixed `random_state`/`random_seed` for reproducibility.

**Reference:** [Optuna configurations](https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/002_configurations.html).

---

## 2. Wrap data and split logic for Optuna

**Where:** Refactor inside `train_and_evaluate()` in `run_final_model.py`.

**Current:** Data prep and MCCV loop are inline in `train_and_evaluate` (features → X,y → loop over `n_runs` splits → train three fixed models per split).

**Target:**

- **Data loading/preprocessing:** Keep as-is but **run once** at the start of `train_and_evaluate` (already the case: `X`, `y`, `numeric_feature_cols`, `cat_feature_indices` are computed once).
- **Split generator:** Extract a **reusable** MCCV split generator so Optuna can use the same splits:
  - e.g. `def generate_mc_splits(X, y, n_splits, test_size=0.3, random_state=1997):` yielding `(train_idx, test_idx)` or `(X_train, X_test, y_train, y_test)` for each split.
  - Use the same `random_state` so that “first 5 splits” are deterministic for HPO and “all 25 splits” for final evaluation.
- **Optuna objective:** Receives `(X, y, cat_feature_indices, split_generator)` (or indices generator) in closure/global so it does **not** reload data. Inside the objective, for each trial:
  - Call `trial.suggest_categorical("model_type", [...])` and `build_model_from_trial(...)`.
  - Loop over the **HPO subset** of splits (e.g. first `N_MCCV_HPO`), train the model, compute Recall and AUC-PR per split, then average.

**Concrete steps:**

1. Add `generate_mc_splits(X, y, n_splits, test_size=0.3, random_state=1997)` that yields `(X_train, X_test, y_train, y_test)` using `train_test_split(..., random_state=random_state + split_idx)`.
2. Add constant `N_MCCV_HPO = 5` (or 10) for Optuna trials.
3. In the Optuna objective, iterate only over the first `N_MCCV_HPO` splits from that generator, train the trial’s model, collect per-split Recall and AUC-PR, return a tuple of two values (see §3).

**Reference:** [K-fold CV with Optuna](https://discuss.pytorch.org/t/k-fold-cross-validation-with-optuna/182229) (same idea: reuse splits inside objective).

---

## 3. Optuna objective: multi-objective API (Recall + AUC-PR)

**Where:** Single `objective(trial)` function used by `study.optimize()` in `run_final_model.py`.

**Metrics:** For each MCCV split in the HPO loop, compute:

- `recall` = `recall_score(y_test, y_pred, zero_division=0)`
- `auprc` = `average_precision_score(y_test, y_proba)`

Then:

- `mean_recall` = mean over the `N_MCCV_HPO` splits  
- `mean_auprc` = mean over the `N_MCCV_HPO` splits  

**Multi-objective API:**

- Use Optuna's **multi-objective** API: the objective returns a **sequence of two values** in a fixed order.
- **Return value:** `return (mean_recall, mean_auprc)` — order must match `directions` (see §4).
- **Study:** `create_study(directions=["maximize", "maximize"])` so both Recall and AUC-PR are maximized.
- Optuna uses a multi-objective sampler (e.g. **NSGA-II** by default when `directions` is a list) and maintains a **Pareto front** of non-dominated trials.
- Optionally set `trial.set_user_attr("mean_recall", mean_recall)` and `mean_auprc` for logging.

**Reference:** [Optuna multi-objective tutorial](https://optuna.readthedocs.io/en/stable/tutorial/20_recipes/002_multi_objective.html).

---

## 3b. Selecting one trial from the Pareto front

After `study.optimize()` completes, `study.best_trial` is **not** defined for multi-objective studies; use **`study.best_trials`** (list of Pareto-optimal trials). Pick **one** trial to train the final model and export.

**Options (choose one and document in code):**

1. **Best AUC-PR on the front:**  
   `best_trial = max(study.best_trials, key=lambda t: t.values[1])`  
   (index 1 = second objective = AUC-PR). Favors discrimination.

2. **Best Recall on the front:**  
   `best_trial = max(study.best_trials, key=lambda t: t.values[0])`  
   (index 0 = Recall). Favors catching more positives.

3. **Knee / compromise:**  
   Pick the trial that maximizes a scalarized score over the front, e.g. `0.5 * recall + 0.5 * auprc`. Requires iterating over `study.best_trials`.

4. **Recall threshold then best AUC-PR:**  
   Among trials on the front with `t.values[0] >= RECALL_MIN`, take the one with largest `t.values[1]`. If none, fall back to best AUC-PR on the full front.

**Recommendation:** Option 4 (constraint then AUC-PR) keeps a clear clinical rule; default `RECALL_MIN = 0.7`. Alternatively option 1 (best AUC-PR on front) for simplicity.

**Implementation:** Add a helper, e.g. `select_best_trial_from_pareto(study, strategy="auprc" | "recall" | "recall_threshold", recall_min=0.7)`, returning the chosen trial.

---

## 4. Integrate Optuna into run_final_model.py

**Where:** Inside `train_and_evaluate()`, **replace** the current “train three fixed models on all n_runs splits then select by AUC-PR/Recall” block with:

1. **Study creation**
   - `study = optuna.create_study(direction="maximize")`.
   - Optionally: `sampler=optuna.samplers.TPESampler(seed=1997, n_startup_trials=10)`, `pruner=optuna.pruners.MedianPruner()` (if using iterative training that supports pruning; may require reporting intermediate values for XGB/CatBoost).

2. **Optimization**
   - `study.optimize(objective, n_trials=N_OPTUNA_TRIALS, timeout=... optional, show_progress_bar=True)`.
   - Choose `N_OPTUNA_TRIALS` (e.g. 50–100) based on runtime; keep `N_MCCV_HPO` small (5–10) so each trial is affordable.

3. **Best trial**
   - `best_trial = study.best_trial`.
   - `selected_model = best_trial.params["model_type"]` (map to `"xgb"` / `"xgb_rf"` / `"catboost"` as used elsewhere).
   - Best hyperparameters = `best_trial.params`.

4. **Final training**
   - **Option A (recommended):** Run **one final** MCCV with the **full** `n_runs` (e.g. 25) using the **best** model type and **best** hyperparameters only (no fixed three-model comparison). Compute and store the same metrics (Recall, AUC-PR, LogLoss) for reporting and for `selection_metadata`.
   - **Option B:** Skip full MCCV and train a single model on **full** `X`, `y` with best params; report that best trial’s `mean_recall` and `mean_auprc` as the “final” metrics (simpler but no variance estimate).

5. **Artifacts and metadata**
   - Save in `selection_metadata` (and optionally a separate `*_optuna_results.json`):
     - `optuna_best_params`, `optuna_best_value`, `optuna_best_recall`, `optuna_best_pr_auc`, `optuna_n_trials`, `optuna_model_type`.
   - Downstream “final training” (export JSON, .cbm, joblib, etc.) exports the Optuna-selected model and **always** the best XGBoost model for FFA (no longer train both XGB variants unless needed for FFA; see “Final model export” below).

**Final model export — always keep best XGBoost for FFA:** For formal feature attribution (FFA) analysis we **always** need to keep the best XGBoost model. After Optuna: (1) Export the Optuna-selected model (best trial from Pareto front) as the primary model for prediction and SHAP when applicable. (2) **Always** train and save the **best XGBoost** model (xgb or xgb_rf): if the selected model is xgb/xgb_rf, use that trial's params; otherwise train the better-performing XGB variant (e.g. with tuned or default params) so FFA has the required XGBoost JSON/artifact. Do not drop the XGBoost export when CatBoost wins. Optionally keep training CatBoost when the selected model is XGBoost, for SHAP consistency.

**Reference:** [Optuna efficient optimization](https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/003_efficient_optimization_algorithms.html).

---

## 5. MCCV design for HPO vs final evaluation

**Current:** 25 (or `n_runs`) MCCV splits; all three models trained per split.

**Target:**

- **HPO phase:** Use only **5–10** MCCV splits per Optuna trial to limit cost.
  - Introduce `N_MCCV_HPO = 5` (or 10) in `run_final_model.py`.
  - In `objective(trial)`, iterate only over the first `N_MCCV_HPO` splits from `generate_mc_splits(..., n_splits=n_runs, ...)` (or a dedicated HPO split count).
- **Final evaluation:** After Optuna completes, run your **existing 25-split (full `n_runs`) routine once** using the best config. Use the same split generator and seed so the 25 splits are identical to what you would have used for the old three-model comparison. Report mean (and optionally std) Recall, AUC-PR, LogLoss for metadata and comparison.
- **Reproducibility:** Same `random_state` for split generator so “first 5” and “all 25” are deterministic across runs.

**Concrete:**

- Add `N_MCCV_HPO = 5` and `N_OPTUNA_TRIALS = 50` (or from env/CLI).
- In the objective, call `generate_mc_splits(X, y, n_splits=N_MCCV_HPO, random_state=1997)` (or take first `N_MCCV_HPO` from a generator with `n_splits=n_runs`).

---

## 6. Code organization and reproducibility

**Modularize inside `run_final_model.py` (or a submodule):**

| Function / unit | Responsibility |
|-----------------|----------------|
| `get_data(df)` | From `df` (already built before `train_and_evaluate`), return `X`, `y`, `numeric_feature_cols`, `cat_feature_indices`. (Extract from current `train_and_evaluate` body.) |
| `generate_mc_splits(X, y, n_splits, test_size=0.3, random_state=1997)` | Yields `(X_train, X_test, y_train, y_test)` for each split. |
| `build_model_from_trial(trial, model_type, device, nthread, cat_feature_indices)` | Returns configured estimator for `model_type` using `trial.suggest_*` for that branch. |
| `evaluate_model_mccv(model, splits, cat_feature_indices?)` | Trains `model` on each split’s train, evaluates on each split’s test; returns `(mean_recall, mean_auprc)` (and optionally per-split lists). |
| `optuna_objective(trial)` | Closure over `X`, `y`, splits (or split generator), `cat_feature_indices`, device, nthread. Calls `build_model_from_trial`, then `evaluate_model_mccv` over HPO splits; applies constraint or scalarization; returns scalar and sets user attributes. |

**Logging:**

- Use `trial.set_user_attr("mean_recall", ...)` and `mean_auprc` for each trial.
- Optionally log `model_type` and key hyperparameters in your existing logger (e.g. fe_monitor) so runs are traceable.
- After optimization, log `study.best_trial.params` and `study.best_value`.

**Reproducibility:**

- Fix `random_state` / `random_seed` in: split generator (e.g. 1997), XGB/CatBoost (e.g. 1997), Optuna sampler (`TPESampler(seed=1997)`).
- Document in code or README that Optuna + MCCV are deterministic for a given `n_runs` and `N_MCCV_HPO`.

---

## 7. Notebook 3 and 3b

- **3_model_train_shap_ffa:** No change to **how** it runs Step 6 (it will still call `run_final_model.py --cohort --age_band`). The change is entirely inside `run_final_model.py`. Optionally add a short note in the notebook’s Step 6 markdown: “Step 6 uses Optuna to select model type and hyperparameters (Recall + AUC-PR).”
- **3b_optuna_shap_ffa:** Once Optuna is integrated into Step 6, 3b can be **simplified or deprecated**: either remove the Optuna cell and keep 3b as “SHAP + FFA only” (using the already-Optuna-tuned model from Step 6), or keep 3b as an optional “extra Optuna pass” with a different objective/n_trials for experimentation. Prefer simplifying 3b to “run SHAP + FFA after Step 6” and document that tuning is done in Step 6.

---

## 8. Implementation order

1. **Data and splits:** Extract `get_data()` and add `generate_mc_splits()`; add `N_MCCV_HPO` and wire a small test that runs 5 splits without Optuna.
2. **Model from trial:** Implement `build_model_from_trial()` for `xgb`, `xgb_rf`, `cat` with the chosen search spaces.
3. **Evaluate:** Implement `evaluate_model_mccv()` that takes one model and the HPO splits and returns mean Recall and mean AUC-PR.
4. **Objective:** Implement `optuna_objective()` with chosen metric strategy (constraint vs scalarized), calling the above.
5. **Study:** In `train_and_evaluate()`, replace the current “train 3 models × n_runs” block with `create_study` → `optimize` → read `best_trial` → run final MCCV (or single full fit) with best config.
6. **Final training and export:** Adapt the “Train final models on full data” section to use **best model type + best params** from Optuna; retain export paths expected by SHAP/FFA (e.g. best XGB and CatBoost if both are still required for downstream).
7. **Metadata and docs:** Write `optuna_*` fields into `model_selection_metadata.json`; update `MODEL_PERFORMANCE_SUMMARY.md` and `README_final_model.md` to describe Optuna-driven selection.
8. **Notebook 3:** Update Step 6 description if desired; simplify 3b as above.

---

## 9. Constants to add (suggested)

```text
N_MCCV_HPO = 5
N_OPTUNA_TRIALS = 50
RECALL_MIN = 0.7   # if using constraint; else unused
RANDOM_STATE = 1997
```

Optionally make these configurable via environment or `run_final_model.py` CLI (e.g. `--n_optuna_trials`, `--n_mccv_hpo`).

---

## 10. Files to touch

| File | Changes |
|------|--------|
| `6_final_model/run_final_model.py` | Main refactor: get_data, generate_mc_splits, build_model_from_trial, evaluate_model_mccv, optuna_objective; replace current MC loop + selection with Optuna study + final training with best config; extend selection_metadata. |
| `requirements.txt` | Already has `optuna`; ensure version >= 3.0. |
| `docs/Step6_FinalModel/README_final_model.md` | Describe Optuna-based model and HPO selection. |
| `8_ffa_analysis/results/MODEL_PERFORMANCE_SUMMARY.md` | Document Optuna search space, objective, and where results are stored. |
| `3_model_train_shap_ffa.ipynb` | Optional: one-sentence update to Step 6 markdown. |
| `3b_optuna_shap_ffa.ipynb` | Simplify to SHAP + FFA only, or document that it is an optional second Optuna pass. |

This plan keeps 3_model_train_shap_ffa as the single entry for “model train + SHAP + FFA” and moves all Optuna logic into `run_final_model.py` for clarity and reproducibility.
