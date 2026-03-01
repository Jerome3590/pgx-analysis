# DTW and BupaR triage plan

Plan to fix DTW/BupaR errors and align visuals with research questions (drug-only where needed, empty-state JSON, Plotly NaN fix, BupaR Trace Explorer Pre-Target JSON/Plotly).

---

## 1. Empty JSON with message for all visuals with no output

**Goal:** Every visual that can have “no output” should return a JSON payload with a `message` and `empty: true` so the dashboard can show a consistent message instead of 404 or broken UI.

| Visual / endpoint | Current behavior | Action |
|-------------------|------------------|--------|
| **DTW** (`/visualizations/dtw`) | Lambda returns 200 with `trajectory_overview_plot` (message/empty when missing). **Pipeline:** `create_dtw_plots.create_trajectory_cluster_plots()` now **always** writes `trajectory_overview_plot.json`—either full Plotly payload or `{"message": "...", "empty": true}` when the visual is not produced (missing features, no codes, etc.). Step 6 syncs it; static or API then returns 200 + JSON. |
| **BupaR** | Manifest lists static files; missing files → 404. API may return URLs only. | Ensure Lambda/backend returns 200 with payload that includes per-artifact `message`/`empty` when an object is missing on S3 (same pattern as DTW). Frontend shows message when `empty` + `message` present. |
| **FP-Growth** | Pipeline can write `empty_state.json`; frontend/API already use it. | Confirm all cohort/age paths get either real data or empty_state.json; document in manifest/README. |
| **Causal** | causal_data.json per cohort/age. | If pipeline does not produce for a cohort/age, write minimal JSON `{"message": "...", "empty": true}` to S3 (or API returns it when object missing). |
| **Feature importance** | Per-cohort and combined files. | When no FI for a cohort, ensure API or static fallback returns JSON with message/empty. |

**Implementation notes:**
- **Lambda:** For each visualization endpoint, when an S3 object is missing, set the corresponding payload key to `{"message": "...", "empty": true}` (already done for DTW trajectory_overview_plot; extend to chart_data/sequence_heatmap if needed, and to BupaR/FP-Growth/Causal where applicable).
- **Pipeline (DTW):** Implemented. `create_dtw_plots.py` writes `trajectory_overview_plot.json` on every run—full payload when the visual is produced, or empty-state JSON on all early-exit paths. Empty-state includes `message` (cohort/age + reason), `empty: true`, `cohort`, `age_band`, and `metrics` (e.g. `reason`: `plotly_unavailable` | `sklearn_unavailable` | `features_csv_missing` | `no_seq_pattern_str` | `no_code_counts` | `no_top_codes`; plus `dtw_rows`, `count_df_rows`, `n_axes_required`, `csv_path` where applicable). Dashboard displays message and optional metrics block. Production-ready.

---

## 2. Common Sequences Heatmap (Drug / ICD / CPT) — CPT single-entry filter

**Goal:** Heatmap continues to support Drug / ICD / CPT slices. For **CPT**, apply a **single-entry filter** (or cap) so that the number of CPT codes/records does not blow up size or frontend rendering.

**Current:** `_build_sequence_heatmap_data` in `create_dtw_visuals.py` builds `drug`, `icd`, `cpt` with `codes`, `positions`, `counts`. All codes are included.

**Action:**
- In `_build_sequence_heatmap_data`, for the **cpt** slice only: limit to a single entry (e.g. top 1 code by total count) or cap at N codes (e.g. 1–5) so response size and frontend heatmap stay bounded.
- Document in code/doc: “CPT heatmap limited to single (or top-N) code(s) due to record count size.”
- Frontend: no change required if payload is already in existing heatmap format; ensure activity-type selector still shows “CPT” and renders the limited CPT data.

**Files:** `9_dashboard_visuals/dtw/create_dtw_visuals.py` (`_build_sequence_heatmap_data`).

---

## 3. Target Pathway Patterns (drugs) — filter to drugs only

**Addressed by drug-only trajectories.** With trajectory construction and plots using drug-only data (see §4), target pathway patterns are derived from the same drug-only sequences. If a separate filter is needed in the chart data builder, add drug-only filter there as well.

**Current:** `_compute_target_pathway_patterns` in `create_dtw_visuals.py` aggregates all tokens from `seq_pattern_str` (DRUG:X, ICD:Y, CPT:Z) and returns top 8 codes.

**Action:**
- In `_compute_target_pathway_patterns`, filter tokens to **DRUG:** prefix only (same parsing as sequence_heatmap: `token.split(":", 1)`, keep only when prefix is DRUG).
- Build code_counts and code_prevalence from drug tokens only.
- Optionally set `x_label`/metadata to indicate “Drug codes only” so the dashboard label is clear.

**Files:** `9_dashboard_visuals/dtw/create_dtw_visuals.py` (`_compute_target_pathway_patterns`).

---

## 4. Trajectory Analysis Overview (drugs) and Sample Trajectories (drugs) — filter to drugs only

**Done.** DTW is **drug-only for both cohorts** at trajectory construction (`create_dtw_trajectories.py`); `seq_pattern_str` and `seq_pattern_monthly` contain only DRUG: tokens. `create_dtw_plots._code_counts_from_seq_pattern_str` counts only DRUG: tokens.

**Previous:** `create_dtw_plots.create_trajectory_cluster_plots` had used `_code_counts_from_seq_pattern_str` counting all tokens (DRUG, ICD, CPT). `_top_codes` then picks top N codes across all types; 3D uses top 3 codes (may mix drug/ICD/CPT), 1D uses top 1.

**Action:**
- **Option A (recommended):** In `create_dtw_plots.py`, add a filter so that when building `count_df` (or when selecting `code_cols`), only **DRUG:** tokens are used. That implies:
  - In `_code_counts_from_seq_pattern_str` (or a new helper), accept an optional `activity_type="drug"` and only count tokens with that prefix.
  - Pass `activity_type="drug"` from `create_trajectory_cluster_plots` so trajectory overview and sample both use drug-only code counts and clusters.
- **Option B:** Filter in `create_dtw_visuals` before calling `create_trajectory_cluster_plots`: e.g. set `dtw_df["seq_pattern_str"]` to a drug-only version of each sequence (strip ICD/CPT tokens). Then existing plot code stays as-is.
- Ensure **trajectory_overview_plot.json** and **simple_traces** use only drug codes so the frontend title “Trajectory Analysis Overview (drugs)” and “Sample Trajectories (drugs)” are accurate.

**Files:** `9_dashboard_visuals/dtw/create_dtw_plots.py` (`_code_counts_from_seq_pattern_str`, `create_trajectory_cluster_plots`), and optionally `create_dtw_visuals.py` if pre-filtering in the dataframe.

---

## 5. Plotly NaN errors in trajectory_overview_plot.json

**Error seen:**  
`Error: <g> attribute transform: Trailing garbage, "…0.9300000000001,NaN)"` and `translate(519.83,NaN)` — Plotly is receiving NaN in trace coordinates, which produces invalid SVG.

**Root cause:** In `create_dtw_plots.py`, `simple_traces` and the Plotly figure use `x_m`, `y_m`, `z_m` from the dataframe. If `count_df` has NaN (e.g. missing values in code columns or after year filter), `.tolist()` and Plotly get NaN and the browser renders invalid transforms.

**Action:**
- In `create_dtw_plots.py`, before building `simple_traces` and before calling `fig.add_trace`:
  - Replace NaN in numeric arrays with a finite value (e.g. `np.nan_to_num(x_m, nan=0.0)` or `pd.Series(x_m).fillna(0).values`).
  - When writing `simple_traces` to JSON, sanitize lists so no NaN is written (JSON does not have NaN; Python’s `json` module emits `NaN` which is invalid JSON; frontend then passes it to Plotly). Use a small helper that converts float('nan') to `None` or `0` in lists/dicts before `json.dump`.
- Apply the same sanitization in the 1D (polypharmacy) and 3D paths for `x`, `y`, `z` (and any other numeric fields used in layout/traces).

**Files:** `9_dashboard_visuals/dtw/create_dtw_plots.py` (all branches that build traces and the payload for `trajectory_overview_plot.json`).

---

## 6. BupaR: Trace Explorer Pre-Target (drugs) — use JSON + Plotly instead of PNG

**Goal:** The “Trace Explorer Pre-Target (drugs)” panel should use the **JSON + Plotly** pattern (like the main Trace Explorer) instead of the static PNG (`*_trace_explorer_pre_f1120.png` / `*_trace_explorer_pre_hcg.png`).

**Current:** Frontend uses `data.trace_explorer_pre_image` (PNG URL) for the panel `trace_explorer_pre` and renders an `<img>`. The pipeline produces `trace_explorer_plot.json` (single JSON for trace explorer) and separate PNGs for pre-target/post-target.

**Action:**
- **Backend / pipeline:** Confirm whether `trace_explorer_plot.json` already contains pre-target (and post-target) data or only “overall” traces. If R only writes one JSON with all trace data, frontend can filter or select “pre-target” view from that JSON. If R can write a dedicated `*_trace_explorer_pre_*_plot.json` (or include a `pre_target` key in the existing JSON), prefer that so the frontend has an explicit pre-target structure.
- **Frontend:** For the Trace Explorer Pre-Target panel:
  - Prefer loading `trace_explorer_plot.json` (or a pre-target-specific JSON if added).
  - If the JSON has a structure for pre-target (e.g. `pre_target` traces or a filter flag), render with Plotly (same pattern as main trace explorer).
  - Only fall back to `trace_explorer_pre_image` (PNG) when JSON is missing or invalid.
- Update manifest/static_files if a new pre-target JSON artifact is added.

**Files:**  
`9_dashboard_visuals/bupar/` (R/Python: ensure JSON exposes pre-target data); `10_risk_dashboard/frontend/index.html` (render Trace Explorer Pre-Target from JSON + Plotly, PNG fallback).

---

## 7. Research questions alignment (summary)

| Research question | Visual | Triage action |
|-------------------|--------|----------------|
| Drug-only trajectory patterns | Common Sequences Heatmap | Keep Drug/ICD/CPT; CPT limited to single (or top-N) entry. |
| Drug-only pathway to target | Target Pathway Patterns | Filter to drugs only (DRUG: tokens). |
| Drug-only cluster overview | Trajectory Analysis Overview | Use drug-only code counts for clustering and plot. |
| Drug-only sample trajectories | Sample Trajectories | Same as overview (drug-only in trajectory_overview_plot.json). |
| Pre-target drug sequences | Trace Explorer Pre-Target | Use JSON + Plotly instead of PNG. |
| No data / missing cohort-age | All visuals | Return JSON with message + empty; fix NaN in DTW Plotly. |

---

## 8. Suggested implementation order

1. **Plotly NaN fix** (create_dtw_plots.py) — unblocks rendering for existing data.
2. **Empty JSON / message** — confirm Lambda and (optionally) pipeline for all visuals; document.
3. **Target Pathway Patterns drugs-only** — small, clear change in create_dtw_visuals.
4. **Sequence heatmap CPT single-entry** — limit CPT in _build_sequence_heatmap_data.
5. **Trajectory Overview + Sample drug-only** — add drug filter in create_dtw_plots (and optionally create_dtw_visuals).
6. **BupaR Trace Explorer Pre-Target JSON/Plotly** — frontend + optional R/JSON shape.

---

## References

- DTW outputs and when they exist: `9_dashboard_visuals/dtw/README.md`
- Manifest and S3 paths: `10_risk_dashboard/docs/README_dashboard_visual_artifact_paths.md`
- Dashboard validation: `README_dashboard_validation.md` (project root)
- BupaR artifact list: `9_dashboard_visuals/bupar/README_bupaR.md`
