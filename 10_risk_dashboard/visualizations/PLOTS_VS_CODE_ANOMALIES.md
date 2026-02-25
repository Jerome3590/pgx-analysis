# Plots vs Code Anomaly Report

**Archived.** For current workflow and outputs see **`10_risk_dashboard/docs/README_visualization_plan.md`**. Lesson: after sync failures (WinError 5 on rename), run **`9_dashboard_visuals/cleanup_aws_temp_files.py`** to remove AWS CLI temp files (`*.png.[0-9A-Za-z]{6,}`). See `archived/dashboard_docs/README.md`.

---

Comparison of synced dashboard visuals (from jerome.dixon.io S3) to the code that generates them.

## 1. Expected outputs by generator

### BupaR — opioid_ed (`create_bupar_outputs_opioid_ed.R`)

Trace explorer is **pre target (F1120) only**; no post-F1120 or overall target trace explorer.

| File pattern | Required? | Notes |
|-------------|-----------|--------|
| `{cohort}_{age}_trace_explorer_pre_f1120.png` | Yes | When pre-F1120 eventlog has data |
| `{cohort}_{age}_trace_explorer_interactive.html` | Yes | Pre-F1120 only; when years_with_data present |
| `{cohort}_{age}_pre_f1120_activity_frequency.png` | Yes | |
| `{cohort}_{age}_performance_spectrum.png` | Optional | Requires psmineR |
| `{cohort}_{age}_frequency_map.png` | Optional | Requires processmapR::export_map |
| `{cohort}_{age}_overall_activity_frequency.png` | Yes | |
| `{cohort}_{age}_activity_frequency_interactive.html` | Yes | When years_with_data present |

**Removed:** `trace_explorer_post_f1120.png`, `trace_explorer.png` (overall target), entire post-F1120 block (no `post_f1120_activity_frequency.png` or post-F1120 CSVs).

**Expected count per cohort/age:** 6–7 PNG + 2 HTML (frequency_map and performance_spectrum can be skipped).

### BupaR — non_opioid_ed (`create_bupar_outputs_non_opioid_ed.R`)

Trace explorer is **pre target (HCG) only**; no overall target trace explorer.

| File pattern | Required? | Notes |
|-------------|-----------|--------|
| `{cohort}_{age}_trace_explorer_pre_hcg.png` | Conditional | Only when n_pre > 0 |
| `{cohort}_{age}_trace_explorer_interactive.html` | Conditional | Pre-HCG only; when n_pre > 0 and years_with_data |
| `{cohort}_{age}_performance_spectrum.png` | Optional | Requires psmineR |
| `{cohort}_{age}_frequency_map.png` | Optional | Requires export_map |
| `{cohort}_{age}_overall_activity_frequency.png` | Yes | When n_target > 0 |
| `{cohort}_{age}_activity_frequency_interactive.html` | Yes | When n_target > 0 and years_with_data |

**Removed:** `trace_explorer.png` (overall target).

**No** `*_pre_f1120_*` or `*_post_f1120_*` — non_opioid_ed uses pre-HCG only.

**Expected count per cohort/age:** 3–5 PNG + 2 HTML.

### DTW (`create_dtw_plots.py` + `create_dtw_visuals.py`)

- **Generated:** `dtw_trajectory_cluster_{1d|3d}_{cohort}_{age}.html` (and optional `.png` via kaleido).
- **API copies:** `dtw_trajectory_analysis_{cohort}_{age}.png`, `dtw_sample_trajectories_{cohort}_{age}.png` (copied from first cluster PNG).
- **Location:** `outputs/{cohort}/{age_band_fname}/plots/` and `chart_data.json` at parent.

### FP-Growth (`create_fpgrowth_visualizations.py`)

- **Patterns:** `{cohort}_{age}_drug_name_combined_top_itemsets.png`, `*_itemsets_interactive.html`, `*_target_rules_network.png`, `*_network_interactive.html`. Production pipeline produces **drug_name** only.
- **Location:** `outputs/{cohort}/{age_band_fname}/plots/`.

---

## 2. Anomalies found

### A. AWS CLI leftover temp files (fixed)

- **What:** After the first sync run (which hit WinError 5 on rename), AWS CLI left behind temp files named like `opioid_ed_0_12_trace_explorer.png.bcAbD772` or `*.html.CFBb4CCb`.
- **Where:** `10_risk_dashboard/visualizations/bupar/outputs/**/plots/`.
- **Count:** 106 files across all cohort/age_band.
- **Action:** Removed via `9_dashboard_visuals/cleanup_aws_temp_files.py` (pattern: `*.png.[0-9A-Za-z]{6,}` and `*.html.[0-9A-Za-z]{6,}`). Re-run that script after any sync that fails with WinError 5 on rename.

### B. frequency_map.png not in S3/synced set

- **Code:** Both opioid_ed and non_opioid_ed R scripts attempt to write `{cohort}_{age}_frequency_map.png` via `processmapR::export_map()`.
- **Reality:** If `export_map` is missing or fails, the script skips with `[skip] frequency_map`. None of the synced BupaR plot lists include `*_frequency_map.png`.
- **Conclusion:** Likely not generated in the pipeline (processmapR/export_map not available or failing). Dashboard and sync do not expect it; no code bug, but optional artifact is consistently absent.

### C. DTW and FP-Growth local outputs empty

- **Observation:** `10_risk_dashboard/visualizations/dtw/outputs/` and `10_risk_dashboard/visualizations/fpgrowth/outputs/` contain **0 files** after sync.
- **Possible reasons:**
  1. S3 prefix `vcu/pgx-risk-calculator/dtw/` or `.../fpgrowth/` has no objects (dashboard never populated or different path).
  2. Sync script created empty dirs and sync ran but found nothing to download.
- **Recommendation:** Confirm on S3 whether `s3://jerome-dixon.io/vcu/pgx-risk-calculator/dtw/` and `.../fpgrowth/` have objects; if not, either run DTW/FP-Growth pipelines and upload, or document that the live dashboard uses only BupaR for those slots.

### D. non_opioid_ed: trace_explorer_pre_hcg.png

- **Code:** Written only when `n_pre > 0L` (pre-HCG eventlog non-empty).
- **Sync:** Not checked per age band; if some age bands have no pre-HCG events, absence of this file is correct.

### E. Naming consistency

- **Code** uses `age_band_fname` with underscore (e.g. `0_12`). **S3** uses hyphen in key prefix (`0-12`). Sync script correctly maps `0-12` → `0_12` for local paths. No naming anomaly.

---

## 3. Summary

| Issue | Severity | Status |
|-------|----------|--------|
| AWS temp files in bupar/plots | Medium (clutter, possible confusion) | Cleaned |
| frequency_map.png never present | Low (optional in code) | Documented |
| DTW/FP-Growth local dirs empty | Medium (verify S3 vs pipeline) | Needs S3/pipeline check |
| BupaR opioid vs non_opioid file sets | — | Matches code (opioid has pre/post F1120; non_opioid has pre_hcg only) |

---

## 4. How to re-check

- **List BupaR plots (canonical only):**  
  `Get-ChildItem -Path "10_risk_dashboard\visualizations\bupar\outputs" -Recurse -File | Where-Object { $_.Name -match '\.(png|html)$' -and $_.Name -notmatch '\.(png|html)\.[0-9A-Za-z]+$' }`
- **Count per cohort/age:** Compare to expected 9–10 (opioid_ed) or 4–6 (non_opioid_ed) PNG + 2 HTML per folder (excluding temp files and optional frequency_map/performance_spectrum if skipped).
