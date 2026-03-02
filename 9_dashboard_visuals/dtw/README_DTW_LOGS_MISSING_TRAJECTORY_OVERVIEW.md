# DTW logs: cohorts missing or empty trajectory_overview_plot.json

Summary of DTW logs under `logs/dtw_s3/` for cohorts that lack a **full** trajectory overview (missing file or empty-state only).

---

## 1. Cohorts with **empty data** (trajectory_overview_plot.json is written as empty-state)

For these, the DTW features CSV has **0 rows** (header only). `create_dtw_visuals` loads 0 patients; `create_trajectory_cluster_plots` is still called and returns after writing an empty-state `trajectory_overview_plot.json`. So the file exists but has `"empty": true` and a message.

| Cohort | Age band | Log (latest) | Log message |
|--------|----------|--------------|-------------|
| **non_opioid_ed** | **65-74** | `logs/dtw_s3/non_opioid_ed/65-74/create_dtw_visuals_non_opioid_ed_65_74_20260302_033516.log` | `Loaded 0 patients with 12 DTW features` → chart_data/sequence_heatmap/plots all empty-state; success line includes `plots: trajectory_overview_plot.json` |
| **non_opioid_ed** | **75-84** | `logs/dtw_s3/non_opioid_ed/75-84/create_dtw_visuals_non_opioid_ed_75_84_20260302_033516.log` | Same: `Loaded 0 patients with 12 DTW features`; `plots: trajectory_overview_plot.json` in success |
| opioid_ed | 25-44 | create_dtw_visuals_opioid_ed_25_44_20260301_155706.log | Loaded 0 patients |
| opioid_ed | 45-54 | create_dtw_visuals_opioid_ed_45_54_20260301_143738.log | Loaded 0 patients |
| opioid_ed | 55-64 | create_dtw_visuals_opioid_ed_55_64_20260302_025000.log | Loaded 0 patients |
| opioid_ed_extreme_density | 25-44, 45-54, 75-84 | various 20260301/20260302 | Loaded 0 patients |

**Cause for non_opioid_ed 65-74 and 75-84:** Upstream: `create_dtw_trajectories` wrote a placeholder CSV (0 rows) because the SHAP/FFA drug filter matched no events in model_events. The pipeline now has a fallback (retry with all drug events) in `create_dtw_trajectories.py`; re-run trajectories + features + visuals for those cohort/age_bands to get real data and full trajectory overview.

---

## 2. Cohorts where **plot step crashed** (trajectory_overview_plot.json not written)

When `create_trajectory_cluster_plots` raises, the success line does **not** include `plots: trajectory_overview_plot.json`, and the file may be missing under `plots/`.

### Error: `The truth value of a Series is ambiguous`

| Cohort | Age band | Log | Note |
|--------|----------|-----|------|
| **opioid_ed** | **75-84** | `create_dtw_visuals_opioid_ed_75_84_20260301_145535.log` | Loaded **8037 patients**; then `DTW trajectory cluster plots failed: The truth value of a Series is ambiguous. Use a.empty, a.bool(), a.item(), a.any() or a.all().` Success: only `['chart_data.json', 'sequence_heatmap.json']` — **no** trajectory_overview_plot.json |
| opioid_ed | 85-114 | create_dtw_visuals_opioid_ed_85_114_20260301_155710.log, 20260301_145523.log | Same error with data loaded |

**Cause:** Bug in `create_dtw_plots.py`: a pandas Series is used in a boolean context (`if series:` or similar). Needs a fix to use `.any()`, `.all()`, or `.empty` as appropriate.

### Error: `invalid syntax (create_dtw_plots.py, line 302)`

Seen in older logs (e.g. 20260225–20260227) for multiple cohorts (opioid_ed and non_opioid_ed, various age bands). Line 302 was in the year-mapping block; that syntax error has likely been fixed in current code.

---

## 3. Where to look for DTW logs

- **Local (EC2):** `logs/dtw_s3/{cohort}/{age_band}/create_dtw_visuals_{cohort}_{age_band_fname}_{timestamp}.log`
- **Step 5 DTW:** `9_dashboard_visuals/logs/5_dtw/create_dtw_visuals_{cohort}_{age_band_fname}_{timestamp}.log`
- **S3 (if mirrored):** `s3://pgx-repository/5_dtw_log/{cohort}/{age_band}/` (see `py_helpers/fe_monitor.py` and README_DTW_S3_CHECKPOINTS.md)

---

## 4. Recommended next steps

1. **non_opioid_ed 65-74 and 75-84:** Re-run `create_dtw_trajectories` (with fallback), then `create_dtw_features`, then `create_dtw_visuals` so the features CSV gets rows and the trajectory overview has real data.
2. **opioid_ed 75-84 (and 85-114 if missing):** Fix the “truth value of a Series is ambiguous” bug in `create_dtw_plots.py`, then re-run `create_dtw_visuals` for those cohort/age_bands so `trajectory_overview_plot.json` is written (and optionally ensure create_dtw_visuals writes an empty-state JSON when the plot step raises, so the file is never missing).
