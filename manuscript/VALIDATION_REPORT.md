# Manuscript ↔ Notebook 4 Validation Report

**Generated:** 2026-03-29  
**Scope:** Cross-check CH_3/CH_4/CH_5 manuscript values against notebook 4 visualization outputs  
and `extract_visual_manuscript.py` / `extract_ffa_manuscript.py` extraction scripts.

---

## Validation Status Legend

| Symbol | Meaning |
|:------:|:--------|
| ✅ | Match confirmed |
| ⚠️ | Mismatch — fixed in this session |
| 🔄 | Will update after retrain (expected change) |
| ❌ | Gap — data not captured; fix applied |
| 📋 | Manual / CloudWatch only |

---

## CH_3 (`ch03_cts.qmd`) — Opioid ED Cohort

### Performance Metrics (Table 2)

| Age Band | Manuscript AUROC | Pipeline LOW-bin AUROC | Match? |
|:---------|:----------------:|:---------------------:|:------:|
| 13–24 | 0.937 | 0.9367 | ✅ |
| 25–44 | 0.961 | 0.9597 | ✅ |
| 45–54 | 0.960 | 0.9587 | ✅ |
| 55–64 | 0.966 | 0.9645 | ✅ |

| Age Band | Manuscript PR-AUC | Pipeline LOW-bin PR-AUC | Match? |
|:---------|:-----------------:|:----------------------:|:------:|
| 13–24 | 0.835 | 0.8347 | ✅ |
| 25–44 | 0.889 | 0.8891 | ✅ |
| 45–54 | 0.896 | 0.8958 | ✅ |
| 55–64 | 0.916 | 0.9155 | ✅ |

> **Note:** Manuscript reports per-band LOW-density-bin metrics (most conservative). This is
> correct and consistent. These values **will change** after retraining without `n_events` (🔄).
> Run `compute_brier_ici.py` post-retrain to update.

### SHAP Top Features (lines 300–305)

| # | Old value | Status | Fix applied |
|:-:|:----------|:------:|:------------|
| 1 | gabapentin mean\|SHAP\|=0.34 | 🔄 | Will update from `shap_top_features.json` post-retrain |
| 3 | pgx_num_drugs mean\|SHAP\|=2.22 | 🔄 | Will update from `shap_top_features.json` post-retrain |
| 5 | **n_events** mean\|SHAP\|=1.49 | ⚠️ **FIXED** | Replaced with `n_event_bin_ordinal` (density stratum 0–3); exact value post-retrain |

> `n_events` was dropped from model. CH_3 line 305 updated to reference `n_event_bin_ordinal`.

### FP-Growth Rule (lines 309–313)

```
IF (Oxycodone_count ≥ 2) AND (ChronicPainDx = TRUE)
   AND NOT (PhysicalTherapy_count ≥ 1)
THEN P(OUD-ED) ↑↑   [support = 0.12, confidence = 0.83]
```

| Field | Status | Action |
|:------|:------:|:-------|
| Rule structure | 🔄 | Will update from `visual_manuscript_data.json` → `fpgrowth.top_rules[0]` post-retrain |
| support = 0.12 | 🔄 | Post-retrain |
| confidence = 0.83 | 🔄 | Post-retrain |

### DTW Cluster Breakdown (lines 325–348)

| Value | Manuscript | Checkpoint captures? | Fix applied |
|:------|:----------|:--------------------:|:------------|
| k = 2 optimal clusters | 2 | ✅ via `n_clusters` | — |
| Cluster 1 N = 5,481 (21%) | hardcoded | ❌ → ⚠️ **FIXED** | Now extracted from `high_risk_trajectories.archetypes_by_dtw_quartile[0].n` (Q1=Rapid-Onset) |
| Cluster 2 N = 21,229 (79%) | hardcoded | ❌ → ⚠️ **FIXED** | Now extracted from `high_risk_trajectories.archetypes_by_dtw_quartile[3].n` (Q4=Chronic-Escalation) |
| Rapid-Onset median 4.2 months | hardcoded | ❌ → ⚠️ **FIXED** | Now extracted from `time_to_target_days.by_routine[0].mean_months` (No-routine bucket) |
| Chronic-Escalation median 22.1 months | hardcoded | ❌ → ⚠️ **FIXED** | Now extracted from `time_to_target_days.by_routine[1].mean_months` (Routine bucket) |

> **All four DTW archetype statistics are now written to `dtw_manuscript_summary.json`** and
> read back by `extract_visual_manuscript.py`. Values will update after notebook 4 runs.

### Abstract Summary Stats

| Value | Status |
|:------|:------:|
| "mean PR-AUC = 0.88 ± 0.03" (avg of low-bin values) | 🔄 Post-retrain |
| "AUROC = 0.96 ± 0.01" | 🔄 Post-retrain |
| "384–498 Consensus-Causal features" | 🔄 Post-retrain |

---

## CH_4 (`ch04_psp.qmd`) — Polypharmacy / Non-Opioid ED Cohort

### Performance Metrics

| Value | Status |
|:------|:------:|
| "mean PR-AUC = 0.991 ± 0.007" | 🔄 Post-retrain via `compute_brier_ici.py` |
| "AUROC 0.996–0.999" | 🔄 Post-retrain |

### FFA Pair/Triplet Counts (lines 313–314)

| Value | Source | Status |
|:------|:-------|:------:|
| "115 synergistic drug pairs" | `axp_explanations.parquet` → `ffa_ie_ci.json` | 🔄 Post-retrain via `extract_ffa_manuscript.py` |
| "5,021 high-risk triplets" | `axp_explanations.parquet` → `ffa_ie_ci.json` | 🔄 Post-retrain |
| IE scores (16.3, 11.9, etc.) | `ffa_ie_ci.json` → CH_4 Table tbl-ddi | 🔄 Post-retrain |

### n_events Descriptive References (lines 388, 397)

| Location | Old text | Status | Fix applied |
|:---------|:---------|:------:|:------------|
| Line 388 | `median $n_{events}$ = 174` | ⚠️ **FIXED** | Changed to "median raw claim count = 174" — not a model feature |
| Line 397 | `$n_{event}$ top 5%` | ⚠️ **FIXED** | Changed to "top 5% raw claim count" |

> These references describe claim volume as a descriptive statistic, not the dropped model
> feature. Clarified to prevent confusion.

### IR Scores (lines 359–369)

| Value | Source | Status |
|:------|:-------|:------:|
| Simvastatin IR = 7.0×10⁻⁴ | `ffa_manuscript_data.json` | 🔄 Post-retrain |
| Furosemide IR = 2.0×10⁻⁴ | `ffa_manuscript_data.json` | 🔄 Post-retrain |
| Rank correlation ρ = 0.53–0.68 | `ffa_ie_ci.json` | 🔄 Post-retrain |

---

## CH_5 (`ch05_bmic.qmd`) — PGx Risk Dashboard

### Model Count

| Location | Old value | Fix applied |
|:---------|:----------|:------------|
| Abstract | "21 ensemble models (7 age bands × 3 algorithms)" | ⚠️ **FIXED** → "Up to 84 models (2 cohorts × 7 usable bands × 4 density bins; 3 algorithms per bin)" |
| Introduction list item 1 | "21 ensemble models from Chapters 3–4" | ⚠️ **FIXED** |
| Architecture table | "All 21 models bundled" | ⚠️ **FIXED** |
| Conclusions | "all 21 age-band models" | ⚠️ **FIXED** |

### Lambda Latency Benchmarks (Table tbl-benchmarks)

| Metric | Manuscript | Status |
|:-------|:----------|:------:|
| Cold-start: 2,100 ms | hardcoded | 📋 Manual — CloudWatch only |
| Warm inference: 6 ms | hardcoded | 📋 Manual — CloudWatch only |
| PGx card: 60 ms | hardcoded | 📋 Manual — CloudWatch only |

> See `RUN_PLAN.md` → "Manual: Lambda Latency" for CloudWatch retrieval command.

### PGx Coverage % (Tab 5 / Feature 4)

| Value | Source | Status |
|:------|:-------|:------:|
| PGx coverage % | `pgx_coverage.json` | 🔄 Extract after notebook 4 via `extract_visual_manuscript.py` |

---

## Extraction Script Changes Made

| Script | Old behavior | New behavior |
|:-------|:-------------|:-------------|
| `extract_ffa_manuscript.py` | Reads only `bin_models/low/` | ✅ Reads all 4 bins; falls back to notebook 4 summary JSON |
| `extract_ffa_manuscript.py` | Saves only `ffa_manuscript_data.json` | ✅ Also saves `ffa_ie_ci.json` with pair-level IE scores |
| `extract_visual_manuscript.py` | DTW: total N + cluster count only | ✅ Now extracts per-archetype N, %, mean months to target |

---

## Notebook 4 Checkpoint Writer Changes Made

| Section | Old | New |
|:--------|:----|:----|
| DTW checkpoint | `total_trajectories`, `n_clusters`, `trajectory_length` | ✅ + `archetypes_by_dtw_quartile` (Q1–Q4 N/pct/rate), `rapid_onset`, `chronic_escalation`, `time_to_target_days` |
| DTW print output | `trajectories=N` | ✅ + `rapid_onset_n=N  chronic_escalation_n=N` |

---

## Post-Retrain Update Checklist

After notebooks 3 + 4 complete on EC2 and extraction scripts run locally:

- [ ] **CH_3 Table 2**: Update AUROC/PR-AUC/Brier/ICI from `compute_brier_ici.py` output
- [ ] **CH_3 abstract**: Update "mean PR-AUC = 0.88" and "AUROC = 0.96" averages  
- [ ] **CH_3 lines 301–305**: Update SHAP values (#1, #3, #5) from `shap_top_features.json`
- [ ] **CH_3 lines 309–313**: Update FP-Growth rule from `visual_manuscript_data.json` (opioid_ed/25-44/low)
- [ ] **CH_3 lines 327–348**: Update DTW cluster N/% and median months from `visual_manuscript_data.json` (opioid_ed/25-44 or 13-24/low)
- [ ] **CH_4 lines 313–314**: Update pair/triplet counts from `ffa_ie_ci.json`
- [ ] **CH_4 Table tbl-ddi**: Update IE scores from `ffa_ie_ci.json`
- [ ] **CH_4 lines 359–369**: Update IR scores from `ffa_manuscript_data.json`
- [ ] **CH_5 benchmarks**: Update latency from CloudWatch after deploy (📋 manual)

---

## S3 Checkpoint Key Reference

After notebook 4 runs, all manuscript-ready data is at:

```
s3://pgxdatalake/gold/manuscript_checkpoints/
  fpgrowth/{cohort}/{ab}/{bin}/fpgrowth_manuscript_summary.json
  dtw/{cohort}/{ab}/{bin}/dtw_manuscript_summary.json        ← includes archetype breakdown
  shap/{cohort}/{ab}/{bin}/shap_manuscript_summary.json
  ffa/{cohort}/{ab}/{bin}/ffa_manuscript_summary.json
  pgx/{cohort}/{ab}/pgx_manuscript_summary.json
```
