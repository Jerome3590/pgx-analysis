# Manuscript Next Steps

_Last updated: 2026-03-31 after new pipeline run (all notebooks re-run on EC2)._

---

## ✅ Completed This Session

| Item | File(s) |
|:-----|:--------|
| CH_3 ICI range corrected (0.163 → 0.164) | `CH_3/ch03_cts.qmd` lines 427, 513 |
| CH_3 abstract SHAP: `pgx_num_drugs` rank #1 / 1.22 | `CH_3/ch03_cts.qmd` line 66 |
| CH_3 Consensus-Causal section SHAP values updated | `CH_3/ch03_cts.qmd` lines 556, 576–586 |
| CH_3 oxycodone stale mean\|SHAP\| removed | `CH_3/ch03_cts.qmd` line 606 |
| CH_5 PGx coverage table added (`{#tbl-pgx-coverage}`) | `CH_5/ch05_bmic.qmd` lines 703–728 |
| METRICS.md model table: post-retrain + Brier/ICI columns | `METRICS.md` |
| METRICS.md checklist: 7 of 11 items confirmed ✅ | `METRICS.md` |

---

## ⏳ Run Now (PowerShell)

`.venv/` is **gitignored** (not in the repo clone). Create it locally at the project root if needed (see project `.cursorrules`), then activate—or call `.\.venv\Scripts\python.exe` directly so activation is optional.

```powershell
# From project root
cd C:\Projects\pgx-analysis
& .\.venv\Scripts\Activate.ps1   # omit if you use system Python or .venv\Scripts\python.exe

# Step 1 — regenerate figures for CH_3, CH_4, CH_5
# (UTF-8 avoids UnicodeEncodeError on Windows consoles when the script prints ✓)
$env:PYTHONIOENCODING = "utf-8"
python manuscript/scripts/generate_figures/generate_figures.py

# Step 2 — rebuild PDFs (changed chapters only)
cd manuscript
.\build.ps1 -Chapter 3
.\build.ps1 -Chapter 4
.\build.ps1 -Chapter 5
```

Output PDFs land in `manuscript/output/`.

---

## 🔲 Still Pending

### Manual / CloudWatch
- **CH_5 Lambda benchmark table** (`{#tbl-benchmarks}`) — pull fresh latency
  numbers from CloudWatch after `prepare_models.py` redeploy:
  - Cold-start mean/SD
  - Warm inference mean/SD
  - Risk, causal importance, visualization endpoint latencies

### After next pipeline run (FP-Growth)
- **CH_3 FP-Growth top rule** — `opioid_ed/25-44/low` returned 0 rules this run.
  Medium bin rules are respiratory (benzonatate/azithromycin), not opioid-relevant.
  High bin target-only rules show baclofen+prednisone+lamotrigine (lift=49) but need
  clinical narrative review before inserting.
  _Action_: re-run FP-Growth with lower support threshold OR accept that
  this band's FP-Growth adds no manuscript-level rule, and rely on the
  FFA pair (Feature 4: gabapentin ⊕ alprazolam) instead.

### Author metadata (manual, all chapters)
- **CRediT author contributions** — MDPI JPM and Wiley both require this field.
  Add to each chapter's YAML front-matter.
  Template:
  ```
  R.J.D.: Conceptualization, Methodology, Software, Formal Analysis,
  Writing – Original Draft. E.T.P.: Supervision, Writing – Review & Editing.
  ```

### PROSPERO (CH_1)
- Replace `[CRD-XXXXXX]` with `CRD420261354089` if not already in CH_1.

---

## 📁 Generated Data Files (manuscript/)

| File | Contents | Used In |
|:-----|:---------|:--------|
| `data/brier_ici_results.json` | Brier + ICI per cohort/band | CH_3, CH_4 |
| `data/ffa_ie_ci.json` | IE scores + 95% CI (top 5 DDI pairs) | CH_4 |
| `data/ffa_manuscript_data.json` | FFA rules, IR scores, top drugs | CH_4 |
| `shap_top_features.json` | SHAP top-10 per cohort/band/bin | CH_3 |
| `visual_manuscript_data.json` | FP-Growth + DTW + SHAP per cohort/band/bin | reference |
| `pgx_coverage.json` | PGx feature coverage % per cohort/band | CH_5 |
