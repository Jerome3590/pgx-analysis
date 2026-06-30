"""Generate NIH AI Checklist domain coverage figure from evidence-map inputs.

Visualises how many eligible articles address each of the 12 NIH AI reporting
checklist domains, using nih_ai_tags already present in articles_screened.csv.
"""

import csv, collections, pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── paths ────────────────────────────────────────────────────────────────────
ROOT   = pathlib.Path(__file__).resolve().parents[2]
CSV    = ROOT / "inputs" / "ch01" / "articles_screened.csv"
OUTDIR = ROOT / "outputs" / "ch01" / "figures"
OUTDIR.mkdir(parents=True, exist_ok=True)

# ── domain display labels (same order as NIH_AI_ORDER in classifier) ─────────
DOMAIN_LABELS = {
    "study_design":               "Study Design &\nData Reporting",
    "model_transparency":         "Model Transparency\n& Reproducibility",
    "bias_fairness":              "Bias & Fairness\nAssessment",
    "performance_metrics":        "Performance\nMetrics",
    "explainability":             "Explainability &\nInterpretability",
    "external_validation":        "External /\nTemporal Validation",
    "uncertainty_quantification": "Uncertainty\nQuantification",
    "clinical_utility":           "Clinical Utility\nDemonstration",
    "deployment_implementation":  "Deployment &\nImplementation",
    "safety_monitoring":          "Safety &\nADR Monitoring",
    "regulatory_ethics":          "Regulatory &\nEthics Compliance",
}
# data_reporting sometimes appears in tags; merge with study_design bucket
TAG_MERGE = {"data_reporting": "study_design"}

YEARS = [str(y) for y in range(2019, 2026)]

# ── load ──────────────────────────────────────────────────────────────────────
rows    = list(csv.DictReader(open(CSV, encoding="utf-8-sig")))
included = [r for r in rows if r.get("human_decision", "").strip().lower() == "include"]
n_eligible = len(included)
print(f"Eligible: {n_eligible}")

domain_total = collections.Counter()
year_domain  = {yr: collections.Counter() for yr in YEARS}

for r in included:
    tags = r.get("nih_ai_tags", "").strip()
    if not tags:
        continue
    yr = r.get("pubdate", "")[:4]
    seen = set()
    for tag in tags.split("|"):
        tag = TAG_MERGE.get(tag.strip(), tag.strip())
        if tag in DOMAIN_LABELS and tag not in seen:
            seen.add(tag)
            domain_total[tag] += 1
            if yr in YEARS:
                year_domain[yr][tag] += 1

# ── sort by total (descending) ────────────────────────────────────────────────
DOMAIN_ORDER = sorted(DOMAIN_LABELS, key=lambda d: -domain_total[d])
labels       = [DOMAIN_LABELS[d] for d in DOMAIN_ORDER]
totals       = [domain_total[d] for d in DOMAIN_ORDER]
pcts         = [t / n_eligible * 100 for t in totals]

print("\nDomain coverage (sorted):")
for d, t, p in zip(DOMAIN_ORDER, totals, pcts):
    print(f"  {t:5d}  {p:5.1f}%  {d}")

# ── colour tiers: High ≥100, Medium 50-99, Low <50 ───────────────────────────
def tier_colour(t):
    if t >= 100: return "#1d4ed8"   # blue — strong coverage
    if t >= 50:  return "#d97706"   # amber — partial coverage
    return "#dc2626"                # red — gap

colours = [tier_colour(t) for t in totals]

# ── heatmap matrix ─────────────────────────────────────────────────────────────
matrix = np.array(
    [[year_domain[yr].get(d, 0) for d in DOMAIN_ORDER] for yr in YEARS],
    dtype=float,
)

# ── figure ────────────────────────────────────────────────────────────────────
fig, (ax_bar, ax_heat) = plt.subplots(
    1, 2, figsize=(14, 6),
    gridspec_kw={"width_ratios": [1, 1.6]},
)
fig.patch.set_facecolor("white")

# ── left: horizontal bar (count + % axis) ─────────────────────────────────────
y_pos = np.arange(len(DOMAIN_ORDER))
bars  = ax_bar.barh(y_pos, totals, color=colours, edgecolor="white", height=0.65)

for bar, val, pct in zip(bars, totals, pcts):
    ax_bar.text(
        bar.get_width() + max(totals) * 0.01,
        bar.get_y() + bar.get_height() / 2,
        f"{val}  ({pct:.1f}%)",
        va="center", ha="left", fontsize=8, color="#374151",
    )

ax_bar.set_yticks(y_pos)
ax_bar.set_yticklabels(labels, fontsize=8.5)
ax_bar.set_xlabel("Articles addressing domain (n)", fontsize=9)
ax_bar.set_title("Coverage across SQLR corpus (2015–2026)", fontsize=10,
                 fontweight="bold", pad=8)
ax_bar.set_xlim(0, max(totals) * 1.35)
ax_bar.spines[["top", "right"]].set_visible(False)
ax_bar.tick_params(axis="x", labelsize=8)

# legend tiers
legend_patches = [
    mpatches.Patch(color="#1d4ed8", label="Strong (≥100 articles)"),
    mpatches.Patch(color="#d97706", label="Partial (50–99 articles)"),
    mpatches.Patch(color="#dc2626", label="Gap (<50 articles)"),
]
ax_bar.legend(handles=legend_patches, fontsize=7.5, loc="lower right",
              framealpha=0.8, edgecolor="#d1d5db")

ax_bar.text(
    0.98, -0.10,
    f"% of {n_eligible:,} eligible articles",
    transform=ax_bar.transAxes, ha="right", fontsize=7,
    color="#6b7280", style="italic",
)

# ── right: heatmap ────────────────────────────────────────────────────────────
im = ax_heat.imshow(matrix, aspect="auto", cmap="YlOrBr",
                    vmin=0, vmax=matrix.max())
ax_heat.set_xticks(np.arange(len(DOMAIN_ORDER)))
ax_heat.set_xticklabels(labels, fontsize=7.5, rotation=40, ha="right")
ax_heat.set_yticks(np.arange(len(YEARS)))
ax_heat.set_yticklabels(YEARS, fontsize=9)
ax_heat.set_title("Annual domain coverage (2019–2025)", fontsize=10,
                  fontweight="bold", pad=8)

for i in range(len(YEARS)):
    for j in range(len(DOMAIN_ORDER)):
        val = int(matrix[i, j])
        txt = str(val) if val > 0 else "—"
        txt_col = "white" if val > matrix.max() * 0.6 else "#374151"
        ax_heat.text(j, i, txt, ha="center", va="center",
                     fontsize=7.5, color=txt_col)

cbar = fig.colorbar(im, ax=ax_heat, fraction=0.025, pad=0.03)
cbar.ax.tick_params(labelsize=7)
cbar.set_label("Article count", fontsize=8)

fig.suptitle(
    "NIH AI Reporting Checklist Domain Coverage in the SQLR Eligible Corpus\n"
    r"($n$ = 5,839 eligible articles; articles may address multiple domains)",
    fontsize=11, fontweight="bold", y=1.02,
)

plt.tight_layout()

out_pdf = OUTDIR / "fig_ml_methods.pdf"
out_png = OUTDIR / "fig_ml_methods.png"
fig.savefig(out_pdf, dpi=150, bbox_inches="tight")
fig.savefig(out_png, dpi=150, bbox_inches="tight")
print(f"\nSaved: {out_pdf}")
print(f"Saved: {out_png}")
plt.close()
