"""Generate operational-metrics bar chart + heatmap for ch01_bmic.qmd Figure 4."""

import csv, collections, pathlib, textwrap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── paths ────────────────────────────────────────────────────────────────────
BASE   = pathlib.Path(__file__).resolve().parents[1]
CSV    = BASE / "data" / "ontology" / "articles_screened.csv"
OUTDIR = pathlib.Path(r"c:\Projects\pgx-analysis\manuscript\figures\ch01")
OUTDIR.mkdir(parents=True, exist_ok=True)

# ── tag → category mapping (user-requested categories) ───────────────────────
CATEGORY_MAP = {
    "improved_patient_outcomes":   "Improved Patient Outcomes",
    "improved_outcomes":           "Improved Patient Outcomes",
    "improved_healthcare_outcomes":"Improved Patient Outcomes",
    "improved_process_performance":"Improved Patient Outcomes",
    "human_resources":             "Physician / Nurse Utilization",
    "cost":                        "Healthcare Cost Analysis",
    "process_throughput":          "Patient Throughput / Flow",
    "process_capacity":            "Hospital / System Capacity",
}

CATEGORY_ORDER = [
    "Improved Patient Outcomes",
    "Physician / Nurse Utilization",
    "Healthcare Cost Analysis",
    "Patient Throughput / Flow",
    "Hospital / System Capacity",
]

YEARS = [str(y) for y in range(2019, 2026)]

# ── load data ─────────────────────────────────────────────────────────────────
rows = list(csv.DictReader(open(CSV, encoding="utf-8-sig")))
included = [r for r in rows if r.get("human_decision","").strip().lower() == "include"]

# per-category totals and year×category matrix (deduplicate per article)
cat_total   = collections.Counter()
year_cat    = {yr: collections.Counter() for yr in YEARS}

for r in included:
    tags = r.get("op_perf_tags","").strip()
    if not tags:
        continue
    yr = r.get("pubdate","")[:4]
    cats_seen = set()
    for tag in tags.split("|"):
        tag = tag.strip()
        cat = CATEGORY_MAP.get(tag)
        if cat and cat not in cats_seen:
            cats_seen.add(cat)
            cat_total[cat] += 1
            if yr in YEARS:
                year_cat[yr][cat] += 1

print("Category totals:", dict(cat_total))
print("Year×category:")
for yr in YEARS:
    print(f"  {yr}: {dict(year_cat[yr])}")

# ── build heatmap matrix ──────────────────────────────────────────────────────
matrix = np.array([
    [year_cat[yr].get(cat, 0) for cat in CATEGORY_ORDER]
    for yr in YEARS
], dtype=float)

# ── figure layout: left bar chart, right heatmap ──────────────────────────────
fig, (ax_bar, ax_heat) = plt.subplots(
    1, 2,
    figsize=(12, 5),
    gridspec_kw={"width_ratios": [1, 2]},
)
fig.patch.set_facecolor("white")

# colour palette (one per category)
COLOURS = ["#2563eb", "#16a34a", "#ca8a04", "#dc2626", "#7c3aed"]

# ── left: horizontal bar chart (totals) ───────────────────────────────────────
totals = [cat_total.get(c, 0) for c in CATEGORY_ORDER]
short_labels = [
    "Improved\nPatient Outcomes",
    "Physician /\nNurse Utilization",
    "Healthcare\nCost Analysis",
    "Patient\nThroughput / Flow",
    "Hospital /\nSystem Capacity",
]
y_pos = np.arange(len(CATEGORY_ORDER))

bars = ax_bar.barh(y_pos, totals, color=COLOURS, edgecolor="white", height=0.6)
for bar, val in zip(bars, totals):
    ax_bar.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                str(val), va="center", ha="left", fontsize=9, color="#374151")

ax_bar.set_yticks(y_pos)
ax_bar.set_yticklabels(short_labels, fontsize=9)
ax_bar.set_xlabel("Number of eligible articles", fontsize=9)
ax_bar.set_title("Total counts (2015–2026)", fontsize=10, fontweight="bold", pad=8)
ax_bar.set_xlim(0, max(totals) * 1.18)
ax_bar.spines[["top","right"]].set_visible(False)
ax_bar.tick_params(axis="x", labelsize=8)
# annotation: total articles and % of corpus
n_eligible = len(included)
n_with_tag = sum(1 for r in included if r.get("op_perf_tags","").strip())
ax_bar.text(0.98, -0.14,
    f"Based on {n_with_tag:,} of {n_eligible:,} eligible articles ({n_with_tag/n_eligible*100:.1f}%) "
    f"with ≥1 operational tag",
    transform=ax_bar.transAxes, ha="right", fontsize=7, color="#6b7280", style="italic")

# ── right: heatmap (year × category) ─────────────────────────────────────────
im = ax_heat.imshow(matrix, aspect="auto", cmap="YlOrRd",
                    vmin=0, vmax=matrix.max())
ax_heat.set_xticks(np.arange(len(CATEGORY_ORDER)))
ax_heat.set_xticklabels(short_labels, fontsize=8, rotation=30, ha="right")
ax_heat.set_yticks(np.arange(len(YEARS)))
ax_heat.set_yticklabels(YEARS, fontsize=9)
ax_heat.set_title("Annual distribution by category (2019–2025)", fontsize=10,
                  fontweight="bold", pad=8)

for i, yr in enumerate(YEARS):
    for j, cat in enumerate(CATEGORY_ORDER):
        val = int(matrix[i, j])
        txt_col = "white" if val > matrix.max() * 0.6 else "#374151"
        ax_heat.text(j, i, str(val) if val > 0 else "—",
                     ha="center", va="center", fontsize=8, color=txt_col)

cbar = fig.colorbar(im, ax=ax_heat, fraction=0.03, pad=0.04)
cbar.ax.tick_params(labelsize=7)
cbar.set_label("Article count", fontsize=8)

fig.suptitle(
    "Operational Performance Metrics Addressed in the SQLR Corpus\n"
    r"($n$ = 5,839 eligible articles; articles may address multiple categories)",
    fontsize=11, fontweight="bold", y=1.02,
)

plt.tight_layout()

# save PDF + PNG
out_pdf = OUTDIR / "fig_op_metrics.pdf"
out_png = OUTDIR / "fig_op_metrics.png"
fig.savefig(out_pdf, dpi=150, bbox_inches="tight")
fig.savefig(out_png, dpi=150, bbox_inches="tight")
print(f"Saved: {out_pdf}")
print(f"Saved: {out_png}")
plt.close()
