"""Generate ML model family distribution figure for ch01_bmic.qmd Figure 4.

Keyword-searches title + key_phrases in the 5,839-article eligible corpus
to produce a horizontal bar chart (totals) + heatmap (2019-2025 annual trends).
"""

import csv, collections, pathlib, re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── paths ────────────────────────────────────────────────────────────────────
BASE   = pathlib.Path(__file__).resolve().parents[1]
CSV    = BASE / "data" / "ontology" / "articles_screened.csv"
OUTDIR = pathlib.Path(r"c:\Projects\pgx-analysis\manuscript\figures\ch01")
OUTDIR.mkdir(parents=True, exist_ok=True)

# ── ML method keyword groups ──────────────────────────────────────────────────
ML_METHODS: dict[str, list[str]] = {
    "Gradient Boosting\n(XGBoost / LightGBM)": [
        "xgboost", "lightgbm", "gradient boost", "gradient-boost", "gbm", "gbdt",
        "extreme gradient",
    ],
    "CatBoost": [
        "catboost", "cat boost",
    ],
    "Random Forest": [
        "random forest", "random forests",
    ],
    "Deep Learning /\nNeural Networks": [
        "deep learning", "neural network", "lstm", "transformer",
        "convolutional", "cnn", "rnn", "bert", "llm", "large language",
        "attention mechanism",
    ],
    "Logistic Regression": [
        "logistic regression", "logit model",
    ],
    "Support Vector\nMachine (SVM)": [
        "support vector", "svm", "svr", "svc",
    ],
    "SHAP / Explainability\nMethods": [
        "shap", "shapley", "lime ", "lime,", "formal feature attribution",
        "ffa", "explainable ai", "xai", "interpretab",
    ],
    "Process Mining /\nSequence Methods": [
        "process mining", "bupar", "fp-growth", "apriori", "association rule",
        "dtw", "dynamic time warp",
    ],
    "Naïve Bayes /\nLinear Models": [
        "naive bayes", "naïve bayes", "linear model", "lasso", "ridge regression",
        "elastic net",
    ],
    "Ensemble /\nStacking": [
        "ensemble", "stacking", "bagging", "boosting", "voting classifier",
        "meta-learner",
    ],
}

METHOD_ORDER = list(ML_METHODS.keys())
YEARS = [str(y) for y in range(2019, 2026)]

def _norm(text: str) -> str:
    return text.lower()

def _match(text: str, kws: list[str]) -> bool:
    t = _norm(text)
    return any(k in t for k in kws)

# ── load data ─────────────────────────────────────────────────────────────────
rows = list(csv.DictReader(open(CSV, encoding="utf-8-sig")))
included = [r for r in rows if r.get("human_decision", "").strip().lower() == "include"]
print(f"Eligible articles: {len(included)}")

method_total = collections.Counter()
year_method  = {yr: collections.Counter() for yr in YEARS}

for r in included:
    text = (r.get("title", "") or "") + " " + (r.get("key_phrases", "") or "")
    yr   = r.get("pubdate", "")[:4]
    for method, kws in ML_METHODS.items():
        if _match(text, kws):
            method_total[method] += 1
            if yr in YEARS:
                year_method[yr][method] += 1

print("\nMethod totals:")
for m in METHOD_ORDER:
    print("  %5d  %s" % (method_total[m], m.replace('\n', ' ')))

# ── build heatmap matrix ──────────────────────────────────────────────────────
matrix = np.array(
    [[year_method[yr].get(m, 0) for m in METHOD_ORDER] for yr in YEARS],
    dtype=float,
)

# ── colour palette ────────────────────────────────────────────────────────────
COLOURS = [
    "#1d4ed8", "#7c3aed", "#059669", "#dc2626", "#d97706",
    "#0891b2", "#be185d", "#65a30d", "#6b7280", "#374151",
]

# ── figure ────────────────────────────────────────────────────────────────────
fig, (ax_bar, ax_heat) = plt.subplots(
    1, 2, figsize=(13, 5.5),
    gridspec_kw={"width_ratios": [1, 1.8]},
)
fig.patch.set_facecolor("white")

# ── left: horizontal bar chart ────────────────────────────────────────────────
totals = [method_total[m] for m in METHOD_ORDER]
y_pos  = np.arange(len(METHOD_ORDER))

bars = ax_bar.barh(y_pos, totals, color=COLOURS, edgecolor="white", height=0.65)
for bar, val in zip(bars, totals):
    ax_bar.text(
        bar.get_width() + max(totals) * 0.01,
        bar.get_y() + bar.get_height() / 2,
        str(val), va="center", ha="left", fontsize=8.5, color="#374151",
    )

ax_bar.set_yticks(y_pos)
ax_bar.set_yticklabels(METHOD_ORDER, fontsize=8.5)
ax_bar.set_xlabel("Articles mentioning method", fontsize=9)
ax_bar.set_title("Total mentions (2015–2026)", fontsize=10, fontweight="bold", pad=8)
ax_bar.set_xlim(0, max(totals) * 1.2)
ax_bar.spines[["top", "right"]].set_visible(False)
ax_bar.tick_params(axis="x", labelsize=8)

n_eligible = len(included)
ax_bar.text(
    0.98, -0.12,
    f"Based on title + key-phrase search across {n_eligible:,} eligible articles",
    transform=ax_bar.transAxes, ha="right", fontsize=7, color="#6b7280", style="italic",
)

# ── right: heatmap ────────────────────────────────────────────────────────────
im = ax_heat.imshow(matrix, aspect="auto", cmap="Blues",
                    vmin=0, vmax=matrix.max())
ax_heat.set_xticks(np.arange(len(METHOD_ORDER)))
ax_heat.set_xticklabels(METHOD_ORDER, fontsize=7.5, rotation=35, ha="right")
ax_heat.set_yticks(np.arange(len(YEARS)))
ax_heat.set_yticklabels(YEARS, fontsize=9)
ax_heat.set_title("Annual mention volume (2019–2025)", fontsize=10,
                  fontweight="bold", pad=8)

for i in range(len(YEARS)):
    for j in range(len(METHOD_ORDER)):
        val = int(matrix[i, j])
        txt_col = "white" if val > matrix.max() * 0.55 else "#374151"
        ax_heat.text(j, i, str(val) if val > 0 else "—",
                     ha="center", va="center", fontsize=7.5, color=txt_col)

cbar = fig.colorbar(im, ax=ax_heat, fraction=0.025, pad=0.03)
cbar.ax.tick_params(labelsize=7)
cbar.set_label("Article count", fontsize=8)

fig.suptitle(
    "Machine Learning and Explainability Method Mentions in the SQLR Corpus\n"
    r"($n$ = 5,839 eligible articles; articles may mention multiple methods)",
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
