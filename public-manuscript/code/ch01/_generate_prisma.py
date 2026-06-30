"""
Generate PRISMA 2020 flowchart + counts CSV for the systematic literature review.

Outputs
-------
  outputs/ch01/figures/fig_prisma_flowchart.pdf   (publication-quality PDF)
  outputs/ch01/figures/fig_prisma_flowchart.png   (PNG)
  outputs/ch01/prisma_counts_current.csv          (all stage counts)

Usage
-----
  python code/ch01/_generate_prisma.py
  python code/ch01/_generate_prisma.py --no-figure
"""
import argparse, csv, json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT          = Path(__file__).resolve().parents[2]
INPUT_DIR     = ROOT / "inputs" / "ch01"
OUTPUT_DIR    = ROOT / "outputs" / "ch01"
SCREENED      = INPUT_DIR / "articles_screened.csv"
SCHOLAR_JSON  = INPUT_DIR / "scholar_json"
SCHOLAR_PDFS  = INPUT_DIR / "scholar_pdfs"
FIGURES_DIR   = OUTPUT_DIR / "figures"
OUT_CSV       = OUTPUT_DIR / "prisma_counts_current.csv"

SOURCE_CSVS = [
    "search_exports/blackbox_cds_articles.csv",
    "search_exports/apcd_analysis_articles.csv",
    "search_exports/pharmacovigilance_articles.csv",
    "search_exports/interpretability_articles.csv",
    "search_exports/fpgrowth_articles.csv",
    "search_exports/process_mining_articles.csv",
    "search_exports/catboost_xgboost_articles.csv",
    "search_exports/dtw_articles.csv",
    "search_exports/temporal_causality_articles.csv",
    "search_exports/target_leakage_articles.csv",
    "search_exports/opioid_disorder_articles.csv",
    "search_exports/polypharmacy_articles.csv",
    "search_exports/drug_interactions_articles.csv",
    "search_exports/duckdb_articles.csv",
    "search_exports/pgx_risk_classification_articles.csv",
    "search_exports/risk_model_ehr_articles.csv",
    "search_exports/fhir_ehr_articles.csv",
]
SEARCH_LABELS = [
    "Black-Box ML + CDS", "APCD Analysis", "Pharmacovigilance",
    "Interpretability / SHAP", "FP-Growth / Association Rules",
    "Process Mining (BupaR)", "CatBoost / XGBoost", "Dynamic Time Warping",
    "Temporal Causality", "Target Leakage Prevention", "Opioid Use Disorder",
    "Polypharmacy", "Drug-Drug Interactions", "DuckDB / OLAP",
    "PGx Classification Models", "Risk Models (EHR/CDS)", "Risk Models (FHIR)",
]


# ── Compute PRISMA counts ─────────────────────────────────────────────────────
def compute_counts() -> dict:
    # Stage 1 — Identification: raw rows across all source CSVs
    raw_titles = []
    per_search = {}
    for path, label in zip(SOURCE_CSVS, SEARCH_LABELS):
        p = INPUT_DIR / path
        if not p.exists():
            continue
        rows = list(csv.DictReader(open(p, encoding="utf-8-sig")))
        titles = [r.get("title", "").strip().lower() for r in rows if r.get("title","").strip()]
        raw_titles.extend(titles)
        per_search[label] = len(rows)

    n_identified = len(raw_titles)

    # Stage 2 — Deduplication (by normalised title)
    seen, dedup_titles = set(), []
    for t in raw_titles:
        if t not in seen:
            seen.add(t)
            dedup_titles.append(t)
    n_duplicates = n_identified - len(dedup_titles)
    n_after_dedup = len(dedup_titles)

    # Stage 3 — Screened (articles_screened.csv)
    screened_rows = list(csv.DictReader(open(SCREENED, encoding="utf-8-sig")))
    n_screened    = len(screened_rows)

    n_excluded_screen  = sum(1 for r in screened_rows if r.get("human_decision") == "exclude")
    n_included_screen  = sum(1 for r in screened_rows if r.get("human_decision") == "include")

    # Stage 4 — Full-text retrieval
    json_stems = {p.stem for p in SCHOLAR_JSON.glob("*.json")}
    pdf_stems  = {p.stem for p in SCHOLAR_PDFS.glob("*.pdf")}
    all_ft     = json_stems | pdf_stems

    included_rows = [r for r in screened_rows if r.get("human_decision") == "include"]
    n_ft_retrieved     = sum(1 for r in included_rows if r.get("pmc_id","") in all_ft)
    n_ft_not_retrieved = n_included_screen - n_ft_retrieved

    # Pending manual full-text review — count as included for now
    n_included_final   = n_ft_retrieved

    return {
        "n_identified":        n_identified,
        "n_duplicates":        n_duplicates,
        "n_after_dedup":       n_after_dedup,
        "n_screened":          n_screened,
        "n_excluded_screen":   n_excluded_screen,
        "n_included_screen":   n_included_screen,
        "n_ft_retrieved":      n_ft_retrieved,
        "n_ft_not_retrieved":  n_ft_not_retrieved,
        "n_included_final":    n_included_final,
        "per_search":          per_search,
        "n_databases":         len([p for p in SOURCE_CSVS if (ROOT / p).exists()]),
    }


# ── PRISMA flowchart ──────────────────────────────────────────────────────────
def draw_box(ax, x, y, w, h, text, color="#D6EAF8", fontsize=8.5, bold_first=False):
    box = FancyBboxPatch(
        (x - w/2, y - h/2), w, h,
        boxstyle="round,pad=0.02",
        linewidth=0.8, edgecolor="#2C3E50",
        facecolor=color, zorder=3,
    )
    ax.add_patch(box)
    lines = text.split("\n")
    if bold_first and lines:
        ax.text(x, y + h/2 - 0.07, lines[0],
                ha="center", va="top", fontsize=fontsize,
                fontweight="bold", color="#1A252F", zorder=4,
                wrap=False)
        body = "\n".join(lines[1:])
        ax.text(x, y + h/2 - 0.19, body,
                ha="center", va="top", fontsize=fontsize - 0.5,
                color="#1A252F", zorder=4, linespacing=1.4)
    else:
        ax.text(x, y, text,
                ha="center", va="center", fontsize=fontsize,
                color="#1A252F", zorder=4, linespacing=1.4)


def draw_arrow(ax, x, y1, y2):
    ax.annotate("", xy=(x, y2 + 0.005), xytext=(x, y1 - 0.005),
                arrowprops=dict(arrowstyle="-|>", color="#2C3E50",
                                lw=0.9, mutation_scale=10), zorder=2)


def draw_side_arrow(ax, x_main, y, x_side):
    mid_x = (x_main + x_side) / 2
    ax.annotate("", xy=(x_side - 0.18, y), xytext=(x_main + 0.15, y),
                arrowprops=dict(arrowstyle="-|>", color="#2C3E50",
                                lw=0.8, mutation_scale=9), zorder=2)


def generate_figure(c: dict, out_pdf: Path, out_png: Path):
    fig, ax = plt.subplots(figsize=(7.5, 11))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    # ── colour scheme ──────────────────────────────────────────────────────────
    C_HEAD  = "#1B4F72"   # header bars
    C_MAIN  = "#D6EAF8"   # main flow boxes
    C_EXCL  = "#FDEDEC"   # exclusion boxes
    C_INCL  = "#D5F5E3"   # included box
    C_TITLE = "#F2F3F4"   # section label

    # ── column positions ───────────────────────────────────────────────────────
    X_MAIN = 0.38
    X_SIDE = 0.80
    BOX_W  = 0.44
    BOX_W2 = 0.33
    BOX_H  = 0.075

    # ── section headers (left strip) ──────────────────────────────────────────
    sections = [
        (0.94, 0.82, "IDENTIFICATION"),
        (0.94, 0.60, "SCREENING"),
        (0.94, 0.40, "ELIGIBILITY"),
        (0.94, 0.16, "INCLUDED"),
    ]
    for y, _, label in sections:
        pass  # drawn below with rotated text

    # ── Row y-positions ───────────────────────────────────────────────────────
    Y = {
        "identified":     0.90,
        "dedup":          0.79,
        "screened":       0.67,
        "eligible":       0.50,
        "ft_assessed":    0.33,
        "included":       0.13,
    }

    # Section header strips
    for y_start, y_end, label, y_text in [
        (0.84, 1.00, "IDENTIFICATION", 0.92),
        (0.61, 0.84, "SCREENING",      0.73),
        (0.24, 0.61, "ELIGIBILITY",    0.435),
        (0.04, 0.24, "INCLUDED",       0.14),
    ]:
        rect = mpatches.FancyBboxPatch(
            (0.01, y_start), 0.07, y_end - y_start,
            boxstyle="round,pad=0.005",
            linewidth=0.6, edgecolor=C_HEAD, facecolor=C_HEAD, zorder=1,
        )
        ax.add_patch(rect)
        ax.text(0.045, (y_start + y_end) / 2, label,
                ha="center", va="center", fontsize=7, fontweight="bold",
                color="white", rotation=90, zorder=2)

    # ── BOX 1: Identified ─────────────────────────────────────────────────────
    db_list = f"({c['n_databases']} PubMed search strategies)"
    draw_box(ax, X_MAIN, Y["identified"], BOX_W, BOX_H,
             f"Records identified from databases\n"
             f"(PubMed; n = {c['n_identified']:,})\n{db_list}",
             color=C_MAIN, bold_first=False)

    # ── BOX 1b: Duplicates removed (side box, same row) ───────────────────────
    draw_arrow(ax, X_MAIN, Y["identified"] - BOX_H/2, Y["dedup"] + BOX_H/2)

    draw_box(ax, X_MAIN, Y["dedup"], BOX_W, BOX_H,
             f"Records after duplicates removed\n(n = {c['n_after_dedup']:,})",
             color=C_MAIN)
    draw_box(ax, X_SIDE, Y["identified"] - BOX_H/2 - 0.02, BOX_W2, 0.065,
             f"Duplicate records removed\n(n = {c['n_duplicates']:,})",
             color=C_EXCL)
    # horizontal connector from dedup row level to side box
    ax.annotate("", xy=(X_SIDE - BOX_W2/2, Y["identified"] - BOX_H/2 - 0.02),
                xytext=(X_MAIN + BOX_W/2, Y["dedup"]),
                arrowprops=dict(arrowstyle="-|>", color="#2C3E50",
                                lw=0.8, mutation_scale=9), zorder=2)

    # ── BOX 2: Screened ───────────────────────────────────────────────────────
    draw_arrow(ax, X_MAIN, Y["dedup"] - BOX_H/2, Y["screened"] + BOX_H/2)
    draw_box(ax, X_MAIN, Y["screened"], BOX_W, BOX_H,
             f"Records screened\n(title & abstract; n = {c['n_screened']:,})",
             color=C_MAIN)
    draw_box(ax, X_SIDE, Y["screened"], BOX_W2, BOX_H,
             f"Records excluded\n"
             f"(algorithm + manual; n = {c['n_excluded_screen']:,})",
             color=C_EXCL)
    ax.annotate("", xy=(X_SIDE - BOX_W2/2, Y["screened"]),
                xytext=(X_MAIN + BOX_W/2, Y["screened"]),
                arrowprops=dict(arrowstyle="-|>", color="#2C3E50",
                                lw=0.8, mutation_scale=9), zorder=2)

    # ── BOX 3: Reports sought ─────────────────────────────────────────────────
    draw_arrow(ax, X_MAIN, Y["screened"] - BOX_H/2, Y["eligible"] + BOX_H/2)
    draw_box(ax, X_MAIN, Y["eligible"], BOX_W, BOX_H,
             f"Reports sought for retrieval\n(n = {c['n_included_screen']:,})",
             color=C_MAIN)
    draw_box(ax, X_SIDE, Y["eligible"], BOX_W2, BOX_H,
             f"Reports not retrieved\n(n = {c['n_ft_not_retrieved']:,})",
             color=C_EXCL)
    ax.annotate("", xy=(X_SIDE - BOX_W2/2, Y["eligible"]),
                xytext=(X_MAIN + BOX_W/2, Y["eligible"]),
                arrowprops=dict(arrowstyle="-|>", color="#2C3E50",
                                lw=0.8, mutation_scale=9), zorder=2)

    # ── BOX 4: Full-text assessed ─────────────────────────────────────────────
    draw_arrow(ax, X_MAIN, Y["eligible"] - BOX_H/2, Y["ft_assessed"] + BOX_H/2)
    draw_box(ax, X_MAIN, Y["ft_assessed"], BOX_W, BOX_H,
             f"Reports assessed for eligibility (full text)\n"
             f"(n = {c['n_ft_retrieved']:,})",
             color=C_MAIN)
    draw_box(ax, X_SIDE, Y["ft_assessed"], BOX_W2, BOX_H,
             f"Reports excluded\n(pending manual review;\n"
             f"n = 0 to date)",
             color=C_EXCL)
    ax.annotate("", xy=(X_SIDE - BOX_W2/2, Y["ft_assessed"]),
                xytext=(X_MAIN + BOX_W/2, Y["ft_assessed"]),
                arrowprops=dict(arrowstyle="-|>", color="#2C3E50",
                                lw=0.8, mutation_scale=9), zorder=2)

    # ── BOX 5: Included ───────────────────────────────────────────────────────
    draw_arrow(ax, X_MAIN, Y["ft_assessed"] - BOX_H/2, Y["included"] + BOX_H/2)
    draw_box(ax, X_MAIN, Y["included"], BOX_W, BOX_H,
             f"Studies included in qualitative synthesis\n"
             f"(n = {c['n_included_final']:,})",
             color=C_INCL, fontsize=9)

    # ── Title ─────────────────────────────────────────────────────────────────
    ax.text(0.50, 0.985,
            "PRISMA 2020 Flow Diagram — Systematic Literature Review",
            ha="center", va="top", fontsize=9.5, fontweight="bold",
            color=C_HEAD)
    ax.text(0.50, 0.968,
            "Pharmacogenomics, Opioid Risk Prediction, and Clinical Decision Support",
            ha="center", va="top", fontsize=8, color="#555555")

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, format="pdf", bbox_inches="tight", dpi=300)
    fig.savefig(out_png, format="png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved: {out_pdf}")
    print(f"Saved: {out_png}")


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-figure", action="store_true",
                        help="Output counts CSV only, skip figure generation")
    args = parser.parse_args()

    print("Computing PRISMA counts from current data...")
    c = compute_counts()

    print("\n── PRISMA Counts ─────────────────────────────────────────")
    print(f"  Identified (raw across {c['n_databases']} searches): {c['n_identified']:,}")
    print(f"  Duplicates removed                                 : {c['n_duplicates']:,}")
    print(f"  After deduplication                                : {c['n_after_dedup']:,}")
    print(f"  Screened (title/abstract + algorithm)              : {c['n_screened']:,}")
    print(f"  Excluded at screen                                 : {c['n_excluded_screen']:,}")
    print(f"  Eligible (human_decision=include)                  : {c['n_included_screen']:,}")
    print(f"  Full-text retrieved (JSON or PDF)                  : {c['n_ft_retrieved']:,}")
    print(f"  Full-text not retrieved                            : {c['n_ft_not_retrieved']:,}")
    print(f"  Included in synthesis                              : {c['n_included_final']:,}")

    print("\n── Per-search counts ─────────────────────────────────────")
    for label, n in sorted(c["per_search"].items(), key=lambda x: -x[1]):
        print(f"  {label:<40} {n:>5}")

    # Save CSV
    rows = [
        ("Identified (raw)",              c["n_identified"]),
        ("Duplicates removed",            c["n_duplicates"]),
        ("After deduplication",           c["n_after_dedup"]),
        ("Screened",                      c["n_screened"]),
        ("Excluded at screening",         c["n_excluded_screen"]),
        ("Eligible (include decision)",   c["n_included_screen"]),
        ("Full-text retrieved",           c["n_ft_retrieved"]),
        ("Full-text not retrieved",       c["n_ft_not_retrieved"]),
        ("Included in synthesis",         c["n_included_final"]),
    ]
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["stage", "n"])
        w.writerows(rows)
    print(f"\nCounts written: {OUT_CSV}")

    if not args.no_figure:
        out_pdf = FIGURES_DIR / "fig_prisma_flowchart.pdf"
        out_png = FIGURES_DIR / "fig_prisma_flowchart.png"
        print("\nGenerating PRISMA flowchart...")
        generate_figure(c, out_pdf, out_png)


if __name__ == "__main__":
    main()
