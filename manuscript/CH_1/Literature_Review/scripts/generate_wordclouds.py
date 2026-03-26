"""
generate_wordclouds.py
Generates word cloud visualisations for the literature review corpus.

Outputs (PNG + PDF) saved to data/wordclouds/:
  wordcloud_overall.png            — full 9,454-article corpus
  wordcloud_ooda_grid.png          — 2×2 grid by OODA phase
  wordcloud_ooda_observe.png       ┐
  wordcloud_ooda_orient.png        │ individual OODA phases
  wordcloud_ooda_decide.png        │
  wordcloud_ooda_act.png           ┘
  wordcloud_rq1.png                — RQ1 (non-opioid ED / pharmacogenomics)
  wordcloud_rq2.png                — RQ2 (opioid ED prediction)
  wordcloud_methods.png            — methodological nodes (orient+decide)
  wordcloud_top_nodes_grid.png     — 3×2 grid of top 6 ontology nodes

Run from: manuscript/CH_1/Literature_Review/
  python scripts/generate_wordclouds.py
"""

import re
import sys
from pathlib import Path
from collections import Counter

import pandas as pd
import matplotlib
matplotlib.use("Agg")   # headless
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
from wordcloud import WordCloud, STOPWORDS

# ── Paths ─────────────────────────────────────────────────────────────────────
TAGGED_CSV   = Path("data/ontology/articles_tagged.csv")
SCREENED_CSV = Path("data/ontology/articles_screened.csv")
OUT_DIR      = Path("data/wordclouds")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Colour palette (OODA phases) ──────────────────────────────────────────────
OODA_COLOURS = {
    "observe": "#2563EB",   # blue
    "orient":  "#7C3AED",   # purple
    "decide":  "#059669",   # green
    "act":     "#DC2626",   # red
}
OODA_LABELS = {
    "observe": "Observe\n(Data & Infrastructure)",
    "orient":  "Orient\n(Methods & Analysis)",
    "decide":  "Decide\n(Models & Interpretation)",
    "act":     "Act\n(Clinical Outcomes)",
}

# RQ colour map
RQ_COLOURS = {
    "RQ1": "#0EA5E9",   # sky blue
    "RQ2": "#F97316",   # orange
    "methods": "#8B5CF6",
}

# ── Stop words ────────────────────────────────────────────────────────────────
EXTRA_STOPS = {
    # generic academic terms
    "using", "based", "study", "studies", "analysis", "data", "model", "models",
    "approach", "method", "methods", "system", "systems", "review", "paper",
    "results", "outcome", "outcomes", "case", "new", "novel", "large",
    "high", "low", "two", "three", "one", "first", "second", "across",
    "associated", "association", "impact", "effect", "effects", "role",
    "use", "used", "within", "among", "comparing", "comparison", "versus",
    "vs", "type", "types", "factor", "factors", "evaluation", "performance",
    "development", "identifying", "identification", "detection", "prediction",
    "predicting", "investigating", "investigation", "related", "different",
    "multiple", "single", "multi", "between", "application", "applications",
    "real", "world", "evidence", "based", "retrospective", "prospective",
    "cohort", "population", "national", "hospital", "clinical", "patients",
    "patient", "healthcare", "health", "care", "medical",
    # year noise
    "2021", "2022", "2023", "2024", "2025", "2026",
}
STOPS = STOPWORDS | EXTRA_STOPS


def clean_text(titles: pd.Series) -> str:
    """Combine titles into one clean text blob."""
    text = " ".join(titles.dropna().astype(str))
    text = re.sub(r"[^a-zA-Z\s\-]", " ", text)
    text = re.sub(r"\b\w{1,2}\b", " ", text)   # remove 1-2 char tokens
    text = re.sub(r"\s+", " ", text)
    return text.lower()


def make_wc(text: str, color: str = "#1E293B", max_words: int = 120,
            width: int = 1200, height: int = 700,
            bg: str = "white") -> WordCloud:
    return WordCloud(
        width=width, height=height,
        background_color=bg,
        max_words=max_words,
        stopwords=STOPS,
        collocations=True,
        collocation_threshold=10,
        min_word_length=3,
        color_func=lambda *a, **kw: color,
        prefer_horizontal=0.85,
        relative_scaling=0.5,
        max_font_size=120,
        min_font_size=10,
        random_state=42,
    ).generate(text)


def save_fig(fig: plt.Figure, stem: str, dpi: int = 150):
    png = OUT_DIR / f"{stem}.png"
    pdf = OUT_DIR / f"{stem}.pdf"
    fig.savefig(png, dpi=dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    print(f"  Saved: {png.name}  {pdf.name}")
    plt.close(fig)


# ── Load data ─────────────────────────────────────────────────────────────────
def load_data():
    if not TAGGED_CSV.exists():
        sys.exit(f"ERROR: {TAGGED_CSV} not found. Run organize_by_ontology.R first.")
    df = pd.read_csv(TAGGED_CSV, dtype=str).fillna("")

    # Merge key_phrases from screened CSV if available
    if SCREENED_CSV.exists():
        sc = pd.read_csv(SCREENED_CSV, dtype=str,
                         usecols=["article_id", "key_phrases"]
                         ).fillna("")
        df = df.merge(sc, on="article_id", how="left")
        df["key_phrases"] = df["key_phrases"].fillna("")
    else:
        df["key_phrases"] = ""

    # Combined text: title + key phrases (key phrases weighted 2x by repetition)
    df["text"] = df["title"] + " " + df["key_phrases"] + " " + df["key_phrases"]
    return df


# ── 1. Overall corpus ─────────────────────────────────────────────────────────
def plot_overall(df: pd.DataFrame):
    text = clean_text(df["text"])
    wc   = make_wc(text, color="#1E40AF", max_words=150, width=1600, height=900)

    fig, ax = plt.subplots(figsize=(16, 9))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    fig.suptitle("Literature Review Corpus\n"
                 f"({len(df):,} articles · 18 PubMed searches + fallback)",
                 fontsize=18, fontweight="bold", y=0.98, color="#1E293B")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save_fig(fig, "wordcloud_overall")


# ── 2. OODA 2×2 grid ──────────────────────────────────────────────────────────
def plot_ooda_grid(df: pd.DataFrame):
    phases = ["observe", "orient", "decide", "act"]
    fig = plt.figure(figsize=(18, 10))
    fig.patch.set_facecolor("#F8FAFC")
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.08, wspace=0.04)

    for i, phase in enumerate(phases):
        subset = df[df["ooda_phase_primary"] == phase]
        n      = len(subset)
        text   = clean_text(subset["text"])
        colour = OODA_COLOURS[phase]
        label  = OODA_LABELS[phase]

        ax = fig.add_subplot(gs[i // 2, i % 2])
        if text.strip():
            wc = make_wc(text, color=colour, max_words=80, width=900, height=500)
            ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")

        # Phase label box
        ax.text(0.02, 0.97, label,
                transform=ax.transAxes, fontsize=11, fontweight="bold",
                color="white", va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.3", facecolor=colour, alpha=0.9))
        ax.text(0.98, 0.97, f"n = {n:,}",
                transform=ax.transAxes, fontsize=9, color=colour,
                va="top", ha="right", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.85))

    fig.suptitle("Literature Review — Word Clouds by OODA Phase",
                 fontsize=16, fontweight="bold", color="#1E293B", y=1.01)
    save_fig(fig, "wordcloud_ooda_grid", dpi=180)


# ── 3. Individual OODA phase panels ───────────────────────────────────────────
def plot_ooda_individual(df: pd.DataFrame):
    for phase in ["observe", "orient", "decide", "act"]:
        subset = df[df["ooda_phase_primary"] == phase]
        text   = clean_text(subset["text"])
        colour = OODA_COLOURS[phase]
        label  = OODA_LABELS[phase].replace("\n", " — ")
        n      = len(subset)

        wc  = make_wc(text, color=colour, max_words=100, width=1400, height=800)
        fig, ax = plt.subplots(figsize=(14, 8), facecolor="#F8FAFC")
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        ax.set_facecolor("#F8FAFC")
        fig.suptitle(f"OODA Phase: {label}  (n = {n:,})",
                     fontsize=14, fontweight="bold", color=colour, y=1.01)
        save_fig(fig, f"wordcloud_ooda_{phase}")


# ── 4. RQ1 vs RQ2 ─────────────────────────────────────────────────────────────
def plot_rq(df: pd.DataFrame):
    rq_map = {
        "RQ1": {
            "label": "RQ1 — Non-Opioid ED & Pharmacogenomics",
            "nodes": ["polypharmacy_ed", "pharmacovigilance", "claims_apcd",
                      "gradient_boosting", "association_rules", "target_leakage"],
            "colour": RQ_COLOURS["RQ1"],
        },
        "RQ2": {
            "label": "RQ2 — Opioid ED Risk Prediction",
            "nodes": ["opioid_ed", "cpt_icd_codes", "gradient_boosting",
                      "explainable_ai", "temporal_analysis"],
            "colour": RQ_COLOURS["RQ2"],
        },
    }

    fig, axes = plt.subplots(1, 2, figsize=(20, 9), facecolor="#F8FAFC")
    fig.suptitle("Literature Review — Word Clouds by Research Question",
                 fontsize=15, fontweight="bold", color="#1E293B", y=1.01)

    for ax, (rq_key, meta) in zip(axes, rq_map.items()):
        pattern = "|".join(meta["nodes"])
        subset  = df[df["ontology_nodes"].str.contains(pattern, na=False)]
        n       = len(subset)
        text    = clean_text(subset["text"])
        colour  = meta["colour"]

        if text.strip():
            wc = make_wc(text, color=colour, max_words=90, width=900, height=700)
            ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        ax.set_title(f"{meta['label']}\n(n = {n:,})",
                     fontsize=11, fontweight="bold", color=colour, pad=8)

    fig.tight_layout()
    save_fig(fig, "wordcloud_rq_comparison")

    # Individual RQ saves
    for rq_key, meta in rq_map.items():
        pattern = "|".join(meta["nodes"])
        subset  = df[df["ontology_nodes"].str.contains(pattern, na=False)]
        n       = len(subset)
        text    = clean_text(subset["text"])
        colour  = meta["colour"]
        wc      = make_wc(text, color=colour, max_words=110, width=1400, height=800)
        fig2, ax2 = plt.subplots(figsize=(14, 8), facecolor="#F8FAFC")
        ax2.imshow(wc, interpolation="bilinear")
        ax2.axis("off")
        fig2.suptitle(f"{meta['label']}  (n = {n:,})",
                      fontsize=13, fontweight="bold", color=colour, y=1.01)
        save_fig(fig2, f"wordcloud_{rq_key.lower()}")


# ── 5. Methodological (orient + decide) ───────────────────────────────────────
def plot_methods(df: pd.DataFrame):
    subset = df[df["ooda_phase_primary"].isin(["orient", "decide"])]
    text   = clean_text(subset["text"])
    n      = len(subset)
    wc     = make_wc(text, color=RQ_COLOURS["methods"], max_words=120,
                     width=1400, height=800)
    fig, ax = plt.subplots(figsize=(14, 8), facecolor="#F8FAFC")
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    fig.suptitle(f"Methodological Articles (Orient + Decide phases)  (n = {n:,})",
                 fontsize=13, fontweight="bold", color=RQ_COLOURS["methods"], y=1.01)
    save_fig(fig, "wordcloud_methods")


# ── 6. Top-6 ontology nodes 3×2 grid ─────────────────────────────────────────
def plot_node_grid(df: pd.DataFrame):
    # Top 6 by article count (from previous analysis)
    top_nodes = [
        ("technical_infrastructure::scalable_analytics", "Scalable Analytics\n(DuckDB/OLAP)",   "#2563EB"),
        ("clinical_outcomes::polypharmacy_ed",           "Polypharmacy ED",                      "#DC2626"),
        ("clinical_outcomes::opioid_ed",                 "Opioid ED",                            "#F97316"),
        ("analytical_methods::association_rules",        "Association Rules\n(FP-Growth)",        "#7C3AED"),
        ("analytical_methods::process_mining",           "Process Mining\n(BupaR)",               "#059669"),
        ("data_sources::claims_apcd",                    "Claims / APCD",                         "#0891B2"),
    ]

    fig = plt.figure(figsize=(21, 9))
    fig.patch.set_facecolor("#F8FAFC")
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.12, wspace=0.04)

    for i, (node_key, label, colour) in enumerate(top_nodes):
        subset = df[df["ontology_nodes"].str.contains(
            node_key.split("::")[-1], na=False)]
        n      = len(subset)
        text   = clean_text(subset["text"])

        ax = fig.add_subplot(gs[i // 3, i % 3])
        if text.strip():
            wc = make_wc(text, color=colour, max_words=70, width=700, height=420)
            ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        ax.text(0.03, 0.97, label,
                transform=ax.transAxes, fontsize=10, fontweight="bold",
                color="white", va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.3", facecolor=colour, alpha=0.9))
        ax.text(0.97, 0.97, f"n={n:,}",
                transform=ax.transAxes, fontsize=9, color=colour,
                va="top", ha="right", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.85))

    fig.suptitle("Top-6 Ontology Nodes — Term Frequency",
                 fontsize=16, fontweight="bold", color="#1E293B", y=1.01)
    save_fig(fig, "wordcloud_top_nodes_grid", dpi=180)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print(f"Loading articles from {TAGGED_CSV}")
    df = load_data()
    print(f"  {len(df):,} articles loaded")
    print(f"  OODA distribution:")
    print(df["ooda_phase_primary"].value_counts().to_string())
    print()
    print(f"Generating word clouds -> {OUT_DIR}/")

    plot_overall(df)
    plot_ooda_grid(df)
    plot_ooda_individual(df)
    plot_rq(df)
    plot_methods(df)
    plot_node_grid(df)

    print(f"\nAll word clouds saved to {OUT_DIR}/")
    print(f"  PNG (screen) and PDF (print/figure insertion) formats")
    print(f"\nTo use in manuscript (ch01_bmic.qmd):")
    print(f"  {{fig-wordcloud-overall}}  -> data/wordclouds/wordcloud_overall.pdf")
    print(f"  {{fig-wordcloud-ooda}}     -> data/wordclouds/wordcloud_ooda_grid.pdf")
    print(f"  {{fig-wordcloud-rq}}       -> data/wordclouds/wordcloud_rq_comparison.pdf")


if __name__ == "__main__":
    main()
