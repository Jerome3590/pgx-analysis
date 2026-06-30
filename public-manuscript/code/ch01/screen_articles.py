"""
screen_articles.py
NLP pre-screening pipeline using spaCy + pytextrank + RQ keyword scoring.
Scores every article against each research question and recommends include/exclude.

Outputs:
  outputs/ch01/articles_screened.csv  — original columns + scoring columns
  outputs/ch01/screening_summary.csv  — aggregate counts by RQ

Run from the public code companion root:
  python code/ch01/screen_articles.py [--threshold 0.15] [--use-comprehend]
"""

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
INPUT_DIR = ROOT / "inputs" / "ch01"
OUTPUT_DIR = ROOT / "outputs" / "ch01"

# ── RQ keyword taxonomy (maps to docs/CrossStep_Workflow/README_research_questions_mapping.md)
RQ_KEYWORDS = {
    "RQ1": [
        "non.opioid", "non opioid", "pharmacogenomic", "pharmacogenetic", "pgx",
        "drug.drug interaction", "ddi", "adverse drug event", "ade",
        "polypharmacy", "drug combination", "clinical decision support", "cds",
        "emergency department", "ed visit", "hospital admission",
        "administrative claim", "apcd", "all.payer", "claims data",
    ],
    "RQ2": [
        "opioid", "opioid use disorder", "oud", "substance use disorder", "sud",
        "opioid risk", "opioid prediction", "morphine", "fentanyl", "hydrocodone",
        "oxycodone", "naloxone", "buprenorphine", "methadone",
        "overdose", "cpt code", "icd.10", "icd code",
        "risk score", "risk prediction", "risk stratif",
    ],
    "N1_DTW": [
        "dynamic time warping", "dtw", "time series cluster", "time series classif",
        "longitudinal cluster", "trajectory cluster", "patient trajectory",
    ],
    "N2_N3_process_mining": [
        "process mining", "event log", "bupar", "clinical pathway", "care pathway",
        "process model", "petri net", "care sequence", "treatment sequence",
        "conformance check",
    ],
    "N4_fpgrowth": [
        "association rule", "frequent itemset", "fp.growth", "fp growth", "apriori",
        "market basket", "co.occurrence", "item co", "frequent pattern",
    ],
    "N5_xai": [
        "shap", "shapley", "explainable ai", "xai", "interpretab", "feature importance",
        "lime", "counterfactual", "model explanation", "black.box",
    ],
    "N6_polypharmacy": [
        "polypharmacy", "drug.drug interaction", "pharmacogenomic",
        "drug combination", "multi.drug", "co.prescrib", "concomitant drug",
        "poly-pharmacy",
    ],
}

# Global inclusion terms — article must match at least one to be considered at all
GLOBAL_INCLUDE = [
    "machine learning", "deep learning", "neural network", "random forest",
    "gradient boost", "xgboost", "catboost", "logistic regression",
    "prediction model", "classification", "clinical", "health", "medical",
    "patient", "hospital", "drug", "pharmacol", "opioid", "claims", "ehr",
    "electronic health record", "administrative data", "cohort", "population",
]

# Hard-exclude patterns (clearly out of scope)
GLOBAL_EXCLUDE = [
    r"\bveterinar\b", r"\banimal model\b", r"\bin vitro\b", r"\bin vivo\b",
    r"\bcell line\b", r"\bmouse model\b", r"\brat model\b",
    r"\bimage segmentation\b", r"\bcomputer vision\b",
    r"\bnatural language processing\b.*\bnot clinical\b",
]


def tokenize(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", " ", text.lower())


def score_text(text: str, keywords: list[str]) -> float:
    """Score 0-1 based on keyword hit density."""
    t = tokenize(text)
    hits = sum(1 for kw in keywords if re.search(kw.replace(".", r"\w*"), t))
    return round(hits / max(len(keywords), 1), 4)


def get_fulltext(pmc_id: str, source_file: str) -> str:
    """Try to read JSON full text if available."""
    if not pmc_id.startswith("PMC"):
        return ""
    json_path = Path(source_file).parent / "pubmed_json_files" / f"{pmc_id}.json"
    if not json_path.exists():
        return ""
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
        # BioC JSON structure: documents[0].passages[*].text
        docs = data if isinstance(data, list) else [data]
        texts = []
        for doc in docs:
            for passage in doc.get("documents", [doc])[0].get("passages", []):
                texts.append(passage.get("text", ""))
        return " ".join(texts)[:4000]  # cap at 4k chars for scoring
    except Exception:
        return ""


def run_pytextrank(texts: list[str]) -> list[list[str]]:
    """Extract key phrases using spaCy + pytextrank."""
    try:
        import spacy
        import pytextrank  # noqa: F401 — registers pipe
        nlp = spacy.load("en_core_web_sm")
        if "textrank" not in nlp.pipe_names:
            nlp.add_pipe("textrank")
        results = []
        for text in texts:
            if not text.strip():
                results.append([])
                continue
            doc = nlp(text[:1000])  # limit to 1k chars per article
            phrases = [p.text.lower() for p in doc._.phrases[:8]]
            results.append(phrases)
        return results
    except Exception as e:
        print(f"  pytextrank unavailable ({e}); using keyword-only scoring")
        return [[] for _ in texts]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=0.12,
                        help="Min composite score to recommend inclusion (default 0.12)")
    parser.add_argument("--use-comprehend", action="store_true",
                        help="Use AWS Comprehend for medical entity detection (costs $)")
    parser.add_argument("--no-textrank", action="store_true",
                        help="Skip pytextrank (faster, titles only)")
    args = parser.parse_args()

    tagged_csv = INPUT_DIR / "articles_tagged.csv"
    if not tagged_csv.exists():
        sys.exit(f"ERROR: {tagged_csv} not found. Provide a compatible tagged-article export.")

    df = pd.read_csv(tagged_csv, dtype=str).fillna("")
    print(f"Loaded {len(df):,} articles for screening")

    # ── Key phrase extraction ─────────────────────────────────────────────────
    titles = df["title"].tolist()
    if not args.no_textrank:
        print("Extracting key phrases with pytextrank...")
        key_phrases_list = run_pytextrank(titles)
    else:
        key_phrases_list = [[] for _ in titles]

    # ── AWS Comprehend medical entity detection ───────────────────────────────
    comprehend_entities: list[list[str]] = [[] for _ in titles]
    if args.use_comprehend:
        try:
            import boto3
            cm = boto3.client("comprehendmedical")
            print("Running AWS Comprehend Medical on titles...")
            for i, title in enumerate(titles):
                if not title.strip():
                    continue
                resp = cm.detect_entities_v2(Text=title[:10000])
                ents = [e["Text"].lower() for e in resp.get("Entities", [])
                        if e["Score"] > 0.7]
                comprehend_entities[i] = ents
                if i % 500 == 0:
                    print(f"  Comprehend: {i}/{len(titles)}")
        except Exception as e:
            print(f"  Comprehend unavailable ({e}); skipping")

    # ── Score each article ────────────────────────────────────────────────────
    scored_rows = []
    n_include = 0

    for i, row in df.iterrows():
        title       = row["title"]
        source_file = row.get("source_file", "")
        pmc_id      = row.get("pmc_id", "")
        ooda_phase  = row.get("ooda_phase_primary", "")
        node_key    = row.get("ontology_nodes", "")

        # Combine title + key phrases + comprehend entities for scoring
        phrases   = key_phrases_list[i]
        comp_ents = comprehend_entities[i]
        full_text = get_fulltext(pmc_id, source_file)
        rich_text = " ".join([title] + phrases + comp_ents + [full_text[:500]])

        # Per-RQ scores
        rq_scores = {rq: score_text(rich_text, kws) for rq, kws in RQ_KEYWORDS.items()}

        # Global include check (must mention at least one domain term)
        global_hit = any(re.search(kw, tokenize(title)) for kw in GLOBAL_INCLUDE)

        # Hard-exclude check
        hard_exclude = any(re.search(pat, title.lower()) for pat in GLOBAL_EXCLUDE)

        # Composite score: max RQ score, with bonus for OODA-aligned phases
        composite = max(rq_scores.values())
        if ooda_phase in ("observe", "orient"):
            composite = min(1.0, composite * 1.15)  # 15% bonus for data/method papers

        # Recommendation
        if hard_exclude:
            recommend = "exclude"
            reason = "hard_exclude_pattern"
        elif not global_hit and composite < args.threshold:
            recommend = "exclude"
            reason = "off_topic"
        elif composite >= args.threshold:
            recommend = "include"
            reason = f"score={composite:.3f}"
            n_include += 1
        else:
            recommend = "exclude"
            reason = f"score={composite:.3f}<{args.threshold}"

        row_out = row.to_dict()
        row_out.update({
            "key_phrases":         "; ".join(phrases[:5]),
            "comprehend_entities": "; ".join(comp_ents[:5]),
            "rq1_score":           rq_scores["RQ1"],
            "rq2_score":           rq_scores["RQ2"],
            "n1_score":            rq_scores["N1_DTW"],
            "n4_score":            rq_scores["N4_fpgrowth"],
            "n5_score":            rq_scores["N5_xai"],
            "composite_score":     round(composite, 4),
            "include_recommended": recommend,
            "screen_reason":       reason,
            "human_decision":      "",   # ← fill during manual review
        })
        scored_rows.append(row_out)

    out_df = pd.DataFrame(scored_rows)
    out_path = OUTPUT_DIR / "articles_screened.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)

    # ── Summary by OODA phase ─────────────────────────────────────────────────
    summary = (out_df.groupby("ooda_phase_primary")["include_recommended"]
               .value_counts().unstack(fill_value=0).reset_index())
    summary_path = OUTPUT_DIR / "screening_summary.csv"
    summary.to_csv(summary_path, index=False)

    n_total   = len(out_df)
    n_exclude = n_total - n_include
    pct       = 100 * n_include / max(n_total, 1)

    print(f"\n── Screening complete ───────────────────────────────────")
    print(f"  Total articles:         {n_total:,}")
    print(f"  Recommended include:    {n_include:,}  ({pct:.1f}%)")
    print(f"  Recommended exclude:    {n_exclude:,}")
    print(f"  Threshold used:         {args.threshold}")
    print(f"\n  Output:  {out_path}")
    print(f"  Summary: {summary_path}")
    print(f"\nNEXT: Review {out_path.name} and apply inclusion decisions as needed.")


if __name__ == "__main__":
    main()
