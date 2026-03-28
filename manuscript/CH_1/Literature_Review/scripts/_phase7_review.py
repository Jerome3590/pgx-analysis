"""
Phase 7 — Final article screening via pytextrank phrase scoring.

Scores each article in articles_screened.csv against dissertation research
topics using:
  1. pytextrank key phrases from full text (abstract + body) via data/scholar_json/
     Falls back to title-only when no JSON available.
  2. Existing key_phrases column (AWS Comprehend)
  3. Existing composite_score (RQ-weighted relevance)

Adds pytextrank_score column to articles_screened.csv.
Only fills human_decision when it is currently blank — existing human decisions
are ALWAYS preserved (idempotent).

Usage:
  python scripts/_phase7_review.py --dry-run              # show distribution only
  python scripts/_phase7_review.py --threshold 0.15       # write decisions
  python scripts/_phase7_review.py --threshold 0.15 --write
"""
import argparse, csv, json, re
from pathlib import Path
from collections import Counter

import spacy
import pytextrank  # noqa: F401  (registers "textrank" pipe)

SCREENED_CSV  = Path("data/ontology/articles_screened.csv")
SCHOLAR_JSON  = Path("data/scholar_json")

# ── Build index: pmc_id / hsh_id → json path ──────────────────────────────────
_JSON_INDEX: dict[str, Path] = {p.stem: p for p in SCHOLAR_JSON.glob("*.json")}

def _load_fulltext(pmc_id: str, article_id: str) -> str:
    """
    Return abstract + first 3000 chars of body text from scholar_json/.
    Tries pmc_id first, then article_id. Returns empty string if not found.
    """
    path = _JSON_INDEX.get(pmc_id) or _JSON_INDEX.get(f"article_{article_id}")
    if not path:
        return ""
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        abstract  = obj.get("abstract", "") or ""
        full_text = obj.get("full_text", "") or ""
        # Abstract in full + first 3000 chars of body (enough for topic detection)
        return (abstract + " " + full_text[:3000]).strip()
    except Exception:
        return ""

# ── Research topic phrase sets ─────────────────────────────────────────────────
# Tokens that strongly signal relevance to dissertation research questions
TOPIC_TOKENS: dict[str, set[str]] = {
    "pgx": {
        "pharmacogenomic", "pharmacogenomics", "pharmacogenetic", "pharmacogenetics",
        "cyp", "cytochrome", "cyp2d6", "cyp3a4", "cyp2c19", "cyp2c9",
        "drug metabolism", "metabolizer", "genotype", "allele", "snp", "haplotype",
        "biomarker", "variant", "polymorphism", "gene", "genetic", "genomic",
        "oprm1", "comt", "abcb1",
    },
    "ddi": {
        "drug-drug", "drug interaction", "ddi", "polypharmacy", "drug combination",
        "interaction", "comedication", "coprescription", "multidrug",
        "contraindication", "interacting",
    },
    "oud_opioid": {
        "opioid", "opioids", "naloxone", "buprenorphine", "methadone", "fentanyl",
        "heroin", "overdose", "opioid use disorder", "oud", "substance use",
        "addiction", "dependence", "neonatal abstinence", "nows", "withdrawal",
        "pain management", "analgesic", "tramadol", "morphine", "hydrocodone",
        "oxycodone", "prescription opioid",
    },
    "pharmacovigilance": {
        "adverse drug", "adverse event", "adr", "ade", "pharmacovigilance",
        "faers", "spontaneous report", "signal detection", "drug safety",
        "drug reaction", "side effect", "toxicity", "hepatotoxic",
        "drug-induced", "iatrogenic",
    },
    "claims_apcd": {
        "claims", "apcd", "all-payer", "all payer", "administrative data",
        "insurance", "medicaid", "medicare", "electronic health record", "ehr",
        "health record", "database", "real-world", "real world", "population-based",
    },
    "ml_cds": {
        "machine learning", "deep learning", "neural network", "xgboost",
        "random forest", "gradient boosting", "classification", "prediction",
        "predictive model", "clinical decision support", "cds", "algorithm",
        "artificial intelligence", "natural language processing", "nlp",
        "transformer", "bert", "explainab", "shap", "feature importance",
    },
    "process_mining": {
        "process mining", "event log", "workflow", "bupar", "ooda", "petri net",
        "conformance", "trace", "case notion", "directly-follows",
    },
}

# Weights per topic (higher = more central to dissertation)
TOPIC_WEIGHTS = {
    "pgx":              3.0,
    "ddi":              2.5,
    "oud_opioid":       2.5,
    "pharmacovigilance":2.0,
    "claims_apcd":      1.5,
    "ml_cds":           1.5,
    "process_mining":   1.0,
}

# ── spaCy + pytextrank setup ───────────────────────────────────────────────────
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    nlp = spacy.load("en_core_web_lg")
nlp.add_pipe("textrank")

def normalize(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]", " ", s.lower())

def topic_score(nlp_text: str, extra_phrases: str = "") -> float:
    """
    Compute weighted topic relevance score.
    nlp_text: full text to run pytextrank on (abstract+body, or title fallback).
    extra_phrases: key_phrases column from CSV (direct token match only).
    """
    combined = normalize(nlp_text + " " + extra_phrases)
    score = 0.0

    # 1. pytextrank on provided text (capped at 4000 chars for performance)
    try:
        doc = nlp(nlp_text[:4000])
        for phrase in doc._.phrases[:30]:
            ph_norm = normalize(phrase.text)
            for topic, tokens in TOPIC_TOKENS.items():
                for tok in tokens:
                    if tok in ph_norm:
                        score += phrase.rank * TOPIC_WEIGHTS[topic]
                        break
    except Exception:
        pass

    # 2. Direct token match on full combined text
    for topic, tokens in TOPIC_TOKENS.items():
        for tok in tokens:
            if tok in combined:
                score += 0.05 * TOPIC_WEIGHTS[topic]

    return round(score, 4)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Show score distribution without writing")
    parser.add_argument("--threshold", type=float, default=0.15,
                        help="Minimum topic score for include (default 0.15)")
    parser.add_argument("--write", action="store_true",
                        help="Write human_decision to articles_screened.csv")
    args = parser.parse_args()

    rows = list(csv.DictReader(open(SCREENED_CSV, encoding="utf-8-sig")))
    json_hit = sum(1 for r in rows
                   if r.get("pmc_id","") in _JSON_INDEX
                   or f"article_{r.get('article_id','')}" in _JSON_INDEX)
    print(f"Loaded {len(rows)} articles from {SCREENED_CSV}")
    print(f"scholar_json/ index: {len(_JSON_INDEX)} files  ({json_hit} articles have full text)")
    print(f"Scoring with pytextrank (full-text where available, threshold={args.threshold})...\n")

    scores = []
    full_text_used = 0
    for i, row in enumerate(rows):
        title       = row.get("title", "") or ""
        key_phrases = row.get("key_phrases", "") or ""
        pmc_id      = row.get("pmc_id", "").strip()
        article_id  = row.get("article_id", "").strip()
        algo_score  = float(row.get("composite_score", 0) or 0)

        # Use full text if available, else fall back to title
        fulltext = _load_fulltext(pmc_id, article_id)
        if fulltext:
            nlp_input = fulltext
            full_text_used += 1
        else:
            nlp_input = title

        ts = topic_score(nlp_input, key_phrases)
        combined = ts + algo_score * 0.3
        scores.append(combined)
        row["_topic_score"] = combined
        if (i + 1) % 500 == 0:
            print(f"  ... {i+1}/{len(rows)}  (full-text: {full_text_used})")

    print(f"\n── Score distribution ───────────────────────────────")
    buckets = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0, 999]
    for lo, hi in zip(buckets, buckets[1:]):
        n = sum(1 for s in scores if lo <= s < hi)
        print(f"  [{lo:.2f}, {hi:.2f}) : {n:5d}  {'█' * (n // 50)}")

    include_n = sum(1 for s in scores if s >= args.threshold)
    exclude_n = len(scores) - include_n
    print(f"\n  Threshold {args.threshold}: include={include_n}  exclude={exclude_n}")

    # Cross-check against algorithm recommendation
    algo_include = sum(1 for r in rows if r.get("include_recommended") == "include")
    overlap = sum(1 for r in rows
                  if r.get("include_recommended") == "include"
                  and r["_topic_score"] >= args.threshold)
    print(f"  Algorithm recommended include : {algo_include}")
    print(f"  Overlap (both agree include)  : {overlap}")
    print(f"  pytextrank-only additions     : {include_n - overlap}")

    if args.dry_run:
        print("\n[dry-run] No changes written.")
        return

    if not args.write:
        print("\nPass --write to commit pytextrank_score to the CSV.")
        return

    # Write pytextrank_score; only fill blank human_decision entries (idempotent)
    filled_new = 0
    preserved  = 0
    for row in rows:
        ts = row["_topic_score"]
        row["pytextrank_score"] = ts
        del row["_topic_score"]
        existing = row.get("human_decision", "").strip()
        if not existing:
            row["human_decision"] = "include" if ts >= args.threshold else "exclude"
            filled_new += 1
        else:
            preserved += 1

    fieldnames = list(rows[0].keys())
    if "pytextrank_score" not in fieldnames:
        fieldnames.append("pytextrank_score")

    with open(SCREENED_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    final_inc = sum(1 for r in rows if r.get("human_decision") == "include")
    final_exc = sum(1 for r in rows if r.get("human_decision") == "exclude")
    print(f"\n✓ Wrote pytextrank_score + human_decision to {SCREENED_CSV}")
    print(f"  Preserved existing decisions : {preserved}")
    print(f"  Filled blank entries         : {filled_new}")
    print(f"  Final include                : {final_inc}")
    print(f"  Final exclude                : {final_exc}")


if __name__ == "__main__":
    main()
