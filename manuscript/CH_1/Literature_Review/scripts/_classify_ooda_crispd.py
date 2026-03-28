"""
_classify_ooda_crispd.py
────────────────────────────────────────────────────────────────────────
Classify every article in articles_screened.csv with a combined
OODA Loop × CRISP-DM taxonomy.

Framework:
  OODA Loop (top-level situational awareness)
    └─ CRISP-DM (data/analytics execution layer)

OODA phases (already in data as ooda_phase_primary):
  observe  – data sensing, surveillance, pharmacovigilance, EHR/claims
  orient   – synthesis, background, context, epidemiology
  decide   – modeling, prediction, decision support
  act      – intervention, deployment, clinical programs

CRISP-DM sub-phases (added by this script):
  business_understanding  – problem framing, reviews, guidelines, policy
  data_understanding      – data sources, EDA, descriptive stats, cohorts
  data_preparation        – feature engineering, preprocessing, linkage
  modeling                – ML/statistical models, process mining, DTW
  evaluation              – SHAP, explainability, AUC, benchmarks
  deployment              – CDS tools, interventions, real-world programs

Adds to articles_screened.csv:
  crisp_dm_phase    – primary CRISP-DM phase
  ooda_crisp_label  – combined "{ooda}:{crisp_dm}"

Also writes:
  scripts/ooda_crisp_taxonomy.csv    – per-article mapping
  scripts/ooda_crisp_summary.md      – counts + hierarchy
  manual_review/OODA_CRISP_MAP.md    – review-ready reference

Usage:
  python scripts/_classify_ooda_crispd.py
  python scripts/_classify_ooda_crispd.py --dry-run
"""
import argparse, csv, json, re
from collections import Counter, defaultdict
from pathlib import Path

SCREENED      = Path("data/ontology/articles_screened.csv")
SCHOLAR_JSON  = Path("data/scholar_json")
TAX_CSV       = Path("scripts/ooda_crisp_taxonomy.csv")
SUMMARY_MD    = Path("scripts/ooda_crisp_summary.md")
REVIEW_HUB    = Path(r"C:\Projects\pgx-analysis\manuscript\infrastructure_setup\manual_review")
MAP_MD        = REVIEW_HUB / "OODA_CRISP_MAP.md"

# ── OODA normalization ────────────────────────────────────────────────────────
OODA_NORM = {"observe":"observe","Observe":"observe","OBSERVE":"observe",
             "orient":"orient","Orient":"orient","ORIENT":"orient",
             "decide":"decide","Decide":"decide","DECIDE":"decide",
             "act":"act","Act":"act","ACT":"act"}
OODA_ORDER  = ["observe","orient","decide","act"]
CRISP_ORDER = ["business_understanding","data_understanding","data_preparation",
               "modeling","evaluation","deployment"]

# ── CRISP-DM keyword signatures ───────────────────────────────────────────────
CRISP_TOKENS: dict[str, list[str]] = {
    "business_understanding": [
        "systematic review","narrative review","scoping review","meta-analysis",
        "literature review","background","overview","guideline","clinical guideline",
        "protocol","policy","regulation","prevalence","incidence","epidemiology",
        "burden","crisis","framework","consensus","pharmacology","pharmacokinetics",
        "pharmacodynamics","mechanism","etiology","pathophysiology","taxonomy",
        "classification","definition","conceptual","theoretical","objectives",
    ],
    "data_understanding": [
        "claims data","administrative claim","apcd","all-payer","ehr",
        "electronic health record","faers","spontaneous report","adverse event report",
        "registry","cohort study","retrospective","cross-sectional","longitudinal",
        "population-based","descriptive study","surveillance","characterization",
        "data source","database","medicare","medicaid","insurance claim","encounter",
        "discharge","medical record","eda","exploratory","data quality",
        "missing data","real-world data","real-world evidence","observational",
    ],
    "data_preparation": [
        "preprocessing","cohort selection","inclusion criteria","exclusion criteria",
        "feature engineering","feature extraction","data cleaning","imputation",
        "propensity score","matching","confounding","covariate","variable selection",
        "encoding","normalization","standardization","data linkage","record linkage",
        "de-identification","train test split","cross-validation setup",
        "oversampling","undersampling","smote","class imbalance","data augmentation",
    ],
    "modeling": [
        "machine learning","deep learning","neural network","xgboost","catboost",
        "random forest","gradient boosting","logistic regression","cox regression",
        "lstm","transformer","bert","gpt","large language model","llm",
        "process mining","event log","petri net","conformance checking",
        "time series","dynamic time warping","dtw","clustering","k-means",
        "drug interaction prediction","ddi prediction","classification algorithm",
        "natural language processing","nlp","text mining","named entity",
        "fp-growth","association rule","graph neural","attention mechanism",
        "prediction model","predictive model","risk model","risk score",
    ],
    "evaluation": [
        "shap","shapley","explainab","interpretab","feature importance",
        "auc","auroc","c-statistic","accuracy","precision","recall","f1",
        "sensitivity","specificity","positive predictive","negative predictive",
        "roc curve","calibration","brier score","confusion matrix",
        "cross-validation","ablation study","benchmark","comparative study",
        "model comparison","performance evaluation","validation study",
        "external validation","internal validation","discrimination",
    ],
    "deployment": [
        "clinical decision support","cds","decision support system",
        "clinical workflow","implementation","real-world deployment",
        "naloxone","buprenorphine program","treatment program","intervention",
        "pharmacist","prescription","prescribing practice","point-of-care",
        "telehealth","telemedicine","mobile health","mhealth","patient portal",
        "community pharmacy","screening program","referral","discharge planning",
        "health policy","quality improvement","care coordination","outcome evaluation",
        "program evaluation","public health program","harm reduction",
    ],
}

# OODA → preferred CRISP-DM phases (for tie-breaking when scores are equal)
AFFINITY: dict[str, list[str]] = {
    "observe": ["data_understanding","data_preparation","business_understanding"],
    "orient":  ["business_understanding","data_understanding","evaluation"],
    "decide":  ["modeling","evaluation","data_preparation"],
    "act":     ["deployment","business_understanding","evaluation"],
}

# ── Helpers ───────────────────────────────────────────────────────────────────
_JSON_INDEX = {p.stem: p for p in SCHOLAR_JSON.glob("*.json")}

def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]"," ", s.lower())

def _load_text(pmc_id: str, article_id: str) -> str:
    path = _JSON_INDEX.get(pmc_id) or _JSON_INDEX.get(f"article_{article_id}")
    if not path:
        return ""
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        return (_norm(obj.get("abstract","") or "") + " " +
                _norm((obj.get("full_text","") or "")[:4000]))
    except Exception:
        return ""

def classify(ooda: str, title: str, key_phrases: str,
             pmc_id: str, article_id: str) -> str:
    full   = _load_text(pmc_id, article_id)
    text   = _norm(title + " " + key_phrases) + " " + full
    scores = {phase: 0.0 for phase in CRISP_ORDER}

    for phase, tokens in CRISP_TOKENS.items():
        for tok in tokens:
            if tok in text:
                scores[phase] += 1.0

    # OODA affinity bonus
    ooda_n = OODA_NORM.get(ooda, "orient")
    for i, pref in enumerate(AFFINITY.get(ooda_n, [])):
        scores[pref] += 0.4 / (i + 1)

    best = max(scores, key=lambda k: scores[k])
    # Default if no signal
    if scores[best] == 0:
        defaults = {"observe":"data_understanding","orient":"business_understanding",
                    "decide":"modeling","act":"deployment"}
        return defaults.get(ooda_n, "business_understanding")
    return best

# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    rows = list(csv.DictReader(open(SCREENED, encoding="utf-8-sig")))
    print(f"Classifying {len(rows)} articles  (scholar_json/: {len(_JSON_INDEX)} available)\n")

    tax_rows = []
    for i, row in enumerate(rows):
        ooda        = row.get("ooda_phase_primary","orient") or "orient"
        title       = row.get("title","") or ""
        key_phrases = row.get("key_phrases","") or ""
        pmc_id      = row.get("pmc_id","").strip()
        article_id  = row.get("article_id","").strip()

        ooda_n     = OODA_NORM.get(ooda, "orient")
        crisp      = classify(ooda, title, key_phrases, pmc_id, article_id)
        combined   = f"{ooda_n}:{crisp}"

        row["crisp_dm_phase"]   = crisp
        row["ooda_crisp_label"] = combined

        tax_rows.append({
            "article_id":      article_id,
            "pmc_id":          pmc_id,
            "title":           title[:80],
            "human_decision":  row.get("human_decision",""),
            "composite_score": row.get("composite_score",""),
            "ooda_phase":      ooda_n,
            "crisp_dm_phase":  crisp,
            "ooda_crisp_label":combined,
        })
        if (i+1) % 2000 == 0:
            print(f"  ... {i+1}/{len(rows)}")

    # ── Distribution ──────────────────────────────────────────────────────────
    counter     = Counter((r["ooda_phase"], r["crisp_dm_phase"]) for r in tax_rows)
    inc_counter = Counter((r["ooda_phase"], r["crisp_dm_phase"])
                          for r in tax_rows if r["human_decision"]=="include")

    print(f"\n{'':28}", end="")
    for c in CRISP_ORDER:
        print(f"  {c[:8]:>8}", end="")
    print(f"  {'TOTAL':>7}")

    for o in OODA_ORDER:
        row_total = sum(counter.get((o,c),0) for c in CRISP_ORDER)
        print(f"  ooda:{o:<22}", end="")
        for c in CRISP_ORDER:
            print(f"  {counter.get((o,c),0):>8}", end="")
        print(f"  {row_total:>7}")

    grand = sum(counter.values())
    print(f"  {'TOTAL':26}", end="")
    for c in CRISP_ORDER:
        print(f"  {sum(counter.get((o,c),0) for o in OODA_ORDER):>8}", end="")
    print(f"  {grand:>7}")

    print(f"\n  (included only):")
    for o in OODA_ORDER:
        row_total = sum(inc_counter.get((o,c),0) for c in CRISP_ORDER)
        print(f"    ooda:{o:<20}", end="")
        for c in CRISP_ORDER:
            print(f"  {inc_counter.get((o,c),0):>6}", end="")
        print(f"  {row_total:>6}")

    if args.dry_run:
        print("\n[dry-run] No files written.")
        return

    # ── Write articles_screened.csv ───────────────────────────────────────────
    fieldnames = list(rows[0].keys())
    for col in ("crisp_dm_phase","ooda_crisp_label"):
        if col not in fieldnames:
            fieldnames.append(col)
    with open(SCREENED, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"\n✓ articles_screened.csv  +crisp_dm_phase +ooda_crisp_label")

    # ── Write taxonomy CSV ────────────────────────────────────────────────────
    with open(TAX_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(tax_rows[0].keys()))
        w.writeheader()
        w.writerows(tax_rows)
    print(f"✓ {TAX_CSV}  ({len(tax_rows)} rows)")

    # ── Write markdown maps ────────────────────────────────────────────────────
    for out_path in (SUMMARY_MD, MAP_MD):
        _write_md(out_path, tax_rows, counter, inc_counter)
    print(f"✓ {SUMMARY_MD}")
    print(f"✓ {MAP_MD}")


def _write_md(path: Path, tax_rows, counter, inc_counter):
    inc = {r["article_id"]: r for r in tax_rows if r["human_decision"]=="include"}

    lines = [
        "# OODA Loop × CRISP-DM Article Taxonomy",
        "",
        "> **OODA Loop** (top level — situational awareness layer)",
        "> **└─ CRISP-DM** (data & analytics execution layer)",
        "",
        "```",
        "OODA: Observe  ── sensing, collecting, monitoring",
        "  └─ CRISP-DM: data_understanding  (EHR/claims, FAERS, registries)",
        "  └─ CRISP-DM: data_preparation   (cohort building, preprocessing)",
        "",
        "OODA: Orient   ── interpreting, synthesizing, situational awareness",
        "  └─ CRISP-DM: business_understanding  (reviews, guidelines, background)",
        "  └─ CRISP-DM: data_understanding      (descriptive/EDA analysis)",
        "",
        "OODA: Decide   ── modeling, pattern recognition, prediction",
        "  └─ CRISP-DM: modeling    (ML, process mining, DDI prediction)",
        "  └─ CRISP-DM: evaluation  (SHAP, AUC, explainability, validation)",
        "",
        "OODA: Act      ── intervention, deployment, clinical implementation",
        "  └─ CRISP-DM: deployment  (CDS tools, naloxone, treatment programs)",
        "  └─ CRISP-DM: evaluation  (program evaluation, outcome assessment)",
        "```",
        "",
        "## Article Counts by OODA × CRISP-DM",
        "",
        "| OODA | CRISP-DM | All | Included |",
        "|------|----------|----:|---------:|",
    ]
    for o in OODA_ORDER:
        for c in CRISP_ORDER:
            n   = counter.get((o,c), 0)
            ni  = inc_counter.get((o,c), 0)
            if n:
                lines.append(f"| {o} | {c} | {n} | {ni} |")
    lines += ["","## Included Articles by Cell",""]

    for o in OODA_ORDER:
        for c in CRISP_ORDER:
            subset = [r for r in tax_rows
                      if r["ooda_phase"]==o and r["crisp_dm_phase"]==c
                      and r["human_decision"]=="include"]
            if not subset:
                continue
            lines.append(f"### {o.title()} → {c.replace('_',' ').title()} ({len(subset)})")
            lines.append("")
            for r in sorted(subset, key=lambda x: float(x["composite_score"] or 0), reverse=True)[:15]:
                lines.append(f"- [{r['pmc_id'] or r['article_id']}] {r['title'][:80]}")
            if len(subset) > 15:
                lines.append(f"- … and {len(subset)-15} more")
            lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
