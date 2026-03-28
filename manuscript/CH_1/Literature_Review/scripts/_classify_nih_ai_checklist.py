"""
_classify_nih_ai_checklist.py
────────────────────────────────────────────────────────────────────────
Tag the Decide+Act risk-model / prediction subset against:

  1. NIH AI Reporting Checklist domains
     (transparency, bias/fairness, validation, clinical utility, etc.)

  2. Operational performance dimensions
     (cost, throughput, patient outcomes, process performance, etc.)

Subset targeted:
  ooda_phase_primary ∈ {decide, act}
  crisp_dm_phase     ∈ {modeling, evaluation}
  (all articles, not just human_decision=include)

Scoring approach:
  • keyword-hit count per domain against: title + key_phrases + full_text[:6000]
  • tags = all domains with ≥1 hit
  • score = number of tagged domains

Adds to articles_screened.csv (only rows in subset):
  nih_ai_tags      – pipe-delimited matched NIH AI checklist domains
  nih_ai_score     – count of NIH AI domains addressed (0–12)
  nih_ai_pct       – nih_ai_score / 12 × 100
  op_perf_tags     – pipe-delimited matched operational performance dims
  op_perf_score    – count of op_perf dimensions (0–8)

Also writes:
  scripts/nih_ai_checklist_tags.csv    – per-article detail (subset only)
  scripts/nih_ai_checklist_summary.md  – domain coverage summary
  manual_review/NIH_AI_CHECKLIST_MAP.md

Usage:
  python scripts/_classify_nih_ai_checklist.py
  python scripts/_classify_nih_ai_checklist.py --dry-run
  python scripts/_classify_nih_ai_checklist.py --min-score 3   # filter summary
"""
import argparse, csv, json, re
from collections import Counter, defaultdict
from pathlib import Path

SCREENED      = Path("data/ontology/articles_screened.csv")
SCHOLAR_JSON  = Path("data/scholar_json")
TAGS_CSV      = Path("scripts/nih_ai_checklist_tags.csv")
SUMMARY_MD    = Path("scripts/nih_ai_checklist_summary.md")
REVIEW_HUB    = Path(r"C:\Projects\pgx-analysis\manuscript\infrastructure_setup\manual_review")
MAP_MD        = REVIEW_HUB / "NIH_AI_CHECKLIST_MAP.md"

# ── NIH AI Reporting Checklist domains ────────────────────────────────────────
# Based on: NIH/NASEM reporting standards + TRIPOD-AI + CONSORT-AI + SPIRIT-AI
NIH_AI: dict[str, list[str]] = {
    "study_design": [
        "study design","prospective","retrospective","cross-sectional","cohort study",
        "randomized controlled","clinical trial","observational study","case-control",
        "longitudinal study","registry study","real-world study","pragmatic trial",
    ],
    "data_reporting": [
        "training set","training data","validation set","test set","data split",
        "sample size","cohort description","inclusion criteria","exclusion criteria",
        "data source","missing data","data quality","data completeness","class imbalance",
        "train test","held-out","hold-out","development cohort","derivation cohort",
    ],
    "model_transparency": [
        "reproducib","open source","code availab","github","gitlab","model card",
        "model architecture","hyperparameter","algorithm description","model specification",
        "model documentation","software availab","publicly available","data availab",
        "replicat",
    ],
    "bias_fairness": [
        "bias","fairness","equity","disparit","subgroup analysis","demographic",
        "race","ethnicit","sex difference","age group","socioeconomic","income",
        "education","rural","urban","underrepresent","health disparit","algorithmic bias",
        "model bias","selection bias","confounding","covariate shift",
    ],
    "performance_metrics": [
        "auc","auroc","c-statistic","area under the curve","sensitivity","specificity",
        "positive predictive","negative predictive","ppv","npv","f1","f-score",
        "accuracy","precision","recall","brier score","calibration","r-squared",
        "mean absolute error","mae","rmse","concordance","kappa",
    ],
    "explainability": [
        "shap","shapley","lime","explainab","interpretab","feature importance",
        "feature attribution","saliency","attention weight","counterfactual",
        "model explanation","black-box","white-box","glass-box","transparent model",
        "local explanation","global explanation","variable importance",
    ],
    "external_validation": [
        "external validation","independent cohort","multi-site","multisite",
        "multicenter","prospective validation","temporal validation","geographic validation",
        "transfer learn","generalizab","transportab","portab","replication study",
        "validation cohort","test cohort","external test",
    ],
    "uncertainty_quantification": [
        "confidence interval","uncertainty","prediction interval","credible interval",
        "probabilistic prediction","bayesian","monte carlo","bootstrap","standard error",
        "variance","model uncertainty","epistemic","aleatoric","reliability diagram",
    ],
    "clinical_utility": [
        "clinical utility","decision curve","net benefit","nri","idi","reclassification",
        "clinical impact","clinical relevance","actionab","clinical significance",
        "incremental value","added value","clinical benefit","net reclassification",
        "decision analytic","clinical meaningf",
    ],
    "deployment_implementation": [
        "deployment","implement","workflow integration","ehr integration","clinical adoption",
        "real-world deployment","point-of-care","embedded","clinical workflow",
        "integration","electronic health record integration","alert","cds tool",
        "clinical decision support system","pilot","feasib",
    ],
    "safety_monitoring": [
        "patient safety","adverse event","harm","risk mitigation","monitoring",
        "surveillance","model drift","concept drift","distribution shift","failure mode",
        "error analysis","false positive rate","false negative rate","missed diagnosis",
        "alert fatigue","overrid","clinical risk",
    ],
    "regulatory_ethics": [
        "fda","regulatory","clearance","510k","de novo","approval","ce mark",
        "irb","institutional review","informed consent","data governance",
        "hipaa","privacy","data protection","ethical","ethics committee",
        "waiver of consent","deidentif","anonymiz",
    ],
}
NIH_AI_ORDER = list(NIH_AI.keys())
NIH_AI_N     = len(NIH_AI_ORDER)

# ── Operational performance dimensions ────────────────────────────────────────
OP_PERF: dict[str, list[str]] = {
    "process_capacity": [
        "process capacity","surge capacity","bed capacity","capacity planning",
        "resource capacity","system capacity","staffing capacity","capacity constraint",
        "capacity utilization","throughput capacity","bottleneck",
    ],
    "human_resources": [
        "human resource","staffing","workforce","staff burden","clinician workload",
        "nurse workload","physician time","labor","personnel","staff time",
        "provider burden","clinician time","staffing level","full-time equivalent","fte",
    ],
    "cost": [
        "cost-effectiveness","cost-benefit","economic analysis","financial","healthcare cost",
        "expenditure","spending","cost reduction","cost saving","resource utilization",
        "budget","reimbursement","cost per","economic burden","cost-utility","qaly",
        "return on investment","roi","economic evaluation",
    ],
    "process_throughput": [
        "throughput","efficiency","turnaround time","wait time","waiting time",
        "length of stay","cycle time","processing time","queue","bottleneck",
        "workflow efficiency","operational efficiency","time-to-treatment","time-to-result",
        "time to","door-to-","discharge time","emergency department flow",
    ],
    "improved_outcomes": [
        "improved outcome","better outcome","outcome improvement","clinical benefit",
        "quality improvement","improved care","better care","improved performance",
        "positive outcome","favorable outcome","enhanced outcome",
    ],
    "improved_healthcare_outcomes": [
        "healthcare outcome","health outcome","mortality reduction","morbidity reduction",
        "readmission reduction","complication reduction","adverse event reduction",
        "reduced mortality","reduced morbidity","reduced readmission","clinical outcome improvement",
        "improved survival","reduced complication","hospitaliz",
    ],
    "improved_process_performance": [
        "process performance","workflow improvement","operational improvement",
        "process optimization","quality metric","performance improvement","lean",
        "six sigma","process redesign","efficiency gain","reduced error",
        "accuracy improvement","model performance improvement","process mining insight",
    ],
    "improved_patient_outcomes": [
        "patient outcome","patient benefit","patient safety improvement",
        "functional outcome","quality of life","patient-reported outcome","pro",
        "patient satisfaction","patient experience","pain reduction","symptom improvement",
        "treatment response","therapeutic outcome","opioid reduction","overdose reduction",
    ],
}
OP_PERF_ORDER = list(OP_PERF.keys())
OP_PERF_N     = len(OP_PERF_ORDER)

# ── Helpers ───────────────────────────────────────────────────────────────────
_JSON_INDEX = {p.stem: p for p in SCHOLAR_JSON.glob("*.json")}

def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]", " ", (s or "").lower())

def _load_text(pmc_id: str, article_id: str) -> str:
    path = _JSON_INDEX.get(pmc_id) or _JSON_INDEX.get(f"article_{article_id}")
    if not path:
        return ""
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        return (_norm(obj.get("abstract", "") or "") + " " +
                _norm((obj.get("full_text", "") or "")[:6000]))
    except Exception:
        return ""

def _tag(text: str, domain_dict: dict) -> list[str]:
    """Return list of domain keys with ≥1 keyword hit."""
    hits = []
    for domain, kws in domain_dict.items():
        if any(kw in text for kw in kws):
            hits.append(domain)
    return hits

# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--min-score", type=int, default=0,
                        help="Filter summary to rows with nih_ai_score >= N")
    args = parser.parse_args()

    rows = list(csv.DictReader(open(SCREENED, encoding="utf-8-sig")))

    # Target subset: decide+act × modeling+evaluation
    TARGET_OODA  = {"decide", "act"}
    TARGET_CRISP = {"modeling", "evaluation"}

    subset_ids = set()
    for r in rows:
        o = (r.get("ooda_phase_primary") or "").lower().strip()
        c = (r.get("crisp_dm_phase") or "").lower().strip()
        if o in TARGET_OODA and c in TARGET_CRISP:
            subset_ids.add(r["article_id"])

    print(f"Total articles        : {len(rows):,}")
    print(f"Decide+Act × model/eval subset: {len(subset_ids):,}")
    print(f"scholar_json/ available: {len(_JSON_INDEX):,}\n")

    tag_rows = []
    nih_domain_counter  = Counter()
    op_perf_counter     = Counter()
    score_hist          = Counter()

    for row in rows:
        if row["article_id"] not in subset_ids:
            row["nih_ai_tags"]   = ""
            row["nih_ai_score"]  = ""
            row["nih_ai_pct"]    = ""
            row["op_perf_tags"]  = ""
            row["op_perf_score"] = ""
            continue

        title       = row.get("title", "") or ""
        key_phrases = row.get("key_phrases", "") or ""
        pmc_id      = row.get("pmc_id", "").strip()
        article_id  = row.get("article_id", "").strip()
        full        = _load_text(pmc_id, article_id)
        text        = _norm(title + " " + key_phrases) + " " + full

        nih_tags  = _tag(text, NIH_AI)
        op_tags   = _tag(text, OP_PERF)
        nih_score = len(nih_tags)
        op_score  = len(op_tags)
        nih_pct   = round(nih_score / NIH_AI_N * 100, 1)

        row["nih_ai_tags"]   = "|".join(nih_tags)
        row["nih_ai_score"]  = nih_score
        row["nih_ai_pct"]    = nih_pct
        row["op_perf_tags"]  = "|".join(op_tags)
        row["op_perf_score"] = op_score

        for d in nih_tags:
            nih_domain_counter[d] += 1
        for d in op_tags:
            op_perf_counter[d] += 1
        score_hist[nih_score] += 1

        if nih_score >= args.min_score:
            tag_rows.append({
                "article_id":      article_id,
                "pmc_id":          pmc_id,
                "title":           title[:90],
                "human_decision":  row.get("human_decision", ""),
                "ooda_phase":      row.get("ooda_phase_primary", ""),
                "crisp_dm_phase":  row.get("crisp_dm_phase", ""),
                "nih_ai_score":    nih_score,
                "nih_ai_pct":      nih_pct,
                "nih_ai_tags":     "|".join(nih_tags),
                "op_perf_score":   op_score,
                "op_perf_tags":    "|".join(op_tags),
            })

    tag_rows.sort(key=lambda r: (-int(r["nih_ai_score"] or 0), r["article_id"]))

    # ── Console summary ────────────────────────────────────────────────────────
    print("NIH AI Checklist domain coverage (subset articles):")
    for d in NIH_AI_ORDER:
        n = nih_domain_counter[d]
        bar = "█" * (n // 5)
        print(f"  {d:<35} {n:>5}  {bar}")

    print()
    print("Operational performance tag coverage:")
    for d in OP_PERF_ORDER:
        n = op_perf_counter[d]
        bar = "█" * (n // 5)
        print(f"  {d:<35} {n:>5}  {bar}")

    print()
    print("NIH AI score distribution (# checklist domains addressed):")
    for s in sorted(score_hist):
        n = score_hist[s]
        bar = "█" * (n // 5)
        print(f"  score {s:>2} : {n:>4}  {bar}")

    inc_with_tags = sum(1 for r in tag_rows
                        if r["human_decision"] == "include" and int(r["nih_ai_score"] or 0) >= 1)
    inc_high      = sum(1 for r in tag_rows
                        if r["human_decision"] == "include" and int(r["nih_ai_score"] or 0) >= 6)
    print(f"\nIncluded subset with ≥1 NIH AI domain  : {inc_with_tags}")
    print(f"Included subset with ≥6 NIH AI domains  : {inc_high}  (≥50% checklist)")

    if args.dry_run:
        print("\n[dry-run] No files written.")
        return

    # ── Write articles_screened.csv ───────────────────────────────────────────
    fieldnames = list(rows[0].keys())
    for col in ("nih_ai_tags","nih_ai_score","nih_ai_pct","op_perf_tags","op_perf_score"):
        if col not in fieldnames:
            fieldnames.append(col)
    with open(SCREENED, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"\n✓ articles_screened.csv  +nih_ai_tags +nih_ai_score +op_perf_tags +op_perf_score")

    # ── Write tags CSV ─────────────────────────────────────────────────────────
    if tag_rows:
        with open(TAGS_CSV, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(tag_rows[0].keys()))
            w.writeheader()
            w.writerows(tag_rows)
        print(f"✓ {TAGS_CSV}  ({len(tag_rows)} rows)")

    # ── Write markdown maps ────────────────────────────────────────────────────
    for out_path in (SUMMARY_MD, MAP_MD):
        _write_md(out_path, tag_rows, nih_domain_counter, op_perf_counter, score_hist)
    print(f"✓ {SUMMARY_MD}")
    print(f"✓ {MAP_MD}")


def _write_md(path: Path, tag_rows, nih_ct, op_ct, score_hist):
    subset_n  = sum(score_hist.values())
    inc_rows  = [r for r in tag_rows if r["human_decision"] == "include"]
    high_rows = [r for r in inc_rows if int(r["nih_ai_score"] or 0) >= 6]

    lines = [
        "# NIH AI Checklist × Operational Performance Tags",
        "",
        "> **Subset**: Decide+Act articles with crisp_dm_phase = modeling or evaluation",
        f"> **N (subset)**: {subset_n:,}  |  **N (included)**: {len(inc_rows):,}",
        f"> **12 NIH AI checklist domains**  |  **8 operational performance dimensions**",
        "",
        "## NIH AI Checklist Domain Coverage",
        "",
        "| Domain | Articles (subset) | Description |",
        "|--------|:-----------------:|-------------|",
    ]
    desc = {
        "study_design":            "Study design, prospective/retrospective, trial type",
        "data_reporting":          "Training/test data, sample size, splits, cohort",
        "model_transparency":      "Reproducibility, open code, model documentation",
        "bias_fairness":           "Bias assessment, equity, subgroup, demographic parity",
        "performance_metrics":     "AUC, sensitivity/specificity, calibration, F1",
        "explainability":          "SHAP, LIME, feature importance, interpretability",
        "external_validation":     "Independent/multi-site/prospective validation",
        "uncertainty_quantification": "CIs, prediction intervals, Bayesian uncertainty",
        "clinical_utility":        "Decision curve, net benefit, NRI/IDI, clinical impact",
        "deployment_implementation": "EHR integration, workflow, clinical adoption",
        "safety_monitoring":       "Patient safety, model drift, failure modes",
        "regulatory_ethics":       "FDA, IRB, HIPAA, ethics, data governance",
    }
    for d in NIH_AI_ORDER:
        lines.append(f"| `{d}` | {nih_ct[d]} | {desc.get(d,'')} |")

    lines += [
        "",
        "## Score Distribution (# NIH AI domains addressed per article)",
        "",
        "| Score | Articles | Bar |",
        "|------:|---------:|-----|",
    ]
    for s in sorted(score_hist):
        n   = score_hist[s]
        bar = "█" * min(n // 3, 40)
        lines.append(f"| {s} | {n} | {bar} |")

    lines += [
        "",
        "## Operational Performance Coverage",
        "",
        "| Dimension | Articles | Description |",
        "|-----------|:--------:|-------------|",
    ]
    op_desc = {
        "process_capacity":           "Capacity planning, surge, bottleneck analysis",
        "human_resources":            "Staffing, workforce, clinician workload/burden",
        "cost":                       "Cost-effectiveness, economic analysis, ROI",
        "process_throughput":         "Wait time, LOS, turnaround, queue, flow",
        "improved_outcomes":          "General outcome improvement, clinical benefit",
        "improved_healthcare_outcomes": "Mortality/morbidity/readmission reduction",
        "improved_process_performance": "Workflow, process optimization, quality metrics",
        "improved_patient_outcomes":  "QoL, functional, patient-reported outcomes",
    }
    for d in OP_PERF_ORDER:
        lines.append(f"| `{d}` | {op_ct[d]} | {op_desc.get(d,'')} |")

    lines += [
        "",
        f"## High-Coverage Included Articles (NIH AI score ≥ 6 of 12)",
        "",
        "| Rank | Article ID | Score | OODA | CRISP-DM | NIH AI Tags | Op Perf Tags |",
        "|-----:|-----------|------:|------|----------|-------------|--------------|",
    ]
    for i, r in enumerate(high_rows[:50], 1):
        lines.append(
            f"| {i} | [{r['article_id']}] {r['title'][:55]}… "
            f"| {r['nih_ai_score']} | {r['ooda_phase']} | {r['crisp_dm_phase']} "
            f"| {r['nih_ai_tags']} | {r['op_perf_tags']} |"
        )
    if len(high_rows) > 50:
        lines.append(f"| … | +{len(high_rows)-50} more articles | | | | | |")

    lines += [
        "",
        "## All Tagged Articles (score ≥ 1, sorted by score)",
        "",
    ]
    for r in tag_rows[:200]:
        score = int(r["nih_ai_score"] or 0)
        lines.append(
            f"- **[{score}/{NIH_AI_N}]** [{r['article_id']}] {r['title'][:70]}  \n"
            f"  NIH: `{r['nih_ai_tags']}`  \n"
            f"  OpPerf: `{r['op_perf_tags']}`"
        )
    if len(tag_rows) > 200:
        lines.append(f"\n… and {len(tag_rows)-200} more — see `{TAGS_CSV.name}` for full list.")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
