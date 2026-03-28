"""
_generate_vcu_queue.py
─────────────────────────────────────────────────────────────────────────
Generates a VCU library download queue for articles missing full-text JSON.

For each article missing from data/scholar_json/:
  • Has real PMC ID  → PMC page via VCU proxy
  • No PMC ID (HSH)  → PubMed title search via VCU proxy

Outputs:
  scripts/vcu_fulltext_queue.csv                  ← machine-readable queue
  infrastructure_setup/manual_review/TO_DOWNLOAD_FULLTEXT.md ← human checklist

After VCU downloads:
  1. Save PDFs to data/vcu_downloads/{pmc_id_or_id}.pdf
  2. Run: python scripts/_import_vcu_pdfs.py
     → extracts text → saves to data/scholar_json/

Usage:
  python scripts/_generate_vcu_queue.py                     # included only
  python scripts/_generate_vcu_queue.py --all               # include + borderline excluded
  python scripts/_generate_vcu_queue.py --min-score 0.05    # custom threshold
  python scripts/_generate_vcu_queue.py --dry-run
"""
import argparse, csv, json, urllib.parse
from pathlib import Path
from collections import Counter

SCREENED      = Path("data/ontology/articles_screened.csv")
SCHOLAR_JSON  = Path("data/scholar_json")
VCU_DL        = Path("data/vcu_downloads")
QUEUE_CSV     = Path("scripts/vcu_fulltext_queue.csv")
REVIEW_HUB    = Path(r"C:\Projects\pgx-analysis\manuscript\infrastructure_setup\manual_review")
OUT_MD        = REVIEW_HUB / "TO_DOWNLOAD_FULLTEXT.md"

VCU_PROXY     = "https://proxy.library.vcu.edu/login?url="
PMC_BASE      = "https://www.ncbi.nlm.nih.gov/pmc/articles/"
PUBMED_SEARCH = "https://pubmed.ncbi.nlm.nih.gov/?term="

QUEUE_FIELDS  = [
    "priority", "article_id", "pmc_id", "title", "human_decision",
    "composite_score", "proxy_url", "fallback_url", "status",
]

def proxy(url: str) -> str:
    return VCU_PROXY + urllib.parse.quote(url, safe=":/?.=&%#")

def build_urls(pmc_id: str, title: str) -> tuple[str, str]:
    """Return (primary_proxy_url, fallback_proxy_url)."""
    if pmc_id.startswith("PMC"):
        primary  = proxy(PMC_BASE + pmc_id + "/")
        fallback = proxy(PUBMED_SEARCH + pmc_id)
    else:
        # HSH ID or no ID — search by title
        q        = urllib.parse.quote(title[:80])
        primary  = proxy(PUBMED_SEARCH + q)
        fallback = f"https://scholar.google.com/scholar?q={q}"
    return primary, fallback

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all",       action="store_true",
                        help="Include borderline excluded (score >= min-score)")
    parser.add_argument("--min-score", type=float, default=0.10,
                        help="Min composite_score for excluded articles (default 0.10)")
    parser.add_argument("--dry-run",   action="store_true")
    args = parser.parse_args()

    json_index = {p.stem for p in SCHOLAR_JSON.glob("*.json")}
    rows = list(csv.DictReader(open(SCREENED, encoding="utf-8-sig")))

    queue = []
    for row in rows:
        pmc_id     = row.get("pmc_id", "").strip()
        article_id = row.get("article_id", "").strip()
        out_id     = pmc_id if pmc_id else f"article_{article_id}"
        decision   = row.get("human_decision", "")
        title      = row.get("title", "") or ""

        try:
            score = float(row.get("composite_score", 0) or 0)
        except:
            score = 0.0

        # Already have full text — skip
        if out_id in json_index:
            continue

        # Decision filter
        if decision == "include":
            pass   # always include
        elif decision == "exclude" and args.all and score >= args.min_score:
            pass   # borderline excluded with --all
        else:
            continue

        primary, fallback = build_urls(pmc_id, title)

        queue.append({
            "priority":        1 if decision == "include" else 2,
            "article_id":      article_id,
            "pmc_id":          pmc_id,
            "title":           title[:120],
            "human_decision":  decision,
            "composite_score": score,
            "proxy_url":       primary,
            "fallback_url":    fallback,
            "status":          "needed",
        })

    # Sort by priority then score
    queue.sort(key=lambda r: (r["priority"], -r["composite_score"]))

    pmc_count = sum(1 for r in queue if r["pmc_id"].startswith("PMC"))
    hsh_count = sum(1 for r in queue if not r["pmc_id"].startswith("PMC"))

    print(f"Articles needing VCU library download: {len(queue)}")
    print(f"  Included missing full text : {sum(1 for r in queue if r['human_decision']=='include')}")
    print(f"  Borderline excluded        : {sum(1 for r in queue if r['human_decision']=='exclude')}")
    print(f"  Via PMC proxy URL          : {pmc_count}")
    print(f"  Via title search           : {hsh_count}")
    print()

    if args.dry_run:
        print("[dry-run] Files not written.")
        return

    if not queue:
        print("No articles need VCU downloads. scholar_json/ is complete for included articles.")
        return

    VCU_DL.mkdir(exist_ok=True)
    REVIEW_HUB.mkdir(parents=True, exist_ok=True)

    # ── Write CSV queue ────────────────────────────────────────────────────────
    with open(QUEUE_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=QUEUE_FIELDS)
        w.writeheader()
        w.writerows(queue)
    print(f"✓ Queue CSV  → {QUEUE_CSV}  ({len(queue)} articles)")

    # ── Write markdown checklist ───────────────────────────────────────────────
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("# VCU Library Full-Text Download Queue\n\n")
        f.write(f"> {len(queue)} articles need full text via VCU institutional access.\n")
        f.write(f"> Save each PDF to: `data/vcu_downloads/{{pmc_id_or_article_id}}.pdf`\n")
        f.write(f"> Then run: `python scripts/_import_vcu_pdfs.py`\n\n")
        f.write("## Instructions\n\n")
        f.write("1. Click the **Proxy URL** link — logs into VCU proxy automatically\n")
        f.write("2. Download the PDF from the journal page\n")
        f.write("3. Save as `{pmc_id}.pdf` (e.g. `PMC12345678.pdf`) in `data/vcu_downloads/`\n")
        f.write("4. Check `[ ]` box when done\n")
        f.write("5. When all done: `python scripts/_import_vcu_pdfs.py`\n\n")
        f.write("---\n\n")

        # Group by decision
        for decision_label, decision_key in [("Priority 1 — Included Articles", "include"),
                                              ("Priority 2 — Borderline Excluded", "exclude")]:
            subset = [r for r in queue if r["human_decision"] == decision_key]
            if not subset:
                continue
            f.write(f"## {decision_label} ({len(subset)} articles)\n\n")
            for r in subset:
                f.write(f"- [ ] **{r['pmc_id'] or r['article_id']}**  ")
                f.write(f"score={r['composite_score']:.3f}  ")
                f.write(f"{r['title'][:70]}\n")
                f.write(f"  - [Proxy URL]({r['proxy_url']})\n")
                if r["fallback_url"] != r["proxy_url"]:
                    f.write(f"  - [Fallback]({r['fallback_url']})\n")
                f.write("\n")

    print(f"✓ Checklist  → {OUT_MD}")
    print(f"\nNext step:")
    print(f"  1. Open {OUT_MD}")
    print(f"  2. Download PDFs → save to data/vcu_downloads/")
    print(f"  3. python scripts/_import_vcu_pdfs.py")


if __name__ == "__main__":
    main()
