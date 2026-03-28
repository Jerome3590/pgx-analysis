"""
_build_vcu_doi_map.py
─────────────────────────────────────────────────────────────────────────
Resolves DOIs for all articles in the VCU download queue and writes
scripts/vcu_queue_with_dois.csv, ready for vcu_download.js --input.

DOI lookup strategy (in order):
  1. NCBI ESummary  — for articles with real PMC IDs (PMC*)
  2. CrossRef API   — title-based lookup for articles without PMC ID
  3. Europe PMC     — secondary title search fallback

Output columns (same schema as screened_doi_map.csv so vcu_download.js
can consume it unchanged):
  screened_pmc_id, doi, title, article_id, pmc_id, score, decision, doi_source

Idempotent: already-resolved entries in the output file are skipped.

Usage:
  python scripts/_build_vcu_doi_map.py              # resolve all VCU queue
  python scripts/_build_vcu_doi_map.py --limit 200  # first 200 (test run)
  python scripts/_build_vcu_doi_map.py --pmc-only   # only PMC-ID articles
  python scripts/_build_vcu_doi_map.py --dry-run    # show queue without hitting APIs
"""
import argparse, csv, json, re, time
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter, Retry

VCU_QUEUE    = Path("scripts/vcu_fulltext_queue.csv")
OUT_CSV      = Path("scripts/vcu_queue_with_dois.csv")
SCHOLAR_JSON = Path("data/scholar_json")

EMAIL = "dixonrj@vcu.edu"

# ── HTTP session ───────────────────────────────────────────────────────────────
SESSION = requests.Session()
SESSION.mount("https://", HTTPAdapter(max_retries=Retry(total=3, backoff_factor=1,
                                                        status_forcelist=[429, 500, 503])))
SESSION.headers.update({"User-Agent": f"LitReviewBot/1.0 (mailto:{EMAIL})"})

NCBI_ESUMMARY = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
CROSSREF_BASE = "https://api.crossref.org/works"
EPMC_SEARCH   = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"


def _numeric(pmc_id: str) -> str:
    """Strip 'PMC' prefix → numeric string."""
    return re.sub(r"[^0-9]", "", pmc_id)


def lookup_ncbi(pmc_id: str, api_key: str = "") -> str:
    """NCBI ESummary: PMC ID → DOI.  Returns '' if not found."""
    num = _numeric(pmc_id)
    if not num:
        return ""
    params = {"db": "pmc", "id": num, "retmode": "json"}
    if api_key:
        params["api_key"] = api_key
    try:
        r = SESSION.get(NCBI_ESUMMARY, params=params, timeout=12)
        if r.status_code != 200:
            return ""
        data = r.json()
        uid  = str(num)
        result = (data.get("result") or {}).get(uid, {})
        doi = result.get("doi", "")
        if doi:
            return doi.strip()
        # Sometimes in articleids list
        for aid in result.get("articleids", []):
            if aid.get("idtype") == "doi":
                return aid["value"].strip()
    except Exception:
        pass
    return ""


def lookup_crossref(title: str) -> str:
    """CrossRef title search → DOI."""
    if not title:
        return ""
    try:
        params = {"query.title": title[:200], "rows": 1,
                  "mailto": EMAIL, "select": "DOI,title,score"}
        r = SESSION.get(CROSSREF_BASE, params=params, timeout=12)
        if r.status_code == 200:
            items = r.json().get("message", {}).get("items", [])
            if items and items[0].get("score", 0) > 60:
                return items[0].get("DOI", "").strip()
    except Exception:
        pass
    return ""


def lookup_epmc(title: str) -> str:
    """Europe PMC title search → DOI (fallback)."""
    if not title:
        return ""
    try:
        params = {"query": f'TITLE:"{title[:100]}"', "format": "json",
                  "resultType": "core", "pageSize": 1}
        r = SESSION.get(EPMC_SEARCH, params=params, timeout=10)
        if r.status_code == 200:
            results = r.json().get("resultList", {}).get("result", [])
            if results:
                doi = results[0].get("doi", "")
                if doi:
                    return doi.strip()
    except Exception:
        pass
    return ""


def load_done() -> dict:
    """Load already-resolved entries keyed by article_id."""
    done = {}
    if OUT_CSV.exists():
        for row in csv.DictReader(open(OUT_CSV, encoding="utf-8-sig")):
            if row.get("doi"):
                done[row["article_id"]] = row["doi"]
    return done


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit",    type=int, default=0)
    parser.add_argument("--pmc-only", action="store_true",
                        help="Only process articles with real PMC IDs")
    parser.add_argument("--api-key",  default="",
                        help="NCBI API key (10 req/s vs 3 req/s)")
    parser.add_argument("--dry-run",  action="store_true")
    args = parser.parse_args()

    if not VCU_QUEUE.exists():
        print(f"ERROR: {VCU_QUEUE} not found. Run: python scripts/_generate_vcu_queue.py")
        return

    queue = list(csv.DictReader(open(VCU_QUEUE, encoding="utf-8-sig")))
    json_index = {p.stem for p in SCHOLAR_JSON.glob("*.json")}
    done  = load_done()

    # Filter out articles that already have full-text JSON
    queue = [r for r in queue if
             (r.get("pmc_id") or f"article_{r['article_id']}") not in json_index]

    if args.pmc_only:
        queue = [r for r in queue if r.get("pmc_id", "").startswith("PMC")]
    if args.limit:
        queue = queue[:args.limit]

    need_doi    = [r for r in queue if r["article_id"] not in done]
    pmc_count   = sum(1 for r in need_doi if r.get("pmc_id","").startswith("PMC"))
    title_count = len(need_doi) - pmc_count

    print(f"VCU queue total         : {len(queue)}")
    print(f"Already resolved (DOI)  : {len(done)}")
    print(f"Need DOI lookup         : {len(need_doi)}")
    print(f"  Via NCBI (PMC ID)     : {pmc_count}")
    print(f"  Via CrossRef (title)  : {title_count}")

    if args.dry_run:
        print("[dry-run] No API calls made.")
        return

    OUT_FIELDS = ["screened_pmc_id","doi","title","article_id",
                  "pmc_id","score","decision","doi_source"]

    # Open output file in append mode (idempotent)
    write_header = not OUT_CSV.exists()
    out_fh = open(OUT_CSV, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(out_fh, fieldnames=OUT_FIELDS, extrasaction="ignore")
    if write_header:
        writer.writeheader()

    ncbi_sleep   = 0.11 if args.api_key else 0.34   # 10/s vs 3/s limit
    cross_sleep  = 0.5

    found_doi = 0
    no_doi    = 0

    for i, row in enumerate(need_doi):
        article_id = row["article_id"]
        pmc_id     = row.get("pmc_id","").strip()
        title      = row.get("title","").strip()
        score      = row.get("composite_score","")
        decision   = row.get("human_decision","")

        print(f"  [{i+1:5}/{len(need_doi)}] {pmc_id or article_id:<15}  {title[:55]}", end="  ")

        doi        = ""
        doi_source = ""

        # ── 1. NCBI ESummary for real PMC IDs ─────────────────────────────────
        if pmc_id.startswith("PMC"):
            doi = lookup_ncbi(pmc_id, args.api_key)
            if doi:
                doi_source = "ncbi_esummary"
            time.sleep(ncbi_sleep)

        # ── 2. CrossRef title search ───────────────────────────────────────────
        if not doi and title:
            doi = lookup_crossref(title)
            if doi:
                doi_source = "crossref"
            time.sleep(cross_sleep)

        # ── 3. Europe PMC fallback ─────────────────────────────────────────────
        if not doi and title:
            doi = lookup_epmc(title)
            if doi:
                doi_source = "epmc"
            time.sleep(0.3)

        if doi:
            found_doi += 1
            print(f"✓  {doi[:50]}  [{doi_source}]")
        else:
            no_doi += 1
            print("—  no DOI found")

        writer.writerow({
            "screened_pmc_id": pmc_id or f"article_{article_id}",
            "doi":              doi,
            "title":            title[:120],
            "article_id":       article_id,
            "pmc_id":           pmc_id,
            "score":            score,
            "decision":         decision,
            "doi_source":       doi_source,
        })
        out_fh.flush()

    out_fh.close()

    # Also write entries already in `done` that aren't in output yet (full refresh)
    print(f"\n── DOI lookup complete ────────────────────────────────")
    print(f"  DOIs resolved   : {found_doi + len(done)}")
    print(f"  No DOI found    : {no_doi}")
    total_with_doi = sum(1 for r in csv.DictReader(open(OUT_CSV, encoding="utf-8-sig")) if r["doi"])
    print(f"  Output CSV rows : {total_with_doi} with DOI  →  {OUT_CSV}")
    print(f"\nNext:")
    print(f"  # Free OA pass (EuropePMC / CORE / SemanticScholar):")
    print(f"  python scripts/scholar_lookup.py --vcu-queue")
    print(f"  # VCU proxy pass (paywalled — needs Puppeteer + VCU credentials):")
    print(f"  node scripts/vcu_download.js --input scripts/vcu_queue_with_dois.csv")


if __name__ == "__main__":
    main()
