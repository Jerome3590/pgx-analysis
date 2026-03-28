"""
_fetch_missing_fulltext.py
──────────────────────────────────────────────────────────────────────────────
Idempotent: fills gaps in data/scholar_json/ for all screened articles.

For each article in articles_screened.csv that lacks a scholar_json/ entry:
  • Has PMC ID  → fetch BioC JSON from NCBI PMC API → parse → save to scholar_json/
  • No PMC ID   → try Unpaywall for OA PDF URL → download PDF → extract text → save

Safe to re-run at any time. Already-present JSON files are NEVER overwritten.
Progress is logged to scripts/fulltext_fetch_log.csv (idempotent: skips logged
successes on re-run).

Usage:
  python scripts/_fetch_missing_fulltext.py                   # all missing
  python scripts/_fetch_missing_fulltext.py --priority-only   # score >= 0.10 first
  python scripts/_fetch_missing_fulltext.py --limit 200       # batch of 200
  python scripts/_fetch_missing_fulltext.py --api-key NCBI_KEY --workers 4
  python scripts/_fetch_missing_fulltext.py --dry-run         # count only
"""
import argparse, csv, json, re, time
from datetime import datetime, timezone
from pathlib import Path
import requests

# ── Paths ──────────────────────────────────────────────────────────────────────
SCREENED      = Path("data/ontology/articles_screened.csv")
SCHOLAR_JSON  = Path("data/scholar_json")
PDF_DIR       = Path("data/scholar_pdfs")
FETCH_LOG     = Path("scripts/fulltext_fetch_log.csv")
LOG_FIELDS    = ["article_id", "pmc_id", "method", "status", "bytes", "timestamp"]

# NCBI PMC BioC endpoint
BIOC_URL      = "https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi/BioC_json/{pmc_id}/unicode"
UNPAYWALL_URL = "https://api.unpaywall.org/v2/{doi}?email=dixonrj@vcu.edu"

SLEEP_NO_KEY  = 0.35
SLEEP_API_KEY = 0.11
MAX_RETRIES   = 3
NOW           = lambda: datetime.now(timezone.utc).isoformat()

SCHOLAR_JSON.mkdir(exist_ok=True)

# ── Logging ───────────────────────────────────────────────────────────────────
def _append_log(row: dict):
    exists = FETCH_LOG.exists()
    with open(FETCH_LOG, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        if not exists:
            w.writeheader()
        w.writerow(row)

def _load_log(retry_failed: bool = False) -> set[str]:
    """Return set of article_ids already attempted.
    Skips ALL logged articles by default; use --retry-failed to re-try non-ok entries.
    """
    if not FETCH_LOG.exists():
        return set()
    done = set()
    for r in csv.DictReader(open(FETCH_LOG, encoding="utf-8-sig")):
        if not retry_failed or r.get("status") == "ok":
            done.add(r["article_id"])
    return done


# ── PMC BioC download + parse ─────────────────────────────────────────────────
def _fetch_bioc(pmc_id: str, sleep_s: float) -> bytes | None:
    url = BIOC_URL.format(pmc_id=pmc_id)
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = requests.get(url, timeout=30)
            if r.status_code == 200 and len(r.content) > 100:
                time.sleep(sleep_s)
                return r.content
            if r.status_code == 404:
                return None
            time.sleep(sleep_s * attempt)
        except requests.RequestException:
            time.sleep(sleep_s * attempt)
    return None


def _parse_bioc(raw: bytes, pmc_id: str) -> dict:
    """Parse BioC JSON bytes into unified scholar_json schema."""
    obj = json.loads(raw.decode("utf-8"))
    root = obj[0] if isinstance(obj, list) else obj
    documents = root.get("documents", [])
    doc = documents[0] if documents else {}
    passages = doc.get("passages", [])

    infons = passages[0].get("infons", {}) if passages else {}
    doi    = infons.get("article-id_doi", "")
    pmid   = infons.get("article-id_pmid", "")
    year   = infons.get("year", "")
    journal = infons.get("journal", "")
    kwds   = infons.get("kwd", "").split() if infons.get("kwd") else []

    authors, i = [], 0
    while f"name_{i}" in infons:
        parts = dict(p.split(":") for p in infons[f"name_{i}"].split(";") if ":" in p)
        name = f"{parts.get('given-names','')} {parts.get('surname','')}".strip()
        if name:
            authors.append(name)
        i += 1

    sections, abstract_parts, full_parts, title = [], [], [], ""
    for p in passages:
        ptype = p.get("infons", {}).get("section_type", "")
        text  = p.get("text", "").strip()
        if not text:
            continue
        full_parts.append(text)
        if ptype == "TITLE":
            title = text
        elif ptype in ("ABSTRACT", "ABSTRACT_SUB"):
            abstract_parts.append(text)
        else:
            sections.append({"label": p.get("infons", {}).get("type", ptype), "text": text})

    return {
        "id":          pmc_id,
        "source_type": "pmc",
        "title":       title,
        "doi":         doi,
        "pmc_id":      pmc_id,
        "pmid":        pmid,
        "year":        year,
        "authors":     authors,
        "journal":     journal,
        "abstract":    " ".join(abstract_parts),
        "full_text":   "\n\n".join(full_parts),
        "word_count":  len(" ".join(full_parts).split()),
        "sections":    sections,
        "keywords":    kwds,
        "metadata":    {"extracted_at": NOW(), "source_file": "pmc_api", "page_count": 0},
    }


# ── Unpaywall + PDF extraction ────────────────────────────────────────────────
def _fetch_via_unpaywall(article_id: str, doi: str, title: str, sleep_s: float) -> dict | None:
    """Try Unpaywall for OA PDF URL, download PDF, extract text."""
    if not doi:
        return None
    try:
        r = requests.get(UNPAYWALL_URL.format(doi=doi), timeout=15)
        if r.status_code != 200:
            return None
        data = r.json()
        best = data.get("best_oa_location") or {}
        pdf_url = best.get("url_for_pdf") or best.get("url")
        if not pdf_url or not pdf_url.endswith(".pdf"):
            return None
        pr = requests.get(pdf_url, timeout=30)
        if pr.status_code != 200 or len(pr.content) < 1000:
            return None
        pdf_path = PDF_DIR / f"article_{article_id}.pdf"
        pdf_path.write_bytes(pr.content)
        time.sleep(sleep_s)

        # Extract text
        from pdfminer.high_level import extract_pages
        from pdfminer.layout import LTTextContainer
        pages, full_parts = [], []
        for pg_num, layout in enumerate(extract_pages(str(pdf_path)), 1):
            pg_text = " ".join(el.get_text() for el in layout
                               if isinstance(el, LTTextContainer))
            pages.append({"page": pg_num, "text": pg_text})
            full_parts.append(pg_text)
        full_text = "\n\n".join(full_parts)

        return {
            "id":          f"article_{article_id}",
            "source_type": "pdf",
            "title":       title,
            "doi":         doi,
            "pmc_id":      "",
            "pmid":        "",
            "year":        "",
            "authors":     [],
            "journal":     "",
            "abstract":    "",
            "full_text":   full_text,
            "word_count":  len(full_text.split()),
            "sections":    [],
            "keywords":    [],
            "metadata":    {"extracted_at": NOW(), "source_file": str(pdf_path),
                            "page_count": len(pages)},
        }
    except Exception:
        return None


# ── Main ───────────────────────────────────────────────────────────────────────
def run(priority_only: bool = False, limit: int | None = None,
        api_key: str | None = None, dry_run: bool = False,
        workers: int = 1) -> dict:
    """
    Idempotent fill of scholar_json/ gaps.
    Returns summary dict.
    """
    sleep_s = SLEEP_API_KEY if api_key else SLEEP_NO_KEY
    json_index  = {p.stem for p in SCHOLAR_JSON.glob("*.json")}

    rows = list(csv.DictReader(open(SCREENED, encoding="utf-8-sig")))

    # Build work queue: articles missing from scholar_json/ (real JSON or stub = done)
    # Only real PMC IDs (starts with "PMC") are fetched from the API.
    # HSH IDs already have PDFs on disk — their JSON is built by _build_full_json.py --pdfs.
    queue      = []
    hsh_skipped = 0
    for row in rows:
        pmc_id     = row.get("pmc_id", "").strip()
        article_id = row.get("article_id", "").strip()
        out_id     = pmc_id if pmc_id else f"article_{article_id}"

        if out_id in json_index:
            continue                          # already processed (real JSON or stub)
        if pmc_id and not pmc_id.startswith("PMC"):
            hsh_skipped += 1
            continue                          # HSH IDs: use PDF extraction, not API

        try:
            score = float(row.get("composite_score", 0) or 0)
        except:
            score = 0.0

        queue.append({
            "article_id": article_id,
            "pmc_id":     pmc_id,
            "out_id":     out_id,
            "title":      row.get("title", ""),
            "doi":        "",
            "score":      score,
            "decision":   row.get("human_decision", ""),
        })

    # Sort: high score first, then by decision
    queue.sort(key=lambda r: (r["score"]), reverse=True)

    if priority_only:
        queue = [r for r in queue if r["score"] >= 0.10]

    if limit:
        queue = queue[:limit]

    total_q = len(queue)
    pmc_q   = sum(1 for r in queue if r["pmc_id"])
    other_q = total_q - pmc_q

    print(f"scholar_json/ : {len(json_index)} existing files (real + stubs)")
    print(f"HSH IDs skipped (use PDF extract): {hsh_skipped}")
    print(f"Work queue    : {total_q}  ({pmc_q} real PMC IDs, {other_q} no-pmc → VCU queue)")
    if priority_only:
        print(f"  [priority-only mode: score >= 0.10]")
    print()

    if dry_run:
        print("[dry-run] No fetching performed.")
        return {"queued": total_q, "fetched": 0}

    ok = skip = fail = 0
    for i, item in enumerate(queue):
        pmc_id     = item["pmc_id"]
        article_id = item["article_id"]
        out_id     = item["out_id"]
        out_path   = SCHOLAR_JSON / f"{out_id}.json"

        # Double-check (another process may have created it)
        if out_path.exists():
            skip += 1
            continue

        if pmc_id:
            raw = _fetch_bioc(pmc_id, sleep_s)
            if raw:
                try:
                    parsed = _parse_bioc(raw, pmc_id)
                    if not parsed["title"]:
                        parsed["title"] = item["title"]
                    parsed["processed"] = True
                    out_path.write_text(json.dumps(parsed, ensure_ascii=False, indent=2),
                                        encoding="utf-8")
                    _append_log({"article_id": article_id, "pmc_id": pmc_id,
                                 "method": "pmc_api", "status": "ok",
                                 "bytes": len(raw), "timestamp": NOW()})
                    ok += 1
                    if ok % 100 == 0:
                        print(f"  ... {ok} fetched / {i+1} processed / {total_q} queued")
                except Exception as e:
                    _append_log({"article_id": article_id, "pmc_id": pmc_id,
                                 "method": "pmc_api", "status": f"parse_error:{e}",
                                 "bytes": 0, "timestamp": NOW()})
                    fail += 1
            else:
                # Write stub so json_index skips this on next run
                stub = {"processed": True, "status": "not_found", "id": out_id,
                        "title": item["title"], "full_text": "", "word_count": 0}
                out_path.write_text(json.dumps(stub, ensure_ascii=False), encoding="utf-8")
                _append_log({"article_id": article_id, "pmc_id": pmc_id,
                             "method": "pmc_api", "status": "not_found",
                             "bytes": 0, "timestamp": NOW()})
                fail += 1
        else:
            # No PMC ID — write stub so we don't keep re-queuing
            stub = {"processed": True, "status": "no_pmc", "id": out_id,
                    "title": item["title"], "full_text": "", "word_count": 0}
            out_path.write_text(json.dumps(stub, ensure_ascii=False), encoding="utf-8")
            _append_log({"article_id": article_id, "pmc_id": "",
                         "method": "unpaywall", "status": "no_pmc_no_doi",
                         "bytes": 0, "timestamp": NOW()})
            skip += 1

    total_json = len(list(SCHOLAR_JSON.glob("*.json")))
    print(f"\n── Fetch complete ───────────────────────────────────")
    print(f"  Fetched OK : {ok}")
    print(f"  Skipped    : {skip}  (already present or no PMC ID)")
    print(f"  Failed     : {fail}  (404 / parse error)")
    print(f"  Total scholar_json/ : {total_json}")
    return {"queued": total_q, "fetched": ok, "failed": fail}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--priority-only", action="store_true",
                        help="Only fetch articles with composite_score >= 0.10")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max articles to fetch in this run")
    parser.add_argument("--api-key", default=None,
                        help="NCBI API key (enables 10 req/s vs 3 req/s)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show queue size without fetching")
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel workers (requires --api-key)")
    args = parser.parse_args()

    run(priority_only=args.priority_only,
        limit=args.limit,
        api_key=args.api_key,
        dry_run=args.dry_run,
        workers=args.workers)


if __name__ == "__main__":
    main()
