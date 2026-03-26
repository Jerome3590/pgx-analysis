"""
scholar_lookup.py
Scans paywalled / no-DOI articles from unpaywall_log.csv for freely
available full-text using three sources (in priority order):

  1. Semantic Scholar API  — DOI or title search, openAccessPdf field
  2. Google Scholar        — via 'scholarly' library (title search)
  3. ResearchGate          — requests + BeautifulSoup title search

Downloads found PDFs to data/scholar_pdfs/{hsh_id}.pdf and extracts
text with pdfminer.six, saving BioC-style JSON to each topic's
pubmed_json_files/ directory (or data/scholar_json/ as fallback).

Progress is appended to scripts/scholar_log.csv.

Usage:
  python scripts/scholar_lookup.py                          # all candidates from unpaywall_log
  python scripts/scholar_lookup.py --screened-only          # only 511 screened-include HSH articles
  python scripts/scholar_lookup.py --limit 20 --source epmc # test Europe PMC on first 20
  python scripts/scholar_lookup.py --source gs              # google scholar only
  python scripts/scholar_lookup.py --source rg              # researchgate only
  python scripts/scholar_lookup.py --retry-failed           # also re-try download_failed
"""

import argparse
import csv
import json
import os
import re
import sys
import time
import urllib.parse
from datetime import datetime
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter, Retry

# ── Optional imports (graceful degrade) ──────────────────────────────────────
try:
    from scholarly import scholarly as _scholarly
    SCHOLARLY_OK = True
except ImportError:
    SCHOLARLY_OK = False
    print("WARNING: 'scholarly' not installed — Google Scholar source disabled.")
    print("         pip install scholarly")

try:
    from bs4 import BeautifulSoup
    BS4_OK = True
except ImportError:
    BS4_OK = False
    print("WARNING: 'beautifulsoup4' not installed — ResearchGate source disabled.")

try:
    from pdfminer.high_level import extract_text as pdf_extract_text
    PDFMINER_OK = True
except ImportError:
    PDFMINER_OK = False

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT          = Path(__file__).resolve().parent.parent
UNPAYWALL_LOG = ROOT / "scripts" / "unpaywall_log.csv"
TAGGED_CSV    = ROOT / "data" / "ontology" / "articles_tagged.csv"
SCHOLAR_LOG          = ROOT / "scripts" / "scholar_log.csv"
SCREENED_MISSING_CSV = ROOT / "scripts" / "screened_missing_fulltext.csv"
PDF_DIR              = ROOT / "data" / "scholar_pdfs"
JSON_FALLBACK        = ROOT / "data" / "scholar_json"
PDF_DIR.mkdir(parents=True, exist_ok=True)
JSON_FALLBACK.mkdir(parents=True, exist_ok=True)

# ── HTTP session ──────────────────────────────────────────────────────────────
SESSION = requests.Session()
SESSION.mount("https://", HTTPAdapter(max_retries=Retry(total=3, backoff_factor=1)))
SESSION.headers.update({
    "User-Agent": (
        "Mozilla/5.0 (compatible; LitReviewBot/1.0; "
        "mailto:dixonrj@vcu.edu)"
    )
})

# ── CrossRef: resolve DOI from title ─────────────────────────────────────────
CR_BASE = "https://api.crossref.org/works"

def crossref_doi(title: str) -> str | None:
    """Try to resolve a DOI for a title-only article via CrossRef."""
    try:
        params = {"query.title": title, "rows": 1,
                  "mailto": "dixonrj@vcu.edu"}
        r = SESSION.get(CR_BASE, params=params, timeout=10)
        if r.status_code == 200:
            items = r.json().get("message", {}).get("items", [])
            if items:
                return items[0].get("DOI", "")
    except Exception:
        pass
    return None


# ── Europe PMC ────────────────────────────────────────────────────────────────
EPMC_BASE = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"

def search_europepmc(doi: str, title: str) -> str | None:
    """Search Europe PMC for open-access full-text PDF."""
    try:
        query = f"DOI:{doi}" if doi else f'TITLE:"{title[:100]}"'
        params = {"query": query, "format": "json",
                  "resultType": "core", "pageSize": 1}
        r = SESSION.get(EPMC_BASE, params=params, timeout=10)
        if r.status_code == 200:
            results = r.json().get("resultList", {}).get("result", [])
            for res in results:
                # fullTextUrlList contains OA PDF links
                urls = res.get("fullTextUrlList", {}).get("fullTextUrl", [])
                for u in urls:
                    if u.get("documentStyle") == "pdf" and u.get("availability") == "Open access":
                        return u["url"]
                # pmcid → direct PMC OA PDF
                pmcid = res.get("pmcid", "")
                if pmcid:
                    return f"https://europepmc.org/articles/{pmcid}/pdf"
    except Exception:
        pass
    return None


# ── CORE.ac.uk ────────────────────────────────────────────────────────────────
CORE_BASE = "https://api.core.ac.uk/v3/search/works"

def search_core(doi: str, title: str,
                api_key: str = os.getenv("CORE_API_KEY", "")) -> str | None:
    """Search CORE for open-access PDF. Free without key (10 req/min)."""
    try:
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        query   = doi if doi else title[:120]
        params  = {"q": query, "limit": 1}
        r = SESSION.get(CORE_BASE, params=params, headers=headers, timeout=10)
        if r.status_code == 200:
            results = r.json().get("results", [])
            for res in results:
                pdf = res.get("downloadUrl") or res.get("sourceFulltextUrls", [None])[0]
                if pdf:
                    return pdf
    except Exception:
        pass
    return None


# ── Semantic Scholar (OA check only — not primary discovery) ──────────────────
SS_BASE   = "https://api.semanticscholar.org/graph/v1/paper"
SS_FIELDS = "title,openAccessPdf,externalIds,year"

def search_semantic_scholar(doi: str, title: str) -> str | None:
    """Check Semantic Scholar openAccessPdf field (complements Unpaywall)."""
    try:
        if doi:
            r = SESSION.get(f"{SS_BASE}/DOI:{doi}",
                            params={"fields": SS_FIELDS}, timeout=10)
        else:
            r = SESSION.get(f"{SS_BASE}/search",
                            params={"query": title, "fields": SS_FIELDS, "limit": 1},
                            timeout=10)
            data = r.json().get("data", []) if r.status_code == 200 else []
            if not data:
                return None
            r = type("R", (), {"status_code": 200,
                               "json": lambda: data[0]})()  # type: ignore
        if r.status_code == 200:
            oa = r.json().get("openAccessPdf", {})
            url = (oa or {}).get("url", "")
            if url:
                return url
    except Exception:
        pass
    return None

# ── Google Scholar via scholarly ──────────────────────────────────────────────
_GS_DELAY = 5.0   # seconds between requests (conservative to avoid blocks)

def search_google_scholar(title: str) -> str | None:
    if not SCHOLARLY_OK:
        return None
    try:
        pub = _scholarly.search_single_pub(title)
        url = pub.get("eprint_url") or pub.get("pub_url")
        if url and url.endswith(".pdf"):
            return url
        # Check if eprint is a PDF link
        eprint = pub.get("eprint_url", "")
        if eprint and ("pdf" in eprint.lower() or "arxiv" in eprint.lower()):
            return eprint
    except Exception:
        pass
    finally:
        time.sleep(_GS_DELAY)
    return None

# ── ResearchGate ──────────────────────────────────────────────────────────────
RG_SEARCH = "https://www.researchgate.net/search/publication"
_RG_DELAY = 3.0

def search_researchgate(title: str) -> str | None:
    if not BS4_OK:
        return None
    try:
        query = urllib.parse.quote_plus(title[:120])
        r = SESSION.get(f"{RG_SEARCH}?q={query}", timeout=15)
        if r.status_code != 200:
            return None
        soup = BeautifulSoup(r.text, "html.parser")

        # Find first result with a full-text link
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if "/publication/" in href and "fulltext" in href.lower():
                full_url = "https://www.researchgate.net" + href if href.startswith("/") else href
                return full_url
            if href.endswith(".pdf") and "researchgate" in href:
                return href

    except Exception:
        pass
    finally:
        time.sleep(_RG_DELAY)
    return None

# ── PDF download + text extraction ───────────────────────────────────────────

def download_pdf(url: str, dest: Path) -> int:
    try:
        r = SESSION.get(url, timeout=30, stream=True)
        if r.status_code == 200 and "pdf" in r.headers.get("content-type", "").lower():
            with open(dest, "wb") as f:
                for chunk in r.iter_content(65536):
                    f.write(chunk)
            return dest.stat().st_size
    except Exception:
        pass
    return 0


def extract_text_from_pdf(pdf_path: Path) -> str:
    if not PDFMINER_OK:
        return ""
    try:
        return pdf_extract_text(str(pdf_path))
    except Exception:
        return ""


def save_bioc_json(hsh_id: str, title: str, text: str,
                   source_label: str, dest_dir: Path) -> Path:
    doc = {
        "source": source_label,
        "date": datetime.now().strftime("%Y-%m-%d"),
        "key": hsh_id,
        "documents": [{
            "id": hsh_id,
            "passages": [{"infons": {"section": "body"}, "text": text}]
        }],
        "title": title,
    }
    out = dest_dir / f"{hsh_id}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=2)
    return out

# ── Load data ─────────────────────────────────────────────────────────────────

MISSING_CSV = ROOT / "scripts" / "missing_articles_combined.csv"


def load_screened_missing() -> list[dict]:
    """Load the 511 screened-include articles that lack full-text as candidates."""
    if not SCREENED_MISSING_CSV.exists():
        print("screened_missing_fulltext.csv not found — run scripts/_check_fulltext.py first.")
        return []
    if not MISSING_CSV.exists():
        return []

    # Build hsh_id -> full_title from missing_articles_combined.csv
    titles: dict[str, str] = {}
    with open(MISSING_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            hsh = row.get("pmc_id", "").strip()
            t   = row.get("title", "").strip()
            if hsh and t:
                titles[hsh] = t

    # Build hsh_id -> doi from unpaywall_log
    dois: dict[str, str] = {}
    if UNPAYWALL_LOG.exists():
        with open(UNPAYWALL_LOG, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                dois[row["hsh_id"]] = row.get("doi", "")

    # Already done
    done: set[str] = set()
    if SCHOLAR_LOG.exists():
        with open(SCHOLAR_LOG, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("status") not in ("error", "not_found", "download_failed"):
                    done.add(row["hsh_id"])

    candidates: list[dict] = []
    with open(SCREENED_MISSING_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            hsh = row.get("pmc_id", "").strip()
            if not hsh or hsh in done:
                continue
            candidates.append({
                "hsh_id":     hsh,
                "title_short": row.get("title", "")[:60],
                "full_title":  titles.get(hsh, row.get("title", "")),
                "doi":         dois.get(hsh, ""),
                "status":      "screened_missing",
            })
    return candidates


def load_candidates(retry_failed: bool = False) -> list[dict]:
    if not UNPAYWALL_LOG.exists():
        sys.exit(f"ERROR: {UNPAYWALL_LOG} not found. Run unpaywall_lookup.py first.")

    # Build hsh_id -> full_title from missing_articles_combined.csv
    # (pmc_id column holds HSH stubs there)
    titles: dict[str, str] = {}
    if MISSING_CSV.exists():
        with open(MISSING_CSV, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                hsh = row.get("pmc_id", "").strip()
                t   = row.get("title", "").strip()
                if hsh and t:
                    titles[hsh] = t

    # Fallback: title-prefix match from articles_tagged.csv
    title_prefix: dict[str, str] = {}
    if TAGGED_CSV.exists():
        with open(TAGGED_CSV, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                t = row.get("title", "").strip()
                if t:
                    title_prefix[t[:40].lower()] = t

    # Already successfully processed in scholar_log
    done: set[str] = set()
    if SCHOLAR_LOG.exists():
        with open(SCHOLAR_LOG, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("status") not in ("error", "not_found", "download_failed"):
                    done.add(row["hsh_id"])

    target_statuses = {"no_oa_pdf", "no_doi"}
    if retry_failed:
        target_statuses.add("pdf_download_failed")

    candidates: list[dict] = []
    with open(UNPAYWALL_LOG, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["status"] not in target_statuses:
                continue
            if row["hsh_id"] in done:
                continue
            # Resolve full title
            full_title = titles.get(row["hsh_id"], "")
            if not full_title:
                # Try prefix match
                ts = row.get("title_short", "")[:40].lower()
                full_title = title_prefix.get(ts, row.get("title_short", ""))
            row["full_title"] = full_title
            candidates.append(row)

    # Prioritise DOI-bearing articles (no_oa_pdf) before no_doi
    candidates.sort(key=lambda r: (0 if r["status"] == "no_oa_pdf" else 1))
    return candidates


def append_log(row: dict):
    fieldnames = ["hsh_id", "title_short", "doi", "source",
                  "pdf_url", "status", "bytes", "json_path", "timestamp"]
    write_header = not SCHOLAR_LOG.exists()
    with open(SCHOLAR_LOG, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            w.writeheader()
        w.writerow(row)

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit",        type=int, default=0,
                        help="Process only first N candidates (0 = all)")
    parser.add_argument("--source",
                        choices=["epmc", "core", "ss", "gs", "rg", "all"],
                        default="all", help="Which source(s) to query")
    parser.add_argument("--screened-only", action="store_true",
                        help="Only process the 511 screened-include HSH articles missing full-text")
    parser.add_argument("--retry-failed", action="store_true",
                        help="Also retry pdf_download_failed articles")
    parser.add_argument("--ss-delay",     type=float, default=0.5,
                        help="Seconds between Semantic Scholar requests")
    args = parser.parse_args()

    if args.screened_only:
        candidates = load_screened_missing()
        suffix = "(screened-include missing full-text)"
    else:
        candidates = load_candidates(retry_failed=args.retry_failed)
        suffix = "(no_doi + no_oa_pdf" + (" + pdf_download_failed" if args.retry_failed else "") + ")"
    if args.limit:
        candidates = candidates[:args.limit]

    total = len(candidates)
    print(f"Candidates: {total:,}  {suffix}")
    print(f"Sources:    {args.source.upper()}")
    print(f"Log:        {SCHOLAR_LOG}\n")

    found = 0
    for i, cand in enumerate(candidates, 1):
        hsh     = cand["hsh_id"]
        doi     = cand.get("doi", "").strip()
        title   = cand["full_title"] or cand.get("title_short", "")
        title_s = title[:60]

        print(f"[{i:5}/{total}] {title_s:<60}", end="  ", flush=True)

        pdf_url   = None
        source_lbl = ""

        # ── 0. Resolve DOI for no_doi articles via CrossRef ──────────────
        if not doi and cand["status"] == "no_doi":
            doi = crossref_doi(title) or ""
            if doi:
                cand["doi"] = doi

        # ── 1. Europe PMC ─────────────────────────────────────────────────
        if args.source in ("epmc", "all"):
            pdf_url = search_europepmc(doi, title)
            if pdf_url:
                source_lbl = "europe_pmc"
            time.sleep(0.3)

        # ── 2. CORE.ac.uk ─────────────────────────────────────────────────
        if pdf_url is None and args.source in ("core", "all"):
            pdf_url = search_core(doi, title)
            if pdf_url:
                source_lbl = "core"
            time.sleep(6.5)  # free tier: 10 req/min

        # ── 3. Semantic Scholar ───────────────────────────────────────────
        if pdf_url is None and args.source in ("ss", "all"):
            pdf_url = search_semantic_scholar(doi, title)
            if pdf_url:
                source_lbl = "semantic_scholar"
            time.sleep(args.ss_delay)

        # ── 4. Google Scholar ─────────────────────────────────────────────
        if pdf_url is None and args.source in ("gs", "all"):
            pdf_url = search_google_scholar(title)
            if pdf_url:
                source_lbl = "google_scholar"

        # ── 5. ResearchGate ───────────────────────────────────────────────
        if pdf_url is None and args.source in ("rg", "all"):
            pdf_url = search_researchgate(title)
            if pdf_url:
                source_lbl = "researchgate"

        if not pdf_url:
            print("not found")
            append_log({"hsh_id": hsh, "title_short": title_s, "doi": doi,
                        "source": "", "pdf_url": "", "status": "not_found",
                        "bytes": 0, "json_path": "", "timestamp": datetime.now().isoformat()})
            continue

        # ── Download PDF ──────────────────────────────────────────────────
        pdf_dest = PDF_DIR / f"{hsh}.pdf"
        nbytes   = download_pdf(pdf_url, pdf_dest)
        if nbytes < 1024:
            print(f"download failed ({source_lbl})")
            append_log({"hsh_id": hsh, "title_short": title_s, "doi": doi,
                        "source": source_lbl, "pdf_url": pdf_url,
                        "status": "download_failed", "bytes": nbytes,
                        "json_path": "", "timestamp": datetime.now().isoformat()})
            continue

        # ── Extract text + save JSON ──────────────────────────────────────
        text      = extract_text_from_pdf(pdf_dest)
        json_path = save_bioc_json(hsh, title, text, source_lbl, JSON_FALLBACK)

        found += 1
        print(f"OK  {nbytes/1024:>7.1f} KB  [{source_lbl}]")
        append_log({"hsh_id": hsh, "title_short": title_s, "doi": doi,
                    "source": source_lbl, "pdf_url": pdf_url,
                    "status": "ok", "bytes": nbytes,
                    "json_path": str(json_path),
                    "timestamp": datetime.now().isoformat()})

    print(f"\n── Scholar scan complete ─────────────────────────────")
    print(f"  Processed:  {total:,}")
    print(f"  Found PDFs: {found:,}  ({found/max(total,1)*100:.1f}%)")
    print(f"  Log:        {SCHOLAR_LOG}")
    print(f"  PDFs:       {PDF_DIR}/")
    print(f"  JSONs:      {JSON_FALLBACK}/")
    print(f"\nRe-run prisma_tracker.R and screen_articles.py after this completes.")


if __name__ == "__main__":
    main()
