"""
unpaywall_lookup.py
For articles without PMC IDs (HSH stubs), finds open-access PDFs via:
  1. CrossRef API: title → DOI
  2. Unpaywall API: DOI → OA PDF URL
  3. Download PDF → AWS Textract → BioC-style JSON

Outputs JSON files to {topic_dir}/pubmed_json_files/{hsh_id}.json
Progress saved to scripts/unpaywall_log.csv (resume-safe).

Run from: manuscript/CH_1/Literature_Review/
  python scripts/unpaywall_lookup.py --email your@email.com [--use-textract]
"""

import argparse
import csv
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import requests

MISSING_CSV  = Path("scripts/missing_articles_combined.csv")
LOG_CSV      = Path("scripts/unpaywall_log.csv")
CROSSREF_URL = "https://api.crossref.org/works"
UNPAYWALL_URL = "https://api.unpaywall.org/v2/{doi}"
SLEEP_S      = 0.5   # CrossRef asks for 1 req/sec politely
MAX_RETRIES  = 2


def crossref_lookup(title: str, email: str) -> str | None:
    """Return DOI for best title match via CrossRef, or None."""
    try:
        r = requests.get(
            CROSSREF_URL,
            params={"query.title": title[:250], "rows": 1,
                    "select": "DOI,title,score", "mailto": email},
            timeout=15
        )
        if r.status_code != 200:
            return None
        items = r.json().get("message", {}).get("items", [])
        if not items:
            return None
        best = items[0]
        # Reject if CrossRef score is low (weak match)
        if best.get("score", 0) < 80:
            return None
        return best.get("DOI")
    except Exception:
        return None


def unpaywall_lookup(doi: str, email: str) -> str | None:
    """Return best OA PDF URL from Unpaywall, or None."""
    try:
        url = UNPAYWALL_URL.format(doi=doi)
        r = requests.get(url, params={"email": email}, timeout=15)
        if r.status_code != 200:
            return None
        data = r.json()
        # Prefer best_oa_location with PDF
        best = data.get("best_oa_location")
        if best and best.get("url_for_pdf"):
            return best["url_for_pdf"]
        # Fall back to any oa_location with PDF
        for loc in data.get("oa_locations", []):
            if loc.get("url_for_pdf"):
                return loc["url_for_pdf"]
        return None
    except Exception:
        return None


def download_pdf(url: str) -> bytes | None:
    """Download PDF bytes from URL."""
    for attempt in range(MAX_RETRIES):
        try:
            r = requests.get(url, timeout=30, allow_redirects=True,
                             headers={"User-Agent": "PGxLitReview/1.0 (research)"})
            if r.status_code == 200 and b"%PDF" in r.content[:16]:
                return r.content
        except Exception:
            time.sleep(1)
    return None


def textract_pdf_to_json(pdf_bytes: bytes, hsh_id: str, title: str) -> dict:
    """Use AWS Textract to extract text from PDF bytes, return BioC-style dict."""
    import boto3
    client = boto3.client("textract")
    resp = client.detect_document_text(Document={"Bytes": pdf_bytes})
    text_blocks = [b["Text"] for b in resp.get("Blocks", [])
                   if b["BlockType"] == "LINE"]
    full_text = " ".join(text_blocks)
    # BioC-compatible structure (matches PMC BioC JSON schema)
    return {
        "source": "unpaywall+textract",
        "date": "",
        "key": hsh_id,
        "infons": {},
        "documents": [{
            "id": hsh_id,
            "infons": {"title": title},
            "passages": [{"offset": 0, "text": full_text, "infons": {}, "annotations": [], "relations": []}],
            "annotations": [],
            "relations": []
        }]
    }


def pdfminer_pdf_to_json(pdf_bytes: bytes, hsh_id: str, title: str) -> dict | None:
    """Fallback: use pdfminer.six if Textract unavailable."""
    try:
        import io
        from pdfminer.high_level import extract_text as pm_extract
        text = pm_extract(io.BytesIO(pdf_bytes))
        return {
            "source": "unpaywall+pdfminer",
            "key": hsh_id,
            "documents": [{
                "id": hsh_id,
                "infons": {"title": title},
                "passages": [{"offset": 0, "text": text[:50000], "infons": {},
                               "annotations": [], "relations": []}],
                "annotations": [], "relations": []
            }]
        }
    except Exception:
        return None


def load_existing_log() -> set[str]:
    """Return set of HSH IDs already processed."""
    if not LOG_CSV.exists():
        return set()
    with open(LOG_CSV, newline="", encoding="utf-8") as f:
        return {row["hsh_id"] for row in csv.DictReader(f)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--email", default="dixonrj@vcu.edu",
                        help="Email for CrossRef/Unpaywall polite pool")
    parser.add_argument("--use-textract", action="store_true",
                        help="Use AWS Textract for PDF text extraction (costs $0.0015/page)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process at most N articles (for testing)")
    args = parser.parse_args()

    if not MISSING_CSV.exists():
        sys.exit(f"ERROR: {MISSING_CSV} not found. Run prisma_tracker.R first.")

    with open(MISSING_CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    already_done = load_existing_log()
    queue = [r for r in rows if r["pmc_id"] not in already_done]
    if args.limit:
        queue = queue[:args.limit]

    print(f"HSH-stub articles (no PMC ID): {len(rows):,}")
    print(f"Already processed:             {len(already_done):,}")
    print(f"To process:                    {len(queue):,}")
    print(f"Email (polite pool):           {args.email}")
    print(f"PDF extraction:                {'AWS Textract' if args.use_textract else 'pdfminer (local)'}")
    print()

    LOG_CSV.parent.mkdir(parents=True, exist_ok=True)
    log_exists = LOG_CSV.exists()
    log_fh = open(LOG_CSV, "a", newline="", encoding="utf-8")
    log_writer = csv.DictWriter(log_fh, fieldnames=[
        "hsh_id", "title_short", "doi", "pdf_url", "status", "bytes"])
    if not log_exists:
        log_writer.writeheader()

    n_found = n_not_found = n_error = 0

    for i, row in enumerate(queue, 1):
        hsh_id      = row["pmc_id"]
        title       = row["title"]
        source_file = row["source_file"]
        out_dir     = Path(source_file).parent / "pubmed_json_files"
        out_path    = out_dir / f"{hsh_id}.json"

        log_row = {"hsh_id": hsh_id, "title_short": title[:60],
                   "doi": "", "pdf_url": "", "status": "", "bytes": 0}

        if i % 25 == 0 or i <= 3:
            print(f"  [{i:>5}/{len(queue)}] {title[:70]}")

        # Skip if output already exists
        if out_path.exists() and out_path.stat().st_size > 100:
            log_row["status"] = "already_exists"
            log_writer.writerow(log_row)
            continue

        time.sleep(SLEEP_S)

        # Step 1: CrossRef → DOI
        doi = crossref_lookup(title, args.email)
        log_row["doi"] = doi or ""
        if not doi:
            log_row["status"] = "no_doi"
            n_not_found += 1
            log_writer.writerow(log_row)
            continue

        time.sleep(SLEEP_S)

        # Step 2: Unpaywall → PDF URL
        pdf_url = unpaywall_lookup(doi, args.email)
        log_row["pdf_url"] = pdf_url or ""
        if not pdf_url:
            log_row["status"] = "no_oa_pdf"
            n_not_found += 1
            log_writer.writerow(log_row)
            continue

        # Step 3: Download PDF
        pdf_bytes = download_pdf(pdf_url)
        if not pdf_bytes:
            log_row["status"] = "pdf_download_failed"
            n_error += 1
            log_writer.writerow(log_row)
            continue

        # Step 4: Extract text → BioC-style JSON
        result_json = None
        if args.use_textract:
            try:
                result_json = textract_pdf_to_json(pdf_bytes, hsh_id, title)
            except Exception as e:
                print(f"    Textract failed: {e}; falling back to pdfminer")

        if result_json is None:
            result_json = pdfminer_pdf_to_json(pdf_bytes, hsh_id, title)

        if result_json is None:
            log_row["status"] = "text_extract_failed"
            n_error += 1
            log_writer.writerow(log_row)
            continue

        # Step 5: Save JSON
        out_dir.mkdir(parents=True, exist_ok=True)
        json_bytes = json.dumps(result_json, ensure_ascii=False).encode("utf-8")
        out_path.write_bytes(json_bytes)

        log_row["status"] = "ok"
        log_row["bytes"]  = len(json_bytes)
        n_found += 1
        log_writer.writerow(log_row)

        if i % 25 == 0:
            print(f"    → found so far: {n_found}, not found: {n_not_found}, errors: {n_error}")

    log_fh.close()

    total = n_found + n_not_found + n_error
    print(f"\n── Unpaywall lookup complete ────────────────────────────")
    print(f"  OA PDFs retrieved:  {n_found:,}")
    print(f"  Not found (no OA):  {n_not_found:,}")
    print(f"  Errors:             {n_error:,}")
    print(f"  Log:                {LOG_CSV}")
    print(f"\nNOTE: Articles with status='no_oa_pdf' require manual download.")
    print(f"      See scripts/unpaywall_log.csv for DOIs — use institutional")
    print(f"      access at https://library.vcu.edu to retrieve remaining PDFs.")
    print(f"\nRe-run prisma_tracker.R to refresh PRISMA full-text counts.")


if __name__ == "__main__":
    main()
