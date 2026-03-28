"""
_import_vcu_pdfs.py
─────────────────────────────────────────────────────────────────────────
Imports PDFs downloaded from VCU library into the full-text pipeline.

Expects PDFs in data/vcu_downloads/ named:
  {pmc_id}.pdf          e.g.  PMC12657415.pdf
  {article_id}.pdf      e.g.  1234.pdf         (fallback)

For each PDF found:
  1. Copies to data/scholar_pdfs/{id}.pdf  (if not already there)
  2. Extracts full text with pdfminer
  3. Saves unified JSON to data/scholar_json/{id}.json  (idempotent — skips existing)
  4. Logs to scripts/vcu_import_log.csv

Idempotent: already-processed PDFs are skipped on re-run.

Usage:
  python scripts/_import_vcu_pdfs.py
  python scripts/_import_vcu_pdfs.py --dry-run
  python scripts/_import_vcu_pdfs.py --overwrite   # re-extract even if JSON exists
"""
import argparse, csv, json, re, shutil
from datetime import datetime, timezone
from pathlib import Path

VCU_DL       = Path("data/vcu_downloads")
PDF_DIR      = Path("data/scholar_pdfs")
SCHOLAR_JSON = Path("data/scholar_json")
SCREENED     = Path("data/ontology/articles_screened.csv")
IMPORT_LOG   = Path("scripts/vcu_import_log.csv")
LOG_FIELDS   = ["pdf_name", "article_id", "pmc_id", "status", "word_count", "timestamp"]

NOW = lambda: datetime.now(timezone.utc).isoformat()

for d in (VCU_DL, PDF_DIR, SCHOLAR_JSON):
    d.mkdir(exist_ok=True)

# ── Build lookup: pmc_id / article_id → screened row ─────────────────────────
screened_by_pmc  = {}
screened_by_id   = {}
for row in csv.DictReader(open(SCREENED, encoding="utf-8-sig")):
    pmc = row.get("pmc_id", "").strip()
    aid = row.get("article_id", "").strip()
    if pmc:
        screened_by_pmc[pmc] = row
    screened_by_id[aid] = row


def _match_row(stem: str) -> tuple[str, str, dict]:
    """Return (article_id, pmc_id, screened_row) for a given PDF stem."""
    # Try as PMC ID directly
    if stem in screened_by_pmc:
        row = screened_by_pmc[stem]
        return row["article_id"], stem, row
    # Try as article_id (numeric)
    if stem in screened_by_id:
        row = screened_by_id[stem]
        return stem, row.get("pmc_id", ""), row
    # Try stripping numeric suffix (e.g. PMC12345678_1)
    base = re.sub(r"_\d+$", "", stem)
    if base in screened_by_pmc:
        row = screened_by_pmc[base]
        return row["article_id"], base, row
    return "", stem, {}


def _extract_pdf(pdf_path: Path) -> dict:
    """Extract full text from PDF, return structured dict."""
    from pdfminer.high_level import extract_pages
    from pdfminer.layout import LTTextContainer

    pages, full_parts = [], []
    try:
        for pg_num, layout in enumerate(extract_pages(str(pdf_path)), 1):
            pg_text = " ".join(el.get_text() for el in layout
                               if isinstance(el, LTTextContainer))
            pages.append({"page": pg_num, "text": pg_text})
            full_parts.append(pg_text)
    except Exception as e:
        pages = [{"page": 1, "text": f"[extraction error: {e}]"}]
        full_parts = [pages[0]["text"]]

    full_text = "\n\n".join(full_parts)

    # Heuristic abstract extraction
    abstract = ""
    for p in pages[:2]:
        m = re.search(
            r"\bAbstract\b[:\s]*(.{100,1500}?)(?=\n\s*\n|\bIntroduction\b|\bBackground\b)",
            p["text"], re.IGNORECASE | re.DOTALL,
        )
        if m:
            abstract = m.group(1).strip()
            break

    return {
        "pages": pages, "full_text": full_text,
        "abstract": abstract, "page_count": len(pages),
        "word_count": len(full_text.split()),
    }


def _build_json(out_id: str, pdf_path: Path, screened_row: dict, extracted: dict) -> dict:
    return {
        "id":          out_id,
        "source_type": "pdf",
        "title":       screened_row.get("title", ""),
        "doi":         "",
        "pmc_id":      screened_row.get("pmc_id", "").strip(),
        "pmid":        "",
        "year":        (screened_row.get("pubdate", "") or "")[:4],
        "authors":     [],
        "journal":     "",
        "abstract":    extracted["abstract"],
        "full_text":   extracted["full_text"],
        "word_count":  extracted["word_count"],
        "sections":    [],
        "keywords":    [],
        "metadata": {
            "extracted_at": NOW(),
            "source_file":  str(pdf_path),
            "page_count":   extracted["page_count"],
            "import_method": "vcu_library",
        },
    }


def _append_log(row: dict):
    exists = IMPORT_LOG.exists()
    with open(IMPORT_LOG, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        if not exists:
            w.writeheader()
        w.writerow(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run",   action="store_true")
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-extract even if JSON already exists")
    args = parser.parse_args()

    pdfs = sorted(VCU_DL.glob("*.pdf"))
    json_index = {p.stem for p in SCHOLAR_JSON.glob("*.json")}

    print(f"VCU downloads folder: {VCU_DL}  ({len(pdfs)} PDFs)")
    print(f"scholar_json/ index : {len(json_index)} existing")
    print()

    if not pdfs:
        print("No PDFs found in data/vcu_downloads/  —  nothing to import.")
        print(f"Download PDFs and save them there, then re-run.")
        return

    ok = skip = unmatched = 0
    for pdf in pdfs:
        stem = pdf.stem
        article_id, pmc_id, screened_row = _match_row(stem)
        out_id = pmc_id or (f"article_{article_id}" if article_id else stem)
        out_json = SCHOLAR_JSON / f"{out_id}.json"
        out_pdf  = PDF_DIR / f"{out_id}.pdf"

        if out_json.exists() and not args.overwrite:
            print(f"  skip  {stem}  (JSON exists)")
            skip += 1
            continue

        if not screened_row:
            print(f"  ⚠️   {stem}  — not matched to any article in articles_screened.csv")
            unmatched += 1
            _append_log({"pdf_name": pdf.name, "article_id": "", "pmc_id": pmc_id,
                         "status": "unmatched", "word_count": 0, "timestamp": NOW()})
            continue

        title = (screened_row.get("title") or "")[:60]
        print(f"  proc  {stem}  → {out_id}  {title}")

        if args.dry_run:
            continue

        # Copy PDF to scholar_pdfs/
        if not out_pdf.exists():
            shutil.copy2(pdf, out_pdf)

        # Extract and save JSON
        extracted = _extract_pdf(pdf)
        data = _build_json(out_id, out_pdf, screened_row, extracted)
        out_json.write_text(json.dumps(data, ensure_ascii=False, indent=2),
                            encoding="utf-8")
        _append_log({
            "pdf_name":   pdf.name,
            "article_id": article_id,
            "pmc_id":     pmc_id,
            "status":     "ok",
            "word_count": extracted["word_count"],
            "timestamp":  NOW(),
        })
        ok += 1

    total_json = len(list(SCHOLAR_JSON.glob("*.json")))
    print()
    if args.dry_run:
        print(f"[dry-run] Would process {len(pdfs) - skip} PDFs.")
    else:
        print(f"── Import complete ───────────────────────────────")
        print(f"  Imported   : {ok}")
        print(f"  Skipped    : {skip}  (JSON already exists)")
        print(f"  Unmatched  : {unmatched}  (no match in articles_screened.csv)")
        print(f"  scholar_json/ total: {total_json}")

    if ok > 0 and not args.dry_run:
        print(f"\nNext: re-run Phase 7 + rebuild checklist:")
        print(f"  python scripts/_run_fulltext_pipeline.py --step 4")


if __name__ == "__main__":
    main()
