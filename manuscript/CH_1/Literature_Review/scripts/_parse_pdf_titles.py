"""
_parse_pdf_titles.py
Extracts title from each PDF in manual_review/ AND Zotero storage by reading:
  1. PDF metadata /Title field
  2. First ~600 chars of page-1 text (title is usually at the top)
Then fuzzy-matches against remaining missing articles and imports matches.

Run from: manuscript/CH_1/Literature_Review/
  python scripts/_parse_pdf_titles.py [--dry-run] [--threshold N]
"""
import argparse
import csv
import io
import re
import shutil
from datetime import datetime
from pathlib import Path

import spacy
import pytextrank
from pdfminer.high_level import extract_text_to_fp
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage

# Load spaCy with pytextrank (en_core_web_sm sufficient for phrase extraction)
try:
    _nlp = spacy.load("en_core_web_sm")
except OSError:
    _nlp = spacy.load("en_core_web_lg")
_nlp.add_pipe("textrank")

MANUAL_DIR     = Path(r"C:\Projects\pgx-analysis\manuscript\infrastructure_setup\manual_review")
ZOTERO_STORAGE = Path(r"C:\Users\jerom\Zotero\storage")
DOI_MAP        = Path("scripts/screened_doi_map.csv")
LOG_CSV        = Path("scripts/vcu_download_log.csv")
PDF_DIR        = Path("data/scholar_pdfs")

LOG_FIELDS = ["hsh_id", "title", "doi", "proxy_url", "status", "bytes", "timestamp"]

STOP = {"a","an","the","of","in","on","and","for","to","with","from","at","by",
        "is","are","its","as","or","that","this","via","using","based","among",
        "between","after","before","during","within","through","were","was",
        "have","has","been","their","its","which","who"}

# ── PDF title extraction ───────────────────────────────────────────────────────

def extract_pdf_title(pdf_path: Path) -> str:
    """Try metadata /Title first, then first-page text."""
    try:
        from pdfminer.pdfparser import PDFParser
        from pdfminer.pdfdocument import PDFDocument

        with open(pdf_path, "rb") as f:
            parser = PDFParser(f)
            doc    = PDFDocument(parser)
            if doc.info:
                for info in doc.info:
                    title = info.get("Title", b"")
                    if isinstance(title, bytes):
                        title = title.decode("utf-8", errors="ignore").strip()
                    if title and len(title) > 10:
                        return title
    except Exception:
        pass

    # Fallback: first 600 chars of page 1 text
    try:
        buf = io.StringIO()
        with open(pdf_path, "rb") as f:
            pages = list(PDFPage.get_pages(f, maxpages=1))
            if pages:
                extract_text_to_fp(open(pdf_path, "rb"), buf,
                                   page_numbers=[0],
                                   laparams=LAParams(line_margin=0.5))
        text = buf.getvalue().strip()
        # Return first non-trivial line block (skip very short lines like volume/date)
        lines = [ln.strip() for ln in text.splitlines() if len(ln.strip()) > 15]
        if lines:
            # Join first 3 lines — often title spans multiple lines
            return " ".join(lines[:3])[:300]
    except Exception:
        pass

    return ""

# ── Helpers ────────────────────────────────────────────────────────────────────

def tok(s: str) -> set:
    return set(re.sub(r"[^a-z0-9]", " ", s.lower()).split()) - STOP

def fuzzy_score(a: str, b: str) -> int:
    """Token overlap score (fast baseline)."""
    return len(tok(a) & tok(b))

def textrank_score(pdf_text: str, article_title: str) -> float:
    """
    Extract key phrases from PDF text via pytextrank, then measure overlap
    with article title tokens.  Returns weighted phrase-overlap score.
    """
    try:
        doc = _nlp(pdf_text[:2000])  # limit to first 2000 chars for speed
        title_toks = tok(article_title)
        score = 0.0
        for phrase in doc._.phrases[:20]:  # top-20 ranked phrases
            phrase_toks = tok(phrase.text)
            overlap = len(phrase_toks & title_toks)
            if overlap:
                score += overlap * phrase.rank  # weight by TextRank score
        return score
    except Exception:
        return 0.0

def load_missing() -> dict:
    doi_rows = {r["screened_pmc_id"]: r
                for r in csv.DictReader(open(DOI_MAP, encoding="utf-8-sig"))}
    latest = {}
    for r in csv.DictReader(open(LOG_CSV, encoding="utf-8-sig")):
        hid = r["hsh_id"]
        if hid not in latest or r["timestamp"] > latest[hid]["timestamp"]:
            latest[hid] = r
    have = ({p.stem for p in PDF_DIR.glob("*.pdf")} |
            {hid for hid, r in latest.items()
             if r["status"] in ("ok","manual","zotero","oa_url","reconcile")})
    return {hid: r
            for hid, r in doi_rows.items()
            if hid not in have and r.get("doi")}

def append_log(row: dict):
    exists = LOG_CSV.exists()
    with open(LOG_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        if not exists:
            w.writeheader()
        w.writerow(row)

# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--threshold", type=float, default=0.1)
    parser.add_argument("--dir", type=Path, default=None,
                        help="Override manual review directory")
    parser.add_argument("--zotero", action="store_true",
                        help="Also scan Zotero storage (slow — 1400+ PDFs)")
    args = parser.parse_args()

    missing = load_missing()
    print(f"Missing articles : {len(missing)}")

    # Collect PDFs to scan
    pdfs_to_scan: list[tuple[Path, str]] = []  # (path, source_label)

    manual_dir  = args.dir if args.dir else MANUAL_DIR
    manual_pdfs = list(manual_dir.glob("*.pdf"))
    pdfs_to_scan += [(p, "manual") for p in manual_pdfs]
    print(f"manual_review/   : {len(manual_pdfs)} PDFs")

    if args.zotero:
        z_pdfs = list(ZOTERO_STORAGE.rglob("*.pdf"))
        pdfs_to_scan += [(p, f"zotero:{p.parent.name}") for p in z_pdfs]
        print(f"Zotero storage/  : {len(z_pdfs)} PDFs")

    print(f"\nScanning {len(pdfs_to_scan)} PDFs for titles...\n")

    imported = 0
    already_matched: set[str] = set()  # hids already matched this run

    for pdf_path, source in pdfs_to_scan:
        extracted = extract_pdf_title(pdf_path)

        # Get first-page text for textrank scoring
        try:
            buf = io.StringIO()
            extract_text_to_fp(open(pdf_path, "rb"), buf,
                               page_numbers=[0], laparams=LAParams())
            page_text = buf.getvalue()[:2000]
        except Exception:
            page_text = extracted

        if not page_text.strip():
            print(f"  ✗ NO TEXT  : {pdf_path.name[:60]}")
            continue

        # Score: TextRank on extracted title; fall back to token overlap on page text
        score_text = extracted if extracted and len(extracted) > 15 else page_text[:300]
        best_hid, best_score = None, 0.0
        for hid, row in missing.items():
            if hid in already_matched:
                continue
            tr = textrank_score(score_text, row["title"])
            fb = fuzzy_score(score_text, row["title"]) * 0.05  # scale to same range
            score = max(tr, fb)
            if score > best_score:
                best_score, best_hid = score, hid

        display_score = f"{best_score:.3f}"
        if best_score < args.threshold or not best_hid:
            if best_score > 0:
                print(f"  ~ LOW [{display_score}]: {(extracted or page_text[:60])[:60]}")
            continue

        want_title = missing[best_hid]["title"]
        print(f"  ✓ MATCH [{best_score:.3f}]: {(extracted or page_text[:65])[:65]}")
        print(f"           → {want_title[:65]}")
        print(f"             hid={best_hid}  src={source}")

        dest = PDF_DIR / f"{best_hid}.pdf"
        if not args.dry_run:
            shutil.copy2(pdf_path, dest)
            append_log({
                "hsh_id":    best_hid,
                "title":     want_title[:70],
                "doi":       missing[best_hid].get("doi",""),
                "proxy_url": source,
                "status":    "manual" if source == "manual" else "reconcile",
                "bytes":     dest.stat().st_size,
                "timestamp": datetime.utcnow().isoformat(),
            })
            already_matched.add(best_hid)
        imported += 1

    print(f"\n── Done {'(dry run) ' if args.dry_run else ''}───────────────────────────────")
    print(f"  Imported  : {imported}")
    print(f"  On disk   : {len(list(PDF_DIR.glob('*.pdf')))}/117")
    if not args.dry_run and imported:
        print(f"\n  Regenerate checklist: python scripts/_gen_checklist_by_publisher.py")

if __name__ == "__main__":
    main()
