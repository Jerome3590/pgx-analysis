"""
Build data/scholar_json/{id}.json for:
  1. All 119 PDF articles (data/scholar_pdfs/) → PDF text extraction
  2. All included PMC articles (human_decision=include, have pmc_id) → BioC parse

Unified output schema:
{
  "id":          str,          # hsh_id or pmc_id
  "source_type": "pdf"|"pmc",
  "title":       str,
  "doi":         str,
  "pmc_id":      str,
  "pmid":        str,
  "year":        str,
  "authors":     [str, ...],
  "journal":     str,
  "abstract":    str,
  "full_text":   str,
  "word_count":  int,
  "sections":    [{"label": str, "text": str}, ...],
  "keywords":    [str, ...],
  "metadata":    {"extracted_at": str, "source_file": str, "page_count": int}
}

Usage:
  python scripts/_build_full_json.py            # all articles
  python scripts/_build_full_json.py --pdfs     # only 119 PDF articles
  python scripts/_build_full_json.py --pmc      # only PMC articles
  python scripts/_build_full_json.py --skip-existing   # skip already-built
"""
import argparse, csv, json, re
from datetime import datetime, timezone
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
PDF_DIR     = Path("data/scholar_pdfs")
SCHOLAR_JSON = Path("data/scholar_json")
DATA_DIR    = Path("data")
DOI_MAP_CSV = Path("scripts/screened_doi_map.csv")
SCREENED_CSV = Path("data/ontology/articles_screened.csv")
SCHOLAR_JSON.mkdir(exist_ok=True)

NOW = datetime.now(timezone.utc).isoformat()

# ── PDF extraction ─────────────────────────────────────────────────────────────
def extract_pdf(pdf_path: Path) -> dict:
    """Extract text from PDF using pdfminer; return pages list and full_text."""
    from pdfminer.high_level import extract_pages
    from pdfminer.layout import LTTextContainer

    pages = []
    try:
        for page_num, layout in enumerate(extract_pages(str(pdf_path)), 1):
            page_text = []
            for element in layout:
                if isinstance(element, LTTextContainer):
                    page_text.append(element.get_text())
            pages.append({"page": page_num, "text": " ".join(page_text)})
    except Exception as e:
        pages = [{"page": 1, "text": f"[extraction error: {e}]"}]

    full_text = "\n\n".join(p["text"] for p in pages)
    return {"pages": pages, "full_text": full_text, "page_count": len(pages)}


def guess_abstract(pages: list) -> str:
    """Heuristic: find text after 'abstract' keyword on first 2 pages."""
    for p in pages[:2]:
        text = p["text"]
        m = re.search(r"\bAbstract\b[:\s]*(.{100,1500}?)(?=\n\s*\n|\bIntroduction\b|\bBackground\b)",
                      text, re.IGNORECASE | re.DOTALL)
        if m:
            return m.group(1).strip()
    return ""


def build_pdf_json(hsh_id: str, doi_row: dict) -> dict:
    """Build unified JSON from a PDF file."""
    pdf_path = PDF_DIR / f"{hsh_id}.pdf"
    extracted = extract_pdf(pdf_path)
    pages = extracted["pages"]
    full_text = extracted["full_text"]
    abstract = guess_abstract(pages)

    return {
        "id":          hsh_id,
        "source_type": "pdf",
        "title":       doi_row.get("title", ""),
        "doi":         doi_row.get("doi", ""),
        "pmc_id":      "",
        "pmid":        "",
        "year":        "",
        "authors":     [],
        "journal":     "",
        "abstract":    abstract,
        "full_text":   full_text,
        "word_count":  len(full_text.split()),
        "sections":    [],
        "keywords":    [],
        "metadata": {
            "extracted_at": NOW,
            "source_file":  str(pdf_path),
            "page_count":   extracted["page_count"],
        },
    }


# ── BioC/PMC JSON parsing ──────────────────────────────────────────────────────
def parse_bioc(bioc_path: Path) -> dict:
    """Parse a BioC-format PMC JSON into unified schema."""
    with open(bioc_path, encoding="utf-8") as f:
        obj = json.load(f)
    root = obj[0] if isinstance(obj, list) else obj

    documents = root.get("documents", [])
    doc = documents[0] if documents else {}
    passages = doc.get("passages", [])

    # Extract metadata from first passage infons
    infons = passages[0].get("infons", {}) if passages else {}
    doi    = infons.get("article-id_doi", "")
    pmc_id = infons.get("article-id_pmc", "")
    pmid   = infons.get("article-id_pmid", "")
    year   = infons.get("year", "")
    journal = infons.get("journal", "")
    kwds   = infons.get("kwd", "").split() if infons.get("kwd") else []

    # Authors
    authors = []
    i = 0
    while f"name_{i}" in infons:
        entry = infons[f"name_{i}"]
        parts = dict(part.split(":") for part in entry.split(";") if ":" in part)
        name = f"{parts.get('given-names','')} {parts.get('surname','')}".strip()
        if name:
            authors.append(name)
        i += 1

    # Sections
    sections = []
    abstract_parts = []
    full_parts = []
    title = ""

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
            label = p.get("infons", {}).get("type", ptype)
            sections.append({"label": label, "text": text})

    abstract  = " ".join(abstract_parts)
    full_text = "\n\n".join(full_parts)

    return {
        "id":          pmc_id or bioc_path.stem,
        "source_type": "pmc",
        "title":       title,
        "doi":         doi,
        "pmc_id":      pmc_id,
        "pmid":        pmid,
        "year":        year,
        "authors":     authors,
        "journal":     journal,
        "abstract":    abstract,
        "full_text":   full_text,
        "word_count":  len(full_text.split()),
        "sections":    sections,
        "keywords":    kwds,
        "metadata": {
            "extracted_at": NOW,
            "source_file":  str(bioc_path),
            "page_count":   0,
        },
    }


# ── Find PMC JSON for an article ───────────────────────────────────────────────
# Build index: pmc_id → json_path
def build_pmc_index() -> dict[str, Path]:
    index = {}
    for jp in DATA_DIR.rglob("*.json"):
        if jp.parent.name == "scholar_json":
            continue
        index[jp.stem] = jp
    return index


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdfs",          action="store_true")
    parser.add_argument("--pmc",           action="store_true")
    parser.add_argument("--skip-existing", action="store_true", default=True)
    args = parser.parse_args()
    do_pdfs = args.pdfs or not args.pmc
    do_pmc  = args.pmc  or not args.pdfs

    doi_map = {r["screened_pmc_id"]: r
               for r in csv.DictReader(open(DOI_MAP_CSV, encoding="utf-8-sig"))}
    screened = list(csv.DictReader(open(SCREENED_CSV, encoding="utf-8-sig")))

    pmc_index = build_pmc_index()
    print(f"PMC JSON index: {len(pmc_index)} files")

    ok = err = skip = 0

    # ── Part 1: PDF articles ─────────────────────────────────────────────────
    if do_pdfs:
        print(f"\n── PDF articles ({len(doi_map)}) ─────────────────────────")
        for hsh_id, row in doi_map.items():
            out = SCHOLAR_JSON / f"{hsh_id}.json"
            if args.skip_existing and out.exists():
                skip += 1
                continue
            pdf = PDF_DIR / f"{hsh_id}.pdf"
            if not pdf.exists():
                print(f"  MISSING PDF: {hsh_id}")
                err += 1
                continue
            try:
                data = build_pdf_json(hsh_id, row)
                out.write_text(json.dumps(data, ensure_ascii=False, indent=2),
                               encoding="utf-8")
                wc = data["word_count"]
                pg = data["metadata"]["page_count"]
                print(f"  ✓ {hsh_id}  {pg}p  {wc:,}w  {row.get('title','')[:55]}")
                ok += 1
            except Exception as e:
                print(f"  ✗ {hsh_id}  {e}")
                err += 1

    # ── Part 2: PMC articles ─────────────────────────────────────────────────
    if do_pmc:
        included = [r for r in screened if r.get("human_decision") == "include"]
        print(f"\n── PMC included articles ({len(included)}) ─────────────────")
        no_pmc = no_json = 0
        for row in included:
            pmc_id = row.get("pmc_id", "").strip()
            art_id = row.get("article_id", "")
            out_id = pmc_id if pmc_id else f"article_{art_id}"
            out    = SCHOLAR_JSON / f"{out_id}.json"

            if args.skip_existing and out.exists():
                skip += 1
                continue
            if not pmc_id:
                no_pmc += 1
                continue
            jp = pmc_index.get(pmc_id)
            if not jp:
                no_json += 1
                continue
            try:
                data = parse_bioc(jp)
                # Merge metadata from screened CSV if BioC title is empty
                if not data["title"]:
                    data["title"] = row.get("title", "")
                out.write_text(json.dumps(data, ensure_ascii=False, indent=2),
                               encoding="utf-8")
                ok += 1
                if ok % 200 == 0:
                    print(f"  ... {ok} written")
            except Exception as e:
                print(f"  ✗ {pmc_id}  {e}")
                err += 1

        print(f"  No pmc_id:   {no_pmc}")
        print(f"  No JSON file: {no_json}")

    total = len(list(SCHOLAR_JSON.glob("*.json")))
    print(f"\n── Complete ──────────────────────────────────────────────")
    print(f"  Written  : {ok}")
    print(f"  Skipped  : {skip}")
    print(f"  Errors   : {err}")
    print(f"  Total in scholar_json/: {total}")


if __name__ == "__main__":
    main()
