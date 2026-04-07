"""validate_docx.py — structural validator for manuscript DOCX files.

Checks each chapter's DOCX against the canonical front-matter structure:

  [Title]              Title style
  [Author ×N]          Author style
  [Authors+markers]    BodyText  — first tp_elem (Dixon,^1,2^ ...)
  [Affiliations]       BodyText
  [^N affiliations]    BodyText
  [Corresponding Author] BodyText
  [Running Title]      BodyText
  [Figures · Tables]   BodyText
  [Keywords]           BodyText  — last tp_elem
  [page break paragraph]
  [Abstract heading]   AbstractTitle or Heading1 "Abstract"
  [abstract text]      Abstract style / BodyText starting with "Background"/"The"
  [Introduction H1]    present somewhere after abstract (may follow Study Highlights)
  [no "TITLE PAGE" text] anywhere in document

Usage:
  python templates/validate_docx.py                    # all chapters
  python templates/validate_docx.py output/edits/cts/ch01_cts_draft.docx
"""

from __future__ import annotations
import sys
from pathlib import Path
from docx import Document
from docx.oxml.ns import qn

# ── chapter map ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
CHAPTERS = {
    "ch01": ROOT / "output/edits/cts/ch01_cts_draft.docx",
    "ch02": ROOT / "output/edits/cpt_psp/ch02_psp_draft.docx",
    "ch03": ROOT / "output/edits/cts/ch03_cts_draft.docx",
    "ch04": ROOT / "output/edits/cpt_psp/ch04_psp_draft.docx",
    "ch05": ROOT / "output/edits/cpt/ch05_cpt_draft.docx",
}

# ── helpers ──────────────────────────────────────────────────────────────────
def get_style(elem) -> str:
    pPr = elem.find(qn("w:pPr"))
    if pPr is not None:
        ps = pPr.find(qn("w:pStyle"))
        if ps is not None:
            return ps.get(qn("w:val"), "")
    return ""

def get_text(elem) -> str:
    return "".join(t.text or "" for t in elem.iter(qn("w:t"))).strip()

def has_page_break(elem) -> bool:
    for br in elem.iter(qn("w:br")):
        if br.get(qn("w:type"), "") == "page":
            return True
    return False

def is_heading1(elem) -> bool:
    return get_style(elem) in ("Heading1", "1")

def paragraphs(doc) -> list:
    return [c for c in doc.element.body if c.tag == qn("w:p")]

# ── validation logic ─────────────────────────────────────────────────────────
def validate(path: Path) -> list[str]:
    errors: list[str] = []
    warnings: list[str] = []

    if not path.exists():
        return [f"FILE NOT FOUND: {path}"]

    doc = Document(str(path))
    paras = paragraphs(doc)

    texts  = [get_text(p) for p in paras]
    styles = [get_style(p) for p in paras]

    # ── 1. No "TITLE PAGE" text anywhere ─────────────────────────────────────
    for i, t in enumerate(texts):
        if t.strip() == "TITLE PAGE":
            errors.append(f"  [FAIL] 'TITLE PAGE' label still present at para {i}")

    # ── 2. Document starts with Title style ──────────────────────────────────
    if styles[0] != "Title":
        errors.append(f"  [FAIL] para 0 style='{styles[0]}' — expected 'Title'")

    # ── 3. Author paragraphs follow immediately ───────────────────────────────
    author_idxs = [i for i, s in enumerate(styles) if s == "Author"]
    if not author_idxs:
        errors.append("  [FAIL] no 'Author' style paragraphs found")
    elif author_idxs[0] != 1:
        errors.append(f"  [FAIL] first Author para at {author_idxs[0]}, expected 1")

    # ── 4. Title page block: Dixon, Affiliations:, Corresponding Author,
    #       Running Title, Figures, Keywords (all BodyText) ─────────────────
    last_author = max(author_idxs) if author_idxs else 0
    tp_start = last_author + 1
    tp_texts = texts[tp_start:tp_start + 20]   # wide enough for CH_3's extra fields

    checks = {
        "author line (Dixon,":   any("Dixon" in t and ("1" in t or "^" in t) for t in tp_texts),
        "Affiliations:":         any(t.strip() == "Affiliations:" for t in tp_texts),
        "affiliation 1 (Dept)":  any("Pharmacotherapy" in t for t in tp_texts),
        "Corresponding Author":  any(t.startswith("Corresponding Author") or t.startswith("Corresponding author") for t in tp_texts),
        "Running Title":         any("Running Title" in t or "Running title" in t for t in tp_texts),
        "Figures":               any(t.startswith("Figures") for t in tp_texts),
        "Keywords":              any(t.startswith("Keywords") or t.startswith("keyword") for t in tp_texts),
    }
    for label, ok in checks.items():
        if not ok:
            errors.append(f"  [FAIL] title page block missing: {label}")

    # ── 5. Page break after title page block ─────────────────────────────────
    # Find first page break paragraph after the title page block
    pb_idx = None
    for i in range(tp_start, min(tp_start + 15, len(paras))):
        if has_page_break(paras[i]):
            pb_idx = i
            break
    if pb_idx is None:
        errors.append("  [FAIL] no page break found after title page block")

    # ── 6. Abstract heading present ───────────────────────────────────────────
    abstract_idx = None
    for i, (s, t) in enumerate(zip(styles, texts)):
        if s in ("AbstractTitle",) and t == "Abstract":
            abstract_idx = i
            break
        if s in ("Heading1", "1") and t.strip().lower() == "abstract":
            abstract_idx = i
            break
    if abstract_idx is None:
        errors.append("  [FAIL] no 'Abstract' heading found")
    elif pb_idx is not None and abstract_idx <= pb_idx:
        errors.append(f"  [FAIL] Abstract heading (para {abstract_idx}) before page break (para {pb_idx})")

    # ── 7. Abstract text follows heading ─────────────────────────────────────
    if abstract_idx is not None:
        next_text = texts[abstract_idx + 1] if abstract_idx + 1 < len(texts) else ""
        if not next_text:
            errors.append("  [FAIL] abstract heading not followed by abstract text")

    # ── 8. First main-text heading present after abstract ────────────────────
    # Accepts "Introduction" or "Background" as the opening section label.
    intro_idx = None
    for i, t in enumerate(texts):
        if is_heading1(paras[i]) and ("Introduction" in t or "Background" in t):
            intro_idx = i
            break
    if intro_idx is None:
        errors.append("  [FAIL] Opening Heading1 (Introduction / Background) not found")
    elif abstract_idx is not None and intro_idx < abstract_idx:
        errors.append(f"  [FAIL] Opening heading (para {intro_idx}) appears before Abstract (para {abstract_idx})")

    # ── 9. No duplicate Abstract heading ─────────────────────────────────────
    abstract_hits = [i for i, t in enumerate(texts) if t.strip() == "Abstract" and styles[i] in ("AbstractTitle", "Heading1", "1")]
    if len(abstract_hits) > 1:
        errors.append(f"  [FAIL] duplicate Abstract headings at paras {abstract_hits}")

    # ── 10. Study Highlights: PSP/CPT only — must appear BEFORE Introduction ─
    # CTS chapters (ch01, ch03) do not move Study Highlights; skip check.
    chapter_stem = path.stem  # e.g. "ch01_cts_draft"
    is_psp_cpt = any(x in chapter_stem for x in ("psp", "cpt"))
    sh_idx = None
    for i, t in enumerate(texts):
        if is_heading1(paras[i]) and "Study Highlights" in t:
            sh_idx = i
            break
    if is_psp_cpt and sh_idx is not None and intro_idx is not None and sh_idx > intro_idx:
        errors.append(f"  [FAIL] Study Highlights (para {sh_idx}) comes AFTER Introduction (para {intro_idx})")

    return errors


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    if len(sys.argv) > 1:
        targets = {Path(a).stem: Path(a) for a in sys.argv[1:]}
    else:
        targets = CHAPTERS

    any_fail = False
    for name, path in targets.items():
        errors = validate(path)
        if errors:
            any_fail = True
            print(f"\n[FAIL] {name} — {path.name}")
            for e in errors:
                print(e)
        else:
            print(f"[PASS] {name} — {path.name}")

    if any_fail:
        sys.exit(1)
    else:
        print("\nAll checks passed.")


if __name__ == "__main__":
    main()
