"""
move_titlepage.py
Post-process a Quarto-generated DOCX to move the embedded title page block
(marked by "TITLE PAGE" ... "---" HR) to the very beginning of the document,
before the YAML-generated abstract and TOC.

Usage:
    python templates/move_titlepage.py output/edits/cpt_psp/ch02_psp_draft.docx
    python templates/move_titlepage.py output/edits/cpt_psp/   # all *.docx in dir
"""
import sys
import copy
from pathlib import Path
from docx import Document
from docx.oxml.ns import qn
from docx.oxml import OxmlElement


def get_text(elem) -> str:
    return "".join(t.text for t in elem.iter(qn("w:t")))


def get_style(elem) -> str:
    """Return the paragraph style name, or empty string."""
    if elem.tag != qn("w:p"):
        return ""
    pPr = elem.find(qn("w:pPr"))
    if pPr is not None:
        pStyle = pPr.find(qn("w:pStyle"))
        if pStyle is not None:
            return pStyle.get(qn("w:val"), "")
    return ""


def is_heading1(elem) -> bool:
    """True if the paragraph uses Heading1 style."""
    style = get_style(elem).lower()
    return "heading1" in style or style == "1"


def move_titlepage_to_front(docx_path: Path) -> None:
    doc = Document(str(docx_path))
    body = doc.element.body
    children = list(body)

    # 1. Find the TITLE PAGE marker paragraph
    tp_start = None
    for i, elem in enumerate(children):
        if elem.tag == qn("w:p") and "TITLE PAGE" in get_text(elem):
            tp_start = i
            break

    if tp_start is None:
        print(f"  {docx_path.name}: no TITLE PAGE marker — skipped")
        return

    # 2. Find the end: first Heading 1 paragraph after the title page block
    tp_end = None
    for i in range(tp_start + 1, len(children)):
        if is_heading1(children[i]):
            tp_end = i
            break

    if tp_end is None:
        print(f"  {docx_path.name}: no Heading 1 end boundary found — skipped")
        return

    # 3. Deep-copy title page elements (up to but not including the first heading).
    #    Skip the "TITLE PAGE" label paragraph itself — it is an internal marker only.
    tp_elems = [
        copy.deepcopy(children[i])
        for i in range(tp_start, tp_end)
        if get_text(children[i]).strip() != "TITLE PAGE"
    ]

    # 4. Remove them from their current position (reverse order to keep indices valid).
    #    Range is tp_start..tp_end-1 — do NOT remove tp_end, which is the first
    #    Heading 1 (Introduction) that acts only as the end-boundary marker.
    for i in range(tp_end - 1, tp_start - 1, -1):
        body.remove(children[i])

    # 5. Find insertion point.
    #
    #    Two cases:
    #    (a) Quarto already rendered an "Abstract" Heading 1 from the abstract: YAML
    #        key (typical for CTS and other journals).  Insert title page block
    #        BEFORE that heading so the abstract section is self-contained on page 2.
    #        Do NOT add a second "Abstract" heading.
    #    (b) Quarto did not render an "Abstract" heading (PSP/CPT with custom template).
    #        Insert before the "Background:" paragraph and add an "Abstract" Heading 1.
    body_after = list(body)
    insert_before          = None
    abstract_heading_exists = False

    for elem in body_after:
        if elem.tag != qn("w:p"):
            continue
        txt = get_text(elem).strip()
        if txt == "Abstract":
            insert_before = elem
            abstract_heading_exists = True
            break
        if "Background" in txt:
            insert_before = elem
            break

    if insert_before is None:
        insert_before = body_after[1] if len(body_after) > 1 else body_after[0]

    for elem in tp_elems:
        insert_before.addprevious(elem)

    # 6. Add "Abstract" Heading 1 only when Quarto did not already render one.
    if not abstract_heading_exists:
        abstract_h = OxmlElement("w:p")
        a_pPr     = OxmlElement("w:pPr")
        a_pStyle  = OxmlElement("w:pStyle")
        a_pStyle.set(qn("w:val"), "Heading1")
        a_pPr.append(a_pStyle)
        abstract_h.append(a_pPr)
        a_r = OxmlElement("w:r")
        a_t = OxmlElement("w:t")
        a_t.text = "Abstract"
        a_r.append(a_t)
        abstract_h.append(a_r)
        insert_before.addprevious(abstract_h)

    # 7. Insert a page break after the last title page element so the abstract
    #    starts on page 2.
    page_break_para = OxmlElement("w:p")
    page_break_run  = OxmlElement("w:r")
    page_break_br   = OxmlElement("w:br")
    page_break_br.set(qn("w:type"), "page")
    page_break_run.append(page_break_br)
    page_break_para.append(page_break_run)
    tp_elems[-1].addnext(page_break_para)

    doc.save(str(docx_path))
    print(f"  {docx_path.name}: TITLE PAGE label removed; 'Abstract' heading added; "
          f"Introduction heading preserved; page break inserted")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    target = Path(sys.argv[1])
    if target.is_file():
        move_titlepage_to_front(target)
    elif target.is_dir():
        found = list(target.rglob("*.docx"))
        if not found:
            print(f"No .docx files found in {target}")
            sys.exit(1)
        for f in found:
            move_titlepage_to_front(f)
    else:
        print(f"Not found: {target}")
        sys.exit(1)


if __name__ == "__main__":
    main()
