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

    # 3. Deep-copy title page elements (up to but not including the first heading)
    tp_elems = [copy.deepcopy(children[i]) for i in range(tp_start, tp_end)]

    # 4. Remove them from their current position (reverse order to keep indices valid)
    for i in range(tp_end, tp_start - 1, -1):
        body.remove(children[i])

    # 5. Find insertion point: just before the abstract ("Background:" paragraph).
    #    This keeps the YAML-generated title + authors at the top of page 1 and
    #    places our affiliations/COI block between the authors and the abstract.
    body_after = list(body)
    insert_before = None
    for elem in body_after:
        if elem.tag == qn("w:p") and "Background" in get_text(elem):
            insert_before = elem
            break

    if insert_before is None:
        # Fallback: insert after first element (YAML title heading)
        insert_before = body_after[1] if len(body_after) > 1 else body_after[0]

    for elem in reversed(tp_elems):
        insert_before.addprevious(elem)

    # 6. Insert a page break after the last title page element so the abstract
    #    starts on page 2.
    page_break_para = OxmlElement("w:p")
    page_break_run  = OxmlElement("w:r")
    page_break_br   = OxmlElement("w:br")
    page_break_br.set(qn("w:type"), "page")
    page_break_run.append(page_break_br)
    page_break_para.append(page_break_run)
    tp_elems[-1].addnext(page_break_para)

    doc.save(str(docx_path))
    print(f"  {docx_path.name}: title page block inserted before abstract (page break added)")


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
