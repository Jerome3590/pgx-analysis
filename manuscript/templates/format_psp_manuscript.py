"""
format_psp_manuscript.py
PSP-specific DOCX post-processor (replaces Pass 2 insert_docx_images.py).

For CPT:PSP submission, figures must be uploaded as SEPARATE files — NOT
embedded in the manuscript. This script:

  1. Replaces every [IMAGE: path] placeholder paragraph with a callout
     paragraph styled as:  [Figure N near here]
  2. Collects every [LEGEND:] paragraph (emitted by suppress_images_psp.lua),
     strips the [LEGEND:] prefix, and moves them to a "Figure Legends" section
     appended at the end of the document (after References), per PSP style guide.

Usage:
    python templates/format_psp_manuscript.py <file.docx> --chapter N
"""
import re
import sys
import argparse
import copy
from pathlib import Path
from docx import Document
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
from docx.shared import Pt

# ── Placeholder patterns ────────────────────────────────────────────────────

IMAGE_RE  = re.compile(r'^\[IMAGE:(.+?)(?::[\d.]+%?)?\]$')
LEGEND_RE = re.compile(r'^\[LEGEND:\]\s*')

# ── Figure filename → number maps (must match export_figures_psp.py order) ─

FIGURE_MAP = {
    2: {
        "pgx_architecture_clinical_ooda_loop.png": 1,
        "pgx_architecture_pipeline.png":           2,
        "pgx_architecture_consensus_filter.png":   3,
        "pgx_architecture_risk_dashboard.png":     4,
        "fig_attrition.png":                       5,
    },
    4: {
        "fig_shap.png":    1,
        "fig_network.png": 2,
        "fig_ir.png":      3,
        "fig_zcode.png":   4,
    },
    5: {
        "pgx_architecture_risk_dashboard.png": 1,
        "fig_imputation.png":                  2,
        "pgx_dashboard.png":                   3,
        "fig_latency.png":                     4,
    },
}


def get_text(para) -> str:
    return "".join(r.text for r in para.runs).strip()


def para_starts_with_legend(para) -> bool:
    text = get_text(para)
    return text.startswith("[LEGEND:]")


def strip_legend_marker(para) -> None:
    """Remove the leading [LEGEND:] text from the first run of a paragraph."""
    for run in para.runs:
        if "[LEGEND:]" in run.text:
            run.text = run.text.replace("[LEGEND:]", "").lstrip()
            return


def resolve_figure_num(path_str: str, ch_map: dict) -> int | None:
    filename = Path(path_str.strip()).name
    return ch_map.get(filename)


def add_heading(doc: Document, text: str) -> None:
    """Append a bold heading paragraph at the end of the document body."""
    body = doc.element.body
    p = OxmlElement("w:p")
    r = OxmlElement("w:r")
    rpr = OxmlElement("w:rPr")
    bold = OxmlElement("w:b")
    sz = OxmlElement("w:sz")
    sz.set(qn("w:val"), "24")  # 12pt
    rpr.append(bold)
    rpr.append(sz)
    r.append(rpr)
    t = OxmlElement("w:t")
    t.text = text
    r.append(t)
    p.append(r)
    # Insert before the last sectPr if present
    last = list(body)[-1]
    if last.tag == qn("w:sectPr"):
        last.addprevious(p)
    else:
        body.append(p)


def replace_runs_text(para_elem, new_text: str, italic: bool = False) -> None:
    """Clear all runs in a paragraph and set the first run to new_text."""
    from docx.text.paragraph import Paragraph
    ns = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
    runs = para_elem.findall(f"{{{ns}}}r")
    # Clear all runs
    for r in runs:
        for t in r.findall(f"{{{ns}}}t"):
            t.text = ""
    if runs:
        # Set text on first run; set italic if requested
        t_elem = runs[0].find(f"{{{ns}}}t")
        if t_elem is None:
            t_elem = OxmlElement("w:t")
            runs[0].append(t_elem)
        t_elem.text = new_text
        t_elem.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
        if italic:
            rpr = runs[0].find(f"{{{ns}}}rPr")
            if rpr is None:
                rpr = OxmlElement("w:rPr")
                runs[0].insert(0, rpr)
            i_elem = OxmlElement("w:i")
            rpr.append(i_elem)
    else:
        # No existing runs — create one
        r = OxmlElement("w:r")
        t_elem = OxmlElement("w:t")
        t_elem.text = new_text
        t_elem.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
        if italic:
            rpr = OxmlElement("w:rPr")
            i_elem = OxmlElement("w:i")
            rpr.append(i_elem)
            r.append(rpr)
        r.append(t_elem)
        para_elem.append(r)


def para_elem_text(para_elem) -> str:
    """Get full text of a w:p element by concatenating all w:t descendants."""
    ns = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
    return "".join(t.text or "" for t in para_elem.iter(f"{{{ns}}}t")).strip()


def format_psp(docx_path: Path, chapter: int) -> None:
    ch_map = FIGURE_MAP.get(chapter, {})
    if not ch_map:
        print(f"  WARNING: no figure map for chapter {chapter} — placeholders left as-is")

    doc = Document(str(docx_path))
    body = doc.element.body
    ns = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"

    # Iterate ALL w:p elements recursively (pandoc wraps figures in w:tbl/w:tr/w:tc)
    legend_elems        = []  # deep copies of caption paragraphs
    legend_to_remove    = []  # (parent, elem) to remove from table cells
    replaced = 0
    collected = 0

    for para_elem in list(body.iter(f"{{{ns}}}p")):
        text = para_elem_text(para_elem)
        m = IMAGE_RE.match(text)
        if not m:
            continue

        fig_num = resolve_figure_num(m.group(1), ch_map)
        label = f"Figure {fig_num}" if fig_num else Path(m.group(1)).stem
        replace_runs_text(para_elem, f"[{label} near here]", italic=True)
        replaced += 1
        print(f"  [{label} near here]  ← {Path(m.group(1)).name}")

        # Caption is the next w:p sibling inside the same w:tc parent
        parent = para_elem.getparent()
        if parent is not None and parent.tag == qn("w:tc"):
            siblings = list(parent)
            idx = siblings.index(para_elem)
            for sibling in siblings[idx + 1:]:
                if sibling.tag == f"{{{ns}}}p":
                    legend_elems.append(copy.deepcopy(sibling))
                    legend_to_remove.append((parent, sibling))
                    collected += 1
                    break

    # Remove caption paragraphs from their table cells
    for parent, elem in legend_to_remove:
        parent.remove(elem)

    # Append Figure Legends section at the end of the body
    if legend_elems:
        add_heading(doc, "Figure Legends")
        last = list(body)[-1]
        sect = last if last.tag == qn("w:sectPr") else None
        for legend_elem in legend_elems:
            if sect is not None:
                sect.addprevious(legend_elem)
            else:
                body.append(legend_elem)

    doc.save(str(docx_path))
    print(f"  {docx_path.name}: {replaced} callout(s) inserted, "
          f"{collected} legend(s) moved to Figure Legends section")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("docx", help="Path to the .docx file")
    parser.add_argument("--chapter", type=int, required=True,
                        help="Chapter number (for figure→number mapping)")
    args = parser.parse_args()

    docx_path = Path(args.docx).resolve()
    if not docx_path.exists():
        print(f"ERROR: {docx_path} not found")
        sys.exit(1)

    print(f"Processing (PSP): {docx_path.name}  [chapter {args.chapter}]")
    format_psp(docx_path, args.chapter)


if __name__ == "__main__":
    main()
