"""
One-shot refactor: extract per-tab HTML from index.html into tabs/ files.
Run from any directory; uses absolute paths.
"""
import re, os, textwrap
from pathlib import Path

FRONTEND = Path(__file__).parent / "frontend"
INDEX    = FRONTEND / "index.html"
TABS_DIR = FRONTEND / "tabs"
TABS_DIR.mkdir(exist_ok=True)

# ── tab id → output filename ──────────────────────────────────────────────────
TAB_MAP = [
    ("risk-assessment-tab",               "risk-assessment.html"),
    ("drugs-tab",                         "drugs.html"),
    ("icd-codes-tab",                     "icd-codes.html"),
    ("cpt-codes-tab",                     "cpt-codes.html"),
    ("pgx-card-tab",                      "pgx-card.html"),
    ("causal-analysis-tab",               "causal-analysis.html"),
    ("feature-importance-visualizations-tab", "feature-importance.html"),
    ("bupar-visualizations-tab",          "bupar.html"),
    ("dtw-visualizations-tab",            "dtw.html"),
    ("fpgrowth-visualizations-tab",       "fpgrowth.html"),
    ("cohort-pgx-visualizations-tab",     "cohort-pgx.html"),
    ("documentation-tab",                 "documentation.html"),
]

html = INDEX.read_text(encoding="utf-8")

# ── Tab loader snippet inserted at the TOP of the <script> block ──────────────
TAB_ENTRIES = "\n".join(
    f"    ['{tid}', 'tabs/{fname}']," for tid, fname in TAB_MAP
)
LOADER = textwrap.dedent(f"""\
(async () => {{
  // ── Tab loader: eagerly populate all tab wrappers before JS init ──────────
  // Each tabs/*.html contains only the inner markup for that tab.
  // This keeps index.html lean (~400 lines) while all existing JS works
  // unchanged (DOM elements are present before any querySelector runs).
  const _TABS = [
{TAB_ENTRIES}
  ];
  await Promise.all(_TABS.map(async ([id, file]) => {{
    try {{
      const r = await fetch(file);
      if (r.ok) {{
        const el = document.getElementById(id);
        if (el) el.innerHTML = await r.text();
      }}
    }} catch (e) {{ console.warn('[tab-loader] Failed:', file, e); }}
  }}));

""")

LOADER_CLOSE = "\n})(); // end async IIFE – tab loader + dashboard init\n"

# ── 1. Extract each tab's inner content + blank it in the source ──────────────
def extract_tab(src: str, tab_id: str) -> tuple[str, str]:
    """
    Find <div id="{tab_id}" class="tab-content..."> ... </div>
    Return (inner_html, src_with_tab_emptied).
    Uses a simple depth counter to handle nested divs correctly.
    """
    # Find the opening tag (may include extra attributes like 'active')
    open_pat = re.compile(
        rf'(<div\b[^>]*\bid="{re.escape(tab_id)}"[^>]*>)', re.DOTALL
    )
    m = open_pat.search(src)
    if not m:
        raise ValueError(f"Tab not found: {tab_id}")

    tag_start = m.start()
    inner_start = m.end()
    opening_tag = m.group(1)

    # Walk forward counting open/close divs to find the matching close tag
    depth = 1
    pos = inner_start
    while depth > 0 and pos < len(src):
        open_m  = re.search(r'<div\b', src[pos:])
        close_m = re.search(r'</div>', src[pos:])
        if close_m is None:
            raise ValueError(f"Unbalanced div in tab: {tab_id}")
        if open_m and open_m.start() < close_m.start():
            depth += 1
            pos += open_m.start() + 4
        else:
            depth -= 1
            if depth == 0:
                inner_end = pos + close_m.start()
                tag_end   = pos + close_m.end()
            pos += close_m.start() + 6

    inner_html = src[inner_start:inner_end]

    # Replace inner content with nothing (keep wrapper div)
    emptied = src[:inner_start] + src[tag_end - len("</div>"):tag_end]
    # Rebuild: opening tag + </div>, then the rest
    emptied = src[:tag_start] + opening_tag + "</div>" + src[tag_end:]
    return inner_html, emptied


for tab_id, fname in TAB_MAP:
    inner, html = extract_tab(html, tab_id)
    out = TABS_DIR / fname
    out.write_text(inner, encoding="utf-8")
    print(f"  wrote {out.relative_to(FRONTEND.parent)}  ({len(inner):,} chars)")

# ── 2. Wrap existing JS in async IIFE ────────────────────────────────────────
SCRIPT_MARKER = "<!-- Main dashboard script moved to end of body for proper DOM loading -->\n<script>"
END_MARKER     = "\n</script>\n</body>"

idx_s = html.index(SCRIPT_MARKER) + len(SCRIPT_MARKER)
idx_e = html.rindex(END_MARKER)

js_body = html[idx_s:idx_e]

# Don't double-wrap if script was already rewritten
if "(async () => {" not in js_body:
    new_js = "\n" + LOADER + js_body + LOADER_CLOSE
    html = html[:idx_s] + new_js + html[idx_e:]
    print("  wrapped JS in async IIFE + tab loader")
else:
    print("  JS already wrapped – skipped")

INDEX.write_text(html, encoding="utf-8")
print(f"\nDone. index.html → {INDEX.stat().st_size:,} bytes")
print(f"Tab files in: {TABS_DIR}")
