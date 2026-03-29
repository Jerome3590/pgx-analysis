"""
stop_slop_validate.py  —  Stop Slop AI Prose Refiner
Scans manuscript QMD files for AI-slop patterns per the Stop Slop skill.
Categories: banned phrases, structural clichés, sentence-level rules.
Scores each chapter 1-10 on Directness, Rhythm, Trust, Authenticity, Density.
Usage:  python scripts/stop_slop_validate.py
"""
import re
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).parent.parent
CHAPTERS = [
    "CH_1/ch01_bmic.qmd",
    "CH_2/ch02_psp.qmd",
    "CH_3/ch03_cts.qmd",
    "CH_4/ch04_psp.qmd",
    "CH_5/ch05_bmic.qmd",
    "CH_6/ch06_conclusion.qmd",
]
WORD_TARGETS = {
    "ch01_bmic.qmd":       (5500, 7000),
    "ch02_psp.qmd":        (4500, 5500),
    "ch03_cts.qmd":        (4000, 5000),
    "ch04_psp.qmd":        (4500, 5500),
    "ch05_bmic.qmd":       (5500, 7000),
    "ch06_conclusion.qmd": (4000, 6000),
}

# ── Pattern definitions ──────────────────────────────────────────────────────
PATTERNS = {
    # BANNED PHRASES
    "throat_clear":  [
        r"\bit is worth noting\b", r"\bit is important to\b",
        r"\bit should be noted\b", r"\bit is noteworthy\b",
        r"\bit is interesting to note\b", r"\bit is evident that\b",
        r"\bnotably,\b", r"\bimportantly,\b", r"\bcrucially,\b",
    ],
    "emphasis_crutch": [
        r"\bpivotal\b", r"\bgroundbreaking\b", r"\bseamlessly\b",
        r"\bunprecedented\b", r"\brevolutionar\w+\b", r"\bstate-of-the-art\b",
        r"\bcutting-edge\b", r"\bcutting edge\b", r"\bgame-changer\b",
        r"\btransformative\b(?! research| implications)",
    ],
    "business_jargon": [
        r"\bleverag\w+\b", r"\bdelve\b", r"\bsynerg\w+\b(?! drug| DDI| pair| inter)",
        r"\bholistic\b", r"\becosystem\b(?! of drug| model| data lake)",
        r"\bparadigm\b(?! shift in| of pharmacoep)", r"\bharnessing\b",
        r"\bstakeholder\b", r"\bsolution\b(?! to the| space| set)",
    ],
    "vague_declarative": [
        r"\bthis (?:study|work|paper|approach) (?:demonstrates|shows|illustrates|highlights)\b",
        r"\bthese results (?:demonstrate|show|highlight|underscore|suggest)\b",
        r"\bthis (?:represents|constitutes) a significant\b",
        r"\bit can be seen that\b", r"\bit is clear that\b",
    ],
    "meta_commentary": [
        r"\bin the following (?:section|paragraph)\b",
        r"\bwe will (?:now |next )?discuss\b",
        r"\bthe rest of this (?:section|paper|chapter)\b",
        r"\bas mentioned (?:above|previously|earlier)\b",
        r"\bas (?:discussed|noted|stated) (?:above|previously|in)\b",
    ],
    "adverbs_filler": [
        r"\bsubstantially\b(?! reduc| improv| differ| increas| decreas)",
        r"\bsignificantly\b(?! associat| differ| P\s*[<=]| p\s*[<=]| reduc| improv)",
        r"\beffectively\b(?! address| prevent| eliminat| identify| detect)",
        r"\bnoticeably\b", r"\bmarkedly\b(?! higher| lower| differ)",
        r"\bdramatically\b(?! reduc| increas| improv)",
    ],
    # STRUCTURAL CLICHÉS
    "binary_contrast": [
        r"\bon one hand.*on the other hand\b",
        r"\bwhile.*conversely\b",
        r"\bnot only.*but also\b",
    ],
    "false_agency": [
        r"\bthis (?:paper|study|work|approach|method|framework) (?:seeks|aims|tries|attempts|strives)\b",
        r"\bthe (?:data|results|findings|analysis) (?:reveals|shows|tells|argues|claims)\b",
        r"\bour (?:results|findings|models) (?:suggest|indicate) that (?:we|one should)\b",
    ],
    "rhetorical_setup": [
        r"\bthe question (?:remains|is|arises|then becomes)\b",
        r"\bone (?:might|may|could) (?:ask|wonder|argue|suggest)\b",
        r"\bthis raises the (?:question|issue|concern)\b",
        r"\bwhy (?:does|is|are|do) this matter\b",
    ],
    # SENTENCE-LEVEL RULES
    "wh_starter": [
        r"^(?:What|When|Where|Which|While|Who|Whose|Whom|Why|How) ",
    ],
    "em_dash": [
        r"—",
        r" -- ",
    ],
    "lazy_extremes": [
        r"\b(?:always|never|all|every|none|no one|everyone)\b"
        r"(?! (?:age band|cohort|partition|model|year|patient|claim|drug|band|bin"
        r"|feature|threshold|case|split|hold|train|test|data|measure|variable"
        r"|band|result|metric|version|arm|stage|phase|step|method))",
    ],
}

# Lines to skip (code blocks, YAML, table rows, LaTeX, comments)
SKIP_RE = re.compile(
    r"^\s*(?:```|%|#|---|abstract:|keywords:|affiliations?:|author|"
    r"title:|date:|format:|bibliography:|execute:|lang:|tbl-|fig-|"
    r"\|.*\||\$\$|\\|\[.*\]:\s*http)"
)


def in_skip_line(line: str) -> bool:
    return bool(SKIP_RE.match(line))


def scan_file(path: Path) -> dict:
    """Return dict of category -> list of (lineno, text) hits."""
    hits = defaultdict(list)
    in_code = False
    for lineno, raw in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        stripped = raw.strip()
        if stripped.startswith("```"):
            in_code = not in_code
        if in_code or in_skip_line(raw):
            continue
        for cat, pats in PATTERNS.items():
            for pat in pats:
                flags = re.IGNORECASE if cat != "wh_starter" else 0
                if re.search(pat, raw, flags):
                    hits[cat].append((lineno, raw.strip()[:110]))
                    break  # one hit per line per category
    return hits


def score_chapter(hits: dict, total_lines: int) -> dict:
    """
    Score 1-10 on each dimension. Higher hit rate → lower score.
    """
    n = sum(len(v) for v in hits.values())
    density_ratio = n / max(total_lines, 1)

    def dim(cats, weight=1.0):
        count = sum(len(hits.get(c, [])) for c in cats)
        # 10 = zero hits; degrades ~1 pt per 3 hits (weighted)
        return max(1, round(10 - (count * weight)))

    directness = dim(["throat_clear", "meta_commentary", "vague_declarative"])
    rhythm     = dim(["binary_contrast", "wh_starter", "em_dash"])
    trust      = dim(["rhetorical_setup", "false_agency", "vague_declarative"])
    authentic  = dim(["emphasis_crutch", "business_jargon", "adverbs_filler"])
    density    = dim(["adverbs_filler", "throat_clear", "meta_commentary"])

    # Cap at 10
    directness = min(directness, 10)
    rhythm     = min(rhythm, 10)
    trust      = min(trust, 10)
    authentic  = min(authentic, 10)
    density    = min(density, 10)

    total = directness + rhythm + trust + authentic + density
    return {
        "Directness": directness, "Rhythm": rhythm, "Trust": trust,
        "Authenticity": authentic, "Density": density,
        "TOTAL": total, "hits": n,
    }


def word_count(path: Path) -> int:
    return len(path.read_text(encoding="utf-8", errors="replace").split())


def main():
    print("\n" + "=" * 70)
    print("STOP SLOP VALIDATION — Manuscript QMD Files")
    print("=" * 70)

    chapter_results = []

    for ch in CHAPTERS:
        p = ROOT / ch
        if not p.exists():
            print(f"\n[MISSING] {ch}")
            continue

        lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
        hits  = scan_file(p)
        sc    = score_chapter(hits, len(lines))
        wc    = word_count(p)
        lo, hi = WORD_TARGETS.get(p.name, (0, 99999))
        wc_status = "OK" if lo <= wc <= hi else ("OVER" if wc > hi else "UNDER")

        chapter_results.append((p.name, sc, hits, wc, wc_status))

        flag = " *** NEEDS REVISION" if sc["TOTAL"] < 35 else ""
        print(f"\n{'─'*70}")
        print(f"  {p.name}   [{wc:,} words / {wc_status}]   Slop hits: {sc['hits']}{flag}")
        print(f"  Score: Directness={sc['Directness']}  Rhythm={sc['Rhythm']}  "
              f"Trust={sc['Trust']}  Authenticity={sc['Authenticity']}  "
              f"Density={sc['Density']}   TOTAL={sc['TOTAL']}/50")

        if hits:
            # Show top 5 hits per category (suppress empty)
            for cat, items in sorted(hits.items()):
                if not items:
                    continue
                print(f"\n    [{cat}]  ({len(items)} hit{'s' if len(items)>1 else ''})")
                for lineno, text in items[:4]:
                    print(f"      L{lineno:4d}: {text}")
                if len(items) > 4:
                    print(f"             … +{len(items)-4} more")

    # ── Summary table ────────────────────────────────────────────────────────
    print("\n\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'File':<28} {'Words':>7}  {'WC':>5}  {'Score':>7}  {'Hits':>5}  {'Status'}")
    print("-" * 70)
    for name, sc, _, wc, wc_status in chapter_results:
        flag = "REVISE" if sc["TOTAL"] < 35 else "PASS"
        print(f"{name:<28} {wc:>7,}  {wc_status:>5}  {sc['TOTAL']:>4}/50  "
              f"{sc['hits']:>5}  {flag}")


if __name__ == "__main__":
    main()
