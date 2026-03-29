"""
stop_slop_validate.py  —  Stop Slop AI Prose Refiner
Source: https://github.com/hardikpandya/stop-slop  (MIT License, Hardik Pandya)
Pattern taxonomy sourced directly from:
  references/phrases.md   — throat-clearing, jargon, adverbs, meta-commentary
  references/structures.md — binary contrasts, false agency, em-dashes, Wh-starters
  SKILL.md                — scoring rubric (1-10 per dimension; <35/50 = revise)

Academic-writing exemptions are noted inline where standard scientific prose
would otherwise trigger a pattern (e.g., methods-section passive voice,
statistical adverbs adjacent to p-values).

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

# ── Pattern definitions (sourced from hardikpandya/stop-slop) ────────────────
#
# ACADEMIC EXEMPTIONS (patterns valid in scientific prose — not flagged):
#   • Methods-section passive voice ("were collected", "was computed") — standard
#   • Statistical adverbs paired with explicit p-values ("significantly, P<0.001")
#   • "All-Payer Claims Database" — proper noun, not lazy extreme
#   • "synergistic" as DDI pharmacology term (CH_4 subject matter)
#   • "data suggests/shows/indicates" — standard academic hedging with citation
#   • Author-contribution statement em-dashes (MDPI journal format)
#   • YAML front-matter em-dashes (institution names)

PATTERNS = {

    # ── phrases.md: Throat-Clearing Openers ──────────────────────────────────
    # "Remove these announcement phrases. State the content directly."
    "throat_clear": [
        r"\bhere'?s the thing\b",
        r"\bhere'?s (?:what|this|that|why)\b",
        r"\bthe uncomfortable truth is\b",
        r"\bit turns out\b",
        r"\bthe real \w+ is\b",
        r"\blet me be clear\b",
        r"\bthe truth is,",
        r"\bi'?ll say it again",
        r"\bi'?m going to be honest\b",
        r"\bcan we talk about\b",
        r"\bhere'?s what i find interesting\b",
        r"\bhere'?s the problem though\b",
        # filler phrases (also phrases.md)
        r"\bat its core\b",
        r"\bin today'?s \w+",
        r"\bit'?s worth noting\b",
        r"\bat the end of the day\b",
        r"\bwhen it comes to\b",
        r"\bin a world where\b",
        r"\bthe reality is\b",
    ],

    # ── phrases.md: Emphasis Crutches ────────────────────────────────────────
    # "These add no meaning. Delete them."
    "emphasis_crutch": [
        r"\bfull stop\.?\b",
        r"\blet that sink in\b",
        r"\bthis matters because\b",
        r"\bmake no mistake\b",
        r"\bhere'?s why that matters\b",
    ],

    # ── phrases.md: Business Jargon ──────────────────────────────────────────
    # "Replace with plain language."
    "business_jargon": [
        r"\bnavigate (?:the |these |those |our |this )?(?:challenge|complex|difficult|uncertain)\w*",
        r"\bunpack\b(?! the (?:model|algorithm|formula|equation|result))",
        r"\blean into\b",
        r"\b(?:the |this |our |a )?landscape\b(?! (?:of drug|pharmacoepid|clinical|of opioid))",
        r"\bgame.changer\b",
        r"\bdouble down\b",
        r"\bdeep dive\b",
        r"\btake a step back\b",
        r"\bmoving forward\b",
        r"\bcircle back\b",
        r"\bon the same page\b",
        # common LLM-tell jargon not in repo list but per Stop Slop spirit:
        r"\bleverag\w+\b",
        r"\bdelve\b",
        r"\bseamlessly\b",
        r"\bunprecedented\b",
        r"\bgroundbreaking\b",
        r"\bcutting.edge\b",
        r"\bstate.of.the.art\b",
        r"\btransformative\b",
        r"\bpivotal\b",
        r"\bharnessing\b",
    ],

    # ── phrases.md: Adverbs ───────────────────────────────────────────────────
    # "Kill all adverbs. No -ly words. No softeners, no intensifiers, no hedges."
    # Specific offenders listed in phrases.md:
    "adverbs": [
        r"\breally\b",
        r"\bjust\b(?! (?:\d|one|two|three|the|a |as|in|for|below|above|over|under))",
        r"\bliterally\b",
        r"\bgenuinely\b",
        r"\bhonestly\b",
        r"\bsimply\b(?! (?:put|stated|because|the|a |an ))",
        r"\bactually\b",
        r"\bdeeply\b(?! (?:nested|embedded|ingrained))",
        r"\btruly\b",
        r"\bfundamentally\b",
        r"\binherently\b",
        r"\binevitably\b",
        r"\binterestingly\b",
        r"\bimportantly,\b",
        r"\bcrucially,\b",
        # unlisted but common AI adverb offenders:
        r"\bnotably,\b",
        r"\bmarkedly\b",
        r"\bdramatically\b",
        r"\bsubstantially\b",
    ],

    # ── phrases.md: Meta-Commentary ──────────────────────────────────────────
    # "Remove self-referential asides. The essay should move."
    "meta_commentary": [
        r"\bhint:\b",
        r"\bplot twist:\b",
        r"\bspoiler:\b",
        r"\byou already know this, but\b",
        r"\bbut that'?s another (?:post|paper|chapter|section)\b",
        r"\bthe rest of this (?:essay|paper|section|chapter) (?:explains|describes|covers)\b",
        r"\blet me walk you through\b",
        r"\bin this section,? (?:we|i)'?ll\b",
        r"\bas we'?ll see\b",
        r"\bi want to explore\b",
        # academic variants:
        r"\bin the following (?:section|paragraph|chapter)\b",
        r"\bwe will (?:now |next )?discuss\b",
        r"\bas (?:discussed|noted|stated|mentioned) (?:above|previously|earlier|in section)\b",
    ],

    # ── phrases.md: Vague Declaratives ───────────────────────────────────────
    # "Sentences that announce importance without naming the specific thing."
    "vague_declarative": [
        r"\bthe reasons are structural\b",
        r"\bthe implications are significant\b",
        r"\bthis is the deepest problem\b",
        r"\bthe stakes are high\b",
        r"\bthe consequences are real\b",
        r"\bit can be seen that\b",
        r"\bit is clear that\b",
        r"\bit is evident that\b",
        # common academic vague-declarative variants:
        r"\bthis (?:represents|constitutes) a significant\b",
        r"\bthese (?:results|findings) (?:underscore|highlight) the (?:importance|need|value)\b",
    ],

    # ── structures.md: Binary Contrasts ──────────────────────────────────────
    # "These create false drama. State the point directly."
    "binary_contrast": [
        r"\bnot because .{5,60}\.? (?:but )?because\b",
        r"\bisn'?t the problem\.? .{0,20} is\b",
        r"\bthe answer isn'?t\b",
        r"\bnot X\.? (?:but|it'?s) Y\b",
        r"\bisn'?t X,? it'?s Y\b",
        r"\bnot just \w+ but also\b",
        r"\bstops being .{5,40} and starts being\b",
        r"\bdoesn'?t mean .{5,60} but actually\b",
    ],

    # ── structures.md: Rhetorical Setups ─────────────────────────────────────
    # "These announce insight rather than deliver it."
    "rhetorical_setup": [
        r"^what if ",
        r"\bhere'?s what i mean\b",
        r"\bthink about it:\b",
        r"\band that'?s okay\.?\b",
        # academic variants:
        r"\bthe question (?:remains|then becomes|arises)\b",
        r"\bthis raises the (?:question|issue|concern)\b",
        r"\bone (?:might|may|could) (?:wonder|ask whether)\b",
    ],

    # ── structures.md: False Agency ──────────────────────────────────────────
    # "AI loves this because it avoids naming the actor."
    # NOTE: "the data suggests/shows" is standard academic hedging with a citation;
    # flagged here but may be acceptable when followed by a numeric result/citation.
    "false_agency": [
        r"\bthe (?:culture|market|system|paradigm) (?:shifts?|rewards?|demands?|decides?)\b",
        r"\bthe conversation (?:moves?|shifts?) (?:toward|away)\b",
        r"\bthe decision emerges?\b",
        r"\ba (?:complaint|bug|error) becomes? (?:a )?(?:fix|feature|solution)\b",
        r"\bbets? (?:live|lives?|die|dies?) in\b",
        # narrator-from-a-distance (structures.md §Narrator-from-a-Distance):
        r"^nobody designed this\b",
        r"^this (?:is why|happens because)\b",
        r"^people tend to\b",
    ],

    # ── structures.md: Sentence Starters to Avoid ────────────────────────────
    # "Wh- openers become a crutch. Restructure. Lead with subject or verb."
    "wh_starter": [
        r"^(?:What|When|Where|Which|Who|Whose|Whom|Why|How) ",
        r"^So,? ",
        r"^Look,? ",
    ],

    # ── structures.md: Em-Dashes ──────────────────────────────────────────────
    # "Em-dash anywhere? Remove it." — exempt YAML front-matter and author
    # contribution lines (checked in scan_file via in_skip_line + in_yaml).
    "em_dash": [
        r"—",
    ],

    # ── structures.md: Word Patterns — Lazy Extremes ─────────────────────────
    # "False authority. Use specifics instead of sweeping claims."
    # Exempt: "All-Payer Claims Database" (proper noun), "all age bands" when
    # followed by a specific numeric qualifier, abbreviation table rows.
    "lazy_extremes": [
        r"\b(?:always|never|nobody|everybody|everyone)\b"
        r"(?! (?:in the |with a |who |that |achieve|exceed|pass|fail|reach))",
        r"\bevery (?:patient|study|model|partition|band|case)\b"
        r"(?! (?:partition|age band|\d))",
    ],
}

# Lines to skip: code blocks, YAML front-matter, table rows, LaTeX/TikZ,
# figure/table labels, abbreviation lists, references, author contributions
SKIP_RE = re.compile(
    r"^\s*(?:"
    r"```"
    r"|%"
    r"|#"
    r"|---"
    r"|abstract:"
    r"|keywords:"
    r"|affiliations?:"
    r"|author"
    r"|title:"
    r"|date:"
    r"|format:"
    r"|bibliography:"
    r"|execute:"
    r"|lang:"
    r"|tbl-"
    r"|fig-"
    r"|\|.*\|"          # table rows
    r"|\$\$"
    r"|\\"              # LaTeX commands
    r"|\[.*\]:\s*http"  # reference links
    r"|APCD,|ADE,|AUC|API,|AUROC|PR-AUC|IRB|NDC|ICD|CPT|SHAP|FFA|DTW|MCCV"
    r"|writing.{0,5}original draft"   # MDPI author contribution format
    r"|writing.{0,5}review"
    r")"
)


def in_skip_line(line: str) -> bool:
    return bool(SKIP_RE.match(line))


def scan_file(path: Path) -> dict:
    """Return dict of category -> list of (lineno, text) hits."""
    hits = defaultdict(list)
    in_code = False
    in_yaml = False
    for lineno, raw in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        stripped = raw.strip()
        # YAML front-matter toggle (first --- opens, second --- closes)
        if stripped == "---":
            in_yaml = not in_yaml
        if in_yaml:
            continue
        if stripped.startswith("```"):
            in_code = not in_code
        if in_code or in_skip_line(raw):
            continue
        for cat, pats in PATTERNS.items():
            for pat in pats:
                flags = re.IGNORECASE if cat not in ("wh_starter", "rhetorical_setup") else 0
                if re.search(pat, raw, flags):
                    hits[cat].append((lineno, raw.strip()[:115]))
                    break  # one hit per line per category
    return hits


# Dimension → category mapping per SKILL.md rubric
DIM_CATS = {
    # "Are they direct statements or just announcements?"
    "Directness":    ["throat_clear", "emphasis_crutch", "meta_commentary", "vague_declarative"],
    # "Is the text varied or metronomic?"
    "Rhythm":        ["em_dash", "wh_starter", "binary_contrast"],
    # "Does it respect reader intelligence?"
    "Trust":         ["rhetorical_setup", "false_agency", "vague_declarative"],
    # "Does it actually sound human?"
    "Authenticity":  ["business_jargon", "adverbs"],
    # "Is there anything cuttable?"
    "Density":       ["adverbs", "throat_clear", "meta_commentary", "lazy_extremes"],
}


def score_chapter(hits: dict) -> dict:
    """Score 1–10 on each SKILL.md dimension; total < 35 requires revision."""
    def dim_score(cats: list) -> int:
        count = sum(len(hits.get(c, [])) for c in cats)
        return max(1, min(10, 10 - count))

    scores = {d: dim_score(cats) for d, cats in DIM_CATS.items()}
    scores["TOTAL"] = sum(scores.values())
    scores["hits"]  = sum(len(v) for v in hits.values())
    return scores


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
        sc    = score_chapter(hits)
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
