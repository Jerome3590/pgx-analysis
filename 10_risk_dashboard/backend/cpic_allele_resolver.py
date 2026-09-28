"""
Exploratory matcher for unphased consumer array genotypes.

Uses official CPIC allele-definition rows only (typically from
cpic_allele_rsid.parquet). Does not invent rsid→star mappings.

Ancestry, 23andMe, MyHeritage, and unphased VCF files are not phased and do
not measure copy number. This matcher:

1. Records each official defining rsid as detected or "Data Not Present in File".
2. Lists named alleles whose defining variant alleles were observed.
3. Never assigns a diplotype, never fills a missing site with *1, and never
   returns a metabolizer phenotype.

Manual Gene,*1,*4 input should skip this module and go straight to phenotype
lookup.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

CALLING_ALGORITHM = "cpic-unphased-exploratory-v2"
MISSING_SITE_STATUS = "Data Not Present in File"
NUDGE_GENES = frozenset({"CYP2C19", "CYP2D6", "VKORC1", "SLCO1B1", "HLA-B"})
UNPHASED_NOTICE = (
    "Consumer raw data is unphased. The system cannot confirm whether these "
    "variants are on the same chromosome or opposite chromosomes, preventing a "
    "definitive CPIC star-allele diplotype assignment."
)
CPIC_BOUNDARY = "CPIC defines clinical guidelines for this gene when phased."

_DEL_TOKENS = frozenset({"D", "DEL", "-", "N", "."})
_INS_TOKENS = frozenset({"I", "INS", "+"})


def normalize_rsid(value: Any) -> str:
    text = str(value or "").strip()
    if not text or text.lower() in ("nan", "none", "null"):
        return ""
    return text.lower()


def normalize_allele_name(value: Any) -> str:
    text = str(value or "").strip()
    if not text or text.lower() in ("nan", "none", "null", "0"):
        return ""
    if text.lower().startswith("rs"):
        return text
    if not text.startswith("*"):
        return "*" + text.lstrip("*")
    return text


def _norm_base(value: Any) -> str:
    return str(value or "").strip().upper()


def _token_set(variantallele: Any) -> List[str]:
    raw = _norm_base(variantallele)
    if not raw or raw in ("NAN", "NONE", "NULL", "REF", "REFERENCE", "."):
        return []
    if raw in _DEL_TOKENS:
        return ["D", "DEL", "-"]
    if raw in _INS_TOKENS:
        return ["I", "INS", "+"]
    return [raw]


def parse_genotype(raw: Any) -> List[str]:
    """Return up to two observed bases/tokens from a genotype string or list."""
    if raw is None:
        return []
    if isinstance(raw, (list, tuple)):
        parts = [_norm_base(x) for x in raw if _norm_base(x)]
    else:
        text = str(raw).strip().upper().replace("|", "/").replace(" ", "")
        if "/" in text:
            parts = [_norm_base(p) for p in text.split("/") if _norm_base(p)]
        else:
            cleaned = "".join(ch for ch in text if ch.isalnum() or ch in "-+")
            if len(cleaned) >= 2 and cleaned[:2] in ("DEL", "INS"):
                parts = [cleaned]
            elif len(cleaned) == 1:
                parts = [cleaned]
            elif len(cleaned) >= 2:
                parts = [cleaned[0], cleaned[1]]
            else:
                parts = []
    out = []
    for part in parts:
        if part in (".", "N", "0"):
            continue
        if part in _DEL_TOKENS:
            out.append("D")
        elif part in _INS_TOKENS:
            out.append("I")
        else:
            out.append(part)
    return out[:2]


def variant_copy_count(observed: Sequence[str], variantallele: Any) -> int:
    tokens = set(_token_set(variantallele))
    if not tokens or not observed:
        return 0
    return sum(1 for base in observed if base in tokens or (base == "D" and tokens & _DEL_TOKENS) or (base == "I" and tokens & _INS_TOKENS))


def is_reference_allele(name: str, sites: Sequence[Mapping[str, Any]]) -> bool:
    bare = normalize_allele_name(name).upper().lstrip("*")
    if bare in {"1", "1A", "1.001", "REF", "REFERENCE", "WT"}:
        return True
    defining = [s for s in sites if _token_set(s.get("variantallele"))]
    return len(defining) == 0


def _pick_col(columns: Iterable[Any], *candidates: str) -> Optional[Any]:
    lowered = {str(col).lower().replace(" ", "").replace("_", ""): col for col in columns}
    for name in candidates:
        key = name.lower().replace(" ", "").replace("_", "")
        if key in lowered:
            return lowered[key]
    return None


def allele_index_from_records(rows: Iterable[Mapping[str, Any]]) -> Dict[str, Dict[str, List[Dict[str, str]]]]:
    """
    Build gene → allele → defining sites from official-shaped records.

    Accepted keys (official ingest / parquet): genesymbol, allele_name, rsid,
    variantallele. Also accepts name/dbsnpid aliases from the source join.
    """
    index: Dict[str, Dict[str, List[Dict[str, str]]]] = defaultdict(lambda: defaultdict(list))
    seen = set()
    for row in rows:
        gene = str(row.get("genesymbol") or row.get("gene") or "").upper().strip()
        allele = normalize_allele_name(row.get("allele_name") or row.get("name") or row.get("allele"))
        rsid = normalize_rsid(row.get("rsid") or row.get("dbsnpid"))
        variant = _norm_base(row.get("variantallele") or row.get("variant_allele"))
        if not gene or not allele or not rsid or gene == "NAN":
            continue
        key = (gene, allele, rsid, variant)
        if key in seen:
            continue
        seen.add(key)
        index[gene][allele].append({"rsid": rsid, "variantallele": variant})
    return {gene: dict(alleles) for gene, alleles in index.items()}


def allele_index_from_df(df: Any) -> Dict[str, Dict[str, List[Dict[str, str]]]]:
    """Build the allele index from an official CPIC allele-rsid DataFrame."""
    if df is None or getattr(df, "empty", True):
        return {}
    gene_col = _pick_col(df.columns, "genesymbol", "gene")
    allele_col = _pick_col(df.columns, "allele_name", "allelename", "name", "allele")
    rsid_col = _pick_col(df.columns, "rsid", "dbsnpid")
    var_col = _pick_col(df.columns, "variantallele", "variant_allele")
    if not gene_col or not allele_col or not rsid_col:
        return {}
    records = []
    for _, row in df.iterrows():
        records.append(
            {
                "genesymbol": row.get(gene_col, ""),
                "allele_name": row.get(allele_col, ""),
                "rsid": row.get(rsid_col, ""),
                "variantallele": row.get(var_col, "") if var_col else "",
            }
        )
    return allele_index_from_records(records)


def official_rsids(index: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]]) -> List[str]:
    out = set()
    for alleles in index.values():
        for sites in alleles.values():
            for site in sites:
                rsid = normalize_rsid(site.get("rsid"))
                if rsid:
                    out.add(rsid)
    return sorted(out)


def observed_from_genotypes(genotypes: Sequence[Mapping[str, Any]]) -> Dict[str, List[str]]:
    observed: Dict[str, List[str]] = {}
    for item in genotypes or []:
        rsid = normalize_rsid(item.get("rsid") or item.get("id"))
        alleles = parse_genotype(item.get("genotype") or item.get("alleles") or item.get("gt"))
        if not rsid or not alleles:
            continue
        observed[rsid] = alleles
    return observed


def _defining_sites(sites: Sequence[Mapping[str, Any]]) -> List[Tuple[str, str]]:
    out = []
    seen = set()
    for site in sites:
        rsid = normalize_rsid(site.get("rsid"))
        variant = _norm_base(site.get("variantallele"))
        if not rsid or not _token_set(variant):
            continue
        key = (rsid, variant)
        if key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out


def _score_allele(
    sites: Sequence[Mapping[str, Any]],
    observed: Mapping[str, Sequence[str]],
) -> Optional[Dict[str, Any]]:
    defining = _defining_sites(sites)
    if not defining:
        return None
    present = []
    missing = []
    copies = []
    for rsid, variant in defining:
        if rsid not in observed:
            missing.append(rsid)
            continue
        count = variant_copy_count(observed[rsid], variant)
        present.append(rsid)
        copies.append(count)
    if not present or any(c == 0 for c in copies):
        return None
    return {
        "sites_total": len(defining),
        "sites_matched": len(present),
        "sites_missing": missing,
        "complete": len(missing) == 0,
        "copies": min(copies) if copies else 0,
        "rsids": [rsid for rsid, _ in defining],
    }


def _prune_subset_alleles(candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Drop official alleles whose defining rsids are a proper subset of another complete match."""
    complete = [c for c in candidates if c["complete"]]
    drop = set()
    for i, left in enumerate(complete):
        left_set = set(left["rsids"])
        for j, right in enumerate(complete):
            if i == j:
                continue
            right_set = set(right["rsids"])
            if left_set < right_set:
                drop.add(left["name"])
    return [c for c in candidates if c["name"] not in drop]


def format_genotype(bases: Sequence[str]) -> str:
    if not bases:
        return ""
    if len(bases) == 1:
        return f"{bases[0]}/{bases[0]}"
    return f"{bases[0]}/{bases[1]}"


def _variant_hit_rsids(
    allele_defs: Mapping[str, Sequence[Mapping[str, Any]]],
    observed: Mapping[str, Sequence[str]],
) -> Dict[str, bool]:
    hits: Dict[str, bool] = {}
    for name, sites in allele_defs.items():
        if is_reference_allele(name, sites):
            continue
        for site in sites:
            rsid = normalize_rsid(site.get("rsid"))
            if not rsid or rsid not in observed:
                continue
            if variant_copy_count(observed[rsid], site.get("variantallele")) > 0:
                hits[rsid] = True
    return hits


def _site_reports(
    gene_rsids: Sequence[str],
    observed_gene: Mapping[str, Sequence[str]],
    hit_rsids: Mapping[str, bool],
) -> List[Dict[str, Any]]:
    reports = []
    for rsid in sorted(set(gene_rsids)):
        if rsid in observed_gene:
            genotype = format_genotype(observed_gene[rsid])
            reports.append({
                "rsid": rsid,
                "genotype": genotype,
                "status": "detected",
                "variantAlleleObserved": bool(hit_rsids.get(rsid)),
            })
        else:
            reports.append({
                "rsid": rsid,
                "genotype": None,
                "status": MISSING_SITE_STATUS,
                "variantAlleleObserved": False,
            })
    return reports


def _exploratory_summary(gene: str, sites: Sequence[Mapping[str, Any]], candidate_names: Sequence[str]) -> str:
    hits = [site for site in sites if site.get("variantAlleleObserved")]
    if len(hits) >= 2 or len(candidate_names) >= 2:
        return f"Multiple {gene} variants found. Phasing required for diplotype call."
    if len(hits) == 1:
        hit = hits[0]
        return f"Variant {hit.get('rsid')} ({hit.get('genotype')}) in {gene} detected in raw file."
    return (
        f"No defining variant allele was detected at measured {gene} sites. "
        "Sites missing from the file are marked Data Not Present in File and were not called as a normal (*1) allele."
    )


def resolve_gene(
    gene: str,
    allele_defs: Mapping[str, Sequence[Mapping[str, Any]]],
    observed: Mapping[str, Sequence[str]],
) -> Optional[Dict[str, Any]]:
    """Observe official sites. Do not assign a diplotype or a *1 default."""
    gene_u = (gene or "").upper()
    gene_rsids = {
        normalize_rsid(site.get("rsid"))
        for sites in allele_defs.values()
        for site in sites
        if normalize_rsid(site.get("rsid"))
    }
    observed_gene = {rsid: list(alleles) for rsid, alleles in observed.items() if rsid in gene_rsids}
    if not observed_gene:
        return None

    candidates = []
    for name, sites in allele_defs.items():
        if is_reference_allele(name, sites):
            continue
        scored = _score_allele(sites, observed)
        if not scored or scored["copies"] <= 0:
            continue
        scored["name"] = normalize_allele_name(name)
        candidates.append(scored)

    pruned = _prune_subset_alleles(candidates)
    names = sorted({c["name"] for c in pruned if c.get("name")})
    incomplete = [c for c in candidates if not c["complete"]]
    hit_rsids = _variant_hit_rsids(allele_defs, observed)
    sites = _site_reports(gene_rsids, observed_gene, hit_rsids)
    limitations = [UNPHASED_NOTICE, CPIC_BOUNDARY]
    missing = [site["rsid"] for site in sites if site["status"] == MISSING_SITE_STATUS]
    if missing:
        limitations.append(
            "Unmeasured variant sites are marked Data Not Present in File. "
            "They were not defaulted to a reference (*1) allele."
        )
    if incomplete:
        partial = sorted({c["name"] for c in incomplete})
        limitations.append(
            "Defining variants were only partly observed for: " + ", ".join(partial) + "."
        )
    if gene_u == "CYP2D6":
        limitations.append("CYP2D6 region evaluated for target SNPs. CNV/Duplication not measured.")
    else:
        limitations.append("Copy-number variation was not measured by this array file.")

    summary = _exploratory_summary(gene_u, sites, names)
    return {
        "gene": gene_u,
        "alleleCalls": [],
        "diplotype": None,
        "phenotype": None,
        "phenotypeConfidence": "EXPLORATORY",
        "analysisMode": "exploratory",
        "limitations": limitations,
        "candidateAlleles": names,
        "observedSites": sites,
        "exploratorySummary": summary,
        "clinicalNudge": gene_u in NUDGE_GENES and any(site.get("variantAlleleObserved") for site in sites),
        "cnvMeasured": False,
        "cpicBoundary": CPIC_BOUNDARY,
        "callingAlgorithm": CALLING_ALGORITHM,
    }


def resolve_star_alleles(
    genotypes: Sequence[Mapping[str, Any]],
    allele_index: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
    genes: Optional[Sequence[str]] = None,
) -> List[Dict[str, Any]]:
    """Resolve uploaded {rsid, genotype} rows to per-gene star calls."""
    observed = observed_from_genotypes(genotypes)
    if not observed or not allele_index:
        return []
    wanted = {(g or "").upper() for g in (genes or []) if g}
    calls = []
    for gene, defs in sorted(allele_index.items()):
        if wanted and gene not in wanted:
            continue
        call = resolve_gene(gene, defs, observed)
        if call:
            calls.append(call)
    return calls


def resolved_to_variants(calls: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Shape resolver output for generate_pgx_card()."""
    variants = []
    for call in calls:
        gene = str(call.get("gene") or "").upper()
        if not gene:
            continue
        alleles = list(call.get("alleleCalls") or [])
        variants.append(
            {
                "gene": gene,
                "variants": alleles,
                "resolved": True,
                "diplotype": None,
                "phenotype": None,
                "phenotypeConfidence": call.get("phenotypeConfidence") or "EXPLORATORY",
                "analysisMode": call.get("analysisMode") or "exploratory",
                "limitations": list(call.get("limitations") or []),
                "candidateAlleles": list(call.get("candidateAlleles") or []),
                "observedSites": list(call.get("observedSites") or []),
                "exploratorySummary": call.get("exploratorySummary"),
                "clinicalNudge": bool(call.get("clinicalNudge")),
                "cnvMeasured": False,
                "cpicBoundary": call.get("cpicBoundary") or CPIC_BOUNDARY,
                "callingAlgorithm": call.get("callingAlgorithm") or CALLING_ALGORITHM,
            }
        )
    return variants
