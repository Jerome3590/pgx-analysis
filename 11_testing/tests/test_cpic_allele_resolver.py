"""
Unit tests for the conservative unphased CPIC star-allele matcher.

Fixture rows use official public CPIC defining variants (same columns as
cpic_allele_rsid.parquet). They are not a substitute for the S3 tables and
do not invent extra rsid→star mappings.
"""

import sys
from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parents[2] / "10_risk_dashboard" / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from cpic_allele_resolver import (  # noqa: E402
    allele_index_from_records,
    official_rsids,
    parse_genotype,
    resolve_star_alleles,
    variant_copy_count,
)


# Official public CPIC defining variants used only to exercise the matcher.
# CYP2C19*2 rs4244285 A; *3 rs4986893 A; *17 rs12248560 T; *4 rs28399504 G
# CYP2D6*4 rs3892097 A; SLCO1B1*5 rs4149056 C
# Multi-variant example uses two official CYP2B6 *6 sites (rs3745274 T + rs2279343 G)
OFFICIAL_ROWS = [
    {"genesymbol": "CYP2C19", "allele_name": "*2", "rsid": "rs4244285", "variantallele": "A"},
    {"genesymbol": "CYP2C19", "allele_name": "*3", "rsid": "rs4986893", "variantallele": "A"},
    {"genesymbol": "CYP2C19", "allele_name": "*17", "rsid": "rs12248560", "variantallele": "T"},
    {"genesymbol": "CYP2C19", "allele_name": "*4", "rsid": "rs28399504", "variantallele": "G"},
    {"genesymbol": "CYP2D6", "allele_name": "*4", "rsid": "rs3892097", "variantallele": "A"},
    {"genesymbol": "SLCO1B1", "allele_name": "*5", "rsid": "rs4149056", "variantallele": "C"},
    {"genesymbol": "CYP2B6", "allele_name": "*6", "rsid": "rs3745274", "variantallele": "T"},
    {"genesymbol": "CYP2B6", "allele_name": "*6", "rsid": "rs2279343", "variantallele": "G"},
]


@pytest.fixture
def allele_index():
    return allele_index_from_records(OFFICIAL_ROWS)


def _call(calls, gene):
    return next((c for c in calls if c["gene"] == gene), None)


def test_parse_genotype_formats():
    assert parse_genotype("AG") == ["A", "G"]
    assert parse_genotype("A/G") == ["A", "G"]
    assert parse_genotype("A|T") == ["A", "T"]
    assert parse_genotype(["T", "C"]) == ["T", "C"]
    assert parse_genotype("DI") == ["D", "I"]


def test_variant_copy_count_exact_and_deletion():
    assert variant_copy_count(["A", "G"], "A") == 1
    assert variant_copy_count(["A", "A"], "A") == 2
    assert variant_copy_count(["G", "G"], "A") == 0
    assert variant_copy_count(["D", "G"], "del") == 1


def test_official_rsids_are_from_table_only(allele_index):
    rsids = official_rsids(allele_index)
    assert "rs4244285" in rsids
    assert "rs28399504" in rsids
    assert "rs4477212" not in rsids


def test_single_snp_het_stays_exploratory(allele_index):
    calls = resolve_star_alleles(
        [{"rsid": "rs4244285", "genotype": "AG"}],
        allele_index,
    )
    cyp = _call(calls, "CYP2C19")
    assert cyp["diplotype"] is None
    assert cyp["phenotype"] is None
    assert cyp["phenotypeConfidence"] == "EXPLORATORY"
    assert cyp["alleleCalls"] == []
    assert "*1" not in cyp["candidateAlleles"]
    assert "*2" in cyp["candidateAlleles"]
    detected = next(site for site in cyp["observedSites"] if site["rsid"] == "rs4244285")
    assert detected["status"] == "detected"
    assert detected["variantAlleleObserved"] is True
    assert any(site["status"] == "Data Not Present in File" for site in cyp["observedSites"])


def test_single_snp_hom_does_not_assign_diplotype(allele_index):
    calls = resolve_star_alleles(
        [{"rsid": "rs3892097", "genotype": "AA"}],
        allele_index,
    )
    cyp = _call(calls, "CYP2D6")
    assert cyp["diplotype"] is None
    assert "*4" in cyp["candidateAlleles"]
    assert any("CNV/Duplication not measured" in text for text in cyp["limitations"])


def test_two_disjoint_hets_require_phasing(allele_index):
    calls = resolve_star_alleles(
        [
            {"rsid": "rs4244285", "genotype": "AG"},
            {"rsid": "rs12248560", "genotype": "CT"},
        ],
        allele_index,
    )
    cyp = _call(calls, "CYP2C19")
    assert cyp["diplotype"] is None
    assert cyp["clinicalNudge"] is True
    assert "*2" in cyp["candidateAlleles"]
    assert "*17" in cyp["candidateAlleles"]
    assert "Phasing required" in cyp["exploratorySummary"]


def test_official_rsid_beyond_old_23_is_used(allele_index):
    calls = resolve_star_alleles(
        [{"rsid": "rs28399504", "genotype": "AG"}],
        allele_index,
    )
    cyp = _call(calls, "CYP2C19")
    assert cyp is not None
    assert cyp["diplotype"] is None
    assert "*4" in cyp["candidateAlleles"]


def test_unknown_rsid_is_ignored(allele_index):
    calls = resolve_star_alleles(
        [{"rsid": "rs4477212", "genotype": "AA"}],
        allele_index,
    )
    assert calls == []


def test_reference_at_defining_site_is_not_called_star1(allele_index):
    calls = resolve_star_alleles(
        [{"rsid": "rs4149056", "genotype": "TT"}],
        allele_index,
    )
    slco = _call(calls, "SLCO1B1")
    assert slco["diplotype"] is None
    assert slco["alleleCalls"] == []
    assert "*1" not in slco["candidateAlleles"]
    site = next(row for row in slco["observedSites"] if row["rsid"] == "rs4149056")
    assert site["status"] == "detected"
    assert site["variantAlleleObserved"] is False
    assert "not called as a normal (*1) allele" in slco["exploratorySummary"]


def test_incomplete_multivariant_is_indeterminate(allele_index):
    calls = resolve_star_alleles(
        [{"rsid": "rs3745274", "genotype": "GT"}],
        allele_index,
    )
    cyp = _call(calls, "CYP2B6")
    assert cyp["phenotypeConfidence"] == "EXPLORATORY"
    assert cyp["diplotype"] is None
    assert "*6" in cyp["candidateAlleles"]
    assert any("partly observed" in text.lower() for text in cyp["limitations"])


def test_complete_multivariant_prunes_subset_allele():
    index = allele_index_from_records(
        OFFICIAL_ROWS
        + [{"genesymbol": "CYP2B6", "allele_name": "*9", "rsid": "rs3745274", "variantallele": "T"}]
    )
    calls = resolve_star_alleles(
        [
            {"rsid": "rs3745274", "genotype": "TT"},
            {"rsid": "rs2279343", "genotype": "GG"},
        ],
        index,
    )
    cyp = _call(calls, "CYP2B6")
    assert cyp["diplotype"] is None
    assert "*6" in cyp["candidateAlleles"]
    assert "*9" not in cyp["candidateAlleles"]


def test_empty_genotypes_return_empty(allele_index):
    assert resolve_star_alleles([], allele_index) == []
    assert resolve_star_alleles([{"rsid": "rs4244285", "genotype": "AG"}], {}) == []
