"""Local tests for the raw-DNA card pipeline. No network."""

import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[2] / "10_risk_dashboard" / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from cpic_allele_resolver import allele_index_from_records, resolve_star_alleles
from pgx_exploratory_pipeline import run_exploratory_pipeline


ROWS = [
    {"genesymbol": "CYP2C19", "allele_name": "*2", "rsid": "rs4244285", "variantallele": "A"},
    {"genesymbol": "CYP2C19", "allele_name": "*17", "rsid": "rs12248560", "variantallele": "T"},
]


def test_pipeline_keeps_report_exploratory_and_opens_referral():
    index = allele_index_from_records(ROWS)
    genotypes = [
        {"rsid": "rs4244285", "genotype": "AG", "chromosome": "10", "position": 96541616},
        {"rsid": "rs12248560", "genotype": "CT", "chromosome": "10", "position": 96522463},
    ]
    calls = resolve_star_alleles(genotypes, index)
    report = run_exploratory_pipeline(
        genotypes,
        calls,
        local_pairs={"CYP2C19": [{"drug": "clopidogrel", "cpic_level": "A"}]},
        fetch=False,
    )
    assert report["stages"]["coordinates"]["withPosition"] == 2
    assert report["stages"]["coverage"]["sitesPresent"] == 2
    assert report["stages"]["cpicPharmvar"]["diplotypeAssigned"] is False
    finding = report["findings"][0]
    assert finding["diplotype"] is None
    assert finding["phenotype"] is None
    assert finding["associatedDrugs"][0]["drug"] == "clopidogrel"
    assert "Gene Coverage:" in finding["coverageText"]
    assert report["technicalNotice"]["isPhased"] is False
    assert report["technicalNotice"]["cnvSupported"] is False
    assert report["stages"]["upload"]["engine"] in ("duckdb-parquet", "python-rows")
    assert "DuckDB" in " ".join(report["architecture"])
    assert report["clinicalNudges"][0]["gene"] == "CYP2C19"
    assert report["clinicalNudges"][0]["clinicalAction"]["recommendation"] == "Follow-On Clinical Test Required"
    assert "CPIC guideline reference available for clopidogrel." in finding["guidelineReferences"]


def test_duckdb_parquet_interval_join(tmp_path):
    from pgx_exploratory_pipeline import (
        ancestry_text_to_parquet,
        join_user_to_cpic_targets,
        write_cpic_targets_parquet,
    )

    raw = tmp_path / "ancestry_raw.txt"
    raw.write_text(
        "# AncestryDNA\n"
        "rsid\tchromosome\tposition\tallele1\tallele2\n"
        "rs4244285\t10\t96541616\tA\tG\n",
        encoding="utf-8",
    )
    user_parquet = tmp_path / "user_dna.parquet"
    targets = tmp_path / "cpic_targets.parquet"
    written = ancestry_text_to_parquet(str(raw), str(user_parquet))
    assert written["parquet"] is True
    assert written["rowCount"] == 1
    write_cpic_targets_parquet(
        {"CYP2C19": {"chrom": "10", "start": 96540000, "stop": 96550000}},
        str(targets),
    )
    joined = join_user_to_cpic_targets(str(user_parquet), str(targets))
    assert joined == [{
        "chromosome": "chr10",
        "rsid": "rs4244285",
        "genotype": "AG",
        "gene": "CYP2C19",
    }]
