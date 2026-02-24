# PGx (Pharmacogenomics) — Legacy Doc

**Use [README.md](README.md) for current documentation.**

The current Step 5 pipeline is **CPIC drug counts only**:

- **Features:** `pgx_num_drugs`, `pgx_num_cpic_drugs` (no alleles, no gene-level features in this step).
- **Alleles:** Not used here. Alleles are used in the **PGx card** when patients submit SNP data with alleles encoded (dashboard/Lambda).

This file described an older workflow (e.g. `7_pgx_analysis` paths, `add_allele_frequencies.py`, drug–gene exposure counts). That workflow is superseded by the simpler CPIC drug-count approach in `README.md`.
