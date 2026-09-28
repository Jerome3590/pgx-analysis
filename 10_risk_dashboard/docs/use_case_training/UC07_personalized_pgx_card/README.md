# UC07 — Personalized PGx Card from DNA

**Live:** https://pgx.jerome-dixon.io/

**Persona:** Clinician / pharmacist (Clinician / pharmacist view). Patient / self-assessment uses Patient view.

**Live tabs:** PGx Card

## Numbered steps

Optional refinement after (or instead of) the claims radar.

1. Under **1. Gene data**, leave **Session / patient pseudonym** blank for an anonymous session, or enter a label.
2. Enter **Gene / allele** lines (`CYP2D6,*1,*4`) **or** **Or upload file**: AncestryDNA, 23andMe, or MyHeritage (txt/csv/zip), Excel `.xlsx` (`Gene,Allele1,Allele2`), or unphased VCF.
3. Files are parsed **in this browser**. They are not uploaded to S3. Array rows go to `POST /pgx/card`, where DuckDB writes Snappy Parquet and joins positions to CPIC gene intervals. The card reports detected variants and gene coverage. Missing sites are **Data Not Present in File** and are not called `*1`. This path does not assign a diplotype, metabolizer status, or dose. A detected variant in `CYP2C19`, `CYP2D6`, `VKORC1`, `SLCO1B1`, or `HLA-B` opens a clinical-test referral. A lab line such as `CYP2C19,*1,*2` still uses the official phenotype table.
4. Under **2. Virginia APCD medications**, search generics (two characters), set **Drug scope** and **Actionable only**, and choose **Patient** or **Clinician / pharmacist**. Those controls shape lab-allele medication rows. They do not turn an array file into a dose list.
5. Click **Generate PGx results**. The request sends parsed rsid rows or lab alleles, plus selected APCD drugs and scope — not the raw genome file.
6. For lab alleles, review **Medication-first action queue**, **Gene–drug actionability matrix**, **Polypharmacy triplet engine**, and **Genes tested**. Unlisted pairs stay **indeterminate**, never “normal.” For an array file, review the exploratory finding, coverage line, and referral.
7. Export: **Print**, **Download JSON**, **Download CSV**, **Download PNG**, **Download PDF**, **Download summary for your doctor**, **Copy to clipboard**, **Technical appendix**, **Pharmacy handoff**.
8. **Send to pharmacy (coming soon)** stays disabled. There is no live e-prescribe and no in-dashboard lab order.


## Live button names

- **Generate PGx results**
- **Or upload file**
- **Drug scope**
- **Actionable only**
- **View (Patient / Clinician / pharmacist)**
- **Print / Download JSON / CSV / PNG / PDF / Download summary for your doctor / Copy to clipboard / Technical appendix / Pharmacy handoff**

## Screenshot index

PNGs in `screenshots/`. These are captures of the live dashboard. Do not replace them with generated mockups. Recaptured 2026-09-28 as **clipped sections**.

- `01-gene-data-entry.png` — **1. Gene data**: genealogy-file notice, medical notice, session label, lab `Gene,*allele` lines plus `rs4149056,TC`, **Or upload file**
- `02-apcd-meds-and-scope.png` — **2. Virginia APCD medications**: **Selected drugs**, three chips, **Generate PGx results**
- `03-generate-pgx-results.png` — **PGx results** with the further-testing card: **Further clinical test recommended** for SLCO1B1, plus the genealogy-claim notice
- `04-action-queue-matrix.png` — medication-first queue and gene–drug matrix for the lab-allele rows
- `05-exports.png` — export row, including **Download summary for your doctor**
- `06-triplets-genes-pharmacy.png` — triplets, **Genes tested**, **Send to pharmacy (coming soon)** disabled

Not a separate PNG (covered elsewhere or not a screen): an actual 23andMe/VCF file chosen in the picker (the control is in `01`); Patient view (UC08 `02-pgx-card-patient.png`); downloaded export files.

Canonical workflow text: repo `10_risk_dashboard/docs/DASHBOARD_USE_CASES.md`.

## NotebookLM prompt (required)

When creating Audio Overview, Video Overview, Slide Deck, or Briefing Doc:

- Use only the uploaded screenshots for any visual depiction of the PGx dashboard.
- Do not generate dashboard images, synthetic UI, or illustrative mockups.
- Describe the real screenshots; if a slide needs a figure, embed/refer to the uploaded PNG.
- Training slides should show or cite the captured screenshots, not AI-drawn interfaces.

