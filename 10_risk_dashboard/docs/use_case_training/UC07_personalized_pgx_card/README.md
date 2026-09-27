# UC07 — Personalized PGx Card from DNA

**Live:** https://pgx.jerome-dixon.io/

**Persona:** Clinician / pharmacist (Clinician / pharmacist view). Patient / self-assessment uses Patient view.

**Live tabs:** PGx Card

## Numbered steps

Optional refinement after (or instead of) the claims radar.

1. Under **1. Gene data**, leave **Session / patient pseudonym** blank for an anonymous session, or enter a label.
2. Enter **Gene / allele** lines (`CYP2D6,*1,*4`) **or** **Or upload file**: AncestryDNA or 23andMe (txt/csv/zip), Excel `.xlsx` (`Gene,Allele1,Allele2`), or VCF.
3. Files are parsed **in this browser**. They are not uploaded to S3. Official CPIC tables resolve stars and phenotypes. A VCF or DTC file is **unphased**: it shows which variants are present, not which chromosome copy each one sits on. Single-SNP stars can often be called; multi-variant or overlapping official alleles stay **indeterminate** (candidate alleles listed) so the card does not invent a haplotype. Prefer `CYP2C19,*1,*2` when the diplotype is already known. This is **not** full haplotype calling.
4. Under **2. Virginia APCD medications**, search generics (two characters), set **Drug scope** and **Actionable only**, and choose **Patient** or **Clinician / pharmacist**.
5. Click **Generate PGx results**. The request is **POST `/pgx/card`** with **parsed variants** (plus selected APCD drugs and scope) — not the raw genome file.
6. Review **Medication-first action queue**, **Gene–drug actionability matrix**, **Polypharmacy triplet engine**, and **Genes tested**. Rows are **action categories** plus guideline URLs. Unlisted allele pairs stay **indeterminate**, never “normal.”
7. Export: **Print**, **Download JSON**, **Download CSV**, **Download PNG**, **Download PDF**, **Copy to clipboard**, **Technical appendix**, **Pharmacy handoff**.
8. **Send to pharmacy (coming soon)** stays disabled. There is no live e-prescribe.


## Live button names

- **Generate PGx results**
- **Or upload file**
- **Drug scope**
- **Actionable only**
- **View (Patient / Clinician / pharmacist)**
- **Print / Download JSON / CSV / PNG / PDF / Copy to clipboard / Technical appendix / Pharmacy handoff**

## Screenshot index

PNGs in `screenshots/`. These are captures of the live dashboard. Do not replace them with generated mockups. Recaptured 2026-09-26 as **clipped sections** (the first pass used full-page shots; `04` and `05` were identical).

- `01-gene-data-entry.png` — **1. Gene data**: unphased-file helper, session label, `Gene,*allele` lines plus an official rsid genotype (`rs4149056,TC`), **Or upload file**
- `02-apcd-meds-and-scope.png` — **2. Virginia APCD medications**: **Selected drugs**, three chips (clopidogrel, gabapentin, alprazolam), Clinician / pharmacist, **Generate PGx results** plus export buttons
- `03-generate-pgx-results.png` — **PGx results** header after generate (session metadata, official phenotype-table / unphased notice, verification QR)
- `04-action-queue-matrix.png` — **Medication-first action queue** + **Gene–drug actionability matrix**
- `05-exports.png` — **Print / JSON / CSV / PNG / PDF / clipboard / Technical appendix / Pharmacy handoff**
- `06-triplets-genes-pharmacy.png` — **Polypharmacy triplet engine** (regimen three-way, not CPIC), **Genes tested**, **Gene details** (unphased limitations), **Send to pharmacy (coming soon)** disabled

Not a separate PNG (covered elsewhere or not a screen): an actual 23andMe/VCF file chosen in the picker (the control is in `01`); Patient view (UC08 `02-pgx-card-patient.png`); downloaded export files.

Canonical workflow text: repo `10_risk_dashboard/docs/DASHBOARD_USE_CASES.md`.

## NotebookLM prompt (required)

When creating Audio Overview, Video Overview, Slide Deck, or Briefing Doc:

- Use only the uploaded screenshots for any visual depiction of the PGx dashboard.
- Do not generate dashboard images, synthetic UI, or illustrative mockups.
- Describe the real screenshots; if a slide needs a figure, embed/refer to the uploaded PNG.
- Training slides should show or cite the captured screenshots, not AI-drawn interfaces.

