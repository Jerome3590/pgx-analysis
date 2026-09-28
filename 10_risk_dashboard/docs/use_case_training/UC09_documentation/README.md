# UC09 — Documentation / model trust

**Live:** https://pgx.jerome-dixon.io/

**Persona:** All users. In-app User Guide is the trust surface for people who never open the repo.

**Live tabs:** User Guide

## Numbered steps

1. Open **User Guide**. Use the **Training** table for video / slides / audio. Files are in the [public Drive folder](https://drive.google.com/drive/folders/1cGbdoH-HDEooRyqWnhS6btWyhD4XOKQt?usp=drive_link) as `{nn}_{use_case}.mp4` / `.pdf` / `.m4a`.
2. Read **How to use**, research-question coverage, and the compact use-case summary.
3. Review **Model performance and at-risk identification** (2019 temporal holdout after leakage correction; selected model by cohort and age band from [Dixon & Price, *Clin Transl Sci*, doi:10.1111/cts.70690](https://doi.org/10.1111/cts.70690) Table 2 and [Dixon & Price, *Clin Transl Sci*, doi:10.1111/cts.70718](https://doi.org/10.1111/cts.70718) Table 2).
4. Review **Dashboard visual artifacts (from manifest)**.
5. Use **Feature importance sources for visuals** and **Event density bins** to interpret why BupaR/DTW vs FP-Growth can disagree, and why the badge matters for model routing.
6. Read **Unphased DNA files and star alleles** so AncestryDNA, 23andMe, MyHeritage, and VCF uploads are read as gene coverage and a clinical-test referral, not as a diplotype or a dose.


## Live button names

- **User Guide (primary tab)**

## Screenshot index

PNGs in `screenshots/`. These are captures of the live dashboard. Do not replace them with generated mockups. Recaptured 2026-09-28 as **clipped sections**.

- `01-documentation-how-to.png` — **Overview** + **Tabs**
- `02-model-performance.png` — **Model performance and at-risk identification**
- `03-visual-artifacts.png` — **Dashboard visual artifacts (from manifest)**
- `04-density-bins.png` — **Feature importance sources for visuals** + **Event density bins**
- `05-unphased-dna.png` — **Unphased DNA files and star alleles**: genealogy files cannot support a prescribing claim; coverage and **Data Not Present in File**

Canonical workflow text: repo `10_risk_dashboard/docs/DASHBOARD_USE_CASES.md`.

## NotebookLM prompt (required)

When creating Audio Overview, Video Overview, Slide Deck, or Briefing Doc:

- Use only the uploaded screenshots for any visual depiction of the PGx dashboard.
- Do not generate dashboard images, synthetic UI, or illustrative mockups.
- Describe the real screenshots; if a slide needs a figure, embed/refer to the uploaded PNG.
- Training slides should show or cite the captured screenshots, not AI-drawn interfaces.

