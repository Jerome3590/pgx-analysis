<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# yes

Here’s a Quarto project layout you can use for your F31, plus example YAML that bakes in the NIH PDF rules and keeps each section as a separate attachment.[^1][^2]

***

## 1. Project structure

Create a project directory like:

- `f31/`
    - `_quarto.yml`
    - `specific-aims.qmd`
    - `research-strategy.qmd`
    - `training-career-goals.qmd`
    - `sponsor-statement.qmd`
    - `institutional-environment.qmd`
    - `rcr-plan.qmd`
    - `human-subjects-data.qmd` (Protection of Human Subjects + Data Management/Sharing)
    - `biosketch-candidate.qmd`
    - `biosketch-sponsor.qmd`

Each `.qmd` will compile to its own PDF with NIH‑compliant formatting.

***

## 2. Global Quarto config (`_quarto.yml`)

This sets NIH‑style margins, fonts, and disables headers/footers. Adjust file list as needed.

```yaml
project:
  type: book
  output-dir: output

execute:
  echo: false
  warning: false
  message: false

format:
  pdf:
    documentclass: article
    geometry: "margin=0.5in"
    mainfont: "Arial"
    fontsize: 11pt
    number-sections: false
    toc: false
    colorlinks: false
    linkcolor: blue
    urlcolor: blue
    header-includes: |
      \usepackage{setspace}
      \setstretch{1.05}
      \usepackage{fancyhdr}
      \pagestyle{plain} % no custom headers/footers; NIH will add
```

You don’t actually need the “book” features; this just makes it easy to manage multiple outputs. Each `.qmd` will override the title and output file name.

***

## 3. Example: `specific-aims.qmd`

```yaml
---
title: "Specific Aims"
format:
  pdf:
    output-file: "F31_Specific_Aims.pdf"
    keep-tex: true
---
```

Then paste the Specific Aims text we just drafted under that header.

***

## 4. Example: `research-strategy.qmd`

```yaml
---
title: "Research Strategy"
format:
  pdf:
    output-file: "F31_Research_Strategy.pdf"
---
```

Then use the Significance / Innovation / Approach skeleton, filling in from your attached draft.

***

## 5. Other common sections

You can clone this pattern for each attachment, just change `title` and `output-file`:

```yaml
---
title: "Training and Career Goals"
format:
  pdf:
    output-file: "F31_Training_Career_Goals.pdf"
---
```

```yaml
---
title: "Responsible Conduct of Research"
format:
  pdf:
    output-file: "F31_RCR.pdf"
---
```

```yaml
---
title: "Protection of Human Subjects and Data Management"
format:
  pdf:
    output-file: "F31_Human_Subjects_Data.pdf"
---
```

For Biosketches, you might prefer to generate them via SciENcv and not Quarto; if you do want them in Quarto, you can mirror the NIH biosketch headings and still use the same PDF formatting.

***

## 6. Workflow tip

- Use VS Code or RStudio with Quarto: open each `.qmd`, Render → you get a single NIH‑styled PDF per section.
- Periodically print to check true page count and adjust content to hit F31 limits (e.g., 1 page Specific Aims, 6 pages Research Strategy, etc.).[^3]

Do you want to prioritize a particular next attachment (e.g., Training \& Career Goals or Sponsor/Environment) so we can outline that content in the same level of detail as the Research Strategy?

<div align="center">⁂</div>

[^1]: https://grants.nih.gov/grants-process/write-application/how-to-apply-application-guide/format-attachments

[^2]: https://www.feinberg.northwestern.edu/sites/bsa/docs/nih-documentsetupguidelines.pdf

[^3]: https://grants.nih.gov/grants/how-to-apply-application-guide/forms-i/fellowship-forms-i.pdf

