

## 1. Files to create in your F31 Quarto project

Create these `.qmd` files in your `f31/` project (or whatever name you chose):

- `specific-aims.qmd`
- `research-strategy.qmd`
- `training-plan.qmd` (Research Training Plan / Candidate Training)
- `candidate-goals.qmd` (Candidate’s Goals, Preparedness, Potential)
- `environment-sponsor.qmd` (Institutional Environment + Sponsor/Co-sponsor statements, once you have their text)

Optional but useful for the exam hand‑off:

- `project-summary.qmd` (Project Summary/Abstract)
- `project-narrative.qmd` (Public Health Relevance)


## 2. Global Quarto config (`_quarto.yml`)

In your project root, use this as a starting point:

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
      \pagestyle{plain}
```

This enforces NIH’s PDF rules (margins, font, no custom header/footer).[^3][^4]

## 3. Content blocks you already have (what to paste where)

### `specific-aims.qmd`

YAML:

```yaml
---
title: "Specific Aims"
format:
  pdf:
    output-file: "F31_Specific_Aims.pdf"
---
```

Body: paste the 1‑page Specific Aims I wrote for you earlier, adapted from your Causal‑AI/PGx proposal.[^1]

### `research-strategy.qmd`

YAML:

```yaml
---
title: "Research Strategy"
format:
  pdf:
    output-file: "F31_Research_Strategy.pdf"
bibliography: refs.bib
---
```

Body:

- Significance: from our skeleton, using your opioid/polypharmacy/PGx text.
- Innovation: Consensus‑Causal Filter, DTW archetypes, APCD pipeline, PGx from claims.[^1]
- Approach:
    - Overall Strategy + prelim data (APCD, throughput, PROBAST).
    - Aim 1 methods.
    - Aim 2 methods.
    - Aim 3 dashboard + privacy.
    - Include the **model‑choice paragraph** with specific GBT vs DL references:

```markdown
### Model selection for tabular APCD data

Although I have formal training in deep learning, this project deliberately
prioritizes tree-based gradient boosting methods (e.g., XGBoost, CatBoost)
for modeling high-dimensional tabular data from the Virginia All-Payer Claims
Database (APCD). Comparative studies of recent deep models for tabular data
have shown that XGBoost outperforms or matches state-of-the-art deep learning
architectures on most benchmark datasets, often with substantially less
hyperparameter tuning and greater robustness across tasks (Kadra et al., 2022).[web:131][web:147]
In a large-scale meta-analysis across 176 tabular datasets, Grinsztajn et al.
(2023) further report that, on average, gradient-boosted decision trees (GBDTs)
perform comparatively better than neural networks on larger and statistically
irregular datasets with skewed or heavy-tailed feature distributions—properties
characteristic of real-world claims data (Grinsztajn et al., 2023).[web:130][web:148]
Given these empirical results and the additional requirement in this project
for stable, interpretable feature attributions via SHAP and Formal Feature
Attribution (FFA), GBDTs are the most appropriate primary modeling choice,
while deep neural networks serve primarily as background knowledge rather
than benchmarks.
```


### `training-plan.qmd`

YAML:

```yaml
---
title: "Research Training Plan"
format:
  pdf:
    output-file: "F31_Training_Plan.pdf"
bibliography: refs.bib
---
```

Body:

- Use the four Areas we just finalized (Claims‑based Pharmacoepi; ML/Causal Discovery with updated Area 2; Translational Informatics/Architecture; Geriatrics \& Communication), plus the milestone summary toward June 2026.[^2]
- Make sure Area 2 uses the updated GBT‑vs‑DL wording and cites Kadra and Grinsztajn:

```markdown
### Area 2: Advanced Machine Learning and Deep Learning for Causal Discovery (Orient)

This area focuses on moving beyond purely predictive “black-box” models toward
causally interpretable ensembles that support the Consensus-Causal Filter and
Intervention Rate (IR) scoring.

- **Modeling mastery**

  - BIOS 691: ST: Deep Learning with R – Provided conceptual and practical
    background in neural network architectures and regularization, which I use
    to understand the broader model landscape and its limitations on clinical
    tabular data. I do not rely on neural networks as primary benchmarks in
    this project, because their opacity conflicts with the explainability and
    causal interpretability requirements for my “causal calculator” and PGx
    Risk Dashboard.[file:129]

  - Consistent with large comparative studies of tabular learning, I focus my
    modeling work on gradient-boosted trees (CatBoost/XGBoost) and related
    ensembles. Empirical evaluations by Kadra et al. and Grinsztajn et al.
    show that tree-based gradient boosting typically outperforms or matches
    deep neural networks on heterogeneous tabular datasets—especially on large,
    skewed, and irregular datasets that closely resemble all-payer claims—
    while requiring less hyperparameter tuning and integrating naturally with
    SHAP-based explanation (Kadra et al., 2022; Grinsztajn et al., 2023).[web:131][web:147][web:130][web:148]

  - STAT 591: Applied Bayesian Statistical Analysis – Provides the theoretical
    basis for uncertainty quantification and motivates the use of Bayesian TPE
    optimization (Optuna) for tuning these tree-based ensembles, rather than
    relying on over-parameterized deep nets as default models.[file:129]

  - DAPT 631 (Data Mining) and DAPT 632 (Forecasting Methods) – Supply applied
    predictive modeling techniques for longitudinal clinical trajectories,
    informing my use of Dynamic Time Warping (DTW) and trajectory archetyping
    within a tree-based ensemble framework.[file:129]

- **Applied certification**

  - AWS Certified Machine Learning – Specialty (previously held) – Validated
    my ability to design, deploy, and monitor large-scale inference pipelines
    and stratified ensemble models across age bands and risk strata, directly
    supporting the implementation of up to 84 gradient-boosted ensembles for
    opioid and polypharmacy risk prediction.[file:129][file:108]
```


### `candidate-goals.qmd` and `project-summary.qmd`

- `candidate-goals.qmd`: pull from your CV, transcript, and our earlier discussion: prior military/OODA background, APCD/causal AI work, AWS certs, clear gaps (causal inference, pharmacoepi, implementation science).
- `project-summary.qmd`: adapt the “Project Summary / Abstract” and “Project Narrative” from your Causal‑AI/PGx Markdown.[^1]


## 4. Reference file (`refs.bib`)

Create a `refs.bib` in the project root and include at least:

```bibtex
@article{kadra2022tabular,
  title   = {Tabular data: Deep learning is not all you need},
  author  = {Kadra, Ahmed and Lindauer, Marius and Hutter, Frank and Grabocka, Josif},
  journal = {Information Fusion},
  year    = {2022},
  volume  = {81},
  pages   = {84--90}
}

@article{grinsztajn2023when,
  title   = {When Do Neural Nets Outperform Boosted Trees on Tabular Data?},
  author  = {Grinsztajn, Leandro and Oyallon, Edouard and Varoquaux, Ga{\"e}l},
  journal = {arXiv preprint arXiv:2305.02997},
  year    = {2023}
}
```

You can add any other references you like (CPIC, PROBAST, etc.) as you grow the document.

## 5. Producing your “final artifacts”

Once everything is in place:

1. Open the project in VS Code or RStudio.
2. For each `.qmd` (Specific Aims, Research Strategy, Training Plan, Candidate, Summary), click **Render** to produce the corresponding PDF in `output/`.
3. Those PDFs are what you send to your committee as your F31‑formatted written exam package.

If you tell me which section you expect them to read most closely before the exam (Specific Aims vs Research Strategy vs Training Plan), I can give you a one‑page checklist for that section to skim right before you walk into the room.
<span style="display:none">[^10][^11][^12][^13][^14][^15][^16][^17][^18][^19][^5][^6][^7][^8][^9]</span>

<div align="center">⁂</div>

[^1]: Causal-AI-and-Pharmacogenomics-for-Opioid-Risk-Prediction.md

[^2]: F31-Training-Plan-for-Clinical-OODA-Loop-Research.md

[^3]: https://grants.nih.gov/grants-process/write-application/how-to-apply-application-guide/format-attachments

[^4]: https://www.feinberg.northwestern.edu/sites/bsa/docs/nih-documentsetupguidelines.pdf

[^5]: https://arxiv.org/abs/2305.02997

[^6]: https://www.aidancooper.co.uk/tree-based-models-vs-deep-learning/

[^7]: https://news.ycombinator.com/item?id=39601710

[^8]: https://dl.acm.org/doi/10.5555/3666122.3669459

[^9]: https://www.sciencedirect.com/science/article/abs/pii/S1566253521002360

[^10]: https://www.scribd.com/document/949560943/1-s2-0-S1566253521002360-main

[^11]: https://arxiv.org/pdf/2305.02997.pdf

[^12]: https://arxiv.org/html/2402.03970v3

[^13]: https://www.marktechpost.com/2024/07/05/beyond-deep-learning-evaluating-and-enhancing-model-performance-for-tabular-data-with-xgboost-and-ensembles/

[^14]: https://thesis.eur.nl/pub/67870/Bachelor-Thesis-Ruben-Timmer.pdf

[^15]: https://deepai.org/publication/when-do-neural-nets-outperform-boosted-trees-on-tabular-data

[^16]: https://sebastianraschka.com/blog/2022/deep-learning-for-tabular-data.html

[^17]: http://www.diva-portal.org/smash/get/diva2:1885958/FULLTEXT02.pdf

[^18]: https://www.youtube.com/shorts/hnMz2i_gSCs

[^19]: https://arxiv.org/html/2408.14817v1

