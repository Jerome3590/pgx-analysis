<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# I want specific references. One of my committtee members very knowledgeable of deep learning

You can absolutely cite concrete papers to justify favoring GBTs for APCD‑style tabular data. Here are two strong, citable anchors plus how to work them into your text.

## Key references you can use

1. **“Tabular data: Deep learning is not all you need”** (Kadra et al., 2021/2022)
    - Systematic comparison of several deep models vs XGBoost on diverse tabular datasets.
    - Main finding: XGBoost **outperforms** these deep models on most datasets and requires **less hyperparameter tuning**.[^1][^2][^3]
2. **“When do neural nets outperform boosted trees on tabular data?” (TabZilla)** (Grinsztajn et al., 2023/2024)
    - Compares 19 algorithms (NNs and GBDTs) across 176 tabular datasets.[^4][^5]
    - Main finding:
        - On average, GBDTs **do outperform** NNs, especially on **larger** and more **irregular** (skewed/heavy‑tailed) datasets.[^5]
        - For many datasets, differences are negligible; light tuning of a GBDT often matters more than choosing NNs vs GBDTs.[^4][^5]

These map nicely onto your APCD context: large, messy, skewed healthcare claims.

## How to phrase this in your F31

### In the Research Strategy – Approach

You might write something like:

> For high‑dimensional tabular data such as the Virginia All‑Payer Claims Database, we prioritize gradient‑boosted decision trees (GBDTs; e.g., XGBoost, CatBoost) over deep neural networks. Comparative studies of recent deep models for tabular data have found that XGBoost outperforms or matches state‑of‑the‑art deep architectures on most benchmarks while requiring substantially less hyperparameter tuning. A large‑scale meta‑analysis across 176 tabular datasets further shows that, on average, GBDTs outperform neural networks, particularly on large and statistically irregular datasets with skewed or heavy‑tailed feature distributions—properties characteristic of real‑world claims data. Given the additional requirement for stable, interpretable feature attributions in this project, tree‑based ensembles are the most appropriate primary modeling choice.[^2][^3][^1][^5][^4]

### In the Training Plan (clarifying your stance)

You can refine the Area 2 paragraph to:

> BIOS 691 (Deep Learning with R) provided essential background in neural architectures and regularization, which I use to understand the broader model landscape and its limits on clinical tabular data. However, following evidence that tree‑based gradient boosting (e.g., XGBoost, CatBoost) tends to outperform or match deep learning on heterogeneous tabular datasets—and especially on large, skewed datasets typical of health claims—I focus my research on GBDTs integrated with SHAP and Formal Feature Attribution rather than deep neural networks as primary models.[^1][^2][^5][^4]

If you’d like, I can help you translate those into properly formatted references (authors/year/journal) matching whatever citation style you use in the F31 (usually simple numeric or author–year).
<span style="display:none">[^10][^11][^12][^13][^14][^15][^6][^7][^8][^9]</span>

<div align="center">⁂</div>

[^1]: https://www.sciencedirect.com/science/article/abs/pii/S1566253521002360

[^2]: https://www.scribd.com/document/607128544/Tabular-Data-Deep-Learning-is-Not-All-You-Need

[^3]: https://huggingface.co/papers/2106.03253

[^4]: https://arxiv.org/abs/2305.02997

[^5]: https://arxiv.org/html/2305.02997v4

[^6]: https://www.aidancooper.co.uk/tree-based-models-vs-deep-learning/

[^7]: http://www.diva-portal.org/smash/get/diva2:1885958/FULLTEXT02.pdf

[^8]: https://www.emergentmind.com/papers/2305.02997

[^9]: https://arxiv.org/html/2402.03970v3

[^10]: https://www.marktechpost.com/2024/07/05/beyond-deep-learning-evaluating-and-enhancing-model-performance-for-tabular-data-with-xgboost-and-ensembles/

[^11]: https://github.com/orgs/community/discussions/188857

[^12]: https://thesis.eur.nl/pub/67870/Bachelor-Thesis-Ruben-Timmer.pdf

[^13]: https://sebastianraschka.com/blog/2022/deep-learning-for-tabular-data.html

[^14]: https://www.youtube.com/watch?v=59uGzJaVzYc

[^15]: https://kth.diva-portal.org/smash/get/diva2:1885958/FULLTEXT02.pdf

