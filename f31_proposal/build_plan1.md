<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# I dont use neural networks as a benchmark due to explainability issues. I learn how to use them for background information but GBTs show better performance on tabular data (ie APCD)

You’re right to distinguish between “I learned DL” and “I use it as a benchmark.” For your F31 training plan, we should frame deep learning as conceptual background, while clearly stating that explainability‑constrained GBTs are your primary modeling choice for APCD tabular data.[^1][^2][^3]

Here’s a revised version of the relevant training‑plan language you can swap in.

***

### Updated wording for Area 2 (for your F31)

Replace the earlier “Modeling Mastery” bullets with something like:

- **Modeling mastery**
    - BIOS 691: ST: Deep Learning with R – Provided conceptual and practical background in neural network architectures and regularization, which I use to understand the broader model landscape and limitations of deep learning in clinical prediction. I do **not** rely on neural networks as primary benchmarks in this project, because their opacity conflicts with the explainability and causal interpretability requirements for my “causal calculator” and PGx Risk Dashboard.[^2][^4][^1]
    - Instead, my primary modeling work focuses on gradient‑boosted trees (CatBoost/XGBoost) and related ensemble methods, which consistently outperform or match deep learning on high‑dimensional tabular data such as the Virginia APCD while allowing richer post‑hoc explanation and rule extraction.[^3][^2]
    - STAT 591: Applied Bayesian Statistical Analysis – Provides the theoretical basis for uncertainty quantification and motivates the use of Bayesian TPE (Optuna) for tuning these tree‑based ensembles, rather than relying on opaque, over‑parameterized deep nets in this setting.[^1][^2]

You can keep the rest of Area 2 (DTW, causal discovery, etc.) as is; this edit simply clarifies your stance.

***

If you’d like, I can now give you a clean, copy‑ready paragraph for the **Research Strategy – Approach** section that explicitly justifies: “We considered deep learning, but prioritize GBTs because of performance on tabular APCD data and compatibility with SHAP ∩ FFA causal explanations.”

<div align="center">⁂</div>

[^1]: F31-Training-Plan-for-Clinical-OODA-Loop-Research.md

[^2]: Causal-AI-and-Pharmacogenomics-for-Opioid-Risk-Prediction.md

[^3]: https://manusights.com/resources/nih-f31-guide

[^4]: https://grants.nih.gov/grants-process/write-application/advice-on-application-sections

