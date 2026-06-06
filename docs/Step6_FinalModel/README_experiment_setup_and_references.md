# Step 6 Experiment Setup and References

This note documents the metric rationale and citation set for interpreting Step 6 final-model outputs in the context of feature importance, SHAP, and formal feature attribution.

## Experiment setup

The final-model workflow is driven by `3_model_train_shap_ffa.ipynb`, which calls `6_final_model/run_final_model.py` for each `(cohort, age_band)`.

Current production setup:

- **Cohorts:** `opioid_ed` and `non_opioid_ed`
- **Age bands:** all configured age bands in `py_helpers.constants.REQUIRED_COHORTS`
- **Training window:** 2016-2018
- **Holdout window:** 2019
- **Default Step 6 mode:** `--train-mode per_bin`
- **Sparse-bin behavior:** density bins without enough class support use pooled full-cohort fallback artifacts and write `INFERENCE_SOURCE.txt`
- **Primary modeling objective:** stable feature importance, SHAP, and formal feature attribution
- **Operational prediction objective:** monitored, but secondary to attribution readiness for this analysis

Recommended idempotent notebook flags when rerunning both cohorts:

```python
FORCE_STEP4_ALL = False
FORCE_STEP4_NON_OPIOID = False
FORCE_STEP5 = False
CLEAN_STEP6_DOWNSTREAM_ARTIFACTS = False
FORCE_STEP6 = False
FORCE_STEP6_REBUILT_ONLY = False
```

With these settings, complete local/S3 artifacts are skipped, while missing or incomplete cohort/age-band artifacts regenerate.

## Metric interpretation for attribution

For imbalanced adverse-event cohorts, PR-AUC/AUPRC is prioritized over ROC-AUC and fixed-threshold recall because it summarizes positive-class retrieval across thresholds and is less dominated by true negatives.

Recall at threshold `0.5` is treated as an operating-point metric. Low recall at this threshold may indicate conservative calibration or threshold choice rather than absence of learnable signal.

For feature attribution, the key requirement is that the model captures non-random predictive structure with stable feature variance. SHAP and model-based feature importance explain the fitted model's predictions; they are not causal effects by themselves. Causal interpretation requires additional assumptions, study design, and attention to confounding.

## Suggested README wording

```markdown
For imbalanced adverse-event cohorts, PR-AUC/AUPRC is prioritized over ROC-AUC and fixed-threshold recall because it summarizes positive-class retrieval across thresholds and is less dominated by true negatives. Recall at threshold 0.5 is treated as an operating-point metric: low recall may indicate conservative calibration or threshold choice rather than absence of learnable signal. For feature attribution, the key requirement is that the model captures non-random predictive structure with stable feature variance; SHAP and model-based feature importance explain the fitted model's predictions, not causal effects by themselves. Causal interpretation requires additional assumptions and study design.
```

## Core references

### AUPRC / PR-AUC for imbalanced prediction

#### Saito and Rehmsmeier, 2015

Saito T, Rehmsmeier M. The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets. *PLOS ONE*. 2015.

- **Use:** Primary citation for preferring PR-AUC/AUPRC under class imbalance.
- **Key point:** ROC-AUC can look strong when true negatives dominate; PR-AUC focuses on positive-class retrieval.
- **DOI:** https://doi.org/10.1371/journal.pone.0118432

#### Davis and Goadrich, 2006

Davis J, Goadrich M. The Relationship Between Precision-Recall and ROC Curves. *Proceedings of the 23rd International Conference on Machine Learning*. 2006.

- **Use:** Foundational PR-vs-ROC reference.
- **Key point:** ROC and PR spaces are related, but PR curves are more informative when positive-class performance is the focus.
- **DOI:** https://doi.org/10.1145/1143844.1143874

#### Boyd, Eng, and Page, 2013

Boyd K, Eng KH, Page CD. Area under the Precision-Recall Curve: Point Estimates and Confidence Intervals. *Machine Learning and Knowledge Discovery in Databases*. 2013.

- **Use:** Quantitative support for AUPRC estimation and uncertainty.
- **DOI:** https://doi.org/10.1007/978-3-642-40994-3_29

### Recall as a threshold-dependent operating metric

#### scikit-learn precision-recall documentation

- **Use:** Practical implementation reference for precision, recall, F-measure, and PR curves.
- **Key point:** Fixed-threshold recall reflects one operating point; PR curves and AUPRC summarize behavior across thresholds.
- **Link:** https://scikit-learn.org/stable/modules/model_evaluation.html#precision-recall-f-measure-metrics

#### Hand, 2009

Hand DJ. Measuring classifier performance: a coherent alternative to the area under the ROC curve. *Machine Learning*. 2009.

- **Use:** Broader support for metric choice depending on decision costs and use case.
- **Key point:** Metric choice should match the application; screening, ranking, and attribution can prioritize different metrics.
- **DOI:** https://doi.org/10.1007/s10994-009-5119-5

### SHAP and feature attribution

#### Lundberg and Lee, 2017

Lundberg SM, Lee SI. A Unified Approach to Interpreting Model Predictions. *NeurIPS*. 2017.

- **Use:** Primary SHAP citation.
- **Key point:** SHAP attributes model predictions to input features; it explains fitted model behavior.
- **Link:** https://proceedings.neurips.cc/paper_files/paper/2017/hash/8a20a8621978632eace20843c7fd984-paper.pdf

#### Lundberg et al., 2020

Lundberg SM, Erion G, Chen H, et al. From local explanations to global understanding with explainable AI for trees. *Nature Machine Intelligence*. 2020.

- **Use:** TreeSHAP / tree ensemble explanation citation.
- **Key point:** Supports local-to-global explanation workflows for tree models.
- **DOI:** https://doi.org/10.1038/s42256-019-0138-9

#### Molnar, Interpretable Machine Learning

Molnar C. *Interpretable Machine Learning: A Guide for Making Black Box Models Explainable.*

- **Use:** Practical reference for feature importance, permutation importance, SHAP, partial dependence, and limitations.
- **Key point:** Feature attribution describes model behavior and should not be automatically interpreted as causal effect.
- **Link:** https://christophm.github.io/interpretable-ml-book/

### Causal interpretation caveats

#### Janzing, Minorics, and Blöbaum, 2020

Janzing D, Minorics L, Blöbaum P. Feature relevance quantification in explainable AI: A causal problem. *AISTATS*. 2020.

- **Use:** Direct citation for the relationship between feature relevance and causal assumptions.
- **Key point:** Feature relevance questions can be causal in nature and require assumptions beyond predictive attribution.
- **Link:** https://proceedings.mlr.press/v108/janzing20a.html

#### Hernán and Robins, Causal Inference: What If

Hernán MA, Robins JM. *Causal Inference: What If.*

- **Use:** Canonical causal inference reference.
- **Key point:** Causal claims require explicit assumptions about interventions, confounding, and identification.
- **Link:** https://www.hsph.harvard.edu/miguel-hernan/causal-inference-book/

#### Pearl, 2009

Pearl J. *Causality: Models, Reasoning, and Inference.* 2nd ed. Cambridge University Press. 2009.

- **Use:** Canonical DAG and structural-causal-model reference.
- **Key point:** Causal effects require assumptions about the causal graph, intervention, and confounding structure.

## Compact citation set

For a concise methods section, cite:

- **Saito and Rehmsmeier 2015** for PR-AUC under class imbalance.
- **Davis and Goadrich 2006** for PR/ROC relationship.
- **Lundberg and Lee 2017** for SHAP.
- **Janzing et al. 2020** or **Hernán and Robins** for causal interpretation caveats.
