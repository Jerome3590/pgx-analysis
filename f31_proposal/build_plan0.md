<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# update training plan based on attached

Here’s an updated F31‑style Training Plan based on your attached document, with a bit more NIH‑friendly framing and explicit linkage to your AWS certifications and dissertation timeline. You can drop this into `training-career-goals.qmd` or a separate “Research Training Plan” attachment.[^1][^2][^3]

***

## F31 Research Training Plan: Clinical OODA Loop and Causal PGx Analytics

This fellowship will consolidate and extend my preparation in causal machine learning, pharmacoepidemiology, and translational informatics to support my long‑term goal of becoming an independent investigator in medication safety and pharmacogenomics. The plan is organized into four complementary training areas aligned with the Clinical OODA Loop (Observe–Orient–Decide–Act) and the CRISP‑DM framework.[^2][^4][^1]

### Area 1: Advanced Claims‑Based Pharmacoepidemiology (Observe)

This area builds the foundational skills needed to manage and analyze the Virginia All‑Payer Claims Database (APCD) and its 1.8 TB of longitudinal records for adverse drug event (ADE) risk.[^4][^1]

- Formal doctoral coursework
    - HGEN 611: Data Science I and HGEN 691: Methods in Data Science – Established the computational foundation for high‑throughput genomic and claims‑based data engineering, including reproducible pipelines and feature extraction.[^1]
    - BIOS 572: Biomedical Data I – Provided statistical methods for analyzing large‑scale biomedical datasets, directly informing my ADE risk modeling and validation strategy.[^1]
    - FMBA 614, 615, 616: Health Care Management I–III – Supplied regulatory and systems‑level context for healthcare delivery, payer behavior, and outcomes research, which shapes my APCD cohort definitions and policy‑relevant endpoints.[^1]
- Professional training
    - DAPT 611 (Analysis \& Design of Database) and DAPT 614 (Advanced SQL) – Support the partition‑first data architecture and efficient querying strategies needed to stage the APCD from Bronze to Gold layers and eliminate shuffle bottlenecks.[^4][^1]

Together, these experiences ensure I can reliably “observe” the clinical system through high‑quality claims data before moving to causal modeling.

### Area 2: Advanced Machine Learning and Deep Learning for Causal Discovery (Orient)

This area focuses on moving beyond black‑box accuracy toward causally interpretable models that support the Consensus‑Causal Filter and Intervention Rate (IR) scoring.[^4][^1]

- Modeling mastery
    - BIOS 691: ST: Deep Learning with R – Developed mastery of neural network ensembles that serve as a benchmark against gradient‑boosted trees (CatBoost/XGBoost) in my opioid and polypharmacy risk models.[^1]
    - STAT 591: Applied Bayesian Statistical Analysis – Provided the theoretical basis for uncertainty quantification and motivates the use of Bayesian TPE optimization in Optuna for hyperparameter tuning.[^4][^1]
    - DAPT 631 (Data Mining) and DAPT 632 (Forecasting Methods) – Offered applied predictive modeling tools for longitudinal clinical trajectories, informing my use of Dynamic Time Warping and trajectory archetyping.[^4][^1]
- Applied certification
    - AWS Certified Machine Learning – Specialty (previously held) – Validated my ability to manage large‑scale inference engines and orchestrate stratified ensemble models across age bands and risk strata, providing practical grounding for deploying up to 84 ensemble models in the dissertation work.[^1][^4]

This area ensures that I can “orient” within complex, high‑dimensional feature spaces and design models that are both robust and suitable for causal interrogation.

### Area 3: Translational Informatics and Big Data Architecture (Decide and Act)

This area targets the Decide and Act phases of the Clinical OODA Loop by ensuring that model insights are deployed as production‑grade, privacy‑preserving clinical decision support tools.[^4][^1]

- Applied certifications
    - AWS Certified Big Data Analytics – Provided expertise in scaling analytical workloads (e.g., DuckDB over S3 data lakes), which underpins the 15.1× throughput improvement achieved in my partition‑first APCD pipeline.[^1][^4]
    - AWS Certified Solutions Architect – Professional – Supplies the technical foundation for designing the serverless PGx Risk Dashboard, including VPC, IAM, and Lambda patterns that implement an ephemeral “compute, respond, discard” architecture consistent with HIPAA’s minimum‑necessary standard.[^4][^1]
- Genomic foundations
    - PHAR 691: Advanced Pharmacogenomics and HGEN 619: Quantitative Genetics – Provide the biological context for recovering pharmacogenomic signals from administrative claims and for embedding CPIC Level A/B evidence into causal feature engineering.[^1][^4]

This area is critical for translating causal rules and PGx insights into a real‑world “causal calculator” and PGx Risk Dashboard that clinicians can use at the point of care.

### Area 4: Geriatrics and Scientific Communication (Orient and Act)

This area supports my transition to an independent investigator capable of communicating complex explainable AI (XAI) outputs to geriatric and multidisciplinary clinical teams.[^1]

- Clinical integration
    - PHAR 691: Geriatrics Research Group – Ongoing participation in interdisciplinary case conferences and research discussions where I will iteratively present causal calculator outputs and PGx Risk Dashboard prototypes to geriatric pharmacists, physicians, and trainees, validating model findings against geriatric prescribing standards.[^4][^1]
- Risk and quality systems
    - Lean Six Sigma Black Belt (Villanova) and DAPT 642: Introduction to Risk Analysis – Provide a framework for managing reliability, safety, and continuous improvement in the clinical software lifecycle, including monitoring for alert fatigue and unintended consequences of decision support.[^1]

These experiences ensure that my work remains clinically grounded and that I can effectively “close the loop” between algorithm development and bedside decision‑making.

### Summary of Fellowship Milestones and Timeline

By the start of the F31 award period, I will have completed GRAD 691 (Dissertation Research) credits and the core PHAR/HGEN sequences, as well as the majority of the claims, ML, and PGx coursework outlined above. During the fellowship, I will:[^1]

- Finalize and validate the Consensus‑Causal Filter and IR scoring framework across opioid and polypharmacy outcomes (Year 1).
- Complete remaining advanced coursework and targeted workshops in causal inference and implementation science, while iteratively testing dashboard prototypes with the Geriatrics Research Group (Years 1–2).[^5][^2]
- Deploy and evaluate the serverless PGx Risk Dashboard, preparing manuscripts and an F31‑aligned dissertation for a planned June 2026 defense (Year 2).[^4][^1]

This integrated training plan leverages my prior military/OODA background, doctoral coursework, and AWS certifications while clearly delineating new skills and experiences needed to become an independent causal AI and pharmacogenomics researcher in medication safety.

Would you like a shorter, 1–2 paragraph “Training Plan overview” version as well, to use in the Candidate section or in emails to your committee when you send them the draft?

<div align="center">⁂</div>

[^1]: F31-Training-Plan-for-Clinical-OODA-Loop-Research.md

[^2]: https://support.ucsd.edu/research?id=kb_article_view\&sysparm_article=KB0034036

[^3]: https://manusights.com/resources/nih-f31-guide

[^4]: Causal-AI-and-Pharmacogenomics-for-Opioid-Risk-Prediction.md

[^5]: https://career.ucsf.edu/sites/g/files/tkssra15591/files/wysiwyg/ResearchersSlidesF30F31.pdf

