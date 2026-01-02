L1:# Step 4: Model Data, DTW Protocol Filter, and Extreme-Density Split
L2:
L3:This folder documents the Step 4 pipeline stages that prepare cohort event data for downstream feature engineering and final modeling.
L4:
L5:## Sub-Steps
L6:
L7:- **4a – Model Data Extraction (`4a_model_data/`)**  
L8:  - Creates compact, model-ready `model_events.parquet` datasets for each `(cohort, age_band)` based on MC‑CV feature importance.  
L9:  - Writes paired target (`opioid_ed`) and control (`non_opioid_ed`) cohorts under:  
L10:    - `4a_model_data/cohort_name={cohort}/age_band={band}/model_events.parquet`.
L11:
L12:- **4b – DTW-Based Protocol Filtering (`4b_dtw_filter/`)**  
L13:  - Removes protocol-like or repetitive time-windowed events to reduce noise before feature engineering.  
L14:  - Produces `model_events_no_protocols.parquet` alongside the base `model_events.parquet` where applicable.
L15:
L16:- **4c – Extreme-Density Cohort Split (`5b_fpgrowth_analysis/extract_extreme_density_cohort.py`)**  
L17:  - Uses the same transaction-density logic as FP-Growth to identify **extreme** patients with very dense medical histories.  
L18:  - For each `(cohort, age_band)`:
L19:    - Writes an extreme-only cohort to  
L20:      `4a_model_data/cohort_name={cohort}_extreme_density/age_band={band}/model_events.parquet`.  
L21:    - Rewrites the base `model_events.parquet` in `4a_model_data/cohort_name={cohort}/age_band={band}/` with extreme patients removed (backing up the original as `model_events_with_extreme.parquet`).  
L22:  - Ensures all **main models and feature engineering steps** run on the non-extreme base cohorts, while `_extreme_density` cohorts are available for exploratory FP-Growth, BupaR, DTW, and process mining.
L23:
L24:## Purpose in the Workflow
L25:
L26:Step 4 is the bridge between raw gold cohorts and structured feature engineering:
L27:
L28:- Enforces a consistent, size-controlled unit of analysis (`model_events.parquet`).  
L29:- Cleans protocol-like sequences that can distort pattern mining and trajectory analysis.  
L30:- Splits off extreme-density patients so they do not dominate the main models, while preserving them for dedicated “extreme” analyses and transition-risk markers.
L31:
L32:## Related Documentation
L33:
L34:- `docs/README_data_pipeline_architecture.md` – Upstream data pipeline and partition-first architecture.  
L35:- `docs/README_analysis_workflow.md` – Detailed Step 4c description and extreme-density sub-pipeline.  
L36:- `docs/README_overview.md` – High-level repository and workflow overview.  
L37:
