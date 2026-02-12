# Cohort DTW Analysis: Clinical Pathway Discovery

This README describes the **DTW data visualization pipeline** and a structured framework for analyzing 20,000+ patient insurance claim journeys (ICD, CPT, and Drug codes) using Dynamic Time Warping (DTW).

---

## Pipeline in This Repo: Filter by Feature Importance, Then DTW

We **filter the dataset by model feature importances first**, then build and analyze **DTW cohort trajectories** on that filtered set.

1. **Filter by model feature importances**  
   Trajectories are built only from events whose codes appear in the **SHAP/FFA important codes** set (same source as BupaR and FP-Growth). We use `get_shap_ffa_allowed_codes_combined()` from the repo; when available, only ICD, CPT, and Drug codes that are feature-important are kept. Filter is applied in SQL with **OR semantics**: keep an event if its drug **or** any ICD **or** procedure code is in the allowed set (normalized for dots/dashes). If no SHAP/FFA set is available, we fall back to all events.

2. **Build cohort trajectories**  
   For each patient we build a time-ordered sequence of activity codes (e.g. `ICD:E11.9`, `DRUG:Metformin`, `CPT:99213`) from model_events (Step 4), using cutoff dates for target patients to avoid leakage.

3. **Apply DTW**  
   We compute DTW distances to prototype trajectories and optional sequence/time-window features. Outputs are used for dashboard visuals (cluster plots, trajectory archetypes) and are **not** merged back into model training data.

So: **feature-importance filter → trajectory construction → DTW (and optional barycenter/archetype reporting)**.

---

## Target-Event Pathway Analysis (Both Cohorts)

Our DTW logic is **target-aligned**: sequences are anchored at the **target event** so we analyze the clinical "on-ramp" leading to that outcome.

### Anchor and lookback (both cohorts)

- **Opioid-ED cohort:** Anchor = first occurrence of **ICD-10 F11.20 (Opioid Dependence)** (`first_opioid_ed_date`). We use the 24 months *prior* to that date (configurable) so trajectories are the pre-diagnosis pathway.
- **Non–opioid-ED (polypharmacy) cohort:** Anchor = first **ED non-opioid** event (`first_ed_non_opioid_date`). Same idea: lookback window of events *before* that anchor.

All sequences use a **target-relative timeline**: events are ordered chronologically and restricted to **before** the anchor (no leakage). An optional **max lookback** (e.g. 24 months) limits how far back we go so pathways are comparable and focused on the near-term on-ramp.

### On-ramp archetypes

We apply clustering (e.g. TimeSeriesKMeans with DTW + Sakoe-Chiba) to these lookback sequences to find **4–5 primary pathways** to the target event. Example archetypes:

1. **Acute traumatic pathway** – Short, intense sequences (surgery CPTs, high-dose initial Rx).
2. **Chronic pain drift** – Long sequences, multiple providers, low-potency drug rotations over years.
3. **Failed conservative care** – Early PT (e.g. CPT 97110) then escalating pharmacy claims.

### Core metrics for risk stratification

For each patient we compute:

- **Path velocity (warp factor):** How quickly the patient moves along the on-ramp relative to the cluster average.
- **Distance to archetype:** How closely their history matches the cluster’s consensus journey. *High match ⇒ high predictive risk* for that pathway.

### Visualization and business use

- **Barycenter alignment plot:** Average journey per cluster, with ribbons for CPT (procedures), ICD (comorbidities), and Drug codes converging on the target event.
- **Early intervention:** Flag patients whose sequences are aligning with a high-risk archetype *before* the target event.
- **Policy:** Identify which CPT/sequence patterns are most associated with “rapid progression” clusters.

### Technical (target-event config)

- Sakoe-Chiba radius (e.g. **6**) = max ~6-month “flex” in event timing when aligning.
- **24-month lookback** is the default for target patients (configurable); controls use all events before reference or the same window depending on design.

---

## Project Overview

This analysis transforms high-dimensional, irregularly spaced insurance claim sequences into interpretable **Archetypal Patient Journeys**. By using **Dynamic Time Warping (DTW)**, we align patients who follow similar clinical paths even if their treatment speeds or encounter frequencies differ.

---

## 1. Data Preparation & Encoding

To perform math on categorical codes, we must convert them into a numeric or vector space.

- **Embedding Layer:** Convert ICD-10, CPT, and NDC codes into vectors.
  - *Simple:* Integer mapping (e.g., `I10` → `101`).
  - *Advanced:* **Medical Embeddings** (e.g., Word2Vec trained on claims) so that "Hypertension" and "Heart Failure" are mathematically closer than "Hypertension" and "Broken Arm."
- **Sequence Construction:** Create a time-ordered list of codes for each `Member_ID`.
- **Sparsity Handling:** Aggregate claims into **Time Buckets** (e.g., 1-week or 1-month intervals) to reduce noise.

---

## 2. Dimensionality Reduction (The 20k Problem)

Processing 20,000 × 20,000 (400 million) DTW combinations is computationally expensive. We use a **Representative Sampling** strategy:

1. **Pilot Sample:** Select a random, stratified sample of 2,000 patients.
2. **Distance Matrix:** Calculate the DTW distance matrix using a **Sakoe-Chiba Window** so we only align events within a reasonable clinical timeframe (e.g., 6 months).
3. **Clustering:** Apply **K-Medoids** or **TimeSeriesKMeans** to identify 5–10 distinct patient clusters (e.g., "The Rapid Treatment Group" vs. "The Chronic Maintenance Group").

---

## 3. Barycenter Averaging (Consensus Paths)

For each cluster, we calculate the **DTW Barycenter Average (DBA)**. This creates a "Master Sequence" that represents the central tendency of that entire group.

- **The "Thick" Path:** High variance in the cluster; treatments are fragmented.
- **The "Thin" Path:** High standardization; most patients follow the same protocol.

---

## 4. Inverse Mapping & Visualization

To make the output readable for clinical stakeholders, we translate the mathematical Barycenters back into medical terminology.

### The Clinical "Archetype" Table

| Cluster | Phase 1 (Initial) | Phase 2 (Mid-Treatment) | Phase 3 (Outcome) |
|---------|-------------------|-------------------------|-------------------|
| **Cluster 1** | ICD: I10 (Hyper) | CPT: 99213 (Visit) | NDC: Lisinopril |
| **Cluster 2** | ICD: E11 (Diabetes) | CPT: 82947 (Glucose) | NDC: Metformin |

### Visualizing the Warping

We use **Alignment Link Plots** to show how an individual patient "drifts" from the Cluster Average.

- **Lagging:** The patient is receiving care slower than the archetype.
- **Leading:** The patient is progressing through the protocol faster than expected.

---

## 5. Implementation Roadmap

1. **Pre-process:** Clean claims and map to numeric IDs/Embeddings.
2. **Compute:** Use `tslearn` or `dtaidistance` in Python to run `TimeSeriesKMeans` (or hierarchical clustering on a precomputed DTW distance matrix).
3. **Synthesize:** Extract the Barycenter for the top 8 clusters (e.g. `tslearn.barycenters.dtw_barycenter_averaging`).
4. **Report:** Generate a dashboard showing the 8 "Master Paths" and the percentage of the 20,000 patients belonging to each.

---

## 6. Code Example: tslearn TimeSeriesKMeans with Sakoe-Chiba

Use a **warping window** (e.g. 30 days or 3 steps) so a 2022 diagnosis does not align with a 2025 procedure. Resample sequences to a fixed length so the distance matrix stays manageable, then cluster with DTW + Sakoe-Chiba and read off the archetypes (cluster centers).

```python
from tslearn.clustering import TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesResampler
import numpy as np

# 1. Parameterize the warping window
# For insurance claims, a 30-day or 3-step window is usually clinically sound.
WINDOW_SIZE = 3

# 2. Data resampling (critical for speed)
# cohort_data: shape (n_patients, max_len) or (n_patients, max_len, n_features)
# Resample to fixed length (e.g., 20 steps).
X_train = TimeSeriesResampler(sz=20).fit_transform(cohort_data)

# 3. Clustering with DTW + Sakoe-Chiba
# metric_params applies the Sakoe-Chiba band; n_jobs=-1 uses all cores.
model = TimeSeriesKMeans(
    n_clusters=8,
    metric="dtw",
    metric_params={"global_constraint": "sakoe_chiba", "sakoe_chiba_radius": WINDOW_SIZE},
    max_iter=10,
    n_jobs=-1,
    random_state=42,
)
labels = model.fit_predict(X_train)

# 4. Archetypes = average journey per cluster
archetypes = model.cluster_centers_
```

Then map each archetype back to discrete codes with `barycenter_reporting.map_barycenter_to_codes()` or `mode_based_consensus_table()` for categorical sequences.

### Why this works for 20k patients

- **Radius constraint:** `sakoe_chiba_radius` tells the algorithm: if two events are more than **r** steps apart, do not align them. That turns an O(n²) search space into a narrow band of width ~2r, often cutting DTW cost by **60–80%**.
- **Parallelization:** `n_jobs=-1` uses all CPU cores for distance computation.
- **Resampling:** Claim sequences are ragged (e.g. 5 vs 500 events per patient). Resampling to a fixed length (e.g. 20 steps) gives a uniform tensor and faster, vectorized operations.

### Outlier metric for reporting

After you have archetypes (cluster centers), score each patient by **distance to centroid** or **Silhouette Score**:

- **Low distance:** Patient is a "standard case" (follows the consensus path).
- **High distance:** Patient is "non-conformant." In insurance terms, these often correspond to **unnecessary spend** or **care gaps** (drifting from the established clinical pathway) and are candidates for case management.

Use these scores to flag high-distance patients for review or to report "% of cohort within X of their cluster center."

---

## Related Scripts in This Repo

- **`create_dtw_features.py`** – **Filters events by SHAP/FFA feature importances**, then builds per-patient trajectories (ICD/CPT/Drug) and DTW distances to prototypes; uses DuckDB for trajectory aggregation. Same code set as BupaR/FP-Growth.
- **`create_dtw_visuals.py`** – Publishes DTW outputs (CSV, plots) to S3 and dashboard bucket; writes pipeline checkpoint `9_dashboard_visuals`.
- **`barycenter_reporting.py`** – Inverse mapping: numeric barycenters or mode-based consensus → clinical journey table (Step, Code, Description, Category).
- **`create_dtw_plots.py`** – Cluster plots (1D/3D) from DTW features for dashboard.

---

## References

- Sakoe-Chiba band: limits warping to a band around the diagonal; use `window` in `dtaidistance.dtw` or `dtwalign`.
- DBA: `tslearn.barycenters.dtw_barycenter_averaging`.
- For 20k patients: sample 2k → distance matrix with `window` → cluster → assign remaining 18k to nearest center; then compute barycenters per cluster.
