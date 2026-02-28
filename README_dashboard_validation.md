# Dashboard validation

Use this README to validate changes to the dashboard frontend (`10_risk_dashboard/frontend/index.html`) and to Step 6 (S3 sync) so that **tabs**, **visual headings**, **S3 path-style URLs**, and **API usage** stay aligned with the **manifest** and **path mapping**.

**Referenced by:** [10_risk_dashboard/frontend/README.md](10_risk_dashboard/frontend/README.md).

---

## Mapping overview

| Layer | Source of truth | Purpose |
|-------|-----------------|--------|
| **Manifest** | `10_risk_dashboard/visualizations/dashboard_visual_objects.json` | **Single source of truth for all data visual requirements:** `metadata_files` (Documentation/dropdowns) and `visual_objects[]` (per-tab `s3_path` + `static_files`). Frontend loads manifest first and builds static URLs from it. |
| **Path mapping** | [10_risk_dashboard/docs/README_dashboard_visual_artifact_paths.md](10_risk_dashboard/docs/README_dashboard_visual_artifact_paths.md) | Tab & visual (heading) → data artifact → EC2 path → S3 object key (path-style). |
| **S3** | Bucket + prefix (e.g. `vcu/pgx-risk-calculator/`) | After Step 6, objects under the prefix must match what the manifest and frontend expect. |
| **Frontend** | `10_risk_dashboard/frontend/index.html` | Tabs, section IDs, visual headings, and URLs must align with manifest and path mapping; all asset URLs use **path-style** S3. |

**Path-style URL only:**  
`https://s3.{region}.amazonaws.com/{bucket}/{prefix}/{object_key}`  
Do not use virtual-hosted style. See [README_dashboard_visual_artifact_paths.md](10_risk_dashboard/docs/README_dashboard_visual_artifact_paths.md) for full path template and examples.

**Age bands:** EC2/local use **underscore** (e.g. `25_44`); S3 keys use **hyphen** (e.g. `25-44`).

---

## Manifest-first and S3 sync (Step 6)

- **Manifest first:** In `5_build_and_deploy.ipynb` Step 6, the manifest (`dashboard_visual_objects.json`) is uploaded to S3 **immediately after** the frontend sync, before BupaR/DTW/FP-Growth sync. The frontend can then load the manifest same-origin and resolve static paths.
- **S3 matches manifest:** Sync excludes non-manifest files so S3 does not contain cruft the frontend never uses:
  - **DTW:** `--exclude "*.csv"`, `--exclude "*/.ipynb_checkpoints/*"`, `--exclude "*checkpoint*"`
  - **BupaR:** `--exclude "Rplots.pdf"`
  - **FP-Growth:** `--exclude "*/.ipynb_checkpoints/*"`, `--exclude "*checkpoint*"`

After deploy, verify S3 against the manifest using [10_risk_dashboard/docs/S3_VERIFICATION_REPORT.md](10_risk_dashboard/docs/S3_VERIFICATION_REPORT.md) (or by comparing `aws s3 ls` output to `dashboard_visual_objects.json` and the path mapping doc).

---

## Validation checklist (when changing `index.html` or deploying)

- [ ] **Tabs:** Tab order and IDs match the manifest and [DASHBOARD_TABS.md](10_risk_dashboard/docs/DASHBOARD_TABS.md) (if present).
- [ ] **Visual headings:** Section headings for Feature Importance, Causal, BupaR, DTW, FP-Growth, PGx Cohort match the path mapping doc and research-question artifacts ([RESEARCH_QUESTIONS_ARTIFACTS.md](10_risk_dashboard/docs/RESEARCH_QUESTIONS_ARTIFACTS.md)).
- [ ] **BupaR copy:** Pre-target labels use `pre_f1120` (opioid_ed) or `pre_hcg` (non_opioid_ed) where the path mapping expects them.
- [ ] **S3 URLs:** All visualization asset URLs are **path-style** (`https://s3.{region}.amazonaws.com/{bucket}/{prefix}/...`). No virtual-hosted style. Base URL / prefix come from config or manifest.
- [ ] **API:** Risk, metadata, and visualization fallback use the same `API_BASE` and endpoints as in [backend/README.md](10_risk_dashboard/backend/README.md).
- [ ] **Manifest:** After Step 6, `visualizations/dashboard_visual_objects.json` is present on S3 and its `static_files` and `s3_path` entries match what the frontend requests.
- [ ] **JSON-first panels:** Any panel with a JSON artifact in the manifest uses it first (Plotly or data), PNG/image fallback only. Check: Feature Importance (JSON then PNG), FP-Growth Top Itemsets and Itemset Support Distribution (drug_name_itemsets.json then PNG), BupaR per manifest notes.
- [ ] **Sync exclusions:** Step 6 does not upload `.ipynb_checkpoints`, `*checkpoint*`, or `Rplots.pdf`; see “Manifest-first and S3 sync” above.

---

## Related docs

| Doc | Purpose |
|-----|--------|
| [10_risk_dashboard/frontend/README.md](10_risk_dashboard/frontend/README.md) | **Artifact usage:** Manifest-driven URLs; JSON + Plotly first for BupaR (Trace Explorer, Trace Explorer Pre-Target, Process Matrix, etc.); DTW/FP-Growth/Causal static-first. |
| [README_dashboard_visual_artifact_paths.md](10_risk_dashboard/docs/README_dashboard_visual_artifact_paths.md) | Full mapping: tab & visual → artifact → EC2 path → S3 key. |
| [S3_VERIFICATION_REPORT.md](10_risk_dashboard/docs/S3_VERIFICATION_REPORT.md) | Check S3 keys against manifest and frontend expectations. |
| [RESEARCH_QUESTIONS_ARTIFACTS.md](10_risk_dashboard/docs/RESEARCH_QUESTIONS_ARTIFACTS.md) | RQ artifact allowlists (e.g. BupaR filenames). |
| [README_validate_frontend_updates.md](10_risk_dashboard/docs/README_validate_frontend_updates.md) | Redirects here (project root). |
