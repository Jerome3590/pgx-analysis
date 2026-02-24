# FP-Growth code mapping table

This folder holds **code → description** mappings so FP-Growth visuals (network graph and itemset bar charts) show human-readable labels instead of raw codes.

## File

- **`fpgrowth_code_descriptions.csv`**  
  - Columns: `code`, `description`  
  - `code`: exact string as in the FP-Growth JSON (with prefix).  
  - **CPT:** We use the **first 3 characters** only (category level), e.g. `CPT:992`, `CPT:100`. The mapping table must use the same form (`CPT:992`, not `CPT:99213`).  
  - **ICD / DRUG:** Full code as in data, e.g. `ICD:J069`, `DRUG:oxycodone`.  
  - `description`: short label for display (used in network nodes and itemset bars)

## How it’s used

- When creating plots, the visualization code loads this CSV (if present) and uses it to replace or augment node/label text:
  - **Network graph**: node labels and hover show description when a mapping exists; otherwise the raw code is shown.
  - **Itemset bar charts**: itemset labels can show descriptions for each item when mapped.
- If the file is missing or a code has no row, the raw code is shown (e.g. `CPT:99213`).

## Populating the table

1. **CPT (procedure codes, first 3 chars)**  
   FP-Growth uses the **first 3 characters** of each procedure code (e.g. 99213 → 992, 10004 → 100). Add one row per **3-character prefix** with `code` = `CPT:XXX` (e.g. `CPT:992`, `CPT:100`) and `description` = category label (e.g. "E/M", "Surgery range"). Reference: first digits indicate category (99xxx = E/M, 10004–69990 = Surgery, 70010–79999 = Radiology, etc.).

2. **ICD (diagnosis codes)**  
   Use ICD-10-CM descriptions (e.g. [CDC ICD-10](https://www.cdc.gov/nchs/icd/icd-10-cm.htm) or your reference). `code` = `ICD:<diagnosis_code>`.

3. **Drug names**  
   `code` = `DRUG:<drug_name>` as in your data. `description` can be the same as the name or a shorter/readable form (e.g. drug class or generic name).

You can build the CSV from existing code lists (e.g. export from your EHR or claims code tables) and add or edit rows as needed. Keep descriptions short so they fit in the graph; longer text can go in tooltips if the visualization supports it.

## CPT code dictionary (final feature importance only)

We keep a separate **CPT dictionary** that includes only CPT codes that have **final feature importance** (i.e. appear in the SHAP/FFA allowed-codes used by FP-Growth and BupaR).

- **`cpt_code_dictionary.csv`**  
  - Columns: `cpt_code`, `first_three`, `definition`  
  - `cpt_code`: full normalized CPT code as in allowed codes.  
  - `first_three`: first 3 characters (category level, e.g. 992, 100).  
  - `definition`: human-readable description (fill from AMA/CMS or your reference).

**How to build it**

1. Generate allowed codes (run the risk dashboard pipeline so that `10_risk_dashboard/visualizations/bupar/outputs/allowed_codes_shap_ffa_{cohort}_{age_band}.json` exist).
2. From repo root:
   ```bash
   python 9_dashboard_visuals/fpgrowth/code_mappings/build_cpt_dictionary_from_allowed_codes.py
   ```
   This scans all `allowed_codes_shap_ffa_*.json` files, collects unique CPT codes, and writes/updates `cpt_code_dictionary.csv` with `cpt_code` and `first_three`; `definition` is left blank (or preserved if you already filled it).  
3. Fill the `definition` column from [AMA CPT](https://www.ama-assn.org/practice-management/cpt) or [CMS Physician Fee Schedule](https://www.cms.gov/medicare/physician-fee-schedule/search).

The same 3-character prefixes can be used in `fpgrowth_code_descriptions.csv` as `CPT:992`, `CPT:100`, etc., for display labels in the network/itemset views.

## Location and overrides

- Default path used by the dashboard visuals:  
  `9_dashboard_visuals/fpgrowth/code_mappings/fpgrowth_code_descriptions.csv`  
  (relative to the repo root when running from `create_plots.py` / `py_helpers`).
- Override the path when creating plots:  
  `python 9_dashboard_visuals/fpgrowth/create_plots.py ... --code-mapping /path/to/your_mapping.csv`
