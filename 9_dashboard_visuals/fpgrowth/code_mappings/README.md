# FP-Growth code mapping table

This folder holds **code → description** mappings so FP-Growth visuals (network graph and itemset bar charts) show human-readable labels instead of raw codes.

## File

- **`fpgrowth_code_descriptions.csv`**  
  - Columns: `code`, `description`  
  - `code`: exact string as in the FP-Growth JSON (with prefix), e.g. `CPT:99213`, `ICD:J069`, `DRUG:oxycodone`  
  - `description`: short label for display (used in network nodes and itemset bars)

## How it’s used

- When creating plots, the visualization code loads this CSV (if present) and uses it to replace or augment node/label text:
  - **Network graph**: node labels and hover show description when a mapping exists; otherwise the raw code is shown.
  - **Itemset bar charts**: itemset labels can show descriptions for each item when mapped.
- If the file is missing or a code has no row, the raw code is shown (e.g. `CPT:99213`).

## Populating the table

1. **CPT (procedure codes)**  
   Use AMA/CMS descriptors (e.g. [CMS PFS](https://www.cms.gov/medicare/physician-fee-schedule/search), or your source). Add one row per code with `code` = `CPT:<procedure_code>` and `description` = short procedure name.

2. **ICD (diagnosis codes)**  
   Use ICD-10-CM descriptions (e.g. [CDC ICD-10](https://www.cdc.gov/nchs/icd/icd-10-cm.htm) or your reference). `code` = `ICD:<diagnosis_code>`.

3. **Drug names**  
   `code` = `DRUG:<drug_name>` as in your data. `description` can be the same as the name or a shorter/readable form (e.g. drug class or generic name).

You can build the CSV from existing code lists (e.g. export from your EHR or claims code tables) and add or edit rows as needed. Keep descriptions short so they fit in the graph; longer text can go in tooltips if the visualization supports it.

## Location and overrides

- Default path used by the dashboard visuals:  
  `9_dashboard_visuals/fpgrowth/code_mappings/fpgrowth_code_descriptions.csv`  
  (relative to the repo root when running from `create_plots.py` / `py_helpers`).
- Override the path when creating plots:  
  `python 9_dashboard_visuals/fpgrowth/create_plots.py ... --code-mapping /path/to/your_mapping.csv`
