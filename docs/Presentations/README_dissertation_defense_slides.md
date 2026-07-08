# Dissertation Defense Slides — NotebookLM Workflow

## Overview

This document describes the workflow for generating dissertation defense presentation slides using Google NotebookLM. All input artifacts are staged in the `notebookLM-edit-content` project and organized per chapter.

## Input Package Location

```
C:\Projects\notebookLM-edit-content\input\slides\
├── Dissertation_Slides_Combined.md   # Master outline + figure/doc index
├── CH1/                               # Literature Review
├── CH2/                               # Clinical OODA Loop
├── CH3/                               # Opioid ED Cohort
├── CH4/                               # Non-Opioid ED Cohort
├── CH5/                               # Dashboard
└── CH6/                               # Conclusion
```

Each `CH*/` folder is flat (no subdirectories) and contains:

| Artifact Type | Naming Convention | Example |
|---------------|-------------------|---------|
| Outline + prompt | `CH{N}_{title}.md` | `CH3_Opioid_ED_Cohort.md` |
| Manuscript DOCX | `CTS-*_revised_manuscript.docx` | `CTS-2026-0196_revised_manuscript.docx` |
| Figures (PNG) | `fig_*.png` or `pgx_*.png` | `fig_shap.png` |
| SHAP CSVs | `{cohort}_{age}_shap_global_importance_{model}.csv` | `opioid_ed_25_44_shap_global_importance_xgboost.csv` |
| FFA CSVs | `{cohort}_{age}_{bin}_ffa_causal_factors.csv` | `opioid_ed_25-44_low_ffa_causal_factors.csv` |
| Supporting docs | `README_*.md` | `README_shap_analysis.md` |

## Source Repositories

| Repo | Purpose |
|------|---------|
| `pgx-analysis` | Manuscript submodule, figures, docs, SHAP/FFA data |
| `pgx-analysis/manuscript` | Submodule — CTS submission packages, chapter QMDs, rendered DOCXs |
| `notebookLM-edit-content` | Staged input packages for NotebookLM |

## Scripts

### 1. Refresh manuscript submodule

```powershell
# Pull latest manuscript commits
cd C:\Projects\pgx-analysis\manuscript
git fetch origin
git pull origin master

# Update parent repo submodule pointer
cd C:\Projects\pgx-analysis
git add manuscript
git commit -m "Update manuscript submodule pointer"
git push origin main
```

### 2. Stage figures for a chapter

```powershell
# Example: refresh CH3 figures
$CH = "CH3"
$SRC = "C:\Projects\pgx-analysis\manuscript\figures\ch03"
$DST = "C:\Projects\notebookLM-edit-content\input\slides\$CH"

Get-ChildItem "$SRC\*.png" | Copy-Item -Destination $DST -Force
```

### 3. Stage latest CTS manuscript DOCX

```powershell
# Find the latest due_date folder for a given CTS ID
$CTS_ID = "CTS-2026-0196"
$CH = "CH3"
$SRC = Get-ChildItem "C:\Projects\pgx-analysis\manuscript\cts\due_date\*$CTS_ID*\edits\*revised_manuscript.docx" |
       Sort-Object LastWriteTime -Descending | Select-Object -First 1
$DST = "C:\Projects\notebookLM-edit-content\input\slides\$CH"

# Remove old manuscript DOCX
Get-ChildItem "$DST\*revised_manuscript*.docx" | Remove-Item -Force
Copy-Item $SRC.FullName -Destination $DST -Force
Write-Host "Copied: $($SRC.Name) -> $DST"
```

### 4. Stage SHAP global importance CSVs

```powershell
$COHORT = "opioid_ed"     # or "non_opioid_ed"
$CH = "CH3"               # or "CH4"
$DST = "C:\Projects\notebookLM-edit-content\input\slides\$CH"
$AGE_BANDS = @("25-44", "45-54", "55-64", "65-74")

foreach ($ab in $AGE_BANDS) {
    $src = "C:\Projects\pgx-analysis\7_shap_analysis\outputs\$COHORT\$ab"
    Get-ChildItem "$src\*shap_global_importance*.csv" | Copy-Item -Destination $DST -Force
}
```

### 5. Stage FFA causal factors CSVs (with descriptive names)

```powershell
$COHORT = "opioid_ed"     # or "non_opioid_ed"
$CH = "CH3"               # or "CH4"
$DST = "C:\Projects\notebookLM-edit-content\input\slides\$CH"
$AGE_BANDS = @("25-44", "45-54", "55-64", "65-74")
$BINS = @("low", "medium", "high", "extreme")

foreach ($ab in $AGE_BANDS) {
    foreach ($bin in $BINS) {
        $src = "C:\Projects\pgx-analysis\8_ffa_analysis\outputs\$COHORT\$ab\bin_models\$bin\ffa_causal_factors.csv"
        if (Test-Path $src) {
            $dest_name = "${COHORT}_${ab}_${bin}_ffa_causal_factors.csv"
            Copy-Item $src -Destination "$DST\$dest_name" -Force
        }
    }
}
```

### 6. Stage supporting README docs

```powershell
$CH = "CH3"
$DST = "C:\Projects\notebookLM-edit-content\input\slides\$CH"
$DOCS = @(
    "docs\Step1-2_DataPipeline\README_create_cohort.md",
    "docs\Step6_FinalModel\README_final_model.md",
    "docs\Step7_SHAP\README_shap_analysis.md",
    "docs\Step8_FFA\README_ffa_analysis.md"
    # Add more as needed per chapter
)

foreach ($doc in $DOCS) {
    $src = "C:\Projects\pgx-analysis\$doc"
    if (Test-Path $src) {
        Copy-Item $src -Destination $DST -Force
    }
}
```

### 7. Full chapter refresh (combines all steps)

```powershell
param(
    [string]$Chapter = "CH3",
    [string]$Cohort = "opioid_ed",
    [string]$CtsId = "CTS-2026-0196",
    [string]$FigDir = "ch03"
)

$DST = "C:\Projects\notebookLM-edit-content\input\slides\$Chapter"
$PGX = "C:\Projects\pgx-analysis"

# Figures
Get-ChildItem "$PGX\manuscript\figures\$FigDir\*.png" | Copy-Item -Destination $DST -Force

# Latest CTS manuscript
$ms = Get-ChildItem "$PGX\manuscript\cts\due_date\*$CtsId*\edits\*revised_manuscript.docx" |
      Sort-Object LastWriteTime -Descending | Select-Object -First 1
if ($ms) {
    Get-ChildItem "$DST\*revised_manuscript*.docx" | Remove-Item -Force
    Copy-Item $ms.FullName -Destination $DST -Force
}

# SHAP + FFA data
$AGE_BANDS = @("25-44", "45-54", "55-64", "65-74")
foreach ($ab in $AGE_BANDS) {
    Get-ChildItem "$PGX\7_shap_analysis\outputs\$Cohort\$ab\*shap_global_importance*.csv" |
        Copy-Item -Destination $DST -Force
    foreach ($bin in @("low", "medium", "high", "extreme")) {
        $ffa = "$PGX\8_ffa_analysis\outputs\$Cohort\$ab\bin_models\$bin\ffa_causal_factors.csv"
        if (Test-Path $ffa) {
            Copy-Item $ffa -Destination "$DST\${Cohort}_${ab}_${bin}_ffa_causal_factors.csv" -Force
        }
    }
}

Write-Host "$Chapter refreshed at $DST"
```

### 8. Commit and push staged slides

```powershell
cd C:\Projects\notebookLM-edit-content
git add input/slides/
git commit -m "Refresh dissertation slide input packages"
git push origin main
```

## NotebookLM Workflow

1. **Open** Google NotebookLM and create a new notebook per chapter (or one combined)
2. **Upload** all files from the chapter's `CH*/` folder as sources
3. **Paste** the NotebookLM Slide Generation Prompt from the chapter's `CH{N}_*.md` outline
4. **Generate** the slide deck
5. **Review** and iterate — adjust the prompt as needed
6. **Export** to Google Slides for final formatting

## Chapter → CTS Manuscript Mapping

| Chapter | CTS ID | Due Date | Status |
|---------|--------|----------|--------|
| CH1 | CTS-2026-0197 | Jul 10 | Revision |
| CH2 | CTS-2026-0230-T | — | Under Consideration |
| CH3 | CTS-2026-0196R1 | Jul 13 | R1 Revision |
| CH4 | CTS-2026-0235-T | Jun 29 | Revision |
| CH5 | CTS-2026-0255-T | Jun 29 | Revision |
| CH6 | — | — | Dissertation only |
