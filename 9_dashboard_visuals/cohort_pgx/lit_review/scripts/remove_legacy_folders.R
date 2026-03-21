# Script to remove legacy folders after reorganization
# This removes the old Chapter1_* directories and other moved folders

library(here)
library(fs)

# List of legacy directories to remove
legacy_dirs <- c(
  "Chapter1_BlackBox_CDS",
  "Chapter1_Interpretability",
  "Chapter1_OpioidDisorder",
  "Chapter1_Polypharmacy",
  "Chapter1_DrugInteractions",
  "Chapter1_Pharmacovigilance",
  "Chapter1_APCD_Analysis",
  "Chapter1_FPGrowth",
  "Chapter1_ProcessMining",
  "Chapter1_DTW",
  "Chapter1_TemporalCausality",
  "Chapter1_TargetLeakage",
  "Chapter1_CatBoost",
  "Chapter1_DuckDB",
  "PGx_Classification_Models",
  "Risk_Models_EHR",
  "Risk_Models_FHIR_EHR"
)

# List of legacy files to remove (if they exist at root)
legacy_files <- c(
  "run_chapter1_searches.R",
  "run_additional_searches.R",
  "run_apcd_broad_search.R",
  "run_apcd_search.R",
  "run_broader_additional_searches.R",
  "run_pharmacovigilance_broad_search.R",
  "install_packages.R",
  "set_r_library_env.ps1",
  "change_r_library_permissions.ps1"
)

cat("=== Removing Legacy Folders ===\n\n")

# Remove directories
for (dir in legacy_dirs) {
  dir_path <- here(dir)
  if (dir_exists(dir_path)) {
    dir_delete(dir_path)
    cat("Removed directory:", dir, "\n")
  } else {
    cat("Directory not found (already removed?):", dir, "\n")
  }
}

cat("\n=== Removing Legacy Files ===\n\n")

# Remove files
for (file in legacy_files) {
  file_path <- here(file)
  if (file_exists(file_path)) {
    file_delete(file_path)
    cat("Removed file:", file, "\n")
  } else {
    cat("File not found (already moved?):", file, "\n")
  }
}

cat("\n=== Cleanup Complete ===\n")
cat("Legacy folders and files have been removed.\n")
cat("All content has been moved to the new organized structure.\n")
