#!/usr/bin/env Rscript
# Wrapper script to create visualizations for feature importance analysis

# Get project root (two levels up from this script)
script_dir <- dirname(normalizePath(sys.frame(1)$ofile))
project_root <- dirname(dirname(script_dir))
source(file.path(project_root, "r_helpers", "create_visualizations.R"))

# Get command line arguments or use defaults
args <- commandArgs(trailingOnly = TRUE)

if (length(args) >= 1) {
  aggregated_file <- args[1]
} else {
  aggregated_file <- "outputs/opioid_ed_0_12_aggregated_feature_importance.csv"
}

if (length(args) >= 2) {
  output_dir <- args[2]
} else {
  output_dir <- "outputs"
}

if (length(args) >= 3) {
  cohort_name <- args[3]
} else {
  cohort_name <- "opioid_ed"
}

if (length(args) >= 4) {
  age_band <- args[4]
} else {
  age_band <- "0-12"
}

if (length(args) >= 5) {
  event_year <- as.integer(args[5])
} else {
  event_year <- 2019
}

cat(sprintf("Creating visualizations:\n"))
cat(sprintf("  File: %s\n", aggregated_file))
cat(sprintf("  Output: %s\n", output_dir))
cat(sprintf("  Cohort: %s, Age Band: %s, Year: %d\n\n", cohort_name, age_band, event_year))

create_feature_importance_plots(
  aggregated_file = aggregated_file,
  output_dir = output_dir,
  s3_upload = FALSE,
  cohort_name = cohort_name,
  age_band = age_band,
  event_year = event_year
)

