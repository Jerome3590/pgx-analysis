#!/usr/bin/env Rscript
#
# Build bupaR event logs for cohort 1 (opioid_ed), age 0–12 using local model_data
# and FP-Growth TRAIN target-only itemsets.
#
# This script is a local, script-based counterpart to the first part of
# `5_bupaR_analysis/bupaR_pipeline.ipynb` and is intended for quick testing
# on a workstation before running the full notebook on EC2.
#

suppressPackageStartupMessages({
  library(duckdb)
  library(arrow)
  library(dplyr)
  library(tidyr)
  library(jsonlite)
  library(readr)
  library(bupaR)
})

# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------

# Assume we run this from the project root (e.g., C:/Projects/pgx-analysis)
project_root <- getwd()

# Optional command-line arguments to configure cohort / age band / control / target ICDs
# Usage (examples):
#   Rscript build_bupar_eventlogs_opioid_ed_0_12.R opioid_ed 0-12 non_opioid_ed F1120
#   Rscript build_bupar_eventlogs_opioid_ed_0_12.R non_opioid_ed 65-74 opioid_ed HCG

args <- commandArgs(trailingOnly = TRUE)

cohort_name  <- if (length(args) >= 1) args[[1]] else "opioid_ed"
age_band     <- if (length(args) >= 2) args[[2]] else "0-12"
control_cohort_name <- if (length(args) >= 3) args[[3]] else if (cohort_name == "opioid_ed") "non_opioid_ed" else "opioid_ed"

# Comma-separated list of target ICD codes/patterns to always include in the alphabet
target_icd_arg <- if (length(args) >= 4) args[[4]] else "F1120"
target_icd_codes <- if (nzchar(target_icd_arg)) strsplit(target_icd_arg, ",")[[1]] else character(0)

age_band_fname <- gsub("-", "_", age_band)
train_years  <- c(2016L, 2017L, 2018L)

cat("=== bupaR Event Log Builder ===\n")
cat("  Cohort:          ", cohort_name, "\n", sep = "")
cat("  Age band:        ", age_band, "\n", sep = "")
cat("  Control cohort:  ", control_cohort_name, "\n", sep = "")
if (length(target_icd_codes) > 0) {
  cat("  Target ICD codes:", paste(target_icd_codes, collapse = ", "), "\n", sep = " ")
}
cat("\n")

model_data_path <- file.path(
  project_root,
  "model_data",
  paste0("cohort_name=", cohort_name),
  paste0("age_band=", age_band),
  "model_events.parquet"
)

fpgrowth_root <- file.path(
  project_root,
  "4_fpgrowth_analysis",
  "outputs",
  cohort_name
)

target_dir_train <- file.path(
  fpgrowth_root,
  "target",
  age_band_fname,
  "train"
)

itemsets_drug_target_path    <- file.path(target_dir_train, "drug_name_itemsets_target_only.json")
itemsets_icd_target_path     <- file.path(target_dir_train, "icd_code_itemsets_target_only.json")
itemsets_medical_target_path <- file.path(target_dir_train, "medical_code_itemsets_target_only.json")

cat("Project root:         ", project_root, "\n", sep = "")
cat("Model data path:      ", model_data_path, "\n", sep = "")
cat("FP-Growth target dir: ", target_dir_train, "\n\n", sep = "")

if (!file.exists(model_data_path)) {
  stop("model_data parquet not found at: ", model_data_path,
       "\nRun 3_feature_importance/create_model_data.py for this cohort/age band first.")
}

# -------------------------------------------------------------------
# Load model_data for train years
# -------------------------------------------------------------------

con <- dbConnect(duckdb::duckdb())

query <- sprintf(
  "SELECT * FROM read_parquet('%s') WHERE event_year IN (%s)",
  model_data_path,
  paste(train_years, collapse = ",")
)

pgx_df <- dbGetQuery(con, query)

cat("Loaded ", nrow(pgx_df), " events for ", cohort_name,
    " age_band=", age_band,
    " across years ", paste(train_years, collapse = ","), "\n", sep = "")

# -------------------------------------------------------------------
# Build target-only subset and allowed codes from FP-Growth itemsets
# -------------------------------------------------------------------

pgx_df_target1 <- pgx_df %>%
  filter(target == 1L)

cat("Target=1 rows: ", nrow(pgx_df_target1), "\n", sep = "")

allowed_codes <- character(0)

if (file.exists(itemsets_drug_target_path)) {
  drug_itemsets_target <- fromJSON(itemsets_drug_target_path, simplifyDataFrame = TRUE)
  drug_codes <- unique(unlist(drug_itemsets_target$itemsets))
  allowed_codes <- union(allowed_codes, drug_codes)
  cat("Loaded ", length(drug_codes), " unique drug codes from target-only itemsets.\n", sep = "")
} else {
  warning("Drug target-only itemsets not found at ", itemsets_drug_target_path)
}

if (file.exists(itemsets_icd_target_path)) {
  icd_itemsets_target <- fromJSON(itemsets_icd_target_path, simplifyDataFrame = TRUE)
  icd_codes <- unique(unlist(icd_itemsets_target$itemsets))
  allowed_codes <- union(allowed_codes, icd_codes)
  cat("Loaded ", length(icd_codes), " unique ICD codes from target-only itemsets.\n", sep = "")
} else {
  warning("ICD target-only itemsets not found at ", itemsets_icd_target_path)
}

if (file.exists(itemsets_medical_target_path)) {
  medical_itemsets_target <- fromJSON(itemsets_medical_target_path, simplifyDataFrame = TRUE)
  medical_codes <- unique(unlist(medical_itemsets_target$itemsets))
  allowed_codes <- union(allowed_codes, medical_codes)
  cat("Loaded ", length(medical_codes), " unique medical (ICD+CPT) codes from target-only itemsets.\n", sep = "")
} else {
  warning("Medical target-only itemsets not found at ", itemsets_medical_target_path)
}

# Always ensure configured target ICD codes are included in the activity alphabet
if (length(target_icd_codes) > 0) {
  allowed_codes <- union(allowed_codes, target_icd_codes)
}

cat("Total unique allowed codes from FP-Growth itemsets (incl. target ICDs, if any): ",
    length(allowed_codes), "\n\n", sep = "")

# -------------------------------------------------------------------
# Build target-only eventlog (DRUG + ICD(1–10) + CPT)
# -------------------------------------------------------------------

pgx_df_target1_long <- pgx_df_target1 %>%
  transmute(
    mi_person_key,
    event_date,
    drug_name,
    primary_icd_diagnosis_code,
    two_icd_diagnosis_code,
    three_icd_diagnosis_code,
    four_icd_diagnosis_code,
    five_icd_diagnosis_code,
    six_icd_diagnosis_code,
    seven_icd_diagnosis_code,
    eight_icd_diagnosis_code,
    nine_icd_diagnosis_code,
    ten_icd_diagnosis_code,
    procedure_code
  ) %>%
  # Ensure all pivoted columns are character to avoid type mixing
  mutate(across(
    c(
      drug_name,
      primary_icd_diagnosis_code,
      two_icd_diagnosis_code,
      three_icd_diagnosis_code,
      four_icd_diagnosis_code,
      five_icd_diagnosis_code,
      six_icd_diagnosis_code,
      seven_icd_diagnosis_code,
      eight_icd_diagnosis_code,
      nine_icd_diagnosis_code,
      ten_icd_diagnosis_code,
      procedure_code
    ),
    as.character
  )) %>%
  pivot_longer(
    cols = c(
      drug_name,
      primary_icd_diagnosis_code,
      two_icd_diagnosis_code,
      three_icd_diagnosis_code,
      four_icd_diagnosis_code,
      five_icd_diagnosis_code,
      six_icd_diagnosis_code,
      seven_icd_diagnosis_code,
      eight_icd_diagnosis_code,
      nine_icd_diagnosis_code,
      ten_icd_diagnosis_code,
      procedure_code
    ),
    names_to = "source",
    values_to = "code"
  ) %>%
  filter(!is.na(code), code != "", code != "NA") %>%
  {
    if (length(allowed_codes) > 0) {
      dplyr::filter(., code %in% allowed_codes)
    } else {
      .
    }
  } %>%
  mutate(
    activity = dplyr::case_when(
      source == "drug_name" ~ paste0("DRUG:", code),
      grepl("icd_diagnosis_code", source) ~ paste0("ICD:", code),
      source == "procedure_code" ~ paste0("CPT:", code),
      TRUE ~ code
    ),
    timestamp = as.POSIXct(event_date)
  )

target_eventlog <- pgx_df_target1_long %>%
  transmute(
    case_id              = mi_person_key,
    activity             = activity,
    timestamp            = timestamp,
    activity_instance_id = dplyr::row_number(),
    lifecycle_id         = "complete",
    resource_id          = "Patient"
  ) %>%
  eventlog(
    case_id              = "case_id",
    activity_id          = "activity",
    activity_instance_id = "activity_instance_id",
    lifecycle_id         = "lifecycle_id",
    resource_id          = "resource_id",
    timestamp            = "timestamp"
  )

cat("Target eventlog created.\n")
print(target_eventlog)

# -------------------------------------------------------------------
# Build combined TARGET + CONTROL eventlog for Sankey
# -------------------------------------------------------------------

control_model_data_path <- file.path(
  project_root,
  "model_data",
  paste0("cohort_name=", control_cohort_name),
  paste0("age_band=", age_band),
  "model_events.parquet"
)

if (file.exists(control_model_data_path)) {
  query_control <- sprintf(
    "SELECT * FROM read_parquet('%s') WHERE event_year IN (%s)",
    control_model_data_path,
    paste(train_years, collapse = ",")
  )
  pgx_df_control <- dbGetQuery(con, query_control)
  cat("Loaded ", nrow(pgx_df_control), " control events for ", control_cohort_name,
      " age_band=", age_band, " across years ", paste(train_years, collapse=","), "\n", sep = "")
} else {
  warning("Control model_data parquet not found: ", control_model_data_path)
  pgx_df_control <- pgx_df[0, ]
}

pgx_df_all <- bind_rows(
  pgx_df_target1 %>% mutate(group = "target"),
  pgx_df_control %>% mutate(group = "control")
)

pgx_df_all_long <- pgx_df_all %>%
  transmute(
    mi_person_key,
    event_date,
    group,
    drug_name,
    primary_icd_diagnosis_code,
    two_icd_diagnosis_code,
    three_icd_diagnosis_code,
    four_icd_diagnosis_code,
    five_icd_diagnosis_code,
    six_icd_diagnosis_code,
    seven_icd_diagnosis_code,
    eight_icd_diagnosis_code,
    nine_icd_diagnosis_code,
    ten_icd_diagnosis_code,
    procedure_code
  ) %>%
  # Ensure all pivoted columns are character to avoid type mixing
  mutate(across(
    c(
      drug_name,
      primary_icd_diagnosis_code,
      two_icd_diagnosis_code,
      three_icd_diagnosis_code,
      four_icd_diagnosis_code,
      five_icd_diagnosis_code,
      six_icd_diagnosis_code,
      seven_icd_diagnosis_code,
      eight_icd_diagnosis_code,
      nine_icd_diagnosis_code,
      ten_icd_diagnosis_code,
      procedure_code
    ),
    as.character
  )) %>%
  pivot_longer(
    cols = c(
      drug_name,
      primary_icd_diagnosis_code,
      two_icd_diagnosis_code,
      three_icd_diagnosis_code,
      four_icd_diagnosis_code,
      five_icd_diagnosis_code,
      six_icd_diagnosis_code,
      seven_icd_diagnosis_code,
      eight_icd_diagnosis_code,
      nine_icd_diagnosis_code,
      ten_icd_diagnosis_code,
      procedure_code
    ),
    names_to = "source",
    values_to = "code"
  ) %>%
  filter(!is.na(code), code != "", code != "NA") %>%
  {
    if (length(allowed_codes) > 0) {
      dplyr::filter(., code %in% allowed_codes)
    } else {
      .
    }
  } %>%
  mutate(
    activity = dplyr::case_when(
      source == "drug_name" ~ paste0("DRUG:", code),
      grepl("icd_diagnosis_code", source) ~ paste0("ICD:", code),
      source == "procedure_code" ~ paste0("CPT:", code),
      TRUE ~ code
    ),
    timestamp = as.POSIXct(event_date)
  )

sankey_eventlog <- pgx_df_all_long %>%
  transmute(
    case_id              = mi_person_key,
    activity             = activity,
    timestamp            = timestamp,
    group                = group,
    activity_instance_id = dplyr::row_number(),
    lifecycle_id         = "complete",
    resource_id          = "Patient"
  ) %>%
  eventlog(
    case_id              = "case_id",
    activity_id          = "activity",
    activity_instance_id = "activity_instance_id",
    lifecycle_id         = "lifecycle_id",
    resource_id          = "resource_id",
    timestamp            = "timestamp"
  )

cat("\nCombined TARGET + CONTROL sankey_eventlog created.\n")
print(sankey_eventlog)

cat("\n=== Done. You can now use these event logs in R (e.g., trace explorer, process maps, Sankey plots). ===\n")


