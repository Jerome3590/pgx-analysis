library(dplyr)
library(readr)
library(stringr)
library(here)
library(purrr)
library(digest)

# ─────────────────────────────────────────────────────────────────────────────
# PRISMA Tracker
# Maps to repo RQs: docs/CrossStep_Workflow/README_research_questions_mapping.md
#   RQ1  non_opioid_ed  — drug window → non-opioid ED
#   RQ2  opioid_ed      — ICD/CPT/drug → OPIOID_ED prediction
#   N1   routine vs. no-routine trajectories (DTW)
#   N2   sequences to target (BupaR)
#   N3   times between sequences (BupaR)
#   N4   ICD/CPT/Drug connections (FP-Growth)
#   N5   feature drivers + relations (FFA/SHAP)
#   N6   drug combinations → polypharmacy ED (FFA/SHAP + BupaR)
# ─────────────────────────────────────────────────────────────────────────────

here::i_am("lit_review.qmd")

# ── 1. CSV manifest ──────────────────────────────────────────────────────────
# Each entry: path relative to Literature_Review/, RQ tags, search label
csv_manifest <- tribble(
  ~rel_path,                                                                 ~rq,               ~label,
  "data/chapter1/1.1_introduction/blackbox_cds/blackbox_cds_articles.csv",  "N5",              "Black-Box ML + CDS",
  "data/chapter1/1.3_methodological/apcd_analysis/apcd_analysis_articles.csv", "RQ1,RQ2",     "APCD Analysis",
  "data/chapter1/1.2_clinical_background/pharmacovigilance/pharmacovigilance_articles.csv", "RQ1,RQ2", "Pharmacovigilance",
  "data/chapter1/1.1_introduction/interpretability/interpretability_articles.csv", "N5",       "Interpretability / SHAP",
  "data/chapter1/1.3_methodological/pattern_mining/fpgrowth/fpgrowth_articles.csv", "N4",     "FP-Growth / Association Rules",
  "data/chapter1/1.3_methodological/pattern_mining/process_mining/process_mining_articles.csv", "N2,N3", "Process Mining (BupaR)",
  "data/chapter1/1.4_technical/catboost_xgboost/catboost_xgboost_articles.csv", "RQ1,RQ2",   "CatBoost / XGBoost",
  "data/chapter1/1.3_methodological/pattern_mining/dtw/dtw_articles.csv",   "N1",              "Dynamic Time Warping (DTW)",
  "data/chapter1/1.3_methodological/temporal_causality/temporal_causality_articles.csv", "RQ1", "Temporal Causality",
  "data/chapter1/1.3_methodological/target_leakage/target_leakage_articles.csv", "RQ1,RQ2",  "Target Leakage Prevention",
  "data/chapter1/1.2_clinical_background/opioid_disorder/opioid_disorder_articles.csv", "RQ2", "Opioid Use Disorder",
  "data/chapter1/1.2_clinical_background/polypharmacy/polypharmacy_articles.csv", "RQ1",     "Polypharmacy",
  "data/chapter1/1.2_clinical_background/drug_interactions/drug_interactions_articles.csv", "RQ1,N6", "Drug-Drug Interactions",
  "data/chapter1/1.4_technical/duckdb_olap/duckdb_articles.csv",            "RQ1,RQ2",         "DuckDB / OLAP Analytics",
  "data/other_chapters/pgx_models/pgx_risk_classifcation_articles.csv",     "RQ1,RQ2",         "PGx Classification Models",
  "data/other_chapters/ehr_models/risk_model_ehr_articles.csv",             "RQ1,RQ2",         "Risk Models with EHR/CDS",
  "data/other_chapters/fhir_models/fhir_ehr_articles.csv",                  "RQ1,RQ2",         "Risk Models with FHIR"
)

# ── 2. Load all CSVs ─────────────────────────────────────────────────────────
load_csv_safe <- function(rel_path, rq, label) {
  full_path <- here(rel_path)
  if (!file.exists(full_path)) {
    message("  [MISSING] ", rel_path)
    return(NULL)
  }
  df <- read_csv(full_path, show_col_types = FALSE)
  df$source_file  <- rel_path
  df$rq_tags      <- rq
  df$search_label <- label
  df
}

all_results <- pmap(csv_manifest, load_csv_safe)
all_results <- keep(all_results, Negate(is.null))

if (length(all_results) == 0) {
  stop("No CSV files found. Run lit_review.qmd search chunks first.")
}

combined <- bind_rows(all_results)

# Normalise key columns
combined <- combined %>%
  mutate(
    title   = str_trim(str_to_lower(coalesce(title, ""))),
    pmc_id  = str_trim(coalesce(as.character(pmc_id), "")),
    authors = str_trim(coalesce(as.character(authors), ""))
  )

# ── 3. PRISMA Stage 1 — Identified ──────────────────────────────────────────
n_identified <- nrow(combined)

# ── 4. PRISMA Stage 2 — Duplicates removed ───────────────────────────────────
# Deduplicate on normalised title; keep first occurrence
combined_dedup <- combined %>%
  filter(title != "") %>%
  distinct(title, .keep_all = TRUE)

n_after_title_dedup <- nrow(combined_dedup)
n_duplicates        <- n_identified - n_after_title_dedup

# ── 5. PRISMA Stage 3 — Screened ─────────────────────────────────────────────
n_screened <- n_after_title_dedup

# ── 6. Assign HSH stubs for articles without PMC ID ──────────────────────────
combined_dedup <- combined_dedup %>%
  rowwise() %>%
  mutate(
    pmc_id_clean = case_when(
      !is.na(pmc_id) & pmc_id != "" & !str_starts(pmc_id, "HSH") ~ pmc_id,
      TRUE ~ paste0("HSH", substr(digest(authors, algo = "md5"), 1, 8))
    )
  ) %>%
  ungroup()

n_has_pmc_id <- sum(!str_starts(combined_dedup$pmc_id_clean, "HSH"))
n_hsh_only   <- sum( str_starts(combined_dedup$pmc_id_clean, "HSH"))

# ── 7. PRISMA Stage 4 — Full-text sought ─────────────────────────────────────
# Count JSON files downloaded across all topic directories
json_dirs <- c(
  here("data/chapter1/1.1_introduction/blackbox_cds/pubmed_json_files"),
  here("data/chapter1/1.3_methodological/apcd_analysis/pubmed_json_files"),
  here("data/chapter1/1.2_clinical_background/pharmacovigilance/pubmed_json_files"),
  here("data/chapter1/1.1_introduction/interpretability/pubmed_json_files"),
  here("data/chapter1/1.3_methodological/pattern_mining/fpgrowth/pubmed_json_files"),
  here("data/chapter1/1.3_methodological/pattern_mining/process_mining/pubmed_json_files"),
  here("data/chapter1/1.4_technical/catboost_xgboost/pubmed_json_files"),
  here("data/chapter1/1.3_methodological/pattern_mining/dtw/pubmed_json_files"),
  here("data/chapter1/1.3_methodological/temporal_causality/pubmed_json_files"),
  here("data/chapter1/1.3_methodological/target_leakage/pubmed_json_files"),
  here("data/chapter1/1.2_clinical_background/opioid_disorder/pubmed_json_files"),
  here("data/chapter1/1.2_clinical_background/polypharmacy/pubmed_json_files"),
  here("data/chapter1/1.2_clinical_background/drug_interactions/pubmed_json_files"),
  here("data/chapter1/1.4_technical/duckdb_olap/pubmed_json_files"),
  here("data/other_chapters/pgx_models/pubmed_json_files"),
  here("data/other_chapters/ehr_models/pubmed_json_files"),
  here("data/other_chapters/fhir_models/pubmed_json_files")
)

count_json_files <- function(dir) {
  if (!dir.exists(dir)) return(0L)
  length(list.files(dir, pattern = "\\.json$", recursive = FALSE))
}

n_fulltext_retrieved <- sum(map_int(json_dirs, count_json_files))
n_fulltext_not_retrieved <- n_has_pmc_id - n_fulltext_retrieved

# ── 8. PRISMA Stage 5 — Excluded at screen (no full text, not Zotero) ────────
# Articles with HSH stubs and no JSON downloaded are excluded at screening
n_excluded_screen <- n_hsh_only  # no PMC ID and assumed no Zotero PDF yet

# ── 9. PRISMA Stage 6 — Eligibility: full-text assessed ──────────────────────
n_fulltext_assessed <- n_fulltext_retrieved

# Check for Selected column (added during manual review)
if ("Selected" %in% names(combined_dedup)) {
  selected_vals <- combined_dedup %>%
    filter(!is.na(Selected), Selected != "") %>%
    mutate(Selected = as.logical(Selected))

  n_included          <- sum(selected_vals$Selected,  na.rm = TRUE)
  n_excluded_fulltext <- sum(!selected_vals$Selected, na.rm = TRUE)
} else {
  n_included          <- NA_integer_
  n_excluded_fulltext <- NA_integer_
  message("  [INFO] No 'Selected' column found — run full-text review to populate.")
}

# ── 10. Per-RQ article counts ─────────────────────────────────────────────────
rq_counts <- csv_manifest %>%
  left_join(
    combined_dedup %>%
      group_by(search_label) %>%
      summarise(n_dedup = n(), .groups = "drop"),
    by = c("label" = "search_label")
  ) %>%
  select(label, rq, n_dedup) %>%
  arrange(rq, label)

# ── 11. Build PRISMA counts object ───────────────────────────────────────────
prisma_counts <- list(
  n_identified          = n_identified,
  n_duplicates          = n_duplicates,
  n_screened            = n_screened,
  n_excluded_screen     = n_excluded_screen,
  n_fulltext_assessed   = n_fulltext_assessed,
  n_fulltext_not_found  = n_fulltext_not_retrieved,
  n_excluded_fulltext   = n_excluded_fulltext,
  n_included            = n_included,
  n_has_pmc_id          = n_has_pmc_id,
  n_hsh_only            = n_hsh_only,
  rq_counts             = rq_counts
)

# ── 12. Save outputs ──────────────────────────────────────────────────────────
saveRDS(prisma_counts, here("scripts", "prisma_counts.rds"))

write_csv(
  tibble(
    stage              = c("Identified", "Duplicates removed", "Screened",
                           "Excluded (no full text)", "Full-text assessed",
                           "Full-text not retrieved", "Excluded (full-text review)",
                           "Included"),
    n                  = c(n_identified, n_duplicates, n_screened,
                           n_excluded_screen, n_fulltext_assessed,
                           n_fulltext_not_retrieved, n_excluded_fulltext,
                           n_included)
  ),
  here("scripts", "prisma_counts.csv")
)

write_csv(rq_counts, here("scripts", "prisma_rq_counts.csv"))

# Generate missing articles CSV for manual Zotero download
missing_articles <- combined_dedup %>%
  filter(str_starts(pmc_id_clean, "HSH")) %>%
  select(title, authors, pubdate, pmc_id = pmc_id_clean, search_label, rq_tags, source_file) %>%
  arrange(rq_tags, search_label, title)

write_csv(missing_articles, here("scripts", "missing_articles_combined.csv"))

# ── 13. Console summary ───────────────────────────────────────────────────────
cat("\n── PRISMA Summary ─────────────────────────────────────────\n")
cat(sprintf("  Identified (total rows):         %6d\n", n_identified))
cat(sprintf("  Duplicates removed:              %6d\n", n_duplicates))
cat(sprintf("  Screened (after dedup):          %6d\n", n_screened))
cat(sprintf("    With PMC ID (OA):              %6d\n", n_has_pmc_id))
cat(sprintf("    Without PMC ID (HSH stub):     %6d\n", n_hsh_only))
cat(sprintf("  Full-text retrieved (JSON):      %6d\n", n_fulltext_retrieved))
cat(sprintf("  Full-text not retrieved:         %6d\n", n_fulltext_not_retrieved))
cat(sprintf("  Excluded at screen (no text):    %6d\n", n_excluded_screen))
cat(sprintf("  Full-text assessed:              %6d\n", n_fulltext_assessed))
if (!is.na(n_excluded_fulltext)) {
  cat(sprintf("  Excluded (full-text review):     %6d\n", n_excluded_fulltext))
  cat(sprintf("  Included in synthesis:           %6d\n", n_included))
} else {
  cat("  Included in synthesis:           [run full-text review]\n")
}
cat("\n── Per-RQ counts (after dedup) ────────────────────────────\n")
print(rq_counts, n = Inf)
cat("\n── Outputs written ────────────────────────────────────────\n")
cat("  scripts/prisma_counts.rds\n")
cat("  scripts/prisma_counts.csv\n")
cat("  scripts/prisma_rq_counts.csv\n")
cat("  scripts/missing_articles_combined.csv\n\n")
