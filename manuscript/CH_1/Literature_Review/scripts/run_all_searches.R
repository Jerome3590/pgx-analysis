# ─────────────────────────────────────────────────────────────────────────────
# run_all_searches.R
# Runs all 18 Chapter 1 PubMed searches and saves CSVs to correct data paths.
#
# Run from: manuscript/CH_1/Literature_Review/
#   Rscript.exe scripts/run_all_searches.R
#
# Outputs:
#   data/chapter1/<topic>/<filename>.csv  — one CSV per search
#   scripts/search_status_log.csv         — live per-search status log
# ─────────────────────────────────────────────────────────────────────────────

library(rentrez)
library(dplyr)
library(readr)
library(xml2)
library(purrr)
library(here)
library(stringr)
library(digest)

here::i_am("lit_review.qmd")

# ── Optional: NCBI API key (free) ────────────────────────────────────────────
# Register at https://www.ncbi.nlm.nih.gov/account/
# With key: 10 req/sec. Without: 3 req/sec.
# Uncomment and set your key:
# rentrez::set_entrez_key("YOUR_NCBI_API_KEY_HERE")

# ── Search function ───────────────────────────────────────────────────────────
search_pubmed_all <- function(query, out_dir, filename) {
  dir.create(here(out_dir), recursive = TRUE, showWarnings = FALSE)
  out_path <- here(out_dir, filename)

  current_year <- as.integer(format(Sys.Date(), "%Y"))
  start_year   <- current_year - 5
  full_query   <- paste0(query, " AND ", start_year, ":", current_year, "[PDAT]")

  cat("  Query:", full_query, "\n")

  initial_search <- tryCatch(
    entrez_search(db = "pubmed", term = full_query, use_history = TRUE),
    error = function(e) { cat("  ERROR:", e$message, "\n"); return(NULL) }
  )
  if (is.null(initial_search)) return(NA_integer_)

  total_count <- initial_search$count
  cat("  Found:", total_count, "records\n")

  if (total_count == 0) {
    write_csv(
      tibble(title = character(), authors = character(),
             pubdate = character(), pmc_id = character()),
      out_path
    )
    cat("  Saved empty CSV:", out_path, "\n")
    return(0L)
  }

  batch_size   <- 200  # 200/batch = 4x fewer API calls vs 50; safe under both key/no-key limits
  all_batches  <- list()

  for (start in seq(1, min(total_count, 5000), by = batch_size)) {
    tryCatch({
      fetched  <- entrez_fetch(
        db          = "pubmed",
        web_history = initial_search$web_history,
        retstart    = start - 1,
        retmax      = batch_size,
        rettype     = "xml"
      )
      xml_doc  <- read_xml(fetched)
      articles <- xml_find_all(xml_doc, "//PubmedArticle")

      if (length(articles) > 0) {
        xt1 <- function(n, xp) { v <- xml_text(xml_find_first(n, xp)); if (length(v) == 0L || is.null(v)) NA_character_ else as.character(v[[1L]]) }
        batch_df <- tibble(
          title   = vapply(articles, function(n) xt1(n, ".//ArticleTitle"),                       FUN.VALUE = character(1L)),
          authors = vapply(articles, function(n) paste(xml_text(xml_find_all(n, ".//Author//LastName")), collapse = ", "), FUN.VALUE = character(1L)),
          pubdate = vapply(articles, function(n) xt1(n, ".//PubDate/Year"),                        FUN.VALUE = character(1L)),
          pmc_id  = vapply(articles, function(n) xt1(n, ".//ArticleId[@IdType='pmc']"),            FUN.VALUE = character(1L))
        )
        all_batches <- c(all_batches, list(batch_df))
      }
      Sys.sleep(0.12)  # 0.12s ≈ 8 req/sec; safe with API key (10/s) and well under no-key limit at 200/batch
    }, error = function(e) {
      cat("  Batch error at", start, ":", e$message, "\n")
    })
  }

  result_df <- if (length(all_batches) > 0) {
    bind_rows(lapply(all_batches, function(b) mutate(b, across(everything(), as.character)))) %>%
      distinct(title, pubdate, pmc_id, .keep_all = TRUE) %>%
      mutate(
        pmc_id = case_when(
          is.na(pmc_id) | pmc_id == "" ~
            paste0("HSH", substr(digest(title, algo = "md5"), 1, 8)),
          !str_starts(pmc_id, "PMC") ~ paste0("PMC", pmc_id),
          TRUE ~ pmc_id
        )
      )
  } else {
    tibble(title = character(), authors = character(),
           pubdate = character(), pmc_id = character())
  }

  write_csv(result_df, out_path)
  cat("  Saved:", nrow(result_df), "unique articles →", out_path, "\n")
  invisible(nrow(result_df))
}

# ── Search manifest (all 18 Chapter 1 searches) ───────────────────────────────
searches <- list(
  list(n =  1, label = "Black-Box ML + CDS",
       query = "black box machine learning clinical decision support interpretability explainable AI",
       dir   = "data/chapter1/1.1_introduction/blackbox_cds",
       file  = "blackbox_cds_articles.csv"),

  list(n =  2, label = "APCD Analysis",
       query = "all payers claim database",
       dir   = "data/chapter1/1.3_methodological/apcd_analysis",
       file  = "apcd_analysis_articles.csv"),

  list(n =  3, label = "Pharmacovigilance",
       query = "pharmacovigilance pharmacogenomics",
       dir   = "data/chapter1/1.2_clinical_background/pharmacovigilance",
       file  = "pharmacovigilance_articles.csv"),

  list(n =  4, label = "Interpretability / SHAP",
       query = "SHAP Shapley additive explanations feature importance interpretability healthcare machine learning",
       dir   = "data/chapter1/1.1_introduction/interpretability",
       file  = "interpretability_articles.csv"),

  list(n =  5, label = "FP-Growth / Association Rules",
       query = "association rules healthcare",
       dir   = "data/chapter1/1.3_methodological/pattern_mining/fpgrowth",
       file  = "fpgrowth_articles.csv"),

  list(n =  6, label = "Process Mining / BupaR",
       query = "process mining healthcare",
       dir   = "data/chapter1/1.3_methodological/pattern_mining/process_mining",
       file  = "process_mining_articles.csv"),

  list(n =  7, label = "Opioid Use Disorder",
       query = "opioid use disorder risk factors",
       dir   = "data/chapter1/1.2_clinical_background/opioid_disorder",
       file  = "opioid_disorder_articles.csv"),

  list(n =  8, label = "Polypharmacy",
       query = "polypharmacy elderly drug interactions adverse events",
       dir   = "data/chapter1/1.2_clinical_background/polypharmacy",
       file  = "polypharmacy_articles.csv"),

  list(n =  9, label = "Drug-Drug Interactions",
       query = "drug-drug interactions DDI synergistic adverse drug events",
       dir   = "data/chapter1/1.2_clinical_background/drug_interactions",
       file  = "drug_interactions_articles.csv"),

  list(n = 10, label = "CatBoost / XGBoost",
       query = "CatBoost XGBoost gradient boosting healthcare claims data",
       dir   = "data/chapter1/1.4_technical/catboost_xgboost",
       file  = "catboost_xgboost_articles.csv"),

  list(n = 11, label = "Dynamic Time Warping",
       query = "dynamic time warping healthcare",
       dir   = "data/chapter1/1.3_methodological/pattern_mining/dtw",
       file  = "dtw_articles.csv"),

  list(n = 12, label = "Temporal Causality",
       query = "temporal causality healthcare claims data temporal windows",
       dir   = "data/chapter1/1.3_methodological/temporal_causality",
       file  = "temporal_causality_articles.csv"),

  list(n = 13, label = "Target Leakage Prevention",
       query = "target leakage data leakage machine learning healthcare prevention",
       dir   = "data/chapter1/1.3_methodological/target_leakage",
       file  = "target_leakage_articles.csv"),

  list(n = 14, label = "DuckDB / OLAP Analytics",
       query = "analytical database healthcare",
       dir   = "data/chapter1/1.4_technical/duckdb_olap",
       file  = "duckdb_articles.csv"),

  list(n = 15, label = "CPT Codes + Opioid Risk (NEW)",
       query = "CPT procedure codes opioid risk prediction claims",
       dir   = "data/chapter1/1.2_clinical_background/opioid_disorder/cpt_opioid",
       file  = "cpt_opioid_articles.csv"),

  list(n = 16, label = "Opioid ED Prediction (NEW)",
       query = "opioid use disorder emergency department visit prediction machine learning",
       dir   = "data/chapter1/1.2_clinical_background/opioid_disorder/opioid_ed_prediction",
       file  = "opioid_ed_prediction_articles.csv"),

  list(n = 17, label = "Polypharmacy ED / Drug Combinations (NEW)",
       query = "drug combination polypharmacy adverse drug event elderly emergency",
       dir   = "data/chapter1/1.2_clinical_background/drug_interactions/polypharmacy_ed",
       file  = "polypharmacy_ed_articles.csv"),

  list(n = 18, label = "Routine vs. Non-Routine Care (NEW)",
       query = "healthcare utilization patterns routine care administrative claims",
       dir   = "data/chapter1/1.3_methodological/routine_care",
       file  = "routine_care_articles.csv")
)

# ── Run all searches ──────────────────────────────────────────────────────────
status_log <- tibble(
  search_num = integer(),
  label      = character(),
  n_articles = integer(),
  status     = character(),
  out_path   = character(),
  timestamp  = character()
)

cat("\n════════════════════════════════════════════════════════════════\n")
cat("  PubMed Literature Search — All 18 Searches\n")
cat("  Started:", format(Sys.time()), "\n")
cat("════════════════════════════════════════════════════════════════\n\n")

for (s in searches) {
  cat(sprintf("\n[%2d/18] %s\n", s$n, s$label))

  n <- tryCatch(
    search_pubmed_all(s$query, s$dir, s$file),
    error = function(e) { cat("  FAILED:", e$message, "\n"); NA_integer_ }
  )

  status_log <- bind_rows(status_log, tibble(
    search_num = s$n,
    label      = s$label,
    n_articles = if (is.na(n)) NA_integer_ else as.integer(n),
    status     = if (is.na(n)) "ERROR" else "DONE",
    out_path   = here(s$dir, s$file),
    timestamp  = format(Sys.time())
  ))

  write_csv(status_log, here("scripts", "search_status_log.csv"))
  Sys.sleep(1)
}

cat("\n════════════════════════════════════════════════════════════════\n")
cat("  All searches complete:", format(Sys.time()), "\n")
cat("  Status log: scripts/search_status_log.csv\n")
cat("════════════════════════════════════════════════════════════════\n\n")
print(status_log, n = 18)
