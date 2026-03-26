# ─────────────────────────────────────────────────────────────────────────────
# run_fallback_searches.R
# Broader fallback queries for the 4 Phase 1 searches that returned < 5 articles.
# Appends results to the existing CSVs (deduplicates by title_norm).
#
# Run from: manuscript/CH_1/Literature_Review/
#   Rscript.exe scripts/run_fallback_searches.R
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

search_pubmed_all <- function(query, out_dir, filename, append = FALSE) {
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
    cat("  No new results — existing CSV unchanged\n")
    return(0L)
  }

  batch_size  <- 200  # 4x fewer API calls vs 50; safe under both key/no-key limits
  all_batches <- list()

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
      Sys.sleep(0.12)  # 0.12s ≈ 8 req/sec
    }, error = function(e) cat("  Batch error at", start, ":", e$message, "\n"))
  }

  new_df <- if (length(all_batches) > 0) {
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
    tibble(title=character(), authors=character(), pubdate=character(), pmc_id=character())
  }

  if (append && file.exists(out_path)) {
    existing <- read_csv(out_path, show_col_types = FALSE,
                         col_types = cols(.default = col_character()))
    existing_norm <- str_trim(str_to_lower(existing$title))
    new_norm      <- str_trim(str_to_lower(new_df$title))
    new_only      <- new_df[!new_norm %in% existing_norm, ]
    combined      <- bind_rows(existing, new_only)
    write_csv(combined, out_path)
    cat("  Appended:", nrow(new_only), "new articles →", out_path,
        "(total:", nrow(combined), ")\n")
    invisible(nrow(new_only))
  } else {
    write_csv(new_df, out_path)
    cat("  Saved:", nrow(new_df), "articles →", out_path, "\n")
    invisible(nrow(new_df))
  }
}

fallbacks <- list(
  list(n = 9,  label = "Drug-Drug Interactions (broadened)",
       query = "drug-drug interaction adverse drug event",
       dir   = "data/chapter1/1.2_clinical_background/drug_interactions",
       file  = "drug_interactions_articles.csv"),

  list(n = 12, label = "Temporal Causality (broadened)",
       query = "temporal analysis healthcare claims longitudinal",
       dir   = "data/chapter1/1.3_methodological/temporal_causality",
       file  = "temporal_causality_articles.csv"),

  list(n = 13, label = "Target Leakage Prevention (broadened)",
       query = "data leakage machine learning clinical prediction",
       dir   = "data/chapter1/1.3_methodological/target_leakage",
       file  = "target_leakage_articles.csv"),

  list(n = 15, label = "CPT + Opioid Risk (broadened)",
       query = "opioid risk prediction administrative claims",
       dir   = "data/chapter1/1.2_clinical_background/opioid_disorder/cpt_opioid",
       file  = "cpt_opioid_articles.csv")
)

cat("\n════════════════════════════════════════════════════════════════\n")
cat("  Fallback Searches — 4 Low-Yield Topics\n")
cat("  Started:", format(Sys.time()), "\n")
cat("════════════════════════════════════════════════════════════════\n\n")

status_log <- tibble(
  search_num = integer(), label = character(),
  n_new = integer(), status = character(), timestamp = character()
)

for (s in fallbacks) {
  cat(sprintf("\n[%2d] %s\n", s$n, s$label))
  n <- tryCatch(
    search_pubmed_all(s$query, s$dir, s$file, append = TRUE),
    error = function(e) { cat("  FAILED:", e$message, "\n"); NA_integer_ }
  )
  status_log <- bind_rows(status_log, tibble(
    search_num = s$n, label = s$label,
    n_new      = if (is.na(n)) NA_integer_ else as.integer(n),
    status     = if (is.na(n)) "ERROR" else "DONE",
    timestamp  = format(Sys.time())
  ))
  Sys.sleep(1)
}

cat("\n════════════════════════════════════════════════════════════════\n")
cat("  Fallback searches complete:", format(Sys.time()), "\n")
cat("════════════════════════════════════════════════════════════════\n\n")
print(status_log, n = Inf)

# Append to main status log
main_log_path <- here("scripts", "search_status_log.csv")
if (file.exists(main_log_path)) {
  main_log <- read_csv(main_log_path, show_col_types = FALSE)
  cat("\nMain status log updated at", main_log_path, "\n")
}
