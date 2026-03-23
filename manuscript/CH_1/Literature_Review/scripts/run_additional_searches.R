# Additional literature searches for Chapter 1 topics
library(rentrez)
library(dplyr)
library(readr)
library(xml2)
library(purrr)
library(here)
library(stringr)
library(digest)

search_pubmed_all <- function(query, filename) {
  current_year <- format(Sys.Date(), "%Y")
  start_year <- as.integer(current_year) - 5
  query <- paste(query, "AND", paste(start_year, ":", current_year, "[PDAT]"))

  initial_search <- entrez_search(db="pubmed", term=query, use_history = TRUE)
  total_count <- initial_search$count
  
  if (total_count == 0) {
    return(list(count = 0, articles = tibble()))
  }
  
  batch_size <- 50
  all_articles <- list()
  
  for (start in seq(1, total_count, by = batch_size)) {
    tryCatch({
      fetched_articles <- entrez_fetch(db="pubmed", 
                                       web_history = initial_search$web_history, 
                                       retstart = start - 1,
                                       retmax = batch_size,
                                       rettype = "xml")
      xml_articles <- read_xml(fetched_articles)
      articles <- xml_find_all(xml_articles, "//PubmedArticle")
      
      if (length(articles) > 0) {
        article_details <- map_df(articles, ~{
          title <- xml_text(xml_find_first(.x, ".//ArticleTitle"))
          authors <- xml_text(xml_find_all(.x, ".//Author//LastName"))
          pubdate <- xml_text(xml_find_first(.x, ".//PubDate/Year"))
          pmc_id <- xml_text(xml_find_first(.x, ".//ArticleId[@IdType='pmc']"))
          tibble(title = title, authors = authors, pubdate = pubdate, pmc_id = pmc_id)
        })
        all_articles <- bind_rows(all_articles, article_details)
      }
    }, error = function(e) {
      cat("Warning:", e$message, "\n")
    })
  }
  
  if (!is.null(all_articles) && length(all_articles) > 0 && nrow(all_articles) > 0) {
    all_articles <- all_articles %>% 
      group_by(title, pubdate, pmc_id) %>%
      summarise(authors = paste(authors, collapse = ", "), .groups = 'drop') %>% 
      mutate(pmc_id = if_else(str_starts(pmc_id, "PMC"), pmc_id, paste0("PMC", pmc_id)))
  } else {
    all_articles <- tibble(title = character(), authors = character(), pubdate = character(), pmc_id = character())
  }

  write_csv(all_articles, filename)
  return(list(count = total_count, articles = all_articles))
}

# Create directories
dirs_to_create <- c("Chapter1_FPGrowth", 
                    "Chapter1_ProcessMining",
                    "Chapter1_OpioidDisorder",
                    "Chapter1_Polypharmacy",
                    "Chapter1_DrugInteractions",
                    "Chapter1_CatBoost",
                    "Chapter1_DuckDB",
                    "Chapter1_TemporalCausality",
                    "Chapter1_TargetLeakage")

for (dir in dirs_to_create) {
  if (!dir.exists(here(dir))) {
    dir.create(here(dir), recursive = TRUE)
    cat("Created directory:", dir, "\n")
  }
}

# Search queries
searches <- list(
  "FP-Growth" = list(
    dir = "Chapter1_FPGrowth",
    query = "FP-Growth association rules market basket analysis healthcare",
    filename = "fpgrowth_articles.csv"
  ),
  "Process Mining" = list(
    dir = "Chapter1_ProcessMining",
    query = "process mining BupaR patient journey healthcare temporal sequences",
    filename = "process_mining_articles.csv"
  ),
  "Opioid Use Disorder" = list(
    dir = "Chapter1_OpioidDisorder",
    query = "opioid use disorder OUD risk factors temporal addiction trajectories",
    filename = "opioid_disorder_articles.csv"
  ),
  "Polypharmacy" = list(
    dir = "Chapter1_Polypharmacy",
    query = "polypharmacy elderly drug interactions adverse events",
    filename = "polypharmacy_articles.csv"
  ),
  "Drug-Drug Interactions" = list(
    dir = "Chapter1_DrugInteractions",
    query = "drug-drug interactions DDI synergistic adverse drug events",
    filename = "drug_interactions_articles.csv"
  ),
  "CatBoost XGBoost" = list(
    dir = "Chapter1_CatBoost",
    query = "CatBoost XGBoost gradient boosting healthcare claims data",
    filename = "catboost_xgboost_articles.csv"
  ),
  "DuckDB OLAP" = list(
    dir = "Chapter1_DuckDB",
    query = "DuckDB OLAP healthcare analytics big data",
    filename = "duckdb_articles.csv"
  ),
  "Temporal Causality" = list(
    dir = "Chapter1_TemporalCausality",
    query = "temporal causality healthcare claims data temporal windows",
    filename = "temporal_causality_articles.csv"
  ),
  "Target Leakage" = list(
    dir = "Chapter1_TargetLeakage",
    query = "target leakage data leakage machine learning healthcare prevention",
    filename = "target_leakage_articles.csv"
  )
)

cat("=== Running Additional Literature Searches ===\n\n")

results_summary <- tibble()

for (name in names(searches)) {
  search_info <- searches[[name]]
  setwd(here(search_info$dir))
  
  cat("Search:", name, "\n")
  cat("Query:", search_info$query, "\n")
  
  result <- search_pubmed_all(search_info$query, search_info$filename)
  
  cat("Found:", result$count, "articles\n\n")
  
  results_summary <- bind_rows(results_summary, 
                                tibble(topic = name, 
                                       count = result$count,
                                       filename = search_info$filename))
  
  Sys.sleep(0.5)  # Be nice to PubMed API
}

cat("=== Search Summary ===\n")
print(results_summary)
cat("\nTotal articles found:", sum(results_summary$count), "\n")
