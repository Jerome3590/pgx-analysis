# Broader searches for topics that returned 0 results
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

# Broader search queries
searches <- list(
  "FP-Growth Broader" = list(
    dir = "Chapter1_FPGrowth",
    queries = c(
      "association rules healthcare",
      "frequent pattern mining healthcare",
      "market basket analysis healthcare"
    ),
    filename = "fpgrowth_broader_articles.csv"
  ),
  "Process Mining Broader" = list(
    dir = "Chapter1_ProcessMining",
    queries = c(
      "process mining healthcare",
      "patient journey analysis",
      "temporal sequence analysis healthcare"
    ),
    filename = "process_mining_broader_articles.csv"
  ),
  "Opioid Use Disorder Broader" = list(
    dir = "Chapter1_OpioidDisorder",
    queries = c(
      "opioid use disorder risk factors",
      "opioid dependence prediction",
      "opioid addiction trajectories"
    ),
    filename = "opioid_disorder_broader_articles.csv"
  ),
  "DTW Dynamic Time Warping" = list(
    dir = "Chapter1_DTW",
    queries = c(
      "dynamic time warping healthcare",
      "DTW time series healthcare",
      "temporal alignment healthcare"
    ),
    filename = "dtw_articles.csv"
  ),
  "DuckDB Broader" = list(
    dir = "Chapter1_DuckDB",
    queries = c(
      "columnar database healthcare",
      "analytical database healthcare",
      "OLAP healthcare analytics"
    ),
    filename = "duckdb_broader_articles.csv"
  )
)

cat("=== Running Broader Additional Searches ===\n\n")

for (name in names(searches)) {
  search_info <- searches[[name]]
  
  if (!dir.exists(here(search_info$dir))) {
    dir.create(here(search_info$dir), recursive = TRUE)
  }
  
  setwd(here(search_info$dir))
  
  all_results <- tibble()
  
  cat("Topic:", name, "\n")
  
  for (query in search_info$queries) {
    cat("  Query:", query, "\n")
    result <- search_pubmed_all(query, paste0("temp_", gsub(" ", "_", query), ".csv"))
    cat("  Found:", result$count, "articles\n")
    
    if (result$count > 0 && nrow(result$articles) > 0) {
      all_results <- bind_rows(all_results, result$articles)
    }
    
    Sys.sleep(0.5)
  }
  
  # Deduplicate and save
  if (nrow(all_results) > 0) {
    all_results <- all_results %>%
      distinct(title, pubdate, pmc_id, .keep_all = TRUE)
    
    write_csv(all_results, search_info$filename)
    cat("  Total unique articles:", nrow(all_results), "\n\n")
  } else {
    cat("  No articles found\n\n")
  }
}

cat("=== Searches Complete ===\n")
