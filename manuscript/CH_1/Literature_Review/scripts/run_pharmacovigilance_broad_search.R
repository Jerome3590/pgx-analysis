# Broader pharmacovigilance searches with pharmacogenomics
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

setwd(here("Chapter1_Pharmacovigilance"))

# Try multiple search variations
queries <- list(
  "pharmacovigilance pharmacogenomics" = "pharmacovigilance pharmacogenomics",
  "pharmacovigilance adverse drug events" = "pharmacovigilance adverse drug events",
  "pharmacogenomics adverse drug events" = "pharmacogenomics adverse drug events",
  "pharmacovigilance FAERS" = "pharmacovigilance FAERS",
  "pharmacogenomics FAERS" = "pharmacogenomics FAERS",
  "pharmacovigilance machine learning" = "pharmacovigilance machine learning",
  "pharmacogenomics machine learning" = "pharmacogenomics machine learning"
)

all_results <- tibble()

cat("=== Testing Multiple Pharmacovigilance/Pharmacogenomics Search Queries ===\n\n")

for (name in names(queries)) {
  query <- queries[[name]]
  filename <- paste0("pharmacovigilance_", gsub(" ", "_", name), "_articles.csv")
  
  cat("Query:", query, "\n")
  result <- search_pubmed_all(query, filename)
  cat("Found:", result$count, "articles\n\n")
  
  if (result$count > 0 && nrow(result$articles) > 0) {
    result$articles$query <- name
    all_results <- bind_rows(all_results, result$articles)
  }
  
  # Be nice to PubMed API
  Sys.sleep(0.5)
}

# Combine and deduplicate all results
if (nrow(all_results) > 0) {
  all_results <- all_results %>%
    distinct(title, pubdate, pmc_id, .keep_all = TRUE) %>%
    select(-query)
  
  write_csv(all_results, "pharmacovigilance_articles_combined.csv")
  cat("=== Combined Results ===\n")
  cat("Total unique articles:", nrow(all_results), "\n")
  cat("Saved to: pharmacovigilance_articles_combined.csv\n")
} else {
  cat("No articles found across all queries.\n")
}
