# Quick script to rerun APCD search with updated query
library(rentrez)
library(dplyr)
library(readr)
library(xml2)
library(purrr)
library(here)
library(stringr)
library(digest)

# Define the search function
search_pubmed_all <- function(query, filename) {
  current_year <- format(Sys.Date(), "%Y")
  start_year <- as.integer(current_year) - 5
  query <- paste(query, "AND", paste(start_year, ":", current_year, "[PDAT]"))

  initial_search <- entrez_search(db="pubmed", term=query, use_history = TRUE)
  total_count <- initial_search$count
  
  if (total_count == 0) {
    return(paste("No articles found for query:", query))
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
      cat("Warning: Error fetching batch starting at", start, ":", e$message, "\n")
    })
  }
  
  if (!is.null(all_articles) && length(all_articles) > 0) {
    if (is.data.frame(all_articles) && nrow(all_articles) > 0) {
      all_articles <- all_articles %>% 
        group_by(title, pubdate, pmc_id) %>%
        summarise(authors = paste(authors, collapse = ", "), .groups = 'drop') %>% 
        mutate(pmc_id = if_else(str_starts(pmc_id, "PMC"), pmc_id, paste0("PMC", pmc_id)))
    } else {
      all_articles <- tibble(title = character(), authors = character(), pubdate = character(), pmc_id = character())
    }
  } else {
    all_articles <- tibble(title = character(), authors = character(), pubdate = character(), pmc_id = character())
  }

  write_csv(all_articles, filename)
  return(paste("Data saved to", filename, "- Found", total_count, "articles"))
}

# Run APCD search with updated query
setwd(here("Chapter1_APCD_Analysis"))
base_term <- "all payers claim database APCD predictive modeling healthcare claims"
filename <- "apcd_analysis_articles.csv"

cat("=== APCD Search (Updated Query) ===\n")
cat("Search term:", base_term, "\n\n")
result <- search_pubmed_all(base_term, filename)
cat(result, "\n")
