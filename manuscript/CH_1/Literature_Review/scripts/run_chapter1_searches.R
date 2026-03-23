# Script to run all Chapter 1 PubMed literature searches
# This script executes the PubMed queries for Chapter 1 topics

# Load required libraries
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
  # Get the current year
  current_year <- format(Sys.Date(), "%Y")
  # Calculate the start year (5 years ago)
  start_year <- as.integer(current_year) - 5
  
  # Modify the query to include the date range
  query <- paste(query, "AND", paste(start_year, ":", current_year, "[PDAT]"))

  initial_search <- entrez_search(db="pubmed", term=query, use_history = TRUE)
  total_count <- initial_search$count
  
  if (total_count == 0) {
    return(paste("No articles found for query:", query))
  }
  
  batch_size <- 50
  all_articles <- list()
  
  # Fetch articles in batches
  for (start in seq(1, total_count, by = batch_size)) {
    tryCatch({
      fetched_articles <- entrez_fetch(db="pubmed", 
                                       web_history = initial_search$web_history, 
                                       retstart = start - 1,  # PubMed uses 0-based indexing
                                       retmax = batch_size,
                                       rettype = "xml")
      # Parse XML to get details
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
  
  # Process and clean the articles
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

  # Save the data frame to a CSV file
  write_csv(all_articles, filename)
  return(paste("Data saved to", filename, "- Found", total_count, "articles"))
}

# Create directories if they don't exist
dirs_to_create <- c("Chapter1_BlackBox_CDS", 
                    "Chapter1_APCD_Analysis", 
                    "Chapter1_Pharmacovigilance", 
                    "Chapter1_Interpretability")

for (dir in dirs_to_create) {
  if (!dir.exists(here(dir))) {
    dir.create(here(dir), recursive = TRUE)
    cat("Created directory:", dir, "\n")
  }
}

# Search 1: Black-Box Machine Learning and Clinical Decision Support
cat("\n=== Search 1: Black-Box ML and Clinical Decision Support ===\n")
setwd(here("Chapter1_BlackBox_CDS"))
base_term1 <- "black box machine learning clinical decision support interpretability explainable AI"
filename1 <- "blackbox_cds_articles.csv"
result1 <- search_pubmed_all(base_term1, filename1)
cat(result1, "\n\n")

# Search 2: All Payer Claims Database Analysis
cat("=== Search 2: APCD Analysis ===\n")
setwd(here("Chapter1_APCD_Analysis"))
base_term2 <- "all payers claim database"
filename2 <- "apcd_analysis_articles.csv"
result2 <- search_pubmed_all(base_term2, filename2)
cat(result2, "\n\n")

# Search 3: Pharmacovigilance
cat("=== Search 3: Pharmacovigilance ===\n")
setwd(here("Chapter1_Pharmacovigilance"))
base_term3 <- "pharmacovigilance pharmacogenomics"
filename3 <- "pharmacovigilance_articles.csv"
result3 <- search_pubmed_all(base_term3, filename3)
cat(result3, "\n\n")

# Search 4: Interpretability Methods
cat("=== Search 4: Interpretability (SHAP/Feature Importance) ===\n")
setwd(here("Chapter1_Interpretability"))
base_term4 <- "SHAP Shapley additive explanations feature importance interpretability healthcare machine learning"
filename4 <- "interpretability_articles.csv"
result4 <- search_pubmed_all(base_term4, filename4)
cat(result4, "\n\n")

cat("=== All searches completed! ===\n")
