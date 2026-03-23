# Script to automatically import all publications to Zotero
# Uses PMC IDs and titles to fetch complete citations and add to Zotero

library(httr)
library(jsonlite)
library(dplyr)
library(readr)
library(here)
library(stringr)
library(xml2)
library(rentrez)

# Zotero API credentials (from your existing scripts)
ZOTERO_USER_ID <- "6037399"
ZOTERO_API_KEY <- "xxjsStqHkKgaSNnzb8FmG3Zb"
ZOTERO_COLLECTION_ID <- "LS75EWXU"  # Optional: specific collection

# Zotero API base URL
ZOTERO_API_BASE <- paste0("https://api.zotero.org/users/", ZOTERO_USER_ID)

# Function to create Zotero item from PubMed data
create_zotero_item <- function(title, authors, year, pmc_id, doi, journal, volume, issue, pages) {
  # Format authors for Zotero (array of creator objects)
  author_list <- str_split(authors, ",\\s*")[[1]]
  creators <- lapply(author_list, function(author) {
    # Try to split first/last name if possible
    parts <- str_split(author, "\\s+")[[1]]
    if (length(parts) >= 2) {
      list(
        creatorType = "author",
        firstName = paste(parts[-length(parts)], collapse = " "),
        lastName = parts[length(parts)]
      )
    } else {
      list(
        creatorType = "author",
        lastName = author
      )
    }
  })
  
  # Build item
  item <- list(
    itemType = "journalArticle",
    title = title,
    creators = creators,
    date = year,
    publicationTitle = journal %||% "",
    volume = volume %||% "",
    issue = issue %||% "",
    pages = pages %||% "",
    DOI = doi %||% "",
    extra = paste0("PMC ID: ", pmc_id %||% "")
  )
  
  # Add URL if PMC ID available
  if (!is.na(pmc_id) && pmc_id != "" && pmc_id != "NA") {
    pmc_clean <- str_replace(pmc_id, "PMC", "")
    item$url <- paste0("https://www.ncbi.nlm.nih.gov/pmc/articles/", pmc_clean, "/")
  }
  
  return(item)
}

# Function to fetch complete citation from PubMed
fetch_pubmed_citation <- function(pmcid_or_title) {
  tryCatch({
    # Try PMC ID first
    if (str_detect(pmcid_or_title, "PMC")) {
      pmc_clean <- str_replace(pmcid_or_title, "PMC", "")
      search_result <- entrez_search(db="pubmed", term=paste0(pmc_clean, "[PMCID]"))
    } else {
      # Search by title
      search_result <- entrez_search(db="pubmed", term=paste0('"', str_sub(pmcid_or_title, 1, 200), '"[Title]'))
    }
    
    if (length(search_result$ids) == 0) {
      return(NULL)
    }
    
    pmid <- search_result$ids[1]
    record <- entrez_fetch(db="pubmed", id=pmid, rettype="xml")
    xml_record <- read_xml(record)
    
    # Extract all fields
    title <- xml_text(xml_find_first(xml_record, ".//ArticleTitle"))
    
    # Authors with full names
    author_nodes <- xml_find_all(xml_record, ".//Author")
    authors <- map_chr(author_nodes, ~{
      last <- xml_text(xml_find_first(.x, ".//LastName"))
      first <- xml_text(xml_find_first(.x, ".//ForeName"))
      if (!is.na(first) && first != "") {
        paste0(last, ", ", first)
      } else {
        last
      }
    })
    authors_str <- paste(authors, collapse = ", ")
    
    journal <- xml_text(xml_find_first(xml_record, ".//Journal/Title"))
    year <- xml_text(xml_find_first(xml_record, ".//PubDate/Year"))
    volume <- xml_text(xml_find_first(xml_record, ".//Volume"))
    issue <- xml_text(xml_find_first(xml_record, ".//Issue"))
    pages <- xml_text(xml_find_first(xml_record, ".//Pagination/MedlinePgn"))
    doi <- xml_text(xml_find_first(xml_record, ".//ArticleId[@IdType='doi']"))
    pmc_id <- xml_text(xml_find_first(xml_record, ".//ArticleId[@IdType='pmc']"))
    if (!is.na(pmc_id) && pmc_id != "") {
      pmc_id <- paste0("PMC", pmc_id)
    }
    
    return(list(
      title = title,
      authors = authors_str,
      journal = journal,
      year = year,
      volume = volume,
      issue = issue,
      pages = pages,
      doi = doi,
      pmc_id = pmc_id
    ))
  }, error = function(e) {
    return(NULL)
  })
}

# Function to add item to Zotero
add_to_zotero <- function(item, collection_id = NULL) {
  # Prepare API endpoint
  endpoint <- paste0(ZOTERO_API_BASE, "/items")
  
  # Prepare headers
  headers <- add_headers(
    "Zotero-API-Key" = ZOTERO_API_KEY,
    "Content-Type" = "application/json"
  )
  
  # Add to collection if specified
  if (!is.null(collection_id)) {
    item$collections <- list(collection_id)
  }
  
  # Convert to JSON
  body <- toJSON(list(item), auto_unbox = TRUE)
  
  # Make request
  response <- POST(
    url = endpoint,
    headers,
    body = body,
    encode = "raw"
  )
  
  if (status_code(response) == 200 || status_code(response) == 201) {
    return(TRUE)
  } else {
    cat("Error adding item:", status_code(response), "-", content(response, "text"), "\n")
    return(FALSE)
  }
}

# Function to check if item already exists in Zotero (by title)
item_exists <- function(title) {
  endpoint <- paste0(ZOTERO_API_BASE, "/items")
  
  headers <- add_headers(
    "Zotero-API-Key" = ZOTERO_API_KEY
  )
  
  # Search for title (simplified - Zotero doesn't have great search API)
  # We'll just try to add and handle duplicates
  return(FALSE)
}

# Function to process a CSV file and import to Zotero
import_csv_to_zotero <- function(csv_path, collection_id = NULL, limit = NULL) {
  cat("\n=== Processing:", csv_path, "===\n")
  
  # Read CSV
  articles <- read_csv(csv_path, show_col_types = FALSE)
  
  if (nrow(articles) == 0) {
    cat("No articles found\n")
    return(NULL)
  }
  
  # Limit if specified
  if (!is.null(limit)) {
    articles <- articles[1:min(limit, nrow(articles)), ]
  }
  
  cat("Found", nrow(articles), "articles\n")
  cat("Importing to Zotero...\n\n")
  
  success_count <- 0
  error_count <- 0
  skipped_count <- 0
  
  for (i in 1:nrow(articles)) {
    cat(sprintf("[%d/%d] ", i, nrow(articles)))
    
    # Get identifier
    identifier <- ifelse(!is.na(articles$pmc_id[i]) && articles$pmc_id[i] != "NA",
                        articles$pmc_id[i],
                        articles$title[i])
    
    # Fetch complete citation
    citation <- fetch_pubmed_citation(identifier)
    
    if (is.null(citation)) {
      # Use basic data from CSV
      citation <- list(
        title = articles$title[i],
        authors = articles$authors[i],
        year = articles$pubdate[i],
        journal = NA,
        volume = NA,
        issue = NA,
        pages = NA,
        doi = NA,
        pmc_id = articles$pmc_id[i]
      )
      cat("Using basic data for:", str_sub(articles$title[i], 1, 50), "...\n")
    } else {
      cat("Fetched:", str_sub(citation$title, 1, 50), "...\n")
    }
    
    # Create Zotero item
    zotero_item <- create_zotero_item(
      title = citation$title,
      authors = citation$authors,
      year = citation$year,
      pmc_id = citation$pmc_id,
      doi = citation$doi,
      journal = citation$journal,
      volume = citation$volume,
      issue = citation$issue,
      pages = citation$pages
    )
    
    # Add to Zotero
    if (add_to_zotero(zotero_item, collection_id)) {
      success_count <- success_count + 1
      cat("  ✓ Added successfully\n")
    } else {
      error_count <- error_count + 1
      cat("  ✗ Failed to add\n")
    }
    
    # Rate limiting
    Sys.sleep(1)  # 1 second between requests (Zotero allows 1 req/sec for free tier)
    
    # Also rate limit PubMed API
    if (!is.null(citation) && i %% 3 == 0) {
      Sys.sleep(0.35)  # PubMed API rate limiting
    }
  }
  
  cat("\n=== Summary ===\n")
  cat("Successfully added:", success_count, "\n")
  cat("Errors:", error_count, "\n")
  cat("Skipped:", skipped_count, "\n\n")
  
  return(list(success = success_count, errors = error_count, skipped = skipped_count))
}

# Main execution
cat("=== Zotero Import Script ===\n")
cat("User ID:", ZOTERO_USER_ID, "\n")
cat("Collection ID:", ZOTERO_COLLECTION_ID, "\n\n")

# Find all CSV files in Chapter 1
chapter1_dirs <- list.dirs(here("data/chapter1"), recursive = TRUE)
csv_files <- list.files(chapter1_dirs, pattern = "\\.csv$", full.names = TRUE, recursive = TRUE)
csv_files <- csv_files[!str_detect(csv_files, "_with_citations\\.csv$")]

cat("Found", length(csv_files), "CSV files\n")
cat("Starting import...\n\n")

# Process each file
total_success <- 0
total_errors <- 0

for (csv_file in csv_files) {
  tryCatch({
    # Process with limit for testing (remove limit for full import)
    result <- import_csv_to_zotero(csv_file, 
                                   collection_id = ZOTERO_COLLECTION_ID,
                                   limit = NULL)  # Set to 5 for testing
    
    if (!is.null(result)) {
      total_success <- total_success + result$success
      total_errors <- total_errors + result$errors
    }
  }, error = function(e) {
    cat("Error processing", csv_file, ":", e$message, "\n\n")
  })
}

cat("\n=== Final Summary ===\n")
cat("Total successfully added:", total_success, "\n")
cat("Total errors:", total_errors, "\n")
cat("\nImport complete! Check your Zotero library.\n")
