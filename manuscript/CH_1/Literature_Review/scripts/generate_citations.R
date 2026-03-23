# Script to generate complete citations from PubMed data
# This script fetches full citation information including journal, volume, DOI, etc.

library(rentrez)
library(dplyr)
library(readr)
library(xml2)
library(purrr)
library(here)
library(stringr)

# Function to fetch complete citation data from PubMed
fetch_complete_citation <- function(pmcid_or_title) {
  tryCatch({
    # Try to search by PMC ID first
    if (str_detect(pmcid_or_title, "PMC")) {
      pmc_clean <- str_replace(pmcid_or_title, "PMC", "")
      search_result <- entrez_search(db="pubmed", term=paste0(pmc_clean, "[PMCID]"))
    } else {
      # Search by title
      search_result <- entrez_search(db="pubmed", term=paste0('"', pmcid_or_title, '"[Title]'))
    }
    
    if (length(search_result$ids) == 0) {
      return(NULL)
    }
    
    pmid <- search_result$ids[1]
    
    # Fetch full record
    record <- entrez_fetch(db="pubmed", id=pmid, rettype="xml")
    xml_record <- read_xml(record)
    
    # Extract citation information
    title <- xml_text(xml_find_first(xml_record, ".//ArticleTitle"))
    
    # Authors with first names
    author_nodes <- xml_find_all(xml_record, ".//Author")
    authors <- map_chr(author_nodes, ~{
      last <- xml_text(xml_find_first(.x, ".//LastName"))
      first <- xml_text(xml_find_first(.x, ".//ForeName"))
      initials <- xml_text(xml_find_first(.x, ".//Initials"))
      if (!is.na(first) && first != "") {
        paste0(last, ", ", str_sub(first, 1, 1), ".")
      } else if (!is.na(initials) && initials != "") {
        paste0(last, ", ", initials)
      } else {
        last
      }
    })
    authors_str <- paste(authors, collapse = ", ")
    if (length(authors) > 6) {
      authors_str <- paste0(paste(authors[1:6], collapse = ", "), ", et al.")
    }
    
    # Journal information
    journal <- xml_text(xml_find_first(xml_record, ".//Journal/Title"))
    journal_abbr <- xml_text(xml_find_first(xml_record, ".//Journal/ISOAbbreviation"))
    
    # Publication date
    pub_date <- xml_find_first(xml_record, ".//PubDate")
    year <- xml_text(xml_find_first(pub_date, ".//Year"))
    month <- xml_text(xml_find_first(pub_date, ".//Month"))
    day <- xml_text(xml_find_first(pub_date, ".//Day"))
    
    # Volume and issue
    volume <- xml_text(xml_find_first(xml_record, ".//Volume"))
    issue <- xml_text(xml_find_first(xml_record, ".//Issue"))
    
    # Pages
    pages <- xml_text(xml_find_first(xml_record, ".//Pagination/MedlinePgn"))
    
    # DOI
    doi <- xml_text(xml_find_first(xml_record, ".//ArticleId[@IdType='doi']"))
    
    # PMC ID
    pmc_id <- xml_text(xml_find_first(xml_record, ".//ArticleId[@IdType='pmc']"))
    if (!is.na(pmc_id) && pmc_id != "") {
      pmc_id <- paste0("PMC", pmc_id)
    }
    
    # PubMed ID
    pmid <- xml_text(xml_find_first(xml_record, ".//ArticleId[@IdType='pubmed']"))
    
    # Generate APA citation
    apa_citation <- paste0(
      authors_str, " (", year, "). ",
      title, ". ",
      ifelse(!is.na(journal_abbr), journal_abbr, journal),
      ifelse(!is.na(volume), paste0(", ", volume), ""),
      ifelse(!is.na(issue), paste0("(", issue, ")"), ""),
      ifelse(!is.na(pages), paste0(", ", pages), ""),
      ifelse(!is.na(doi), paste0(". https://doi.org/", doi), "")
    )
    
    # Generate BibTeX key
    first_author <- authors[1]
    bibtex_key <- paste0(
      str_replace_all(tolower(first_author), "[^a-z]", ""),
      year,
      str_replace_all(tolower(str_sub(title, 1, 3)), "[^a-z]", "")
    )
    
    return(list(
      title = title,
      authors_full = authors_str,
      authors_list = paste(authors, collapse = "; "),
      journal = journal,
      journal_abbr = journal_abbr,
      year = year,
      month = month,
      day = day,
      volume = volume,
      issue = issue,
      pages = pages,
      doi = doi,
      pmc_id = pmc_id,
      pmid = pmid,
      apa_citation = apa_citation,
      bibtex_key = bibtex_key
    ))
  }, error = function(e) {
    return(NULL)
  })
}

# Function to process a CSV file and add citations
add_citations_to_csv <- function(csv_path) {
  cat("Processing:", csv_path, "\n")
  
  # Read the CSV
  articles <- read_csv(csv_path, show_col_types = FALSE)
  
  if (nrow(articles) == 0) {
    cat("  No articles found\n")
    return(NULL)
  }
  
  cat("  Found", nrow(articles), "articles\n")
  cat("  Fetching citations...\n")
  
  # Fetch citations (with rate limiting)
  citations_list <- list()
  for (i in 1:nrow(articles)) {
    if (i %% 10 == 0) {
      cat("    Processing article", i, "of", nrow(articles), "\n")
    }
    
    # Try PMC ID first, then title
    identifier <- ifelse(!is.na(articles$pmc_id[i]) && articles$pmc_id[i] != "NA",
                        articles$pmc_id[i],
                        articles$title[i])
    
    citation <- fetch_complete_citation(identifier)
    
    if (!is.null(citation)) {
      citations_list[[i]] <- citation
    } else {
      # Create minimal citation from available data
      citations_list[[i]] <- list(
        title = articles$title[i],
        authors_full = articles$authors[i],
        journal = NA,
        year = articles$pubdate[i],
        doi = NA,
        pmc_id = articles$pmc_id[i],
        pmid = NA,
        apa_citation = paste0(articles$authors[i], " (", articles$pubdate[i], "). ", articles$title[i], "."),
        bibtex_key = NA
      )
    }
    
    # Rate limiting - be nice to PubMed API
    Sys.sleep(0.35)  # ~3 requests per second
  }
  
  # Convert to data frame
  citations_df <- bind_rows(citations_list)
  
  # Combine with original data
  result <- bind_cols(articles, citations_df)
  
  # Save to new file
  output_path <- str_replace(csv_path, "\\.csv$", "_with_citations.csv")
  write_csv(result, output_path)
  cat("  Saved citations to:", output_path, "\n\n")
  
  return(result)
}

# Process all CSV files in Chapter 1
cat("=== Generating Citations for Chapter 1 Articles ===\n\n")

chapter1_dirs <- list.dirs(here("data/chapter1"), recursive = TRUE)
csv_files <- list.files(chapter1_dirs, pattern = "\\.csv$", full.names = TRUE, recursive = TRUE)
csv_files <- csv_files[!str_detect(csv_files, "_with_citations\\.csv$")]  # Skip already processed files

cat("Found", length(csv_files), "CSV files to process\n\n")

# Process each file
for (csv_file in csv_files) {
  tryCatch({
    add_citations_to_csv(csv_file)
  }, error = function(e) {
    cat("Error processing", csv_file, ":", e$message, "\n\n")
  })
}

cat("=== Citation Generation Complete ===\n")
