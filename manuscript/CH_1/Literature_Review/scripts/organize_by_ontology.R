library(dplyr)
library(readr)
library(stringr)
library(here)
library(purrr)
library(yaml)
library(jsonlite)
library(tidyr)

# ─────────────────────────────────────────────────────────────────────────────
# organize_by_ontology.R
# Tags every article CSV row and downloaded JSON file with ontology nodes
# from scripts/keyword_ontology.yaml.
#
# Outputs:
#   data/ontology/articles_tagged.csv      — all articles with ontology columns
#   data/ontology/ontology_summary.csv     — article count per node
#   data/ontology/<domain>/<node>/         — symlinked/copied JSON files
#   data/ontology/ontology_index.json      — machine-readable index
# ─────────────────────────────────────────────────────────────────────────────

here::i_am("lit_review.qmd")

# ── 1. Load ontology ──────────────────────────────────────────────────────────
ontology_path <- here("scripts", "keyword_ontology.yaml")
if (!file.exists(ontology_path)) stop("keyword_ontology.yaml not found.")
ontology_raw <- yaml::read_yaml(ontology_path)

# Flatten ontology into a lookup table: node_id, domain, label, terms, rq, aims, searches
flatten_ontology <- function(ont) {
  rows <- list()
  for (domain_name in names(ont)) {
    domain <- ont[[domain_name]]
    for (node_name in names(domain)) {
      node <- domain[[node_name]]
      if (!is.list(node) || is.null(node$terms)) next
      node_label   <- node[["label"]]      %||% node_name
      node_ooda    <- node[["ooda_phase"]] %||% NA_character_
      node_terms   <- list(unlist(node[["terms"]]))
      node_rq      <- list(unlist(node[["rq"]]      %||% list()))
      node_aims    <- list(unlist(node[["aims"]]    %||% list()))
      node_srch    <- list(unlist(node[["searches"]]%||% list()))

      rows[[length(rows) + 1]] <- tibble(
        domain      = domain_name,
        node        = node_name,
        label       = node_label,
        ooda_phase  = node_ooda,
        terms       = node_terms,
        rq          = node_rq,
        aims        = node_aims,
        searches    = node_srch
      )
    }
  }
  bind_rows(rows)
}

`%||%` <- function(a, b) if (!is.null(a)) a else b

# Exclude top-level YAML keys that are not domain nodes (causal_links, ooda_loop)
ont_df <- flatten_ontology(ontology_raw[!names(ontology_raw) %in% c("causal_links", "ooda_loop")])
message(sprintf("Ontology loaded: %d nodes across %d domains", nrow(ont_df),
                length(setdiff(names(ontology_raw), c("causal_links", "ooda_loop")))))

# ── 2. Load all article CSVs (same manifest as prisma_tracker.R) ─────────────
csv_manifest <- tribble(
  ~rel_path,                                                                           ~search_num,
  "data/chapter1/1.1_introduction/blackbox_cds/blackbox_cds_articles.csv",            1L,
  "data/chapter1/1.3_methodological/apcd_analysis/apcd_analysis_articles.csv",        2L,
  "data/chapter1/1.2_clinical_background/pharmacovigilance/pharmacovigilance_articles.csv", 3L,
  "data/chapter1/1.1_introduction/interpretability/interpretability_articles.csv",    4L,
  "data/chapter1/1.3_methodological/pattern_mining/fpgrowth/fpgrowth_articles.csv",  5L,
  "data/chapter1/1.3_methodological/pattern_mining/process_mining/process_mining_articles.csv", 6L,
  "data/chapter1/1.4_technical/catboost_xgboost/catboost_xgboost_articles.csv",      7L,
  "data/chapter1/1.3_methodological/pattern_mining/dtw/dtw_articles.csv",            8L,
  "data/chapter1/1.3_methodological/temporal_causality/temporal_causality_articles.csv", 9L,
  "data/chapter1/1.3_methodological/target_leakage/target_leakage_articles.csv",     10L,
  "data/chapter1/1.2_clinical_background/opioid_disorder/opioid_disorder_articles.csv", 11L,
  "data/chapter1/1.2_clinical_background/polypharmacy/polypharmacy_articles.csv",    12L,
  "data/chapter1/1.2_clinical_background/drug_interactions/drug_interactions_articles.csv", 13L,
  "data/chapter1/1.4_technical/duckdb_olap/duckdb_articles.csv",                     14L,
  "data/chapter1/1.2_clinical_background/opioid_disorder/cpt_opioid/cpt_opioid_articles.csv", 15L,
  "data/chapter1/1.2_clinical_background/opioid_disorder/opioid_ed_prediction/opioid_ed_prediction_articles.csv", 16L,
  "data/chapter1/1.2_clinical_background/drug_interactions/polypharmacy_ed/polypharmacy_ed_articles.csv", 17L,
  "data/chapter1/1.3_methodological/routine_care/routine_care_articles.csv",         18L,
  "data/other_chapters/pgx_models/pgx_risk_classifcation_articles.csv",              NA_integer_,
  "data/other_chapters/ehr_models/risk_model_ehr_articles.csv",                      NA_integer_,
  "data/other_chapters/fhir_models/fhir_ehr_articles.csv",                           NA_integer_
)

load_csv_safe <- function(rel_path, search_num) {
  full <- here(rel_path)
  if (!file.exists(full)) { message("  [MISSING] ", rel_path); return(NULL) }
  df <- read_csv(full, show_col_types = FALSE)
  df$search_num  <- search_num
  df$source_file <- rel_path
  df
}

all_articles <- bind_rows(pmap(csv_manifest, load_csv_safe))

if (nrow(all_articles) == 0) stop("No article CSVs found. Run search chunks first.")

# Deduplicate on normalised title
all_articles <- all_articles %>%
  mutate(title_norm = str_trim(str_to_lower(coalesce(title, "")))) %>%
  filter(title_norm != "") %>%
  distinct(title_norm, .keep_all = TRUE) %>%
  mutate(article_id = row_number())

message(sprintf("Articles loaded: %d (after dedup)", nrow(all_articles)))

# ── 3. Tag each article with matching ontology nodes ─────────────────────────
tag_article <- function(title_norm, search_num) {
  matched_nodes <- character(0)
  for (i in seq_len(nrow(ont_df))) {
    terms    <- ont_df$terms[[i]]
    node_key <- paste0(ont_df$domain[i], "::", ont_df$node[i])
    # Match on title
    if (any(str_detect(title_norm, fixed(str_to_lower(terms), ignore_case = FALSE)))) {
      matched_nodes <- c(matched_nodes, node_key)
      next
    }
    # Also match by search number if no title hit
    node_searches <- as.integer(unlist(ont_df$searches[[i]]))
    if (!is.na(search_num) && search_num %in% node_searches) {
      matched_nodes <- c(matched_nodes, node_key)
    }
  }
  if (length(matched_nodes) == 0) matched_nodes <- "unclassified::unclassified"
  paste(unique(matched_nodes), collapse = "|")
}

message("Tagging articles with ontology nodes...")
all_articles <- all_articles %>%
  mutate(
    ontology_nodes = map2_chr(title_norm, search_num, tag_article),
    domain_primary = str_extract(ontology_nodes, "^[^:]+"),
    node_primary   = str_extract(ontology_nodes, "::([^|]+)", group = 1)
  )

# ── 4. Join ontology labels onto articles ────────────────────────────────────
ont_labels <- ont_df %>%
  mutate(node_key = paste0(domain, "::", node)) %>%
  select(node_key, label, ooda_phase, rq, aims)

# Explode multi-node articles for summary counts
articles_exploded <- all_articles %>%
  select(article_id, title, pubdate, pmc_id, ontology_nodes) %>%
  mutate(node_key = str_split(ontology_nodes, "\\|")) %>%
  unnest(node_key) %>%
  left_join(ont_labels, by = "node_key")

# Derive primary OODA phase from primary node
ooda_lookup <- ont_df %>%
  mutate(node_key = paste0(domain, "::", node)) %>%
  select(node_key, ooda_phase)

all_articles <- all_articles %>%
  mutate(
    primary_node_key  = paste0(domain_primary, "::", node_primary),
    ooda_phase_primary = ooda_lookup$ooda_phase[
      match(primary_node_key, ooda_lookup$node_key)
    ]
  )

# ── 5. Save tagged articles CSV ───────────────────────────────────────────────
dir.create(here("data/ontology"), recursive = TRUE, showWarnings = FALSE)

write_csv(
  all_articles %>%
    select(article_id, title, pubdate, pmc_id, authors,
           search_num, source_file, ontology_nodes,
           domain_primary, node_primary, ooda_phase_primary),
  here("data/ontology", "articles_tagged.csv")
)

# ── 6. Ontology summary: article count per node ───────────────────────────────
ontology_summary <- articles_exploded %>%
  filter(node_key != "unclassified::unclassified") %>%
  group_by(node_key, label, ooda_phase) %>%
  summarise(
    n_articles  = n_distinct(article_id),
    rq_tags     = paste(unique(unlist(rq)),   collapse = ", "),
    aim_tags    = paste(unique(unlist(aims)),  collapse = ", "),
    .groups = "drop"
  ) %>%
  arrange(
    factor(ooda_phase, levels = c("observe", "orient", "decide", "act")),
    desc(n_articles)
  )

write_csv(ontology_summary, here("data/ontology", "ontology_summary.csv"))

# OODA phase summary (article counts per phase)
ooda_summary <- all_articles %>%
  filter(!is.na(ooda_phase_primary), ooda_phase_primary != "unclassified") %>%
  count(ooda_phase_primary, name = "n_articles") %>%
  mutate(ooda_phase_primary = factor(
    ooda_phase_primary, levels = c("observe", "orient", "decide", "act")
  )) %>%
  arrange(ooda_phase_primary)

write_csv(ooda_summary, here("data/ontology", "ooda_phase_summary.csv"))

unclassified_count <- sum(all_articles$domain_primary == "unclassified", na.rm = TRUE)

# ── 7. Organise JSON files by ontology node ───────────────────────────────────
# For each article with a downloaded JSON, copy it into:
#   data/ontology/<domain>/<node>/<pmc_id>.json
#
# JSON source dirs follow the pattern: <csv_dir>/pubmed_json_files/

json_source_dirs <- csv_manifest %>%
  mutate(json_dir = here(str_replace(rel_path, "/[^/]+\\.csv$", "/pubmed_json_files"))) %>%
  filter(map_lgl(json_dir, dir.exists))

# Build pmc_id → json_path lookup
build_json_index <- function(json_dir) {
  files <- list.files(json_dir, pattern = "\\.json$", full.names = TRUE)
  tibble(
    pmc_id    = tools::file_path_sans_ext(basename(files)),
    json_path = files
  )
}

json_index <- bind_rows(map(json_source_dirs$json_dir, build_json_index)) %>%
  distinct(pmc_id, .keep_all = TRUE)

message(sprintf("JSON files found: %d", nrow(json_index)))

# Join articles to their JSON paths
articles_with_json <- all_articles %>%
  inner_join(json_index, by = "pmc_id")

# Copy JSON files into ontology directories (by domain/node AND by ooda_phase)
copy_to_ontology <- function(json_path, dest_dir, pmc_id) {
  dir.create(dest_dir, recursive = TRUE, showWarnings = FALSE)
  dest_file <- file.path(dest_dir, paste0(pmc_id, ".json"))
  if (!file.exists(dest_file)) file.copy(json_path, dest_file)
}

message("Copying JSON files into ontology directories...")
walk(seq_len(nrow(articles_with_json)), function(i) {
  row <- articles_with_json[i, ]
  if (is.na(row$domain_primary) || row$domain_primary == "unclassified") return()
  # domain/node directory
  copy_to_ontology(
    row$json_path,
    here("data/ontology", row$domain_primary, row$node_primary),
    row$pmc_id
  )
  # ooda_phase directory (parallel structure)
  if (!is.na(row$ooda_phase_primary)) {
    copy_to_ontology(
      row$json_path,
      here("data/ontology", "ooda", row$ooda_phase_primary),
      row$pmc_id
    )
  }
})

# ── 8. Write machine-readable ontology index JSON ────────────────────────────
ontology_index <- ont_df %>%
  mutate(
    node_key   = paste0(domain, "::", node),
    n_articles = map_int(node_key, function(k) {
      sum(str_detect(all_articles$ontology_nodes, fixed(k)), na.rm = TRUE)
    })
  ) %>%
  select(domain, node, label, ooda_phase, rq, aims, searches, n_articles) %>%
  group_by(domain) %>%
  group_split() %>%
  set_names(map_chr(., ~unique(.x$domain))) %>%
  map(~select(.x, -domain))

write_json(ontology_index, here("data/ontology", "ontology_index.json"),
           pretty = TRUE, auto_unbox = TRUE)

# ── 9. Console summary ────────────────────────────────────────────────────────
cat("\n── Ontology Tagging Summary ───────────────────────────────\n")
cat(sprintf("  Total articles tagged:          %6d\n", nrow(all_articles)))
cat(sprintf("  Unclassified (no term match):   %6d\n", unclassified_count))
cat(sprintf("  JSON files organised:           %6d\n", nrow(articles_with_json)))
cat("\n── Top Ontology Nodes by Article Count ────────────────────\n")
print(head(ontology_summary, 15), n = 15)
cat("\n── Articles by OODA Phase ───────────────────────────────────\n")
print(ooda_summary, n = Inf)
cat("\n── Outputs written ────────────────────────────────────────\n")
cat("  data/ontology/articles_tagged.csv       (+ ooda_phase_primary col)\n")
cat("  data/ontology/ontology_summary.csv      (+ ooda_phase col, sorted)\n")
cat("  data/ontology/ooda_phase_summary.csv    (counts per OODA phase)\n")
cat("  data/ontology/ontology_index.json\n")
cat("  data/ontology/<domain>/<node>/*.json\n")
cat("  data/ontology/ooda/<phase>/*.json\n\n")
