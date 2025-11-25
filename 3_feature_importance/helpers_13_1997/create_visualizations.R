#!/usr/bin/env Rscript
# Feature Importance Visualizations for MC-CV Classification
# Uses Recall-scaled importance from CatBoost + Random Forest

library(tidyverse)
library(ggplot2)
library(here)

#' Create visualizations from aggregated feature importance
#' 
#' @param aggregated_file Path to aggregated feature importance CSV
#' @param output_dir Directory to save plots
#' @param s3_upload Whether to upload plots to S3
#' @param cohort_name Cohort name for S3 path
#' @param age_band Age band for S3 path  
#' @param event_year Event year for S3 path
#' @return List of plot file paths
create_feature_importance_plots <- function(aggregated_file, 
                                           output_dir = NULL,
                                           s3_upload = TRUE,
                                           cohort_name = NULL,
                                           age_band = NULL,
                                           event_year = NULL) {
  
  # Determine output directory
  if (is.null(output_dir)) {
    if (dir.exists("outputs")) {
      output_dir <- "outputs"
    } else if (dir.exists(here("outputs"))) {
      output_dir <- here("outputs")
    } else {
      stop("Cannot find outputs directory")
    }
  }
  
  plot_dir <- file.path(output_dir, "plots")
  dir.create(plot_dir, showWarnings = FALSE, recursive = TRUE)
  
  cat("Reading aggregated feature importance...\n")
  if (!file.exists(aggregated_file)) {
    stop(sprintf("File not found: %s", aggregated_file))
  }
  
  features_df <- readr::read_csv(aggregated_file, show_col_types = FALSE)
  cat(sprintf("✓ Loaded %d features\n", nrow(features_df)))
  
  # Extract cohort info from filename if not provided
  if (is.null(cohort_name) || is.null(age_band) || is.null(event_year)) {
    basename_parts <- strsplit(basename(aggregated_file), "_")[[1]]
    if (is.null(cohort_name)) cohort_name <- basename_parts[1]
    if (is.null(age_band)) age_band <- basename_parts[2]
    if (is.null(event_year)) event_year <- as.integer(sub("\\.csv$", "", basename_parts[4]))
  }
  
  plot_files <- list()
  
  # ============================================================================
  # PLOT 1: Top 50 Features Bar Chart (Scaled Importance)
  # ============================================================================
  cat("\nCreating top 50 features bar chart...\n")
  
  top50 <- features_df %>%
    head(50) %>%
    mutate(feature = factor(feature, levels = rev(feature)))
  
  p1 <- ggplot(top50, aes(x = feature, y = importance_scaled)) +
    geom_bar(stat = "identity", fill = "steelblue", alpha = 0.8) +
    coord_flip() +
    labs(
      title = sprintf("Top 50 Features by Scaled Importance\n%s / %s / %d", 
                      cohort_name, age_band, event_year),
      subtitle = sprintf("Importance scaled by MC-CV Recall (mean=%.3f)", 
                        features_df$recall_mean[1]),
      x = "Feature",
      y = "Scaled Importance"
    ) +
    theme_minimal() +
    theme(
      plot.title = element_text(size = 14, face = "bold"),
      plot.subtitle = element_text(size = 11),
      axis.text.y = element_text(size = 8)
    )
  
  plot_file <- file.path(plot_dir, 
                         sprintf("%s_%s_%d_top50_features.png", 
                                cohort_name, age_band, event_year))
  ggplot2::ggsave(plot_file, p1, width = 12, height = 14, dpi = 300)
  cat(sprintf("✓ Saved: %s\n", basename(plot_file)))
  plot_files$top50_bar <- plot_file
  
  # ============================================================================
  # PLOT 2: Top 50 Features with Recall Confidence and Scaled Importance
  # ============================================================================
  cat("\nCreating top 50 features with Recall confidence and scaled importance...\n")
  
  top50_recall <- features_df %>%
    head(50) %>%
    mutate(
      feature = factor(feature, levels = rev(feature)),
      recall_lower = recall_mean - 1.96 * recall_std,  # 95% CI
      recall_upper = recall_mean + 1.96 * recall_std
    )
  
  p2 <- ggplot(top50_recall, aes(x = feature, y = importance_scaled)) +
    geom_bar(stat = "identity", aes(fill = recall_mean), alpha = 0.8) +
    geom_errorbar(aes(ymin = 0, ymax = 0), width = 0) +  # Placeholder for legend
    scale_fill_gradient(low = "orange", high = "darkblue", 
                        name = "MC-CV\nRecall",
                        limits = c(min(top50_recall$recall_lower, na.rm = TRUE),
                                   max(top50_recall$recall_upper, na.rm = TRUE))) +
    coord_flip() +
    labs(
      title = sprintf("Top 50 Features: Scaled Importance with Recall Confidence\n%s / %s / %d", 
                      cohort_name, age_band, event_year),
      subtitle = sprintf("Bar height = scaled importance | Color = MC-CV Recall (mean ± 1.96 SD) | Mean Recall: %.3f",
                        mean(top50_recall$recall_mean)),
      x = "Feature",
      y = "Scaled Importance"
    ) +
    theme_minimal() +
    theme(
      plot.title = element_text(size = 14, face = "bold"),
      plot.subtitle = element_text(size = 10),
      axis.text.y = element_text(size = 8),
      legend.position = "right"
    )
  
  plot_file <- file.path(plot_dir,
                         sprintf("%s_%s_%d_top50_with_recall.png",
                                cohort_name, age_band, event_year))
  ggplot2::ggsave(plot_file, p2, width = 12, height = 14, dpi = 300)
  cat(sprintf("✓ Saved: %s\n", basename(plot_file)))
  plot_files$top50_recall <- plot_file
  
  # ============================================================================
  # PLOT 3: Normalized vs Scaled Importance (Top 50)
  # ============================================================================
  cat("\nCreating normalized vs scaled comparison...\n")
  
  top50_comparison <- features_df %>%
    head(50) %>%
    mutate(feature = factor(feature, levels = rev(feature))) %>%
    pivot_longer(
      cols = c(importance_normalized, importance_scaled),
      names_to = "importance_type",
      values_to = "value"
    ) %>%
    mutate(
      importance_type = case_when(
        importance_type == "importance_normalized" ~ "Normalized",
        importance_type == "importance_scaled" ~ "Scaled by Recall"
      )
    )
  
  p3 <- ggplot(top50_comparison, aes(x = feature, y = value, fill = importance_type)) +
    geom_bar(stat = "identity", position = "dodge", alpha = 0.8) +
    scale_fill_manual(values = c("Normalized" = "gray70", 
                                  "Scaled by Recall" = "steelblue"),
                      name = "Importance Type") +
    coord_flip() +
    labs(
      title = sprintf("Normalized vs Recall-Scaled Importance (Top 50)\n%s / %s / %d",
                      cohort_name, age_band, event_year),
      subtitle = "Shows impact of model quality weighting on feature importance",
      x = "Feature",
      y = "Importance Value"
    ) +
    theme_minimal() +
    theme(
      plot.title = element_text(size = 14, face = "bold"),
      plot.subtitle = element_text(size = 11),
      axis.text.y = element_text(size = 8),
      legend.position = "bottom"
    )
  
  plot_file <- file.path(plot_dir,
                         sprintf("%s_%s_%d_normalized_vs_scaled.png",
                                cohort_name, age_band, event_year))
  ggplot2::ggsave(plot_file, p3, width = 12, height = 14, dpi = 300)
  cat(sprintf("✓ Saved: %s\n", basename(plot_file)))
  plot_files$normalized_vs_scaled <- plot_file
  
  # ============================================================================
  # PLOT 4: Feature Categories Distribution (Top 50)
  # ============================================================================
  cat("\nCreating feature categories distribution...\n")
  
  top50_with_categories <- top50 %>%
    mutate(
      category = case_when(
        grepl("^[A-Z][0-9]{2}\\.", feature) ~ "ICD Code",
        grepl("^[0-9]{5}$", feature) ~ "CPT Code",
        TRUE ~ "Drug Name"
      )
    )
  
  p4 <- ggplot(top50_with_categories, aes(x = category, fill = category)) +
    geom_bar(alpha = 0.8) +
    scale_fill_manual(values = c(
      "Drug Name" = "steelblue",
      "ICD Code" = "darkgreen", 
      "CPT Code" = "darkorange"
    )) +
    labs(
      title = sprintf("Feature Category Distribution (Top 50)\n%s / %s / %d",
                      cohort_name, age_band, event_year),
      subtitle = "Shows which item types are most important",
      x = "Feature Category",
      y = "Count"
    ) +
    theme_minimal() +
    theme(
      plot.title = element_text(size = 14, face = "bold"),
      plot.subtitle = element_text(size = 11),
      legend.position = "none"
    )
  
  plot_file <- file.path(plot_dir,
                         sprintf("%s_%s_%d_category_distribution.png",
                                cohort_name, age_band, event_year))
  ggplot2::ggsave(plot_file, p4, width = 10, height = 6, dpi = 300)
  cat(sprintf("✓ Saved: %s\n", basename(plot_file)))
  plot_files$category_dist <- plot_file
  
  # ============================================================================
  # Upload to S3 if requested
  # ============================================================================
  if (s3_upload) {
    cat("\n" , rep("=", 70), "\n", sep="")
    cat("Uploading plots to S3...\n")
    cat(rep("=", 70), "\n", sep="")
    
    # Find AWS CLI
    aws_cmd <- Sys.which("aws")
    if (aws_cmd == "") {
      aws_paths <- c("/usr/local/bin/aws", "/usr/bin/aws", 
                    "/home/ec2-user/.local/bin/aws")
      for (path in aws_paths) {
        if (file.exists(path)) {
          aws_cmd <- path
          break
        }
      }
    }
    
    if (!is.null(aws_cmd) && aws_cmd != "") {
      s3_base <- sprintf("s3://pgxdatalake/gold/feature_importance/cohort_name=%s/age_band=%s/event_year=%d/plots",
                        cohort_name, age_band, event_year)
      
      upload_count <- 0
      for (plot_name in names(plot_files)) {
        local_file <- plot_files[[plot_name]]
        s3_path <- sprintf("%s/%s", s3_base, basename(local_file))
        
        upload_cmd <- sprintf('"%s" s3 cp "%s" "%s"', aws_cmd, local_file, s3_path)
        result <- system(upload_cmd, intern = FALSE, ignore.stdout = TRUE, ignore.stderr = TRUE)
        
        if (result == 0) {
          cat(sprintf("✓ Uploaded: %s\n", basename(local_file)))
          upload_count <- upload_count + 1
        } else {
          warning(sprintf("Failed to upload: %s", basename(local_file)))
        }
      }
      
      if (upload_count == length(plot_files)) {
        cat(sprintf("\n✓ All %d plots uploaded to S3\n", upload_count))
        cat(sprintf("Location: %s\n", s3_base))
      }
    } else {
      cat("⚠ AWS CLI not found. Plots saved locally only.\n")
      cat(sprintf("To upload manually:\n"))
      cat(sprintf("  aws s3 sync %s s3://pgxdatalake/gold/feature_importance/cohort_name=%s/age_band=%s/event_year=%d/plots/\n",
                 plot_dir, cohort_name, age_band, event_year))
    }
  }
  
  cat("\n", rep("=", 70), "\n", sep="")
  cat("Visualization Summary\n")
  cat(rep("=", 70), "\n", sep="")
  cat(sprintf("Local plots directory: %s\n", normalizePath(plot_dir)))
  cat(sprintf("Plots created: %d\n", length(plot_files)))
  for (plot_name in names(plot_files)) {
    cat(sprintf("  - %s\n", basename(plot_files[[plot_name]])))
  }
  
  return(invisible(plot_files))
}

# Run when executed as standalone script
if (!interactive()) {
  # Look for aggregated file in outputs directory
  output_dir <- if (dir.exists("outputs")) "outputs" else here("outputs")
  agg_files <- list.files(output_dir, pattern = "*_feature_importance_aggregated.csv$",
                          full.names = TRUE)
  
  if (length(agg_files) == 0) {
    stop("No aggregated feature importance files found in outputs directory")
  }
  
  # Use the most recently modified file
  latest_file <- agg_files[which.max(file.mtime(agg_files))]
  cat(sprintf("Using: %s\n\n", basename(latest_file)))
  
  create_feature_importance_plots(latest_file, output_dir = output_dir)
}
