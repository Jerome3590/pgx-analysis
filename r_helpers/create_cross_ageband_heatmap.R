#!/usr/bin/env Rscript
# Cross-Age-Band Feature Importance Heatmap
# Shows how feature importance changes across age bands for a cohort

library(tidyverse)
library(ggplot2)
library(here)

#' Create heatmap comparing feature importance across age bands
#' 
#' @param cohort_name Cohort name (e.g., "opioid_ed")
#' @param event_year Event year (e.g., 2016)
#' @param age_bands Vector of age bands to compare (e.g., c("13-24", "25-44", "45-54", "55-64", "65-74"))
#' @param output_dir Directory containing feature importance files
#' @param s3_upload Whether to upload heatmap to S3
#' @param top_n Number of top features to show in heatmap (default: 50)
#' @return Plot file path
create_ageband_heatmap <- function(cohort_name,
                                   event_year,
                                   age_bands = c("0-12", "13-24", "25-44", "45-54", "55-64", "65-74", "75+"),
                                   output_dir = NULL,
                                   s3_upload = TRUE,
                                   top_n = 50) {
  
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
  
  cat("\n", rep("=", 80), "\n", sep="")
  cat(sprintf("Creating Age Band Heatmap: %s / %d\n", cohort_name, event_year))
  cat(rep("=", 80), "\n", sep="")
  
  # Load feature importance files for each age band
  cat("\nLoading feature importance files...\n")
  
  all_features <- list()
  loaded_age_bands <- c()
  
  for (age_band in age_bands) {
    file_pattern <- sprintf("%s_%s_%d_feature_importance_aggregated.csv",
                           cohort_name, age_band, event_year)
    file_path <- file.path(output_dir, file_pattern)
    
    if (file.exists(file_path)) {
      df <- readr::read_csv(file_path, show_col_types = FALSE) %>%
        select(feature, importance_scaled, recall_mean) %>%
        mutate(age_band = age_band)
      
      all_features[[age_band]] <- df
      loaded_age_bands <- c(loaded_age_bands, age_band)
      cat(sprintf("  ✓ Loaded: %s (%d features)\n", age_band, nrow(df)))
    } else {
      cat(sprintf("  ⚠ Not found: %s\n", age_band))
    }
  }
  
  if (length(loaded_age_bands) < 2) {
    stop("Need at least 2 age bands to create comparison heatmap")
  }
  
  # Combine all features
  combined <- bind_rows(all_features)
  cat(sprintf("\nCombined %d age bands with %d total feature records\n",
              length(loaded_age_bands), nrow(combined)))
  
  # Get union of top N features across all age bands
  top_features_per_band <- combined %>%
    group_by(age_band) %>%
    arrange(desc(importance_scaled)) %>%
    slice_head(n = top_n) %>%
    ungroup() %>%
    pull(feature) %>%
    unique()
  
  cat(sprintf("Identified %d unique features in top %d across age bands\n",
              length(top_features_per_band), top_n))
  
  # Create feature matrix (feature × age_band)
  feature_matrix <- combined %>%
    filter(feature %in% top_features_per_band) %>%
    select(feature, age_band, importance_scaled) %>%
    pivot_wider(
      names_from = age_band,
      values_from = importance_scaled,
      values_fill = 0  # Features not in top N for an age band get 0
    )
  
  # Calculate average importance across age bands to order features
  feature_order <- feature_matrix %>%
    mutate(avg_importance = rowMeans(select(., -feature), na.rm = TRUE)) %>%
    arrange(desc(avg_importance)) %>%
    pull(feature)
  
  # Reshape for plotting
  heatmap_data <- feature_matrix %>%
    pivot_longer(
      cols = -feature,
      names_to = "age_band",
      values_to = "importance_scaled"
    ) %>%
    mutate(
      feature = factor(feature, levels = rev(feature_order)),
      age_band = factor(age_band, levels = loaded_age_bands)
    )
  
  # ============================================================================
  # PLOT: Feature Importance Heatmap Across Age Bands
  # ============================================================================
  cat("\nCreating heatmap...\n")
  
  p <- ggplot(heatmap_data, aes(x = age_band, y = feature, fill = importance_scaled)) +
    geom_tile(color = "white", linewidth = 0.5) +
    scale_fill_gradient2(
      low = "white",
      mid = "orange", 
      high = "darkblue",
      midpoint = median(heatmap_data$importance_scaled[heatmap_data$importance_scaled > 0]),
      name = "Scaled\nImportance"
    ) +
    labs(
      title = sprintf("Feature Importance Across Age Bands\n%s / %d",
                      cohort_name, event_year),
      subtitle = sprintf("Top %d features (union across all age bands) | Color: white (absent) → orange (medium) → dark blue (high)",
                        top_n),
      x = "Age Band",
      y = "Feature"
    ) +
    theme_minimal() +
    theme(
      plot.title = element_text(size = 14, face = "bold"),
      plot.subtitle = element_text(size = 10),
      axis.text.x = element_text(angle = 45, hjust = 1, size = 10),
      axis.text.y = element_text(size = 7),
      axis.title = element_text(size = 11),
      legend.position = "right",
      panel.grid = element_blank()
    )
  
  # Adjust height based on number of features
  plot_height <- max(12, length(feature_order) * 0.2)
  
  plot_file <- file.path(plot_dir,
                         sprintf("%s_%d_ageband_heatmap_top%d.png",
                                cohort_name, event_year, top_n))
  
  ggplot2::ggsave(plot_file, p, width = 10 + length(loaded_age_bands) * 0.8, 
                  height = plot_height, dpi = 300, limitsize = FALSE)
  cat(sprintf("✓ Saved: %s\n", basename(plot_file)))
  
  # ============================================================================
  # Create summary statistics table
  # ============================================================================
  cat("\nCreating summary statistics...\n")
  
  summary_stats <- combined %>%
    filter(feature %in% top_features_per_band) %>%
    group_by(feature) %>%
    summarise(
      n_age_bands = n(),
      mean_importance = mean(importance_scaled, na.rm = TRUE),
      sd_importance = sd(importance_scaled, na.rm = TRUE),
      min_importance = min(importance_scaled, na.rm = TRUE),
      max_importance = max(importance_scaled, na.rm = TRUE),
      range_importance = max_importance - min_importance,
      cv_importance = sd_importance / mean_importance * 100,  # Coefficient of variation
      .groups = "drop"
    ) %>%
    arrange(desc(mean_importance))
  
  summary_file <- file.path(plot_dir,
                            sprintf("%s_%d_ageband_summary_top%d.csv",
                                   cohort_name, event_year, top_n))
  readr::write_csv(summary_stats, summary_file)
  cat(sprintf("✓ Saved summary: %s\n", basename(summary_file)))
  
  # Print key insights
  cat("\n", rep("=", 80), "\n", sep="")
  cat("Key Insights:\n")
  cat(rep("=", 80), "\n", sep="")
  
  # Most consistent features (low CV)
  most_consistent <- summary_stats %>%
    filter(n_age_bands >= 3) %>%
    arrange(cv_importance) %>%
    head(5)
  
  cat("\nMost Consistent Features Across Age Bands (Low Variability):\n")
  for (i in 1:min(5, nrow(most_consistent))) {
    cat(sprintf("  %d. %s (CV=%.1f%%, present in %d age bands)\n",
                i, most_consistent$feature[i], 
                most_consistent$cv_importance[i],
                most_consistent$n_age_bands[i]))
  }
  
  # Most variable features (high CV)
  most_variable <- summary_stats %>%
    filter(n_age_bands >= 3) %>%
    arrange(desc(cv_importance)) %>%
    head(5)
  
  cat("\nMost Variable Features Across Age Bands (High Age-Specificity):\n")
  for (i in 1:min(5, nrow(most_variable))) {
    cat(sprintf("  %d. %s (CV=%.1f%%, range=%.3f-%.3f)\n",
                i, most_variable$feature[i],
                most_variable$cv_importance[i],
                most_variable$min_importance[i],
                most_variable$max_importance[i]))
  }
  
  # Age band specific features
  age_specific <- summary_stats %>%
    filter(n_age_bands == 1) %>%
    arrange(desc(mean_importance)) %>%
    head(10)
  
  if (nrow(age_specific) > 0) {
    cat(sprintf("\nAge Band-Specific Features (appear in only 1 age band): %d features\n",
                nrow(age_specific)))
    cat("  Top 5:\n")
    for (i in 1:min(5, nrow(age_specific))) {
      # Find which age band
      age_band_for_feature <- combined %>%
        filter(feature == age_specific$feature[i]) %>%
        pull(age_band)
      cat(sprintf("    %d. %s (importance=%.3f, age band=%s)\n",
                  i, age_specific$feature[i],
                  age_specific$mean_importance[i],
                  age_band_for_feature[1]))
    }
  }
  
  # ============================================================================
  # Upload to S3 if requested
  # ============================================================================
  if (s3_upload) {
    cat("\n", rep("=", 70), "\n", sep="")
    cat("Uploading to S3...\n")
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
      s3_base <- sprintf("s3://pgxdatalake/gold/feature_importance/cohort_name=%s/cross_ageband_analysis",
                        cohort_name)
      
      # Upload heatmap
      s3_plot_path <- sprintf("%s/%s", s3_base, basename(plot_file))
      upload_cmd <- sprintf('"%s" s3 cp "%s" "%s"', aws_cmd, plot_file, s3_plot_path)
      result <- system(upload_cmd, intern = FALSE, ignore.stdout = TRUE, ignore.stderr = TRUE)
      
      if (result == 0) {
        cat(sprintf("✓ Uploaded heatmap: %s\n", s3_plot_path))
      }
      
      # Upload summary
      s3_summary_path <- sprintf("%s/%s", s3_base, basename(summary_file))
      upload_cmd <- sprintf('"%s" s3 cp "%s" "%s"', aws_cmd, summary_file, s3_summary_path)
      result <- system(upload_cmd, intern = FALSE, ignore.stdout = TRUE, ignore.stderr = TRUE)
      
      if (result == 0) {
        cat(sprintf("✓ Uploaded summary: %s\n", s3_summary_path))
      }
      
      cat(sprintf("\n✓ Files uploaded to: %s\n", s3_base))
    } else {
      cat("⚠ AWS CLI not found. Files saved locally only.\n")
    }
  }
  
  cat("\n", rep("=", 80), "\n", sep="")
  cat("Heatmap Analysis Complete\n")
  cat(rep("=", 80), "\n", sep="")
  cat(sprintf("Heatmap: %s\n", normalizePath(plot_file)))
  cat(sprintf("Summary: %s\n", normalizePath(summary_file)))
  
  return(invisible(list(plot = plot_file, summary = summary_file)))
}

# Run when executed as standalone script
if (!interactive()) {
  # Example usage - modify these parameters
  cohort_name <- Sys.getenv("COHORT_NAME", "opioid_ed")
  event_year <- as.integer(Sys.getenv("EVENT_YEAR", "2016"))
  
  cat(sprintf("\nRunning cross-age-band analysis for: %s / %d\n", cohort_name, event_year))
  cat("Looking for files in outputs directory...\n\n")
  
  create_ageband_heatmap(
    cohort_name = cohort_name,
    event_year = event_year,
    age_bands = c("0-12", "13-24", "25-44", "45-54", "55-64", "65-74", "75+"),
    output_dir = "outputs",
    s3_upload = TRUE,
    top_n = 50
  )
}

