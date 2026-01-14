# ============================================================
# LOGGING UTILITIES
# ============================================================
# R logging functions aligned with Python logging_utils
# Provides console, file, and S3 logging capabilities

# Setup R logging similar to Python logging_utils.setup_logging()
# Creates log file and returns logger object for logging to file and console
setup_r_logging <- function(cohort_name, age_band, event_year) {
  # Args:
  #   cohort_name: Cohort name (e.g., 'opioid_ed')
  #   age_band: Age band (e.g., '25-44')
  #   event_year: Event year (e.g., 2016)
  # Returns:
  #   List with logger, log_file_path, and log_connection
  
  # Create logs directory
  logs_dir <- here::here("logs")
  if (!dir.exists(logs_dir)) {
    dir.create(logs_dir, showWarnings = FALSE, recursive = TRUE)
  }
  
  # Generate unique log filename (similar to Python setup)
  timestamp <- format(Sys.time(), "%Y%m%d_%H%M%S")
  process_id <- Sys.getpid()
  log_prefix <- paste0(cohort_name, "_feature_importance_output")
  log_filename <- sprintf("%s_%s_%d.txt", log_prefix, timestamp, process_id)
  log_file_path <- file.path(logs_dir, log_filename)
  
  # Open connection for logging (append mode)
  log_con <- file(log_file_path, open = "wt")
  
  # Function to log messages to both console and file
  log_message <- function(level, message, ...) {
    timestamp_str <- format(Sys.time(), "%Y-%m-%d %H:%M:%S")
    formatted_msg <- sprintf("%s - %s - %s", timestamp_str, level, sprintf(message, ...))
    
    # Write to console
    cat(formatted_msg, "\n")
    
    # Write to file
    writeLines(formatted_msg, log_con)
    flush(log_con)
  }
  
  # Create logging functions
  logger <- list(
    info = function(msg, ...) log_message("INFO", msg, ...),
    warning = function(msg, ...) log_message("WARNING", msg, ...),
    error = function(msg, ...) log_message("ERROR", msg, ...),
    debug = function(msg, ...) log_message("DEBUG", msg, ...),
    close = function() {
      if (isOpen(log_con)) {
        close(log_con)
      }
    },
    get_log_path = function() log_file_path,
    get_log_content = function() {
      if (file.exists(log_file_path)) {
        paste(readLines(log_file_path), collapse = "\n")
      } else {
        ""
      }
    }
  )
  
  return(list(
    logger = logger,
    log_file_path = log_file_path,
    log_connection = log_con
  ))
}

# Function to save logs to S3 (similar to Python save_logs_to_s3)
# Save log file to S3 using AWS CLI (aligned with Python save_logs_to_s3)
# Args:
#   log_file_path: Path to local log file
#   cohort_name: Cohort name
#   age_band: Age band
#   event_year: Event year
#   logger: Optional logger object for logging the save operation
save_logs_to_s3_r <- function(log_file_path, cohort_name, age_band, event_year, logger = NULL) {
  if (!file.exists(log_file_path)) {
    if (!is.null(logger)) {
      logger$warning("Log file does not exist: %s", log_file_path)
    }
    return(invisible(NULL))
  }
  
  # Clean up age_band and event_year if they have prefixes
  age_band_clean <- gsub("^age_band=", "", as.character(age_band))
  event_year_clean <- gsub("^event_year=", "", as.character(event_year))
  
  # Generate S3 path (aligned with Python logging_utils)
  timestamp <- format(Sys.time(), "%Y%m%d_%H%M%S")
  s3_path <- sprintf(
    "s3://pgx-repository/build_logs/feature_importance/%s/%s/%s/log_%s.txt",
    cohort_name,
    age_band_clean,
    event_year_clean,
    timestamp
  )
  
  # Use AWS CLI to copy to S3
  aws_cmd <- Sys.which("aws")
  if (aws_cmd == "") {
    # Try common AWS CLI paths
    aws_paths <- c(
      "/usr/local/bin/aws",
      "/usr/bin/aws",
      "/home/ec2-user/.local/bin/aws",
      "C:/Program Files/Amazon/AWSCLIV2/aws.exe"
    )
    for (path in aws_paths) {
      if (file.exists(path)) {
        aws_cmd <- path
        break
      }
    }
  }
  
  if (aws_cmd != "" && file.exists(aws_cmd)) {
    # Extract bucket and key from s3:// path
    s3_parts <- gsub("^s3://", "", s3_path)
    parts <- strsplit(s3_parts, "/", fixed = TRUE)[[1]]
    bucket <- parts[1]
    key <- paste(parts[-1], collapse = "/")
    
    # Build AWS CLI command
    aws_args <- c(
      "s3", "cp",
      log_file_path,
      s3_path,
      "--content-type", "text/plain"
    )
    
    # Execute AWS CLI command
    result <- tryCatch({
      system2(aws_cmd, aws_args, stdout = TRUE, stderr = TRUE)
    }, error = function(e) {
      if (!is.null(logger)) {
        logger$warning("Failed to save logs to S3: %s", e$message)
      }
      return(NULL)
    })
    
    if (!is.null(result) && !is.null(logger)) {
      logger$info("✓ Logs saved to S3: %s", s3_path)
    }
  } else {
    if (!is.null(logger)) {
      logger$warning("AWS CLI not found. Logs saved locally only: %s", log_file_path)
      logger$warning("To upload manually: aws s3 cp %s %s", log_file_path, s3_path)
    }
  }
  
  return(invisible(s3_path))
}

# ============================================================
# MEMORY LOGGING (aligned with 2_create_cohort Python check_memory_usage)
# ============================================================
# Function to check and log R memory usage
# Similar to Python's check_memory_usage from duckdb_utils
check_memory_usage_r <- function(logger = NULL, step_name = "Memory check") {
  # Args:
  #   logger: Optional logger object (if NULL, prints to console only)
  #   step_name: Name of the step being checked
  # Returns:
  #   List with memory stats
  
  # Force garbage collection to get accurate memory stats
  gc_result <- gc(verbose = FALSE)
  
  # Get memory stats from gc()
  # gc() returns a matrix with columns: used, gc trigger, max used, max trigger
  # Row 1 = Ncells (cons cells), Row 2 = Vcells (vector cells)
  if (nrow(gc_result) >= 2) {
    # Total memory used (in MB)
    total_used_mb <- sum(gc_result[, "used"]) / 1024^2  # Convert from KB to MB
    total_max_mb <- sum(gc_result[, "max used"]) / 1024^2
    
    # Format memory string
    if (total_used_mb >= 1024) {
      memory_str <- sprintf("%.1f GiB", total_used_mb / 1024)
    } else {
      memory_str <- sprintf("%.1f MiB", total_used_mb)
    }
    
    # Log memory usage
    msg <- sprintf("📊 %s: Memory usage: %s (max: %.1f MiB)", step_name, memory_str, total_max_mb)
    
    if (!is.null(logger)) {
      logger$info(msg)
    } else {
      cat(msg, "\n")
    }
    
    return(list(
      used_mb = total_used_mb,
      max_mb = total_max_mb,
      gc_result = gc_result
    ))
  } else {
    warning_msg <- sprintf("⚠️ %s: Could not get memory stats", step_name)
    if (!is.null(logger)) {
      logger$warning(warning_msg)
    } else {
      cat(warning_msg, "\n")
    }
    return(list(used_mb = 0, max_mb = 0, gc_result = NULL))
  }
}

