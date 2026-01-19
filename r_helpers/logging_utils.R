# ============================================================
# LOGGING UTILITIES FOR R SCRIPTS
# ============================================================
# Shared logging functions for BupaR analysis and other R scripts

# Helper function for timestamped logging
# Args:
#   msg: Message to log
#   level: Log level (INFO, WARN, ERROR, etc.)
log_msg <- function(msg, level = "INFO") {
  timestamp <- format(Sys.time(), "%Y-%m-%d %H:%M:%S")
  cat(sprintf("[%s] [%s] %s\n", timestamp, level, msg))
  flush.console()  # Ensure output appears immediately in Jupyter
}
