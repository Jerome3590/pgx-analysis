# Configure R to use a user-writable library location
# This prevents "library is not writable" errors on Windows

# Get the user's home directory
user_home <- Sys.getenv("USERPROFILE")

# Create a personal library directory if it doesn't exist
# Use R version components to create version-specific library path
r_version <- paste(R.version$major, strsplit(R.version$minor, "\\.")[[1]][1], sep = ".")
personal_lib <- file.path(user_home, "R", "win-library", r_version)
dir.create(personal_lib, recursive = TRUE, showWarnings = FALSE)

# Add the personal library to the library paths (at the beginning)
.libPaths(c(personal_lib, .libPaths()))

# Optional: Print a message (comment out if you don't want to see this)
# cat("Using personal library:", personal_lib, "\n")
