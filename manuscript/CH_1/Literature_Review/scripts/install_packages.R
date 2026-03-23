# Script to install R packages to a user-writable library location
# This solves the "library is not writable" error on Windows

# Get the user's home directory
user_home <- Sys.getenv("USERPROFILE")

# Create a personal library directory if it doesn't exist
personal_lib <- file.path(user_home, "R", "win-library", R.version$major, ".", R.version$minor)
dir.create(personal_lib, recursive = TRUE, showWarnings = FALSE)

# Add the personal library to the library paths
.libPaths(c(personal_lib, .libPaths()))

# Verify the library path is writable
cat("Installing packages to:", personal_lib, "\n")
cat("Library is writable:", file.access(personal_lib, 2) == 0, "\n\n")

# Install bslib package
cat("Installing bslib package...\n")
install.packages("bslib", lib = personal_lib, repos = "https://cran.rstudio.com/")

cat("\nInstallation complete!\n")
cat("To use this library path in future sessions, add this to your .Rprofile:\n")
cat(".libPaths(c(file.path(Sys.getenv('USERPROFILE'), 'R', 'win-library', R.version$major, '.', R.version$minor), .libPaths()))\n")
