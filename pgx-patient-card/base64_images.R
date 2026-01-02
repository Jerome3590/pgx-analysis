library(base64enc)
library(knitr)
library(reshape2)
library(dplyr)
library(tools)
library(here)
library(readr)



image_to_base64 <- function(folder_path) {
  
  # Grab filenames and place in list vector
  files <- list.files(path=folder_path, pattern="*.*", full.names=TRUE, recursive=FALSE)
  lst <- vector("list", length(files))
  names(lst) <- files
  
  # Process files
  for (i in 1:length(files)) {
    out <- knitr::image_uri(files[i])
    ## store object in list, by file path and name
    lst[[files[i]]] <- out
  }
  
  # Format dataframe
  image_df <- melt(lst) %>% select(2,1)
  names(image_df)[1] <- "Image"
  names(image_df)[2] <- "Value"
  
  # Strip out file path from name
  image_df$Image <- basename(file_path_sans_ext(image_df$Image))
  
  # Strip out file path from folder
  filename <- basename(file_path_sans_ext(folder_path))
  
  # Save to .csv
  write.csv(image_df, paste0("base64_", filename, ".csv"), row.names=T)
  
}
