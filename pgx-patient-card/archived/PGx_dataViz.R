
library(renv)
renv:::activate()

library(readxl)
library(readr)
library(dplyr) 
library(magrittr) 
library(janitor) 
library(tidyr) 
library(purrr) 
library(tidyverse) 
library(reshape2)
library(here)
library(gridExtra)
library(grid)
library(gtable)
library(flextable)
library(officer)


pgx_data <- read_csv("pgx_data.csv")


cards <- pgx_data %>% group_by(`Sample ID`) %>% select(1:6,8,25) %>% unique()


sample1 <- cards %>% filter(`Sample ID`== "R208")
sample1 %<>% group_by(Gene) %>% unique()


# Format Drugs
allele2 <- sample1 %>% filter(allele_count == "2") 
drugs_allele2 <- rbind(allele2["Drug"]) %>% unique()
drugs_allele2 %<>% filter(Drug != "0")
allele2_drugs <- drugs_allele2 %>% dplyr::summarise(Drugs = paste(Drug, collapse = ", "))


allele1 <- sample1 %>% filter(allele_count == "1") 
drugs_allele1 <- rbind(allele1["Drug"]) %>% unique()
drugs_allele1 %<>% filter(Drug != "0")
allele1_drugs <- drugs_allele1 %>% dplyr::summarise(Drugs = paste(Drug, collapse = ", "))


drugs_wrapped <- strwrap(allele2_drugs$Drugs, width = 65, simplify = FALSE) # modify 30 to your needs
drugs_new <- sapply(drugs_wrapped, paste, collapse = "\n")
Allele2 <- data.frame(drugs_new)
names(Allele2) <- "Drugs"

drugs_wrapped <- strwrap(allele1_drugs$Drugs, width = 65, simplify = FALSE) # modify 30 to your needs
drugs_new <- sapply(drugs_wrapped, paste, collapse = "\n")
Allele1 <- data.frame(drugs_new)
names(Allele1) <- "Drugs"


# Format Genes and QR Codes
QR_codes <- sample1 %>% select(1:6,8) %>% unique()


# Genes with more than one entry - Consolidate variants
gene_multiple <- QR_codes %>%
  group_by(Gene) %>%
  filter(n() >1 )


Variants_single <- gather(gene_multiple, Variant, Variants, Variant1:Variant3, factor_key=TRUE)
Variants_single %<>% filter(Variants != "0")
allele_variants <- Variants_single %>% dplyr::summarise(Variants = paste(Variants, collapse = ", "))

gene_card_step1 <- cbind(Variants_single[1,1], allele_variants, Variants_single[1,3], Variants_single[1,4])



# Genes with one entry
gene <- QR_codes %>%
  group_by(Gene) %>%
  filter(n() == 1 )


Variants_df <- gather(gene, Variant, Variants, Variant1:Variant3, factor_key=TRUE)
Variants_df %<>% filter(Variants != "0")
allele_variants <- Variants_df %>% dplyr::summarise(Variants = paste(Variants, collapse = ", "))

genes_df <- left_join(allele_variants, gene, by = c('Gene' = 'Gene'))

gene_card_step2 <- genes_df %>% select(3,1,2,7,8)



# Final Gene Info
card_gene_info <- rbind(gene_card_step1, gene_card_step2)


# Final Drug Info
card_drug_info <- rbind(Allele2, Allele1)
rownames(card_drug_info)[1] <- "Alleles_2"
rownames(card_drug_info)[2] <- "Alleles_1"



# PGx Card Layout
# Row Counts   1----- 2 -------3------- 4 ----------------- 5 -------------- 6 -- 7 - 8 - 9 -10 -- 11  ----- 12 -------- 13 ----------- 14   
column1 <- c("Icon1", "", "Patient ID", "", "Genes Tested But Not Relevant", "", "", "", "", "", "Gene", "Variants", "Allele Count", "QR Code")

# Row Counts   1----------------------- 2 -----------------------3-------------------------- 4 --------------------------------------- 5 ------------------------------------ 6 -- 7 - 8 - 9 -10 -- 11  ----- 12 -------- 13 ----------- 14   
column2 <- c("VCU PGx Research Study", "", "I particiapte in a VCU medication safety study", "", "The Following drugs may require dosing modifications due to gene markers:", "", "", "", "", "", "Gene", "Variants", "Allele Count", "QR Code")

# Row Counts   1----- 2 --3-- 4 - 5 -------------- 6 ------------------ 7 --------------- 8 --------------- 9 ------------10 ------------- 11  ----- 12 -------- 13 ----------- 14  
column3 <- c("Icon2", "", "", "", "", "VCU Points of Contact:", "Dr. Mackiewicz", "tel: 804.314.1417", "Dr. Price", "tel: 804.314.1417", "Gene", "Variants", "Allele Count", "QR Code")


card.layout <- data.frame(column1, column2, column3)

images <- read_csv("data/base64_icons.csv")
names(images)[1] <- "Index"






