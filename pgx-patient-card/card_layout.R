
# PGx Card Layout
# Row Counts   1----- 2 -------3------- 4 ----------------- 5 -------------- 6 -- 7 - 8 - 9 -10 -- 11  ----- 12 -------- 13 ----------- 14   
PGx1 <- c("Icon1", "", "Patient ID", "", "Genes Tested But Not Relevant", "", "", "", "", "", "Gene", "Variants", "Allele Count", "QR Code")

# Row Counts   1----------------------- 2 -----------------------3-------------------------- 4 --------------------------------------- 5 ------------------------------------ 6 -- 7 - 8 - 9 -10 -- 11  ----- 12 -------- 13 ----------- 14   
PGx2 <- c("VCU PGx Research Study", "", "I particiapte in a VCU medication safety study", "", "The Following drugs may require dosing modifications due to gene markers:", "", "", "", "", "", "Gene", "Variants", "Allele Count", "QR Code")

# Row Counts   1----- 2 --3-- 4 - 5 -------------- 6 ------------------ 7 --------------- 8 --------------- 9 ------------10 ------------- 11  ----- 12 -------- 13 ----------- 14  
PGx3 <- c("Icon2", "", "", "", "", "VCU Points of Contact:", "Dr. Mackiewicz", "tel: 804.314.8263", "Dr. Price", "tel: 804.314.1417", "Gene", "Variants", "Allele Count", "QR Code")


card.layout <- data.frame(PGx1, PGx2, PGx3)

images <- read_csv("data/base64_icons.csv")
names(images)[1] <- "Index"

card.layout[1,1] <- images[8,3]
card.layout[1,3] <- images[13,3]

write.csv(card.layout, "data/card_layout.csv")