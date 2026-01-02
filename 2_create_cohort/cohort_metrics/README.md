# Cohort Final Metrics - Visualization Guide

This directory contains final metrics CSV files for each cohort, generated from gold S3 parquet files.

## Generated Files

### For each cohort (`opioid_ed` and `non_opioid_ed`):

1. **`{cohort}_event_counts.csv`** - Total target and control event counts
2. **`{cohort}_yearly_metrics.csv`** - Patients and transactions by year (target and control)
3. **`{cohort}_distinct_patients_by_year.csv`** - Distinct patient counts by year
4. **`{cohort}_drug_frequency_target_by_year.csv`** - Drug frequency counts for target cases by year
5. **`{cohort}_drug_frequency_control_by_year.csv`** - Drug frequency counts for control cases by year

## R Visualization Code

### Setup

```r
# Load required libraries
library(ggplot2)
library(dplyr)
library(tidyr)
library(scales)
library(RColorBrewer)

# Set working directory to metrics folder
setwd("cohort_metrics")

# Define color palette
cohort_colors <- list(
  target = "#E74C3C",      # Red for target
  control = "#3498DB"      # Blue for control
)
```

### 1. Event Counts Visualization

**Bar chart comparing target vs control events**

```r
# Load event counts
opioid_ed_counts <- read.csv("opioid_ed_event_counts.csv")
non_opioid_ed_counts <- read.csv("non_opioid_ed_event_counts.csv")

# Combine cohorts
all_counts <- rbind(
  cbind(opioid_ed_counts, cohort_type = "Opioid ED"),
  cbind(non_opioid_ed_counts, cohort_type = "Non-Opioid ED")
)

# Create bar chart
ggplot(all_counts, aes(x = cohort_name, fill = cohort_type)) +
  geom_bar(aes(y = target_events), stat = "identity", position = "dodge", alpha = 0.7) +
  geom_bar(aes(y = control_events), stat = "identity", position = "dodge", alpha = 0.7) +
  scale_fill_manual(values = c("Opioid ED" = cohort_colors$target, 
                                "Non-Opioid ED" = cohort_colors$control)) +
  labs(title = "Total Event Counts by Cohort",
       x = "Cohort",
       y = "Number of Events",
       fill = "Event Type") +
  scale_y_continuous(labels = comma_format()) +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# Stacked bar chart showing proportions
all_counts_long <- all_counts %>%
  pivot_longer(cols = c(target_events, control_events), 
               names_to = "event_type", 
               values_to = "count") %>%
  mutate(event_type = ifelse(event_type == "target_events", "Target", "Control"))

ggplot(all_counts_long, aes(x = cohort_name, y = count, fill = event_type)) +
  geom_bar(stat = "identity", position = "stack") +
  scale_fill_manual(values = c("Target" = cohort_colors$target, 
                                "Control" = cohort_colors$control)) +
  labs(title = "Event Counts by Cohort (Stacked)",
       x = "Cohort",
       y = "Number of Events",
       fill = "Event Type") +
  scale_y_continuous(labels = comma_format()) +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))
```

### 2. Yearly Metrics Visualization

**Line chart showing trends over time**

```r
# Load yearly metrics
opioid_ed_yearly <- read.csv("opioid_ed_yearly_metrics.csv")
non_opioid_ed_yearly <- read.csv("non_opioid_ed_yearly_metrics.csv")

# Function to plot yearly metrics for a cohort
plot_yearly_metrics <- function(data, cohort_label) {
  data_long <- data %>%
    pivot_longer(cols = c(target_patients, control_patients, 
                         target_transactions, control_transactions),
                 names_to = "metric_type",
                 values_to = "value") %>%
    separate(metric_type, into = c("group", "metric"), sep = "_", extra = "merge")
  
  # Plot patients
  p1 <- data_long %>%
    filter(metric == "patients") %>%
    ggplot(aes(x = year, y = value, color = group)) +
    geom_line(size = 1.2) +
    geom_point(size = 3) +
    scale_color_manual(values = c("target" = cohort_colors$target,
                                   "control" = cohort_colors$control),
                       labels = c("target" = "Target", "control" = "Control")) +
    labs(title = paste(cohort_label, "- Distinct Patients by Year"),
         x = "Year",
         y = "Number of Patients",
         color = "Group") +
    scale_y_continuous(labels = comma_format()) +
    theme_minimal() +
    theme(legend.position = "bottom")
  
  # Plot transactions
  p2 <- data_long %>%
    filter(metric == "transactions") %>%
    ggplot(aes(x = year, y = value, color = group)) +
    geom_line(size = 1.2) +
    geom_point(size = 3) +
    scale_color_manual(values = c("target" = cohort_colors$target,
                                   "control" = cohort_colors$control),
                       labels = c("target" = "Target", "control" = "Control")) +
    labs(title = paste(cohort_label, "- Transactions by Year"),
         x = "Year",
         y = "Number of Transactions",
         color = "Group") +
    scale_y_continuous(labels = comma_format()) +
    theme_minimal() +
    theme(legend.position = "bottom")
  
  return(list(patients = p1, transactions = p2))
}

# Generate plots
opioid_plots <- plot_yearly_metrics(opioid_ed_yearly, "Opioid ED")
non_opioid_plots <- plot_yearly_metrics(non_opioid_ed_yearly, "Non-Opioid ED")

# Display plots
print(opioid_plots$patients)
print(opioid_plots$transactions)
print(non_opioid_plots$patients)
print(non_opioid_plots$transactions)

# Combined comparison plot
combined_yearly <- rbind(
  cbind(opioid_ed_yearly, cohort_type = "Opioid ED"),
  cbind(non_opioid_ed_yearly, cohort_type = "Non-Opioid ED")
)

ggplot(combined_yearly, aes(x = year, y = target_patients, color = cohort_type)) +
  geom_line(size = 1.2) +
  geom_point(size = 3) +
  scale_color_manual(values = c("Opioid ED" = cohort_colors$target,
                                 "Non-Opioid ED" = "#9B59B6")) +
  labs(title = "Target Patients by Year - Cohort Comparison",
       x = "Year",
       y = "Number of Target Patients",
       color = "Cohort") +
  scale_y_continuous(labels = comma_format()) +
  theme_minimal() +
  theme(legend.position = "bottom")
```

### 3. Distinct Patients by Year

**Dual-axis or faceted plot**

```r
# Load distinct patient data
opioid_ed_patients <- read.csv("opioid_ed_distinct_patients_by_year.csv")
non_opioid_ed_patients <- read.csv("non_opioid_ed_distinct_patients_by_year.csv")

# Plot for opioid_ed
opioid_ed_patients %>%
  pivot_longer(cols = c(target_patients, control_patients),
               names_to = "group",
               values_to = "count") %>%
  mutate(group = ifelse(group == "target_patients", "Target", "Control")) %>%
  ggplot(aes(x = year, y = count, fill = group)) +
  geom_bar(stat = "identity", position = "dodge") +
  scale_fill_manual(values = c("Target" = cohort_colors$target,
                                "Control" = cohort_colors$control)) +
  labs(title = "Opioid ED - Distinct Patients by Year",
       x = "Year",
       y = "Number of Distinct Patients",
       fill = "Group") +
  scale_y_continuous(labels = comma_format()) +
  theme_minimal() +
  theme(legend.position = "bottom")

# Plot for non_opioid_ed
non_opioid_ed_patients %>%
  pivot_longer(cols = c(target_patients, control_patients),
               names_to = "group",
               values_to = "count") %>%
  mutate(group = ifelse(group == "target_patients", "Target", "Control")) %>%
  ggplot(aes(x = year, y = count, fill = group)) +
  geom_bar(stat = "identity", position = "dodge") +
  scale_fill_manual(values = c("Target" = cohort_colors$target,
                                "Control" = cohort_colors$control)) +
  labs(title = "Non-Opioid ED - Distinct Patients by Year",
       x = "Year",
       y = "Number of Distinct Patients",
       fill = "Group") +
  scale_y_continuous(labels = comma_format()) +
  theme_minimal() +
  theme(legend.position = "bottom")
```

### 4. Drug Frequency Visualizations

**Top N drugs by frequency**

```r
# Function to plot top N drugs
plot_top_drugs <- function(data, cohort_label, n_top = 20, group_type = "target") {
  # Aggregate across all years
  drug_totals <- data %>%
    group_by(drug_name) %>%
    summarise(total_frequency = sum(frequency), .groups = "drop") %>%
    arrange(desc(total_frequency)) %>%
    head(n_top)
  
  # Get data for top drugs
  top_drug_data <- data %>%
    filter(drug_name %in% drug_totals$drug_name) %>%
    group_by(drug_name) %>%
    summarise(total_frequency = sum(frequency), .groups = "drop") %>%
    arrange(desc(total_frequency)) %>%
    mutate(drug_name = factor(drug_name, levels = drug_name))
  
  ggplot(top_drug_data, aes(x = drug_name, y = total_frequency)) +
    geom_bar(stat = "identity", fill = ifelse(group_type == "target", 
                                               cohort_colors$target, 
                                               cohort_colors$control)) +
    coord_flip() +
    labs(title = paste(cohort_label, "- Top", n_top, "Drugs (", 
                       ifelse(group_type == "target", "Target", "Control"), ")"),
         x = "Drug Name",
         y = "Total Frequency") +
    scale_y_continuous(labels = comma_format()) +
    theme_minimal() +
    theme(axis.text.y = element_text(size = 8))
}

# Load drug frequency data
opioid_ed_drugs_target <- read.csv("opioid_ed_drug_frequency_target_by_year.csv")
opioid_ed_drugs_control <- read.csv("opioid_ed_drug_frequency_control_by_year.csv")
non_opioid_ed_drugs_target <- read.csv("non_opioid_ed_drug_frequency_target_by_year.csv")
non_opioid_ed_drugs_control <- read.csv("non_opioid_ed_drug_frequency_control_by_year.csv")

# Generate plots
plot_top_drugs(opioid_ed_drugs_target, "Opioid ED", n_top = 20, "target")
plot_top_drugs(opioid_ed_drugs_control, "Opioid ED", n_top = 20, "control")
plot_top_drugs(non_opioid_ed_drugs_target, "Non-Opioid ED", n_top = 20, "target")
plot_top_drugs(non_opioid_ed_drugs_control, "Non-Opioid ED", n_top = 20, "control")
```

**Drug frequency trends over time (heatmap)**

```r
# Function to create heatmap of top drugs over time
plot_drug_heatmap <- function(data, cohort_label, n_top = 15, group_type = "target") {
  # Get top N drugs by total frequency
  top_drugs <- data %>%
    group_by(drug_name) %>%
    summarise(total_freq = sum(frequency), .groups = "drop") %>%
    arrange(desc(total_freq)) %>%
    head(n_top) %>%
    pull(drug_name)
  
  # Filter and prepare data
  heatmap_data <- data %>%
    filter(drug_name %in% top_drugs) %>%
    group_by(drug_name) %>%
    mutate(total_freq = sum(frequency)) %>%
    ungroup() %>%
    mutate(drug_name = factor(drug_name, 
                              levels = top_drugs[order(sapply(top_drugs, function(d) {
                                sum(data$frequency[data$drug_name == d])
                              }), decreasing = TRUE)]))
  
  ggplot(heatmap_data, aes(x = year, y = drug_name, fill = frequency)) +
    geom_tile() +
    scale_fill_gradient(low = "white", 
                        high = ifelse(group_type == "target", 
                                     cohort_colors$target, 
                                     cohort_colors$control),
                        labels = comma_format()) +
    labs(title = paste(cohort_label, "- Top", n_top, "Drugs Frequency Over Time (", 
                       ifelse(group_type == "target", "Target", "Control"), ")"),
         x = "Year",
         y = "Drug Name",
         fill = "Frequency") +
    theme_minimal() +
    theme(axis.text.y = element_text(size = 8),
          legend.position = "right")
}

# Generate heatmaps
plot_drug_heatmap(opioid_ed_drugs_target, "Opioid ED", n_top = 15, "target")
plot_drug_heatmap(opioid_ed_drugs_control, "Opioid ED", n_top = 15, "control")
plot_drug_heatmap(non_opioid_ed_drugs_target, "Non-Opioid ED", n_top = 15, "target")
plot_drug_heatmap(non_opioid_ed_drugs_control, "Non-Opioid ED", n_top = 15, "control")
```

**Drug frequency comparison: Target vs Control**

```r
# Function to compare top drugs between target and control
compare_drugs_target_vs_control <- function(target_data, control_data, cohort_label, n_top = 20) {
  # Get top drugs from target group
  top_target_drugs <- target_data %>%
    group_by(drug_name) %>%
    summarise(target_freq = sum(frequency), .groups = "drop") %>%
    arrange(desc(target_freq)) %>%
    head(n_top)
  
  # Get frequencies for same drugs in control group
  top_control_drugs <- control_data %>%
    filter(drug_name %in% top_target_drugs$drug_name) %>%
    group_by(drug_name) %>%
    summarise(control_freq = sum(frequency), .groups = "drop")
  
  # Combine and prepare
  comparison <- top_target_drugs %>%
    left_join(top_control_drugs, by = "drug_name") %>%
    mutate(control_freq = ifelse(is.na(control_freq), 0, control_freq)) %>%
    arrange(desc(target_freq)) %>%
    mutate(drug_name = factor(drug_name, levels = drug_name)) %>%
    pivot_longer(cols = c(target_freq, control_freq),
                 names_to = "group",
                 values_to = "frequency") %>%
    mutate(group = ifelse(group == "target_freq", "Target", "Control"))
  
  ggplot(comparison, aes(x = drug_name, y = frequency, fill = group)) +
    geom_bar(stat = "identity", position = "dodge") +
    scale_fill_manual(values = c("Target" = cohort_colors$target,
                                  "Control" = cohort_colors$control)) +
    coord_flip() +
    labs(title = paste(cohort_label, "- Top", n_top, "Drugs: Target vs Control"),
         x = "Drug Name",
         y = "Total Frequency",
         fill = "Group") +
    scale_y_continuous(labels = comma_format()) +
    theme_minimal() +
    theme(axis.text.y = element_text(size = 8),
          legend.position = "bottom")
}

# Generate comparisons
compare_drugs_target_vs_control(opioid_ed_drugs_target, opioid_ed_drugs_control, 
                                "Opioid ED", n_top = 20)
compare_drugs_target_vs_control(non_opioid_ed_drugs_target, non_opioid_ed_drugs_control, 
                                "Non-Opioid ED", n_top = 20)
```

### 5. Comprehensive Dashboard

**Create a multi-panel dashboard**

```r
# Create comprehensive dashboard function
create_cohort_dashboard <- function(cohort_name, cohort_label) {
  # Load all data
  event_counts <- read.csv(paste0(cohort_name, "_event_counts.csv"))
  yearly_metrics <- read.csv(paste0(cohort_name, "_yearly_metrics.csv"))
  distinct_patients <- read.csv(paste0(cohort_name, "_distinct_patients_by_year.csv"))
  drugs_target <- read.csv(paste0(cohort_name, "_drug_frequency_target_by_year.csv"))
  drugs_control <- read.csv(paste0(cohort_name, "_drug_frequency_control_by_year.csv"))
  
  # Panel 1: Event counts
  p1 <- event_counts %>%
    pivot_longer(cols = c(target_events, control_events),
                 names_to = "event_type",
                 values_to = "count") %>%
    mutate(event_type = ifelse(event_type == "target_events", "Target", "Control")) %>%
    ggplot(aes(x = cohort_name, y = count, fill = event_type)) +
    geom_bar(stat = "identity", position = "dodge") +
    scale_fill_manual(values = c("Target" = cohort_colors$target,
                                  "Control" = cohort_colors$control)) +
    labs(title = "Total Event Counts", x = "", y = "Events", fill = "") +
    scale_y_continuous(labels = comma_format()) +
    theme_minimal() +
    theme(legend.position = "bottom")
  
  # Panel 2: Patients over time
  p2 <- yearly_metrics %>%
    pivot_longer(cols = c(target_patients, control_patients),
                 names_to = "group",
                 values_to = "count") %>%
    mutate(group = ifelse(group == "target_patients", "Target", "Control")) %>%
    ggplot(aes(x = year, y = count, color = group)) +
    geom_line(size = 1.2) +
    geom_point(size = 3) +
    scale_color_manual(values = c("Target" = cohort_colors$target,
                                  "Control" = cohort_colors$control)) +
    labs(title = "Patients by Year", x = "Year", y = "Patients", color = "") +
    scale_y_continuous(labels = comma_format()) +
    theme_minimal() +
    theme(legend.position = "bottom")
  
  # Panel 3: Transactions over time
  p3 <- yearly_metrics %>%
    pivot_longer(cols = c(target_transactions, control_transactions),
                 names_to = "group",
                 values_to = "count") %>%
    mutate(group = ifelse(group == "target_transactions", "Target", "Control")) %>%
    ggplot(aes(x = year, y = count, color = group)) +
    geom_line(size = 1.2) +
    geom_point(size = 3) +
    scale_color_manual(values = c("Target" = cohort_colors$target,
                                  "Control" = cohort_colors$control)) +
    labs(title = "Transactions by Year", x = "Year", y = "Transactions", color = "") +
    scale_y_continuous(labels = comma_format()) +
    theme_minimal() +
    theme(legend.position = "bottom")
  
  # Panel 4: Top 10 drugs (target)
  top_target_drugs <- drugs_target %>%
    group_by(drug_name) %>%
    summarise(total_freq = sum(frequency), .groups = "drop") %>%
    arrange(desc(total_freq)) %>%
    head(10) %>%
    mutate(drug_name = factor(drug_name, levels = drug_name))
  
  p4 <- ggplot(top_target_drugs, aes(x = drug_name, y = total_freq)) +
    geom_bar(stat = "identity", fill = cohort_colors$target) +
    coord_flip() +
    labs(title = "Top 10 Drugs (Target)", x = "", y = "Frequency") +
    scale_y_continuous(labels = comma_format()) +
    theme_minimal() +
    theme(axis.text.y = element_text(size = 8))
  
  # Combine panels
  library(gridExtra)
  grid.arrange(p1, p2, p3, p4, ncol = 2, 
               top = paste(cohort_label, "Dashboard"))
}

# Generate dashboards
create_cohort_dashboard("opioid_ed", "Opioid ED")
create_cohort_dashboard("non_opioid_ed", "Non-Opioid ED")
```

## Usage Notes

1. **Install required R packages:**
   ```r
   install.packages(c("ggplot2", "dplyr", "tidyr", "scales", "RColorBrewer", "gridExtra"))
   ```

2. **Set working directory:**
   ```r
   setwd("path/to/cohort_metrics")
   ```

3. **Customize visualizations:**
   - Adjust `n_top` parameter to show more/fewer top drugs
   - Modify colors using `cohort_colors` list
   - Change plot themes and labels as needed

4. **Save plots:**
   ```r
   ggsave("plot_name.png", width = 10, height = 6, dpi = 300)
   ```

## Data Dictionary

- **target_events**: Total number of target case events
- **control_events**: Total number of control case events
- **target_patients**: Number of distinct patients in target group
- **control_patients**: Number of distinct patients in control group
- **target_transactions**: Number of transactions/events for target group
- **control_transactions**: Number of transactions/events for control group
- **drug_name**: Name of the drug
- **frequency**: Number of times the drug appears in the data
- **year**: Event year (2016-2020)

