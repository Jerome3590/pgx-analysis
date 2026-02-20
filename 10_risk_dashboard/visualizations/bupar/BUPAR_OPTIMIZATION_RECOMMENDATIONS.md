# BupaR Visualization Optimization Recommendations

**Archived.** Interactive Plotly (year dropdown) is implemented; Gantt charts are not produced (see `9_dashboard_visuals/bupar/ARCHIVE_GANTT_REMOVAL.md`). For current workflow see **`10_risk_dashboard/docs/README_visualization_plan.md`**. See `archived/dashboard_docs/README.md`.

---

## Executive Summary

Based on review of current BupaR visualizations and dashboard configuration, this document provides recommendations for optimizing the 5 core visualizations for the PGx risk dashboard:
1. **Trace Explorer** - Activity sequence patterns
2. **Performance Spectrum** - Aggregated temporal activity trace
3. **Overall Activity Frequency** - Event distribution analysis
4. **Process Map** - Directly-follows graph with frequencies
5. **Process Matrix** - Drug interaction discovery

---

## Current State Assessment

### ✅ What's Working Well

1. **Data Pipeline**: Successfully filtering to SHAP/FFA important codes (top 500 features)
2. **Event Log Creation**: Properly building bupaR eventlogs from model_events.parquet
3. **Static PNG Generation**: High-quality 300 DPI images for all visualizations
4. **S3 Integration**: Automated upload and serving via CloudFront/API Gateway
5. **Pre/Post Target Analysis**: Separate analysis for pre-F1120 (opioid_ed) and pre-HCG (non_opioid_ed)

### ⚠️ Areas for Improvement

1. **Process Matrix**: Generated but **not displayed** on dashboard
2. **Interactive Elements**: All visualizations are static PNGs (no zoom, filter, drill-down)
3. **Large Cohort Scalability**: Current visualizations may be cluttered for high-patient-count age bands
4. **Activity Labels**: Drug/ICD/CPT codes can overlap on charts (label_size = 3.5)
5. **Color Coding**: Limited use of color to distinguish drug vs diagnosis vs procedure events
6. **Frequency Map**: Currently uses `export_map()` which may not be available in all bupaR versions

---

## Detailed Recommendations by Visualization

### 1. Trace Explorer

**Current Implementation:**
```r
trace_explorer(target_eventlog, n_traces = 20, label_size = 3.5, 
               abbreviate = FALSE, coverage_labels = c("relative"))
```

**Recommendations:**

#### A. Make it Interactive (Plotly/HTML)
Convert from static PNG to interactive HTML using `plotly` or `htmlwidgets`:

```r
# Generate interactive trace explorer
library(plotly)
library(htmlwidgets)

# Create trace frequency data
traces_df <- target_eventlog %>%
  bupaR::traces() %>%
  arrange(desc(absolute_frequency)) %>%
  head(30)  # Top 30 traces

# Build interactive plot
p <- plot_ly(traces_df, x = ~trace_id, y = ~absolute_frequency, 
             type = 'bar', 
             text = ~paste("Trace:", trace, "<br>Count:", absolute_frequency, 
                          "<br>Coverage:", scales::percent(relative_frequency)),
             hoverinfo = 'text',
             marker = list(color = ~absolute_frequency, 
                          colorscale = 'Viridis',
                          showscale = TRUE,
                          colorbar = list(title = "Frequency"))) %>%
  layout(title = paste("Trace Explorer:", cohort_name, age_band),
         xaxis = list(title = "Trace Variant", tickangle = -45),
         yaxis = list(title = "Absolute Frequency"),
         hovermode = 'closest')

# Save as self-contained HTML
htmlwidgets::saveWidget(p, 
                       file.path(plots_dir, sprintf("%s_%s_trace_explorer_interactive.html", 
                                                    cohort_name, age_band_fname)),
                       selfcontained = TRUE)
```

**Benefits:**
- Users can hover to see full trace sequences
- Zoom into specific traces
- Filter by frequency threshold
- Better for large cohorts (can show more than 20 traces)

#### B. Improve Static PNG Version
For static version, enhance readability:

```r
# Enhanced static trace explorer with better formatting
p_te <- trace_explorer(target_eventlog, 
                      n_traces = 30,  # Increase from 20
                      label_size = 3.0,  # Reduce for better fit
                      abbreviate = TRUE,  # Abbreviate long codes
                      coverage_labels = c("relative", "absolute"),  # Show both
                      show_labels = TRUE) +
  theme_minimal(base_size = 14) +
  theme(axis.text.y = element_text(size = 10, hjust = 1),
        plot.title = element_text(face = "bold", size = 16),
        plot.margin = margin(10, 10, 10, 40))  # Add left margin for labels

ggsave(..., width = 16, height = 12, dpi = 300)  # Larger dimensions
```

---

### 2. Performance Spectrum

**Current Implementation:**
```r
p_ps <- target_eventlog %>% psmineR::ps_aggregated()
ggsave(..., width = 12, height = 8, dpi = 300)
```

**Recommendations:**

#### A. Add Temporal Segmentation
Break down by time periods (pre-target, post-target, time-to-target):

```r
library(psmineR)

# Add time-to-target feature
target_eventlog_enriched <- target_eventlog %>%
  group_by(case_id) %>%
  mutate(days_to_target = as.numeric(difftime(max(timestamp), timestamp, units = "days")),
         time_segment = case_when(
           days_to_target > 180 ~ ">6 months before",
           days_to_target > 90 ~ "3-6 months before",
           days_to_target > 30 ~ "1-3 months before",
           days_to_target > 7 ~ "1-4 weeks before",
           TRUE ~ "<1 week before"
         )) %>%
  ungroup()

# Create segmented performance spectrum
p_ps_segmented <- target_eventlog_enriched %>%
  ps_aggregated(segments = "time_segment") +
  scale_fill_viridis_d(option = "plasma") +
  labs(title = "Performance Spectrum by Time to Target",
       subtitle = paste(cohort_name, age_band, "- Temporal Activity Patterns"),
       fill = "Time Segment") +
  theme_minimal(base_size = 14)

ggsave(..., width = 14, height = 10, dpi = 300)
```

#### B. Add Activity Type Colors
Distinguish drugs, diagnoses, and procedures:

```r
# Color-code by activity type (drug, ICD, CPT)
target_eventlog_typed <- target_eventlog %>%
  mutate(activity_type = case_when(
    grepl("^DRUG:", activity) ~ "Drug",
    grepl("^ICD:", activity) ~ "Diagnosis",
    grepl("^CPT:", activity) ~ "Procedure",
    TRUE ~ "Other"
  ))

p_ps_typed <- target_eventlog_typed %>%
  ps_aggregated(color = "activity_type") +
  scale_fill_manual(values = c("Drug" = "#3b82f6", 
                               "Diagnosis" = "#ef4444", 
                               "Procedure" = "#10b981",
                               "Other" = "#gray50")) +
  labs(title = "Performance Spectrum by Activity Type",
       fill = "Event Type")
```

---

### 3. Overall Activity Frequency

**Current Implementation:**
```r
target_activity_freq <- target_eventlog %>%
  group_by(activity) %>%
  summarise(count = n(), .groups = "drop") %>%
  arrange(desc(count)) %>%
  head(30)

p3 <- ggplot(target_activity_freq, aes(x = reorder(activity, count), y = count)) +
  geom_bar(stat = "identity", fill = "darkgreen") +
  coord_flip() +
  labs(title = paste("Overall Activity Frequency:", cohort_name, age_band),
       x = "Activity", y = "Frequency") +
  theme_bw()
```

**Recommendations:**

#### A. Add Activity Type Color Coding
```r
target_activity_freq <- target_eventlog %>%
  mutate(activity_type = case_when(
    grepl("^DRUG:", activity) ~ "Drug",
    grepl("^ICD:", activity) ~ "Diagnosis",
    grepl("^CPT:", activity) ~ "Procedure",
    TRUE ~ "Other"
  )) %>%
  group_by(activity, activity_type) %>%
  summarise(count = n(), .groups = "drop") %>%
  arrange(desc(count)) %>%
  head(40)  # Increase from 30

# Enhanced visualization with color coding
p3 <- ggplot(target_activity_freq, 
            aes(x = reorder(activity, count), y = count, fill = activity_type)) +
  geom_col() +
  coord_flip() +
  scale_fill_manual(values = c("Drug" = "#3b82f6", 
                               "Diagnosis" = "#ef4444", 
                               "Procedure" = "#10b981",
                               "Other" = "#64748b"),
                   name = "Event Type") +
  labs(title = paste("Overall Activity Frequency:", cohort_name, age_band),
       subtitle = "Top 40 activities by frequency",
       x = NULL, y = "Frequency") +
  theme_minimal(base_size = 13) +
  theme(axis.text.y = element_text(size = 10),
        legend.position = "top",
        panel.grid.minor = element_blank())

ggsave(..., width = 14, height = 11, dpi = 300)
```

#### B. Add Interactive Version (Plotly)
```r
library(plotly)

p3_interactive <- plot_ly(target_activity_freq, 
                         x = ~count, 
                         y = ~reorder(activity, count),
                         type = 'bar',
                         orientation = 'h',
                         color = ~activity_type,
                         colors = c("Drug" = "#3b82f6", 
                                   "Diagnosis" = "#ef4444", 
                                   "Procedure" = "#10b981"),
                         text = ~paste("Activity:", activity, 
                                      "<br>Type:", activity_type,
                                      "<br>Count:", count),
                         hoverinfo = 'text') %>%
  layout(title = paste("Overall Activity Frequency:", cohort_name, age_band),
         xaxis = list(title = "Frequency"),
         yaxis = list(title = "", tickfont = list(size = 10)),
         margin = list(l = 200),
         legend = list(orientation = "h", y = 1.1))

htmlwidgets::saveWidget(p3_interactive, 
                       file.path(plots_dir, sprintf("%s_%s_activity_frequency_interactive.html", 
                                                    cohort_name, age_band_fname)),
                       selfcontained = TRUE)
```

---

### 4. Process Map (Frequency Map)

**Current Implementation:**
```r
pm_freq <- process_map(target_eventlog, type = frequency("absolute"), render = FALSE)
processmapR::export_map(pm_freq, file_name = freq_map_path, file_type = "png", 
                       width = 1200, height = 900)
```

**Recommendations:**

#### A. Use DiagrammeR for Better Quality
```r
library(DiagrammeR)
library(DiagrammeRsvg)
library(rsvg)

# Generate process map as DiagrammeR graph
pm_graph <- target_eventlog %>%
  process_map(type = frequency("absolute"),
              layout = "layout_with_sugiyama",  # Better layout algorithm
              edge_cutoff = 0.01,  # Hide rare transitions (< 1%)
              node_label = c("activity"),
              node_size = "frequency",
              edge_label = "absolute") %>%
  render_graph()

# Export as high-quality PNG
pm_svg <- export_svg(pm_graph)
rsvg_png(charToRaw(pm_svg), 
         file.path(plots_dir, sprintf("%s_%s_frequency_map.png", cohort_name, age_band_fname)),
         width = 2400, height = 1800)  # High resolution
```

#### B. Create Interactive HTML Process Map
```r
library(visNetwork)
library(htmlwidgets)

# Build nodes and edges from eventlog
edges_df <- target_eventlog %>%
  group_by(case_id) %>%
  arrange(timestamp) %>%
  mutate(from_activity = activity,
         to_activity = lead(activity)) %>%
  ungroup() %>%
  filter(!is.na(to_activity)) %>%
  group_by(from_activity, to_activity) %>%
  summarise(frequency = n(), .groups = "drop") %>%
  arrange(desc(frequency)) %>%
  filter(frequency >= quantile(frequency, 0.25))  # Keep top 75% transitions

nodes_df <- data.frame(
  id = unique(c(edges_df$from_activity, edges_df$to_activity)),
  label = unique(c(edges_df$from_activity, edges_df$to_activity))
) %>%
  left_join(target_activity_freq, by = c("id" = "activity")) %>%
  mutate(value = count,  # Node size based on frequency
         group = activity_type,  # Color by type
         title = paste0(id, "<br>Count: ", count))  # Hover text

# Create interactive network
network <- visNetwork(nodes_df, 
                     edges_df %>% rename(from = from_activity, to = to_activity, value = frequency),
                     main = paste("Process Map:", cohort_name, age_band),
                     width = "100%", height = "800px") %>%
  visNodes(shape = "box", font = list(size = 14)) %>%
  visEdges(arrows = "to", smooth = list(type = "cubicBezier")) %>%
  visGroups(groupname = "Drug", color = list(background = "#3b82f6", border = "#1e40af")) %>%
  visGroups(groupname = "Diagnosis", color = list(background = "#ef4444", border = "#b91c1c")) %>%
  visGroups(groupname = "Procedure", color = list(background = "#10b981", border = "#047857")) %>%
  visOptions(highlightNearest = list(enabled = TRUE, degree = 1, hover = TRUE),
             selectedBy = "group") %>%
  visPhysics(stabilization = TRUE, solver = "forceAtlas2Based")

saveWidget(network, 
          file.path(plots_dir, sprintf("%s_%s_process_map_interactive.html", 
                                       cohort_name, age_band_fname)),
          selfcontained = TRUE)
```

---

### 5. Process Matrix (Drug Interactions)

**Current Implementation:**
```r
pm_target <- process_matrix(target_eventlog, type = "frequency")
pm_target_df <- as.data.frame(pm_target)
save_bupar_csv(pm_target_df, sprintf("%s_%s_train_target_process_matrix_bupar.csv", 
                                     cohort_name, age_band_fname))
```

**⚠️ CRITICAL ISSUE**: Process matrix is generated as CSV but **NOT displayed on dashboard**!

**Recommendations:**

#### A. Add Process Matrix to Dashboard HTML

**1. Update HTML (10_risk_dashboard/pgx_dashboard.html):**

```html
<!-- Add this panel inside bupar-visualizations-tab -->
<div class="panel">
  <h2>Process Matrix (Drug Interaction Patterns)</h2>
  <p style="font-size: 0.9em; color: #64748b;">
    Heatmap showing frequency of directly-follows relationships. 
    High values indicate common activity sequences (e.g., Drug A → Drug B).
  </p>
  <div id="bupar-process-matrix-image"></div>
</div>
```

**2. Update JavaScript rendering function:**

```javascript
function renderBupaRVisualizations(data) {
  const imageMap = {
    'bupar-activity-freq-image': data.activity_frequency_image,
    'bupar-pre-freq-image': data.pre_target_frequency_image,
    'bupar-sequence-image': data.sequence_image,
    'bupar-trace-explorer-image': data.trace_explorer_image,
    'bupar-trace-explorer-pre-image': data.trace_explorer_pre_image,
    'bupar-performance-spectrum-image': data.performance_spectrum_image,
    'bupar-frequency-map-image': data.frequency_map_image,
    'bupar-process-matrix-image': data.process_matrix_image  // ADD THIS
  };
  for (const [elementId, imageUrl] of Object.entries(imageMap)) {
    if (imageUrl) {
      document.getElementById(elementId).innerHTML = `<img src="${imageUrl}" style="max-width: 100%;" />`;
    }
  }
}
```

**3. Update Lambda function (10_risk_dashboard/backend/lambda_function.py):**

```python
def handle_visualizations_bupar(event: Dict[str, Any]) -> Dict[str, Any]:
    # ... existing code ...
    
    payload = {
        "activity_frequency_image": f"{base_url}/{base_key}/{cohort}_{age_band_fname}_overall_activity_frequency.png",
        "pre_target_frequency_image": f"{base_url}/{base_key}/{cohort}_{age_band_fname}_{pre_suffix}_activity_frequency.png",
        "sequence_image": f"{base_url}/{base_key}/{cohort}_{age_band_fname}_activity_sequence_top.png",
        "trace_explorer_image": f"{base_url}/{base_key}/{cohort}_{age_band_fname}_trace_explorer.png",
        "trace_explorer_pre_image": f"{base_url}/{base_key}/{cohort}_{age_band_fname}_trace_explorer_{pre_suffix}.png",
        "performance_spectrum_image": f"{base_url}/{base_key}/{cohort}_{age_band_fname}_performance_spectrum.png",
        "frequency_map_image": f"{base_url}/{base_key}/{cohort}_{age_band_fname}_frequency_map.png",
        "process_matrix_image": f"{base_url}/{base_key}/{cohort}_{age_band_fname}_process_matrix.png",  # ADD THIS
    }
    return _response(200, payload)
```

#### B. Generate Process Matrix Visualization (R Script)

Add this to `create_bupar_outputs_opioid_ed.R` after process matrix CSV export:

```r
# Generate process matrix heatmap visualization
if (!is.null(pm_target)) {
  pm_target_df <- as.data.frame(pm_target)
  
  # Convert to long format for ggplot
  pm_long <- pm_target_df %>%
    tibble::rownames_to_column("from_activity") %>%
    tidyr::pivot_longer(cols = -from_activity, 
                       names_to = "to_activity", 
                       values_to = "frequency") %>%
    filter(frequency > 0)  # Remove zero-frequency cells
  
  # Filter to top activities (reduce clutter)
  top_activities <- target_activity_freq %>% 
    head(25) %>% 
    pull(activity)
  
  pm_long_filtered <- pm_long %>%
    filter(from_activity %in% top_activities,
           to_activity %in% top_activities)
  
  # Create heatmap
  p_matrix <- ggplot(pm_long_filtered, 
                    aes(x = to_activity, y = from_activity, fill = frequency)) +
    geom_tile(color = "white", size = 0.5) +
    geom_text(aes(label = ifelse(frequency > 0, frequency, "")), 
             size = 2.5, color = "white") +
    scale_fill_viridis_c(option = "magma", 
                        trans = "log10",
                        breaks = c(1, 10, 100, 1000),
                        labels = scales::comma) +
    labs(title = paste("Process Matrix:", cohort_name, age_band),
         subtitle = "Frequency of directly-follows relationships (top 25 activities)",
         x = "To Activity →", 
         y = "← From Activity",
         fill = "Frequency\n(log scale)") +
    theme_minimal(base_size = 12) +
    theme(axis.text.x = element_text(angle = 45, hjust = 1, size = 9),
          axis.text.y = element_text(size = 9),
          panel.grid = element_blank(),
          legend.position = "right")
  
  ggsave(file.path(plots_dir, sprintf("%s_%s_process_matrix.png", cohort_name, age_band_fname)),
         plot = p_matrix, width = 16, height = 14, dpi = 300)
  
  cat("Saved process_matrix.png\n")
}
```

#### C. Create Interactive Process Matrix (Plotly)

```r
library(plotly)

# Interactive heatmap with hover details
p_matrix_interactive <- plot_ly(
  data = pm_long_filtered,
  x = ~to_activity,
  y = ~from_activity,
  z = ~frequency,
  type = "heatmap",
  colors = viridis::magma(100),
  text = ~paste("From:", from_activity, 
               "<br>To:", to_activity,
               "<br>Frequency:", frequency),
  hoverinfo = "text",
  colorbar = list(title = "Frequency")
) %>%
  layout(
    title = paste("Process Matrix (Interactive):", cohort_name, age_band),
    xaxis = list(title = "To Activity", tickangle = -45, tickfont = list(size = 10)),
    yaxis = list(title = "From Activity", tickfont = list(size = 10)),
    margin = list(l = 150, b = 150)
  )

htmlwidgets::saveWidget(p_matrix_interactive,
                       file.path(plots_dir, sprintf("%s_%s_process_matrix_interactive.html",
                                                    cohort_name, age_band_fname)),
                       selfcontained = TRUE)
```

---

## Implementation Priority

### Phase 1: Critical Fixes (Immediate)
1. **Add Process Matrix to Dashboard** - Currently missing despite being generated
2. **Fix Activity Frequency Color Coding** - Distinguish drug/ICD/CPT events
3. **Increase Trace Explorer Count** - From 20 to 30 traces for better coverage

### Phase 2: Enhanced Static Visualizations (Week 1)
1. **Performance Spectrum Segmentation** - Add time-to-target breakdown
2. **Process Matrix Heatmap** - Replace CSV with visual heatmap
3. **Improve Label Readability** - Abbreviate codes, adjust font sizes

### Phase 3: Interactive Visualizations (Week 2-3)
1. **Interactive Trace Explorer** - Plotly HTML with hover details
2. **Interactive Activity Frequency** - Sortable, filterable bar chart
3. **Interactive Process Map** - visNetwork with zoom/pan/filter

### Phase 4: Advanced Analytics (Future)
1. **Patient-Level Filtering** - On-demand R execution for custom cohorts
2. **Temporal Clustering** - Group patients by trajectory similarity (DTW)
3. **Drug Interaction Scoring** - Highlight high-risk combinations in process matrix

---

## Code Changes Summary

### Files to Modify:

1. **9_dashboard_visuals/bupar/create_bupar_outputs_opioid_ed.R**
   - Add process matrix heatmap generation (lines ~1135)
   - Add color-coded activity frequency (lines ~1170)
   - Add interactive HTML exports (optional)

2. **9_dashboard_visuals/bupar/create_bupar_outputs_non_opioid_ed.R**
   - Same changes as opioid_ed script

3. **10_risk_dashboard/pgx_dashboard.html**
   - Add process matrix panel (line ~856)
   - Update renderBupaRVisualizations() to include process_matrix_image

4. **10_risk_dashboard/backend/lambda_function.py**
   - Add process_matrix_image to payload (line ~1493)

---

## Testing Checklist

- [ ] Process matrix PNG generates successfully
- [ ] Process matrix uploads to S3 with correct path
- [ ] Dashboard displays process matrix in new panel
- [ ] Activity frequency colors distinguish drug/ICD/CPT
- [ ] Trace explorer shows 30 traces (or configurable)
- [ ] Performance spectrum renders without errors
- [ ] All visualizations load for all 16 cohort/age_band combinations
- [ ] Interactive HTML files are self-contained (no external dependencies)
- [ ] S3 bucket permissions allow public read access to visualization files

---

## Performance Considerations

### Current Bottlenecks:
1. **Large Cohorts**: Age bands 25-44 and 45-54 may have 20,000+ patients
2. **Process Map Complexity**: Too many nodes/edges cause DiagrammeR to slow down
3. **PNG File Sizes**: High-DPI images (300 DPI, 16" wide) can be 5-10 MB each

### Optimizations:
1. **Sampling for Visualization**: Use max 10,000 patients for process map/trace explorer
2. **Edge Pruning**: Filter out transitions with frequency < 1% of total
3. **WebP Format**: Convert PNGs to WebP for 30-50% smaller file sizes
4. **Lazy Loading**: Only load visualizations when tab is clicked (not all at once)

---

## Related Documentation

- [BupaR Trace Explorer](https://bupaverse.github.io/docs/trace_explorer.html)
- [BupaR Performance Spectrum](https://bupaverse.github.io/docs/performance_spectrum.html)
- [BupaR Process Matrix](https://bupaverse.github.io/docs/process_matrix.html)
- [BupaR Frequency Maps](https://bupaverse.github.io/docs/frequency_maps.html)

---

## Contact

For questions or implementation assistance, see:
- **Pipeline Code**: `9_dashboard_visuals/bupar/`
- **Output Location**: `10_risk_dashboard/visualizations/bupar/outputs/`
- **Dashboard Frontend**: `10_risk_dashboard/pgx_dashboard.html`
- **Backend API**: `10_risk_dashboard/backend/lambda_function.py`
