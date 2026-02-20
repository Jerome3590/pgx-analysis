# Production HTML pattern for dashboard visuals

Interactive HTML outputs (Plotly / htmlwidgets) must render correctly when served from S3, in iframes, or opened locally. Use a **single self-contained file** with all JS/CSS embedded so there are no external dependencies.

## FP-Growth (Python/Plotly)

- **Implementation:** `py_helpers.create_fpgrowth_visualizations.write_plotly_html_for_production()`
- **Pattern:** `fig.write_html(path, config={"responsive": True, "displayModeBar": True}, include_plotlyjs=True)`
- **Template:** `4_fpgrowth_analysis`; dashboard pipeline uses same helper in `9_dashboard_visuals/fpgrowth` via `create_plots.py` → `create_all_fpgrowth_plots()`.

## BupaR (R/htmlwidgets)

- **Implementation:** `htmlwidgets::saveWidget(..., selfcontained = TRUE)` in `create_bupar_outputs_*.R`
- **Pattern:** Single HTML file with embedded Plotly/HTMLWidgets (no `libdir`). Equivalent to FP-Growth’s `include_plotlyjs=True`.
- **Do not use** `selfcontained = FALSE` with `libdir = "lib"` for dashboard outputs; the `lib/` folder is often missing when files are synced or served from S3, which leads to blank or “builder” UI instead of the chart.

## Summary

| Pipeline   | Writer                          | Option / behavior                    |
|-----------|----------------------------------|--------------------------------------|
| FP-Growth | `write_plotly_html_for_production()` | `include_plotlyjs=True`              |
| BupaR     | `saveWidget(..., selfcontained = TRUE)` | Embed all deps in one file           |

Both produce one HTML file per chart that renders in production without extra assets.
