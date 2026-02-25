# BupaR visualization outputs

This directory contains **outputs only** for BupaR dashboard visualizations (plots, feature CSVs).  
**Creation code** lives in `9_dashboard_visuals/bupar/` (pipeline step 9). Do not add scripts here; follow the same pattern as `6_final_model`, `7_shap_analysis`, and `8_ffa_analysis` (code in step folder, outputs in designated output location).

**JSON vs PNG:** Prefer JSON where available for dashboard flexibility (see `9_dashboard_visuals/bupar/README_bupaR.md`). JSON available: `*_activity_frequency.json`, `*_pre_target_activity_frequency.json`, `*_post_target_activity_frequency.json` (frontend renders Chart.js from these); optional with `--export-csv-to-json`: `*_traces_top.json`, `*_traces_rare.json`, `*_pre_target_traces_*.json`. Process matrix, trace explorer, frequency map are PNG (or interactive HTML) only until pipeline adds JSON.
