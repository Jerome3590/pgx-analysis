# Frontend Dashboard

## Overview

The frontend dashboard is a single-page application (SPA) built with vanilla HTML, CSS, and JavaScript. It provides an interactive interface for risk assessment and PGx patient card generation.

## Files

- **`index.html`** - Main dashboard HTML file with all tabs and JavaScript
- **`assets/`** - Static assets (CSS, JavaScript, images) - currently inline in HTML

## Tabs

1. **Risk Assessment** - Calculate risk scores for opioid ED visits or polypharmacy
2. **Causal Analysis** - Explore FFA causal factors and SHAP importance
3. **DTW Trajectories** - View patient trajectory patterns
4. **FP-Growth Patterns** - Explore drug-name itemsets and association rules (drugs only)
5. **BupaR Process Mining** - View process flows, activity sequences, and Drug × Drug process matrix
6. **PGx Patient Card** - Generate pharmacogenomic cards

## Dependencies

- **Plotly.js** (CDN) - For interactive charts
- **Chart.js** (CDN) - For additional visualizations (if needed)

## API Integration

The frontend communicates with the Lambda backend via API Gateway:
- Base URL: Configured in `index.html` (`API_BASE` constant)
- Endpoints: See `../backend/README.md` for API documentation

## Validating frontend updates

When changing `index.html`, use the checklist in **[README_dashboard_validation.md](../../README_dashboard_validation.md) (project root)** to ensure tabs, visual headings, BupaR copy, S3 path-style URLs, and API usage stay aligned with the path mapping and research-question artifacts.

## Deployment

The frontend is deployed as a static website on S3:
- Build: No build step required (vanilla HTML/JS)
- Deploy: Upload `index.html` to S3 bucket
- CDN: Can be served via CloudFront for better performance
