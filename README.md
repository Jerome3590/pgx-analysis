# Prescription Drug Analysis with FpGrowth, BupaR and CatBoost Integration

End-to-end workflow for feature discovery, noise reduction, and causal-oriented modeling using drug exposures, ICD/CPT codes, and classification outcomes.

## 📚 Documentation

This project is organized into four main sections:

1. **[Overview](README_overview.md)** - Project structure, components, and high-level workflow
2. **[Data Pipeline](README_data_pipeline.md)** - Data processing, cohort creation, and data flow
3. **[Analysis Workflow](README_analysis_workflow.md)** - Feature importance, pattern mining, and final model development
4. **[Data Visualizations](README_data_visualizations.md)** - Visualization approaches, interpretation, and network analysis

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Configure AWS credentials for S3 access
aws configure
```

## Repository Structure

```
pgx-analysis/
├── 1_apcd_input_data/          # Data preprocessing and cleaning
├── 2_create_cohort/            # Cohort creation and QA
├── 3_feature_importance/       # MC-CV feature importance analysis
├── 4_fpgrowth_analysis/        # Frequent pattern mining
├── 5_bupaR_analysis/           # Process mining
├── 6_dtw_analysis/             # Trajectory analysis
├── 7_final_model/              # Final model development
├── 8_ffa_analysis/             # Feature attribution analysis
├── helpers_1997_13/            # Utility functions
└── docs/                       # Documentation
```

## High-Level Workflow

```mermaid
flowchart TD
    subgraph "Phase 1: Data Preparation"
        A1[APCD Input Data] --> A2[Data Cleaning]
        A2 --> A3[Cohort Creation]
        A3 --> A4[Quality Assurance]
    end
    
    subgraph "Phase 2: Feature Discovery"
        A4 --> B1[Monte Carlo CV]
        B1 --> B2[Feature Importance - Model Ensemble]
        B2 --> B3[Top Features Selection]
    end
    
    subgraph "Phase 3: Pattern Mining"
        B3 --> C1[FPGrowth Analysis<br/>Frequent Itemsets]
        C1 --> C2[BupaR Process Mining<br/>Temporal Pathways]
        C2 --> C3[DTW Trajectory Analysis<br/>Patient Clustering]
    end
    
    subgraph "Phase 4: Feature Engineering"
        C1 --> D1[FPGrowth Features<br/>Itemsets & Rules]
        C2 --> D2[BupaR Features<br/>Process Patterns]
        C3 --> D3[DTW Features<br/>Trajectory Clusters]
        D1 --> D4[Final Feature Schema]
        D2 --> D4
        D3 --> D4
    end
    
    subgraph "Phase 5: Final Model"
        D4 --> E1[Feature Integration]
        E1 --> E2[CatBoost Training]
        E1 --> E3[Random Forest Training]
        E2 --> E4[Model Evaluation]
        E3 --> E4
        E4 --> E5[Feature Attribution]
    end
    
    subgraph "Phase 6: Causal Analysis"
        E5 --> F1[Tree Export JSON]
        F1 --> F2[Subgroup Identification]
        F2 --> F3[Causal Inference]
    end
    
    style A1 fill:#f9f,stroke:#333,stroke-width:2px
    style B2 fill:#bbf,stroke:#333,stroke-width:2px
    style C1 fill:#bfb,stroke:#333,stroke-width:2px
    style C2 fill:#bfb,stroke:#333,stroke-width:2px
    style C3 fill:#bfb,stroke:#333,stroke-width:2px
    style D4 fill:#fbb,stroke:#333,stroke-width:2px
    style E2 fill:#fbb,stroke:#333,stroke-width:2px
    style E3 fill:#fbb,stroke:#333,stroke-width:2px
    style F3 fill:#f9f,stroke:#333,stroke-width:2px
```

## Key Features

- **Feature Screening** with a focused model ensemble (CatBoost, XGBoost boosted trees, XGBoost RF mode) + Monte Carlo cross-validation
- **Structure Discovery** and noise reduction with FP-Growth, process mining (BupaR), and dynamic time warping (DTW)
- **Final Model Development** combining features from all analysis methods for prediction and causal inference

## Related Documentation

- [`docs/README_create_cohort.md`](docs/README_create_cohort.md) - Comprehensive cohort creation guide
- [`docs/README_feature_importance.md`](docs/README_feature_importance.md) - Feature importance analysis
- [`docs/README_fpgrowth.md`](docs/README_fpgrowth.md) - FP-Growth pattern mining
- [`docs/README_bupaR.md`](docs/README_bupaR.md) - Process mining with BupaR
- [`docs/README_dtw_feature_extraction.md`](docs/README_dtw_feature_extraction.md) - DTW trajectory analysis
