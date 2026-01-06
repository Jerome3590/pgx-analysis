#!/usr/bin/env bash
# Clear old model outputs to force regeneration
# Usage: ./utility_scripts/clear_models.sh [--cohort <cohort>] [--age-band <age_band>] [--s3] [--all]

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# Default: clear all cohorts/age bands, including S3
CLEAR_ALL=true
CLEAR_S3=true
COHORT=""
AGE_BAND=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --cohort)
            COHORT="$2"
            CLEAR_ALL=false
            shift 2
            ;;
        --age-band)
            AGE_BAND="$2"
            CLEAR_ALL=false
            shift 2
            ;;
    --s3)
        CLEAR_S3=true
        shift
        ;;
    --no-s3)
        CLEAR_S3=false
        shift
        ;;
        --all)
            CLEAR_ALL=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--cohort <cohort>] [--age-band <age_band>] [--s3] [--no-s3] [--all]"
            echo ""
            echo "Options:"
            echo "  --cohort <cohort>     Clear specific cohort (requires --age-band)"
            echo "  --age-band <age_band> Clear specific age band (requires --cohort)"
            echo "  --s3                  Clear S3 paths (default: enabled)"
            echo "  --no-s3               Skip clearing S3 paths"
            echo "  --all                 Clear all cohorts/age bands (default)"
            exit 1
            ;;
    esac
done

echo "=========================================="
echo "Clearing Model Outputs"
echo "=========================================="

if [ "$CLEAR_ALL" = true ]; then
    echo "Clearing ALL model outputs..."
    MODEL_DIR="$PROJECT_ROOT/6_final_model/outputs"
    if [ -d "$MODEL_DIR" ]; then
        echo "Removing: $MODEL_DIR"
        echo "  This includes all cohort/age_band model outputs:"
        echo "    - Model selection metadata JSON files"
        echo "    - Final features CSV files (train_final_features_no_leakage.csv)"
        echo "    - Model JSON files (XGBoost, CatBoost) in final_model_json/"
        echo "    - Model binaries (.joblib, .cbm) in models/"
        echo "    - Feature importance CSV files"
        rm -rf "$MODEL_DIR"
        echo "✅ Local model outputs cleared"
    else
        echo "No local model outputs found at $MODEL_DIR"
    fi
    
    MODEL_OUTPUTS_DIR="$PROJECT_ROOT/6_final_model/model_outputs"
    if [ -d "$MODEL_OUTPUTS_DIR" ]; then
        echo "Removing: $MODEL_OUTPUTS_DIR"
        echo "  This includes all mirrored model files for FFA/SHAP analysis"
        rm -rf "$MODEL_OUTPUTS_DIR"
        echo "✅ Model outputs directory cleared"
    fi
    
    if [ "$CLEAR_S3" = true ]; then
        echo ""
        echo "Clearing S3 model outputs..."
        aws s3 rm s3://pgxdatalake/gold/final_model/ --recursive || {
            echo "⚠️  Warning: Could not clear S3 models (may not exist or no permissions)"
        }
        echo "✅ S3 model outputs cleared"
    fi
else
    # Clear specific cohort/age band
    if [ -z "$COHORT" ] || [ -z "$AGE_BAND" ]; then
        echo "Error: --cohort and --age-band must be specified together"
        exit 1
    fi
    
    AGE_BAND_FNAME=$(echo "$AGE_BAND" | tr '-' '_')
    echo "Clearing models for cohort=$COHORT, age_band=$AGE_BAND"
    
    MODEL_DIR="$PROJECT_ROOT/6_final_model/outputs/$COHORT/$AGE_BAND_FNAME"
    if [ -d "$MODEL_DIR" ]; then
        echo "Removing: $MODEL_DIR"
        echo "  This includes:"
        echo "    - Model selection metadata JSON"
        echo "    - Final features CSV (train_final_features_no_leakage.csv)"
        echo "    - Model JSON files (XGBoost, CatBoost)"
        echo "    - Model binaries (.joblib, .cbm)"
        echo "    - Feature importance CSV"
        rm -rf "$MODEL_DIR"
        echo "✅ Local model outputs cleared for $COHORT/$AGE_BAND"
    else
        echo "No local model outputs found at $MODEL_DIR"
    fi
    
    MODEL_OUTPUTS_DIR="$PROJECT_ROOT/6_final_model/model_outputs/$COHORT/$AGE_BAND_FNAME"
    if [ -d "$MODEL_OUTPUTS_DIR" ]; then
        echo "Removing: $MODEL_OUTPUTS_DIR"
        echo "  This includes mirrored model files for FFA/SHAP"
        rm -rf "$MODEL_OUTPUTS_DIR"
        echo "✅ Model outputs directory cleared"
    fi
    
    if [ "$CLEAR_S3" = true ]; then
        echo ""
        echo "Clearing S3 model outputs for $COHORT/$AGE_BAND..."
        aws s3 rm "s3://pgxdatalake/gold/final_model/$COHORT/$AGE_BAND/" --recursive || {
            echo "⚠️  Warning: Could not clear S3 models (may not exist or no permissions)"
        }
        echo "✅ S3 model outputs cleared"
    fi
fi

echo ""
echo "=========================================="
echo "Model clearing complete!"
echo "Next run will regenerate models from scratch."
echo "=========================================="

