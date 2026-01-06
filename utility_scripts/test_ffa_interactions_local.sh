#!/bin/bash
# Test FFA multi-feature interaction analysis locally
# Downloads CatBoost model from S3 and runs interaction analysis

set -e

# Default values (use 13-24 as it has models available)
COHORT="${1:-opioid_ed}"
AGE_BAND="${2:-13-24}"
AGE_BAND_FNAME=$(echo "$AGE_BAND" | tr '-' '_')
S3_BUCKET="${S3_BUCKET:-pgxdatalake}"
ENABLE_INTERACTIONS="${ENABLE_INTERACTIONS:-true}"

echo "=========================================="
echo "Testing FFA Multi-Feature Interactions"
echo "=========================================="
echo "Cohort: $COHORT"
echo "Age Band: $AGE_BAND"
echo "S3 Bucket: $S3_BUCKET"
echo "Enable Interactions: $ENABLE_INTERACTIONS"
echo ""

# Set up paths
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_DIR="$PROJECT_ROOT/6_final_model/outputs/$COHORT/$AGE_BAND_FNAME/final_model_json"
DATA_DIR="$PROJECT_ROOT/6_final_model/outputs/$COHORT/$AGE_BAND_FNAME"
SHAP_DIR="$PROJECT_ROOT/8_shap_analysis/outputs/$COHORT/$AGE_BAND_FNAME"
OUTPUT_DIR="$PROJECT_ROOT/7_ffa_analysis/outputs/$COHORT/$AGE_BAND_FNAME"

echo "Creating directories..."
mkdir -p "$MODEL_DIR"
mkdir -p "$DATA_DIR"
mkdir -p "$SHAP_DIR"
mkdir -p "$OUTPUT_DIR"

# Download CatBoost model from S3
echo ""
echo "Downloading CatBoost model from S3..."
# Try multiple S3 path patterns (based on actual S3 structure)
MODEL_S3_PATHS=(
    "s3://$S3_BUCKET/gold/final_model/$COHORT/$AGE_BAND/${COHORT}_${AGE_BAND_FNAME}_best_catboost_model.json"
    "s3://$S3_BUCKET/gold/final_model/$COHORT/$AGE_BAND/final_model_json/${COHORT}_${AGE_BAND_FNAME}_best_catboost_model.json"
    "s3://$S3_BUCKET/gold/final_model/$COHORT/$AGE_BAND_FNAME/${COHORT}_${AGE_BAND_FNAME}_best_catboost_model.json"
    "s3://$S3_BUCKET/gold/final_model/$COHORT/$AGE_BAND_FNAME/final_model_json/${COHORT}_${AGE_BAND_FNAME}_best_catboost_model.json"
    "s3://$S3_BUCKET/gold/dashboard/models/$COHORT/$AGE_BAND_FNAME/catboost.json"
)

MODEL_DOWNLOADED=false
for MODEL_S3_PATH in "${MODEL_S3_PATHS[@]}"; do
    if aws s3 ls "$MODEL_S3_PATH" > /dev/null 2>&1; then
        aws s3 cp "$MODEL_S3_PATH" "$MODEL_DIR/${COHORT}_${AGE_BAND_FNAME}_best_catboost_model.json"
        echo "✓ Downloaded CatBoost model from: $MODEL_S3_PATH"
        MODEL_DOWNLOADED=true
        break
    fi
done

if [ "$MODEL_DOWNLOADED" = false ]; then
    echo "✗ CatBoost model not found at any of the following paths:"
    for path in "${MODEL_S3_PATHS[@]}"; do
        echo "    - $path"
    done
    echo ""
    echo "Please check S3 paths or ensure the model exists."
    exit 1
fi

# Download feature data
echo ""
echo "Downloading feature data from S3..."
FEATURE_S3_PATHS=(
    "s3://$S3_BUCKET/gold/final_model/$COHORT/$AGE_BAND/${COHORT}_${AGE_BAND_FNAME}_train_final_features_no_leakage.csv"
    "s3://$S3_BUCKET/gold/final_model/$COHORT/$AGE_BAND_FNAME/${COHORT}_${AGE_BAND_FNAME}_train_final_features_no_leakage.csv"
    "s3://$S3_BUCKET/gold/final_model/$COHORT/$AGE_BAND/inputs/model_train/final_features.parquet"
    "s3://$S3_BUCKET/gold/final_model/$COHORT/$AGE_BAND_FNAME/inputs/model_train/final_features.parquet"
)

FEATURE_DOWNLOADED=false
for FEATURE_S3_PATH in "${FEATURE_S3_PATHS[@]}"; do
    if aws s3 ls "$FEATURE_S3_PATH" > /dev/null 2>&1; then
        if [[ "$FEATURE_S3_PATH" == *.parquet ]]; then
            # If it's a parquet file, we need to convert it or use it directly
            # For now, download and note that FFA expects CSV
            echo "⚠ Found Parquet file: $FEATURE_S3_PATH"
            echo "  FFA analysis expects CSV format. Checking for CSV version..."
        else
            aws s3 cp "$FEATURE_S3_PATH" "$DATA_DIR/${COHORT}_${AGE_BAND_FNAME}_train_final_features_no_leakage.csv"
            echo "✓ Downloaded feature data from: $FEATURE_S3_PATH"
            FEATURE_DOWNLOADED=true
            break
        fi
    fi
done

if [ "$FEATURE_DOWNLOADED" = false ]; then
    echo "⚠ Feature data not found on S3. Will try to use local file if it exists"
    if [ ! -f "$DATA_DIR/${COHORT}_${AGE_BAND_FNAME}_train_final_features_no_leakage.csv" ]; then
        echo "✗ Local feature file also not found."
        echo ""
        echo "The feature CSV file contains:"
        echo "  - Binary features (0/1) for each important code from aggregated feature importance"
        echo "  - PGx features (from Step 5)"
        echo "  - One row per patient with target column"
        echo ""
        echo "To generate this file, you need:"
        echo "  1. Aggregated feature importance CSV (from Step 3)"
        echo "  2. Raw event data: model_events.parquet (from Step 4a)"
        echo "  3. PGx features CSV (from Step 5)"
        echo ""
        echo "Run Step 6 to generate it:"
        echo "  python 6_final_model_selection/run_final_model.py --cohort $COHORT --age_band $AGE_BAND"
        echo ""
        echo "Or download manually from S3 if it exists elsewhere."
        exit 1
    else
        echo "✓ Using local feature file"
    fi
fi

# Download SHAP values (required for interaction analysis)
echo ""
echo "Downloading SHAP values from S3..."
SHAP_GLOBAL_S3="s3://$S3_BUCKET/gold/shap_analysis/$COHORT/$AGE_BAND/${COHORT}_${AGE_BAND_FNAME}_shap_global_importance_catboost.csv"
SHAP_VALUES_S3="s3://$S3_BUCKET/gold/shap_analysis/$COHORT/$AGE_BAND/${COHORT}_${AGE_BAND_FNAME}_shap_sample_values_catboost.parquet"

if aws s3 ls "$SHAP_GLOBAL_S3" > /dev/null 2>&1; then
    aws s3 cp "$SHAP_GLOBAL_S3" "$SHAP_DIR/${COHORT}_${AGE_BAND_FNAME}_shap_global_importance_catboost.csv"
    echo "✓ Downloaded SHAP global importance"
else
    echo "✗ SHAP global importance not found. Required for interaction analysis."
    exit 1
fi

if aws s3 ls "$SHAP_VALUES_S3" > /dev/null 2>&1; then
    aws s3 cp "$SHAP_VALUES_S3" "$SHAP_DIR/${COHORT}_${AGE_BAND_FNAME}_shap_sample_values_catboost.parquet"
    echo "✓ Downloaded SHAP sample values (Parquet)"
else
    echo "✗ SHAP sample values not found. Required for interaction analysis."
    exit 1
fi

# Check if Python environment is set up
echo ""
echo "Checking Python environment..."
if ! command -v python3 &> /dev/null; then
    echo "✗ python3 not found"
    exit 1
fi

# Check for required packages
echo "Checking required packages..."
python3 -c "import duckdb" 2>/dev/null || { echo "✗ duckdb not installed. Install with: pip install duckdb"; exit 1; }
python3 -c "import pandas" 2>/dev/null || { echo "✗ pandas not installed"; exit 1; }
python3 -c "import numpy" 2>/dev/null || { echo "✗ numpy not installed"; exit 1; }

echo "✓ Python environment ready"

# Run FFA analysis with interaction analysis enabled
echo ""
echo "=========================================="
echo "Running FFA Analysis with Interactions"
echo "=========================================="

cd "$PROJECT_ROOT"

# Run with interaction analysis enabled
# Note: Use 'catboost' as model type to test with CatBoost model
python3 7_ffa_analysis/run_full_ffa_analysis.py \
    --cohort-name "$COHORT" \
    --age-band "$AGE_BAND" \
    --model-type catboost \
    --enable-interaction-analysis \
    --max-interaction-size 2 \
    --interaction-top-k 20 \
    --interaction-sample-size 100

echo ""
echo "=========================================="
echo "Analysis Complete"
echo "=========================================="
echo ""
echo "Check outputs in: $OUTPUT_DIR/catboost/"
echo "  - interaction_analysis.csv (multi-feature interactions)"
echo "  - causal_importance.csv (univariate causal analysis)"
echo "  - feature_importance_axp.csv (AXP feature importance)"
echo ""

