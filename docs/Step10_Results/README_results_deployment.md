# Risk Dashboard Implementation Plan

## Overview

This document outlines the implementation plan for a user-facing risk assessment dashboard that provides:

1. **F1120 Opioid ED Visit Risk Score** (ages 13-64)
2. **Polypharmacy Risk Score** (ages 65+)

The dashboard uses age to automatically select the appropriate model and provides an interactive interface for exploring risk factors.

---

## Architecture

### Components

```
┌─────────────────┐
│  S3 Static Site │  ← HTML/JS Dashboard (index.html)
│  (CloudFront)   │
└────────┬────────┘
         │ HTTPS
         ▼
┌─────────────────┐
│  API Gateway    │  ← REST API Endpoints
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  AWS Lambda     │  ← Model Inference & Metadata
│  (Python 3.10+) │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  S3 Data Lake   │  ← Models, Feature Importances, Metadata
│  (pgxdatalake)  │
└─────────────────┘
```

---

## Data Sources

### Cohort 1: Opioid ED (`opioid_ed`)
- **Age Range**: 13-64 (age bands: 13-24, 25-44, 45-54, 55-64)
- **Models**: `8_final_model/outputs/opioid_ed/{age_band}/models/`
- **Feature Importances**: `3_feature_importance/outputs/opioid_ed_{age_band}_aggregated_feature_importance.csv`
- **Input Features**: Age, ICD codes, CPT codes, Drug names
- **Note**: Age band 0-12 excluded due to small cohort size

### Cohort 2: Polypharmacy (`non_opioid_ed`)
- **Age Range**: 65-114 (age bands: 65-74, 75-84, 85-94)
- **Models**: `8_final_model/outputs/non_opioid_ed/{age_band}/models/`
- **Feature Importances**: `3_feature_importance/outputs/non_opioid_ed_{age_band}_aggregated_feature_importance.csv`
- **Input Features**: Age, Drug names (single or combinations)
- **Note**: Ages 95-114 are mapped to age band 85-94 (uses 85-94 model) due to small cohort size

---

## Implementation Steps

### Step 1: Generate Metadata Files

**Script**: `generate_metadata.py`

**Purpose**: Extract valid codes (ICD, CPT, Drug) from feature importance files and create metadata JSON files for each cohort/age_band combination.

**Output Structure**:
```json
{
  "cohort": "opioid_ed",
  "age_bands": ["13-24", "25-44", "45-54", "55-64"],
  "codes": {
    "13-24": {
      "drugs": [
        {"code": "AMOXICILLIN", "display": "Amoxicillin", "importance": 0.85},
        ...
      ],
      "icds": [
        {"code": "R51", "display": "Headache", "importance": 0.95},
        {"code": "G89", "display": "Pain, not elsewhere classified", "importance": 0.87},
        ...
      ],
      "cpts": [
        {"code": "80305", "display": "Drug test, definitive", "importance": 0.87},
        ...
      ]
    },
    ...
  }
}
```

**S3 Location**: `s3://pgxdatalake/gold/dashboard/metadata/metadata_{cohort}.json`

---

### Step 2: Prepare Models for Lambda Container (ECR)

**Script**: `prepare_models.py`

**Purpose**: Package models and feature schemas for Lambda container deployment.

**Actions**:
1. Load models from `8_final_model/outputs/`
2. Extract feature schemas
3. Create model packages in `models/` directory for Docker build
4. Models will be bundled in container image (up to 10GB supported)

**Model Package Structure**:
```
models/
├── opioid_ed/
│   ├── 13_24/
│   │   ├── catboost.joblib (or .json)
│   │   ├── xgboost.joblib
│   │   ├── xgboost_rf.joblib
│   │   └── feature_schema.json
│   └── ...
└── non_opioid_ed/
    └── ...
```

**Deployment**: Models are bundled directly in the container image at `/var/task/models/` for fast loading. S3 is used as fallback for metadata and as backup.

**Note**: Lambda with ECR supports up to 10GB container images, so all models can be included directly in the image for optimal performance.

---

### Step 3: Build Lambda Function

**File**: `lambda_function.py`

**Endpoints**:

#### 3.1 GET /metadata
**Purpose**: Return valid age bands and code lists for a cohort.

**Query Parameters**:
- `cohort` (required): `opioid_ed` or `non_opioid_ed`

**Response**:
```json
{
  "age_bands": ["13-24", "25-44", ...],
  "codes": {
    "13-24": {
      "drugs": [...],
      "icds": [...],
      "cpts": [...]
    },
    ...
  }
}
```

#### 3.2 POST /risk
**Purpose**: Calculate risk score using ensemble of all three models (CatBoost, XGBoost, XGBoost RF).

**Request Body**:
```json
{
  "age": 35,
  "cohort": "opioid_ed",  // auto-determined from age if not provided
  "drugs": ["AMOXICILLIN", "METHYLPHENIDATE HYDROCHLO"],
  "icds": ["R51", "G89"],  // Note: F1120 is excluded (it's the target, not an input)
  "cpts": ["80305", "99213"]
}
```

**Response**:
```json
{
  "risk_score": 0.65,
  "risk_band": "high",
  "model_breakdown": {
    "catboost": 0.64,
    "xgboost": 0.66,
    "xgboost_rf": 0.65
  },
  "ensemble_info": {
    "method": "weighted_average",
    "models_used": 3,
    "models_failed": [],
    "weights": {
      "catboost": 1.0,
      "xgboost": 1.0,
      "xgboost_rf": 1.0
    }
  },
  "age_band_used": "25-44",
  "cohort_used": "opioid_ed"
}
```

**Ensemble Logic**:
1. Determine cohort from age (13-64 → `opioid_ed`, 65-114 → `non_opioid_ed`)
2. Determine age band from age:
   - Validates age is 13-114 (excludes 0-12)
   - Maps ages 95-114 to age band 85-94 (uses 85-94 model)
3. Load all three models (CatBoost, XGBoost, XGBoost RF) from container/S3 (with caching)
4. Load model weights from `feature_schema.json` (calculated from MC-CV performance metrics)
5. Build feature vector from inputs
6. Run predictions on all three models
7. Combine predictions using **performance-based weighted average**:
   - Weights based on composite score: `0.5 × PR-AUC + 0.5 × (1/(1+logloss))`
   - Weights normalized to sum to 1.0
   - Only successful models contribute to ensemble
8. Return ensemble risk score with per-model breakdown and weight information
9. If any model fails, ensemble continues with remaining models (weights renormalized)

#### 3.3 POST /risk/comparison
**Purpose**: Compare risk scores for different combinations.

**Request Body**:
```json
{
  "base": {
    "age": 35,
    "drugs": ["AMOXICILLIN"],
    "icds": [],
    "cpts": []
  },
  "scenarios": [
    {
      "name": "Add Headache",
      "drugs": ["AMOXICILLIN"],
      "icds": ["R51"],
      "cpts": []
    },
    {
      "name": "Add Drug Test",
      "drugs": ["AMOXICILLIN"],
      "icds": [],
      "cpts": ["80305"]
    }
  ]
}
```

**Response**:
```json
{
  "base_risk": 0.15,
  "scenarios": [
    {
      "name": "Add Headache",
      "risk_score": 0.45,
      "delta": 0.30
    },
    {
      "name": "Add Drug Test",
      "risk_score": 0.28,
      "delta": 0.13
    }
  ]
}
```

---

### Step 4: Build Dashboard UI

**File**: `index.html`

**Features**:

1. **Age Input**
   - Number input field
   - Auto-detects cohort and age band
   - Shows which model will be used

2. **Code Selection Dropdowns**
   - Populated from `/metadata` endpoint
   - Multi-select dropdowns
   - Grouped by type (Drugs, ICDs, CPTs)
   - Show importance scores next to codes
   - Search/filter capability

3. **Risk Score Display**
   - Large, prominent risk score (0-100%)
   - Color-coded risk band (low/medium/high)
   - Model breakdown chart (bar chart)
   - Risk distribution histogram

4. **Comparison Mode**
   - Side-by-side comparison of scenarios
   - Show delta changes
   - Visual indicators for risk increases/decreases

5. **Responsive Design**
   - Mobile-friendly
   - Accessible (WCAG 2.1 AA)

**Technology Stack**:
- Pure HTML/CSS/JavaScript (no build step)
- Plotly.js for visualizations
- Fetch API for backend calls

---

### Step 5: Deploy Infrastructure

#### 5.1 S3 Static Website Hosting

```bash
# Upload dashboard
aws s3 cp index.html s3://pgxdatalake/dashboard/index.html --content-type text/html

# Enable static website hosting
aws s3 website s3://pgxdatalake/dashboard/ --index-document index.html

# Set bucket policy for public read access
aws s3api put-bucket-policy --bucket pgxdatalake --policy file://bucket-policy.json
```

**Optional**: Use CloudFront for:
- HTTPS
- Custom domain
- Caching
- Better performance

#### 5.2 Lambda Container Deployment (ECR)

**Container Structure**:
```
lambda-container/
├── Dockerfile
├── lambda_function.py
├── requirements.txt
└── models/          (prepared by prepare_models.py)
    ├── opioid_ed/
    └── non_opioid_ed/
```

**Build and Deploy**:
```bash
# 1. Prepare models
python prepare_models.py --all

# 2. Build and push container image
./docker_build.sh

# Or manually:
docker build -t pgx-risk-dashboard:latest .
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com
docker tag pgx-risk-dashboard:latest <ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com/pgx-risk-dashboard:latest
docker push <ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com/pgx-risk-dashboard:latest

# 3. Create/update Lambda function from container image
aws lambda create-function \
  --function-name pgx-risk-dashboard-api \
  --package-type Image \
  --code ImageUri=<ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com/pgx-risk-dashboard:latest \
  --role arn:aws:iam::<ACCOUNT_ID>:role/lambda-execution-role \
  --timeout 60 \
  --memory-size 3008 \
  --environment Variables="{PGX_RESULTS_BUCKET=pgxdatalake,MODEL_CACHE_TTL=3600}"
```

**Lambda Configuration**:
- **Package Type**: Container Image (ECR)
- **Container Image**: Up to 10GB supported
- **Memory**: 3008 MB (for model inference)
- **Timeout**: 60 seconds (allows time for cold starts)
- **Environment Variables**:
  - `PGX_RESULTS_BUCKET=pgxdatalake`
  - `MODEL_CACHE_TTL=3600` (seconds)
  - `MODEL_BASE_PATH=/var/task/models` (default, models bundled in container)

**Model Loading Strategy**:
1. **Primary**: Load from container filesystem (`/var/task/models/`) - fastest, no network latency
2. **Fallback**: Load from S3 if container models not available (for development/testing)

**IAM Permissions**:
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:GetObject"
      ],
      "Resource": [
        "arn:aws:s3:::pgxdatalake/gold/*",
        "arn:aws:s3:::pgxdatalake/gold/dashboard/*"
      ]
    },
    {
      "Effect": "Allow",
      "Action": [
        "logs:CreateLogGroup",
        "logs:CreateLogStream",
        "logs:PutLogEvents"
      ],
      "Resource": "arn:aws:logs:*:*:*"
    }
  ]
}
```

#### 5.3 API Gateway Setup

**Configuration**:
- Type: HTTP API (or REST API)
- Integration: Lambda Proxy
- CORS: Enabled
- Authentication: None (or API Key for production)

**Endpoints**:
- `GET /metadata`
- `POST /risk`
- `POST /risk/comparison`

---

## Feature Engineering Logic

### Building Feature Vectors

**For Opioid ED (Cohort 1)**:
```python
def build_feature_vector(age, drugs, icds, cpts, feature_schema):
    """
    Build feature vector matching the model's expected schema.
    
    Features are typically:
    - item_{DRUG_NAME}: binary (1 if present)
    - item_{ICD_CODE}: binary (1 if present)
    - item_{CPT_CODE}: binary (1 if present)
    - trajectory_*: numeric (set to 0 or median if not available)
    - pre_*: numeric (set to 0 or median if not available)
    - itemset_*: binary (set to 0 if not available)
    """
    features = {}
    
    # Initialize all features to 0
    for feature in feature_schema['features']:
        features[feature] = 0.0
    
    # Set age-related features
    features['age'] = age
    
    # Set item features (drugs, ICDs, CPTs)
    for drug in drugs:
        feature_name = f"item_{drug.upper()}"
        if feature_name in features:
            features[feature_name] = 1.0
    
    for icd in icds:
        feature_name = f"item_{icd.upper()}"
        if feature_name in features:
            features[feature_name] = 1.0
    
    for cpt in cpts:
        feature_name = f"item_{cpt.upper()}"
        if feature_name in features:
            features[feature_name] = 1.0
    
    # Set default values for trajectory/sequence features
    # (These would ideally come from patient history, but for dashboard
    #  we use median/default values)
    for feature in features:
        if feature.startswith('trajectory_') or feature.startswith('pre_'):
            if features[feature] == 0.0:
                features[feature] = feature_schema['defaults'].get(feature, 0.0)
    
    return features
```

**For Polypharmacy (Cohort 2)**:
- Similar logic but focuses on drug combinations
- May include drug-drug interaction features

---

## Age-Based Model Selection

```python
def determine_cohort_and_age_band(age):
    """
    Determine which cohort and age band to use based on age.
    
    Rules:
    - Ages 13-64: opioid_ed cohort
    - Ages 65+: non_opioid_ed (polypharmacy) cohort
    """
    if age < 13:
        raise ValueError("Age must be 13 or older")
    elif age >= 13 and age <= 64:
        cohort = "opioid_ed"
        if age <= 24:
            age_band = "13-24"
        elif age <= 44:
            age_band = "25-44"
        elif age <= 54:
            age_band = "45-54"
        else:
            age_band = "55-64"
    else:  # age >= 65
        cohort = "non_opioid_ed"
        if age <= 74:
            age_band = "65-74"
        elif age <= 84:
            age_band = "75-84"
        else:
            age_band = "85-94"
    
    return cohort, age_band
```

---

## Testing Strategy

### Unit Tests
- Feature vector building
- Age-based cohort selection
- Model loading and inference
- Metadata parsing

### Integration Tests
- Lambda function with mock S3
- API Gateway endpoints
- End-to-end dashboard flow

### Manual Testing
- Test with various age inputs
- Test with different code combinations
- Verify risk score calculations
- Test comparison mode

---

## Security Considerations

1. **Input Validation**
   - Validate age ranges
   - Sanitize code inputs
   - Prevent injection attacks

2. **Rate Limiting**
   - Implement API Gateway throttling
   - Consider usage limits

3. **Data Privacy**
   - No PII stored
   - No patient data in requests
   - Logging excludes sensitive data

4. **Access Control**
   - Consider API keys for production
   - CORS configuration
   - S3 bucket policies

---

## Performance Optimization

1. **Model Bundling (ECR Container)**
   - Models bundled directly in container image (up to 10GB)
   - No S3 download latency on cold starts
   - Models available immediately at `/var/task/models/`
   - Fastest possible model loading

2. **Model Caching**
   - Cache loaded models in Lambda container memory
   - Use Lambda container reuse (models persist across invocations)
   - No need to reload models on warm starts

3. **Lazy Loading**
   - Load models on-demand (only when needed)
   - Models cached after first load

4. **CDN Caching**
   - Cache static assets (dashboard HTML/JS)
   - Cache metadata responses

5. **Parallel Processing**
   - Run ensemble models in parallel
   - Use async I/O for S3 calls (metadata only)

---

## Monitoring & Logging

### CloudWatch Metrics
- Lambda invocation count
- Lambda duration
- Lambda errors
- API Gateway 4xx/5xx errors

### CloudWatch Logs
- Request/response logging
- Error details
- Performance metrics

### Alarms
- High error rate
- High latency
- Lambda timeout errors

---

## Deployment Checklist

- [ ] Generate metadata files for all cohorts/age_bands
- [ ] Upload models to S3
- [ ] Deploy Lambda function
- [ ] Create API Gateway
- [ ] Upload dashboard HTML to S3
- [ ] Configure S3 static website hosting
- [ ] Set up CloudFront (optional)
- [ ] Configure IAM permissions
- [ ] Test all endpoints
- [ ] Test dashboard UI
- [ ] Set up monitoring/alarms
- [ ] Document API endpoints
- [ ] Create user guide

---

## File Structure

```
10_results/
├── README_DASHBOARD_IMPLEMENTATION.md  (this file)
├── generate_metadata.py                (Step 1)
├── prepare_models.py                   (Step 2)
├── lambda_function.py                  (Step 3)
├── index.html                         (Step 4)
├── requirements.txt                   (Lambda dependencies)
├── test/
│   ├── test_lambda.py
│   ├── test_feature_building.py
│   └── test_metadata.py
└── deployment/
    ├── terraform/                      (Infrastructure as Code)
    ├── cloudformation/                 (Alternative IaC)
    └── scripts/
        ├── deploy_lambda.sh
        └── deploy_dashboard.sh
```

---

## Next Steps

1. **Immediate**: Generate metadata files from existing feature importance CSVs
2. **Short-term**: Build Lambda function with model loading logic
3. **Short-term**: Create dashboard UI with basic functionality
4. **Medium-term**: Add comparison mode and advanced features
5. **Medium-term**: Deploy to AWS and test end-to-end
6. **Long-term**: Add authentication, rate limiting, and production hardening

---

## References

- Feature Importance Files: `3_feature_importance/outputs/`
- Final Models: `8_final_model/outputs/`
- FFA Analysis: `9_ffa_analysis/outputs/`
- S3 Data Lake: `s3://pgxdatalake/gold/`

