# Dashboard Incremental Build & Deployment

## Overview

The dashboard can be built and deployed **incrementally** - it doesn't require all cohorts to be complete. The system gracefully handles missing cohorts/age_bands and works with whatever is available.

## How It Works

### 1. **Step 10: Per-Cohort/Age-Band Preparation**

During the workflow (`run_cohort_workflow.sh`), Step 10 prepares dashboard artifacts for each cohort/age_band **independently**:

```bash
# Step 10 runs automatically for each cohort/age_band
python 10_risk_dashboard/prepare_models.py --cohort opioid_ed --age-band 13-24
python 10_risk_dashboard/generate_metadata.py --cohort opioid_ed --age-band 13-24
```

**Outputs:**
- Models: `10_results/models/{cohort}/{age_band}/` (catboost.joblib, xgboost.joblib, feature_schema.json)
- Metadata: `10_results/metadata/metadata_{cohort}.json` (updated incrementally)

### 2. **Incremental Dashboard Build**

When ready to build the dashboard (can be done anytime):

```bash
See 10_risk_dashboard/deployment/ or archived/utility_scripts/build_dashboard.sh
```

**What it does:**
1. Checks which cohorts/age_bands have completed Step 6 (models exist)
2. Prepares models for **only available** cohorts
3. Generates metadata for **only available** cohorts
4. Skips missing cohorts gracefully

**Example output:**
```
✓ opioid_ed: N age band(s) available (e.g. 13-24, 25-44, …, 85-114)
✓ non_opioid_ed: N age band(s) available (same full set)

Preparing models for available cohorts...
  Processing cohort: opioid_ed
  Processing cohort: non_opioid_ed

Generating metadata for available cohorts...
  Processing cohort: opioid_ed
  Processing cohort: non_opioid_ed
```

(Both cohorts use the full set of age bands: 0-12, 13-24, 25-44, 45-54, 55-64, 65-74, 75-84, 85-114.)

### 3. **Docker Build (Incremental)**

The Docker build script (`10_risk_dashboard/docker_build.sh`) works with whatever models are available:

```bash
cd 10_risk_dashboard
./docker_build.sh
```

**What it does:**
- Checks if `models/` directory exists
- If empty/missing, runs `prepare_models.py --all` (which skips missing cohorts)
- Builds Docker image with **available models only**
- Missing cohorts are simply not included in the image

### 4. **Lambda Function (Graceful Handling)**

The Lambda function (`lambda_function.py`) handles missing models gracefully:

**Metadata Endpoint (`GET /metadata?cohort=...`):**
- Returns only age bands that have models available
- Returns 404 with helpful message if cohort not found
- Filters out missing age bands automatically

**Risk Endpoint (`POST /risk`):**
- Returns 404 with helpful message if model not available
- Works with partial models (doesn't require all 3 model types)
- Uses ensemble of available models only

**Example responses:**

**Missing cohort:**
```json
{
  "error": "No models available for cohort 'opioid_ed'",
  "message": "The cohort 'opioid_ed' has not been trained yet. Please wait for the pipeline to complete.",
  "cohort": "opioid_ed"
}
```

**Missing age band:**
```json
{
  "error": "Model not available for opioid_ed/25-44",
  "message": "The model for cohort 'opioid_ed' and age band '25-44' has not been trained yet...",
  "cohort": "opioid_ed",
  "age_band": "25-44"
}
```

## Deployment Workflow

### Option 1: Incremental Deployment (Recommended)

1. **Run workflows** for cohorts as they complete:
   ```bash
   archived/utility_scripts/run_cohort_workflow.sh opioid_ed 13-24
   archived/utility_scripts/run_cohort_workflow.sh opioid_ed 25-44
   # ... etc
   ```

2. **Build dashboard** when ready (with available cohorts):
   ```bash
   See 10_risk_dashboard/deployment/ or archived/utility_scripts/build_dashboard.sh
   ```

3. **Build Docker image**:
   ```bash
   cd 10_risk_dashboard
   ./docker_build.sh
   ```

4. **Deploy to AWS**:
   - Push Docker image to ECR
   - Create/update Lambda function
   - Configure API Gateway

5. **Dashboard works** with available cohorts only
   - Users see only available age bands in dropdowns
   - Missing cohorts show helpful error messages
   - As more cohorts complete, rebuild and redeploy

### Option 2: Full Deployment (After All Cohorts Complete)

1. **Wait for all cohorts** to complete Steps 3-9
2. **Build dashboard** with all cohorts:
   ```bash
   See 10_risk_dashboard/deployment/ or archived/utility_scripts/build_dashboard.sh
   ```
3. **Build and deploy** Docker image

## Key Features

### ✅ **Incremental Support**
- Dashboard can be built/deployed with partial data
- Missing cohorts are handled gracefully
- No failures if some cohorts aren't ready

### ✅ **Graceful Degradation**
- Lambda returns helpful error messages for missing models
- Frontend can show "Coming soon" for unavailable age bands
- Works with whatever models are available

### ✅ **Idempotent Builds**
- Can rebuild dashboard anytime
- Only processes available cohorts
- Skips missing ones automatically

### ✅ **User-Friendly Errors**
- Clear error messages when models aren't available
- Suggests what needs to be done
- Doesn't crash the dashboard

## Example Scenarios

### Scenario 1: Partial Deployment
- **Completed:** e.g. `opioid_ed/13-24`, `opioid_ed/25-44`
- **Not completed:** Other cohort/age_band combinations
- **Result:** Dashboard works for completed cohort/age_band combinations only
- **User experience:** Risk and visualizations work for available combinations; others show "Model not available" or empty

### Scenario 2: One Cohort Complete
- **Completed:** All age bands for `opioid_ed` (or `non_opioid_ed`)
- **Not completed:** All age bands for the other cohort
- **Result:** Full functionality for one cohort tab; other cohort tab shows missing models for its bands
- **User experience:** User can switch cohort tab; available cohort works for all its age bands

### Scenario 3: All Cohorts Complete
- **Completed:** All cohort/age_band combinations (both cohorts, full age band set)
- **Result:** Full dashboard functionality
- **User experience:** Both Opioid ED and Polypharmacy tabs work for all age bands (0-12 through 85-114)

## Best Practices

1. **Build incrementally** - Don't wait for all cohorts
2. **Document availability** - Let users know which age bands are available
3. **Rebuild when ready** - Rebuild dashboard as more cohorts complete
4. **Monitor errors** - Check Lambda logs for missing model requests
5. **Update frontend** - Show availability status in UI

## Summary

✅ **Dashboard CAN be built without all cohorts**  
✅ **Lambda handles missing models gracefully**  
✅ **Frontend can show partial functionality**  
✅ **Incremental deployment is supported**  
✅ **No failures if some cohorts aren't ready**

The dashboard is designed to work incrementally - you can deploy it anytime with whatever cohorts are available, and it will gracefully handle missing ones.

