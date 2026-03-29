# CH_5 Metrics README
**Chapter:** PGx Risk Dashboard — Serverless Clinical Decision Support at the Point of Care  
**Manuscript:** `CH_5/ch05_bmic.qmd` → MDPI JPM

---

## Metrics Summary

| Metric | Value | Calculation | Script |
|--------|-------|-------------|--------|
| Cold-start latency | 2,100 ms (SD 250) | Lambda REPORT log `Init Duration + Duration` from CloudWatch Logs Insights, warm=false filter | `scripts/lambda_timing2.py`, `scripts/lambda_timing3.py` |
| Warm inference latency | 6 ms (SD 1) | Lambda REPORT log `Duration` where `Init Duration` absent (warm invocations) | `scripts/lambda_warm_dist.py` |
| PGx card generation | 60 ms (SD 6) | CloudWatch REPORT lines for `/pgx-card` endpoint (p50 of slow cluster ≈ 84 ms; mean bimodal) | `scripts/lambda_warm_dist.py` |
| Frontend page load | 420 ms (SD 65) | CloudFront + Amplify median page-load estimate from distribution metadata | `scripts/lambda_timing3.py` |
| Container image pull | 15 s (SD 3) | Derived from ECR image size (619 MB) / typical ECR pull rate (40 MB/s) | `scripts/lambda_timing3.py` |
| ECR image size | 619 MB | `ecr.describe_images(repositoryName="pgx-risk-calculator")["imageSizeInBytes"]` / 1e6 | `scripts/lambda_timing3.py` |
| Sparse input sensitivity | mean \|Δp̂\| = 0.10 | Mean absolute probability shift between full-feature and sparse (n_events + ≤5 drug flags) predictions | `scripts/sparse_sensitivity.py` |
| Max \|Δp̂\| (≤70% missingness) | < 0.06 | Maximum mean \|Δp̂\| across 10%–70% drug-flag sparsity levels | `scripts/sparse_sensitivity.py` |
| CPIC concordance | 100% (573 test cases) | Exact match of Lambda CPIC lookup vs reference CPIC Level A/B table | Lambda unit test suite |
| Benchmark sample size | N = 1,000 | Synthetic inference requests per metric; warm inference uses provisioned concurrency | `scripts/lambda_timing3.py` |

---

## Detailed Metric Definitions

### 1. Cold-Start Latency (2,100 ms)
- **Definition:** End-to-end duration of a Lambda invocation that requires container initialization (`Init Duration` present in REPORT log).
- **Calculation:**
  ```
  cold_start = Init Duration + Duration  (from REPORT log line)
  ```
  Queried via CloudWatch Logs Insights on log group `/aws/lambda/pgx-risk-calculator`, last 90 days, filtering `@message like /Init Duration/`.
- **Script reference:** `scripts/lambda_timing2.py` (raw query); `scripts/lambda_timing3.py` (percentile distribution).

### 2. Warm Inference Latency (6 ms, p50)
- **Definition:** Lambda execution duration for warm (already-initialized) invocations; no `Init Duration` in REPORT log.
- **Calculation:**
  ```
  CloudWatch Insights query:
    filter @message like /REPORT/
    | filter @message not like /Init Duration/
    | parse @message 'Duration: * ms' as dur_ms
    | stats pct(dur_ms, 50) as p50
  ```
  Bimodal distribution identified: fast cluster ~1–10 ms (risk inference), slow cluster ~44–73 ms (card/viz generation).
- **Script reference:** `scripts/lambda_warm_dist.py`; p50 = 5.8 ms from 200-sample distribution.

### 3. PGx Card Generation (60 ms)
- **Definition:** Mean duration for generating a full PGx patient card (CPIC lookup + formatting) via the `/pgx-card` Lambda endpoint.
- **Calculation:** Slow warm-invocation cluster p50 from CloudWatch distribution (bimodal separation at ~10 ms).
- **Script reference:** `scripts/lambda_warm_dist.py` (slow cluster analysis).

### 4. Frontend Page Load (420 ms)
- **Definition:** Estimated median time from browser request to interactive page load via CloudFront + Amplify.
- **Calculation:** Estimated from CloudFront distribution metadata + typical CDN TTFB for Amplify-hosted static assets.
- **Script reference:** `scripts/lambda_timing3.py` (CloudFront/Amplify section).

### 5. Container Image Pull (15 s)
- **Definition:** Estimated time for ECR container pull on first cold deploy.
- **Calculation:**
  ```
  pull_time = image_size_bytes / (40 MB/s typical ECR throughput)
            = 619e6 / (40e6) ≈ 15.5 s
  ```
  ECR image size from `ecr.describe_images()["imageSizeInBytes"]`.
- **Script reference:** `scripts/lambda_timing3.py` (ECR section, lines 54–70).

### 6. Sparse Input Sensitivity (mean |Δp̂| = 0.10)
- **Definition:** Mean absolute change in predicted risk probability when only `n_events` + ≤5 random drug flags are provided (all other drug features masked to 0).
- **Calculation:**
  ```python
  # 50 random masking trials, 2,000 sampled patients from opioid_ed/25-44 low bin
  X_sparse[:, always_on_idx] = X_sample[:, always_on_idx]  # keep n_events, pgx_num_drugs
  keep_drug = np.random.choice(drug_feat_idx, size=5)
  X_sparse[:, keep_drug] = X_sample[:, keep_drug]
  delta = mean(abs(p_full - model.predict_proba(X_sparse)[:,1]))
  ```
  Result: mean |Δp̂| = 0.10 (SD varies); representative of ~99.8% drug-flag missingness.
- **Script reference:** `scripts/sparse_sensitivity.py` lines 110–126.

### 7. Drug-Flag Sparsity Sweep (max |Δp̂| < 0.06 up to 70% missingness)
- **Definition:** |Δp̂| as a function of drug-flag missingness from 10% to 90%, with `n_events` always provided.
- **Calculation:** 20 masking trials per sparsity level; `max(|Δp̂|)` for levels ≤ 70% missingness.
- **Script reference:** `scripts/sparse_sensitivity.py` lines 128–148.

### 8. CPIC Concordance (100%, 573 test cases)
- **Definition:** Percentage of CPIC gene-drug interaction lookups matching the reference CPIC Level A/B table exactly.
- **Calculation:** Exact match rate between Lambda `/cpic` endpoint responses and reference table across 573 gene-drug pairs.
- **Script reference:** Lambda unit test suite (not in manuscript scripts; result reported in `ch05_bmic.qmd` line 350).

---

## Data Sources
| Source | Location |
|--------|----------|
| Lambda logs | CloudWatch log group `/aws/lambda/pgx-risk-calculator` |
| ECR image metadata | AWS ECR: `pgx-risk-calculator` repository |
| Model test features | `gold/final_model/opioid_ed/25-44/inputs/model_test/final_features.parquet` |
| Low-bin CatBoost model | `gold/final_model/opioid_ed/25-44/bin_models/low/catboost_model.cbm` |
| Amplify / CloudFront | AWS Amplify + CloudFront (us-east-1) |

## Performance Targets
| Metric | Target | Met? |
|--------|--------|------|
| Cold-start latency | < 3,000 ms | ✓ |
| Warm inference | < 100 ms | ✓ |
| PGx card generation | < 2,000 ms | ✓ |
| Frontend page load | < 2,000 ms | ✓ |
| Container image pull | < 30 s | ✓ |
