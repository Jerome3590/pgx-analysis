# S3 CORS configuration for dashboard bucket

When the frontend (origin `https://jerome-dixon.io`) fetches **direct S3 URLs** (e.g. `https://s3.us-east-1.amazonaws.com/jerome-dixon.io/vcu/pgx-risk-calculator/causal/.../causal_data.json`), the browser treats that as **cross-origin**. S3 must return `Access-Control-Allow-Origin` or the request is blocked by CORS.

## When CORS is needed

- **Same-origin requests** (e.g. `https://jerome-dixon.io/vcu/pgx-risk-calculator/metadata/opioid_ed.json` via CloudFront): **no CORS** required.
- **Direct S3 URLs** (path-style `https://s3.region.amazonaws.com/bucket/key`) used by the frontend (e.g. `causal_data_url`, `chart_data_url`, DTW/FP-Growth/BupaR asset URLs): **CORS required** on the bucket.

The Lambda often returns data **inline** (e.g. `causal_data`, `chart_data`). When it cannot (e.g. object missing in the bucket Lambda uses), it only returns a URL and the frontend fetches that URL — that fetch is cross-origin to S3 and needs CORS.

## Apply CORS to the dashboard bucket

Bucket: **jerome-dixon.io** (or the value of `S3_DASHBOARD_BUCKET`).

**Deployment workflow:** Notebook 5 **Step 6** (Sync Dashboard Frontend to S3) runs `apply_dashboard_bucket_cors.py` before syncing, so CORS is applied idempotently on every deploy. No manual step needed for production or when adding new visuals.

**Manual / one-off:**

1. **Using the deployment script (recommended)**
   ```bash
   python 10_risk_dashboard/deployment/apply_dashboard_bucket_cors.py
   # Optional: --bucket NAME, --config PATH, --region us-east-1, --check (print current only)
   ```

2. **Using AWS Console**
   - Open **S3** → bucket **jerome-dixon.io** → **Permissions**.
   - Under **Cross-origin resource sharing (CORS)**, edit and paste the configuration from `s3-cors-config.json` (or the JSON below).

3. **Using AWS CLI**
   ```bash
   aws s3api put-bucket-cors --bucket jerome-dixon.io --cors-configuration file://10_risk_dashboard/docs/s3-cors-config.json
   ```

## Example CORS configuration

File: `10_risk_dashboard/docs/s3-cors-config.json` (format required by `aws s3api put-bucket-cors`)

```json
{
  "CORSRules": [
    {
      "AllowedHeaders": ["*"],
      "AllowedMethods": ["GET", "HEAD"],
      "AllowedOrigins": ["https://jerome-dixon.io", "http://localhost:5500", "http://127.0.0.1:5500"],
      "ExposeHeaders": [],
      "MaxAgeSeconds": 3600
    }
  ]
}
```

- **AllowedOrigins:** Add your dashboard origin(s). Include `https://jerome-dixon.io` and any dev origins (e.g. localhost) you use.
- **AllowedMethods:** `GET` and `HEAD` are enough for loading JSON and assets.
- **AllowedHeaders:** `*` allows any request headers (e.g. `Accept`).

After saving, direct S3 URL fetches from the dashboard origin will receive `Access-Control-Allow-Origin` and the browser will allow the response.

## Optional: avoid CORS by not using direct S3 URLs

To avoid CORS entirely for causal/DTW/FP-Growth data:

- Have the **Lambda** always return data **inline** when it can read from S3 (it already does for causal when the object exists).
- Or serve those assets **same-origin** (e.g. under `https://jerome-dixon.io/vcu/pgx-risk-calculator/...`) via CloudFront so the frontend never hits S3 directly. Then no S3 CORS is needed for those requests.

Applying the CORS config above is the minimal change so existing direct S3 URL usage works.

---

## 403 Forbidden on direct S3 URL

If the request reaches S3 but returns **403 (Forbidden)**, the bucket is denying anonymous `GetObject` for that key.

### 1. Apply the bucket policy (public read for dashboard prefix)

The dashboard bucket must allow public `GetObject` for the prefix used by the app (e.g. `vcu/pgx-risk-calculator/*`).

**CLI** (from repo root, bucket name = `jerome-dixon.io`):

```bash
aws s3api put-bucket-policy --bucket jerome-dixon.io --policy file://10_risk_dashboard/docs/s3-public-read-policy.json
```

**Policy file** (`10_risk_dashboard/docs/s3-public-read-policy.json`): allows `s3:GetObject` for `arn:aws:s3:::jerome-dixon.io/vcu/pgx-risk-calculator/*`.

### 2. Block Public Access can override the policy

If the bucket has **Block public access** enabled (S3 → bucket → Permissions → Block public access), the policy above may still result in 403 because S3 blocks “public” bucket policies by default.

- **Option A:** Edit **Block public access** and **uncheck** “Block public access to buckets and objects granted through new public bucket or access point policies”. Save. Then the bucket policy’s `Principal "*"` will take effect for the dashboard prefix.
- **Option B (recommended long-term):** Do not make S3 public. Serve dashboard assets **only via CloudFront** (same origin as the app, e.g. `https://jerome-dixon.io/vcu/pgx-risk-calculator/...`). Configure CloudFront with an **Origin Access Control (OAC)** so it can read from S3 without public access. The frontend then never uses direct S3 URLs for causal/DTW/FP-Growth; it uses same-origin URLs, so no CORS and no public S3.

### 3. Confirm the object exists

Ensure `causal_data.json` (and other assets) are uploaded to the bucket under the correct key, e.g.:

`vcu/pgx-risk-calculator/causal/opioid_ed/25-44/causal_data.json`

If the object is missing, fix the upload step (e.g. `combine_shap_ffa_results --upload-to-dashboard` or your pipeline).
