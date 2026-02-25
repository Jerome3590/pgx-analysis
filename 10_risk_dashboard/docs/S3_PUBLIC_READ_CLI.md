# S3 public read for dashboard (Option 1) – CLI

Allow unauthenticated `GetObject` only for the dashboard prefix so HTML and images render.

**Bucket:** `jerome-dixon.io`  
**Prefix:** `vcu/pgx-risk-calculator/*`

## 1. Remove block public access (if set)

Otherwise the bucket policy cannot grant public access:

```bash
aws s3api delete-public-access-block --bucket jerome-dixon.io
```

If you see `NoSuchPublicAccessBlockConfiguration`, the bucket has no block; skip or ignore.

## 2. (Optional) Get existing bucket policy

If the bucket already has a policy, merge the new statement into it and use the merged JSON in step 3:

```bash
aws s3api get-bucket-policy --bucket jerome-dixon.io
```

## 3. Apply bucket policy

From repo root (so the path to the JSON is correct):

```bash
cd 10_risk_dashboard/docs
aws s3api put-bucket-policy --bucket jerome-dixon.io --policy file://s3-public-read-policy.json
```

Or with absolute path:

```bash
aws s3api put-bucket-policy --bucket jerome-dixon.io --policy file:///c:/Projects/pgx-analysis/10_risk_dashboard/docs/s3-public-read-policy.json
```

## 4. Verify

Open a dashboard URL in a browser (e.g. network topology or FP-Growth combined rules HTML). They should load instead of 403.

To test from CLI:

```bash
curl -I "https://jerome-dixon.io.s3.amazonaws.com/vcu/pgx-risk-calculator/index.html"
```

Expect `200 OK` (and no `x-amz-server-side-encryption` required for this test).
