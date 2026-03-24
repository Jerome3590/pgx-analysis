# Lambda IAM Policy for S3 Access (pgxdatalake)

The Lambda function (`pgx-risk-calculator`) reads models, metadata, and analytics data from S3 bucket `pgxdatalake`. If the execution role lacks `s3:GetObject`, you will see:

```
AccessDenied: User: arn:aws:sts::535362115856:assumed-role/pgx-lambda-role/pgx-risk-calculator
is not authorized to perform: s3:GetObject on resource:
"arn:aws:s3:::pgxdatalake/gold/dashboard/models/opioid_ed/25_44/risk_distribution_2019.json"
because no identity-based policy allows the s3:GetObject action
```

## Fix: Attach S3 Read Policy to Lambda Role

Attach an inline policy or create a managed policy that grants `s3:GetObject` on the required prefixes.

### Option 1: Inline policy (CLI)

```bash
aws iam put-role-policy \
  --role-name pgx-lambda-role \
  --policy-name PgxS3ReadGold \
  --policy-document '{
    "Version": "2012-10-17",
    "Statement": [
      {
        "Effect": "Allow",
        "Action": ["s3:GetObject", "s3:ListBucket"],
        "Resource": [
          "arn:aws:s3:::pgxdatalake",
          "arn:aws:s3:::pgxdatalake/gold/*"
        ]
      }
    ]
  }'
```

### Option 2: Least-privilege (specific prefixes only)

If you prefer to scope access to only the paths Lambda uses:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": ["s3:GetObject"],
      "Resource": [
        "arn:aws:s3:::pgxdatalake/gold/dashboard/*",
        "arn:aws:s3:::pgxdatalake/gold/ffa_analysis/*",
        "arn:aws:s3:::pgxdatalake/gold/shap_analysis/*",
        "arn:aws:s3:::pgxdatalake/gold/final_model/*"
      ]
    }
  ]
}
```

Save as `lambda-s3-policy.json` and attach:

```bash
aws iam put-role-policy \
  --role-name pgx-lambda-role \
  --policy-name PgxS3ReadGold \
  --policy-document file://lambda-s3-policy.json
```

### Paths Lambda reads from pgxdatalake

| Prefix | Purpose |
|--------|---------|
| `gold/dashboard/models/` | Models, risk_distribution_2019.json, feature_schema.json, n_event_bin_thresholds |
| `gold/dashboard/metadata/` | metadata_{cohort}.json, model_performance_metrics.json |
| `gold/dashboard/data/` | cpic_gene-drug_pairs (PGx card) |
| `gold/ffa_analysis/` | causal_importance.parquet, interaction_analysis |
| `gold/shap_analysis/` | SHAP importance CSVs |
| `gold/final_model/` | Per-bin feature importance (fallback) |

## Verify

After attaching the policy, invoke the Lambda and check that it can load `risk_distribution_2019.json` and other S3 assets without AccessDenied.
