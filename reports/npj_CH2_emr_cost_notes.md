# EMR vs fat Spot EC2 cost attribution (npj CH2)

Source: AWS Cost Explorer (`mushin`, account 535362115856), UnblendedCost.
Retrieved 2026-07-16.

## Attribution rule

1. **EMR instance types** = instance classes appearing under service
   `Amazon Elastic MapReduce` (`BoxUsage:<type>`).
2. **EMR EC2** = EC2 `BoxUsage`/`SpotUsage` for those **same** instance types
   in the same month.
3. **EMR all-in** = EMR surcharge + EMR EC2.
4. **Fat Spot (DuckDB path)** = `x2iedn.*` (and related `x2i*` Spot) on EC2
   that do **not** appear on the EMR service list that month.
   Manuscript per-pass comparator remains published Spot rate × ~4 h
   (~\$6–\$10 on x2iedn.8xlarge-class).

## Peak / log months

| Month | EMR surcharge | EMR-matched EC2 | EMR all-in | Fat Spot not on EMR list | Notes |
|:------|-------------:|----------------:|-----------:|-------------------------:|:------|
| 2024-12 | 5,192.74 | 17,773.89 | **22,966.63** | 0.00 | Peak; m5.8xlarge / m2.2xlarge / c7g.16xlarge fleet |
| 2025-01 | 2,035.04 | 2,580.60 | 4,615.64 | 0.00 | EMR list includes some x2iedn sizes |
| 2025-04 | 549.80 | 381.43 | 931.23 | 17.65 | Spot x2iedn.24xlarge fat-ish |
| 2025-05 | 173.39 | 1,164.72 | 1,338.11 | 423.60 | Spot **x2iedn.8xlarge \$277**; EMR notebook exit 137 |

### Dec 2024 EMR types
`c7g.16xlarge`, `c7g.8xlarge`, `m2.2xlarge`, `m5.2xlarge`, `m5.8xlarge`, `m5a.12xlarge`

### May 2025 fat Spot detail (not on EMR list that month)
- SpotUsage:x2iedn.8xlarge — \$277.21
- BoxUsage:x2iedn.2xlarge — \$95.47
- SpotUsage:x2idn.16xlarge — \$43.19
- SpotUsage:x2iedn.24xlarge — \$7.73

### May 2025 log fingerprint
Cluster `j-14B24584C2HKR` under
`s3://aws-logs-535362115856-us-east-1/elasticmapreduce/`
- `toPandas at <stdin>:70` (EMR Studio / interactive Spark)
- APCD fields: `mi_person_key`, `hcg_setting`, `hcg_line`, …
- Executor exit status **137**; ResultStage cancelled ~556 s
