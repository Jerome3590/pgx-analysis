# AWS Architecture Cost Comparison

Supporting analysis for **CH_2 Methods** of the PGx Risk Dashboard dissertation.
Produces order-of-magnitude cost estimates for two architecture substitutions
made relative to the ICPM 2024 baseline platform.

## Architecture substitutions

| ICPM 2024 baseline | Dissertation implementation | Rationale |
|:---|:---|:---|
| AWS EMR Studio Lab | EC2 Spot Instance + DuckDB | Eliminate managed-cluster overhead; single-node columnar engine sufficient for partition-first workload |
| AWS QuickSight | S3 static hosting + AWS Lambda + API Gateway + CloudFront | Eliminate per-author/session licensing; serverless pay-per-request at research-scale traffic |

## Prerequisites

- AWS CLI v2 configured with credentials that have at minimum:
  - `pricing:GetProducts` (IAM policy: `AWSPricingReadOnlyAccess`)
  - `ec2:DescribeSpotPriceHistory`
- PowerShell 7+
- `us-east-1` access (the AWS Price List API is a global endpoint served from us-east-1)

### Verify credentials

```powershell
aws sts get-caller-identity
aws pricing describe-services --region us-east-1 --output table | Select-String "ElasticMapReduce|AmazonEC2"
```

## Usage

```powershell
# Default: 4-hour runtime, us-east-1, 3 QuickSight authors
.\aws_cost_comparison.ps1

# Custom runtime and output CSV
.\aws_cost_comparison.ps1 -RuntimeHours 6 -OutputCsv .\cost_estimates.csv

# Different region Spot price
.\aws_cost_comparison.ps1 -Region us-east-2 -RuntimeHours 4

# More QuickSight author seats
.\aws_cost_comparison.ps1 -QuickSightUsers 5
```

## What the script queries

### Compute path (EMR vs EC2 + DuckDB)

| Component | API call | Notes |
|:---|:---|:---|
| EMR master node (m5.xlarge) | `aws pricing get-products --service-code AmazonEC2` | On-demand list price |
| EMR core nodes (4× r5.4xlarge) | `aws pricing get-products --service-code AmazonEC2` | On-demand list price × 4 |
| EMR software surcharge | Computed | +25% applied on top of EC2 price (AWS standard EMR rate) |
| EC2 c5.18xlarge on-demand | `aws pricing get-products --service-code AmazonEC2` | List price; 72 vCPU / 144 GB RAM |
| EC2 c5.18xlarge Spot | `aws ec2 describe-spot-price-history` | Current Spot bid in target region |

### Visualisation path (QuickSight vs serverless)

| Component | Source | Notes |
|:---|:---|:---|
| QuickSight Author | Hardcoded list price | $18.00/author/month, standard edition, annual commitment |
| S3 + Lambda + CloudFront | Hardcoded estimate | ~$0.90/month upper bound at research-scale traffic (<200 req/day); Lambda within free tier (1M req/month) |

## Interpreting the output

The script prints:

1. **Formatted table** — per-hour and per-job/per-month costs for each scenario.
2. **Manuscript summary block** — copy-pastable text with the ratio and the caveat.

### Important caveats (printed in output and required in manuscript)

- Prices are from the **AWS published list price** and **EC2 Spot price history** as of the
  date the script is run. Spot prices fluctuate.
- The EMR and EC2+DuckDB architectures **differ in execution model**, cluster provisioning,
  managed-service overhead, and I/O patterns. This is **not** a controlled head-to-head
  benchmark.
- QuickSight and the serverless stack differ in session model, feature set, and
  multi-user concurrency handling.
- Isolated benchmarking on identical datasets and identical transformations was **not**
  performed.
- These figures are cited in the dissertation as **order-of-magnitude estimates** derived
  from AWS Pricing Calculator methodology, not as empirical measurements.

## EMR cluster configuration assumed

```
Master node : 1× m5.xlarge   (4 vCPU / 16 GB)
Core nodes  : 4× r5.4xlarge  (16 vCPU / 128 GB each) — memory-optimised for 2 TB shuffle
EMR surcharge: 25% on top of underlying EC2 on-demand price
```

This configuration was chosen as a reasonable minimum for a managed 2 TB
in-memory transformation workload. The actual ICPM paper used EMR Studio Lab
notebooks rather than a dedicated transformation cluster; the configuration
above is a representative proxy.

## EC2 + DuckDB configuration (dissertation)

```
Instance    : c5.18xlarge  (72 vCPU / 144 GB RAM)
Pricing     : EC2 Spot (us-east-1 default)
Engine      : DuckDB (single-node columnar, partition-first)
Checkpoints : S3 (fault-tolerant recovery — no cluster restart cost)
```

## Adding results to the manuscript

After running the script, insert the ratio and the caveat footnote into the
CH_2 Methods cost discussion paragraph.  The fig-insights TikZ diagram already
contains the architecture-substitution callouts; the body text should provide
the supporting numbers.

Example prose (fill in values from script output):

> AWS Pricing Calculator estimates for a representative 2-TB transformation
> workload (4-hour assumed runtime, us-east-1) indicate an order-of-magnitude
> cost difference: the EMR cluster configuration totalled approximately
> \$XX.XX per job versus \$X.XX per job for the EC2 Spot + DuckDB approach.
> For the visualisation layer, QuickSight standard edition (N author licences)
> costs approximately \$XX/month versus an estimated \$<1/month for the
> S3 + Lambda + CloudFront serverless stack at research-scale traffic.
> These figures are not strictly apples-to-apples: the two compute
> architectures differ in execution model, cluster provisioning, and I/O
> patterns, and isolated head-to-head benchmarking on identical datasets
> was not performed.

## Files

```
cost_analysis/
├── aws_cost_comparison.ps1   # main script
└── README.md                 # this file
```

Output files (not committed):
```
cost_estimates.csv            # optional CSV export (-OutputCsv flag)
```
