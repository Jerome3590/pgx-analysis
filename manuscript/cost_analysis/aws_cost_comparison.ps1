#Requires -Version 7.0
<#
.SYNOPSIS
    AWS architecture cost comparison: EMR vs EC2+DuckDB  |  QuickSight vs S3+Lambda+CloudFront.

.DESCRIPTION
    Queries the AWS Price List API and EC2 Spot price history to produce order-of-magnitude
    cost estimates for two architecture substitutions made in the PGx Risk Dashboard project.
    Estimates are scoped to a representative 2 TB data-transformation workload.

    IMPORTANT CAVEAT
    ----------------
    These are NOT apples-to-apples comparisons.  EMR and EC2+DuckDB differ in execution
    model, cluster provisioning, managed-service overhead, and I/O patterns.  QuickSight
    and the S3+Lambda+CloudFront stack differ in session model and feature set.  Isolated
    head-to-head benchmarking on identical datasets and transformations was not performed.
    Estimates are derived from AWS published list prices and the EC2 Spot price history API
    and are intended solely to characterise order-of-magnitude cost differences.

.PARAMETER Region
    AWS region for Spot price lookups (default: us-east-1).

.PARAMETER RuntimeHours
    Estimated wall-clock hours for the 2 TB transformation workload (default: 4).

.PARAMETER QuickSightUsers
    Number of QuickSight Author users to price (default: 3).

.PARAMETER OutputCsv
    Optional path to write the comparison table as CSV.

.EXAMPLE
    .\aws_cost_comparison.ps1
    .\aws_cost_comparison.ps1 -RuntimeHours 6 -OutputCsv .\cost_estimates.csv
#>

param(
    [string]$Region         = "us-east-1",
    [double]$RuntimeHours   = 4,
    [int]   $QuickSightUsers = 3,
    [string]$OutputCsv      = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# ── Helpers ───────────────────────────────────────────────────────────────────

function Get-OnDemandPrice {
    <#
    Queries the AWS Price List API (must target us-east-1 endpoint regardless of region).
    Returns the USD hourly on-demand price for the given service + filters.
    #>
    param(
        [string]   $ServiceCode,
        [hashtable]$Filters
    )

    $filterList = $Filters.GetEnumerator() | ForEach-Object {
        "Type=TERM_MATCH,Field=$($_.Key),Value=$($_.Value)"
    }

    $raw = aws pricing get-products `
        --service-code $ServiceCode `
        --filters $filterList `
        --region us-east-1 `
        --output json 2>$null | ConvertFrom-Json

    if (-not $raw.PriceList -or $raw.PriceList.Count -eq 0) {
        Write-Warning "No price list entry found for $ServiceCode / $($Filters | Out-String)"
        return $null
    }

    $pl   = $raw.PriceList[0] | ConvertFrom-Json
    $term = $pl.terms.OnDemand.PSObject.Properties.Value | Select-Object -First 1
    $pu   = $term.priceDimensions.PSObject.Properties.Value | Select-Object -First 1
    return [double]$pu.pricePerUnit.USD
}

function Get-SpotPrice {
    param(
        [string]$InstanceType,
        [string]$Region
    )
    $result = aws ec2 describe-spot-price-history `
        --instance-types $InstanceType `
        --product-descriptions "Linux/UNIX" `
        --region $Region `
        --max-items 1 `
        --output json 2>$null | ConvertFrom-Json

    if (-not $result.SpotPriceHistory -or $result.SpotPriceHistory.Count -eq 0) {
        Write-Warning "No Spot price found for $InstanceType in $Region"
        return $null
    }
    return [double]$result.SpotPriceHistory[0].SpotPrice
}

# ── 1.  COMPUTE PATH: EMR cluster vs EC2 Spot + DuckDB ───────────────────────
Write-Host "`n=== Compute path comparison (2 TB transformation workload) ===" -ForegroundColor Cyan
Write-Host "  Assumed runtime : $RuntimeHours hours"
Write-Host "  Region          : $Region"

# EMR cluster configuration used as ICPM baseline:
#   Master  : 1x m5.xlarge
#   Core    : 4x r5.4xlarge   (memory-optimised for 2 TB in-memory shuffle)
#   EMR software surcharge applied on top of EC2 price (~25% for EMR label)

Write-Host "`n  Fetching EMR / EC2 on-demand prices from AWS Price List API..."

$emrSurchargeRate = 0.25   # EMR adds ~25% on top of underlying EC2

$masterOD = Get-OnDemandPrice -ServiceCode "AmazonEC2" -Filters @{
    instanceType    = "m5.xlarge"
    location        = "US East (N. Virginia)"
    operatingSystem = "Linux"
    tenancy         = "Shared"
    preInstalledSw  = "NA"
    capacitystatus  = "Used"
}

$coreOD = Get-OnDemandPrice -ServiceCode "AmazonEC2" -Filters @{
    instanceType    = "r5.4xlarge"
    location        = "US East (N. Virginia)"
    operatingSystem = "Linux"
    tenancy         = "Shared"
    preInstalledSw  = "NA"
    capacitystatus  = "Used"
}

$ec2TargetOD = Get-OnDemandPrice -ServiceCode "AmazonEC2" -Filters @{
    instanceType    = "c5.18xlarge"
    location        = "US East (N. Virginia)"
    operatingSystem = "Linux"
    tenancy         = "Shared"
    preInstalledSw  = "NA"
    capacitystatus  = "Used"
}

Write-Host "  Fetching EC2 Spot price (c5.18xlarge, $Region)..."
$ec2TargetSpot = Get-SpotPrice -InstanceType "c5.18xlarge" -Region $Region

# Cost calculations
$emrEc2RawPerHr  = if ($masterOD -and $coreOD) { $masterOD + (4 * $coreOD) } else { $null }
$emrTotalPerHr   = if ($emrEc2RawPerHr)         { $emrEc2RawPerHr * (1 + $emrSurchargeRate) } else { $null }
$emrJobCost      = if ($emrTotalPerHr)            { $emrTotalPerHr * $RuntimeHours } else { $null }

$ec2ODJobCost    = if ($ec2TargetOD)   { $ec2TargetOD   * $RuntimeHours } else { $null }
$ec2SpotJobCost  = if ($ec2TargetSpot) { $ec2TargetSpot * $RuntimeHours } else { $null }

# ── 2.  VISUALISATION PATH: QuickSight vs S3+Lambda+CloudFront ───────────────
Write-Host "`n=== Visualisation path comparison (monthly, $QuickSightUsers author users) ===" -ForegroundColor Cyan

# QuickSight pricing (standard edition, annual commitment, per-author)
$qsAuthorMonthly  = 18.00   # USD/author/month (standard, annual); reader sessions excluded
$qsMonthlyTotal   = $qsAuthorMonthly * $QuickSightUsers

# S3+Lambda+CloudFront (serverless, pay-per-request)
# Lambda: 128 requests/day * 30 days = 3,840 req/month @ ~200ms each, 1 GB memory
#   Free tier 1M req/month — effectively $0 at research scale
# S3: static hosting ~1 GB frontend, <1000 req/day  → < $0.05/month
# CloudFront: ~10 GB/month transfer → ~$0.085 * 10 = $0.85/month
# API Gateway: 4 endpoints * 5 req/day * 30 days = 600 req/month → negligible
$serverlessMonthly = 0.90  # conservative upper estimate

# ── 3.  Build output table ────────────────────────────────────────────────────
$rows = @(
    [PSCustomObject]@{
        Scenario         = "ICPM baseline: EMR (1x m5.xlarge + 4x r5.4xlarge)"
        Architecture     = "AWS EMR"
        UnitCostPerHr    = if ($emrTotalPerHr)   { "`$$([math]::Round($emrTotalPerHr,2))/hr"   } else { "N/A" }
        JobOrMonthCost   = if ($emrJobCost)       { "`$$([math]::Round($emrJobCost,2)) / job"   } else { "N/A" }
        Notes            = "EC2 on-demand + $([int]($emrSurchargeRate*100))% EMR surcharge; $RuntimeHours-hr job"
    }
    [PSCustomObject]@{
        Scenario         = "Dissertation: EC2 Spot c5.18xlarge + DuckDB (on-demand)"
        Architecture     = "EC2 On-Demand"
        UnitCostPerHr    = if ($ec2TargetOD)      { "`$$([math]::Round($ec2TargetOD,2))/hr"     } else { "N/A" }
        JobOrMonthCost   = if ($ec2ODJobCost)      { "`$$([math]::Round($ec2ODJobCost,2)) / job" } else { "N/A" }
        Notes            = "72 vCPU / 144 GB RAM; $RuntimeHours-hr job; list price"
    }
    [PSCustomObject]@{
        Scenario         = "Dissertation: EC2 Spot c5.18xlarge + DuckDB (spot)"
        Architecture     = "EC2 Spot"
        UnitCostPerHr    = if ($ec2TargetSpot)    { "`$$([math]::Round($ec2TargetSpot,3))/hr"   } else { "N/A" }
        JobOrMonthCost   = if ($ec2SpotJobCost)    { "`$$([math]::Round($ec2SpotJobCost,2)) / job" } else { "N/A" }
        Notes            = "Spot price at time of query ($Region); subject to fluctuation"
    }
    [PSCustomObject]@{
        Scenario         = "ICPM baseline: AWS QuickSight ($QuickSightUsers authors, standard)"
        Architecture     = "QuickSight"
        UnitCostPerHr    = "`$$qsAuthorMonthly/author/mo"
        JobOrMonthCost   = "`$$([math]::Round($qsMonthlyTotal,2)) / month"
        Notes            = "Annual commitment; excludes reader sessions and SPICE"
    }
    [PSCustomObject]@{
        Scenario         = "Dissertation: S3 static + Lambda + CloudFront + API Gateway"
        Architecture     = "Serverless"
        UnitCostPerHr    = "Pay-per-request"
        JobOrMonthCost   = "~`$$serverlessMonthly / month"
        Notes            = "Research-scale traffic (<200 req/day); Lambda within free tier"
    }
)

# ── 4.  Display ───────────────────────────────────────────────────────────────
Write-Host ""
$rows | Format-Table -AutoSize -Wrap

# ── 5.  Manuscript-ready summary ──────────────────────────────────────────────
Write-Host "=== Manuscript summary (copy into CH_2 cost discussion) ===" -ForegroundColor Green
Write-Host @"

AWS Pricing Calculator estimates for a representative 2-TB data-transformation
workload ($RuntimeHours hours assumed runtime, $Region):

  EMR cluster (1x m5.xlarge master + 4x r5.4xlarge core, on-demand + EMR surcharge):
    ~`$$([math]::Round($emrJobCost,2)) per job   [`$([math]::Round($emrTotalPerHr,2))/hr]

  EC2 Spot c5.18xlarge + DuckDB (spot price at query time):
    ~`$$([math]::Round($ec2SpotJobCost,2)) per job  [`$([math]::Round($ec2TargetSpot,3))/hr]

  Ratio (EMR on-demand / EC2 Spot): ~$(if($ec2SpotJobCost -gt 0){[math]::Round($emrJobCost/$ec2SpotJobCost,1)}else{'N/A'})x

QuickSight ($QuickSightUsers authors, standard annual, monthly):  `$$([math]::Round($qsMonthlyTotal,2))
S3 + Lambda + CloudFront (serverless, research-scale, monthly):   ~`$$serverlessMonthly

NOTE: These estimates are derived from AWS published list prices and EC2 Spot
price history as of $(Get-Date -Format 'yyyy-MM-dd'). Direct comparison is not
strictly apples-to-apples: the two compute architectures differ in execution
model, cluster provisioning, managed-service overhead, and I/O patterns.
Isolated head-to-head benchmarking on identical datasets was not performed.
"@

# ── 6.  Optional CSV export ───────────────────────────────────────────────────
if ($OutputCsv -ne "") {
    $rows | Export-Csv -Path $OutputCsv -NoTypeInformation -Encoding UTF8
    Write-Host "`nCSV written to: $OutputCsv" -ForegroundColor Yellow
}
