# Add PHTS-style explicit resources to the pgx-risk-calculator API Gateway:
#   /metadata (GET, OPTIONS), /risk (POST, OPTIONS), /risk/comparison (POST, OPTIONS),
#   /causal/importance (POST, OPTIONS), /causal/interactions (POST, OPTIONS).
# Same Lambda proxy integration; these paths take precedence over {proxy+}.
#
# Usage:
#   .\add_api_gateway_resources_phts_style.ps1 [-Profile PROFILE]
#
# Prerequisites: API "pgx-risk-calculator" and Lambda "pgx-risk-calculator" must exist.

param(
    [string] $Profile = $env:AWS_PROFILE
)

$ErrorActionPreference = "Stop"

$Region = if ($env:AWS_REGION) { $env:AWS_REGION } else { "us-east-1" }
$ApiName = "pgx-risk-calculator"
$LambdaName = "pgx-risk-calculator"

function Run-Aws {
    param([Parameter(ValueFromRemainingArguments = $true)] [string[]] $AwsArgs)
    $base = @("--region", $Region)
    if ($Profile) { $base = @("--profile", $Profile) + $base }
    $prevErrAction = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    $out = & aws @base @AwsArgs 2>&1
    $ErrorActionPreference = $prevErrAction
    if ($LASTEXITCODE -ne 0) { throw "aws exited with $LASTEXITCODE - $out" }
    # Return stdout only (first element if mixed array from 2>&1)
    if ($null -eq $out) { return "" }
    if ($out -is [array]) { ($out | Where-Object { $_ -is [string] }) -join "`n" } else { $out }
}

function Get-ResourceId {
    param([string] $ApiId, [string] $Path)
    $out = Run-Aws apigateway get-resources --rest-api-id $ApiId --query "items[?path=='$Path'].id" --output text
    if ($null -eq $out) { return $null }
    $id = ([string]$out).Trim() -replace "`r", ""
    if ($id -eq "None" -or [string]::IsNullOrWhiteSpace($id)) { return $null }
    return $id
}

function Ensure-Resource {
    param([string] $ApiId, [string] $ParentId, [string] $PathPart, [string] $ExpectedPath)
    $id = Get-ResourceId -ApiId $ApiId -Path $ExpectedPath
    if (-not $id) {
        $raw = Run-Aws apigateway create-resource --rest-api-id $ApiId --parent-id $ParentId --path-part $PathPart --query id --output text
        $id = ([string]$raw).Trim() -replace "`r", ""
        Write-Host "Created resource $ExpectedPath ($id)"
    } else {
        Write-Host "Resource $ExpectedPath exists ($id)"
    }
    return $id
}

function Put-MethodIntegration {
    param([string] $ApiId, [string] $ResourceId, [string] $HttpMethod)
    try {
        Run-Aws apigateway put-method --rest-api-id $ApiId --resource-id $ResourceId --http-method $HttpMethod --authorization-type NONE | Out-Null
    } catch {
        if ($_.ToString() -notmatch "Method already exists") { throw }
    }
    Run-Aws apigateway put-integration --rest-api-id $ApiId --resource-id $ResourceId --http-method $HttpMethod `
        --type AWS_PROXY --integration-http-method POST --uri $script:IntegrationUri | Out-Null
    Write-Host "  $HttpMethod on resource $ResourceId"
}

Write-Host "Region: $Region  API: $ApiName  Lambda: $LambdaName"
Write-Host ""

$AccountId = ([string](Run-Aws sts get-caller-identity --query Account --output text)).Trim() -replace "`r", ""
$ApiId = ([string](Run-Aws apigateway get-rest-apis --query "items[?name=='$ApiName'].id" --output text)).Trim() -replace "`r", ""
if (-not $ApiId -or $ApiId -eq "None") {
    Write-Host "Error: API $ApiName not found. Create it first with create_api_gateway_pgx_risk_calculator.ps1"
    exit 1
}
Write-Host "API id: $ApiId"

$RootId = ([string](Run-Aws apigateway get-resources --rest-api-id $ApiId --query "items[?path=='/'].id" --output text)).Trim() -replace "`r", ""
Write-Host "Root resource id: $RootId"
Write-Host ""

$LambdaArn = "arn:aws:lambda:${Region}:${AccountId}:function:${LambdaName}"
$script:IntegrationUri = "arn:aws:apigateway:${Region}:lambda:path/2015-03-31/functions/${LambdaArn}/invocations"

$MetadataId   = Ensure-Resource -ApiId $ApiId -ParentId $RootId -PathPart "metadata"   -ExpectedPath "/metadata"
$RiskId       = Ensure-Resource -ApiId $ApiId -ParentId $RootId -PathPart "risk"       -ExpectedPath "/risk"
$ComparisonId = Ensure-Resource -ApiId $ApiId -ParentId $RiskId  -PathPart "comparison" -ExpectedPath "/risk/comparison"
$CausalId     = Ensure-Resource -ApiId $ApiId -ParentId $RootId  -PathPart "causal"     -ExpectedPath "/causal"
$ImportanceId = Ensure-Resource -ApiId $ApiId -ParentId $CausalId -PathPart "importance"  -ExpectedPath "/causal/importance"
$InteractionsId = Ensure-Resource -ApiId $ApiId -ParentId $CausalId -PathPart "interactions" -ExpectedPath "/causal/interactions"
Write-Host ""

Write-Host "Putting methods and Lambda proxy integration..."
Put-MethodIntegration -ApiId $ApiId -ResourceId $MetadataId     -HttpMethod "GET"
Put-MethodIntegration -ApiId $ApiId -ResourceId $MetadataId     -HttpMethod "OPTIONS"
Put-MethodIntegration -ApiId $ApiId -ResourceId $RiskId         -HttpMethod "POST"
Put-MethodIntegration -ApiId $ApiId -ResourceId $RiskId         -HttpMethod "OPTIONS"
Put-MethodIntegration -ApiId $ApiId -ResourceId $ComparisonId   -HttpMethod "POST"
Put-MethodIntegration -ApiId $ApiId -ResourceId $ComparisonId   -HttpMethod "OPTIONS"
Put-MethodIntegration -ApiId $ApiId -ResourceId $ImportanceId   -HttpMethod "POST"
Put-MethodIntegration -ApiId $ApiId -ResourceId $ImportanceId   -HttpMethod "OPTIONS"
Put-MethodIntegration -ApiId $ApiId -ResourceId $InteractionsId  -HttpMethod "POST"
Put-MethodIntegration -ApiId $ApiId -ResourceId $InteractionsId  -HttpMethod "OPTIONS"
Write-Host ""

Run-Aws apigateway create-deployment --rest-api-id $ApiId --stage-name prod --description "Add PHTS-style resources" | Out-Null
Write-Host "Deployed to stage: prod"
Write-Host ""
Write-Host "Resources now match PHTS-style: /metadata, /risk, /risk/comparison, /causal/importance, /causal/interactions (plus existing / and {proxy+})."
