# Create API Gateway REST API "pgx-risk-calculator" and wire it to Lambda "pgx-risk-calculator"
# (proxy integration: all paths go to Lambda). Run after Lambda exists.
#
# Usage:
#   .\create_api_gateway_pgx_risk_calculator.ps1 [-Profile PROFILE]
#   Optional: $env:AWS_REGION = "us-east-1" (default)
#
# Prerequisites: Lambda function "pgx-risk-calculator" must exist in the same account/region.

param(
    [string] $Profile = $env:AWS_PROFILE
)

$ErrorActionPreference = "Stop"

$Region = if ($env:AWS_REGION) { $env:AWS_REGION } else { "us-east-1" }
$ApiName = "pgx-risk-calculator"
$LambdaName = "pgx-risk-calculator"

function Run-Aws {
    param([string[]] $AwsArgs)
    $base = @("--region", $Region)
    if ($Profile) { $base = @("--profile", $Profile) + $base }
    & aws @base @AwsArgs
    if ($LASTEXITCODE -ne 0) { throw "aws exited with $LASTEXITCODE" }
}

Write-Host "Region: $Region  API: $ApiName  Lambda: $LambdaName"
Write-Host ""

# Account ID (for ARNs)
$AccountId = (Run-Aws sts get-caller-identity --query Account --output text).Trim()
Write-Host "Account: $AccountId"
Write-Host ""

# 1. Get or create REST API
$Existing = Run-Aws apigateway get-rest-apis --query "items[?name=='$ApiName'].id" --output text 2>$null
$ApiId = ($Existing -replace "`r", "").Trim()
if ($ApiId) {
    Write-Host "API already exists: $ApiName (id: $ApiId)"
} else {
    $ApiId = (Run-Aws apigateway create-rest-api --name $ApiName `
        --description "PGx Risk Calculator API" `
        --endpoint-configuration types=EDGE `
        --query id --output text).Trim()
    Write-Host "Created API: $ApiId"
}
Write-Host ""

# 2. Root resource ID
$RootId = (Run-Aws apigateway get-resources --rest-api-id $ApiId --query "items[?path=='/'].id" --output text).Trim()
Write-Host "Root resource id: $RootId"
Write-Host ""

# 3. Create or get {proxy+} resource
try {
    $ProxyId = (Run-Aws apigateway create-resource --rest-api-id $ApiId --parent-id $RootId `
        --path-part "{proxy+}" --query id --output text 2>$null).Trim()
} catch {
    $ProxyId = (Run-Aws apigateway get-resources --rest-api-id $ApiId --query "items[?pathPart=='{proxy+}'].id" --output text).Trim()
}
Write-Host "Proxy resource id: $ProxyId"
Write-Host ""

$LambdaArn = "arn:aws:lambda:${Region}:${AccountId}:function:${LambdaName}"
$IntegrationUri = "arn:aws:apigateway:${Region}:lambda:path/2015-03-31/functions/${LambdaArn}/invocations"

# 4. ANY method on {proxy+} with Lambda proxy integration
Run-Aws apigateway put-method --rest-api-id $ApiId --resource-id $ProxyId --http-method ANY `
    --authorization-type NONE --request-parameters "method.request.path.proxy=true" | Out-Null
Write-Host "Put method ANY on {proxy+}"

Run-Aws apigateway put-integration --rest-api-id $ApiId --resource-id $ProxyId --http-method ANY `
    --type AWS_PROXY --integration-http-method POST --uri $IntegrationUri | Out-Null
Write-Host "Put Lambda proxy integration on {proxy+}"
Write-Host ""

# 5. Method on root (/) so that GET / and OPTIONS / go to Lambda
try {
    Run-Aws apigateway put-method --rest-api-id $ApiId --resource-id $RootId --http-method ANY --authorization-type NONE | Out-Null
    Run-Aws apigateway put-integration --rest-api-id $ApiId --resource-id $RootId --http-method ANY `
        --type AWS_PROXY --integration-http-method POST --uri $IntegrationUri | Out-Null
} catch { }
Write-Host "Put ANY on root (/)"
Write-Host ""

# 6. Grant API Gateway permission to invoke Lambda
$SourceArn = "arn:aws:execute-api:${Region}:${AccountId}:${ApiId}/*"
try {
    Run-Aws lambda add-permission --function-name $LambdaName --statement-id "apigateway-invoke-$ApiId" `
        --action lambda:InvokeFunction --principal apigateway.amazonaws.com --source-arn $SourceArn | Out-Null
    Write-Host "Lambda invoke permission set"
} catch {
    Write-Host "(Lambda permission may already exist)"
}
Write-Host ""

# 7. Deploy to prod stage
try {
    Run-Aws apigateway create-deployment --rest-api-id $ApiId --stage-name prod --description "Deploy from PowerShell" | Out-Null
} catch {
    Run-Aws apigateway create-deployment --rest-api-id $ApiId --stage-name prod | Out-Null
}
Write-Host "Deployed to stage: prod"
Write-Host ""

# 8. Output invoke URL
$InvokeUrl = "https://${ApiId}.execute-api.${Region}.amazonaws.com/prod"
Write-Host "=============================================="
Write-Host "API Gateway: $ApiName (id: $ApiId)"
Write-Host "Invoke URL:  $InvokeUrl"
Write-Host "=============================================="
Write-Host "Update your frontend API_BASE to: $InvokeUrl"
