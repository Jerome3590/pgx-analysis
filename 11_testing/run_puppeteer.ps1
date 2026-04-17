<#
.SYNOPSIS
    Run Puppeteer E2E tests against the production PGx dashboard.

.PARAMETER Pattern
    Jest testPathPattern (default: tests/)

.PARAMETER Suite
    Shortcut: combo | viz | card | all  (default: all)
#>
param(
    [string]$Pattern = "",
    [ValidateSet("all","combo","viz","card")]
    [string]$Suite = "all"
)

$env:DASHBOARD_URL = "https://jerome-dixon.io/vcu/pgx-risk-calculator/index.html"
$env:API_BASE_URL  = "https://cmv0qislq3.execute-api.us-east-1.amazonaws.com/prod"

$puppeteerDir = Join-Path $PSScriptRoot "puppeteer"

if (-not (Test-Path (Join-Path $puppeteerDir "node_modules"))) {
    Write-Host "Installing npm deps..." -ForegroundColor Yellow
    Push-Location $puppeteerDir
    & npm install
    Pop-Location
}

$pat = switch ($Suite) {
    "combo" { "tests/combinatorial" }
    "viz"   { "tests/viz" }
    "card"  { "tests/pgx-card" }
    default { if ($Pattern) { $Pattern } else { "tests/" } }
}

Write-Host ""
Write-Host "DASHBOARD_URL : $env:DASHBOARD_URL" -ForegroundColor Cyan
Write-Host "API_BASE_URL  : $env:API_BASE_URL"  -ForegroundColor Cyan
Write-Host "Pattern       : $pat"               -ForegroundColor Cyan
Write-Host ""

Push-Location $puppeteerDir
& npx jest --testPathPattern=$pat --forceExit --verbose
$exit = $LASTEXITCODE
Pop-Location

exit $exit
