<#
.SYNOPSIS
    Run all PGx Risk Dashboard tests against the production site.

.DESCRIPTION
    Sets production URLs and runs:
      1. pytest  — combinatorial Lambda/live-API tests (test_combinatorial_risk.py)
      2. cli     — PowerShell combinatorial CLI test (cli_test_risk.ps1)
      3. puppet  — Puppeteer E2E browser tests (11_testing/puppeteer/)

    Production endpoints:
      Dashboard : https://jerome-dixon.io/vcu/pgx-risk-calculator/index.html
      API Base  : https://cmv0qislq3.execute-api.us-east-1.amazonaws.com/prod

.PARAMETER Suite
    Which suite(s) to run: pytest | cli | puppet | all   (default: all)

.PARAMETER Viz
    Also run visualization endpoints in the CLI suite.

.PARAMETER Headless
    Run Puppeteer in headless mode (default: true).

.EXAMPLE
    # Run everything
    .\11_testing\run_prod_tests.ps1

.EXAMPLE
    # pytest live API only
    .\11_testing\run_prod_tests.ps1 -Suite pytest

.EXAMPLE
    # CLI with viz endpoints
    .\11_testing\run_prod_tests.ps1 -Suite cli -Viz

.EXAMPLE
    # Puppeteer only (headless Chromium)
    .\11_testing\run_prod_tests.ps1 -Suite puppet
#>

param(
    [ValidateSet("all","pytest","cli","puppet")]
    [string]$Suite    = "all",
    [switch]$Viz,
    [bool]  $Headless = $true
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot

# ── Production URLs ─────────────────────────────────────────────────────────
$env:BASE_URL         = "https://cmv0qislq3.execute-api.us-east-1.amazonaws.com/prod"
$env:DASHBOARD_URL    = "https://jerome-dixon.io/vcu/pgx-risk-calculator/index.html"
$env:API_BASE_URL     = $env:BASE_URL

Write-Host ""
Write-Host "======================================================" -ForegroundColor Cyan
Write-Host " PGx Risk Dashboard — Production Test Run" -ForegroundColor Cyan
Write-Host "======================================================" -ForegroundColor Cyan
Write-Host "  Dashboard : $env:DASHBOARD_URL" -ForegroundColor Cyan
Write-Host "  API Base  : $env:BASE_URL" -ForegroundColor Cyan
Write-Host "  Suite     : $Suite" -ForegroundColor Cyan
Write-Host "======================================================" -ForegroundColor Cyan
Write-Host ""

$overallPass = $true

# ── 1. pytest live API ───────────────────────────────────────────────────────
if ($Suite -in @("all","pytest")) {
    Write-Host "── [1/3] pytest combinatorial (live API) ──" -ForegroundColor Yellow
    Write-Host "   pytest 11_testing/tests/test_combinatorial_risk.py -v -k live"
    Write-Host ""

    Set-Location $RepoRoot
    & python -m pytest 11_testing/tests/test_combinatorial_risk.py -v -k "live" `
        --tb=short --no-header -q
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  [FAIL] pytest live tests had failures." -ForegroundColor Red
        $overallPass = $false
    } else {
        Write-Host "  [PASS] pytest live tests." -ForegroundColor Green
    }
    Write-Host ""
}

# ── 2. PowerShell CLI test ──────────────────────────────────────────────────
if ($Suite -in @("all","cli")) {
    Write-Host "── [2/3] CLI combinatorial runner ──" -ForegroundColor Yellow
    Write-Host "   cli_test_risk.ps1 -BaseUrl $env:BASE_URL"
    Write-Host ""

    $cliArgs = @("-BaseUrl", $env:BASE_URL)
    if ($Viz) { $cliArgs += "-Viz" }

    & "$RepoRoot\11_testing\cli_test_risk.ps1" @cliArgs
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  [FAIL] CLI tests had failures." -ForegroundColor Red
        $overallPass = $false
    } else {
        Write-Host "  [PASS] CLI tests." -ForegroundColor Green
    }
    Write-Host ""
}

# ── 3. Puppeteer E2E ────────────────────────────────────────────────────────
if ($Suite -in @("all","puppet")) {
    Write-Host "── [3/3] Puppeteer E2E browser tests ──" -ForegroundColor Yellow
    Write-Host "   DASHBOARD_URL=$env:DASHBOARD_URL"
    Write-Host "   API_BASE_URL=$env:API_BASE_URL"
    Write-Host ""

    $puppeteerDir = Join-Path $RepoRoot "11_testing\puppeteer"
    if (-not (Test-Path (Join-Path $puppeteerDir "node_modules"))) {
        Write-Host "  Installing npm deps..." -ForegroundColor Yellow
        & npm install --prefix $puppeteerDir
    }

    & npx --prefix $puppeteerDir jest --testPathPattern=tests/ --forceExit
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  [FAIL] Puppeteer tests had failures." -ForegroundColor Red
        $overallPass = $false
    } else {
        Write-Host "  [PASS] Puppeteer tests." -ForegroundColor Green
    }
    Write-Host ""
}

# ── Summary ─────────────────────────────────────────────────────────────────
Write-Host "======================================================" -ForegroundColor Cyan
if ($overallPass) {
    Write-Host "  ALL SUITES PASSED" -ForegroundColor Green
} else {
    Write-Host "  ONE OR MORE SUITES FAILED — review output above" -ForegroundColor Red
}
Write-Host "======================================================" -ForegroundColor Cyan
Write-Host ""

exit $(if ($overallPass) { 0 } else { 1 })
