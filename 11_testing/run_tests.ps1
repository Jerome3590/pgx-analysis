# Run dashboard tests from repo root (Windows PowerShell).
# Usage: .\11_testing\run_tests.ps1 [pytest args...]
# Optional: $env:BASE_URL = "https://...execute-api.../prod" to run live API tests.

$ErrorActionPreference = "Stop"
# 11_testing is one level under repo root
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

$TestDir = "11_testing\tests"
if (-not (Test-Path $TestDir)) {
    Write-Error "Test directory not found: $RepoRoot\$TestDir"
    exit 1
}

Write-Host "Repo root: $RepoRoot"
Write-Host "Running: pytest $TestDir -v (dashboard + dashboard_visuals)"
Write-Host ""

$pytestArgs = $args
if ($pytestArgs) {
    & python -m pytest $TestDir -v @pytestArgs
} else {
    & python -m pytest $TestDir -v
}
exit $LASTEXITCODE
