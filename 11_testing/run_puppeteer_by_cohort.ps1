<#
.SYNOPSIS
    Run Puppeteer E2E tests one cohort + one age band at a time.
    Writes per-cohort markdown to 11_testing/results/results_{cohort}.md.

.PARAMETER Cohort
    Run only this cohort: opioid_ed | non_opioid_ed | all  (default: all)

.PARAMETER AgeBand
    Run only this age band, e.g. "13-24"  (default: all)
#>
param(
    [ValidateSet("all","opioid_ed","non_opioid_ed")]
    [string]$Cohort  = "all",
    [string]$AgeBand = ""
)

$ErrorActionPreference = "Continue"
$env:DASHBOARD_URL = "https://jerome-dixon.io/vcu/pgx-risk-calculator/index.html"
$env:API_BASE_URL  = "https://cmv0qislq3.execute-api.us-east-1.amazonaws.com/prod"

$puppeteerDir = Join-Path $PSScriptRoot "puppeteer"
$resultsDir   = Join-Path $PSScriptRoot "results"
New-Item -ItemType Directory -Force -Path $resultsDir | Out-Null

$allCohorts   = @("opioid_ed", "non_opioid_ed")
$allAgeBands  = @("0-12","13-24","25-44","45-54","55-64","65-74","75-84","85-114")

$cohortList   = if ($Cohort -eq "all") { $allCohorts } else { @($Cohort) }
$ageBandList  = if ($AgeBand)          { @($AgeBand) } else { $allAgeBands }

# ── helper: run one jest pattern, return output lines + exit code ─────────
function Invoke-Jest {
    param([string]$TestPattern, [string]$NameFilter)
    Push-Location $puppeteerDir
    if ($NameFilter) {
        $out = & npx jest --testPathPattern=$TestPattern --testNamePattern=$NameFilter --forceExit --verbose 2>&1
    } else {
        $out = & npx jest --testPathPattern=$TestPattern --forceExit --verbose 2>&1
    }
    $code = $LASTEXITCODE
    Pop-Location
    return @{ Lines = $out; Exit = $code }
}

# ── helper: append failures from jest output to markdown file ─────────────
function Write-Failures {
    param([string[]]$Lines, [string]$MdFile)
    $failures = @{}
    $currentTest = $null
    $errorLines  = @()

    foreach ($line in $Lines) {
        $plain = $line -replace '\x1B\[[0-9;]*m', ''   # strip ANSI
        if ($plain -match '^\s+●\s+(.+)$') {
            if ($currentTest -and $errorLines.Count) {
                $failures[$currentTest] = $errorLines[0..([Math]::Min(5,$errorLines.Count-1))]
            }
            $currentTest = $Matches[1].Trim()
            $errorLines  = @()
        } elseif ($currentTest -and $plain -match '\S') {
            $errorLines += $plain.Trim()
        }
    }
    if ($currentTest -and $errorLines.Count) {
        $failures[$currentTest] = $errorLines[0..([Math]::Min(5,$errorLines.Count-1))]
    }

    if ($failures.Count -eq 0) { return }

    "#### Failures ($($failures.Count))" | Add-Content $MdFile
    "" | Add-Content $MdFile
    foreach ($test in $failures.Keys) {
        "- **$test**" | Add-Content $MdFile
        $firstError = ($failures[$test] | Where-Object { $_ -match "Error|expect|TypeError|net::" } | Select-Object -First 1)
        if ($firstError) { "  ``$firstError``" | Add-Content $MdFile }
    }
    "" | Add-Content $MdFile
}

# ─────────────────────────────────────────────────────────────────────────────

foreach ($cohort in $cohortList) {
    $cohortLabel = if ($cohort -eq "opioid_ed") { "Opioid ED" } else { "Polypharmacy (non_opioid_ed)" }
    $mdFile      = Join-Path $resultsDir "results_${cohort}.md"
    $timestamp   = Get-Date -Format "yyyy-MM-dd HH:mm:ss"

    "# Test Results: $cohortLabel"       | Set-Content  $mdFile
    ""                                    | Add-Content $mdFile
    "**Run:** $timestamp"                 | Add-Content $mdFile
    "**Dashboard:** $env:DASHBOARD_URL"   | Add-Content $mdFile
    "**API:** $env:API_BASE_URL"          | Add-Content $mdFile
    ""                                    | Add-Content $mdFile

    # ── PGx Card (no age band split, run once per cohort report) ──────────
    if ($cohort -eq "opioid_ed") {
        Write-Host "`n==> pgx-card" -ForegroundColor Cyan
        "## PGx Card" | Add-Content $mdFile
        "" | Add-Content $mdFile
        $r = Invoke-Jest "tests/pgx-card" ""
        $summary = ($r.Lines | Where-Object { $_ -match "^Tests:" } | Select-Object -Last 1) -replace '\x1B\[[0-9;]*m',''
        $icon = if ($r.Exit -eq 0) { "PASS" } else { "FAIL" }
        "**$icon** $summary" | Add-Content $mdFile
        "" | Add-Content $mdFile
        if ($r.Exit -ne 0) { Write-Failures $r.Lines $mdFile }
        "---" | Add-Content $mdFile
        "" | Add-Content $mdFile
        Write-Host "  $icon  $summary" -ForegroundColor $(if ($r.Exit -eq 0) {"Green"} else {"Red"})
    }

    # ── Combinatorial + Viz per age band ──────────────────────────────────
    foreach ($ab in $ageBandList) {
        Write-Host "`n==> $cohort / $ab" -ForegroundColor Cyan
        "## Age Band: $ab" | Add-Content $mdFile
        "" | Add-Content $mdFile

        foreach ($suite in @("combinatorial","viz")) {
            $pattern    = "tests/$suite"
            $nameFilter = "cohort: $cohort.*age_band: $ab"

            Write-Host "    $suite ..." -NoNewline
            $r = Invoke-Jest $pattern $nameFilter
            $summary = ($r.Lines | Where-Object { $_ -match "^Tests:" } | Select-Object -Last 1) -replace '\x1B\[[0-9;]*m',''
            $time    = ($r.Lines | Where-Object { $_ -match "^Time:" }  | Select-Object -Last 1) -replace '\x1B\[[0-9;]*m',''
            $icon    = if ($r.Exit -eq 0) { "PASS" } else { "FAIL" }

            "### $suite" | Add-Content $mdFile
            "**$icon** $summary  $time" | Add-Content $mdFile
            "" | Add-Content $mdFile
            if ($r.Exit -ne 0) {
                # Always dump raw log alongside markdown for full error context
                $logFile = Join-Path $resultsDir "${cohort}_${ab}_${suite}.log"
                ($r.Lines -replace '\x1B\[[0-9;]*[A-Za-z]','') | Set-Content $logFile -Encoding UTF8
                "  [raw log: results/${cohort}_${ab}_${suite}.log]" | Add-Content $mdFile
                "" | Add-Content $mdFile
                Write-Failures $r.Lines $mdFile
            }

            Write-Host " $icon  $summary" -ForegroundColor $(if ($r.Exit -eq 0) {"Green"} else {"Red"})
        }

        "---" | Add-Content $mdFile
        "" | Add-Content $mdFile
    }

    Write-Host "`nResults: $mdFile" -ForegroundColor Yellow
}

# ── Full UI simulation suite (real user workflow, run once after cohort suites) ──
Write-Host "`n==> user-simulation (real user workflow)" -ForegroundColor Cyan
$simFile      = Join-Path $resultsDir "results_user_simulation.md"
$simTimestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"

"# Test Results: Full UI Simulation (Real User Workflow)" | Set-Content  $simFile
""                                                         | Add-Content $simFile
"**Run:** $simTimestamp"                                   | Add-Content $simFile
"**Dashboard:** $env:DASHBOARD_URL"                        | Add-Content $simFile
"**API:** $env:API_BASE_URL"                               | Add-Content $simFile
""                                                         | Add-Content $simFile

$simResult = Invoke-Jest "tests/user-simulation" ""
$simSummary = ($simResult.Lines | Where-Object { $_ -match "^Tests:" } | Select-Object -Last 1) -replace '\x1B\[[0-9;]*m',''
$simTime    = ($simResult.Lines | Where-Object { $_ -match "^Time:" }  | Select-Object -Last 1) -replace '\x1B\[[0-9;]*m',''
$simIcon    = if ($simResult.Exit -eq 0) { "PASS" } else { "FAIL" }

"**$simIcon** $simSummary  $simTime" | Add-Content $simFile
""                                   | Add-Content $simFile

if ($simResult.Exit -ne 0) {
    $simLogFile = Join-Path $resultsDir "user_simulation.log"
    ($simResult.Lines -replace '\x1B\[[0-9;]*[A-Za-z]','') | Set-Content $simLogFile -Encoding UTF8
    "  [raw log: results/user_simulation.log]" | Add-Content $simFile
    "" | Add-Content $simFile
    Write-Failures $simResult.Lines $simFile
}

Write-Host "  $simIcon  $simSummary" -ForegroundColor $(if ($simResult.Exit -eq 0) {"Green"} else {"Red"})
Write-Host "`nResults: $simFile" -ForegroundColor Yellow

Write-Host "`nDone. Results in: $resultsDir" -ForegroundColor Cyan
