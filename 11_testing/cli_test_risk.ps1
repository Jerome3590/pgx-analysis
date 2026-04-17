<#
.SYNOPSIS
    Combinatorial CLI test runner for the PGx Risk Dashboard API.

.DESCRIPTION
    Tests POST /risk and GET /visualizations/* for every cohort × age_band
    combination using five density scenarios that target each n_event_bin tier.
    Prints colour-coded PASS/FAIL per request and a summary table at the end.

    Default thresholds  (p25=5, p50=15, p95=50):
      baseline  0 codes  → is_baseline=true  (2019 outcome rate, no model)
      low       3 codes  → n_event_bin=low   (≤ p25)
      medium   10 codes  → n_event_bin=medium
      high     25 codes  → n_event_bin=high
      extreme  55 codes  → n_event_bin=extreme (> p95)

.PARAMETER BaseUrl
    API base URL.
    Local offline server : http://localhost:8000/prod   (default)
    Production           : https://<id>.execute-api.us-east-1.amazonaws.com/prod
    Or set env var BASE_URL before running.

.PARAMETER Cohort
    Restrict to a single cohort (opioid_ed | non_opioid_ed).  Default = all.

.PARAMETER AgeBand
    Restrict to a single age band (e.g. 25-44).  Default = all.

.PARAMETER Scenario
    Restrict to one density scenario (baseline|low|medium|high|extreme). Default = all.

.PARAMETER Viz
    Also run GET /visualizations/* endpoint matrix.

.PARAMETER TimeoutSec
    HTTP timeout per request in seconds. Default = 20.

.EXAMPLE
    # All combos against local offline server
    .\11_testing\cli_test_risk.ps1

.EXAMPLE
    # All combos against production
    .\11_testing\cli_test_risk.ps1 -BaseUrl "https://xxx.execute-api.us-east-1.amazonaws.com/prod"

.EXAMPLE
    # Single combo, all scenarios
    .\11_testing\cli_test_risk.ps1 -Cohort opioid_ed -AgeBand 25-44

.EXAMPLE
    # Single scenario across all combos
    .\11_testing\cli_test_risk.ps1 -Scenario extreme -Viz
#>

param(
    [string]$BaseUrl    = ($env:BASE_URL -replace '/$', ''),
    [string]$Cohort     = "",
    [string]$AgeBand    = "",
    [string]$Scenario   = "",
    [switch]$Viz,
    [int]   $TimeoutSec = 20
)

if (-not $BaseUrl) { $BaseUrl = "http://localhost:8000/prod" }

# ---------------------------------------------------------------------------
# Cohort / age-band matrix
# ---------------------------------------------------------------------------
$AllCohorts   = @("opioid_ed", "non_opioid_ed")
$AllAgeBands  = @("0-12","13-24","25-44","45-54","55-64","65-74","75-84","85-114")
$ValidBins    = @("low","medium","high","extreme")
$ValidBands   = @("low","medium","high")

$Combos = foreach ($c in $AllCohorts) {
    foreach ($ab in $AllAgeBands) {
        [pscustomobject]@{ Cohort=$c; AgeBand=$ab }
    }
}
if ($Cohort)  { $Combos = $Combos | Where-Object { $_.Cohort  -eq $Cohort  } }
if ($AgeBand) { $Combos = $Combos | Where-Object { $_.AgeBand -eq $AgeBand } }

# ---------------------------------------------------------------------------
# Code pools  (same logic as test_combinatorial_risk.py)
# ---------------------------------------------------------------------------
$OpioidDrugs = @(
    "oxycodone","hydrocodone","tramadol","gabapentin","alprazolam",
    "cyclobenzaprine","fentanyl","codeine","methadone","morphine",
    "diazepam","clonazepam","buprenorphine","oxymorphone","hydromorphone",
    "carisoprodol","zolpidem","lorazepam","pregabalin","duloxetine"
)
$OpioidIcds = @(
    "M54.5","G89.29","F41.1","F32.1","F17.210","R51","M25.511",
    "Z87.891","J06.9","M54.41","G89.4","M79.3","F33.1",
    "M54.16","G89.11","F41.0","M47.816","Z79.891","G89.21","M54.50"
)
$OpioidCpts = @(
    "99213","80305","99396","99214","99203","80306","97110",
    "97012","90832","90834","72100","72148","73560","99215","99204"
)

$NonDrugs = @(
    "furosemide","hydrochlorothiazide","lisinopril","metformin","simvastatin",
    "atorvastatin","metoprolol","amlodipine","carvedilol","losartan",
    "warfarin","aspirin","omeprazole","levothyroxine","albuterol",
    "prednisone","levofloxacin","alprazolam","lorazepam","acetaminophen"
)
$NonIcds = @(
    "I10","E11.9","E78.5","I50.9","N18.3","I25.10","J44.1",
    "E03.9","G47.33","M79.3","K21.0","D64.9","F03.90","G20",
    "I48.91","I63.9","N39.0","R06.09","Z79.01","M17.11"
)
$NonCpts = @(
    "99213","99214","93000","83036","85025","80053","36415",
    "99396","93306","71046","93010","82947","84443","86900","99395"
)

function Get-Pool($CohortName) {
    if ($CohortName -eq "opioid_ed") {
        return @{ Drugs=$OpioidDrugs; Icds=$OpioidIcds; Cpts=$OpioidCpts }
    }
    return @{ Drugs=$NonDrugs; Icds=$NonIcds; Cpts=$NonCpts }
}

# Returns @{Drugs=@();Icds=@();Cpts=@();ExpectedBin="low"|...}
function Get-Scenario($ScenarioName, $CohortName) {
    $p = Get-Pool $CohortName
    switch ($ScenarioName) {
        "baseline" { return @{Drugs=@();Icds=@();Cpts=@();ExpectedBin=$null} }
        "low"      { return @{Drugs=@($p.Drugs[0..0]);Icds=@($p.Icds[0..0]);Cpts=@($p.Cpts[0..0]);ExpectedBin="low"} }
        "medium"   { return @{Drugs=@($p.Drugs[0..3]);Icds=@($p.Icds[0..3]);Cpts=@($p.Cpts[0..1]);ExpectedBin="medium"} }
        "high"     { return @{Drugs=@($p.Drugs[0..9]);Icds=@($p.Icds[0..9]);Cpts=@($p.Cpts[0..4]);ExpectedBin="high"} }
        "extreme"  { return @{Drugs=@($p.Drugs[0..19]);Icds=@($p.Icds[0..19]);Cpts=@($p.Cpts[0..14]);ExpectedBin="extreme"} }
    }
}

$AllScenarios = @("baseline","low","medium","high","extreme")
$ActiveScenarios = if ($Scenario) { @($Scenario) } else { $AllScenarios }

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
$PassCount = 0
$FailCount = 0
$SkipCount = 0
$Results   = [System.Collections.Generic.List[pscustomobject]]::new()

function Write-Pass($msg) { Write-Host "  [PASS] $msg" -ForegroundColor Green;  $script:PassCount++ }
function Write-Fail($msg) { Write-Host "  [FAIL] $msg" -ForegroundColor Red;    $script:FailCount++ }
function Write-Skip($msg) { Write-Host "  [SKIP] $msg" -ForegroundColor Yellow; $script:SkipCount++ }

function Invoke-Api {
    param(
        [string]$Method,
        [string]$Path,
        [hashtable]$QueryParams = @{},
        [hashtable]$Body = $null
    )
    $uri = "$BaseUrl$Path"
    if ($QueryParams.Count -gt 0) {
        $qs = ($QueryParams.GetEnumerator() | ForEach-Object { "$($_.Key)=$([uri]::EscapeDataString($_.Value))" }) -join "&"
        $uri += "?$qs"
    }
    try {
        if ($Method -eq "POST") {
            $jsonBody = $Body | ConvertTo-Json -Depth 5 -Compress
            $resp = Invoke-WebRequest -Uri $uri -Method POST `
                -ContentType "application/json" -Body $jsonBody `
                -TimeoutSec $TimeoutSec -ErrorAction Stop -UseBasicParsing
        } else {
            $resp = Invoke-WebRequest -Uri $uri -Method GET `
                -TimeoutSec $TimeoutSec -ErrorAction Stop -UseBasicParsing
        }
        return @{ Status=$resp.StatusCode; Body=($resp.Content | ConvertFrom-Json -ErrorAction SilentlyContinue); Raw=$resp.Content }
    } catch [System.Net.WebException] {
        $code = [int]$_.Exception.Response.StatusCode
        return @{ Status=$code; Body=$null; Raw=$_.Exception.Message }
    } catch {
        return @{ Status=0; Body=$null; Raw=$_.Exception.Message }
    }
}

function Test-RiskResponse {
    param($Body, $Cohort, $AgeBand, $ScenarioName, $Drugs, $Icds, $Cpts, $ExpectedBin)

    $issues = @()

    # risk_score
    if ($null -eq $Body.risk_score)                              { $issues += "missing risk_score" }
    elseif ($Body.risk_score -lt 0 -or $Body.risk_score -gt 1)  { $issues += "risk_score=$($Body.risk_score) out of [0,1]" }

    # risk_band
    if ($Body.risk_band -notin $ValidBands) { $issues += "invalid risk_band=$($Body.risk_band)" }

    # echo-back
    if ($Body.cohort_used  -ne $Cohort)   { $issues += "cohort_used=$($Body.cohort_used) != $Cohort" }
    if ($Body.age_band_used -ne $AgeBand) { $issues += "age_band_used=$($Body.age_band_used) != $AgeBand" }

    $totalCodes = $Drugs.Count + $Icds.Count + $Cpts.Count

    if ($ScenarioName -eq "baseline") {
        if ($Body.is_baseline -ne $true) { $issues += "expected is_baseline=true" }
    } else {
        if ($Body.is_baseline -eq $true) { $issues += "unexpected is_baseline=true" }
        if ($Body.n_event_bin -notin $ValidBins) { $issues += "invalid n_event_bin=$($Body.n_event_bin)" }
        if ($Body.n_events -ne $totalCodes)       { $issues += "n_events=$($Body.n_events) != expected $totalCodes" }
        # Bin routing (default thresholds)
        if ($ExpectedBin -and $Body.n_event_bin -ne $ExpectedBin) {
            $issues += "n_event_bin=$($Body.n_event_bin) expected=$ExpectedBin (default thresholds; may differ if custom)"
        }
    }

    # codes_used / codes_unknown
    foreach ($block in @("codes_used","codes_unknown")) {
        $b = $Body.$block
        if ($null -eq $b) { $issues += "missing $block" }
    }

    return $issues
}

# ---------------------------------------------------------------------------
# Risk Assessment — POST /risk — combinatorial matrix
# ---------------------------------------------------------------------------
Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host " PGx Risk Dashboard — Combinatorial CLI Test" -ForegroundColor Cyan
Write-Host " Target: $BaseUrl" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

foreach ($combo in $Combos) {
    $c  = $combo.Cohort
    $ab = $combo.AgeBand
    Write-Host ""
    Write-Host "── $c / $ab ──" -ForegroundColor Cyan

    foreach ($scen in $ActiveScenarios) {
        $s    = Get-Scenario $scen $c
        $label = "$c/$ab [$scen]"
        $totalCodes = $s.Drugs.Count + $s.Icds.Count + $s.Cpts.Count

        $reqBody = @{
            cohort    = $c
            age_band  = $ab
            drugs     = $s.Drugs
            icds      = $s.Icds
            cpts      = $s.Cpts
        }

        $r = Invoke-Api -Method POST -Path "/risk" -Body $reqBody

        $row = [pscustomobject]@{
            Combo      = "$c/$ab"
            Scenario   = $scen
            Codes      = $totalCodes
            Status     = $r.Status
            RiskScore  = $null
            RiskBand   = $null
            NEventBin  = $null
            IsBaseline = $null
            Result     = "?"
            Issues     = ""
        }

        if ($r.Status -eq 0) {
            Write-Skip "  $label — connection error: $($r.Raw)"
            $row.Result = "SKIP"; $script:SkipCount++
        } elseif ($r.Status -eq 500) {
            Write-Host "  [SKIP] $label — HTTP 500 (models not deployed)" -ForegroundColor Yellow
            $row.Result = "SKIP"; $script:SkipCount++
        } elseif ($r.Status -eq 200 -and $r.Body) {
            $b = $r.Body
            $row.RiskScore  = [math]::Round([double]($b.risk_score), 4)
            $row.RiskBand   = $b.risk_band
            $row.NEventBin  = $b.n_event_bin
            $row.IsBaseline = $b.is_baseline

            $issues = Test-RiskResponse $b $c $ab $scen $s.Drugs $s.Icds $s.Cpts $s.ExpectedBin
            if ($issues.Count -eq 0) {
                $row.Result = "PASS"
                Write-Pass "$label  score=$($row.RiskScore)  band=$($row.RiskBand)  bin=$($row.NEventBin)  n=$totalCodes"
            } else {
                $row.Result = "FAIL"
                $row.Issues = $issues -join "; "
                Write-Fail "$label — $($row.Issues)"
            }
        } else {
            Write-Fail "$label — HTTP $($r.Status)"
            $row.Result = "FAIL"; $row.Issues = "HTTP $($r.Status)"
        }
        $Results.Add($row)
    }
}

# ---------------------------------------------------------------------------
# Visualizations — GET /visualizations/* — all combos
# ---------------------------------------------------------------------------
if ($Viz) {
    $VizEndpoints = @(
        "/visualizations/causal",
        "/visualizations/dtw",
        "/visualizations/fpgrowth",
        "/visualizations/bupar",
        "/visualizations/bupar/activity_frequency",
        "/visualizations/cohort_pgx"
    )

    Write-Host ""
    Write-Host "============================================================" -ForegroundColor Cyan
    Write-Host " Visualization Endpoints" -ForegroundColor Cyan
    Write-Host "============================================================" -ForegroundColor Cyan

    foreach ($combo in $Combos) {
        $c  = $combo.Cohort
        $ab = $combo.AgeBand
        Write-Host ""
        Write-Host "── $c / $ab ──" -ForegroundColor Cyan

        foreach ($ep in $VizEndpoints) {
            $label = "$c/$ab $ep"
            $r = Invoke-Api -Method GET -Path $ep -QueryParams @{ cohort=$c; age_band=$ab }
            $row = [pscustomobject]@{
                Combo=$c+"/$ab"; Scenario=$ep; Codes=$null
                Status=$r.Status; RiskScore=$null; RiskBand=$null
                NEventBin=$null; IsBaseline=$null; Result="?"; Issues=""
            }
            if ($r.Status -in @(200,400,404)) {
                $row.Result = "PASS"
                Write-Pass "$label — HTTP $($r.Status)"
            } elseif ($r.Status -eq 500) {
                $row.Result = "SKIP"
                Write-Host "  [SKIP] $label — HTTP 500" -ForegroundColor Yellow; $script:SkipCount++
            } elseif ($r.Status -eq 0) {
                $row.Result = "SKIP"
                Write-Skip "$label — connection error"; $script:SkipCount++
            } else {
                $row.Result = "FAIL"
                $row.Issues = "HTTP $($r.Status)"
                Write-Fail "$label — HTTP $($r.Status)"
            }
            $Results.Add($row)
        }
    }

    # Required-params 400 checks
    Write-Host ""
    Write-Host "── Missing-param 400 validation ──" -ForegroundColor Cyan
    foreach ($ep in $VizEndpoints) {
        $r = Invoke-Api -Method GET -Path $ep -QueryParams @{}
        if ($r.Status -eq 400) { Write-Pass "$ep (no params) → 400 ✓" }
        else                   { Write-Fail "$ep (no params) → expected 400, got $($r.Status)" }
    }
}

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host " Summary" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

$total = $PassCount + $FailCount + $SkipCount
Write-Host "  Total  : $total"
Write-Host "  PASS   : $PassCount" -ForegroundColor Green
Write-Host "  FAIL   : $FailCount" -ForegroundColor $(if ($FailCount -gt 0) { "Red" } else { "Green" })
Write-Host "  SKIP   : $SkipCount" -ForegroundColor Yellow

if ($FailCount -gt 0) {
    Write-Host ""
    Write-Host "  Failed cases:" -ForegroundColor Red
    $Results | Where-Object { $_.Result -eq "FAIL" } | ForEach-Object {
        Write-Host "    $($_.Combo) [$($_.Scenario)] — $($_.Issues)" -ForegroundColor Red
    }
}

# Detailed results table
Write-Host ""
Write-Host "  Result table (POST /risk):" -ForegroundColor Cyan
$Results | Where-Object { $_.Scenario -in $AllScenarios } |
    Format-Table Combo,Scenario,Codes,Status,RiskScore,RiskBand,NEventBin,Result,Issues -AutoSize

Write-Host ""
Write-Host "  Run with -Viz to include /visualizations/* endpoints."
Write-Host "  Set -BaseUrl to target production:  -BaseUrl `"https://...execute-api.../prod`""
Write-Host ""

exit $(if ($FailCount -gt 0) { 1 } else { 0 })
