<#
.SYNOPSIS
    PowerShell shim mirroring the Makefile targets, for Windows boxes without
    GNU make on PATH.

.EXAMPLE
    .\make.ps1 check
    .\make.ps1 deep
    .\make.ps1 all
#>
[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [string]$Target = 'help',

    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Rest
)

$ErrorActionPreference = 'Stop'
$Root = $PSScriptRoot
$Py = Join-Path $Root '.venv\Scripts\python.exe'
$Cfg = Join-Path $Root 'config.yaml'

if (-not (Test-Path $Py)) {
    Write-Error "Virtualenv not found at $Py. Run: .\make.ps1 env"
}

function Invoke-Step {
    param([string]$Script, [string[]]$Args = @())
    $path = Join-Path $Root "scripts\$Script"
    Write-Host "==> python scripts/$Script $($Args -join ' ')" -ForegroundColor Cyan
    & $Py $path @Args
    # -1073740791 (0xC0000409) = STATUS_STACK_BUFFER_OVERRUN: benign crash
    # during Python/PyTorch interpreter shutdown on Windows; work is done.
    if ($LASTEXITCODE -eq -1073740791) {
        Write-Host "  (ignored benign Windows exit code 0xC0000409 during interpreter shutdown)" -ForegroundColor Yellow
    }
    elseif ($LASTEXITCODE -ne 0) { throw "scripts/$Script failed with exit code $LASTEXITCODE" }
}

$common = @('--config', $Cfg)

# Cross-city generalisation check. Same pipeline, same grid, same seeds; only
# the record, one meteorological driver and the season definition differ.
$CfgB = Join-Path $Root 'config_beijing.yaml'
$commonB = @('--config', $CfgB)

switch ($Target) {
    'help' {
        Write-Host @'
Targets:
  env         Create .venv and install pinned requirements
  check       Phase 0: environment / CUDA / disk check
  discover    Phase 1: list Dhaka OpenAQ monitors (STOPS for review)
  data        Phase 1: fetch OpenAQ + NASA POWER + UCI Beijing
  audit       Phase 1: build reports/DATA_AUDIT.md  (HARD GATE)
  features    Phase 2: build features + chronological splits
  test        Phase 2: leakage + unit tests
  baselines   Phase 3: Tier 1 baselines + Tier 2 classical ML
  deep        Phase 4: Tier 3 sequence models, all seeds (resumable)
  classify    Phase 5: AQI-category classifier
  green       Phase 5: complexity, latency, energy, CO2e
  eval        Phase 5: stratified eval, skill scores, Diebold-Mariano
  figures     Phase 6: all figures (png + pdf, 300 dpi)
  report      Phase 6: RESULTS.md + abstract_facts.json
  all         Everything above, in order
  beijing     Cross-city: whole pipeline again on the Beijing record
  cross-city  Rank-transfer comparison (needs 'all' and 'beijing')
  tune        Choose the sequence training recipe on val loss, before 'deep'
  ablation    Gap-injection experiment + figure (needs 'beijing')
  everything  all + beijing + cross-city + ablation + report
  lint        ruff check + format check
  clean       Remove caches and checkpoints (keeps raw data)
'@
    }
    'env' {
        python -m venv (Join-Path $Root '.venv')
        & $Py -m pip install --upgrade pip setuptools wheel
        & $Py -m pip install torch --index-url https://download.pytorch.org/whl/cu128
        & $Py -m pip install -r (Join-Path $Root 'requirements.txt')
    }
    'check'     { Invoke-Step '00_check_env.py' }
    'discover'  { Invoke-Step '01_discover_openaq.py' $common }
    'data'      { Invoke-Step '02_fetch_data.py'      $common }
    'audit'     { Invoke-Step '03_data_audit.py'      $common }
    'features'  { Invoke-Step '04_build_features.py'  $common }
    'test'      { & $Py -m pytest (Join-Path $Root 'tests') -v }
    'baselines' { Invoke-Step '05_run_baselines.py'   $common }
    'deep'      { Invoke-Step '06_train_sequence.py'  ($common + @('--resume', 'auto') + $Rest) }
    'classify'  { Invoke-Step '07_train_classifier.py' ($common + @('--resume', 'auto') + $Rest) }
    'green'     { Invoke-Step '08_green_measure.py'   $common }
    'eval'      { Invoke-Step '09_evaluate.py'        $common }
    'figures'   { Invoke-Step '10_make_figures.py'    $common }
    'report'    { Invoke-Step '11_make_report.py'     $common }
    'beijing' {
        # Reuses the Beijing frame already downloaded by the 'data' target; every
        # stage after 12_prepare_beijing is the identical script, other config.
        Invoke-Step '12_prepare_beijing.py'  $commonB
        Invoke-Step '03_data_audit.py'       $commonB
        Invoke-Step '04_build_features.py'   $commonB
        Invoke-Step '05_run_baselines.py'    $commonB
        Invoke-Step '06_train_sequence.py'   ($commonB + @('--resume', 'auto') + $Rest)
        Invoke-Step '07_train_classifier.py' ($commonB + @('--resume', 'auto') + $Rest)
        Invoke-Step '08_green_measure.py'    $commonB
        Invoke-Step '09_evaluate.py'         $commonB
        Invoke-Step '10_make_figures.py'     $commonB
        Invoke-Step '11_make_report.py'      $commonB
    }
    'tune'      { Invoke-Step '06_train_sequence.py' ($common + @('--tune', '--progress', 'plain') + $Rest) }
    'ablation' {
        # Degrades the COMPARISON city (the near-complete record) using the
        # PRIMARY city's gap-length distribution, so both configs are passed.
        Invoke-Step '16_gap_injection.py' ($commonB + @('--profile-config', $Cfg, '--progress', 'plain') + $Rest)
        Invoke-Step '17_ablation_analysis.py' $commonB
    }
    'cross-city' {
        Invoke-Step '13_cross_city.py' @('--config', $Cfg, '--config-b', $CfgB)
    }
    'everything' {
        # 'report' repeats last on purpose: 13_cross_city writes the comparison
        # into results.json, and only a later report carries section 7.
        foreach ($t in @('all', 'beijing', 'cross-city', 'ablation', 'report')) {
            & $PSCommandPath $t
            if ($LASTEXITCODE -ne 0) { throw "target '$t' failed" }
        }
    }
    'lint' {
        & $Py -m ruff check (Join-Path $Root 'src') (Join-Path $Root 'scripts') (Join-Path $Root 'tests')
        & $Py -m ruff format --check (Join-Path $Root 'src') (Join-Path $Root 'scripts') (Join-Path $Root 'tests')
    }
    'fmt' {
        & $Py -m ruff format (Join-Path $Root 'src') (Join-Path $Root 'scripts') (Join-Path $Root 'tests')
        & $Py -m ruff check --fix (Join-Path $Root 'src') (Join-Path $Root 'scripts') (Join-Path $Root 'tests')
    }
    'all' {
        foreach ($t in @('check','data','audit','features','test','baselines','deep','classify','green','eval','figures','report')) {
            & $PSCommandPath $t
            if ($LASTEXITCODE -ne 0) { throw "target '$t' failed" }
        }
    }
    'clean' {
        Get-ChildItem $Root -Recurse -Directory -Filter '__pycache__' -ErrorAction SilentlyContinue |
            Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
        foreach ($p in @('data\interim\cache','data\interim\checkpoints','.ruff_cache','.pytest_cache')) {
            Remove-Item (Join-Path $Root $p) -Recurse -Force -ErrorAction SilentlyContinue
        }
        Write-Host 'Cleaned caches and checkpoints (raw data untouched).'
    }
    default { Write-Error "Unknown target '$Target'. Run: .\make.ps1 help" }
}
