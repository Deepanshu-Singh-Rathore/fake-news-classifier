<#
 .SYNOPSIS
    One-step environment setup and optional baseline training for fakenews project.

 .DESCRIPTION
    Creates (or reuses) .fenv virtual environment, upgrades pip, installs base and optional dev/ui extras,
    then optionally trains a baseline model if a dataset is available.

 .PARAMETER Train
    Switch: if provided, trains a baseline model on data\merged.csv (or data\sample.csv fallback).

 .PARAMETER Extras
    Comma-separated optional extras to install (ui,dev).

 .EXAMPLE
    powershell .\scripts\env_setup.ps1 -Train -Extras ui,dev
    Sets up env, installs UI + dev deps, trains model.
#>
param(
    [switch]$Train,
    [string]$Extras = ""
)

$ErrorActionPreference = 'Stop'

Write-Host "[fakenews] Environment setup starting..." -ForegroundColor Cyan

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path | Split-Path -Parent
Set-Location $projectRoot

$venvPath = Join-Path $projectRoot '.fenv'
if (!(Test-Path $venvPath)) {
    Write-Host "Creating virtual environment (.fenv)..." -ForegroundColor Yellow
    python -m venv .fenv
} else {
    Write-Host "Using existing .fenv virtual environment" -ForegroundColor Yellow
}

& .\.fenv\Scripts\Activate.ps1
Write-Host "Activated .fenv" -ForegroundColor Green

Write-Host "Upgrading pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip | Out-Null

Write-Host "Installing base dependencies via pyproject (editable mode)..." -ForegroundColor Yellow
if (Test-Path (Join-Path $projectRoot 'pyproject.toml')) {
    python -m pip install -e . | Out-Null
} else {
    python -m pip install -r requirements.txt | Out-Null
}

if ($Extras -ne "") {
    $extrasList = $Extras.Split(',') | ForEach-Object { $_.Trim() } | Where-Object { $_ -ne '' }
    foreach ($ex in $extrasList) {
        Write-Host "Installing extra group: $ex" -ForegroundColor Yellow
        python -m pip install -e ."[$ex]" | Out-Null
    }
}

Write-Host "Dependency check:" -ForegroundColor Cyan
python - <<'PY'
import importlib, sys
mods = ["pandas", "sklearn", "joblib"]
missing = []
for m in mods:
    try:
        importlib.import_module(m)
    except Exception as e:
        missing.append(m)
if missing:
    print("Missing:", ", ".join(missing))
    sys.exit(1)
print("All core dependencies present.")
PY
if ($LASTEXITCODE -ne 0) { throw "Missing core dependencies" }

if ($Train) {
    $dataMerged = Join-Path $projectRoot 'data\merged.csv'
    $dataSample = Join-Path $projectRoot 'data\sample.csv'
    $dataToUse = if (Test-Path $dataMerged) { $dataMerged } elseif (Test-Path $dataSample) { $dataSample } else { $null }
    if ($null -eq $dataToUse) {
        Write-Warning "No dataset found (data/merged.csv or data/sample.csv). Skipping training."
    } else {
        Write-Host "Training baseline model on $dataToUse ..." -ForegroundColor Cyan
        python train.py --data "$dataToUse" --model-out models\fake_news_model.joblib --ngrams 1,1 --max-features 2000 --class-weight balanced --cv 0 --min-df 1 --max-df 1.0
    }
}

Write-Host "Setup complete." -ForegroundColor Green