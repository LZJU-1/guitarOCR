param(
    [int]$Epochs = 24,
    [int]$BatchSize = 64,
    [int]$Workers = 4,
    [switch]$SkipTraining
)

$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'common.ps1')
$WorkspaceRoot = Split-Path -Parent $PSScriptRoot
$env:PYTHONPATH = $WorkspaceRoot
$DatabaseRoot = Get-GuitarOcrDatabaseRoot
$Python = Get-GuitarOcrPython

Write-Host '[1/4] Building source-disjoint real-PDF tie-event labels...'
& $Python -m guitarocr.data.build_tie_event_dataset --database $DatabaseRoot
if ($LASTEXITCODE -ne 0) { throw "Tie dataset build failed with exit code $LASTEXITCODE" }

if (-not $SkipTraining) {
    Write-Host '[2/4] Training the compact tie relationship CNN...'
    & $Python -m guitarocr.training.train_tie_context `
        --database $DatabaseRoot --epochs $Epochs --batch-size $BatchSize --workers $Workers --fresh
    if ($LASTEXITCODE -ne 0) { throw "Tie model training failed with exit code $LASTEXITCODE" }
} else {
    Write-Host '[2/4] Reusing the existing tie relationship model.'
}

Write-Host '[3/4] Evaluating page-level validation candidates...'
& $Python -m guitarocr.evaluation.evaluate_tie_event_pages --database $DatabaseRoot --split validation
if ($LASTEXITCODE -ne 0) { throw "Tie validation evaluation failed with exit code $LASTEXITCODE" }

Write-Host '[4/4] Evaluating the independent page-level test split...'
& $Python -m guitarocr.evaluation.evaluate_tie_event_pages --database $DatabaseRoot --split test
if ($LASTEXITCODE -ne 0) { throw "Tie test evaluation failed with exit code $LASTEXITCODE" }

Write-Host 'Complete. Tie models and metrics are in database\tie_events\models.'
