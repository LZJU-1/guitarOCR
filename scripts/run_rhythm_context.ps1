param(
    [int]$Epochs = 30,
    [int]$BatchSize = 64,
    [int]$Workers = 4,
    [switch]$SkipDatasetBuild
)

$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'common.ps1')

$WorkspaceRoot = Split-Path -Parent $PSScriptRoot
$env:PYTHONPATH = $WorkspaceRoot
$DatabaseRoot = Get-GuitarOcrDatabaseRoot
$Python = Get-GuitarOcrPython
$DatasetScript = Join-Path $PSScriptRoot 'build_score_rhythm_dataset.ps1'
$Trainer = Join-Path $WorkspaceRoot 'guitarocr\training\train_rhythm_context.py'

foreach ($required in @($Python, $DatasetScript, $Trainer, $DatabaseRoot)) {
    if (-not (Test-Path -LiteralPath $required)) {
        throw "Required path is missing: $required"
    }
}

if (-not $SkipDatasetBuild) {
    Write-Host '[1/2] Building and validating score_tab rhythm data...'
    & $DatasetScript
    if ($LASTEXITCODE -ne 0) { throw "Dataset build failed with exit code $LASTEXITCODE" }
} else {
    Write-Host '[1/2] Reusing the existing score_tab rhythm dataset.'
}

Write-Host '[2/2] Training the compact rhythm context CNN...'
& $Python -m guitarocr.training.train_rhythm_context `
    --database $DatabaseRoot --epochs $Epochs --batch-size $BatchSize --workers $Workers
if ($LASTEXITCODE -ne 0) { throw "Training failed with exit code $LASTEXITCODE" }

Write-Host 'Complete. Metrics are in database\rhythm_events\models\metrics.json.'
