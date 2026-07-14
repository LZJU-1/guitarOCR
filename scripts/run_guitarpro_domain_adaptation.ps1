[CmdletBinding()]
param(
    [string]$GuitarProDataset,
    [int]$Epochs = 12,
    [int]$BatchSize = 128,
    [int]$Workers = 4,
    [double]$MinimumTechniquePrecision = 0.25,
    [int]$MinimumTechniqueSupport = 10,
    [switch]$SkipDatasetBuild,
    [switch]$SkipFretTraining,
    [switch]$SkipRhythmTraining,
    [switch]$SkipTechniqueTraining,
    [switch]$Promote
)

$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'common.ps1')

$python = Get-GuitarOcrPython
$database = Get-GuitarOcrDatabaseRoot
$workspace = Split-Path -Parent $PSScriptRoot
if (-not $GuitarProDataset) {
    $GuitarProDataset = Join-Path $database 'guitarpro8_multimode_v1'
}
if (-not (Test-Path -LiteralPath $GuitarProDataset)) {
    throw "Guitar Pro multimode dataset is missing: $GuitarProDataset"
}

function Invoke-GuitarOcrPython {
    param([string[]]$Arguments)
    & $python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed with exit code ${LASTEXITCODE}: $($Arguments -join ' ')"
    }
}

if (-not $SkipDatasetBuild) {
    Write-Host '[1/5] Building mixed Guitar Pro 8/TuxGuitar fret-token crops'
    Invoke-GuitarOcrPython @(
        '-m', 'guitarocr.data.build_fret_token_dataset',
        '--database', $database,
        '--gp8-root', $GuitarProDataset,
        '--modes', 'tab,both'
    )
    Write-Host '[2/5] Building source-disjoint Guitar Pro 8 rhythm/technique crops'
    Invoke-GuitarOcrPython @(
        '-m', 'guitarocr.data.build_guitarpro_rhythm_dataset',
        '--database', $database,
        '--gp8-root', $GuitarProDataset,
        '--modes', 'tab,both'
    )
}

if (-not $SkipFretTraining) {
    Write-Host '[3/5] Training the event-conditioned fret/X classifier'
    Invoke-GuitarOcrPython @(
        '-m', 'guitarocr.training.train_fret_token',
        '--database', $database,
        '--epochs', [string]$Epochs,
        '--batch-size', [string]$BatchSize,
        '--workers', [string]$Workers
    )
}

if (-not $SkipRhythmTraining) {
    Write-Host '[4/5] Fine-tuning Guitar Pro 8 rhythm models'
    Invoke-GuitarOcrPython @(
        '-m', 'guitarocr.training.train_rhythm_context',
        '--database', $database,
        '--task-root', 'gp8_tab_rhythm_events',
        '--init-checkpoint', (Join-Path $workspace 'weights\tab_rhythm_context_cnn.pt'),
        '--epochs', [string]$Epochs,
        '--batch-size', [string]$BatchSize,
        '--workers', [string]$Workers,
        '--learning-rate', '0.0005'
    )
    Invoke-GuitarOcrPython @(
        '-m', 'guitarocr.training.train_rhythm_context',
        '--database', $database,
        '--task-root', 'gp8_score_rhythm_events',
        '--init-checkpoint', (Join-Path $workspace 'weights\rhythm_context_cnn.pt'),
        '--epochs', [string]$Epochs,
        '--batch-size', [string]$BatchSize,
        '--workers', [string]$Workers,
        '--learning-rate', '0.0005'
    )
}

if (-not $SkipTechniqueTraining) {
    Write-Host '[5/5] Fine-tuning Guitar Pro 8 technique models with release gates'
    foreach ($configuration in @(
        [pscustomobject]@{
            TaskRoot = 'gp8_tab_rhythm_events'
            Checkpoint = 'tab_technique_context_cnn.pt'
        },
        [pscustomobject]@{
            TaskRoot = 'gp8_score_rhythm_events'
            Checkpoint = 'technique_context_cnn.pt'
        }
    )) {
        $techniqueArguments = @(
            '-m', 'guitarocr.training.train_technique_context',
            '--database', $database,
            '--task-root', $configuration.TaskRoot,
            '--init-checkpoint', (Join-Path $workspace "weights\$($configuration.Checkpoint)"),
            '--epochs', [string]$Epochs,
            '--batch-size', [string]$BatchSize,
            '--workers', [string]$Workers,
            '--learning-rate', '0.0003',
            '--minimum-validation-precision', [string]$MinimumTechniquePrecision,
            '--minimum-validation-support', [string]$MinimumTechniqueSupport
        )
        if ($configuration.TaskRoot -eq 'gp8_score_rhythm_events') {
            $techniqueArguments += @('--disable-class', 'hammer')
        }
        Invoke-GuitarOcrPython $techniqueArguments
    }
}

if ($Promote) {
    & (Join-Path $PSScriptRoot 'promote_models.ps1') `
        -DatabaseRoot $database `
        -WeightsRoot (Join-Path $workspace 'weights') `
        -GuitarProDomain
    if ($LASTEXITCODE -ne 0) {
        throw "Model promotion failed with exit code $LASTEXITCODE"
    }
}

Write-Host "Guitar Pro domain adaptation complete. Database: $database"
