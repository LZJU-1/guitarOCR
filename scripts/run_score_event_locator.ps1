param(
    [int]$Epochs = 35,
    [int]$BatchSize = 32,
    [int]$Workers = 4,
    [switch]$RebuildRhythmLabels,
    [switch]$SkipTraining
)

$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'common.ps1')
$WorkspaceRoot = Split-Path -Parent $PSScriptRoot
$env:PYTHONPATH = $WorkspaceRoot
$DatabaseRoot = Get-GuitarOcrDatabaseRoot
$Python = Get-GuitarOcrPython

if ($RebuildRhythmLabels) {
    Write-Host '[1/11] Rebuilding score_tab rhythm labels...'
    & (Join-Path $PSScriptRoot 'build_score_rhythm_dataset.ps1')
    if ($LASTEXITCODE -ne 0) { throw "Rhythm-label build failed with exit code $LASTEXITCODE" }
} else {
    Write-Host '[1/11] Reusing existing score_tab rhythm labels.'
}

Write-Host '[2/11] Building score-measure event-location tiles...'
& $Python -m guitarocr.data.build_score_event_locator_dataset --database $DatabaseRoot
if ($LASTEXITCODE -ne 0) { throw "Event-location dataset build failed with exit code $LASTEXITCODE" }

Write-Host '[3/11] Validating coordinates and source-isolated splits...'
& $Python -m guitarocr.evaluation.validate_score_event_locator_dataset --database $DatabaseRoot
if ($LASTEXITCODE -ne 0) { throw "Event-location validation failed with exit code $LASTEXITCODE" }

if (-not $SkipTraining) {
    Write-Host '[4/11] Training the compact x-axis event locator...'
    & $Python -m guitarocr.training.train_score_event_locator `
        --database $DatabaseRoot --epochs $Epochs --batch-size $BatchSize --workers $Workers
    if ($LASTEXITCODE -ne 0) { throw "Event-location training failed with exit code $LASTEXITCODE" }
} else {
    Write-Host '[4/11] Reusing the existing event-locator model.'
}

Write-Host '[5/11] Evaluating pixel-only page geometry and event locations...'
& $Python -m guitarocr.evaluation.evaluate_score_event_pages --database $DatabaseRoot --split test
if ($LASTEXITCODE -ne 0) { throw "Page event evaluation failed with exit code $LASTEXITCODE" }

Write-Host '[6/11] Evaluating rhythm from automatically detected event centres...'
& $Python -m guitarocr.evaluation.evaluate_detected_rhythm_pages --database $DatabaseRoot --split test
if ($LASTEXITCODE -ne 0) { throw "Detected-rhythm evaluation failed with exit code $LASTEXITCODE" }

Write-Host '[7/11] Evaluating TAB fingering on score_tab pages...'
& $Python -m guitarocr.evaluation.evaluate_score_tab_fingering --database $DatabaseRoot --split test
if ($LASTEXITCODE -ne 0) { throw "score_tab fingering evaluation failed with exit code $LASTEXITCODE" }

Write-Host '[8/11] Evaluating merged rhythm/fingering Event IR...'
& $Python -m guitarocr.evaluation.evaluate_merged_event_ir --database $DatabaseRoot --split test
if ($LASTEXITCODE -ne 0) { throw "Merged Event IR evaluation failed with exit code $LASTEXITCODE" }

Write-Host '[9/11] Evaluating printed time signatures and document propagation...'
& $Python -m guitarocr.evaluation.evaluate_time_signatures `
    --database $DatabaseRoot --split test --threshold 0.20
if ($LASTEXITCODE -ne 0) { throw "Time-signature evaluation failed with exit code $LASTEXITCODE" }

Write-Host '[10/11] Evaluating exact measure-duration audits and correction proposals...'
& $Python -m guitarocr.evaluation.evaluate_measure_rhythm_constraints `
    --database $DatabaseRoot --split test --time-signature-threshold 0.20
if ($LASTEXITCODE -ne 0) { throw "Measure-constraint evaluation failed with exit code $LASTEXITCODE" }

Write-Host '[11/11] Evaluating tie candidates and conservative relation resolution...'
& $Python -m guitarocr.evaluation.evaluate_tie_event_pages --database $DatabaseRoot --split test
if ($LASTEXITCODE -ne 0) { throw "Tie relation evaluation failed with exit code $LASTEXITCODE" }

Write-Host 'Complete. Metrics are in database\score_event_locator\models.'
