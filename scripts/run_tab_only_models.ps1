param(
    [int]$LocatorEpochs = 35,
    [int]$RhythmEpochs = 30,
    [int]$TechniqueEpochs = 20,
    [int]$TieEpochs = 24,
    [int]$BatchSize = 64,
    [int]$Workers = 4,
    [switch]$SkipDatasetBuild,
    [switch]$SkipTraining
)

$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'common.ps1')
$WorkspaceRoot = Split-Path -Parent $PSScriptRoot
$env:PYTHONPATH = $WorkspaceRoot
$DatabaseRoot = Get-GuitarOcrDatabaseRoot
$Python = Get-GuitarOcrPython

function Invoke-GuitarOcrStep {
    param([string]$Label, [string[]]$Arguments)
    Write-Host $Label
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Label failed with exit code $LASTEXITCODE"
    }
}

if (-not $SkipDatasetBuild) {
    Invoke-GuitarOcrStep '[1/14] Building pure-TAB event-location tiles...' @(
        '-m', 'guitarocr.data.build_tab_event_locator_dataset', '--database', $DatabaseRoot)
    Invoke-GuitarOcrStep '[2/14] Validating event coverage and source isolation...' @(
        '-m', 'guitarocr.evaluation.validate_score_event_locator_dataset',
        '--database', $DatabaseRoot, '--task-root', 'tab_event_locator')
    Invoke-GuitarOcrStep '[3/14] Building semantic pure-TAB rhythm/technique crops...' @(
        '-m', 'guitarocr.data.build_tab_rhythm_dataset', '--database', $DatabaseRoot)
    Invoke-GuitarOcrStep '[4/14] Building pure-TAB tie-relation crops...' @(
        '-m', 'guitarocr.data.build_tie_event_dataset',
        '--database', $DatabaseRoot, '--layout', 'tab_only')
} else {
    Write-Host '[1-4/14] Reusing pure-TAB datasets.'
}

if (-not $SkipTraining) {
    Invoke-GuitarOcrStep '[5/14] Training the pure-TAB event locator...' @(
        '-m', 'guitarocr.training.train_score_event_locator',
        '--database', $DatabaseRoot, '--task-root', 'tab_event_locator',
        '--epochs', "$LocatorEpochs", '--batch-size', "$BatchSize", '--workers', "$Workers")
    Invoke-GuitarOcrStep '[6/14] Training the pure-TAB rhythm CNN...' @(
        '-m', 'guitarocr.training.train_rhythm_context',
        '--database', $DatabaseRoot, '--task-root', 'tab_rhythm_events',
        '--epochs', "$RhythmEpochs", '--batch-size', "$BatchSize", '--workers', "$Workers")
    Invoke-GuitarOcrStep '[7/14] Training the pure-TAB technique CNN...' @(
        '-m', 'guitarocr.training.train_technique_context',
        '--database', $DatabaseRoot, '--task-root', 'tab_rhythm_events',
        '--epochs', "$TechniqueEpochs", '--batch-size', "$BatchSize", '--workers', "$Workers")
    Invoke-GuitarOcrStep '[8/14] Training the pure-TAB tie CNN...' @(
        '-m', 'guitarocr.training.train_tie_context',
        '--database', $DatabaseRoot, '--task-root', 'tab_tie_events', '--fresh',
        '--epochs', "$TieEpochs", '--batch-size', "$BatchSize", '--workers', "$Workers")
} else {
    Write-Host '[5-8/14] Reusing trained pure-TAB models.'
}

Invoke-GuitarOcrStep '[9/14] Evaluating full-page geometry and event locations...' @(
    '-m', 'guitarocr.evaluation.evaluate_tab_event_pages', '--database', $DatabaseRoot, '--split', 'test')
Invoke-GuitarOcrStep '[10/14] Evaluating detected-centre rhythm...' @(
    '-m', 'guitarocr.evaluation.evaluate_detected_rhythm_pages',
    '--database', $DatabaseRoot, '--layout', 'tab_only', '--split', 'test')
Invoke-GuitarOcrStep '[11/14] Evaluating full-page TAB strings/frets/X...' @(
    '-m', 'guitarocr.evaluation.evaluate_tuxguitar_tab_pages',
    '--database', $DatabaseRoot, '--split', 'test')
Invoke-GuitarOcrStep '[12/14] Evaluating merged pure-TAB Event IR...' @(
    '-m', 'guitarocr.evaluation.evaluate_merged_event_ir',
    '--database', $DatabaseRoot, '--layout', 'tab_only', '--split', 'test')
Invoke-GuitarOcrStep '[13/14] Evaluating full-page tie candidates and resolution...' @(
    '-m', 'guitarocr.evaluation.evaluate_tie_event_pages',
    '--database', $DatabaseRoot, '--layout', 'tab_only', '--split', 'test')
Invoke-GuitarOcrStep '[14/14] Evaluating printed and propagated time signatures...' @(
    '-m', 'guitarocr.evaluation.evaluate_time_signatures',
    '--database', $DatabaseRoot, '--layout', 'tab_only', '--split', 'test')

Write-Host 'Complete. Pure-TAB metrics are in database\tab_event_locator\models.'
