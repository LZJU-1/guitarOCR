[CmdletBinding()]
param(
    [int]$SourceCount = 12,
    [ValidateSet('tuxguitar', 'musescore', 'guitarpro')]
    [string[]]$Renderers = @('tuxguitar', 'musescore', 'guitarpro'),
    [ValidateSet('score_only', 'tab_only', 'score_tab')]
    [string[]]$Layouts = @('score_only', 'tab_only', 'score_tab'),
    [string]$OutputDir,
    [string]$MuseScoreExecutable,
    [switch]$OverwriteRenderings,
    [switch]$ForceInference
)

$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'common.ps1')

$python = Get-GuitarOcrPython
$database = Get-GuitarOcrDatabaseRoot
if (-not $OutputDir) {
    $OutputDir = Join-Path $database 'cross_renderer_benchmark'
}

$buildArguments = @(
    '-m', 'guitarocr.data.build_cross_renderer_benchmark',
    '--database', $database,
    '--output', $OutputDir,
    '--source-count', $SourceCount,
    '--renderers', ($Renderers -join ','),
    '--layouts', ($Layouts -join ',')
)
if ($MuseScoreExecutable) {
    $buildArguments += @('--musescore-executable', $MuseScoreExecutable)
}
if ($OverwriteRenderings) {
    $buildArguments += '--overwrite'
}
& $python @buildArguments
if ($LASTEXITCODE -ne 0) {
    throw "Cross-renderer benchmark construction failed with exit code $LASTEXITCODE"
}

$evaluationLayouts = @($Layouts | Where-Object { $_ -in @('tab_only', 'score_tab') })
if ($evaluationLayouts.Count -eq 0) {
    Write-Host 'No currently supported OCR layouts were requested; skipping inference.'
    exit 0
}

$evaluationArguments = @(
    '-m', 'guitarocr.evaluation.evaluate_cross_renderer_benchmark',
    '--manifest', (Join-Path $OutputDir 'benchmark.jsonl'),
    '--output', (Join-Path $OutputDir 'evaluation'),
    '--layouts', ($evaluationLayouts -join ',')
)
if ($ForceInference) {
    $evaluationArguments += '--force'
}
& $python @evaluationArguments
if ($LASTEXITCODE -ne 0) {
    throw "Cross-renderer benchmark evaluation failed with exit code $LASTEXITCODE"
}

Write-Host "Benchmark summary: $(Join-Path $OutputDir 'summary.json')"
Write-Host "Evaluation summary: $(Join-Path $OutputDir 'evaluation\summary.json')"
Write-Host "Pending manual/GUI exports: $(Join-Path $OutputDir 'manual_export_queue.jsonl')"
