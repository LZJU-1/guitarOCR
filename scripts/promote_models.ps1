param(
    [string]$DatabaseRoot,
    [string]$WeightsRoot
)

$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'common.ps1')
$WorkspaceRoot = Split-Path -Parent $PSScriptRoot
$DatabaseRoot = if ($DatabaseRoot) { $DatabaseRoot } else { Get-GuitarOcrDatabaseRoot }
$WeightsRoot = if ($WeightsRoot) { $WeightsRoot } else { Join-Path $WorkspaceRoot 'weights' }

$models = @{
    'symbol_cnn\models\atomic_symbol_cnn.pt' = 'atomic_symbol_cnn.pt'
    'rhythm_events\models\rhythm_context_cnn.pt' = 'rhythm_context_cnn.pt'
    'score_event_locator\models\score_event_locator.pt' = 'score_event_locator.pt'
    'tab_detector\models\tab_symbol_detector.pt' = 'tab_symbol_detector.pt'
    'tie_events\models\tie_context_cnn.pt' = 'tie_context_cnn.pt'
}

New-Item -ItemType Directory -Force -Path $WeightsRoot | Out-Null
foreach ($entry in $models.GetEnumerator()) {
    $source = Join-Path $DatabaseRoot $entry.Key
    if (-not (Test-Path -LiteralPath $source)) {
        throw "Trained model is missing: $source"
    }
    Copy-Item -LiteralPath $source -Destination (Join-Path $WeightsRoot $entry.Value) -Force
    Write-Host "Promoted $($entry.Value)"
}
