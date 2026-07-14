param(
    [string]$DatabaseRoot,
    [string]$WeightsRoot,
    [switch]$GuitarProDomain
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
    'tab_event_locator\models\tab_event_locator.pt' = 'tab_event_locator.pt'
    'tab_detector\models\tab_symbol_detector.pt' = 'tab_symbol_detector.pt'
    'fret_token\models\fret_token_cnn.pt' = 'fret_token_cnn.pt'
    'tab_rhythm_events\models\rhythm_context_cnn.pt' = 'tab_rhythm_context_cnn.pt'
    'tab_tie_events\models\tab_tie_context_cnn.pt' = 'tab_tie_context_cnn.pt'
    'tab_technique_events\models\tab_technique_context_cnn.pt' = 'tab_technique_context_cnn.pt'
    'tie_events\models\tie_context_cnn.pt' = 'tie_context_cnn.pt'
    'technique_events\models\technique_context_cnn.pt' = 'technique_context_cnn.pt'
    'technique_events\models\pick_stroke_context_cnn.pt' = 'pick_stroke_context_cnn.pt'
}

if ($GuitarProDomain) {
    $models.Remove('rhythm_events\models\rhythm_context_cnn.pt')
    $models.Remove('tab_rhythm_events\models\rhythm_context_cnn.pt')
    $models.Remove('technique_events\models\technique_context_cnn.pt')
    $models.Remove('tab_technique_events\models\tab_technique_context_cnn.pt')
    $models['gp8_score_rhythm_events\models\rhythm_context_cnn.pt'] = 'rhythm_context_cnn.pt'
    $models['gp8_tab_rhythm_events\models\rhythm_context_cnn.pt'] = 'tab_rhythm_context_cnn.pt'
    $models['gp8_score_rhythm_events\models\technique_context_cnn.pt'] = 'technique_context_cnn.pt'
    $models['gp8_tab_rhythm_events\models\tab_technique_context_cnn.pt'] = 'tab_technique_context_cnn.pt'
}

New-Item -ItemType Directory -Force -Path $WeightsRoot | Out-Null
foreach ($entry in $models.GetEnumerator()) {
    # The current TuxGuitar task family lives below v2/, while the newer
    # Guitar Pro domain tasks live at the database root.  A few older base
    # tasks were never migrated, so fall back to the root when needed.
    $v2Source = Join-Path (Join-Path $DatabaseRoot 'v2') $entry.Key
    $rootSource = Join-Path $DatabaseRoot $entry.Key
    $source = if (Test-Path -LiteralPath $v2Source) { $v2Source } else { $rootSource }
    if (-not (Test-Path -LiteralPath $source)) {
        throw "Trained model is missing: $source"
    }
    Copy-Item -LiteralPath $source -Destination (Join-Path $WeightsRoot $entry.Value) -Force
    Write-Host "Promoted $($entry.Value)"
}
