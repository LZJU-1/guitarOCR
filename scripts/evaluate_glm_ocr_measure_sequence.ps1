param(
    [string]$Python = "python",
    [string]$Adapter = "",
    [int]$MaxSamples = 3000,
    [switch]$Resume
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
if ([string]::IsNullOrWhiteSpace($Adapter)) {
    $Adapter = Join-Path $ProjectRoot "output\glm_ocr_measure_sequence_v2_lora"
}
$Manifest = Join-Path $ProjectRoot "database\gp8_measure_sequence_v2\manifests\test.jsonl"
$Predictions = Join-Path $ProjectRoot "reports\glm_ocr_measure_sequence_v2_test_predictions.jsonl"
$Metrics = Join-Path $ProjectRoot "reports\glm_ocr_measure_sequence_v2_test_metrics.json"
$Gate = Join-Path $ProjectRoot "configs\measure_sequence_release_gate.json"
$GateReport = Join-Path $ProjectRoot "reports\glm_ocr_measure_sequence_v2_release_gate.json"

if (-not (Test-Path -LiteralPath $Manifest -PathType Leaf)) {
    throw "The source-disjoint v2 test manifest is missing: $Manifest"
}
if (-not (Test-Path -LiteralPath $Adapter -PathType Container)) {
    throw "The trained v2 adapter is missing: $Adapter"
}

$Arguments = @(
    "-m", "guitarocr.evaluation.infer_glm_ocr_measure_sequence",
    "--manifest", $Manifest,
    "--adapter", $Adapter,
    "--predictions", $Predictions,
    "--metrics", $Metrics,
    "--max-samples", $MaxSamples
)
if ($Resume) {
    $Arguments += "--resume"
}

Push-Location $ProjectRoot
try {
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "GLM-OCR held-out inference failed with exit code $LASTEXITCODE"
    }
    & $Python -m guitarocr.evaluation.check_measure_sequence_release `
        $Metrics --gate $Gate --report $GateReport
    if ($LASTEXITCODE -ne 0) {
        throw "The v2 adapter did not pass the M2 release gate. See $GateReport"
    }
}
finally {
    Pop-Location
}
