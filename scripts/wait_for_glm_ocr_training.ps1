param(
    [Parameter(Mandatory = $true)]
    [int]$TrainingProcessId,
    [string]$Python = "python",
    [string]$Adapter = "",
    [int]$EvaluationSamples = 3000
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
if ([string]::IsNullOrWhiteSpace($Adapter)) {
    $Adapter = Join-Path $ProjectRoot "output\glm_ocr_measure_sequence_v2_lora"
}

if (Get-Process -Id $TrainingProcessId -ErrorAction SilentlyContinue) {
    Wait-Process -Id $TrainingProcessId
}

$AdapterWeights = Join-Path $Adapter "adapter_model.safetensors"
$AdapterConfig = Join-Path $Adapter "adapter_config.json"
if (-not (Test-Path -LiteralPath $AdapterWeights -PathType Leaf) -or
    -not (Test-Path -LiteralPath $AdapterConfig -PathType Leaf)) {
    throw "Training ended without a complete adapter in $Adapter"
}

& (Join-Path $PSScriptRoot "evaluate_glm_ocr_measure_sequence.ps1") `
    -Python $Python -Adapter $Adapter -MaxSamples $EvaluationSamples
if ($LASTEXITCODE -ne 0) {
    throw "Post-training held-out evaluation failed with exit code $LASTEXITCODE"
}
