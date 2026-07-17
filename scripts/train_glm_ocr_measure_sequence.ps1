param(
    [string]$Python = "python",
    [string]$Config = "",
    [string]$Adapter = "",
    [string]$ResumeFromCheckpoint = "",
    [int]$EvaluationSamples = 3000,
    [switch]$SkipEvaluation
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
if ([string]::IsNullOrWhiteSpace($Config)) {
    $Config = Join-Path $ProjectRoot "configs\glm_ocr_measure_sequence_v2_lora_fp16.yaml"
}
if ([string]::IsNullOrWhiteSpace($Adapter)) {
    $Adapter = Join-Path $ProjectRoot "output\glm_ocr_measure_sequence_v2_lora"
}
$CliCommand = Get-Command "llamafactory-cli" -ErrorAction SilentlyContinue
$CliPath = if ($null -ne $CliCommand) { $CliCommand.Source } else { "" }
if ([string]::IsNullOrWhiteSpace($CliPath) -and
    (Test-Path -LiteralPath $Python -PathType Leaf)) {
    $LocalCli = Join-Path (Split-Path -Parent $Python) "Scripts\llamafactory-cli.exe"
    if (Test-Path -LiteralPath $LocalCli -PathType Leaf) {
        $CliPath = $LocalCli
    }
}
if ([string]::IsNullOrWhiteSpace($CliPath)) {
    throw "llamafactory-cli is missing from PATH and was not found next to $Python. See README.md for setup."
}

$env:CUDA_VISIBLE_DEVICES = "0"
$env:DISABLE_VERSION_CHECK = "1"
$env:PYTHONUTF8 = "1"
Push-Location $ProjectRoot
try {
    $TrainArguments = @("train", $Config)
    if (-not [string]::IsNullOrWhiteSpace($ResumeFromCheckpoint)) {
        if (-not (Test-Path -LiteralPath $ResumeFromCheckpoint -PathType Container)) {
            throw "Resume checkpoint is missing: $ResumeFromCheckpoint"
        }
        $TrainArguments += "resume_from_checkpoint=$ResumeFromCheckpoint"
    }
    & $CliPath @TrainArguments
    if ($LASTEXITCODE -ne 0) {
        throw "GLM-OCR LoRA training failed with exit code $LASTEXITCODE"
    }
    if (-not $SkipEvaluation) {
        & (Join-Path $PSScriptRoot "evaluate_glm_ocr_measure_sequence.ps1") `
            -Python $Python -Adapter $Adapter -MaxSamples $EvaluationSamples
        if ($LASTEXITCODE -ne 0) {
            throw "GLM-OCR held-out evaluation failed with exit code $LASTEXITCODE"
        }
    }
}
finally {
    Pop-Location
}
