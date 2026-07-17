[CmdletBinding()]
param(
    [Parameter(Mandatory = $true, Position = 0)]
    [string]$InputPath,

    [Parameter(Mandatory = $true, Position = 1)]
    [string]$OutputDirectory,

    [ValidateSet("auto", "tab", "notation", "both")]
    [string]$Mode = "auto",

    [string]$Python = "python",
    [string]$BaseModel,
    [string]$Adapter,

    [ValidateSet("cuda", "cpu")]
    [string]$Device = "cuda",

    [int]$MaximumAttempts = 3,
    [int]$MaxNewTokens = 512,
    [int]$MaxNewTokensCeiling = 2048,

    [string]$Title,
    [string]$Artist,
    [string]$Tuning,
    [int]$Capo = 0,

    [switch]$Resume,
    [switch]$ForcePdfRender,
    [switch]$RenderWithGuitarPro,
    [string]$PreviewPdf
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$ResolvedInput = (Resolve-Path -LiteralPath $InputPath).Path

if ([string]::IsNullOrWhiteSpace($BaseModel)) {
    $BaseModel = Join-Path $ProjectRoot "tools\models\GLM-OCR"
}
if ([string]::IsNullOrWhiteSpace($Adapter)) {
    $Adapter = Join-Path $ProjectRoot "weights\glm_ocr_measure_sequence_v2_lora"
}
if (-not (Test-Path -LiteralPath $BaseModel -PathType Container)) {
    throw "GLM-OCR base model not found: $BaseModel. Run scripts\download_glm_ocr_base.ps1 first."
}
if (-not (Test-Path -LiteralPath (Join-Path $Adapter "adapter_config.json") -PathType Leaf)) {
    throw "LoRA adapter_config.json not found under: $Adapter"
}
if (-not (Test-Path -LiteralPath (Join-Path $Adapter "adapter_model.safetensors") -PathType Leaf)) {
    throw "LoRA adapter_model.safetensors not found under: $Adapter. Pull Git LFS files."
}
if ($MaxNewTokensCeiling -lt $MaxNewTokens) {
    throw "MaxNewTokensCeiling must be greater than or equal to MaxNewTokens."
}

$OutputDirectory = [System.IO.Path]::GetFullPath($OutputDirectory)
New-Item -ItemType Directory -Force -Path $OutputDirectory | Out-Null
$env:PYTHONPATH = if ($env:PYTHONPATH) {
    "$ProjectRoot;$env:PYTHONPATH"
} else {
    $ProjectRoot
}
$env:PYTHONUNBUFFERED = "1"

$InferenceArguments = @(
    "-m", "guitarocr.pipeline.infer_glm_ocr_document",
    $ResolvedInput,
    "--output", $OutputDirectory,
    "--mode", $Mode,
    "--model", $BaseModel,
    "--adapter", $Adapter,
    "--device", $Device,
    "--maximum-attempts", "$MaximumAttempts",
    "--max-new-tokens", "$MaxNewTokens",
    "--max-new-tokens-ceiling", "$MaxNewTokensCeiling",
    "--capo", "$Capo"
)
if ($Resume) {
    $InferenceArguments += "--resume"
}
if ($ForcePdfRender) {
    $InferenceArguments += "--force-pdf-render"
}
if (-not [string]::IsNullOrWhiteSpace($Title)) {
    $InferenceArguments += @("--title", $Title)
}
if (-not [string]::IsNullOrWhiteSpace($Artist)) {
    $InferenceArguments += @("--artist", $Artist)
}
if (-not [string]::IsNullOrWhiteSpace($Tuning)) {
    $InferenceArguments += @("--tuning", $Tuning)
}

Push-Location $ProjectRoot
try {
    & $Python @InferenceArguments
    if ($LASTEXITCODE -ne 0) {
        throw "GuitarOCR inference failed with exit code $LASTEXITCODE."
    }

    $ManifestPath = Join-Path $OutputDirectory "manifest.json"
    $Manifest = Get-Content -LiteralPath $ManifestPath -Raw -Encoding UTF8 | ConvertFrom-Json
    $Gp5Path = [string]$Manifest.gp5
    $M2Path = [string]$Manifest.m2

    if ($RenderWithGuitarPro) {
        if ([string]::IsNullOrWhiteSpace($PreviewPdf)) {
            $PreviewPdf = Join-Path $OutputDirectory "PRE.pdf"
        } else {
            $PreviewPdf = [System.IO.Path]::GetFullPath($PreviewPdf)
        }
        $LayoutJson = [System.IO.Path]::ChangeExtension($PreviewPdf, ".layout.json")
        & $Python -m guitarocr.export.render_gp_to_guitarpro_pdf `
            $Gp5Path $PreviewPdf `
            --layout-json $LayoutJson `
            --display-mode ([string]$Manifest.mode)
        if ($LASTEXITCODE -ne 0) {
            throw "Official Guitar Pro 8 preview rendering failed with exit code $LASTEXITCODE."
        }
    }

    [pscustomobject]@{
        Mode = [string]$Manifest.mode
        Measures = [int]$Manifest.measures
        M2 = $M2Path
        GP5 = $Gp5Path
        PreviewPDF = if ($RenderWithGuitarPro) { $PreviewPdf } else { $null }
        Manifest = $ManifestPath
        RecognitionLog = Join-Path $OutputDirectory "recognition.jsonl"
    } | Format-List
}
finally {
    Pop-Location
}
