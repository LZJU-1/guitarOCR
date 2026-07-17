[CmdletBinding()]
param(
    [string]$Destination,
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
if ([string]::IsNullOrWhiteSpace($Destination)) {
    $Destination = Join-Path $ProjectRoot "tools\models\GLM-OCR"
}
$Destination = [System.IO.Path]::GetFullPath($Destination)
New-Item -ItemType Directory -Force -Path $Destination | Out-Null

$Program = @"
from huggingface_hub import snapshot_download
import sys

snapshot_download(
    repo_id="zai-org/GLM-OCR",
    local_dir=sys.argv[1],
)
"@

& $Python -c $Program $Destination
if ($LASTEXITCODE -ne 0) {
    throw "GLM-OCR base-model download failed with exit code $LASTEXITCODE."
}

Write-Host "GLM-OCR base model: $Destination"
