param(
    [string]$DatagenRoot = '',
    [string]$Python = '',
    [string]$ConfigPath = '',
    [string]$WorkDir = '',
    [string]$RealDatasetDir = ''
)

$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'common.ps1')

$WorkspaceRoot = Split-Path -Parent $PSScriptRoot
$DatabaseRoot = Get-GuitarOcrDatabaseRoot
$env:PYTHONPATH = $WorkspaceRoot

if (-not $DatagenRoot) {
    $DatagenRoot = if ($env:GUITAROCR_GP8_DATAGEN_ROOT) {
        $env:GUITAROCR_GP8_DATAGEN_ROOT
    } else {
        Join-Path $WorkspaceRoot 'guitar-hero-main'
    }
}
$DatagenRoot = [IO.Path]::GetFullPath($DatagenRoot)

if (-not $Python) {
    $Python = if ($env:GUITAROCR_GP8_PYTHON) {
        $env:GUITAROCR_GP8_PYTHON
    } else {
        Join-Path $DatagenRoot '.venv\Scripts\python.exe'
    }
}
if (-not $ConfigPath) {
    $ConfigPath = Join-Path $WorkspaceRoot 'configs\guitarpro8_multimode_v1.json'
}
if (-not $WorkDir) {
    $WorkDir = Join-Path $DatabaseRoot 'guitarpro8_multimode_v1'
}
if (-not $RealDatasetDir) {
    $RealDatasetDir = Join-Path $DatabaseRoot 'v2\source\gp'
}
$env:GUITAROCR_GP8_DATAGEN_ROOT = $DatagenRoot
$env:GUITAROCR_GP8_PYTHON = $Python

foreach ($required in @(
    $Python,
    $ConfigPath,
    $RealDatasetDir,
    (Join-Path $DatagenRoot 'datagen\cli.py'),
    (Join-Path $DatagenRoot 'datagen\vendor\gp8_runtime\GuitarPro.exe'),
    (Join-Path $DatagenRoot 'datagen\gt2pdf\bin\gt2pdf_inject.dll')
)) {
    if (-not (Test-Path -LiteralPath $required)) {
        throw "Required path is missing: $required"
    }
}

Write-Host '[1/3] Validating the pinned Guitar Pro 8.1.2.37 worker and injector...'
& $Python -c @'
from guitarocr.guitarpro_runtime import require_guitarpro_datagen_runtime
runtime = require_guitarpro_datagen_runtime()
print(runtime.executable)
'@
if ($LASTEXITCODE -ne 0) {
    throw "Guitar Pro runtime validation failed with exit code $LASTEXITCODE"
}

Write-Host '[2/3] Generating tab, notation, and score+tab PDFs with native layout labels...'
Push-Location $DatagenRoot
try {
    & $Python -m datagen.cli build-multimode-layout-dataset `
        --config $ConfigPath `
        --work-dir $WorkDir `
        --real-dataset-dir $RealDatasetDir
    if ($LASTEXITCODE -ne 0) {
        throw "Guitar Pro multimode dataset build failed with exit code $LASTEXITCODE"
    }
} finally {
    Pop-Location
}

Write-Host '[3/3] Re-splitting layout COCO by source song across all three display modes...'
$MergedCoco = Join-Path $WorkDir 'layout_coco'
$SourceDisjointCoco = Join-Path $WorkDir 'layout_coco_source_disjoint'
$Config = Get-Content -Raw -LiteralPath $ConfigPath | ConvertFrom-Json
$ValidationRatio = if ($null -ne $Config.layout_coco.val_ratio) {
    [double]$Config.layout_coco.val_ratio
} else {
    0.1
}
$SplitSeed = if ($null -ne $Config.layout_coco.seed) {
    [int]$Config.layout_coco.seed
} else {
    [int]$Config.seed
}
& $Python -m guitarocr.data.build_source_disjoint_layout_coco `
    --input-dir $MergedCoco `
    --output-dir $SourceDisjointCoco `
    --val-ratio $ValidationRatio `
    --seed $SplitSeed `
    --overwrite
if ($LASTEXITCODE -ne 0) {
    throw "Source-disjoint COCO split failed with exit code $LASTEXITCODE"
}

Write-Host "Complete: $WorkDir"
