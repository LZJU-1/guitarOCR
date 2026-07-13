param(
    [int]$Limit = 0,
    [ValidateSet('tab_only', 'score_tab')]
    [string]$Layout = 'tab_only'
)

$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'common.ps1')

$WorkspaceRoot = Split-Path -Parent $PSScriptRoot
$env:PYTHONPATH = $WorkspaceRoot
$DatabaseRoot = Get-GuitarOcrDatabaseRoot
$TuxGuitarRoot = Get-GuitarOcrTuxGuitarRoot
$JavaSource = Join-Path $WorkspaceRoot 'java\TuxGuitarTabAnnotationBuilder.java'
$PythonSource = Join-Path $WorkspaceRoot 'guitarocr\data\build_tuxguitar_page_annotations.py'
$JavaBin = Join-Path $DatabaseRoot 'tmp\java_classes'
$Python = Get-GuitarOcrPython

foreach ($required in @(
    (Join-Path $TuxGuitarRoot 'jre\bin\java.exe'),
    $JavaSource,
    $PythonSource,
    $Python,
    (Join-Path $DatabaseRoot 'source\gp'),
    (Join-Path $DatabaseRoot "output\images\$Layout")
)) {
    if (-not (Test-Path -LiteralPath $required)) {
        throw "Required path is missing: $required"
    }
}

New-Item -ItemType Directory -Force -Path $JavaBin | Out-Null
$ClassPath = @(
    $JavaBin,
    (Join-Path $TuxGuitarRoot 'lib\*'),
    (Join-Path $TuxGuitarRoot 'share\plugins\*'),
    (Join-Path $TuxGuitarRoot 'share'),
    (Join-Path $TuxGuitarRoot 'dist')
) -join ';'

Write-Host '[1/3] Compiling the TuxGuitar layout annotation exporter...'
& javac -encoding UTF-8 -cp $ClassPath -d $JavaBin $JavaSource
if ($LASTEXITCODE -ne 0) {
    throw "javac failed with exit code $LASTEXITCODE"
}

Write-Host "[2/3] Replaying TuxGuitar $Layout layout and recording logical coordinates..."
$Java = Join-Path $TuxGuitarRoot 'jre\bin\java.exe'
$JavaArgs = @(
    '-Xmx3g',
    '-Djava.awt.headless=true',
    "-Dtuxguitar.home.path=$TuxGuitarRoot",
    '-cp',
    $ClassPath,
    'TuxGuitarTabAnnotationBuilder',
    $DatabaseRoot,
    $Limit.ToString(),
    $Layout
)
& $Java @JavaArgs
if ($LASTEXITCODE -ne 0) {
    throw "TuxGuitar coordinate export failed with exit code $LASTEXITCODE"
}

Write-Host '[3/3] Converting to PNG pixels and drawing annotation overlays...'
& $Python -m guitarocr.data.build_tuxguitar_page_annotations --database $DatabaseRoot --limit $Limit --layout $Layout
if ($LASTEXITCODE -ne 0) {
    throw "PNG annotation conversion failed with exit code $LASTEXITCODE"
}

Write-Host "Complete. Inspect overlays under database\output\annotation_overlays\$Layout."
