param(
    [int]$Limit = 0
)

$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'common.ps1')

$WorkspaceRoot = Split-Path -Parent $PSScriptRoot
$env:PYTHONPATH = $WorkspaceRoot
$DatabaseRoot = Get-GuitarOcrDatabaseRoot
$TuxGuitarRoot = Get-GuitarOcrTuxGuitarRoot
$JavaSource = Join-Path $WorkspaceRoot 'java\TuxGuitarScoreRhythmAnnotationBuilder.java'
$PythonSource = Join-Path $WorkspaceRoot 'guitarocr\data\build_score_rhythm_dataset.py'
$Validator = Join-Path $WorkspaceRoot 'guitarocr\evaluation\validate_score_rhythm_dataset.py'
$JavaBin = Join-Path $DatabaseRoot 'tmp\java_classes'
$Python = Get-GuitarOcrPython

foreach ($required in @(
    (Join-Path $TuxGuitarRoot 'jre\bin\java.exe'),
    $JavaSource,
    $PythonSource,
    $Validator,
    $Python,
    (Join-Path $DatabaseRoot 'source\gp'),
    (Join-Path $DatabaseRoot 'output\images\score_tab')
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

Write-Host '[1/4] Compiling the score+TAB rhythm exporter...'
& javac -encoding UTF-8 -cp $ClassPath -d $JavaBin $JavaSource
if ($LASTEXITCODE -ne 0) { throw "javac failed with exit code $LASTEXITCODE" }

Write-Host '[2/4] Replaying TuxGuitar score_tab layout and exporting rhythm events...'
$Java = Join-Path $TuxGuitarRoot 'jre\bin\java.exe'
& $Java '-Xmx3g' '-Djava.awt.headless=true' "-Dtuxguitar.home.path=$TuxGuitarRoot" `
    '-cp' $ClassPath 'TuxGuitarScoreRhythmAnnotationBuilder' $DatabaseRoot $Limit.ToString()
if ($LASTEXITCODE -ne 0) { throw "Rhythm coordinate export failed with exit code $LASTEXITCODE" }

Write-Host '[3/4] Building pixel labels, event crops and QA overlays...'
& $Python -m guitarocr.data.build_score_rhythm_dataset --database $DatabaseRoot --limit $Limit
if ($LASTEXITCODE -ne 0) { throw "Rhythm dataset build failed with exit code $LASTEXITCODE" }

Write-Host '[4/4] Validating semantics, coordinates and source-disjoint splits...'
& $Python -m guitarocr.evaluation.validate_score_rhythm_dataset --database $DatabaseRoot --limit $Limit
if ($LASTEXITCODE -ne 0) { throw "Rhythm dataset validation failed with exit code $LASTEXITCODE" }

Write-Host 'Complete. Inspect database\output\annotation_overlays\score_tab_rhythm.'
