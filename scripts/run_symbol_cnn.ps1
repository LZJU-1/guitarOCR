param(
    [int]$Epochs = 18,
    [int]$BatchSize = 256,
    [int]$TrainPerClass = 400,
    [int]$ValidationPerClass = 80,
    [int]$TestPerClass = 80
)

$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'common.ps1')
$WorkspaceRoot = Split-Path -Parent $PSScriptRoot
$env:PYTHONPATH = $WorkspaceRoot
$DatabaseRoot = Get-GuitarOcrDatabaseRoot
$SymbolRoot = Join-Path $DatabaseRoot 'symbol_cnn'
$TemplateRoot = Join-Path $SymbolRoot 'templates'
$DatasetRoot = Join-Path $SymbolRoot 'dataset'
$ModelRoot = Join-Path $SymbolRoot 'models'
$JavaBin = Join-Path $DatabaseRoot 'tmp\java_classes'
$TuxGuitarRoot = Get-GuitarOcrTuxGuitarRoot
$CudaPython = Get-GuitarOcrPython
$JavaSource = Join-Path $WorkspaceRoot 'java\TuxGuitarAtomicSymbolBuilder.java'

foreach ($required in @(
    $CudaPython,
    $JavaSource,
    (Join-Path $WorkspaceRoot 'guitarocr\data\build_symbol_dataset.py'),
    (Join-Path $WorkspaceRoot 'guitarocr\training\train_symbol_cnn.py'),
    (Join-Path $WorkspaceRoot 'guitarocr\models\symbol_model.py'),
    (Join-Path $TuxGuitarRoot 'jre\bin\java.exe')
)) {
    if (-not (Test-Path -LiteralPath $required)) {
        throw "Required path is missing: $required"
    }
}

New-Item -ItemType Directory -Force -Path $TemplateRoot, $ModelRoot, $JavaBin | Out-Null

Write-Host '[1/4] Verifying the CUDA PyTorch environment...'
& $CudaPython -c "import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name(0))"
if ($LASTEXITCODE -ne 0) { throw 'CUDA PyTorch verification failed' }

Write-Host '[2/4] Rendering exact TuxGuitar atomic templates...'
$resolvedTemplateRoot = (Resolve-Path -LiteralPath $TemplateRoot).Path
$expectedTemplateRoot = [IO.Path]::GetFullPath((Join-Path $DatabaseRoot 'symbol_cnn\templates'))
if (-not $resolvedTemplateRoot.Equals($expectedTemplateRoot, [StringComparison]::OrdinalIgnoreCase)) {
    throw "Refusing to clean unexpected template directory: $resolvedTemplateRoot"
}
Get-ChildItem -LiteralPath $resolvedTemplateRoot -File -Filter '*.png' -ErrorAction SilentlyContinue | Remove-Item -Force

$ClassPath = @(
    $JavaBin,
    (Join-Path $TuxGuitarRoot 'lib\*'),
    (Join-Path $TuxGuitarRoot 'share\plugins\*'),
    (Join-Path $TuxGuitarRoot 'share'),
    (Join-Path $TuxGuitarRoot 'dist')
) -join ';'
& javac -encoding UTF-8 -cp $ClassPath -d $JavaBin $JavaSource
if ($LASTEXITCODE -ne 0) { throw 'Atomic template Java compilation failed' }
& (Join-Path $TuxGuitarRoot 'jre\bin\java.exe') '-Djava.awt.headless=true' -cp $ClassPath `
    TuxGuitarAtomicSymbolBuilder $TemplateRoot
if ($LASTEXITCODE -ne 0) { throw 'Atomic template rendering failed' }

Write-Host '[3/4] Building the labeled synthetic symbol dataset...'
& $CudaPython -m guitarocr.data.build_symbol_dataset `
    --templates $TemplateRoot `
    --output $DatasetRoot `
    --train-per-class $TrainPerClass `
    --validation-per-class $ValidationPerClass `
    --test-per-class $TestPerClass
if ($LASTEXITCODE -ne 0) { throw 'Symbol dataset generation failed' }

Write-Host '[4/4] Training and evaluating the compact CNN...'
& $CudaPython -m guitarocr.training.train_symbol_cnn `
    --data $DatasetRoot `
    --output $ModelRoot `
    --epochs $Epochs `
    --batch-size $BatchSize `
    --workers 4
if ($LASTEXITCODE -ne 0) { throw 'CNN training failed' }

Write-Host "Complete. Model: $(Join-Path $ModelRoot 'atomic_symbol_cnn.pt')"
