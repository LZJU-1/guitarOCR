param(
    [int]$Epochs = 28,
    [int]$BatchSize = 16
)

$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'common.ps1')
$WorkspaceRoot = Split-Path -Parent $PSScriptRoot
$env:PYTHONPATH = $WorkspaceRoot
$DatabaseRoot = Get-GuitarOcrDatabaseRoot
$Python = Get-GuitarOcrPython

if (-not (Test-Path -LiteralPath $Python)) {
    throw "CUDA Python environment is missing: $Python"
}

Write-Host '[1/3] Verifying CUDA PyTorch...'
& $Python -c "import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name(0))"
if ($LASTEXITCODE -ne 0) { throw 'CUDA PyTorch verification failed.' }

Write-Host '[2/3] Building 512x128 TuxGuitar measure crops and detection labels...'
& $Python -m guitarocr.data.build_tab_detector_dataset --database $DatabaseRoot
if ($LASTEXITCODE -ne 0) { throw 'TAB detector dataset build failed.' }

Write-Host '[3/3] Training and evaluating the compact TAB symbol detector...'
& $Python -m guitarocr.training.train_tab_detector `
    --database $DatabaseRoot --epochs $Epochs --batch-size $BatchSize
if ($LASTEXITCODE -ne 0) { throw 'TAB detector training failed.' }

Write-Host 'Complete. Model: database\tab_detector\models\tab_symbol_detector.pt'
