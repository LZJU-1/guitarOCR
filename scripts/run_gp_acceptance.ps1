param(
    [Parameter(Mandatory = $true)][string]$Gp,
    [string]$OutputDir,
    [string]$Python
)

$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'common.ps1')
$WorkspaceRoot = Split-Path -Parent $PSScriptRoot
$env:PYTHONPATH = $WorkspaceRoot
$Python = if ($Python) { $Python } else { Get-GuitarOcrPython }
$Gp = (Resolve-Path -LiteralPath $Gp).Path
$OutputDir = if ($OutputDir) { $OutputDir } else { Join-Path (Split-Path -Parent $Gp) 'guitarocr_acceptance' }
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
$OutputDir = (Resolve-Path -LiteralPath $OutputDir).Path

$gt = Join-Path $OutputDir 'GT.pdf'
$preGp = Join-Path $OutputDir 'PRE.gp5'
$prePdf = Join-Path $OutputDir 'PRE.pdf'
$work = Join-Path $OutputDir 'work'
$metrics = Join-Path $OutputDir 'metrics.json'
$groundTruth = Join-Path $OutputDir 'ground_truth.json'

Write-Host '[1/4] Rendering exact TuxGuitar score+TAB ground truth...'
& $Python -m guitarocr.export.render_gp_to_pdf $Gp $gt
if ($LASTEXITCODE) { throw "GT rendering failed: $LASTEXITCODE" }

Write-Host '[2/4] Running PDF -> IR -> GP5 -> preview PDF...'
& $Python -m guitarocr.cli.pdf_to_gp $gt -o $preGp --work-dir $work --preview-pdf $prePdf --force-pdf-render
if ($LASTEXITCODE) { throw "OCR round trip failed: $LASTEXITCODE" }

Write-Host '[3/4] Comparing OCR IR with exact GPIF semantics...'
& $Python -m guitarocr.evaluation.evaluate_gpif_ir $Gp (Join-Path $work 'document_score_ir.json') `
    --output $metrics --ground-truth-output $groundTruth
if ($LASTEXITCODE) { throw "GPIF evaluation failed: $LASTEXITCODE" }

Write-Host '[4/4] Acceptance artifacts:'
Write-Host "GT=$gt"
Write-Host "PRE_GP5=$preGp"
Write-Host "PRE_PDF=$prePdf"
Write-Host "METRICS=$metrics"
