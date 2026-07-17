param(
    [Parameter(Mandatory = $true)]
    [string]$Corpus,
    [string]$Output = "",
    [int]$SourceCount = 2000,
    [int]$Dpi = 180,
    [int]$RestartEvery = 200,
    [ValidateSet("all", "select", "relabel", "relabel-labels", "render", "crop")]
    [string]$Phase = "all",
    [string]$Python = ""
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
if ([string]::IsNullOrWhiteSpace($Output)) {
    $Output = Join-Path $ProjectRoot "database\gp8_measure_sequence_v2"
}
if ([string]::IsNullOrWhiteSpace($Python)) {
    $BundledEnvironment = Join-Path $ProjectRoot "guitar-hero-main\.venv\Scripts\python.exe"
    $Python = if (Test-Path -LiteralPath $BundledEnvironment -PathType Leaf) {
        $BundledEnvironment
    } else {
        "python"
    }
}
Push-Location $ProjectRoot
try {
    & $Python -m guitarocr.data.build_gp8_measure_sequence_dataset `
        --corpus $Corpus `
        --output $Output `
        --source-count $SourceCount `
        --dpi $Dpi `
        --restart-every $RestartEvery `
        --mode tab `
        --mode notation `
        --mode both `
        --phase $Phase
    if ($LASTEXITCODE -ne 0) {
        throw "GP8 measure-sequence dataset build failed with exit code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}
