param(
    [string]$Python = 'python',
    [switch]$SkipPythonPackages,
    [switch]$SkipTuxGuitar,
    [switch]$SkipPoppler
)

$ErrorActionPreference = 'Stop'
$WorkspaceRoot = Split-Path -Parent $PSScriptRoot
$VenvRoot = Join-Path $WorkspaceRoot '.venv'
$VenvPython = Join-Path $VenvRoot 'Scripts\python.exe'
$ToolsRoot = Join-Path $WorkspaceRoot 'tools'
$DownloadRoot = Join-Path $ToolsRoot 'downloads'

New-Item -ItemType Directory -Force -Path $ToolsRoot, $DownloadRoot, (Join-Path $WorkspaceRoot 'database') | Out-Null

function Assert-Sha256([string]$Path, [string]$Expected) {
    $actual = (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actual -ne $Expected.ToLowerInvariant()) {
        throw "SHA-256 mismatch for $Path. Expected $Expected, received $actual"
    }
}

if (-not (Get-Command javac -ErrorAction SilentlyContinue) -and -not $env:GUITAROCR_JAVAC) {
    throw 'JDK 17+ is required to compile the TuxGuitar bridge. Install a JDK and make javac available in PATH.'
}

if (-not $SkipPythonPackages) {
    if (-not (Test-Path -LiteralPath $VenvPython)) {
        Write-Host '[1/4] Creating Python virtual environment...'
        & $Python -m venv $VenvRoot
        if ($LASTEXITCODE -ne 0) { throw 'Could not create .venv' }
    }
    Write-Host '[1/4] Installing Python package and dependencies...'
    & $VenvPython -m pip install --upgrade pip
    & $VenvPython -m pip install -r (Join-Path $WorkspaceRoot 'requirements.txt')
    if ($LASTEXITCODE -ne 0) { throw 'Python dependency installation failed' }
} elseif (-not (Test-Path -LiteralPath $VenvPython)) {
    $VenvPython = (Get-Command $Python -ErrorAction Stop).Source
}

$TuxGuitarRoot = Join-Path $WorkspaceRoot 'tuxguitar'
if (-not $SkipTuxGuitar -and -not (Test-Path -LiteralPath (Join-Path $TuxGuitarRoot 'lib\tuxguitar.jar'))) {
    Write-Host '[2/4] Downloading TuxGuitar 2.0.1 for Windows x86_64...'
    $archive = Join-Path $DownloadRoot 'tuxguitar-2.0.1-windows-swt-x86_64.zip'
    $url = 'https://github.com/helge17/tuxguitar/releases/download/2.0.1/tuxguitar-2.0.1-windows-swt-x86_64.zip'
    Invoke-WebRequest -Uri $url -OutFile $archive
    Assert-Sha256 $archive 'facc9a81a82f1ce3cc0f5e97f4370cfbd76193a488483b03a9e5b32d15374f55'
    $stage = Join-Path $DownloadRoot 'tuxguitar-stage'
    if (Test-Path -LiteralPath $stage) { Remove-Item -LiteralPath $stage -Recurse -Force }
    Expand-Archive -LiteralPath $archive -DestinationPath $stage -Force
    $runtimeJar = Get-ChildItem -LiteralPath $stage -File -Filter 'tuxguitar.jar' -Recurse | Select-Object -First 1
    if (-not $runtimeJar) { throw 'Downloaded TuxGuitar archive did not contain tuxguitar.jar' }
    $runtimeRoot = Split-Path -Parent (Split-Path -Parent $runtimeJar.FullName)
    New-Item -ItemType Directory -Force -Path $TuxGuitarRoot | Out-Null
    Copy-Item -Path (Join-Path $runtimeRoot '*') -Destination $TuxGuitarRoot -Recurse -Force
}

$PopplerRoot = Join-Path $ToolsRoot 'poppler'
if (-not $SkipPoppler -and -not (Test-Path -LiteralPath (Join-Path $PopplerRoot 'Library\bin\pdftoppm.exe'))) {
    Write-Host '[3/4] Downloading Poppler 26.02.0-0 for Windows...'
    $archive = Join-Path $DownloadRoot 'poppler-26.02.0-0.zip'
    $url = 'https://github.com/oschwartz10612/poppler-windows/releases/download/v26.02.0-0/Release-26.02.0-0.zip'
    Invoke-WebRequest -Uri $url -OutFile $archive
    Assert-Sha256 $archive '993e4a94376ed712fafc7058d724ea0b943d118bbd2305cd9ed55174eb85cda5'
    $stage = Join-Path $DownloadRoot 'poppler-stage'
    if (Test-Path -LiteralPath $stage) { Remove-Item -LiteralPath $stage -Recurse -Force }
    if (Test-Path -LiteralPath $PopplerRoot) { Remove-Item -LiteralPath $PopplerRoot -Recurse -Force }
    Expand-Archive -LiteralPath $archive -DestinationPath $stage -Force
    $pdfToPpm = Get-ChildItem -LiteralPath $stage -File -Filter 'pdftoppm.exe' -Recurse | Select-Object -First 1
    if (-not $pdfToPpm) { throw 'Downloaded Poppler archive did not contain pdftoppm.exe' }
    $runtimeRoot = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $pdfToPpm.FullName))
    New-Item -ItemType Directory -Force -Path $PopplerRoot | Out-Null
    Copy-Item -Path (Join-Path $runtimeRoot '*') -Destination $PopplerRoot -Recurse -Force
}

Write-Host '[4/4] Checking installation...'
& $VenvPython -m guitarocr.cli.check_install
if ($LASTEXITCODE -ne 0) { throw 'GuitarOCR installation check failed' }
Write-Host 'Setup complete. Activate with: .\.venv\Scripts\Activate.ps1'
