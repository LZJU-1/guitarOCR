$script:GuitarOcrWorkspaceRoot = Split-Path -Parent $PSScriptRoot

function Get-GuitarOcrPython {
    if ($env:GUITAROCR_PYTHON) {
        if (-not (Test-Path -LiteralPath $env:GUITAROCR_PYTHON)) {
            throw "GUITAROCR_PYTHON does not exist: $env:GUITAROCR_PYTHON"
        }
        return $env:GUITAROCR_PYTHON
    }
    $venvPython = Join-Path $script:GuitarOcrWorkspaceRoot '.venv\Scripts\python.exe'
    if (Test-Path -LiteralPath $venvPython) {
        return $venvPython
    }
    $command = Get-Command python -ErrorAction Stop
    return $command.Source
}

function Get-GuitarOcrDatabaseRoot {
    $root = if ($env:GUITAROCR_DATABASE_ROOT) {
        $env:GUITAROCR_DATABASE_ROOT
    } else {
        Join-Path $script:GuitarOcrWorkspaceRoot 'database'
    }
    New-Item -ItemType Directory -Force -Path $root | Out-Null
    return (Resolve-Path -LiteralPath $root).Path
}

function Get-GuitarOcrTuxGuitarRoot {
    $root = if ($env:GUITAROCR_TUXGUITAR_ROOT) {
        $env:GUITAROCR_TUXGUITAR_ROOT
    } else {
        Join-Path $script:GuitarOcrWorkspaceRoot 'tuxguitar'
    }
    if (-not (Test-Path -LiteralPath (Join-Path $root 'lib\tuxguitar.jar'))) {
        throw "TuxGuitar runtime not found at $root. Run scripts\setup_windows.ps1 first."
    }
    return $root
}

function Get-GuitarOcrPopplerBin {
    $candidates = @()
    if ($env:GUITAROCR_POPPLER_BIN) {
        $candidates += $env:GUITAROCR_POPPLER_BIN
    }
    $candidates += Join-Path $script:GuitarOcrWorkspaceRoot 'tools\poppler\Library\bin'
    foreach ($candidate in $candidates) {
        if (Test-Path -LiteralPath (Join-Path $candidate 'pdftoppm.exe')) {
            return $candidate
        }
    }
    $command = Get-Command pdftoppm -ErrorAction SilentlyContinue
    if ($command) {
        return Split-Path -Parent $command.Source
    }
    throw 'Poppler was not found. Run scripts\setup_windows.ps1 or set GUITAROCR_POPPLER_BIN.'
}
