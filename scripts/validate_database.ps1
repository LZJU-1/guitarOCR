param([string]$DatabaseRoot)

$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'common.ps1')
$DatabaseRoot = if ($DatabaseRoot) { (Resolve-Path $DatabaseRoot).Path } else { Get-GuitarOcrDatabaseRoot }
$PopplerBin = Get-GuitarOcrPopplerBin
$PdfInfo = Join-Path $PopplerBin 'pdfinfo.exe'
$SourcesManifest = Join-Path $DatabaseRoot 'manifests\sources.jsonl'
$SamplesManifest = Join-Path $DatabaseRoot 'manifests\samples.jsonl'
$SummaryPath = Join-Path $DatabaseRoot 'manifests\dataset_summary.json'

function Resolve-DatasetPath([string]$RelativePath) {
    return Join-Path $DatabaseRoot ($RelativePath -replace '/', '\')
}

foreach ($required in @($PdfInfo, $SourcesManifest, $SamplesManifest, $SummaryPath)) {
    if (-not (Test-Path -LiteralPath $required)) {
        throw "Missing required file: $required"
    }
}

$sources = @(Get-Content -LiteralPath $SourcesManifest -Encoding UTF8 | Where-Object { $_.Trim() } | ForEach-Object { $_ | ConvertFrom-Json })
$samples = @(Get-Content -LiteralPath $SamplesManifest -Encoding UTF8 | Where-Object { $_.Trim() } | ForEach-Object { $_ | ConvertFrom-Json })
$summary = Get-Content -LiteralPath $SummaryPath -Raw -Encoding UTF8 | ConvertFrom-Json

if ($sources.Count -ne $summary.source_song_count) {
    throw "Source count mismatch: manifest=$($sources.Count), summary=$($summary.source_song_count)"
}
if ($samples.Count -ne $summary.image_page_count) {
    throw "Image count mismatch: manifest=$($samples.Count), summary=$($summary.image_page_count)"
}

$sourceIds = [System.Collections.Generic.HashSet[string]]::new()
foreach ($source in $sources) {
    if (-not $sourceIds.Add([string]$source.id)) {
        throw "Duplicate source id: $($source.id)"
    }

    $gpPath = Resolve-DatasetPath $source.source_gp
    $labelPath = Resolve-DatasetPath $source.label_json
    if (-not (Test-Path -LiteralPath $gpPath)) { throw "Missing GP source: $gpPath" }
    if (-not (Test-Path -LiteralPath $labelPath)) { throw "Missing song label: $labelPath" }

    $actualHash = (Get-FileHash -LiteralPath $gpPath -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actualHash -ne $source.sha256) {
        throw "SHA-256 mismatch for $gpPath"
    }
    $null = Get-Content -LiteralPath $labelPath -Raw -Encoding UTF8 | ConvertFrom-Json
}

$splitIds = [System.Collections.Generic.HashSet[string]]::new()
foreach ($split in @('train', 'validation', 'test')) {
    $splitPath = Join-Path $DatabaseRoot "splits\$split.txt"
    foreach ($id in @(Get-Content -LiteralPath $splitPath -Encoding UTF8 | Where-Object { $_.Trim() })) {
        if (-not $sourceIds.Contains($id)) { throw "Unknown id in ${split}: $id" }
        if (-not $splitIds.Add($id)) { throw "Source occurs in multiple splits: $id" }
    }
}
if ($splitIds.Count -ne $sourceIds.Count) {
    throw "Split files cover $($splitIds.Count) of $($sourceIds.Count) sources"
}

$pdfPaths = [System.Collections.Generic.HashSet[string]]::new()
$sampleIds = [System.Collections.Generic.HashSet[string]]::new()
foreach ($sample in $samples) {
    if (-not $sampleIds.Add([string]$sample.sample_id)) { throw "Duplicate sample id: $($sample.sample_id)" }
    if (-not $sourceIds.Contains([string]$sample.source_id)) { throw "Unknown sample source: $($sample.source_id)" }

    $pdfPath = Resolve-DatasetPath $sample.pdf
    $imagePath = Resolve-DatasetPath $sample.image
    if (-not (Test-Path -LiteralPath $pdfPath)) { throw "Missing PDF: $pdfPath" }
    if (-not (Test-Path -LiteralPath $imagePath)) { throw "Missing PNG: $imagePath" }
    $null = $pdfPaths.Add($pdfPath)

    $stream = [IO.File]::OpenRead($imagePath)
    try {
        $header = New-Object byte[] 26
        if ($stream.Read($header, 0, $header.Length) -ne $header.Length) { throw "Truncated PNG: $imagePath" }
    } finally {
        $stream.Dispose()
    }
    $pngSignature = [byte[]](137, 80, 78, 71, 13, 10, 26, 10)
    for ($i = 0; $i -lt $pngSignature.Length; $i++) {
        if ($header[$i] -ne $pngSignature[$i]) { throw "Invalid PNG signature: $imagePath" }
    }
    $width = [Net.IPAddress]::NetworkToHostOrder([BitConverter]::ToInt32($header, 16))
    $height = [Net.IPAddress]::NetworkToHostOrder([BitConverter]::ToInt32($header, 20))
    if ($width -lt 1000 -or $height -lt 1000) { throw "Unexpected PNG dimensions ${width}x${height}: $imagePath" }
}

if ($pdfPaths.Count -ne $summary.pdf_count) {
    throw "PDF count mismatch: manifest=$($pdfPaths.Count), summary=$($summary.pdf_count)"
}
foreach ($pdfPath in $pdfPaths) {
    $pdfInfoOutput = & $PdfInfo $pdfPath 2>$null
    if ($LASTEXITCODE -ne 0 -or -not ($pdfInfoOutput -match '^Pages:\s+\d+')) {
        throw "Unreadable PDF: $pdfPath"
    }
}

Write-Host ("Validation passed: {0} sources, {1} PDFs, {2} PNG pages, {3} labels" -f `
    $sources.Count, $pdfPaths.Count, $samples.Count, $sources.Count)
