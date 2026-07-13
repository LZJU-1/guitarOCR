param(
    [int]$PerFormat = 10,
    [int]$Dpi = 180,
    [string]$CorpusRoot,
    [string]$DatabaseRoot,
    [switch]$RenderOnly
)

$ErrorActionPreference = 'Stop'
. (Join-Path $PSScriptRoot 'common.ps1')

$WorkspaceRoot = Split-Path -Parent $PSScriptRoot
$DatabaseRoot = if ($DatabaseRoot) { $DatabaseRoot } else { Get-GuitarOcrDatabaseRoot }
$CorpusRoot = if ($CorpusRoot) { $CorpusRoot } else { Join-Path $WorkspaceRoot 'music-scores-collection\files\guitar_pro' }
New-Item -ItemType Directory -Force -Path $DatabaseRoot | Out-Null
$DatabaseRoot = (Resolve-Path $DatabaseRoot).Path
$CorpusRoot = (Resolve-Path $CorpusRoot).Path
$TuxGuitarRoot = Get-GuitarOcrTuxGuitarRoot
$JavaSource = Join-Path $WorkspaceRoot 'java\TuxGuitarDatasetBuilder.java'
$JavaBin = Join-Path $DatabaseRoot 'tmp\java_classes'
$BuildLog = Join-Path $DatabaseRoot 'logs\build_java.log'
$SourcesManifest = Join-Path $DatabaseRoot 'manifests\sources.jsonl'
$SamplesManifest = Join-Path $DatabaseRoot 'manifests\samples.jsonl'
$PopplerBin = Get-GuitarOcrPopplerBin
$PdfToPpm = Join-Path $PopplerBin 'pdftoppm.exe'
$PdfInfo = Join-Path $PopplerBin 'pdfinfo.exe'

foreach ($required in @(
    (Join-Path $TuxGuitarRoot 'jre\bin\java.exe'),
    $CorpusRoot,
    $JavaSource,
    $PdfToPpm,
    $PdfInfo
)) {
    if (-not (Test-Path -LiteralPath $required)) {
        throw "Required path is missing: $required"
    }
}

foreach ($directory in @(
    $JavaBin,
    (Join-Path $DatabaseRoot 'source\gp'),
    (Join-Path $DatabaseRoot 'output\pdf'),
    (Join-Path $DatabaseRoot 'output\images'),
    (Join-Path $DatabaseRoot 'labels\songs'),
    (Join-Path $DatabaseRoot 'manifests'),
    (Join-Path $DatabaseRoot 'splits'),
    (Join-Path $DatabaseRoot 'logs'),
    (Join-Path $DatabaseRoot 'tmp\pdfs')
)) {
    New-Item -ItemType Directory -Force -Path $directory | Out-Null
}

$ClassPath = @(
    $JavaBin,
    (Join-Path $TuxGuitarRoot 'lib\*'),
    (Join-Path $TuxGuitarRoot 'share\plugins\*'),
    (Join-Path $TuxGuitarRoot 'share'),
    (Join-Path $TuxGuitarRoot 'dist')
) -join ';'

if (-not $RenderOnly) {
    Write-Host '[1/4] Compiling the headless TuxGuitar exporter...'
    & javac -encoding UTF-8 -cp $ClassPath -d $JavaBin $JavaSource
    if ($LASTEXITCODE -ne 0) {
        throw "javac failed with exit code $LASTEXITCODE"
    }

    Write-Host '[2/4] Selecting GP sources and exporting PDF layouts...'
    $Java = Join-Path $TuxGuitarRoot 'jre\bin\java.exe'
    $JavaArgs = @(
        '-Xmx3g',
        '-Djava.awt.headless=true',
        "-Dtuxguitar.home.path=$TuxGuitarRoot",
        '-cp',
        $ClassPath,
        'TuxGuitarDatasetBuilder',
        $CorpusRoot,
        $DatabaseRoot,
        $PerFormat.ToString()
    )
    $JavaStdoutLog = Join-Path $DatabaseRoot 'logs\tuxguitar_stdout.log'
    $JavaStderrLog = Join-Path $DatabaseRoot 'logs\tuxguitar_stderr.log'
    Remove-Item -LiteralPath $JavaStdoutLog, $JavaStderrLog -Force -ErrorAction SilentlyContinue
    $JavaProcess = Start-Process -FilePath $Java -ArgumentList $JavaArgs -NoNewWindow -Wait -PassThru `
        -RedirectStandardOutput $JavaStdoutLog -RedirectStandardError $JavaStderrLog
    $JavaOutput = @(
        Get-Content -LiteralPath $JavaStdoutLog -Encoding UTF8 -ErrorAction SilentlyContinue
        Get-Content -LiteralPath $JavaStderrLog -Encoding UTF8 -ErrorAction SilentlyContinue
    )
    $JavaOutput | Set-Content -LiteralPath $BuildLog -Encoding UTF8
    $JavaOutput | ForEach-Object { Write-Host $_ }
    if ($JavaProcess.ExitCode -ne 0) {
        throw "TuxGuitar dataset export failed with exit code $($JavaProcess.ExitCode)"
    }
} else {
    Write-Host '[1/4] Reusing the existing source manifest and exported PDFs...'
}

if (-not (Test-Path -LiteralPath $SourcesManifest)) {
    throw "Source manifest was not created: $SourcesManifest"
}

Write-Host '[3/4] Rendering PDF pages to grayscale PNG...'
$utf8 = [System.Text.UTF8Encoding]::new($false)
$sources = @(Get-Content -LiteralPath $SourcesManifest -Encoding UTF8 | Where-Object { $_.Trim() } | ForEach-Object { $_ | ConvertFrom-Json })
$orderedSources = @($sources | Sort-Object sha256)
$testSourceCount = [Math]::Max(1, [int][Math]::Round($sources.Count * 0.10))
$validationSourceCount = [Math]::Max(1, [int][Math]::Round($sources.Count * 0.10))
for ($sourceIndex = 0; $sourceIndex -lt $orderedSources.Count; $sourceIndex++) {
    if ($sourceIndex -lt $testSourceCount) {
        $orderedSources[$sourceIndex].split = 'test'
    } elseif ($sourceIndex -lt ($testSourceCount + $validationSourceCount)) {
        $orderedSources[$sourceIndex].split = 'validation'
    } else {
        $orderedSources[$sourceIndex].split = 'train'
    }
}
$normalisedSourceLines = @($sources | ForEach-Object { $_ | ConvertTo-Json -Compress })
[IO.File]::WriteAllLines($SourcesManifest, $normalisedSourceLines, $utf8)
$styles = @('tab_only', 'score_tab', 'score_only')
$sampleLines = [System.Collections.Generic.List[string]]::new()

function Get-RelativePath([string]$Path) {
    $rootUri = [Uri]::new(($DatabaseRoot.TrimEnd('\') + '\'))
    $pathUri = [Uri]::new($Path)
    return [Uri]::UnescapeDataString($rootUri.MakeRelativeUri($pathUri).ToString())
}

foreach ($source in $sources) {
    foreach ($style in $styles) {
        $pdf = Join-Path $DatabaseRoot ("output\pdf\{0}\{1}.pdf" -f $style, $source.id)
        if (-not (Test-Path -LiteralPath $pdf)) {
            throw "Missing exported PDF: $pdf"
        }

        $pdfInfoText = & $PdfInfo $pdf 2>&1
        if ($LASTEXITCODE -ne 0) {
            throw "pdfinfo failed for $pdf`n$pdfInfoText"
        }
        $pageLine = $pdfInfoText | Where-Object { $_ -match '^Pages:\s+(\d+)' } | Select-Object -First 1
        if (-not $pageLine -or $pageLine -notmatch '^Pages:\s+(\d+)') {
            throw "Could not read page count from $pdf"
        }
        $expectedPages = [int]$Matches[1]

        $imageDir = Join-Path $DatabaseRoot ("output\images\{0}\{1}" -f $style, $source.id)
        New-Item -ItemType Directory -Force -Path $imageDir | Out-Null
        $existingPages = @(Get-ChildItem -LiteralPath $imageDir -File -Filter 'page_*.png' -ErrorAction SilentlyContinue)
        if ($existingPages.Count -ne $expectedPages) {
            Get-ChildItem -LiteralPath $imageDir -File -Filter '*.png' -ErrorAction SilentlyContinue | Remove-Item -Force
            $prefix = Join-Path $imageDir 'render'
            $previousErrorPreference = $ErrorActionPreference
            try {
                $ErrorActionPreference = 'Continue'
                $renderOutput = & $PdfToPpm -r $Dpi -gray -png $pdf $prefix 2>&1
                $renderExitCode = $LASTEXITCODE
            } finally {
                $ErrorActionPreference = $previousErrorPreference
            }
            if ($renderExitCode -ne 0) {
                throw "pdftoppm failed for $pdf`n$renderOutput"
            }
            $rendered = @(Get-ChildItem -LiteralPath $imageDir -File -Filter 'render-*.png' | Sort-Object {
                if ($_.BaseName -match 'render-(\d+)$') { [int]$Matches[1] } else { 0 }
            })
            for ($index = 0; $index -lt $rendered.Count; $index++) {
                $stableName = 'page_{0:D3}.png' -f ($index + 1)
                Move-Item -LiteralPath $rendered[$index].FullName -Destination (Join-Path $imageDir $stableName) -Force
            }
        }

        $pages = @(Get-ChildItem -LiteralPath $imageDir -File -Filter 'page_*.png' | Sort-Object Name)
        if ($pages.Count -ne $expectedPages) {
            throw "Rendered page count mismatch for ${pdf}: expected $expectedPages, found $($pages.Count)"
        }

        for ($pageIndex = 0; $pageIndex -lt $pages.Count; $pageIndex++) {
            if ($pages[$pageIndex].Length -lt 10KB) {
                throw "Rendered image looks too small: $($pages[$pageIndex].FullName)"
            }
            $record = [ordered]@{
                sample_id = ('{0}_{1}_p{2:D3}' -f $source.id, $style, ($pageIndex + 1))
                source_id = $source.id
                split = $source.split
                layout = $style
                page_index = $pageIndex + 1
                page_count = $pages.Count
                dpi = $Dpi
                source_format = $source.source_format
                source_gp = $source.source_gp
                label_json = $source.label_json
                pdf = Get-RelativePath $pdf
                image = Get-RelativePath $pages[$pageIndex].FullName
                target_track_number = $source.target_track_number
                measure_count = $source.measure_count
                note_count = $source.note_count
            }
            $sampleLines.Add(($record | ConvertTo-Json -Compress))
        }
    }
}

[IO.File]::WriteAllLines($SamplesManifest, $sampleLines, $utf8)

Write-Host '[4/4] Writing splits and dataset summary...'
foreach ($split in @('train', 'validation', 'test')) {
    $ids = @($sources | Where-Object split -eq $split | ForEach-Object id | Sort-Object)
    [IO.File]::WriteAllLines((Join-Path $DatabaseRoot "splits\$split.txt"), $ids, $utf8)
}

$summary = [ordered]@{
    schema_version = '1.0'
    generated_at = [DateTimeOffset]::Now.ToString('o')
    source_song_count = $sources.Count
    pdf_count = $sources.Count * $styles.Count
    image_page_count = $sampleLines.Count
    dpi = $Dpi
    layouts = [ordered]@{}
    source_formats = [ordered]@{}
    splits = [ordered]@{}
}
foreach ($style in $styles) {
    $summary.layouts[$style] = @($sampleLines | Where-Object { $_ -match ('"layout":"' + $style + '"') }).Count
}
foreach ($format in @('gp3', 'gp4', 'gp5', 'gtp')) {
    $summary.source_formats[$format] = @($sources | Where-Object source_format -eq $format).Count
}
foreach ($split in @('train', 'validation', 'test')) {
    $summary.splits[$split] = [ordered]@{
        sources = @($sources | Where-Object split -eq $split).Count
        pages = @($sampleLines | Where-Object { $_ -match ('"split":"' + $split + '"') }).Count
    }
}
$summaryJson = $summary | ConvertTo-Json -Depth 6
[IO.File]::WriteAllText((Join-Path $DatabaseRoot 'manifests\dataset_summary.json'), $summaryJson + "`n", $utf8)

Write-Host ("Database complete: {0} sources, {1} PDFs, {2} PNG pages" -f $summary.source_song_count, $summary.pdf_count, $summary.image_page_count)
