[CmdletBinding()]
param(
    [switch]$InstallDissertationFigures
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$sourceDir = $PSScriptRoot
$buildDir = Join-Path $sourceDir 'build'
$repoRoot = Split-Path -Parent (Split-Path -Parent $sourceDir)
$figureDir = Join-Path $repoRoot 'dissertation\figures'

$outputs = [ordered]@{
    'pci'  = 'figure_3_2a_pci_optical_readout.pdf'
    'dgi'  = 'figure_3_2b_dgi_optical_readout.pdf'
    'dffi' = 'figure_3_3a_dffi_optical_readout.pdf'
    'dpfi' = 'figure_3_3b_dpfi_optical_readout.pdf'
}

foreach ($commandName in @('latex', 'dvips', 'ps2pdf')) {
    if (-not (Get-Command $commandName -ErrorAction SilentlyContinue)) {
        throw "Required command '$commandName' was not found on PATH."
    }
}

New-Item -ItemType Directory -Path $buildDir -Force | Out-Null

Push-Location -LiteralPath $sourceDir
try {
    foreach ($entry in $outputs.GetEnumerator()) {
        $stem = $entry.Key
        $pdfName = $entry.Value
        $dviPath = Join-Path $buildDir "$stem.dvi"
        $psPath = Join-Path $buildDir "$stem.ps"
        $pdfPath = Join-Path $buildDir $pdfName

        & latex -interaction=nonstopmode -halt-on-error -file-line-error "-output-directory=$buildDir" "$stem.tex"
        if ($LASTEXITCODE -ne 0) {
            throw "latex failed for $stem."
        }

        & dvips -Ppdf -G0 -o $psPath $dviPath
        if ($LASTEXITCODE -ne 0) {
            throw "dvips failed for $stem."
        }

        & ps2pdf $psPath $pdfPath
        if ($LASTEXITCODE -ne 0) {
            throw "ps2pdf failed for $stem."
        }

        Write-Output "Built $pdfPath"
    }
}
finally {
    Pop-Location
}

if ($InstallDissertationFigures) {
    if (-not (Test-Path -LiteralPath $figureDir -PathType Container)) {
        throw "Dissertation figure directory not found: $figureDir"
    }

    foreach ($pdfName in $outputs.Values) {
        $sourcePath = Join-Path $buildDir $pdfName
        $destinationPath = Join-Path $figureDir $pdfName
        Copy-Item -LiteralPath $sourcePath -Destination $destinationPath -Force
        Write-Output "Installed $destinationPath"
    }
}
else {
    Write-Output 'Build complete. Dissertation figures were not modified.'
    Write-Output 'Pass -InstallDissertationFigures to install all four PDFs explicitly.'
}
