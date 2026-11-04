param(
    [Parameter(Mandatory = $true)]
    [ValidatePattern('^[a-zA-Z0-9_-]+$')]
    [string]$Label
)
$ErrorActionPreference = 'Stop'
Set-Location (Join-Path $PSScriptRoot '..')
$outputDirectory = Join-Path (Get-Location) "docs/performance/$Label"
New-Item -ItemType Directory -Force -Path $outputDirectory | Out-Null
$checks = @(
    @{ Name = 'check'; Arguments = @('check') },
    @{ Name = 'test'; Arguments = @('test') },
    @{ Name = 'clippy'; Arguments = @('clippy', '--all-targets', '--', '-D', 'warnings') },
    @{ Name = 'fmt'; Arguments = @('fmt', '--check') },
    @{ Name = 'release'; Arguments = @('build', '--release') }
)
foreach ($check in $checks) {
    $arguments = $check.Arguments
    & cargo @arguments *> (Join-Path $outputDirectory "$($check.Name).txt")
    if ($LASTEXITCODE -ne 0) {
        Get-Content (Join-Path $outputDirectory "$($check.Name).txt")
        throw "Check failed: $($check.Name)"
    }
    Write-Output "$($check.Name): passed"
}
