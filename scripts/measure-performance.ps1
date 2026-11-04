param(
    [Parameter(Mandatory = $true)]
    [ValidatePattern('^[a-zA-Z0-9_-]+$')]
    [string]$Label
)
$ErrorActionPreference = 'Stop'
Set-Location (Join-Path $PSScriptRoot '..')
$outputDirectory = Join-Path (Get-Location) "docs/performance/$Label"
New-Item -ItemType Directory -Force -Path $outputDirectory | Out-Null
git rev-parse HEAD | Set-Content (Join-Path $outputDirectory 'revision.txt')
rustc -Vv | Set-Content (Join-Path $outputDirectory 'rustc.txt')
$runs = @(
    @{ Name = 'timing'; Feature = ''; Test = 'timing' },
    @{ Name = 'allocations'; Feature = 'profile-allocations'; Test = 'profile' },
    @{ Name = 'heap'; Feature = 'profile-heap'; Test = 'profile' }
)
foreach ($run in $runs) {
    $buildArguments = @('test', '--release', '--lib', '--no-run', '--message-format=json')
    if ($run.Feature) { $buildArguments += @('--features', $run.Feature) }
    $buildOutput = & cargo @buildArguments
    if ($LASTEXITCODE -ne 0) { throw "Build failed: $($run.Name)" }
    $artifacts = @($buildOutput | ForEach-Object { $_ | ConvertFrom-Json } |
        Where-Object { $_.reason -eq 'compiler-artifact' -and $_.executable -and $_.profile.test })
    if ($artifacts.Count -ne 1) { throw 'Expected one library test executable' }
    $testArguments = @("agent::performance::$($run.Test)", '--exact', '--ignored', '--nocapture', '--test-threads=1')
    $process = Start-Process -FilePath $artifacts[0].executable -ArgumentList $testArguments -WindowStyle Hidden -PassThru `
        -RedirectStandardOutput (Join-Path $outputDirectory "$($run.Name).txt") `
        -RedirectStandardError (Join-Path $outputDirectory "$($run.Name)-stderr.txt")
    $peakWorkingSet = 0L
    $peakPrivate = 0L
    while (-not $process.WaitForExit(20)) {
        $process.Refresh()
        if (-not $process.HasExited) {
            $peakWorkingSet = [Math]::Max($peakWorkingSet, $process.PeakWorkingSet64)
            $peakPrivate = [Math]::Max($peakPrivate, $process.PrivateMemorySize64)
        }
    }
    $process.WaitForExit()
    if ($process.ExitCode -ne 0) { throw "Measurement failed: $($run.Name)" }
    [pscustomobject]@{
        run = $run.Name
        observed_peak_working_set_bytes = $peakWorkingSet
        sampled_peak_private_bytes = $peakPrivate
        sampling_interval_ms = 20
    } | ConvertTo-Json | Set-Content (Join-Path $outputDirectory "$($run.Name)-rss.json")
    Get-Content (Join-Path $outputDirectory "$($run.Name).txt")
    if ($run.Name -eq 'timing') {
        $estimates = [ordered]@{}
        Get-ChildItem target/criterion -Directory | ForEach-Object {
            $path = Join-Path $_.FullName 'new/estimates.json'
            if (Test-Path -LiteralPath $path) {
                $estimates[$_.Name] = Get-Content -Raw -LiteralPath $path | ConvertFrom-Json
            }
        }
        $estimates | ConvertTo-Json -Depth 10 | Set-Content (Join-Path $outputDirectory 'estimates.json')
    }
}
