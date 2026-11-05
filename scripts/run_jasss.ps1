# Start, or continue, the JASSS Phase 2 batch (scripts/jasss_runs.py).
#
# Safe to run again after any interruption (closed window, crash, reboot, power cut):
# finished models are skipped and a model stopped mid-training continues from its
# last resume point (every 25k steps). Progress is appended to results/jasss/run.log.
#
#   powershell -ExecutionPolicy Bypass -File scripts\run_jasss.ps1                       # everything, 4 workers
#   powershell -ExecutionPolicy Bypass -File scripts\run_jasss.ps1 -Groups A13 -Workers 3
#   powershell -ExecutionPolicy Bypass -File scripts\run_jasss.ps1 -Groups A14,A17
#
# Splitting the batch between machines: give each machine different -Groups (see
# JASSS_TEAM_RUN.md). A machine running only some groups skips the rule grids and the
# baselines (they are run once, at the end, on the machine that collects the results).
#
# Before starting: plug in, close Chrome and VS Code (16 GB RAM: 4 workers at most),
# and set Windows to never sleep while plugged in.
param([int]$Workers = 4, [string[]]$Groups = @())

$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo
$env:PYTHONPATH = "."
$py = ".venv-win\Scripts\python.exe"
if (-not (Test-Path $py)) { $py = "python" }
$jobArgs = @("--workers", $Workers, "--log", "results\jasss\run.log")
if ($Groups.Count -gt 0) { $jobArgs += @("--groups") + $Groups + @("--stages", "refs", "train") }
# Python writes the log itself (progress, and the traceback if it crashes); piping a
# native program through 2>&1 / Tee-Object in Windows PowerShell 5.1 loses its output.
& $py -u scripts\jasss_runs.py @jobArgs
"batch exited with code $LASTEXITCODE at $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" |
    Out-File -Append -Encoding ascii results\jasss\run.log
