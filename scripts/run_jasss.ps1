# Start, or continue, the JASSS Phase 2 batch (scripts/jasss_runs.py).
#
# Safe to run again after any interruption (closed window, crash, reboot, power cut):
# finished models are skipped and a model stopped mid-training continues from its
# last resume point (every 25k steps). Progress is appended to results/jasss/run.log.
#
#   powershell -ExecutionPolicy Bypass -File scripts\run_jasss.ps1            # 4 workers
#   powershell -ExecutionPolicy Bypass -File scripts\run_jasss.ps1 -Workers 3
#
# Before starting: plug in, close Chrome and VS Code (16 GB RAM: 4 workers at most),
# and set Windows to never sleep while plugged in.
param([int]$Workers = 4)

$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo
$env:PYTHONPATH = "."
# Python writes the log itself (progress, and the traceback if it crashes); piping a
# native program through 2>&1 / Tee-Object in Windows PowerShell 5.1 loses its output.
& ".venv-win\Scripts\python.exe" -u scripts\jasss_runs.py --workers $Workers --log results\jasss\run.log
"batch exited with code $LASTEXITCODE at $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" |
    Out-File -Append -Encoding ascii results\jasss\run.log
