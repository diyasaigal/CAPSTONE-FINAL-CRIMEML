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
New-Item -ItemType Directory -Force "results\jasss" | Out-Null
$log = "results\jasss\run.log"
"===== start $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), commit $(git rev-parse --short HEAD), $Workers workers =====" |
    Out-File -Append -Encoding utf8 $log
& ".venv-win\Scripts\python.exe" -u scripts\jasss_runs.py --workers $Workers 2>&1 |
    ForEach-Object { "$_" } | Tee-Object -FilePath $log -Append
