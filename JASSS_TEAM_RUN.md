# Running part of the JASSS batch on your laptop

The 380 models are split by group. Each laptop runs **different groups**, never the same
ones, because each model comes out the same whoever trains it.

| Laptop | Groups | Models | Rough time (4 workers) |
|---|---|---|---|
| Diya | A10, A11 | 130 (36 already done) | ~30 h |
| Teammate 1 | A13 | 80 | ~25 h |
| Teammate 2 | A14, A17 | 90 | ~28 h |
| Teammate 3 | A6A7 | 80 | ~35–45 h (1000-agent models are slow) |

## 1. Setup (once, ~15 min)

1. Install **Python 3.11** (python.org). On Windows, tick "Add to PATH".
2. Get the code:
   ```
   git clone -b jasss https://github.com/diyasaigal/CAPSTONE-FINAL-CRIMEML.git
   cd CAPSTONE-FINAL-CRIMEML
   ```
3. Create the environment.
   - Windows (PowerShell):
     ```
     python -m venv .venv-win
     .venv-win\Scripts\python.exe -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
     .venv-win\Scripts\python.exe -m pip install -r requirements-jasss.txt
     ```
   - Mac/Linux:
     ```
     python3.11 -m venv .venv && source .venv/bin/activate
     pip install -r requirements-jasss.txt
     ```
4. Check it works. Both commands must succeed:
   ```
   $env:PYTHONPATH="."; .venv-win\Scripts\python.exe -m pytest tests -q
   $env:PYTHONPATH="."; .venv-win\Scripts\python.exe scripts\jasss_runs.py --dry-run --groups A13
   ```
   (Mac/Linux: `PYTHONPATH=. python -m pytest tests -q`, and so on.)

## 2. Laptop settings

- Keep it **plugged in** the whole time. On battery, Windows puts it to sleep and the run stops.
- Turn off sleep while plugged in:
  ```
  powercfg /change standby-timeout-ac 0
  powercfg /change hibernate-timeout-ac 0
  ```
- Settings → System → Power → Lid: "When I close the lid (plugged in)" → **Do nothing**.
- Close Chrome and other heavy apps. With 8 GB RAM, use `-Workers 2`.

## 3. Run your groups

Windows (example for Teammate 1):
```
powershell -ExecutionPolicy Bypass -File scripts\run_jasss.ps1 -Groups A13
```
Teammate 2: `-Groups A14,A17`. Teammate 3: `-Groups A6A7`.

Mac/Linux:
```
PYTHONPATH=. python -u scripts/jasss_runs.py --groups A13 --stages refs train --workers 4 --log results/jasss/run.log
```

Leave the window open. If it stops (crash, restart, power cut), **run the same command again**.
It skips finished models and continues the interrupted ones from their last save.

Progress is in `results/jasss/run.log`. Each finished model adds one line, `N/80 ... trained`.

## 4. When it says `done; failed: none`

Zip and send these two things to Diya (Google Drive; about 5 MB per model):
- the folder `runs/jasss/`
- the file `results/jasss/run.log` (rename it to `run_<yourname>.log`)

If the last line says `failed: [...]` or `CRASHED`, send the log anyway.
