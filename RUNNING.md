# How to open and run this project (step by step)

Written for Windows + VS Code + Miniconda (what Dennis has). If a teammate uses macOS/Linux the
only differences are the shell and the interpreter path.

---

## 0. One-time: get the code and data

1. Copy the whole `CrimeML` folder to your machine. It already contains:
   - `data/processed_crime.csv` (~125 MB) and `data/processed_census.csv`
   - `legacy/` (the old Mesa ABM, for reference)
   - all the new code
2. If `data/processed_crime.csv` is missing (it is git-ignored because of size), get it from
   Dennis or re-run the Phase-2 `legacy/preprocess.py` against the raw Chicago crime export.

---

## 1. Open the folder in VS Code

1. Open **VS Code**.
2. `File -> Open Folder...` and pick the `CrimeML` folder (the one that contains `crimesim/`,
   `scripts/`, `tests/`).
3. Install the **Python** extension if you do not have it (Extensions panel, search "Python",
   by Microsoft).

---

## 2. Create the Python environment (one time)

We use **conda** (Miniconda / Anaconda). Open a terminal in VS Code:
`Terminal -> New Terminal`. Then:

```bash
conda create -n crimeml python=3.12 -y
```

This makes an isolated environment called `crimeml` with Python 3.12 (some RL libraries lag on
3.13, so 3.12 is deliberate).

Now install the packages. From the `CrimeML` folder:

```bash
conda run -n crimeml pip install -r requirements.txt
```

That pulls NumPy, pandas, scikit-learn, matplotlib, Gymnasium, Stable-Baselines3, PyTorch (CPU),
pytest. Takes a few minutes. **No GPU is needed** — everything runs on CPU.

---

## 3. Point VS Code at the environment

1. Press `Ctrl+Shift+P` -> type **"Python: Select Interpreter"** -> Enter.
2. Choose the one that says **`crimeml`** (path looks like
   `C:\Users\<you>\miniconda3\envs\crimeml\python.exe`).
3. Close and reopen the VS Code terminal so it activates the env. The prompt should now start
   with `(crimeml)`.

If it does not auto-activate, run:

```bash
conda activate crimeml
```

---

## 4. One-time data prep

From the `CrimeML` folder, in the `(crimeml)` terminal:

```bash
python scripts/build_ca_lookup.py --grid 20
```

This reads the crime CSV and writes `data/region_district_g20.npz` (the map from grid cells to
Chicago police districts, plus per-district real-crime counts). It prints a short summary — you
should see "22 districts" and the top-5 high-crime districts.

> If you get `ModuleNotFoundError: crimesim`, you forgot to run from the `CrimeML` folder, or
> `PYTHONPATH` is not set. Fix: `set PYTHONPATH=.` (Windows cmd) / `$env:PYTHONPATH="."`
> (PowerShell) / `export PYTHONPATH=.` (bash), then re-run. Or run modules with `python -m`.

---

## 5. Sanity checks (run these first, every time you pull new code)

```bash
python -m pytest tests/ -q
```

Expect: `20 passed`.

```bash
python -m crimesim.abm.calibration
```

Expect: a table and **five `[PASS]` lines**. This is the fidelity gate — it confirms the fast
simulator still reproduces the Phase-2 model's behaviour (hardship raises crime ~27%, night crime
> day crime, spatial clusters emerge from a flat start). If any line says `[FAIL]`, stop and
tell the group — nothing downstream is trustworthy.

---

## 6. See the result quickly (about 15 minutes)

```bash
python scripts/quicktest.py --timesteps 150000
```

This trains a short **naive PPO** rule and a short **fairness-penalised PPO** rule, then
rolls out every policy (baselines + both RL policies) on held-out simulator seeds and prints two
tables:

- **efficiency & fairness** — look at the `ppo` row: `rec_cov` (recorded-crime coverage) is the
  highest of any policy (looks great) but `lat_cov` (real coverage) is near the bottom, and
  `disparity` is large (~15). The `pen_p*` rows: `disparity` drops toward 1 as the penalty grows.
- **E3 incidence** — the `ppo` hardship coefficient is strongly positive with a tiny p-value;
  `ppo_fair` is near zero.

That is the whole paper in one screen.

---

## 7. Full paper results (a few hours, safe to interrupt)

```bash
python scripts/run_all.py --train-seeds 0 1 2 --eval-seeds 200 201 202 203 204 205 206 207 208 209 --penalties 0.02 0.06 0.15 0.4
```

- Trains 18 policies (3 seeds × {PPO, PPO-oracle, 4 penalty weights}). Each trained model is
  cached in `runs/<name>/` — if you stop and re-run, it **skips what is already done**.
- Then evaluates everything on 10 held-out seeds and writes:
  - `results/e1_summary.csv`, `results/e2_feedback.csv`, `results/e3_incidence.csv`,
    `results/e4_pareto.csv`
  - `figures/fig_efficiency_equity.pdf`, `fig_feedback_trajectories.pdf`, `fig_incidence.pdf`,
    `fig_pareto.pdf`
- It prints the E1–E4 tables to the terminal at the end.

To run it unattended, redirect output to a log:

```bash
python scripts/run_all.py --train-seeds 0 1 2 --penalties 0.02 0.06 0.15 0.4 > run.log 2>&1
```

and watch it with `Get-Content run.log -Wait` (PowerShell) or `tail -f run.log` (bash).

---

## 8. Robustness sweeps (E5)

```bash
python scripts/run_ablations.py
```

Trains a small PPO + fair-PPO for each variant (perfect observation, weak/strong deterrence,
low/high detection gain, no socioeconomic features, small/large budget, short/long history),
evaluates them, and writes `results/e5_ablations.csv`. The key row is **`perfect_obs`**: with
`p_report = 1` the detection loop is switched off, and the disparity + hardship coefficient
should collapse toward zero — that is our falsification check.

---

## 9. Where to look at things

| You want to... | Open |
|---|---|
| understand the whole project | `HANDOFF.md`, then `EXPLAINER.md` |
| change a simulator parameter | `crimesim/config.py` (defaults) or `configs/base.yaml` (the RL runs' overlay) |
| change what the RL agent optimises | `EnvConfig.reward_on` in `crimesim/config.py` |
| change the allocation-rule action | `PolicingEnv.action_to_deployment` in `crimesim/env/policing_env.py` |
| change the fairness penalty | `FairnessConfig.penalty` + `crimesim/env/fairness.py` |
| read the model logic | `crimesim/abm/dynamics.py` (functions) + `generator.py` (the loop) |
| read the paper | `paper/main.tex` (LaTeX for Overleaf) |

---

## 10. Common problems

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError: crimesim` | Run from the `CrimeML` folder; set `PYTHONPATH=.` or use `python -m` |
| `conda run ... python -c "..."` errors about newlines | Windows conda bug — call `...\envs\crimeml\python.exe` directly instead |
| calibration `[FAIL]` | Someone changed an ABM constant. `git diff crimesim/config.py`. Do not proceed. |
| training is very slow / uses one core | Expected — the simulator is single-threaded NumPy at ~300–400 steps/s. Use `--timesteps` smaller while iterating; run the full thing overnight. |
| `pyarrow` / parquet error | We switched to `.pkl`; pull the latest code. |
