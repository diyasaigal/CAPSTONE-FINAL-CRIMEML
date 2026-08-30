# CrimeSim-RL — Runaway feedback & fairness in reinforcement-learned patrol allocation

Capstone research (PES University). We study the predictive-policing feedback loop inside an
**unseeded, theory-grounded** agent-based crime simulator whose latent (ground-truth) crime
process is known, so we can measure exactly how far a learned patrol policy's enforcement drifts
from where crime actually is — and show a fairness-constrained RL objective fixes it.

Read `HANDOFF.md` first, then `EXPLAINER.md` for the deep dive. The paper draft is
`paper/main.tex` (IEEE format, for Overleaf).

## Setup

```bash
conda create -n crimeml python=3.12 -y
conda run -n crimeml pip install -r requirements.txt
python scripts/build_ca_lookup.py --grid 20          # one-time: builds data/region_*.npz
```

Run everything through the env's interpreter with `PYTHONPATH=.`, e.g.
`PYTHONPATH=. python -m pytest tests/ -q`.

## Layout

| Path | What |
|---|---|
| `crimesim/abm/` | vectorised NumPy crime generator (ports `legacy/crime_model_v5.py`); `calibration.py` is the fidelity gate |
| `crimesim/abm/regions.py` | 22 Chicago police districts = allocation + fairness unit, with ACS covariates |
| `crimesim/env/` | `PolicingEnv` (Gymnasium), detection/observation model, fairness metrics |
| `crimesim/rl/` | PPO (`ppo.py`), Lagrangian constrained PPO (`constrained_ppo.py`), rule-based baselines |
| `crimesim/eval/` | rollout harness, feedback metrics (E2), incidence regression (E3), Pareto (E4), figures |
| `configs/` | `base.yaml` + `_extends` overlays |
| `scripts/run_all.py` | full pipeline → `results/` + `figures/` (resumable) |
| `scripts/run_ablations.py` | E5 robustness sweeps |
| `legacy/` | the original Mesa ABM + ML pipeline, kept for provenance |

## Reproduce

```bash
# fidelity gate (must pass before trusting anything downstream)
PYTHONPATH=. python -m crimesim.abm.calibration

# fast end-to-end smoke (train short PPO + Lagrangian, compare to baselines)
PYTHONPATH=. python scripts/quicktest.py --timesteps 150000

# full paper results (hours; resumable — safe to re-run)
PYTHONPATH=. python scripts/run_all.py \
    --train-seeds 0 1 2 3 --eval-seeds 200 201 202 203 204 205 206 207 208 209 210 211 \
    --alphas 0.3 0.7 1.5 3.0 --timesteps 220000

# robustness / falsification (incl. perfect-observation control that should kill the loop)
PYTHONPATH=. python scripts/run_ablations.py
```

## The core idea in one paragraph

Patrol has two effects in the sim: it **deters** a bounded fraction of latent crime, and it
**raises the chance** any latent incident is recorded. The RL agent is rewarded for the
operational objective of predictive policing — the share of *recorded* crime that happens under
patrol. Optimising that, it concentrates patrol on the districts where crime is *found*, which
(via the detection effect) makes those districts look even worse, tightening the loop. Because the
sim is unseeded, the latent crime map is a clean function of socioeconomic geography, so we can
show the drift lands on the most disadvantaged districts beyond what their true crime justifies —
and that a Lagrangian parity constraint on enforcement-per-capita removes it cheaply.
