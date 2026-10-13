# Optimising a Self-Contaminated Proxy

**Patrol-allocation bias in a crime simulator with known ground truth.** Capstone research, PES University.

Predictive policing sends patrols to where crime was *recorded*, but patrolling a district raises the share of its crime that gets recorded. Recorded crime is then both the input and the score of the next decision, so the system partly writes its own scorecard. We call this a *self-contaminated proxy* and study what happens when it is optimised.

The simulator is an agent-based model of 22 Chicago police districts built from census data. No historical crime is fed in, so the true (latent) crime in every district is known, and we can measure exactly how far a patrol policy drifts from where crime actually is.

Start with `Understanding update.md` for a plain-language guide. The paper is in `paper/`. `HANDOFF.md`, `EXPLAINER.md` and `JOURNAL_READINESS_AUDIT.md` describe the project before the October 2026 pivot and are kept for history; `PIVOT_AUDIT.md` explains why the earlier numbers were replaced.

## Main results

All disparities are relative to a need-proportional reference (the "oracle", which patrols in proportion to true crime), on the same simulator seeds. 1× is fair.

| Finding | Result |
|---|---|
| RL agent (PPO) rewarded on recorded crime | 3.9× the oracle's disparity against the poorest districts (95% CI 3.0–5.1) |
| True-crime coverage, PPO vs oracle | 4.77% vs 5.37%: unfair and not more effective |
| Best fixed rule from an exhaustive search, no learning, same reward | 13.0× |
| Reward vs disparity across 20 trained models | Rank correlation 0.96: better optimisation means more bias |
| Patrol-dependent detection switched off | PPO's excess falls from 4.9× to 1.5× |
| Mean-field model of the loop vs the simulator | Within 2% on average |
| Seven changed parameter settings | PPO stays 3.2× to 9.1× its own oracle |
| Parity penalty | Overcorrects and under-patrols the districts with the most crime |
| Oracle-targeted penalty | 0.76× the oracle, with higher true-crime coverage (4.90%) |

The bias comes from the objective, not from reinforcement learning. These are results inside a simulator with assumed detection and deterrence parameters; they are not claims about real-world policing.

## Setup

Python 3.11 or 3.12; CPU is enough.

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python scripts/build_ca_lookup.py --grid 20     # one-time: builds data/region_*.npz
```

Run everything from the repo root with `PYTHONPATH=.`, for example:

```bash
PYTHONPATH=. python -m pytest tests -q               # 48 tests
PYTHONPATH=. python -m crimesim.abm.calibration      # simulator fidelity gate, 5 checks
```

## Layout

| Path | What |
|---|---|
| `crimesim/abm/` | Vectorised NumPy crime generator; `calibration.py` is the fidelity gate; `regions.py` holds the 22 districts and their census covariates |
| `crimesim/env/` | `PolicingEnv` (Gymnasium), the detection and deterrence model, fairness metrics |
| `crimesim/rl/` | PPO (`ppo.py`), penalised and Lagrangian PPO (`constrained_ppo.py`), rule-based baselines |
| `crimesim/eval/` | Rollouts, oracle-relative disparity with hierarchical bootstrap (`oracle_relative.py`), statistics, E2 and E3 metrics |
| `crimesim/theory.py` | Mean-field model of the loop: fixed point, signal elasticity, amplification |
| `configs/` | `base.yaml` plus overlays |
| `scripts/` | Training queues, summaries, and the paper asset generators (see below) |
| `results/final/`, `results/journal/`, `results/theory/` | The result tables the paper is built from |
| `figures/` | Generated figures |
| `paper/` | `body.tex` and `abstract.tex` (the text), `main.tex` (Springer wrapper), `main_ieee.tex` (IEEE wrapper), generated `numbers*.tex` and `tables/` |
| `tests/` | Unit tests |
| `legacy/` | The original Mesa model and ML pipeline, kept for provenance |

Trained models (`runs/`) and cached rollouts (`*.pkl`) are not in the repository; the scripts below recreate them.

## Reproduce

Training is slow on a laptop: about 45 minutes per 400k-step model, roughly 90 models in total. Every training script is resumable; rerun the same command after an interruption and finished models are skipped. On a 16 GB machine use at most 4 workers.

```bash
# 1. Fixed-rule search (no RL)
PYTHONPATH=. python scripts/rule_search.py

# 2. Main experiments E1-E5: train and evaluate (about 51 models)
PYTHONPATH=. python scripts/overnight.py --workers 4

# 3. Final tables, then figures, tables and numbers for the paper
PYTHONPATH=. python scripts/summarise_400k.py
PYTHONPATH=. python scripts/make_paper_assets.py

# 4. Journal additions: longer training, hyperparameters, sensitivity, Lagrangian (39 models)
PYTHONPATH=. python scripts/journal_runs.py --workers 4
PYTHONPATH=. python scripts/analytic_model.py           # mean-field model vs simulator
PYTHONPATH=. python scripts/reference_robustness.py     # other reference policies
PYTHONPATH=. python scripts/summarise_journal.py
PYTHONPATH=. python scripts/make_journal_assets.py

# 5. Check the paper sources (every number, table, figure, label and citation resolves)
python scripts/check_paper.py
```

Evaluation protocol: 400,000 training steps, 5 training seeds, 30 held-out evaluation seeds (200–229). E5 and the sensitivity runs use 3 training seeds and evaluation seeds 300–319. Confidence intervals come from a hierarchical bootstrap; pairwise comparisons are Holm-corrected.

## Paper

Edit the text only in `paper/body.tex` and `paper/abstract.tex`. Never type a result into the paper by hand: every number is a macro in `paper/numbers.tex` or `paper/numbers_journal.tex`, and every table is generated into `paper/tables/`, so rerun the asset scripts instead.

Compile with any LaTeX distribution, or with [Tectonic](https://tectonic-typesetting.github.io):

```bash
mkdir -p build
tectonic -X compile paper/main.tex --outdir build        # Springer (sn-jnl) version
tectonic -X compile paper/main_ieee.tex --outdir build   # IEEE version
```

## Limitations

- Simulation only; nothing is validated against real policing data.
- Detection and deterrence parameters are assumed, not calibrated. The sensitivity analysis varies them but does not replace calibration.
- 22 districts, so each hardship quintile holds only 4–5 districts; residential population is uniform.
- Policies act through a three-parameter allocation rule, not a free allocation.
- Default PPO does not reach the optimum of its objective, and results vary across training seeds (2.6× to 5.3×).
