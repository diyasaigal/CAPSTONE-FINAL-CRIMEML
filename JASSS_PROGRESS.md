# JASSS revision: progress

Tracks the change list "JASSS Paper: Changes To Be Made" (items A1-A20, B1-B19, C1-C10). Work happens on the branch `jasss`. All numbers below are produced by the named scripts, and the outputs are in `results/phase0/`.

## Step 0: intake
- [ ] JASSS `.tex`, `.bib` and class from the teammate → `paper/jasss/`
- [ ] Scripts behind the real-record validation, calibration grid, one-at-a-time and Latin-hypercube sensitivity, concave detection curve, strong loop and rule boxes → `scripts/validation/`

## Phase 0: analysis on existing rollouts (done)

| Item | Script | Result |
|---|---|---|
| A1 Crime prevented | `scripts/phase0_prevented.py` | Each policy is paired with a no-patrol run on the same evaluation seed (200–229). PPO prevents **29.4%** of no-patrol crime [29.0, 29.9]. The latent-crime oracle prevents **32.2%** [31.9, 32.5]. PredPol prevents 34.0% and Uniform 25.9%. Paired PPO − oracle: **−2.8 points [−3.2, −2.3], p < 0.001**. The best fixed rule for true coverage prevents the most (36.2%). |
| Gate 0 | | **PPO prevents less crime than the reference.** Keep the cost claim, but state it as crime prevented rather than coverage share. |
| A2 Does PPO adapt? | `scripts/ppo_actions.py`, `figures/fig_ppo_actions.*` | PPO is close to **bang-bang**. β sits at a bound in 52–68% of shifts and w in 59–71%. The exact corner (β, w, μ) = (4, 2, 0), which is the best fixed rule for recorded coverage, is played in 18–44% of shifts. Actions cycle with the hour of day. They are only weakly tied to how concentrated the recorded-crime EWMA is (|ρ| ≤ 0.35). PPO's mean action run as a fixed rule earns slightly less reward (−0.001 to −0.003) and has about the same disparity (2.6–6.4×, against PPO's 2.6–5.3×). **Conclusion:** PPO adapts mainly to time of day, and its disparity is that of its average rule. Present it as one optimiser of the proxy (B11). |
| A3 Provenance | `scripts/provenance.py` → `provenance.csv`, `provenance_families.csv` | Naive PPO by training seed: 5.27, 4.06, 5.31, 3.07, 2.60. The **3.91×** figure is seeds 0–4; **4.85×** is seeds 0–2 on the same evaluation seeds; **4.92×** is seeds 0–2 on evaluation seeds 300–319. Seeds 3–4 were trained two days earlier, but the simulator and PPO code are identical between the two batches (git diff 0a433ce..187491b), so this is seed variance. 10 seeds will settle it. |
| A4 Mean-field error | `scripts/meanfield_error.py` → `meanfield_error.json` | There are 24 pairs (3 settings × 8 values of β). In the model, constant detection and perfect observation coincide, so there are only 2 distinct curves. **Raw disparity:** mean error 1.6%, max 7.4%, R² of log = 0.998, so the draft's "within 2%, at most 7%" holds for raw disparity. **× own oracle** (the Table 4 scale): mean 8.2%, max 13.4%, because the model's oracle (3.07) is 10.7% more concentrated than the simulator's (2.77). The draft's 0.98 is the R² on log × oracle. *Still to do:* the concave-curve rows, which need the teammate's script. |
| A5 Night/day | `crimesim/abm/calibration.py` (`night_day_ratio`) | There is now one hourly definition (night = hours 20–02, the model's own window). The old gate classified a whole shift by the hour it ended. **Original-model settings: 1.20** (real data: 1.18). **Experiment settings (`base.yaml`): 1.01.** The day–night cycle is washed out by the two port additions (stress recovery 0.18, turnover 0.05). The gate now takes `--config configs/base.yaml`; it previously ran only on code defaults, so it never checked the experiment configuration. Both configurations pass all 5 checks, but the night check passes at 1.01 only because its threshold is "> 1". |

Code changes (tests: 48 → 52, all pass):
- `NoPatrolPolicy` in `crimesim/rl/baselines.py`.
- Rollouts log (β, w, μ), the EWMA dispersion and the hour (`crimesim/eval/rollout.py`, `PolicingEnv.rule_params`).
- The generator returns hourly sub-step counts.
- The calibration gate uses hourly night/day and takes a config.

## Phase 1: generator (done; waiting for the freeze commit and tag)

The frozen base is **`configs/jasss_base.yaml`**: it extends `base.yaml` and adds the switches below. The variants are in `crimesim/experiments/jasss.py` (`jasss_config(variant)`). The measurements come from `scripts/phase1_generator.py` and are in `results/phase1/generator_sweep.csv`: proportional patrol, 100-shift warm-up, 140 shifts, 6–8 seeds.

| Decision | What was done | Measured |
|---|---|---|
| A6 Hardship counted once | `hardship_in_risk: false`. Hardship now enters baseline stress only. `hardship_scale` sets the **spread** of hardship around its population-weighted mean. A plain multiplier changed crime volume from 55 to 475 per shift and *lowered* the contrast. | Latent crime per resident, top vs bottom hardship quintile: **base 3.1×** (spread 1.0, the real data). **Low 1.8×** (spread 0.2; 1.8× remains even with no hardship at all, because unemployment and geography alone produce it). **High 4.3×** needs hardship in the risk term as well (the pre-revision world, variant `contrast_high`); hardship in stress alone tops out at about 3.5×. |
| A5 Night/day | Weakening stress recovery does **not** restore the cycle: night/day stays 1.01–1.03 for every recovery value from 0.18 to 0. The cause is that once hardship leaves the risk term, the escalation path dominates, and v5's night factor does not act on it. Neither trigger restores it either (≤ 1.02). Chosen: **`night_interaction: true`**, the night factor on both paths (routine activity applies to both). | **1.17** (real data 1.18). The gate now requires ≥ 1.10. `base.yaml` gives 1.01 and now fails the check, as it should. |
| A8 Population | ACS 2023 5-year community-area totals (City of Chicago, t68z-cikk) → `data/ca_population_acs2023.csv`. Each area's population is split across districts in proportion to where its crime records fall. `population_weighted: true` drives agent homes, the patrol reference and exposure per capita. | Districts range from 52k to 247k residents (4.7×). Real recorded violent crime, top vs bottom hardship quintile: **1.44× by count, 2.58× per resident**. |
| A9 Drift | `warmup_shifts: 100` | After warm-up, drift is between −3% and +3% per 100 shifts (seed SD 3–5%). |
| A7 Agents | 500 and 1,000 agents, **no re-tuning**. Holding total crime fixed by raising the escalation trigger pushed the contrast to 9–16× and night/day to 1.6–2.0, a different world rather than a less noisy one. So crime grows with the number of agents, which is the clean count-noise test. | Crime per shift 384 / 1,025 / 1,940. Contrast 3.1 / 3.8 / 4.6×. District count CV 0.44 / 0.38 / 0.36. |
| A15 Displacement | Skipped. Stated as a limitation; the hot-spots literature often finds diffusion of benefits instead. | |

**Calibration gate changes** (`crimesim/abm/calibration.py`):
- The night check is now ≥ 1.10, per hour.
- The hardship-elasticity floor is lowered from 20% to **10%**. The old window was anchored on v5's +42.6%, which includes the double count. With hardship counted once, a +0.2 rise in citywide hardship raises crime by **18.7%** (16 seeds). That value does not move with the spread (18.5% at 1.2, 14.7% at 1.5). **The paper must report this change and the 18.7%.**
- `configs/jasss_base.yaml` passes all 5 checks (16 seeds). Run it with: `python -m crimesim.abm.calibration --config configs/jasss_base.yaml`.

**Tests:** 55 pass.

**Note for Phase 2 scripts:** exposure per capita now depends on population. Every script that scores rollouts must load the region with `region_for(cfg)`, not `load_region_info(20)`. The same applies to the mean-field model's λ estimate (`crimesim/theory.py`).

**To freeze (Gate 1):** commit on `jasss` and `git tag jasss-frozen-v1`. Every pre-revision model in `runs/` stays valid for `base.yaml` but is not used in the revision.

## Gate 1 decisions (taken 2026-11-03)
1. **Night/day.** Tighten the gate to, say, ≥ 1.10 and fix the generator so the experiment configuration keeps a day–night cycle (for example, apply the night factor on the interaction path too, or weaken stress recovery). The alternative is to keep the generator and state that the experiment configuration has no day–night contrast.
2. **Hardship double count (A6).** Choose which term to remove: in baseline stress (`generator.py:62,151`) or in the risk term (`dynamics.crime_events`). Also choose the contrast settings (about 1.8, about 3, and 4.4).
3. **Agent count (A7)**, population weighting (A8) and the drift fix (A9). A8 needs ACS district population data, because the census file has no population column.
4. **Displacement (A15).** The recommendation is to skip it.
5. Any change to the base generator invalidates every trained model, so all models are retrained at 10 seeds in Phase 2 (about 380 models, about 90–100 h).

## Later phases
See `C:\Users\diyas\.claude\plans\cuddly-churning-sphinx.md` (Phase 1 generator switches, Phase 2 `scripts/jasss_runs.py`, Phase 4 paper edits, Phase 5 archive).
