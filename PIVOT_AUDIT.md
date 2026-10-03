# Pivot Audit — "RL Bias Under Self-Contaminated Observations"

Audit of the reframing handoff (the §53 deliverable) against the code and the data that already
exist. **No experiments were run, nothing was retrained, and nothing in `results/` or `runs/`
was modified.** Every number below comes either from code inspection or from a read-only
reanalysis of the existing trajectories (`results/all_shifts.pkl`, `runs/*/eval_e200_201n10.pkl`,
`results/baselines_e200_201n10.pkl`, `results/e5_ablations_raw.pkl`).

Verification run on this machine (Windows, `.venv-win`, Python 3.11, torch 2.14 CPU, SB3 2.9.0,
gymnasium 1.3.0, numpy 2.4.6, pandas 3.0.6):
- `pytest tests/`: **20 passed** (the handoff says 10).
- `python -m crimesim.abm.calibration`: **5/5 PASS** (SDT elasticity +27.2%, Low<Med<High,
  clustering 12.7, coverage 85%, night > day).

---

## 0. Findings that change the story (read these first)

| # | Finding | Evidence | Consequence |
|---|---|---|---|
| **F1** | **The published E1/E4 numbers for the RL policies come from training seed 2 alone, not from the 3-seed pool.** Before the uncommitted fix, `summarise()` grouped by `(policy, eval seed)` and took `disparity_cum`/`gini_cum` with `"last"`. In pooled data that row belongs to the *last* training seed. | Per-training-seed naive PPO disparity is **13.63 / 8.50 / 8.08**. The CSV reports 8.081, which is exactly seed 2. The same holds for pen_p0.02 (1.4756), ppo_oracle (2.7813), and every Gini value. The means (coverage) were pooled correctly. | The true pooled naive-PPO disparity is **≈10.1**, not 8.1. The uncommitted `rollout.py` change fixes this, but `e1_summary.csv`, `e4_pareto.csv` and `e4_knee.json` were **never regenerated**. Every disparity and Gini figure in the paper, HANDOFF.md and the friend's handoff is affected. |
| **F2** | **Training budgets are mixed.** `runs/*_s0..s2` were trained for **150k** steps; `runs/*_s3,s4` for **400k**. `ensure_trained` skips any run that already exists. | `runs/*/config.yaml` | Running `run_all.py --train-seeds 0 1 2 3 4` now would silently pool two different training budgets. Pick one budget and retrain the odd seeds (or drop them). |
| **F3** | **The "runaway index" does not measure feedback.** It is the partial correlation of `dep[t]` with `cum[t-1]`. Any persistent allocation scores ≈1. | A static hardship-tilted allocation with **zero** observation feedback scores **1.000**. The GreedyOracle, which has no feedback either because it reads latent crime, scores 0.82. | Drop it or replace it (see §B4). E2 cannot be called evidence of a feedback loop as it stands. |
| **F4** | **Most of the E3 "hardship gradient" is an artefact of a noisy control.** The control is the *single-shift* latent share, which is noisy. Hardship correlates 0.73 with true long-run need, so it absorbs need that the control fails to capture. | When the control is long-run expected need (latent share under uniform patrol), the coefficients become: naive PPO **+0.0166 → +0.0034** (t = 3.9, episode-clustered); PredPol **+0.0051 → −0.0010** (n.s.); GreedyOracle **+0.0032 → −0.0011** (n.s.). | The "3× PredPol" claim does not survive. A small positive PPO gradient remains. "p < 10⁻³⁰⁰" is pseudoreplication: with episode-clustered SEs the original spec gives t = 14 for PPO. |
| **F5** | **The fairness penalty overshoots the reference.** `r − w_f·δ` rewards pushing δ toward 0, not toward a target. | Penalised disparities are 1.48 / 0.91 / 0.69 / 0.47, i.e. **0.55× → 0.17× the oracle**. From w_f = 0.06 upward δ < 1, so high-hardship districts get *less* patrol per capita than low-hardship ones. The E3 coefficient goes to −0.026. | Under the new framing this is over-correction, not "removing excess". E4 needs a reference-targeted regulariser (§B5). |
| **F6** | **The core reframing claim holds up.** | Paired by eval seed, with a hierarchical bootstrap over training × eval seeds, disparity relative to GreedyOracle: **naive PPO 3.71× [2.84, 5.07]**, log-shift +1.24 [+1.02, +1.53]. PredPol 1.39×, RF 1.34×. **PPO with latent observation and latent reward: 1.06× [0.93, 1.23]**, consistent with the oracle. | Same learner, same action space: given the latent signal it matches the need-proportional reference, and given the contaminated signal it lands 3.7× above it. This is the paper's strongest result. |

---

## A. Current-state verification

| Handoff claim | Verdict | Where |
|---|---|---|
| Simulator is unseeded: risk grid starts flat at 0.1, no crime points injected | SUPPORTED | `abm/generator.py:73` |
| 20×20 grid, 22 districts, 200 agents, 60% local work | SUPPORTED | `config.py`, `regions.py`, `generator.py:67` |
| Latent and recorded crime are distinct; L is kept as ground truth | SUPPORTED | `generator.py:126-128`, `env/detection.py:28-32`, `policing_env.py:128-148` |
| Deterrence `L ~ Bin(L_raw, 1 − min(0.68, 0.26u))` | SUPPORTED | `generator.py:126`; applied per sub-step |
| Detection `q = 0.30 + 0.55·min(1, u/1.5)` | SUPPORTED, with one caveat | `detection.py:18-25`; q is clipped at **0.995**, so "perfect observation" is q = 0.995, not 1 |
| Observation is 199-dimensional and excludes latent crime for naive PPO | SUPPORTED | `policing_env.py:89-99`: 88 + 22 + 3 + 88 = 201 in code (the doc's "199" is off by 2), latent only if `observe_latent` |
| No latent leak through the reward or the penalty | SUPPORTED | reward uses `rec`; the penalty uses `disparity_shift`, which depends on deployment only; `rollout` passes only `obs` to `predict` |
| Action (β, w, μ) and the allocation formula | SUPPORTED | `policing_env.py:151-166`; uses the EWMA of recorded crime (half-life 5), floor 0.005 |
| Reward `10·Σ a·D / Σ D` | SUPPORTED | `policing_env.py:177-183` |
| Uniform is "equal per-capita" | SUPPORTED | `pop_weight` = 1/22 (`regions.py:93`). The docstrings in `fairness.py` and `baselines.py` that say "share of real crime" are stale. |
| GreedyOracle uses latent crime, the same floor, the same EWMA half-life | SUPPORTED | `baselines.py:115-131`; `_alloc` applies the same floor |
| GreedyOracle is an "upper bound" (paper l.286) | **NOT SUPPORTED** | It allocates ∝ EWMA(latent): need-proportional. Coverage is linear in the shares, so the coverage-maximising allocation is a corner solution. It is a reference, not a bound. |
| All policies evaluated on identical seeds | SUPPORTED | 200–209 for everything in `all_shifts.pkl` |
| E1 numbers in §11 of the handoff | **PARTIALLY SUPPORTED** | Coverage numbers are correct. Disparity and Gini are training-seed-2 only (F1). |
| E2 numbers | SUPPORTED as computed | Interpretation of the runaway index NOT SUPPORTED (F3) |
| E3 coefficients | SUPPORTED as computed | Inference and interpretation NOT SUPPORTED (F4) |
| E4: "every penalised variant has higher latent coverage than naive PPO" (paper l.396) | **NOT SUPPORTED** | All four penalised variants have *lower* latent coverage (0.0473 … 0.0385 vs 0.0479). Naive PPO is dominated only by PredPol, RF and the GreedyOracle. |
| E4 knee "at ~1 pt coverage cost relative to uniform" (paper l.403) | **NOT SUPPORTED** | pen_p0.02 has *higher* latent coverage than uniform (0.0473 vs 0.0455) |
| E4 frontier is monotone | SUPPORTED for (δ, lat_cov) | Gini is non-monotone (0.15 → 0.10 → 0.15 → 0.20) because of the overshoot (F5) |
| E5: perfect observation 9.04 → 2.92 | SUPPORTED as computed | but **1 training seed, 6 eval seeds (300–305)**, not ten as the paper says; trained for 120k steps vs 150k for main; per-episode base disparity ranges **4.4–13.8** |
| E5 configs differ only in detection parameters | SUPPORTED | `runs/abl_base_ppo_s0` vs `abl_perfect_obs_ppo_s0`: only `p_report`/`detect_gain` differ; training and eval seeds are matched |
| `run_ablations.py` accepts `--train-seeds` | SUPPORTED (uncommitted) | `scripts/run_ablations.py:45` |
| Lagrangian PPO works as described | **NOT SUPPORTED** | `constrained_ppo.py:71` targets a hard-coded `0.15` and ignores `alpha` |
| Python at `.venv/bin/python3` | NOT APPLICABLE here | `.venv/` is a macOS venv copied from a teammate. This machine uses `.venv-win/`. |

---

## B. Scientific corrections to the handoff

**B1. Primary metric: use the paired log-shift, and report the ratio as its exponential.**
`Δlog = mean over (train, eval seed) of [ln δ_policy − ln δ_oracle]`, paired on the eval seed.
Reasons:
- it is symmetric (over- and under-allocation weigh equally);
- pairing removes eval-seed variance (oracle sd 0.21);
- uncertainty comes from the hierarchical bootstrap directly, with no ratio-of-means error propagation.

On the handoff's audit points:
- **(A) Ratio instability is not a practical risk.** The 0.005 floor bounds Q1 exposure at ≥ 0.11, and the observed minimum is 0.16 (naive PPO).
- **(C) Use cumulative δ as primary and per-shift δ as secondary.** Per-shift δ for PPO is heavy-tailed (mean 20.8).
- **(D/E) Name the reference precisely:** "need-proportional reference (GreedyOracle)". `ppo_oracle` is a second check: it lands at 1.06×, so the two oracle-like controls agree.
- **(B) Because a ratio hides absolute differences, also report an allocation-level divergence.** For example, L1 distance between the deployment shares and the long-run need shares. This also gives Q2–Q4 a voice: the quintile ratio uses only 9 of 22 districts.

**B2. The handoff's §12 arithmetic needs redoing.** 8.081 / 2.712 = 2.98 uses the seed-2-only value. The pooled, paired estimate is **3.71× [2.84, 5.07]**. The wide CI reflects real training-seed instability (13.6 / 8.5 / 8.1), which needs reporting, not hiding.

**B3. E5 is a structural intervention, not a single-channel one, as currently run.** Setting q ≡ 0.995 (i) removes the action-dependence of detection, **and** (ii) raises the observed count by ~1.5–1.8× (realised mean q: uniform 0.667, naive PPO 0.561), which cuts observation noise. It also (iii) makes recorded crime equal latent crime, so naive PPO's objective *becomes* the oracle objective. That (iii) is exactly why the result (2.9 ≈ oracle 2.7) is meaningful, but (i) and (ii) are confounded.
- **Add a matched control: `p_report = 0.667, detect_gain = 0`.** This gives constant detection at the neutral level, with the same noise but no endogeneity. It is a config-only variant.
- Wording: "mechanism-isolating evidence within the simulator's structural equations". Not "proof", not "causal" beyond the model.

**B4. Replace the runaway index.** Candidates that actually need feedback to be non-zero:
- **Proxy gap** = recorded coverage − latent coverage. This is exactly the measurement inflation the policy induces on its own objective. Existing values: naive PPO 0.0164, PredPol 0.0044, oracle 0.0028, uniform 0. It is already computed per shift, so no new code is needed for the base value.
- **Self-induced signal**: the partial correlation between `dep[t−1]` and `(rec_share[t] − lat_share[t])`. It measures whether past patrol inflates a district's record relative to its truth.
- Keep Gini and the deployment-latent EMD (both valid, but they measure concentration and mismatch, not feedback).

**B5. E4 needs a reference-targeted regulariser.** `−w·δ` optimises toward δ → 0 (F5). Options:
- `−w·max(0, ln δ − ln δ_ref)`, a hinge at the reference;
- `−w·|ln δ − ln δ_ref|`;
- the existing Lagrangian with **`alpha = δ_ref` once the 0.15 bug is fixed**. This is the cleanest option because the code already exists.

Keep the current sweep as "penalty toward parity" for continuity. The trade-off figure's equity axis should be |Δlog|, not raw δ: under the new framing "lower δ" is not "better".

**B6. E3 respecification.**
- Dependent variable: call it **allocation residual relative to need**, `a_r − need_r`. Here `need_r` is long-run latent share under neutral (uniform) patrol, estimated on the same eval seeds.
- Regressors: hardship (+ poverty).
- SEs clustered on episode, and report a seed-level bootstrap. With 22 districts, do not cluster on district alone.
- The current spec (DV = dep − lat_t, control = lat_t) also has a mechanical problem: lat_t sits on both sides.
- State plainly that the coefficient is descriptive, not causal.

**B7. Two baselines are worth adding (optional, cheap):**
- **(a) Static-rule optimiser.** Run a grid search or CMA-ES over fixed (β, w, μ) maximising recorded coverage on the *training* seeds. If it reproduces PPO's behaviour, the finding is "the objective is exploitable" rather than "RL specifically". Either answer is publishable, but the paper must know which it is.
- **(b) 2×2 decomposition** of observation {recorded, latent} × reward {recorded-cov, latent-cov}. Currently naive PPO and `ppo_oracle` differ in both, so the 3.7× → 1.06× drop is not attributed to either one. Both cells are config-only (`observe_latent`, `reward_on`).

**B8. Statistical design.** The unit of analysis is the *episode*, nested in training seed. Report:
- mean ± hierarchical-bootstrap CI over (training seed, eval seed);
- training-seed spread separately;
- paired differences against the reference on shared eval seeds.

The new `summarise()` (uncommitted) computes CIs over all 30 episodes, treating them as independent. That is acceptable as a supplement but not as the primary CI. 22 districts give quintile groups of only 4–5 districts; disclose this.

---

## C. Code impact

| File / function | Current | Required | Risk | Tests |
|---|---|---|---|---|
| `eval/rollout.py::summarise` | Uncommitted fix to episode grouping | **Keep and commit.** Add `tseed` to rollout rows (or keep the pre-pooling frames) so training-seed identity survives `pool_by_family` | Low | Test: pooled frame of 2 training seeds → disparity equals the mean of both |
| `experiments/registry.py::pool_by_family` | Drops training-seed identity | Keep the family name and add a `tseed` column | Low | as above |
| **new** `eval/reference.py` | — | `need_shares(df_uniform)`, `log_shift_vs_reference(ep_table, ref)` with paired hierarchical bootstrap; reference derived from the data, never hard-coded | Medium (statistics) | Synthetic test with a known ratio; identity test (reference vs itself → 0) |
| `eval/feedback_metrics.py` | runaway index | Add `proxy_gap` and `self_induced_signal`; keep `runaway_index` but rename it in figures/paper ("allocation persistence") or drop it | Low | Static allocation → self-induced ≈ 0; uniform → proxy gap 0 |
| `eval/incidence.py` | DV = dep − lat_t, control lat_t, HC1 | Add a `need`-based spec plus episode-clustered SEs; keep the old spec as a robustness row. Remove the unused `last` in `exposure_by_hardship_quintile` | Medium | Cluster SE vs HC1 on i.i.d. data ≈ equal |
| `eval/stats.py::ols` | HC1 only | Add a `cluster=` argument (CR1) | Low | as above |
| `eval/pareto.py`, `figures.py::fig_pareto/fig_efficiency_equity` | x = raw δ, "lower is better" | x = |Δlog| vs reference (or δ with the reference line drawn); relabel axes; no hard-coded 2.71 | Low | Regenerate and inspect |
| `rl/constrained_ppo.py::DualAscentCallback` | targets `0.15`, ignores `alpha` | Use `alpha` with a cost of `max(0, δ_cum − alpha)` → target δ_ref | Medium (needs retrain) | Short training run: λ rises when cost > 0 |
| `rl/constrained_ppo.py` (optional) | `−w·δ` | add `penalty_mode: "linear" \| "hinge_ref" \| "abs_log_ref"` | Medium | Unit test of shaped reward |
| `scripts/run_ablations.py` | 11 variants; stale "Lagrangian" docstring | Add `const_detect` (`p_report 0.667, detect_gain 0`); fix the docstring; output mean ± CI over training seeds | Low | — |
| `scripts/run_all.py` / `registry.ensure_trained` | Reuses any existing run | Store `total_timesteps` in the run name (e.g. `ppo_t150k_s3`) **or** assert the config matches the cache | Low; prevents F2 | Test: mismatched config raises |
| `env/fairness.py`, `rl/baselines.py` docstrings | Say "population ∝ real crime" | Say "uniform residential population" | None | — |
| `requirements.txt` | Lower bounds only | Pin the versions the results were produced with (pandas 3 vs 2.2 differences) | Low | — |
| `env/detection.py`, `abm/*`, `env/policing_env.py`, `rl/ppo.py` | — | **No change** | — | Existing 20 tests + calibration |

---

## D. Experiment impact

| Item | Category |
|---|---|
| E1 pooled numbers (fix F1), oracle-relative log-shift, proxy gap | **Reuse existing data**: reanalysis only, from `runs/*/eval_e200_201n10.pkl` |
| E3 need-controlled, clustered spec | Reanalysis only |
| E2 with the new feedback metrics | Reanalysis only (arrays are stored) |
| E4 re-axis to |Δlog| | Reanalysis only |
| Seeds 3–4 at a consistent budget | **Retrain** (or retrain 0–2 at 400k) |
| E1–E4 on 30 eval seeds | **New eval** (no retrain needed if the budget is settled); baselines on 100–129 are already cached |
| E5 multi-seed (3 training × ≥ 20 eval seeds) + `const_detect` control | **Retrain**: 3 variants × 2 kinds × 3 seeds = 18 runs |
| Reference-targeted regulariser sweep / fixed Lagrangian | **New code + retrain** |
| Static-rule optimiser baseline | **New code**, no RL |
| 2×2 observation × reward | **Retrain**, config only. Optional but strongly recommended (it attributes the main effect). |
| Direct-action ablation, g×κ grid, population weights | Optional |

---

## E. Manuscript impact (`paper/main.tex`)

| Section | Change |
|---|---|
| Title | Rewrite (after results are frozen) |
| Abstract | Rewrite: new framing, final numbers only |
| Introduction (l.75) | Rewrite: endogenous observation / performative feedback first, policing as testbed; contributions list (l.130-139) rewrite |
| Related work (l.141) | Minor edit + **new citations**: performative prediction (Perdomo et al., ICML 2020), selective labels (Lakkaraju et al., KDD 2017). Verify both before inserting. |
| Simulator (l.182-230) | Minor edit: make L ≠ D explicit; keep the details |
| Closed-loop / POMDP (l.232-300) | Rewrite as an explicit POMDP: observation kernel `P(D | L, u)`; **define the reference policy and Δlog**; Oracle l.286 "upper bound" → "need-proportional reference" |
| E1 (l.307) | Rewrite numbers (F1); **new table** with the Δlog column |
| E2 (l.354) | Rewrite: drop the runaway-index claims; proxy gap; **figure regenerated** |
| E3 (l.372) | Rewrite: new spec, clustered SEs; remove "p < 10⁻³⁰⁰" and "3× PredPol" |
| E4 (l.393) | Rewrite: fix the false dominance claim and the "1 pt vs uniform" claim; reference-targeted regulariser; **figure re-axed** |
| E5 (l.412) | Rewrite: multi-seed, constant-detection control, cautious wording; fix "ten eval seeds" → actual |
| Discussion / Limitations | Add: quintile size (4–5 districts), training-seed instability, rule action space, uniform population |
| **New Figure 1** | Architecture diagram (L → u → {deterrence, detection} → D → policy) |

---

## F. Execution plan

```
0. Freeze ─ commit the uncommitted summarise() fix + this audit; tag the current results
   (git tag pre-pivot); copy results/ → results/pre_pivot/ (pkl files are git-ignored)
1. Decisions (team) ─ see list below
2. Code: metrics + statistics (reference.py, clustered OLS, proxy gap, tseed column,
   run-name/config guard)      + unit tests            [no simulator changes]
3. Reanalysis of existing 3×10 data ─ corrected E1, Δlog, E2, E3, E4
   → confirms or kills the story before spending compute
4. Fix the Lagrangian; add the const_detect variant; (optional) static-optimiser baseline
5. Retrain to a single budget; E5 multi-seed (3 × ≥20 eval seeds)
6. E1–E4 at 30 eval seeds; (optional) 2×2 attribution, reference-targeted regulariser sweep
7. Statistics freeze → figures → manuscript → final validation (tests, calibration,
   LaTeX build, number-consistency check of tables vs CSVs)
```

**Decisions needed from the team before step 2:**
1. **Training budget**: 150k (seeds 0–2 exist) or 400k (seeds 3–4 exist)? One must be retrained.
2. **Eval seeds for E1–E4**: 100–129 (baselines already cached, matches EXPERIMENT_ROADMAP.md) or 200–229 (friend's handoff; extends the current 200–209)? Recommendation: **200–229**, so the existing 10 seeds are a subset and stay comparable.
3. **Primary metric**: paired log-shift vs GreedyOracle (recommended) or the ratio of means?
4. **E4 regulariser**: keep the parity penalty only, or add the reference-targeted one (recommended)?
5. **Scope**: include the 2×2 attribution and the static-optimiser baseline? Both are cheap and pre-empt the obvious reviewer question.
