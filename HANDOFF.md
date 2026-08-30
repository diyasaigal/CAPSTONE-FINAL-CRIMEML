# Capstone — Team Handoff

Everything you need to understand the pivot, the codebase, and how to run it.
Read top to bottom once; after that use it as a reference.

---

## 1. Where we were, and why we are pivoting

### 1.1 What we had (Phase 1 + Phase 2)

We built an **agent-based model (ABM)** of urban violent crime for Chicago:

- Agents carry a socioeconomic **stress** level derived from real US Census (ACS) hardship and
  unemployment data for Chicago community areas.
- Violent crime is produced by a decision rule that mixes **Social Disorganization Theory**
  (stress from neighbourhood disadvantage) and **Routine Activity Theory** (agents move between
  home / work / leisure and crime happens where paths cross).
- The environment starts **unseeded**: the risk grid is a flat constant, and *no historical
  crime coordinates are injected*. Any spatial crime pattern that appears is emergent.
- We then did an "ML reality check": trained a RandomForest to confirm the simulated crime map
  correlates with the real Chicago crime map (ROC-AUC ~0.95).

### 1.2 What the review said

Our advisor's feedback was, in effect:

1. This domain (crime ABM) is hard to publish. Pivot toward an **AI/ML method** — RL especially —
   so the work reads as *method research that uses crime as the testbed*, not a crime paper.
2. The ML/RL has to be **substantial**, not a bolt-on validation step. Basic ABM alone is not
   enough for a research contribution.
3. We need **one clear selling point** and **one sharp novel point**.

### 1.3 Continuity — this is not a restart

We keep the same simulator, the same data, the same theory, the same four authors, the same
advisor. What changes is the **research question**. The panel should see a natural progression:

| Phase 1–2 | Now |
|---|---|
| Build an unseeded, census-grounded emergent crime ABM | Use it as the **environment** for a reinforcement-learning study |
| "Unseeded" was a defence against circular validation | "Unseeded" is now the **enabling condition**: because no crime is injected, the true (latent) crime process is known, so we can measure exactly how far a learned policy drifts from it |
| RandomForest = a validation diagnostic | The same RandomForest is now a **baseline** the RL agent is compared against |
| What-if experiments on hardship | The what-if machinery becomes the **robustness sweep** |

One sentence for the panel: *"We took the unseeded simulator we built and asked what happens when
a reinforcement-learning patrol-allocation policy is put in a closed loop with it — a question you
cannot answer with a seeded model, because there the ground truth is unknown."*

---

## 2. The new idea

### 2.1 The phenomenon

Predictive-policing systems send patrols where crime has been **recorded**, then retrain on the
crime those patrols go on to record. An officer present in an area discovers a larger share of
the crime that happens there, so patrolling an area **raises its recorded crime even if the true
rate is unchanged** — which confirms the forecast that sent them. This is the **runaway feedback
loop** (Ensign et al., FAT\* 2018; Lum & Isaac, 2016). It is known to fall hardest on
disadvantaged and minority neighbourhoods.

### 2.2 What is missing from prior work

- Studies of the loop use **abstract urn models** or real data where the **true crime rate is
  unknown**, so they can only measure drift from *previously recorded* crime, not from where
  crime actually is.
- Recent RL / multi-agent-RL papers on patrol allocation (2024–2026) optimise **coverage of a
  fixed historical or predicted signal** and **never close the loop** — the policy does not
  train on data it generated.

### 2.3 Our contribution

We put a **sequential RL policy in a closed loop** with our **unseeded, theory-grounded**
simulator, where the latent crime process is fully known. The policy learns the **parameters of
an allocation rule** each shift — how sharply to concentrate on recently recorded crime (β), how
much to tilt toward high-hardship districts (w), how much to hold uniform (μ). This is learnable
in a noisy environment and is a faithful model of how real predictive-policing systems work
(tuned heuristics). This setup lets us:

1. Measure whether the learned rule covers **real** crime or only **recorded** crime — the
   efficiency axis is *coverage* (share of crime occurring under patrol), which is what patrol
   allocation actually optimizes. Impossible to check without ground truth.
2. Measure the **socioeconomic incidence** of the drift: does the rule over-police disadvantaged
   districts *beyond what their true crime justifies*?
3. Show that a **fairness penalty** on the per-shift enforcement disparity, swept from light to
   heavy, removes the drift along a **monotone efficiency–equity frontier**.

### 2.4 Selling point (the one-liner)

> A reinforcement-learned patrol-allocation rule, trained on the deployable objective of
> predictive policing, learns a sharp concentration and an explicit tilt toward the poorest
> districts — posting the best *recorded*-crime coverage of any method while covering the
> *least* actual crime among targeted policies — and a swept fairness penalty removes the tilt
> along a clean trade-off.

### 2.5 Novelty (the sharp claim)

> First closed-loop study of predictive-policing feedback built around a theory-grounded,
> *unseeded* emergent simulator with known ground-truth latent crime. This is what lets us
> measure the gap between what the learned rule optimizes (recorded-crime coverage) and what it
> actually delivers (latent-crime coverage), and the socioeconomic incidence of that gap, after
> controlling for true crime — and to trace a fairness–efficiency frontier against ground truth.

### 2.6 Why this helps us beyond the capstone

The work sits in constrained/fair RL + AI-for-social-good + simulation — the research areas of
groups at Harvard (Tambe), CMU (Fang, Heidari), Stanford (Goel). A second RL paper (Dennis
already has one) on fair RL with a real societal testbed is a strong, coherent signal for MS
applications.

---

## 3. System architecture

```
                 ACS hardship / unemployment          real Chicago crime CSV
                 (77 community areas)                  (439k violent records, 2020-26)
                          |                                     |
                          v                                     v
              +-----------------------------------------------------------+
              |  RegionInfo (crimesim/abm/regions.py)                      |
              |  22 Chicago police districts = allocation + fairness unit  |
              |  per-district: hardship, poverty, income, real-crime share |
              +-----------------------------------------------------------+
                          |
                          v
   +--------------------------------------------------------------+
   |  CrimeGenerator  (crimesim/abm/generator.py + dynamics.py)   |
   |  vectorised NumPy port of the Phase-2 Mesa model:            |
   |   - N=200 heterogeneous agents, bound to a home district     |
   |   - stress = base + district hardship + noise                |
   |   - routine-activity movement (home/work/leisure by hour)    |
   |   - crime = f(stress, environmental risk, interaction)       |
   |   - risk grid starts FLAT (unseeded)                         |
   |  -> latent crime per district per shift  (GROUND TRUTH)      |
   +--------------------------------------------------------------+
                          |  latent crime L_r
                          v
   +--------------------------------------------------------------+
   |  PolicingEnv  (crimesim/env/policing_env.py)  [Gymnasium]    |
   |                                                              |
   |  action (beta, w, mu) -> allocation RULE -> deployment       |
   |     share_r ~ (1-mu)*softmax(beta*log EWMA_recorded_r        |
   |                              + w*z_hardship_r) + mu*uniform   |
   |  patrol has two effects on the world:                        |
   |    (a) DETERRENCE: removes a bounded fraction of L_r          |
   |    (b) DETECTION:  recorded D_r ~ Binomial(L_r, q_r),         |
   |                    q_r rises with patrol   <-- the loop       |
   |  observation = EWMA of RECORDED crime (never latent)          |
   |  reward "captured" = coverage of recorded crime under patrol  |
   |               (= the operational predictive-policing goal)    |
   +--------------------------------------------------------------+
              |                         |                    |
              v                         v                    v
     baselines (crimesim/rl/     PPO (ppo.py)        penalized / Lagrangian PPO
       baselines.py)             Stable-Baselines3   (constrained_ppo.py)
     uniform / random /                              reward - w_f * disparity,
     PredPol-EWMA / RF /                             w_f swept -> frontier
     greedy-oracle
              \_________________________|____________________/
                                        v
              +---------------------------------------------+
              |  eval harness (crimesim/eval/)              |
              |  E1 efficiency/equity   E2 feedback metrics |
              |  E3 incidence regression E4 Pareto frontier |
              |  -> results/*.csv  +  figures/*.pdf         |
              +---------------------------------------------+
```

---

## 4. Technical plan and how each part is implemented

### 4.1 The simulator (`crimesim/abm/`)

| File | What it does | Key choices |
|---|---|---|
| `regions.py` | Builds the 22 Chicago police districts from the crime data; assigns each 20×20 grid cell its modal district; aggregates ACS covariates to districts (crime-weighted). | Districts, not community areas: a 20×20 grid cannot resolve 77 CAs, and districts are the real patrol-allocation unit. Residential population assumed uniform across districts (documented simplification) so per-capita crime is *emergent*, driven by hardship. |
| `dynamics.py` | Pure, unit-tested functions: toroidal neighbour sums, routine-activity movement, the mean-field interaction-escalation term, the two crime-trigger paths. | The Phase-2 Mesa model assigned each agent a *random* community area unrelated to position — so its interaction terms were already effectively mean-field. We make that explicit and vectorise it, which is why the whole thing runs ~40× faster. |
| `generator.py` | `CrimeGenerator` — holds agent state as NumPy arrays, steps the world one shift at a time, returns latent crime per cell and per district. Agents are **bound to a home district** (home + leisure in it, workplace sometimes elsewhere = commuting). | Binding agents to districts is a genuine improvement over Phase 2: it makes the socioeconomic geography spatially coherent, which is a prerequisite for a spatial fairness study. |
| `calibration.py` | The **fidelity gate**. Run it before trusting anything. Checks the port still shows the Phase-2 signatures. | Passes: SDT elasticity ~+27% between low/high hardship (Phase-2 reported +42.6%), night crime > day crime, emergent spatial clustering from a flat start, monotone Low<Medium<High. We re-tuned two constants (`risk_gain`, `risk_trigger`) to compensate for the vectorised port. |

### 4.2 The closed loop (`crimesim/env/`)

The environment is where the research question lives. One `step` = one 8-hour **shift**.

- **`detection.py`** — the observation model. `q_r = q_0 + g · min(1, u_r / u_sat)` where `u_r` is
  patrol intensity in district `r`. Baseline reporting `q_0 = 0.3`; full patrol pushes detection
  toward ~0.85. Recorded crime is a Binomial thinning of latent crime at rate `q_r`. **This is
  the mechanism that closes the loop**: patrolling a district makes it look worse.
- **`fairness.py`** — enforcement **exposure per capita** = deployment share / population share.
  The headline disparity metric is the ratio of mean exposure in the most-disadvantaged district
  quintile to the least-disadvantaged. Per-shift cost for the constrained learner is
  `max(0, disparity − 1)`.
- **`policing_env.py`** — ties it together.
  - **Action (`action_mode="rule"`, the main setting)**: 3 numbers → `(β, w, μ)` → a deployment
    via the allocation rule `share_r ∝ (1-μ)·softmax(β·log(EWMA_recorded_r) + w·z_hardship_r) + μ·uniform`.
    A 0.5% floor per district. `action_mode="direct"` (raw 22 logits) is kept as an ablation —
    PPO can't explore it in this noisy environment.
  - **Reward (`reward_on`)**: `captured` (default) = *recorded*-crime coverage, the deployable
    "be where crime is found" objective — maximising it drives the loop. `latent_coverage`
    (oracle, + `observe_latent`) = true-crime coverage. `latent` / `recorded_neg` are ablations.
    Coverage rewards are scaled ×10 for PPO value fitting.
  - The agent observes an **EWMA of recorded crime** (+ last deployment, time, static covariates)
    — never latent crime, unless `observe_latent` (oracle only).
  - Each episode warms up (20 shifts under proportional patrol) so the ABM transient has passed.

### 4.3 The learners (`crimesim/rl/`)

- **`baselines.py`**:
  - `uniform` — patrol ∝ population (fair by construction, the reference point).
  - `random`.
  - `ewma_hotspot` — **PredPol-style**: patrol ∝ an exponentially-weighted moving average of
    recorded crime. The exact heuristic Ensign et al. show is loop-prone.
  - `rf_hotspot` — our Phase-2 RandomForest, made sequential: refit every 15 shifts on
    accumulated recorded data, patrol ∝ predicted next-shift recorded crime.
  - `oracle` — patrol ∝ an EWMA of *latent* crime. Not deployable; an upper bound on how much a
    well-targeted policy can cut real crime.
- **`ppo.py`** — Stable-Baselines3 PPO, MLP policy (128×128), `DummyVecEnv` (in-process; the env
  step is cheap NumPy and inter-process communication was slower), `VecNormalize` on the
  observation. Saves the model + the observation-normalisation stats.
- **`constrained_ppo.py`** — the fairness learners.
  - **`train_penalized_ppo` (main)**: a fixed penalty `w_f` on the per-shift enforcement
    disparity — shaped reward `rₜ − w_f·δₜ`. Sweeping `w_f` traces the efficiency–equity frontier
    by construction. Robust, no tuning.
  - **`train_lagrangian_ppo` (alternative, reported for comparison)**: dual ascent on a
    multiplier toward a disparity ceiling `α` (Chow et al. 2017; Tessler et al. 2019; Stooke et
    al. 2020). Finicky about the cost-signal / episode-length interaction; the penalty method is
    what we use for the headline frontier.

### 4.4 Evaluation (`crimesim/eval/`)

Every headline number is a mean over **held-out ABM seeds** (seeds the policy never trained on),
with 95% confidence intervals.

| File | Experiment | Output |
|---|---|---|
| `rollout.py` | runs any policy (RL or rule-based) on held-out seeds, logs per-shift latent/recorded/fairness | tidy DataFrame |
| `stats.py` | OLS with heteroskedasticity-robust (HC1) standard errors | used by E3 |
| `feedback_metrics.py` | **E2** — enforcement Gini over time, earth-mover distance between the recorded and latent crime distributions, and a "runaway index" (partial correlation of this shift's deployment with cumulative past deployment, controlling for current latent crime) | trajectories + figure |
| `incidence.py` | **E3** — regress `over_policing = deployment_share − latent_share` on district hardship, controlling for latent crime; a positive hardship coefficient = discriminatory loop | regression table + exposure-by-quintile figure |
| `pareto.py` | **E4** — place every policy in (disparity, latent crime) space, find the Pareto-efficient set and the knee point | frontier table + figure |
| `figures.py` | all 5 paper figures, matplotlib only | `figures/*.pdf` |

### 4.5 Tools and versions

Python 3.12 · NumPy · pandas · scikit-learn · matplotlib · **Gymnasium 1.3** ·
**Stable-Baselines3 2.9** · PyTorch 2.13 (CPU) · pytest. Everything runs on a laptop CPU; no GPU
needed.

---

## 5. Experiments

| ID | Question | Method |
|---|---|---|
| **E0** | Does the fast simulator still behave like the Phase-2 model? | `crimesim/abm/calibration.py` — the fidelity gate |
| **E1** | Does the trained rule cover *latent* crime, or only *recorded* crime? | Train PPO(`captured`); compare recorded vs latent coverage against baselines and the oracle |
| **E2** | Does the loop run away over a deployment horizon? | Track enforcement Gini, latent coverage, deployment-vs-latent EMD, runaway index, shift by shift |
| **E3** | Does the drift land on disadvantaged districts, after controlling for their true crime? | OLS: `over_policing ~ hardship + latent_share (+ poverty)`; hardship coefficient per policy |
| **E4** | Can a fairness penalty break the loop, along what trade-off? | Penalized PPO swept over `w_f`; efficiency–equity frontier of disparity vs latent coverage; knee |
| **E5** | Is the effect robust? | Sweep detection gain, deterrence, patrol budget, history length, socioeconomic features, grid size, seeds. **Falsification check**: with perfect observation (`p_report = 1`, gain 0) the loop should vanish. |

---

## 6. What the results mean (multi-seed: 3 training × 10 held-out eval seeds)

| policy | recorded cov. | real cov. | disparity (per-shift) | over-policing ∝ hardship |
|---|---|---|---|---|
| PredPol heuristic | 0.059 | **0.055** | 3.8 (3.9) | +0.005 |
| RandomForest | 0.059 | 0.055 | 3.6 (3.8) | +0.007 |
| Greedy oracle | 0.056 | 0.053 | 2.7 (2.8) | +0.003 |
| **naive RL** | **0.064** (best of all) | 0.048 (lowest of targeted) | **8.1 (20.8)** | **+0.017** — 3× PredPol |
| RL oracle (right objective) | 0.055 | 0.049 | 2.8 (6.6) | +0.008 |
| RL + penalty 0.02 (knee) | 0.051 | 0.047 | 1.5 (1.9) | −0.004 |
| RL + penalty 0.40 | 0.041 | 0.039 | 0.5 (0.6) | −0.026 |
| uniform | 0.045 | 0.045 | 1.0 | 0 |

- **The naive rule games the observable metric.** Highest *recorded* coverage of any method
  ("looks best" on the dashboard); *real* coverage lower than every targeted baseline. It
  optimizes what it can see, not what is happening. Learned parameters: `β ≈ 3` (sharp
  concentration), **`w > 0` (explicit tilt toward high-hardship districts)**, `μ ≈ 0` (no
  uniform mix). Cumulative enforcement Gini 0.46 vs the heuristics' 0.28.
- **It is Pareto-dominated** (E4 figure): every heuristic and every penalised variant reaches
  *both* higher real coverage *and* lower disparity. The naive RL sits alone in the "worst
  trade-off" corner.
- **The drift is discriminatory.** Over-policing ~ hardship, controlling for true crime:
  **+0.017 (p ≪ 10⁻³⁰⁰, R² 0.40)** — ~**3× the PredPol baseline** (+0.005). Enforcement per
  capita in the most-disadvantaged quintile: ~3× (naive RL) vs ~1.6× (heuristics) vs ~1 (fair).
- **The fairness penalty gives a clean monotone frontier.** disparity 8.1 → 1.5 → 0.9 → 0.6 →
  0.5; hardship coefficient +0.017 → −0.026. The knee (penalty 0.02) removes the disparity for
  ~1 pt of real coverage.
- **Because the simulator is unseeded**, none of this is a biased-data artefact — it is the
  learned rule in the detection loop.

**Honest weaknesses to disclose:** (a) spatial allocation barely moves *total* crime here
(bounded deterrence), which is why the efficiency axis is *coverage*, not crime totals —
consistent with the displacement literature; (b) the task is largely reactive, so an RL oracle
(right objective + full observation) only *matches*, not beats, the reactive heuristic on
coverage — which sharpens the point that the naive agent *loses* coverage by chasing the
recorded signal. E5 (`run_ablations.py`) sweeps the loop-strength parameters; the
`perfect_obs` variant is the falsification check.

---

## 7. Mini "how to run" (full VS Code guide is in `RUNNING.md`)

```bash
# one-time environment
conda create -n crimeml python=3.12 -y
conda run -n crimeml pip install -r requirements.txt

# one-time data prep (builds data/region_district_g20.npz from the crime CSV)
conda run -n crimeml python scripts/build_ca_lookup.py --grid 20

# sanity checks (both should pass)
conda run -n crimeml python -m pytest tests/ -q
conda run -n crimeml python -m crimesim.abm.calibration

# fast end-to-end demo: trains a short PPO + a fair PPO, prints the comparison (~15 min)
conda run -n crimeml python scripts/quicktest.py --timesteps 150000

# full paper results (a few hours, resumable — safe to Ctrl-C and re-run)
conda run -n crimeml python scripts/run_all.py \
  --train-seeds 0 1 2 --eval-seeds 200 201 202 203 204 205 206 207 208 209 \
  --alphas 0.3 1.0 3.0 --timesteps 170000

# robustness sweeps + the perfect-observation falsification control
conda run -n crimeml python scripts/run_ablations.py
```

Outputs land in `results/` (CSV tables) and `figures/` (PDF figures).

> On Windows, if `conda run ... python -c "..."` fails with a newline error, call the interpreter
> directly: `C:/Users/<you>/miniconda3/envs/crimeml/python.exe`, and set `PYTHONPATH=.`.

---

## 8. Division of work (suggestion)

| Area | Owner |
|---|---|
| Simulator fidelity + E5 ablations (esp. the perfect-observation control) | |
| RL training runs + hyper-parameter stability across seeds | |
| E2/E3 analysis + figures + statistics write-up | |
| Paper: Intro / Related Work / Discussion + submission logistics (JAAMAS) | |

Whoever owns the paper: JAAMAS uses the Springer format; our draft is in IEEE format for now
(easy to switch). Target JAAMAS 2027; AIES / FAccT 2026 workshops are good parallel venues.
