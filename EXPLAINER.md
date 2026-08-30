# The research project, explained in full

This is the "learn it 100%" document. It assumes you know basic probability and have seen
reinforcement learning at a high level. By the end you should be able to defend every design
choice in a viva.

---

## Part 1 — The problem

### 1.1 Predictive policing and the feedback loop

A predictive-policing system does roughly this, every day:

1. Look at recent recorded crime per area.
2. Predict where crime will be highest tomorrow.
3. Send patrols to the top predicted areas.
4. Record the crime those patrols encounter.
5. Go to step 1, now including today's records.

The problem is in the coupling of steps 3 and 4. **An officer who is present in an area discovers
a larger fraction of the crime that occurs there** — more incidents get reported, more stops
happen, more arrests are made. So sending patrols to an area *increases that area's recorded
crime* even if the true underlying crime rate did not change at all.

Now follow the loop. Area A gets a high prediction (maybe just noise). Patrols go there. Recorded
crime in A goes up because of the extra discovery, not because A got more dangerous. Tomorrow's
prediction for A is even higher. More patrols. And so on. The system converges to over-policing
whichever areas it happened to over-police first, **regardless of the true crime distribution**.

Ensign, Friedler, Neville, Scheidegger and Venkatasubramanian (FAT\* 2018) proved this formally
with a Pólya urn model and called it a *runaway feedback loop*. Lum and Isaac (2016) showed the
real-world consequence: when PredPol's algorithm was run on Oakland drug-crime data, it would
have sent police disproportionately to Black and low-income neighbourhoods, amplifying an existing
reporting bias.

### 1.2 Two things missing from prior work

**(a) Ground truth.** With real crime data you never observe the true crime rate — only what was
recorded, which is exactly the biased quantity. So you can measure that a policy's enforcement
drifts away from *previously recorded* crime, but you cannot measure whether it drifts away from
*where crime actually is*. The urn models sidestep this by being abstract (no space, no
socioeconomics, no real dynamics).

**(b) A learning policy.** The recent wave of reinforcement-learning and multi-agent-RL papers on
police patrol (2024–2026) all optimise **coverage of a fixed signal** — historical crime, or an
ML forecast trained once. The policy's actions never feed back into its own training data. So
these papers cannot exhibit the loop even in principle.

### 1.3 What we do

We build a simulator where **the true (latent) crime process is known exactly**, put a
**sequential RL policy** in a **closed loop** with it (the policy's patrols change what it later
observes), and measure:

- **E1**: does the policy reduce *true* crime, or just *recorded* crime?
- **E2**: does its enforcement run away from the true crime distribution over time?
- **E3**: does the runaway fall on disadvantaged districts, *after controlling for their true
  crime*?
- **E4**: can a fairness constraint stop it, and what does it cost?

---

## Part 2 — The simulator (the environment)

### 2.1 Why we can trust it: it is our Phase-2 model

The simulator is a faster reimplementation of the agent-based model we built in Phase 1–2. That
model was validated: it reproduces real Chicago crime hotspots (ROC-AUC ~0.95), it shows a +42.6%
rise in violent crime between low- and high-hardship scenarios (consistent with Social
Disorganization Theory), and it produces spatial crime clusters starting from a completely flat
environment.

The new implementation is in NumPy instead of the Mesa framework, roughly 40× faster (needed
because RL requires millions of simulation steps), and it passes a **calibration gate**
(`crimesim/abm/calibration.py`) that checks all of those signatures still hold. Running that gate
is the first thing anyone should do.

### 2.2 The pieces

**Space.** A 20×20 grid laid over Chicago. Each cell is assigned its **modal police district**
(there are 22) using the real crime data's district column. Police districts, not the 77 census
community areas, because (i) a 20×20 grid cannot resolve 77 areas, and (ii) districts are the
real unit that police departments actually allocate patrols to.

**Agents.** 200 people. Each has:
- a **home district** `r`, assigned so districts have comparable residential population (a
  simplification we state openly — it makes per-capita crime an *emergent* quantity rather than
  something we hard-code);
- a **stress** level `s = s_0 + hardship_r + noise`, where `hardship_r` is the ACS hardship index
  of the district (crime-weighted average of the community areas in it), scaled to `[0, 1]`;
- an **employment** status, drawn at the district's real unemployment rate;
- a **home cell**, **leisure cell** (both inside the home district), and a **workplace cell**
  (in the home district 60% of the time, elsewhere 40% — modelling commuting, so daytime
  activity mixes people of different stress levels).

**Movement (Routine Activity Theory).** Each sub-step, agents step one cell toward a target that
depends on the hour: workplace during 9–17 (if employed), leisure during 18–23, home otherwise.

**Environmental risk.** A grid `T` that starts at a flat `0.1` everywhere (**this is the unseeded
part — no real crime is injected**). Each sub-step it decays slightly and increases where agents
are. Over time, busy cells accumulate high risk. This is emergent, not given.

**Crime.** Each sub-step, an agent commits a violent act through either of two paths (adapted
from Rosés et al. 2021):
- **environmental**: a smoothed risk quantity
  `ρ = (1 − m)·[(s + hardship_r)·T(cell) + γ·I] + m·ρ_previous`
  crosses a threshold. Here `I` is an **interaction-escalation** term that rises with the number
  of nearby agents and how spread-out their stress levels are (encounters between people under
  very different strain are more likely to escalate).
- **interaction**: a random escalation whose probability rises with the count of nearby
  high-stress agents.

Night sub-steps carry a 1.5× activity factor (crime clusters at night, as in the real data).

After a threshold crime, that agent's stress drops (a release), and a small fraction of agents
"turn over" each shift (move away, replaced by fresh agents) so the population stays stationary
over a long deployment horizon.

### 2.3 The output

One **shift** = 4 sub-steps (≈ 8 hours). Calling `run_shift()` advances the world one shift and
returns:
- `latent_region`: the true violent-crime count in each of the 22 districts this shift — **the
  ground truth, hidden from the RL agent**;
- `latent_cell`: the same on the fine grid (for figures).

### 2.4 The key property

Because no crime coordinates ever enter the environment, `latent_region` is a clean function of
the socioeconomic geography (hardship) and the agent dynamics. High-hardship districts genuinely
produce more crime per person, and that structure *emerges* — we can inspect it, and we can check
whether a policy respects it or distorts it.

---

## Part 3 — The closed loop (the RL environment)

This is `crimesim/env/policing_env.py`. It is a standard Gymnasium environment. One `step` is one
shift.

### 3.1 State (what the agent sees)

A flat vector:
- the last `H = 4` shifts of **recorded** crime per district (22 numbers × 4);
- the previous deployment (22 numbers);
- time-of-day (sin/cos of the hour) and how far through the episode we are;
- optionally, static district covariates: hardship, poverty, income, unemployment (22 × 4).

**The agent never sees latent crime.** It only ever sees recorded crime, which is biased by its
own past actions.

### 3.2 Action — the allocation rule

Real predictive-policing systems are **tuned rules** that map a recent-crime signal to a
deployment. We model the policy's action as the choice of that rule's three parameters each
shift:

```
share_r  ∝  (1 - μ) · softmax( β · log EWMA_recorded_r  +  w · z_hardship_r )  +  μ · (1/22)
```

- **β ∈ [0, 4]** — how sharply the rule concentrates on recently recorded crime.
- **w ∈ [-2, 2]** — an explicit tilt *toward* (w > 0) or *away from* (w < 0) high-hardship
  districts (`z_hardship` is standardised district hardship).
- **μ ∈ [0, 1]** — how much to mix back toward a uniform allocation.

A 0.5% floor per district means none is ever fully abandoned. The budget `B = 20` is split in
these proportions.

Why not let the policy output a raw 22-district vector? We tried — PPO can't explore a 22-dim
continuous simplex in this noisy environment; it does no better than uniform even when given the
right objective. The raw version is kept as `action_mode="direct"` for an ablation. The
rule-parameter version is learnable *and* is a more honest model of deployed systems.

### 3.3 Transition (what patrol does to the world)

Let `u_r` = patrol intensity in district `r` = (its share of the budget) ÷ (its share of a
proportional allocation). `u_r = 1` means "policed exactly in proportion to size"; `u_r = 3`
means "three times its fair share".

Two effects:
- **Deterrence** (patrol reduces real crime): a fraction `min(0.55, 0.20 · u_r)` of latent crime
  in district `r` is prevented. Bounded — you cannot drive crime to zero by flooding one district
  (consistent with the hot-spots-policing meta-analysis, which finds ~15–20% local reductions).
- **Detection** (patrol reveals more crime): recorded crime
  `D_r ~ Binomial(L_r, q_r)` where `q_r = 0.30 + 0.55 · min(1, u_r / 1.5)`. With no patrol only
  30% of incidents are recorded; with heavy patrol ~85%.

**This is the loop.** Patrolling a district both lowers its true crime a little and roughly
*triples* how much of the remaining crime gets recorded. Net effect on the record: patrolled
districts look worse.

### 3.4 Reward — coverage, not crime totals

Define the **coverage** of a crime field `X` under a deployment as
`cov = Σ_r share_r · X_r / Σ_r X_r` — the share of that crime which occurred in districts you
were patrolling. (This is essentially the Predictive Accuracy Index used to evaluate real patrol
allocation.)

- **`captured` (default reward)** = coverage of *recorded* crime. This is the deployable
  objective: "be where crime is found." Maximising it means concentrating where recorded crime
  is — which, through the detection effect, is partly a distribution the agent creates.
- **`latent_coverage` (oracle)** = coverage of *true* crime. Not deployable (no real system sees
  latent crime); the oracle also gets latent crime in its observation. Answers "how well could
  an RL agent do with the right objective?"
- **`latent` / `recorded_neg`** — minimise crime totals; ablations only.

**Why coverage and not "crimes prevented"?** In this simulator, deterrence is bounded (~15–25%
in a patrolled district) and there's no displacement modelling, so *total* latent crime barely
moves with allocation (the best heuristic beats uniform by ~12%, the oracle by ~6%). Coverage is
the quantity that actually varies with allocation, and it's what the predictive-policing
literature uses. The finding is about *where* patrol goes, not how many crimes vanish.

### 3.5 Warm-up

Each episode resets the simulator to a fresh random seed and runs 25 shifts under proportional
patrol before the agent takes over, so the simulator's transient (risk grid building up) has
passed.

---

## Part 4 — The learners

### 4.1 PPO (the naive policy)

Proximal Policy Optimization (Schulman et al. 2017) from Stable-Baselines3: a policy network
(two hidden layers of 128 units) outputs the 3 rule parameters `(β, w, μ)`; a value network
estimates expected return; PPO improves the policy with a clipped surrogate objective. Trained on
randomly-seeded episodes; evaluated on **held-out seeds** the policy never saw. Coverage reward
is scaled ×10 so the value network fits it well.

Nothing exotic — the point is not a new RL algorithm, it is what a competent optimiser does when
handed this realistic, widely-used objective.

### 4.2 The fair policy — penalty method (main)

Add a fixed penalty on the per-shift enforcement **disparity** `δ_t` (the ratio of mean
enforcement-per-capita in the most-disadvantaged district quintile to the least-disadvantaged;
`δ = 1` is parity):

```
shaped reward_t  =  reward_t  −  w_f · δ_t
```

Train with ordinary PPO. **Sweeping the fixed weight `w_f`** from 0 upward gives a family of
policies from "concentrate freely" to "hold uniform" — and because a larger `w_f` monotonically
increases the pressure to reduce `δ`, the family traces the efficiency–equity frontier by
construction. No tuning, robust. This is the exterior-penalty method for constrained MDPs.

### 4.3 The fair policy — Lagrangian variant (reported for comparison)

The principled alternative: a constrained MDP `max E[Σ r_t] s.t. E[(1/T) Σ c_t] ≤ 0` with
`c_t = max(0, disparity_cum_t − α)`, solved by dual ascent on a multiplier `λ`
(`λ ← clip(λ + η(mean_cost − ε), 0, λ_max)`; Chow et al. 2017; Tessler et al. 2019; Stooke et al.
2020). In our runs it was sensitive to the interaction between the cost signal and the training
episode length, so the penalty method carries the headline frontier and the Lagrangian is a
robustness comparison. Both are in `crimesim/rl/constrained_ppo.py`.

### 4.3 Baselines

| Baseline | What it does | Why it's here |
|---|---|---|
| **Uniform** | patrol ∝ population | the fairness reference: `disparity = 1` by construction |
| **Random** | random split each shift | sanity floor |
| **PredPol (EWMA)** | patrol ∝ exponentially-weighted moving average of recorded crime | the exact heuristic Ensign et al. analyse; shows the loop in a non-RL system |
| **RandomForest** | our Phase-2 model, refit every 15 shifts on accumulated records, patrol ∝ predicted recorded crime | connects to our earlier work; a "smarter" predictive system |
| **Greedy Oracle** | patrol ∝ EWMA of *latent* crime | upper bound on how much a well-targeted policy can cut real crime; not deployable |

---

## Part 5 — The experiments and what each one shows

### E0 — calibration (`crimesim/abm/calibration.py`)
Confirms the fast simulator reproduces the Phase-2 model: hardship elasticity ~+27% (Phase-2:
+42.6%), night > day crime, emergent spatial clustering, monotone Low < Medium < High. **Gate:
nothing downstream is valid unless this passes.**

### E1 — recorded coverage up, latent coverage not
Roll out every policy on held-out seeds. Report, per policy: coverage of recorded crime, coverage
of latent crime (± 95% CI), enforcement disparity, Gini of enforcement.

**Observed (3 train × 10 eval seeds):** the naive PPO rule posts the **highest recorded coverage
of any method** (0.064 vs the heuristic's 0.059) — it "looks best" on the dashboard — while its
**latent coverage is the lowest of the targeted policies** (0.048 vs the heuristic's 0.055). The
learned rule parameters spell out why: β ≈ 3 (sharp concentration), **w > 0 (a positive hardship
tilt)**, μ ≈ 0. Cumulative enforcement Gini 0.46 vs the heuristics' 0.28. This is the core
"optimising the observable metric" result.

### E2 — the loop runs away
Track, shift by shift, averaged over seeds:
- **enforcement Gini** (cumulative deployment) — rises for loop-prone policies;
- **EMD(recorded, latent)** — how distorted the observed data has become;
- **EMD(deployment, latent)** — how far patrol has drifted from where crime actually is;
- **runaway index** = partial correlation of this shift's deployment with *cumulative past
  deployment*, controlling for *current latent crime*. A positive value means patrol is chasing
  its own history, not crime.

**Expected / observed:** for uniform, all flat. For PredPol and naive PPO, Gini and EMD climb and
the runaway index is clearly positive. For the fair PPO, they stay near the uniform lines.

### E3 — who pays
Build a panel with one row per (policy, seed, shift, district). For each row,
`over_policing = deployment_share − latent_share` (how much more patrol than the district's true
crime warrants). Regress:

```
over_policing  ~  hardship  +  latent_share   (+ poverty)
```

with heteroskedasticity-robust standard errors. **The hardship coefficient is the headline
number**: positive and significant means the policy over-polices disadvantaged districts *beyond
what their true crime justifies* — a discriminatory loop, not just "more crime there".

**Observed (3 train × 10 eval seeds):** naive PPO **+0.017**, p ≪ 10⁻³⁰⁰, R² 0.40 — about **3×
the PredPol baseline** (+0.005), 2× the RL oracle (+0.008). Penalised PPO goes negative
(−0.026 at the heaviest). Enforcement per capita by hardship quintile: naive PPO rises to
**~3× per capita in the most-disadvantaged quintile** (vs ~1.6× for the heuristics), the
penalised rule is flat.

### E4 — the fairness penalty and the frontier
Plot every policy in (disparity, latent coverage) space. Find the Pareto-efficient set and the
**knee** (efficient point closest to the ideal corner: low disparity, high coverage).

**Observed:** the naive PPO rule and the RL oracle are the **only Pareto-dominated policies** —
every heuristic and every penalised variant reaches both higher latent coverage AND lower
disparity. As the penalty weight `w_f` grows the penalised rule moves monotonically along the
frontier: disparity 8.1 → 1.5 → 0.9 → 0.6 → 0.5, hardship coefficient +0.017 → −0.026. The knee
(w_f = 0.02) removes the disparity for about one percentage point of latent coverage relative
to uniform.

### E5 — robustness (`scripts/run_ablations.py`)
Re-run the core comparison under each of: perfect observation, weak/strong deterrence, low/high
detection gain, no socioeconomic features in the state, small/large patrol budget, short/long
observation history. Report latent crime, disparity, and the E3 hardship coefficient for each.

**The falsification check:** with **perfect observation** (`p_report = 1`, detection gain 0),
patrol no longer inflates the record, so the loop should disappear — disparity and the hardship
coefficient should collapse toward zero. If they don't, our mechanism explanation is wrong.

---

## Part 6 — Reading the results

The story, stated carefully:

1. **The naive rule is not buggy — it is optimal for its objective.** Given "cover recorded
   crime", a sharp concentration on high-recorded districts is correct. The failure is that this
   objective and the social objective (cover real crime, spread enforcement in proportion to
   need) come apart *once the rule influences the record*.
2. **It games the observable metric.** It posts the best *recorded* coverage of any method while
   its *real* coverage is near-worst — measurable only because the unseeded simulator gives us
   ground truth. "Patrols high-crime districts" is not over-policing; "patrols them more than
   their true crime warrants" is, and you need latent crime to say that.
3. **The distortion is discriminatory.** It correlates with hardship after controlling for true
   crime, and the rule learns an *explicit* positive hardship tilt (w ≈ +1.3). This is the
   socially important finding.
4. **A cheap fix exists.** A swept penalty on the enforcement disparity removes the tilt along a
   clean monotone frontier; a light penalty gets to parity for a few points of coverage.
5. **None of it is a data-bias artefact.** The simulator injects no historical crime, so the
   distortion is a property of the *learned rule in a feedback loop*, full stop.

---

## Part 7 — Limitations (know these for the viva)

- **It is a simulation.** The numbers are not Chicago's crime totals. The value is counterfactual
  clarity — the same caveat applies to Ensign et al., FairSense, and the Baltimore study. E5
  sweeps the parameters that govern the loop's strength.
- **Deterrence is bounded and parametric**, so spatial allocation barely moves *total* crime.
  That is why the efficiency axis is *coverage*, not crimes prevented — and why an RL oracle
  (right objective + full observation) only *matches*, not beats, the reactive heuristic on
  coverage. This actually sharpens the point (the naive agent *loses* coverage by chasing the
  recorded signal), but a reviewer will ask; E5 varies the deterrence strength.
- **Residential population is assumed uniform across districts.** Real districts differ; we chose
  uniform so per-capita crime is emergent rather than assumed. A version with real district
  populations is a natural extension.
- **Offenders are rule-based.** A two-sided-learning version (offenders also adapt) is sketched
  as future work; there is prior art for RL offenders (Wise & Cheng 2024).
- **The grid is coarse** (20×20) and movement is zone-based, not road-network routing.

---

## Part 8 — Anticipated viva questions

**"Isn't the result obvious — you gave it a bad reward?"**
The reward is the standard operational criterion for patrol allocation (predictive accuracy /
coverage), used to evaluate real systems. The contribution is showing (a) *how much harder* an
RL optimiser presses on it than the PredPol heuristic — 3× the hardship coefficient, it learns
an explicit hardship tilt, and it is Pareto-dominated by every alternative — (b) that the damage
is discriminatory after controlling for true crime, which needs ground truth, and (c) that a
swept penalty fixes it along a clean frontier. "The deployable objective is misspecified" is the
finding, not a flaw in the setup.

**"Why not use real crime data?"**
Then you cannot measure over-policing — only drift from previously recorded crime, which is
already the biased quantity. Ground truth is the whole point.

**"Why police districts and not community areas?"**
A 20×20 grid cannot resolve 77 community areas, and districts are the actual patrol-allocation
unit. District covariates are aggregated from the community-area ACS data.

**"How is this different from the 2025 CEUS fairness-constrained RL patrol paper?"**
That paper uses RL as a fast *solver* for a one-shot facility-location problem with a static
coverage-disparity constraint, over XGBoost-predicted risk. No feedback loop, no sequential
decision-making, no latent ground truth. We study the loop dynamics over a deployment horizon and
constrain them.

**"Is the constrained RL novel?"**
The penalty and Lagrangian schemes are standard (we cite Chow, Tessler, Stooke). The novelty is
the setting: constraining the *feedback-loop dynamics* of a rule that shapes its own training
distribution, evaluated against known latent crime, and the monotone frontier over that
trade-off.

**"Why does the RL learn the parameters of a rule instead of a free allocation?"**
Two reasons. Practically, PPO cannot explore a 22-dim continuous simplex in this noise — even the
oracle version does no better than uniform. Substantively, deployed predictive-policing systems
*are* tuned rules (a recency signal, a concentration knob), so learning the knobs is the
faithful model. The free-allocation version is in the code as an ablation.

**"What would make this a stronger paper?"**
Real district populations, a two-sided-learning extension, and calibrating the detection/deterrence
parameters against an external estimate of the crime reporting rate. All are in future work.
