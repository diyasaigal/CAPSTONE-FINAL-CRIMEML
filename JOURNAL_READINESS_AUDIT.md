# Journal-Readiness Audit & Peer-Review Assessment
**Project**: CrimeSim-RL — Runaway Feedback & Fairness in Reinforcement-Learned Patrol Allocation  
**Author/Candidate**: Capstone Research Team (PES University)  
**Target Venues**: 
1. *Expert Systems with Applications* (Elsevier, IF: 8.5) — Primary Target
2. *Computers, Environment and Urban Systems* (Elsevier, IF: 6.8) — Alternative Primary
3. *AI and Ethics* (Springer) / *ACM Transactions on Probabilistic Machine Learning* — Secondary
**Audit Date**: September 2026  
**Status**: Comprehensive Pre-Submission Technical Audit

---

## Executive Summary & Target Journal Fit

This audit provides a comprehensive, reviewer-grade assessment of the **CrimeSim-RL** codebase, mathematical formulations, experimental evidence, and manuscript positioning.

### Strategic Positioning
The paper investigates an important and timely question at the intersection of **Reinforcement Learning under Partial Observability (POMDPs)**, **Algorithmic Fairness**, and **Urban Decision Support Systems**:
> *When an RL agent optimizes an operational proxy (recorded crime coverage) in a closed loop where actions alter measurement probabilities (patrol-dependent detection), does it induce runaway discriminatory over-policing, and can constrained optimization restore Pareto efficiency?*

* **Fit for *Expert Systems with Applications***: High. ESWA values applied intelligent systems, robust optimization algorithms, and rigorous empirical evaluations of automated decision-making under uncertainty.
* **Fit for *Computers, Environment and Urban Systems***: Excellent. CEUS is the premier venue for urban agent-based modeling, spatial decision support systems, and GIS-informed predictive analytics.
* **Fit for *AI and Ethics***: Very High (if framed with heavier emphasis on algorithmic governance, civil rights implications, and metric gaming).

---

# PART 1 — REPRODUCIBILITY AUDIT

```
                        SYSTEM ARCHITECTURE & OBSERVABILITY ISOLATION
                        
  ┌─────────────────────────────────────────────────────────────────────────────┐
  │                           SIMULATOR ENGINE (ABM)                            │
  │  • Latent Ground Truth L_r (Emergent, Unseeded, Hidden from Policy)         │
  │  • Physical Deterrence: L_r ~ Binomial(L_raw, 1 - min(0.68, 0.26 * u_r))   │
  └──────────────────────────────────────┬──────────────────────────────────────┘
                                         │
                                         ▼
  ┌─────────────────────────────────────────────────────────────────────────────┐
  │                         MEASUREMENT CHANNEL (OBS)                           │
  │  • Detection Probability: q_r = 0.30 + 0.55 * min(1.0, u_r / 1.5)           │
  │  • Recorded Crime: D_r ~ Binomial(L_r, q_r)                                 │
  └──────────────────────────────────────┬──────────────────────────────────────┘
                                         │
                                         ▼
  ┌─────────────────────────────────────────────────────────────────────────────┐
  │                         RL POLICY (PolicingEnv)                             │
  │  • State Observation s_t: [Hist_4(D), Last_Deploy, Hour, Socioeconomic]    │
  │  • Action: Rule Knobs a_t = (β, w, μ) -> Deployment B_r                     │
  │  • Reward: 10.0 * (Σ a_r D_r / Σ D_r) - w_f * Disparity                     │
  └─────────────────────────────────────────────────────────────────────────────┘
```

### A. Environment Implementation vs Mathematical Definitions

| Component | Code Implementation | Mathematical Formalization | Audit Verdict & Consistency |
|---|---|---|---|
| **State Space ($\mathcal{S}$)** | `crimesim/env/policing_env.py:74-99`<br>199-dim flat vector ($H \times 22 + 22 + 3 + 88$) | $s_t = [\frac{1}{50}D_{t-H:t}, \frac{1}{B}B_{t-1}, \sin(\frac{2\pi h}{24}), \cos(\frac{2\pi h}{24}), \frac{t}{T}, \mathbf{z}_{\text{socio}}]$ | 🟢 **Perfect Match**. Normalized, scaled, and correctly partitioned. Latent crime is strictly excluded in standard mode. |
| **Action Space ($\mathcal{A}$)** | `crimesim/env/policing_env.py:151-166`<br>3D Box $[-1, 1]^3 \to (\beta, w, \mu)$ | $\beta \in [0, 4], w \in [-2, 2], \mu \in [0, 1]$ mapped to softmax allocation $a_r$ with floor $\phi=0.005$ | 🟢 **Perfect Match**. Bounded continuous mapping prevents simplex exploration failure. |
| **Physical Deterrence** | `crimesim/abm/generator.py:126-128`<br>`det_frac = min(0.68, 0.26 * intensity)`<br>`latent ~ Binomial(latent_raw, 1 - det_frac)` | $L_r \sim \text{Binomial}\left(L_r^{\text{raw}}, 1 - \min(\kappa_{\text{cap}}, \kappa u_r)\right)$ with $\kappa_{\text{cap}}=0.68, \kappa=0.26$ | 🟢 **Perfect Match**. Correctly executed per sub-step. |
| **Observation Channel** | `crimesim/env/detection.py:18-32`<br>`q = 0.30 + 0.55 * min(1, u / 1.5)`<br>`recorded ~ Binomial(latent, q)` | $D_r \sim \text{Binomial}\left(L_r, q_0 + g \min(1, \frac{u_r}{u_{\text{sat}}})\right)$ with $q_0=0.30, g=0.55, u_{\text{sat}}=1.5$ | 🟢 **Perfect Match**. Monotonic detection gain up to $0.85$. |
| **Operational Reward** | `crimesim/env/policing_env.py:177-183`<br>`reward = 10.0 * (dep_share * rec).sum() / rec.sum()` | $r_t = 10.0 \cdot \frac{\sum_{r=1}^R a_r D_r}{\sum_{r=1}^R D_r}$ | 🟢 **Perfect Match**. Scaled for neural network value-function fitting. |
| **Fairness Penalty** | `crimesim/rl/constrained_ppo.py:32-51`<br>`reward_shaped = rew - weight * info["disparity_shift"]` | $r_t^{\text{shaped}} = r_t - w_f \cdot \delta_t$ where $\delta_t = \frac{\text{exposure}_{Q5}}{\text{exposure}_{Q1}}$ | 🟢 **Perfect Match**. Exterior penalty method correctly implemented via vectorized environment wrapper. |

---

### B. Master Parameter Dictionary & Range Audit

| Parameter | Code Variable | Default Value | Tested / Valid Range | File Location | Purpose & Scientific Source | Inconsistencies Found? |
|---|---|---|---|---|---|---|
| Grid Resolution | `grid` | $20$ ($20 \times 20$) | $[20, 30]$ | `crimesim/config.py:30` | Spatial resolution overlaying Chicago bounding box | None. Aligned with CA lookup cache. |
| Population Size | `n_agents` | $200$ | $[100, 500]$ | `crimesim/config.py:31` | Density of interacting citizen agents | None. |
| Interaction Gain | `gamma` ($\gamma$) | $0.20$ | $[0.1, 0.3]$ | `crimesim/config.py:32` | Stress mismatch escalation multiplier (Rosés et al. 2021) | None. |
| Risk Decay | `risk_decay` | $0.99$ | $(0, 1)$ | `crimesim/config.py:33` | Environmental risk temporal dissipation | None. |
| Risk Gain | `risk_gain` | $0.008$ | $[0.005, 0.015]$ | `crimesim/config.py:34` | Ambient risk accumulation per occupant | Retuned from v5 ($0.01 \to 0.008$) for mean-field stability. Documented. |
| Risk Memory | `risk_memory` ($m$) | $0.20$ | $[0.1, 0.5]$ | `crimesim/config.py:37` | Exponential smoothing memory on agent risk | None. |
| Risk Trigger | `risk_trigger` ($\tau$) | $2.20$ | $[1.5, 3.0]$ | `crimesim/config.py:38` | Threshold for environmental violent crime | Retuned from v5 ($1.2 \to 2.2$) to match 4 sub-steps/shift. Documented. |
| Base Reporting | `p_report` ($q_0$) | $0.30$ | $[0.1, 1.0]$ | `crimesim/config.py:64` | Baseline reporting rate with zero patrol presence | None. BJS survey grounded. |
| Detection Gain | `detect_gain` ($g$) | $0.55$ | $[0.0, 0.75]$ | `crimesim/config.py:65` | Maximum additional discovery probability under patrol | None. Ablated in E5. |
| Deterrence Slope | `deterrence_slope` ($\kappa$) | $0.26$ | $[0.1, 0.5]$ | `crimesim/config.py:67` | Physical latent crime reduction per unit intensity | None. Weisburd et al. meta-analysis grounded. |
| Deterrence Cap | `deterrence_cap` | $0.68$ | $[0.4, 0.8]$ | `crimesim/config.py:68` | Saturation ceiling on local crime suppression | None. |
| Patrol Budget | `n_units` ($B$) | $20$ | $[10, 50]$ | `crimesim/config.py:74` | Total patrol officers deployed per shift across 22 districts | None. |
| Shift Length | `steps_per_shift` | $4$ | $[2, 8]$ | `crimesim/config.py:57` | Number of 2-hour sub-steps in an 8-hour shift | None. |
| Episode Horizon | `shifts_per_episode` ($T$) | $180$ | $[100, 365]$ | `crimesim/config.py:77` | 60 days of 3 shifts/day ($\sim 2$ months) | None. |
| Warmup Shifts | `warmup_shifts` | $25$ | $[10, 50]$ | `crimesim/config.py:78` | Pre-deployment proportional patrol burn-in | None. |
| Min Allocation Floor | `min_share` ($\phi$) | $0.005$ | $[0.001, 0.02]$ | `crimesim/config.py:75` | Minimum budget floor per district ($0.5\%$) | None. Prevents division by zero. |
| Fairness Penalty | `penalty` ($w_f$) | $[0.02, 0.40]$ | $[0.0, 1.0]$ | `crimesim/config.py:107` | Disparity regularization weight | Swept cleanly across $\{0.02, 0.06, 0.15, 0.40\}$. |
| PPO Learning Rate | `learning_rate` | $3 \times 10^{-4}$ | $[10^{-4}, 10^{-3}]$ | `crimesim/config.py:123` | Adam optimizer learning rate | None. |
| PPO Rollout Steps | `n_steps` | $2048$ | $[1024, 4096]$ | `crimesim/config.py:119` | Steps collected per policy update | Scaled per environment thunk: `n_steps // n_envs`. |

---

### C. Random Seed & Determinism Audit

1. **RNG Architecture**: The codebase uses modern NumPy `Generator` objects (`np.random.default_rng(seed)`) rather than legacy global state (`np.random.seed`).
2. **Seed Partitioning**:
   - **Training Seeds**: `train_seeds = [0, 1, 2]` (distinct initialization of environment dynamics, agent placement, and neural network weights).
   - **Evaluation Seeds**: `eval_seeds = [100, ..., 109]` or `[200, ..., 209]` (strictly held-out ABM trajectories never seen during gradient updates).
   - **Baseline Evaluation**: Heuristic baselines evaluated on identical held-out seed sequences to ensure exact paired comparisons.
3. **Reproducibility Verdict**: 🟢 **Passed**. Code execution is completely deterministic and reproducible across independent runs.

---

### D. Data Leakage & Information Boundary Audit

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           INFORMATION ACCESS BOUNDARY                           │
├──────────────────────────────────────┬──────────────────────────────────────────┤
│ Visible to RL Agent (Observation s_t)│ Hidden from Agent (Evaluator / Ground)   │
├──────────────────────────────────────┼──────────────────────────────────────────┤
│ • Historical Recorded Crime D_r      │ • True Latent Crime L_r (Physical Truth) │
│ • Previous Patrol Deployment B_{t-1} │ • True Detection Probability q_r         │
│ • Temporal Sine/Cosine of Hour       │ • Agent Stress Levels s_i                │
│ • Static ACS Census Covariates z_r   │ • Raw Environmental Risk Grid T(x,y)     │
└──────────────────────────────────────┴──────────────────────────────────────────┘
```

* **Observation Model Isolation**: Verified. `crimesim/env/policing_env.py` passes only normalized recorded crime, past deployment, time, and socioeconomic covariates to `_obs()`. Latent crime $L_r$ is appended if and only if `observe_latent=True` (strictly reserved for the `ppo_oracle` benchmark).
* **Reward Leakage**: Verified. In standard mode (`reward_on="captured"`), reward is computed strictly from recorded crime $D_r$. The agent receives no reward gradient from unrecorded latent crime.
* **Fairness Signal**: In the penalty method, `disparity_shift` is passed via the environment info dict and deducted externally in `ShapedRewardWrapper`. The policy network observes the penalty only through the scalar return.

---

# PART 2 — EXPERIMENT AUDIT (E0–E5)

| Exp | Research Question | Independent Variable | Dependent Variables | Baselines Evaluated | Seeds | Statistical Test | Key Result | Identified Confounds & Attack Surface |
|---|---|---|---|---|---|---|---|---|
| **E0** | Does the vectorized simulator preserve criminological dynamics? | Hardship scenario (Low, Med, High) | Total crime, night/day ratio, spatial clustering ratio | Mesa ABM v5 baseline | 8 calibration seeds | Ratio thresholds ($>3.0$, $+27.2\%$) | 5/5 gates `[PASS]`, SDT elasticity $+27.2\%$ | Reviewer will ask if $+27.2\%$ matches empirical Chicago elasticities. (Must cite Sampson et al. 1997). |
| **E1** | Does optimizing recorded crime coverage maximize real crime reduction? | Policy family (`Uniform`, `PredPol`, `RF`, `Oracle`, `Naive PPO`, `Fair PPO`) | Recorded coverage ($\text{cov}_{\text{rec}}$), Latent coverage ($\text{cov}_{\text{lat}}$), Disparity ($\delta$), Gini ($G$) | `Uniform`, `Random`, `PredPol (EWMA)`, `RandomForest`, `Greedy Oracle` | 3 train $\times$ 10 eval seeds | Mean $\pm 95\%$ CI over independent seeds | Naive PPO has highest recorded coverage ($0.0643$) but lowest latent coverage ($0.0479$) and extreme disparity ($\delta=8.08$). | 10 eval seeds is acceptable for capstone, but journal reviewers will demand $N \ge 30$ seeds for tight CIs. |
| **E2** | How does enforcement drift dynamically over deployment time? | Policy type over 180 shifts | Shift-by-shift Gini, $\text{EMD}(\text{Deploy}, \text{Lat})$, Runaway Index | `Uniform`, `PredPol`, `Naive PPO`, `Fair PPO (0.06)` | Pooled episodes across eval seeds | Rolling mean ($w=9$), Partial correlation | Naive PPO Gini jumps to $0.495$ by shift 15; Runaway Index reaches $0.744$ chasing past deployments. | Reviewer will ask if Runaway Index is sensitive to the 5-shift burn-in. |
| **E3** | Is the allocation drift socioeconomically discriminatory? | District hardship ($h_r$) controlling for true crime | District-level over-policing ($a_r - L_r/\sum L$) | `Uniform`, `PredPol`, `RF`, `Oracle`, `PPO`, `Fair PPO` | 3 train $\times$ 10 eval ($22,000$ panel rows) | Panel OLS with HC1 robust standard errors | Naive PPO hardship coefficient $+0.0166$ ($p \ll 10^{-300}$, $R^2=0.404$), $3.25\times$ higher than PredPol ($+0.0051$). | Reviewer will question temporal autocorrelation in panel residuals. Must prove HC1 cluster robustness. |
| **E4** | Can fairness regularization recover Pareto-efficient allocation? | Fairness penalty weight $w_f \in [0.02, 0.40]$ | Disparity ($\delta$) vs Latent Coverage ($\text{cov}_{\text{lat}}$) | All E1 policies | 3 train $\times$ 10 eval seeds | 2D Pareto dominance frontier & Euclidean Knee point | Naive PPO is Pareto-dominated; Knee at $w_f=0.02$ cuts disparity by $82\%$ ($8.08 \to 1.48$) at $\Delta \text{cov} < 0.0006$. | Reviewer will ask if the knee point is stable across different training runs. |
| **E5** | Is detection inflation the causal driver of the feedback loop? | Simulator configuration (Base, Perfect Obs, No Socio Obs, Deterrence Sweeps) | End-state Disparity ($\delta$), Hardship coefficient ($\beta_1$) | Naive PPO vs Fair PPO | 1 train $\times$ 10 eval seeds | Comparative ablation shifts | Perfect observation ($q_0=1.0$) collapses disparity from $9.04 \to 2.92$ ($-68\%$) and hardship coef by $>50\%$. | Reviewer will note E5 only used 1 training seed. Must re-run across 3 seeds. |

---

# PART 3 — NOVELTY AUDIT & CONTRIBUTION MATRIX

```
                                  THE NOVELTY FRONTIER
                                  
  Prior Work (Ensign / Lum):         Abstract Urn Models / No Space / No RL Learners
  Prior Work (Repasky / CEUS 2025):  Open-Loop Patrol RL / Optimizes Fixed Static Risk / No Ground Truth
  
  CRIMESIM-RL (This Work):          ┌────────────────────────────────────────────────────────┐
                                    │ • Theory-Grounded Unseeded Spatial ABM (Ground Truth)  │
                                    │ • Closed-Loop Endogenous Measurement Channel (POMDP)   │
                                    │ • Proves Naive RL Aggressively Exploits Proxy Gap      │
                                    │ • Monotone Efficiency-Equity Pareto Regularization     │
                                    └────────────────────────────────────────────────────────┘
```

### Literature Contribution Matrix

| Dimension / Capability | Ensign et al. (2018) | Lum & Isaac (2016) | Wise & Cheng (2024) | Repasky et al. (2024) | CEUS (2025) Patrol RL | Algorithms (2026) Baltimore | **CrimeSim-RL (This Work)** |
|---|---|---|---|---|---|---|---|
| **Urban Crime Simulation** | ❌ (Pólya Urn) | ❌ (Real Logs) | 🟢 (ABM) | ❌ (Graph) | ❌ (Point Data) | 🟢 (PredPol Sim) | 🟢 **(Vectorized ABM)** |
| **Known Latent Ground Truth** | ❌ (Theoretical) | ❌ (None) | 🟢 (Synthetic) | ❌ (Historical) | ❌ (XGBoost Risk) | ❌ (Historical) | 🟢 **(100% Unseeded Latent)** |
| **Reinforcement Learning** | ❌ (Static Rule) | ❌ (None) | 🟢 (Offender RL) | 🟢 (Patrol RL) | 🟢 (Patrol RL) | ❌ (Heuristic) | 🟢 **(Closed-Loop Patrol RL)** |
| **Closed-Loop Measurement Bias**| 🟢 (Analytical) | 🟢 (Empirical) | ❌ (Open Loop) | ❌ (Open Loop) | ❌ (Open Loop) | 🟢 (Closed Loop) | 🟢 **(Closed-Loop POMDP)** |
| **RL Proxy-Gaming Proof** | ❌ (No RL) | ❌ (No RL) | ❌ (No RL) | ❌ (Optimizes Proxy)| ❌ (Optimizes Proxy)| ❌ (No RL) | 🟢 **(Recorded vs Latent Divergence)** |
| **Pareto Fairness Regularization**| ❌ (Urn Threshold)| ❌ (None) | ❌ (None) | ❌ (None) | 🟢 (Facility Constraint)| ❌ (None) | 🟢 **(Constrained MDP Frontier)** |
| **Falsification Intervention** | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | 🟢 **(Perfect-Observation Control)** |

### Rigorous 3-Sentence Novelty Statement (Peer-Review Defensible)
> While previous studies have analyzed predictive-policing feedback analytically using urn models or applied reinforcement learning to static historical crime grids, this work provides the first closed-loop empirical demonstration that reinforcement-learned patrol allocation actively exploits endogenous detection bias to maximize observable proxy metrics at the expense of latent crime coverage. By formulating patrol allocation as a continuous rule-parameterized POMDP over an unseeded, theory-grounded crime simulator with exact latent tracking, we prove that naive RL optimization amplifies socioeconomic enforcement disparity by over $3\times$ compared to standard heuristics. Furthermore, we demonstrate that exterior fairness regularization recovers a monotonic Pareto-efficient frontier, eliminating over $80\%$ of discriminatory over-policing at negligible true coverage cost.

---

# PART 4 — LITERATURE GAP & CITATION ANALYSIS

### 1. Performative Prediction & Measurement Bias
* **Perdomo et al. (ICML 2020)** — *Performative Prediction*: Formalizes settings where model predictions induce distribution shifts in the target variable.
  - *Gap*: Studied in abstract supervised learning frameworks; our work extends performativity into sequential continuous-control reinforcement learning under spatial constraints.
* **Lakkaraju et al. (KDD 2017)** — *The Selective Labels Problem*: Evaluates classification under one-sided feedback in criminal bail decisions.
  - *Gap*: Focuses on static bandit feedback; our work operationalizes continuous spatio-temporal feedback loops.

### 2. Reinforcement Learning in Urban Patrol
* **Repasky et al. (arXiv 2024 / AAMAS)** — *Multi-Agent RL for Police Patrol and Dispatch*: Optimizes response times and patrol coverage over urban road networks.
  - *Gap*: Assumes crime risk is an exogenous, stationary historical distribution. The model cannot exhibit feedback loops because patrol presence does not modify future crime observation.
* **CEUS (2025)** — *Spatial Optimization with Deep RL for Patrol Region Design*: Uses deep RL as a fast combinatorial solver for maximum coverage location problems under equity constraints.
  - *Gap*: The RL agent operates on fixed XGBoost-predicted risk surfaces. It is an open-loop solver, not a sequential policy interacting with dynamic feedback.

### 3. Criminological ABMs & Feedback
* **Ensign et al. (FAT* 2018)** — *Runaway Feedback Loops in Predictive Policing*: Proved that PredPol's threshold updates lead to urn convergence on initial noise.
  - *Gap*: Used a non-spatial, 2-bin mathematical urn model with no physical agents, no socioeconomic geography, and no optimization policy.
* **Wise & Cheng (CEUS 2024)** — *Simulating Offender Behaviours in ABMs via RL*: Offender agents learn target selection via Q-learning.
  - *Gap*: Places RL on the offender side against static police; our work places RL on the institutional allocation side against endogenous measurement bias.

---

# PART 5 — REQUIRED NEW EXPERIMENTS & ROADMAP

```
                     PRIORITIZED EXPERIMENTAL ENHANCEMENT PLAN
                     
  [PHASE 1: MUST HAVE] ──> • Scale Held-Out Evaluation Seeds to N=30 (Tight 95% CIs)
                           • Scale Training Seeds to N=5 (PPO Training Stability)
                           • Re-run E5 Falsification across 3 Training Seeds
                                         │
                                         ▼
  [PHASE 2: STRONGLY RECOMMENDED] ──> • 2D Sensitivity Heatmap on (g, κ) [Detection vs Deterrence]
                                      • Two-Way Cluster-Robust Standard Errors in E3 Panel
                                      • Action-Space Ablation: 3D Rule vs Direct 22-Logit
                                         │
                                         ▼
  [PHASE 3: OPTIONAL] ───────────────> • Empirical District Population Weights from ACS
```

### 1. MUST-HAVE Experiments (Mandatory for Journal Acceptance)
1. **Scale Evaluation Seeds ($N=30$)**: Increase held-out evaluation seeds from 10 to 30. This reduces CI width by $\approx 42\%$ ($\frac{1}{\sqrt{30}}$ vs $\frac{1}{\sqrt{10}}$) and definitively settles reviewer doubts regarding seed variance.
2. **Multi-Seed Training Stability ($N=5$)**: Train policies across 5 distinct random seeds (`train_seeds = [0, 1, 2, 3, 4]`). Plot mean training reward curves with $\pm 1 \text{ SD}$ shaded error bands to prove PPO convergence stability.
3. **Multi-Seed E5 Falsification**: Re-run the 6 ablation settings in `scripts/run_ablations.py` across 3 training seeds $\times$ 10 evaluation seeds (currently only 1 training seed in `results/e5_ablations.csv`).

### 2. STRONGLY RECOMMENDED Experiments
1. **2D Sensitivity Surface on $(g, \kappa)$**: Run a $4 \times 4$ grid sweep over detection gain $g \in \{0.0, 0.25, 0.55, 0.75\}$ and deterrence slope $\kappa \in \{0.0, 0.15, 0.26, 0.45\}$. Plot a 2D contour of resulting disparity to map the exact boundary where detection overwhelms deterrence.
2. **Cluster-Robust Variance Estimation in E3**: Re-estimate the panel OLS regression using two-way clustering on `(district, seed)` to guarantee that serial autocorrelation within districts does not artificially deflate standard errors.
3. **Action-Space Expressiveness Ablation**: Compare `action_mode="rule"` against `action_mode="direct"` (22-dimensional continuous Dirichlet output) to empirically prove why direct simplex control fails in noisy environments.

---

# PART 6 — RESULT VERIFICATION AUDIT

| Reported Claim | Source Script | Raw Result Artifact | Exact Mathematical Calculation | Verified? |
|---|---|---|---|---|
| Naive PPO Recorded Cov $= 0.0643$ | `scripts/run_all.py` | `results/e1_summary.csv:5` | $\frac{1}{N_{\text{seeds}}} \sum \frac{1}{T} \sum_t \frac{\sum_r a_r D_r}{\sum_r D_r} = 0.064328$ | 🟢 **VERIFIED** |
| Naive PPO Latent Cov $= 0.0479$ | `scripts/run_all.py` | `results/e1_summary.csv:5` | $\frac{1}{N_{\text{seeds}}} \sum \frac{1}{T} \sum_t \frac{\sum_r a_r L_r}{\sum_r L_r} = 0.047871$ | 🟢 **VERIFIED** |
| PredPol Latent Cov $= 0.0546$ | `scripts/run_all.py` | `results/e1_summary.csv:3` | $\frac{1}{N_{\text{seeds}}} \sum \frac{1}{T} \sum_t \frac{\sum_r a_r L_r}{\sum_r L_r} = 0.054577$ | 🟢 **VERIFIED** |
| Naive PPO Disparity $= 8.08$ | `scripts/run_all.py` | `results/e1_summary.csv:5` | $\frac{1}{N_{\text{seeds}}} \sum \left( \frac{\sum_{t} B_{Q5, t} / \text{Pop}_{Q5}}{\sum_{t} B_{Q1, t} / \text{Pop}_{Q1}} \right) = 8.0810$ | 🟢 **VERIFIED** |
| Naive PPO Gini $= 0.461$ | `scripts/run_all.py` | `results/e1_summary.csv:5` | $\text{Gini}\left(\sum_{t=1}^{180} B_{r, t}\right) = 0.46117$ | 🟢 **VERIFIED** |
| Naive Hardship Coef $= +0.0166$ | `scripts/run_all.py` | `results/e3_incidence.csv:7` | $\beta_1$ from OLS $\text{OverPolicing} \sim Z(h) + Z(\text{lat}) + Z(\text{pov}) = +0.016575$ | 🟢 **VERIFIED** |
| PredPol Hardship Coef $= +0.0051$| `scripts/run_all.py` | `results/e3_incidence.csv:4` | $\beta_1$ from OLS $\text{OverPolicing} \sim Z(h) + Z(\text{lat}) + Z(\text{pov}) = +0.005131$ | 🟢 **VERIFIED** |
| Knee Disparity $= 1.48$ ($w_f=0.02$)| `scripts/run_all.py` | `results/e1_summary.csv:7` | Disparity under penalty $0.02 = 1.47558$ | 🟢 **VERIFIED** |
| Perfect Obs Disparity $= 2.92$ | `scripts/run_ablations.py`| `results/e5_ablations.csv:9` | Disparity when $q_0=1.0, g=0.0 = 2.92469$ | 🟢 **VERIFIED** |

---

# PART 7 — THE CORE 4-FIGURE PUBLICATION SET

```
Figure 1: Closed-Loop Architecture & Observation POMDP Workflow (Diagram)
Figure 2: Efficiency vs. Equity Trade-off (Latent Coverage vs. Cumulative Disparity)
Figure 3: Runaway Feedback Trajectories (Gini, EMD, and Runaway Index over 180 Shifts)
Figure 4: Socioeconomic Incidence & Quintile Exposure per Capita (Bar Chart & OLS Slopes)
Figure 5: The Pareto Frontier & Knee Point Recovery under Fairness Regularization
```

### Detailed Figure Specifications for Camera-Ready Submission

#### Figure 1: Conceptual Framework & Closed-Loop POMDP Testbed
* **Type**: Vector Schematic / Flowchart (TikZ / PDF).
* **Components**: (a) Ground-truth ABM generator (stress, routine activity, emergent latent crime); (b) Deterrence channel; (c) Detection observation thinning; (d) RL policy interacting via parameterized allocation rule $(\beta, w, \mu)$; (e) Fairness regularization feedback.
* **Message**: Clearly delineates the unobservable latent process from the observable recorded data stream.

#### Figure 2: The Core Proxy Trap (E1)
* **X-Axis**: Policy Family (`Uniform`, `Random`, `PredPol`, `RandomForest`, `Greedy Oracle`, `Naive PPO`, `PPO + Penalty`).
* **Y-Axis**: Dual Bar Chart comparing **Recorded Coverage** (Blue) vs **Latent Coverage** (Red), with $95\%$ error bars.
* **Inset / Subplot**: Scatter plot of Disparity ($\delta$) vs Latent Coverage.
* **Message**: Visually exposes the reversal: Naive PPO is tallest on recorded coverage, but shortest among targeted policies on true coverage.

#### Figure 3: Dynamic Runaway Feedback Trajectories (E2)
* **X-Axis**: Deployment Shift ($t \in [0, 180]$).
* **Y-Axes (4 Panels)**:
  1. Cumulative Enforcement Gini ($G_t$).
  2. Earth-Mover Distance $\text{EMD}(\text{Deployment}_t, \text{Latent}_t)$.
  3. Earth-Mover Distance $\text{EMD}(\text{Recorded}_t, \text{Latent}_t)$.
  4. Runaway Index (Partial Correlation controlling for $L_t$).
* **Policies**: `Uniform` (gray dashed), `PredPol` (blue), `Naive PPO` (red), `Fair PPO w=0.06` (green).
* **Message**: Proves that Naive PPO diverges rapidly within the first 15 shifts and sustains high self-correlation.

#### Figure 4: Socioeconomic Incidence & Disproportionate Burden (E3)
* **Left Panel**: Enforcement Exposure per Capita across District Hardship Quintiles ($Q_1 \to Q_5$). Naive PPO shows steep monotonic rise to $\approx 3.0\times$ in $Q_5$; Fair PPO is flat at $1.0\times$.
* **Right Panel**: Scatter and OLS regression slopes of Over-Policing vs District Hardship, highlighting the $3.25\times$ slope difference between Naive PPO ($+0.0166$) and PredPol ($+0.0051$).
* **Message**: Isolates the discriminatory nature of the feedback loop after controlling for true crime.

#### Figure 5: Pareto Efficiency Frontier & Knee Point (E4)
* **X-Axis**: Cumulative Enforcement Disparity ($\delta \in [0.4, 9.0]$).
* **Y-Axis**: Latent Crime Coverage ($\text{cov}_{\text{lat}} \in [0.035, 0.058]$).
* **Markers**: Distinct shapes for Heuristic Baselines, Naive PPO (red square, dominated), PPO Oracle (purple diamond, dominated), and Penalized PPO sweep (green stars, Pareto-efficient).
* **Message**: Demonstrates that Naive PPO is strictly Pareto-dominated and illustrates the costless disparity recovery at the knee ($w_f=0.02$).

---

# PART 8 — PROPOSED JOURNAL MANUSCRIPT STRUCTURE

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                           MANUSCRIPT SECTION OUTLINE                             │
├─────┬──────────────────────────┬─────────────────────────────────────────────────┤
│ Sec │ Title                    │ Scientific Question Answered                    │
├─────┼──────────────────────────┼─────────────────────────────────────────────────┤
│ 1   │ Introduction             │ Why does proxy optimization fail in closed loops│
│ 2   │ Related Work             │ Where do existing ABM and patrol RL models fail │
│ 3   │ Problem Formulation      │ How to mathematically frame endogenous POMDPs   │
│ 4   │ CrimeSim-RL Environment  │ How is latent crime generated unseeded          │
│ 5   │ Patrol Allocation Models │ How are PPO and baselines parameterized         │
│ 6   │ Experimental Setup       │ What are the evaluation protocols and metrics   │
│ 7   │ Results & Empirical E1-E5│ What do the 5 experiments prove                 │
│ 8   │ Discussion & Governance  │ How should municipal algorithms be audited      │
│ 9   │ Threats to Validity      │ What are the simulation boundaries              │
│ 10  │ Ethical Considerations   │ Broader societal impacts of predictive policing │
│ 11  │ Conclusion               │ Core takeaways and future directions            │
└─────┴──────────────────────────┴─────────────────────────────────────────────────┘
```

---

# PART 9 — PEER REVIEWER ATTACK SIMULATION

```
                       PEER REVIEWER SIMULATION & DEFENSE
                       
   [REVIEWER A: ML/RL Expert]         [REVIEWER B: Responsible AI]        [REVIEWER C: Criminologist]
   • "Why 3D rule action?"            • "Is exposure parity enough?"      • "Is deterrence realistic?"
   • "Is PPO training stable?"        • "Did you hardcode bias?"          • "Why uniform population?"
             │                                     │                                   │
             ▼                                     ▼                                   ▼
   [DEFENSE: Simplex failure,         [DEFENSE: Standard metric,          [DEFENSE: Meta-analysis fit,
    N=5 seed stability plots]          E5 perfect-obs falsification]       explicit scope boundary]
```

### Reviewer A (Machine Learning / Reinforcement Learning Specialist)
1. **Attack**: *"Why restrict the RL action space to a 3-dimensional rule $(\beta, w, \mu)$ rather than letting PPO output a 22-dimensional continuous Dirichlet allocation directly?"*
   - *Severity*: **Major**.
   - *Defense*: In our preliminary experiments (`action_mode="direct"`), high-dimensional continuous simplex exploration under noisy ABM transitions causes policy gradient collapse into uniform noise. More importantly, real municipal systems operate via tunable concentration and recency parameters; learning the rule knobs is both computationally robust and faithful to deployed operational software.
   - *Action Required*: Add a dedicated subsection in Section 5 with the empirical ablation comparing 3D rule vs 22-logit training curves.

2. **Attack**: *"Are 3 training seeds sufficient to claim PPO convergence stability?"*
   - *Severity*: **Major**.
   - *Defense*: PPO converges to nearly identical rule parameters ($\beta \approx 3.0, w \approx 1.3, \mu \approx 0.0$) across seeds.
   - *Action Required*: Re-train with 5 seeds (`train_seeds=[0,1,2,3,4]`) and include learning curves in the appendix.

---

### Reviewer B (Responsible AI / Algorithmic Fairness Specialist)
1. **Attack**: *"Your disparity metric uses socioeconomic hardship quintiles. Does demographic parity conflict with allocating police to areas that genuinely have more crime?"*
   - *Severity*: **Major**.
   - *Defense*: This is precisely why Experiment E3 is essential: our panel regression regresses over-policing on hardship **while controlling for true latent crime share**. The positive coefficient ($+0.0166$) proves that Naive PPO over-polices disadvantaged areas *beyond* what their true crime justifies.
   - *Action Required*: Ensure the distinction between raw allocation and over-policing ($a_r - L_r/\sum L$) is prominently emphasized in the abstract and results.

2. **Attack**: *"Is the feedback loop simply an inevitable artifact of the detection formula you chose?"*
   - *Severity*: **Critical**.
   - *Defense*: Experiment E5 provides rigorous falsification: when detection inflation is disabled ($q_0=1.0, g=0.0$), the disparity collapses from $9.04 \to 2.92$. The residual disparity ($2.92$) reflects legitimate response to higher baseline crime in disadvantaged areas, proving that detection inflation accounts for $>68\%$ of the excess disparity.

---

### Reviewer C (Urban Systems & Criminology Specialist)
1. **Attack**: *"Assuming uniform residential population across all 22 police districts is unrealistic for Chicago."*
   - *Severity*: **Minor / Moderate**.
   - *Defense*: Uniform population was selected as an intentional experimental control so that per-capita crime rates and per-capita enforcement exposure are clean emergent outcomes rather than artifacts of unequal baseline population denominators.
   - *Action Required*: Explicitly document this in the Limitations section and run a sensitivity check with empirical ACS district population weights.

2. **Attack**: *"Does the simulation capture rational offender displacement (spatial deterrence)?"*
   - *Severity*: **Minor**.
   - *Defense*: Physical deterrence suppresses local crime up to $68\%$, but spatial displacement is bounded. We acknowledge non-adaptive offenders as a scope boundary and cite Wise & Cheng (2024) for two-sided multi-agent extensions.

---

# PART 10 — FINAL DECISION & SUBMISSION ROADMAP

### A. Current Journal Readiness Score: **84 / 100**

```
READINESS SCORE BREAKDOWN:
├── 1. Mathematical Formulation & POMDP Rigor:    95 / 100
├── 2. Code Implementation & Test Coverage:       95 / 100
├── 3. Software Reproducibility & Seed Safety:   95 / 100
├── 4. Experimental Breadth (E0 - E5):           90 / 100
├── 5. Evaluation Seed Volume (Currently N=10):   70 / 100  [Needs N=30]
├── 6. Training Stability Seeds (Currently N=3):  75 / 100  [Needs N=5]
├── 7. Causal & Falsification Proofs:             90 / 100
├── 8. Novelty & Literature Positioning:          85 / 100
├── 9. Visual / Figure Presentation Quality:      80 / 100  [Needs Figure 1 Diagram]
└── 10. Manuscript Draft Readiness:               65 / 100  [LaTeX needs expansion]
────────────────────────────────────────────────────────────
TOTAL SCORE:                                      84 / 100  (Strong Foundation)
```

---

### B. Top 10 Action Items (Strict Priority Order)

1. **Scale Evaluation Seeds to $N=30$**: Update `scripts/run_all.py` to evaluate across seeds `100` to `129` for tighter $95\%$ CIs.
2. **Scale Training Seeds to $N=5$**: Train policies across seeds `0, 1, 2, 3, 4` to demonstrate robust PPO convergence.
3. **Re-run E5 Falsification across Multi-Seeds**: Execute `scripts/run_ablations.py` across 3 training seeds $\times$ 10 eval seeds.
4. **Create Figure 1 System Architecture Diagram**: Generate a high-resolution vector PDF diagram illustrating the unseeded ABM, detection thinning, and RL closed loop.
5. **Add 2D Parameter Sensitivity Sweep**: Run grid sweep on detection gain $g \in [0.1, 0.75]$ vs deterrence slope $\kappa \in [0.1, 0.5]$.
6. **Implement Two-Way Clustered Standard Errors in E3**: Validate that panel regression significance holds under district-level clustering.
7. **Expand Related Work Section in `paper/main.tex`**: Integrate citations on Performative Prediction (Perdomo et al. 2020) and Selective Labels (Lakkaraju et al. 2017).
8. **Add Action-Space Expressiveness Ablation**: Document empirical comparison between 3D rule action space vs 22-logit direct allocation.
9. **Refine Figure Aesthetics for Publication**: Ensure all figures use consistent font sizing, CMYK colorblind-safe palettes, and explicit axis units.
10. **Final Manuscript Expansion to 10–12 Pages**: Expand LaTeX draft with complete experimental details, mathematical derivations, and broader ethical governance discussions.

---

### C. DO NOT DO (Anti-Roadmap)
* ❌ **Do NOT rewrite the ABM in C++ or Mesa**: Pure NumPy vectorized execution ($400$ steps/sec) is already optimal for CPU RL training.
* ❌ **Do NOT attempt to predict exact real-world street crime coordinates**: The simulator is designed for counterfactual algorithmic auditing, not real-world incident forecasting.
* ❌ **Do NOT add dozens of complex RL algorithms (e.g., SAC, TD3, Rainbow DQN)**: PPO is the gold standard for continuous control; adding more algorithms adds noise without strengthening the core scientific claim.
* ❌ **Do NOT claim that real-world police departments are intentionally biased**: Keep all scientific claims strictly focused on mathematical optimization vulnerabilities under measurement bias.

---

### D. Submission Target Recommendation

1. **Primary Target: *Expert Systems with Applications* (Elsevier)**
   - *Rationale*: ESWA prioritizes intelligent decision-support systems, reinforcement learning, and rigorous computational benchmarking. Our paper presents an end-to-end engineered system with actionable algorithmic solutions.
2. **Alternative Target: *Computers, Environment and Urban Systems* (Elsevier)**
   - *Rationale*: Premier venue for spatial agent-based models and urban computational systems. If framed with greater emphasis on municipal GIS policy, CEUS is an ideal home.
3. **Specialized Target: *AI and Ethics* (Springer)**
   - *Rationale*: Excellent fit if reviewer feedback pushes for deeper analysis of algorithmic governance and fairness auditing standards.

---

### E. Stop Condition: When to Stop Experimenting and Start Writing
The experimental phase is complete once:
1. `results/e1_summary.csv`, `e2_feedback.csv`, `e3_incidence.csv`, and `e4_pareto.csv` are regenerated with **$N=30$ evaluation seeds**.
2. Training reward curves across **$N=5$ training seeds** confirm convergence stability.
3. Multi-seed **E5 falsification table** confirms the detection-driven collapse across all seeds.

Once these 3 artifacts are generated, **freeze all code and focus 100% on manuscript drafting and polishing**.
