# CrimeSim-RL: Prioritized Journal Experiment Roadmap

This document outlines the concrete, prioritized experiments required to elevate the **CrimeSim-RL** manuscript to top-tier journal submission standards (*Expert Systems with Applications* / *Computers, Environment and Urban Systems*).

---

## Phase 1: MUST-HAVE Experiments (Mandatory for Acceptance)

```
[Phase 1 Execution Plan]
1. Multi-Seed Training (N=5 Seeds: 0, 1, 2, 3, 4)
2. Scaled Evaluation (N=30 Held-Out Seeds: 100 to 129)
3. Multi-Seed Falsification Ablation (3 Train Seeds x 10 Eval Seeds)
```

### 1.1 Scaled Multi-Seed Evaluation ($N=30$) & Training ($N=5$)
* **Scientific Purpose**: Reduce 95% confidence interval width by $\approx 42\%$ ($\frac{1}{\sqrt{30}}$ vs $\frac{1}{\sqrt{10}}$) and demonstrate PPO convergence stability across 5 independent neural network initializations.
* **Exact Command**:
  ```bash
  PYTHONPATH=. python scripts/run_all.py \
    --train-seeds 0 1 2 3 4 \
    --eval-seeds $(seq 100 129) \
    --penalties 0.02 0.06 0.15 0.40 \
    --timesteps 400000
  ```
* **Expected Output Artifacts**:
  - `results/e1_summary.csv` (Tight CIs over 30 seeds)
  - `results/e2_feedback.csv` (Smooth dynamic trajectories)
  - `results/e3_incidence.csv` (High-power panel regressions, $N=66,000$ rows)
  - `results/e4_pareto.csv` & `results/e4_knee.json` (Validated monotone frontier)
  - `figures/fig_efficiency_equity.pdf`, `fig_feedback_trajectories.pdf`, `fig_incidence.pdf`, `fig_pareto.pdf`

---

### 1.2 Multi-Seed E5 Falsification Suite
* **Scientific Purpose**: Prove that the collapse of the feedback loop under perfect observation ($q_0=1.0$) is robust across multiple training seeds rather than an artifact of a single seed.
* **Exact Command**:
  ```bash
  PYTHONPATH=. python scripts/run_ablations.py --train-seeds 0 1 2 --eval-seeds $(seq 200 209)
  ```
* **Expected Output Artifacts**:
  - `results/e5_ablations.csv` (Mean $\pm \text{CI}$ across 3 training seeds)

---

## Phase 2: STRONGLY RECOMMENDED Experiments

### 2.1 2D Sensitivity Grid: Detection Gain ($g$) vs Deterrence Slope ($\kappa$)
* **Scientific Purpose**: Map the exact phase transition where observation bias overwhelms physical crime deterrence.
* **Parameter Sweep**:
  - Detection Gain $g \in \{0.00, 0.25, 0.55, 0.75\}$
  - Deterrence Slope $\kappa \in \{0.00, 0.15, 0.26, 0.45\}$
  - Total combinations: 16 runs $\times$ 10 evaluation seeds.
* **Output Artifact**:
  - `results/sensitivity_g_kappa.csv`
  - `figures/fig_sensitivity_heatmap.pdf` (2D contour plot of end-state disparity $\delta$)

---

### 2.2 Action-Space Expressiveness Ablation
* **Scientific Purpose**: Defend against reviewer critique regarding why the 3D continuous rule space $(\beta, w, \mu)$ was selected over direct 22-dimensional continuous allocation.
* **Configuration**:
  - Compare `action_mode="rule"` vs `action_mode="direct"` (22-logit softmax).
* **Expected Result**: Show that direct 22-simplex control suffers from policy gradient collapse, verifying the necessity of rule parameterization.
* **Output Artifact**:
  - `results/ablation_action_space.csv`

---

### 2.3 Two-Way Cluster-Robust Standard Errors in E3
* **Scientific Purpose**: Prove to econometric/statistical reviewers that panel regression significance ($p \ll 10^{-300}$) is not an artifact of within-district serial correlation.
* **Implementation**:
  - In `crimesim/eval/incidence.py`, implement two-way cluster variance estimation on `(district, seed)`.

---

## Phase 3: OPTIONAL / Extension Experiments

### 3.1 Empirical District Population Weights
* **Scientific Purpose**: Replace the uniform residential population assumption ($p_r = 1/22$) with empirical ACS district population counts.
* **Configuration**:
  - Load population counts from `data/processed_census.csv` and re-evaluate E1 metrics to demonstrate that per-capita disparity findings remain unchanged.

---

## Summary Timeline & Execution Order

| Priority | Task | Computational Time (CPU) | Priority Level |
|---|---|---|---|
| **Step 1** | Run Scaled Main Pipeline ($N=5$ train, $N=30$ eval) | $\sim 3\text{--}4$ hours | 🔴 **MUST HAVE** |
| **Step 2** | Run Multi-Seed E5 Ablation Suite | $\sim 1\text{--}2$ hours | 🔴 **MUST HAVE** |
| **Step 3** | Run 2D Sensitivity Grid on $(g, \kappa)$ | $\sim 2$ hours | 🟡 **STRONGLY RECOMMENDED** |
| **Step 4** | Run Action Space Direct vs Rule Ablation | $\sim 1$ hour | 🟡 **STRONGLY RECOMMENDED** |
| **Step 5** | Freeze Code & Finalize LaTeX Manuscript Draft | — | 🟢 **MANUSCRIPT PHASE** |
