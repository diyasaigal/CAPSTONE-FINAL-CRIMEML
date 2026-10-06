# Understanding the update

*A plain-language guide to what CrimeML is, what it used to claim, and what the October 2026 update changed and why. Written 2026-10-06. Work is on git branch `pivot`; `master` is unchanged.*

---

## 1. What the project is

**The short version:** we built a fake city where we *know* how much crime really happens, put a learning AI in charge of sending police patrols, and measured whether it becomes unfair.

**The pieces:**

- **The simulator ("the city").** A 20×20 grid laid over Chicago, split into the **22 real police districts**. **200 simulated people** live, work and go out on daily schedules. Their stress depends on how disadvantaged ("high-hardship") their district is, using real US Census (ACS) numbers. Crime emerges from how they behave. We **never feed in real crime locations**: the map starts flat, and crime hot spots form on their own.
- **The key trick: true crime vs recorded crime.** The simulator knows every crime that happens (**true** or **latent** crime). Police only *record* some of it, and the more police are in a district, the more of its crime gets recorded:
  - no patrol → **30%** of crime gets recorded;
  - heavy patrol → up to **85%**;
  - even (uniform) patrol → **66.7%** everywhere.
- **The loop.** Patrol a district more → record more crime there → it *looks* worse → patrol it even more. This is the "runaway feedback loop" known from predictive policing.
- **The AI ("the policy").** A reinforcement-learning agent (**PPO**) chooses how to split patrols each shift. It sees only *recorded* crime and is rewarded for "covering" recorded crime, just like a real predictive-policing dashboard. It picks three dials:
  - **β (beta):** how strongly to chase recorded hot spots (0 to 4);
  - **w:** an extra push toward (+) or away from (−) poor districts (−2 to +2);
  - **μ (mu):** how much to fall back to an even split (0 to 1).
- **The comparison policies.**
  - *Uniform:* even split.
  - *Random.*
  - *PredPol (EWMA):* patrol where crime was recently recorded.
  - *Random forest:* a hotspot model.
  - *Greedy oracle:* patrols in proportion to **true** crime. Not usable in real life, but it shows what "patrol where the need is" looks like.

**How we measure unfairness ("disparity"):** compare patrol per person in the **most disadvantaged fifth** of districts with the **least disadvantaged fifth**. A value of 1 means equal; 4 means the poorest districts get 4× the patrol per person.

---

## 2. What the project was before this update

The paper was titled *"Policing the Data It Creates: Feedback and Fairness in a Reinforcement-Learned Patrol-Allocation Rule over an Unseeded Crime Simulator."* It was trained for **150k steps**, with **3 training seeds** and **10 test seeds**.

**Its main claims:**

| Claim (old paper) | Old number |
|---|---|
| PPO's disparity | **8.1** (oracle 2.7, PredPol 3.8) |
| PPO has the best recorded coverage but worse true coverage | 0.064 recorded, 0.048 true (PredPol 0.055 true) |
| PPO over-polices poor districts beyond their crime | coefficient **+0.017, p < 10⁻³⁰⁰**, "3× PredPol" |
| E2: a "runaway index" proves the loop | index near 1 for PPO |
| E4: fairness penalties reach parity along a smooth trade-off; every penalised version beats PPO on true coverage | knee at penalty 0.02 |
| E5: turning off the detection effect cuts disparity from 9.0 to 2.9 | **1 training seed** |
| The oracle is an "upper bound" | — |

A handoff then proposed **reframing the paper** around a sharper question: *how biased does an RL learner get when its own actions contaminate the data it learns from?* The headline would be "PPO is **3.7×** the oracle".

---

## 3. What was wrong with the old version (the audit)

Before changing anything we checked every claim. The full audit is in `PIVOT_AUDIT.md`. Main problems:

| # | Problem | In easy words |
|---|---|---|
| **F1** | **The published PPO numbers came from only 1 of the 3 training runs.** A code bug kept the *last* run's value instead of the average. | The three runs gave **13.6, 8.5 and 8.1**; the paper printed **8.1** (just the last). The true average was about **10.1**. Every disparity and Gini number in the paper was affected. |
| **F2** | **Training lengths were mixed.** Seeds 0–2 had 150k steps, seeds 3–4 had 400k. | Pooling them would mix apples and oranges. |
| **F3** | **The "runaway index" doesn't measure feedback.** | A fixed allocation with *no* feedback at all scores **1.000** on it. |
| **F4** | **Most of the E3 "poor districts" effect was a measurement artefact.** The control for "how much crime a district has" was noisy, so hardship soaked up the difference. | With a proper control, PPO's coefficient shrank from **+0.0166 to +0.0034**. |
| **F5** | **The fairness penalties overshoot.** They push disparity toward 0, not toward "fair". | Penalised versions gave the poorest districts *less* patrol than their crime needs (**0.55× down to 0.17×** the oracle). |
| **F6** | **The core idea still holds.** | Even after fixes, PPO was clearly above the oracle (**3.7×** on the old data). |

**Also wrong:**
- "Every penalised version beats PPO on true coverage": false, they are all *lower*.
- "Knee costs ~1 point vs uniform": false, it's actually *above* uniform.
- E5 used **6** test seeds, not "ten".
- The oracle is *not* an upper bound: PredPol covers more true crime than it does.

---

## 4. What we did in this update, and why

We followed the 8-phase plan in order. Nothing in the simulator, the detection model, the PPO setup or the baselines was changed. Old results are preserved (git tag `pre-pivot`, folders `results/pre_pivot/` and `runs_pre_pivot/`, old paper in `paper/archive/`).

### Five decisions (made 2026-10-03)

| Decision | Choice | Why |
|---|---|---|
| Training length | **400k steps for everything** | At 150k some models were still changing, and one run (13.6) was an outlier. |
| Test seeds | **200–229** (30 seeds) for E1–E4; **300–319** (20 seeds) for E5 | 3× more test seeds; the old 200–209 are a subset. |
| Main metric | **"× oracle"**: PPO's disparity divided by the oracle's, matched on the same test seed | Poorer districts *do* have more crime, so some disparity is fair. Comparing to "patrol where the need is" separates fair targeting from bias. |
| New penalty | **"Oracle-targeted" penalty**: only punish disparity *above* the oracle's (2.78); keep the old penalty as a contrast | The old penalty overshoots (F5). |
| Extra experiments | A **2×2 experiment** (what PPO sees × what it is rewarded on) and a **non-AI search** for the best fixed rule | To find *where* the bias comes from, and whether it needs AI at all. |

### Phase by phase

| Phase | What we did | Why |
|---|---|---|
| **0. Freeze** | New branch `pivot`, snapshot of everything as found, backups. | So nothing old is lost. |
| **1. Metrics in code** | New file `crimesim/eval/oracle_relative.py`: "× oracle", the proxy gap and the over-recording ratio. Error bars come from a hierarchical bootstrap that resamples both training and test seeds. Fixed E3 with a proper "need" control and per-episode clustered errors. Unit tests added. | The audit's fixes had to live in the codebase, with tests, not in throwaway scripts. |
| **2. Reanalysis** | Re-ran the new metrics on old data (`results/reanalysis/`). | To check the story survives before spending compute. It did. |
| **3. Prep** | Fixed a bug in the Lagrangian method (it used a hard-coded 0.15). Added a guard so a model trained with different settings is never silently reused (F2). Added the new penalty and a new E5 control, **"constant detection"**. Wrote the fixed-rule search. | The old E5 "perfect observation" test changed two things at once: it removed the patrol effect *and* gave the AI more data. Constant detection removes only the patrol effect. |
| **4–5. Overnight run + stats** | Trained **51 models** (about 15 h, no failures) and evaluated everything. Wrote the final tables (`results/final/`). Ran **19 pre-planned comparisons** with Holm correction for multiple testing. | The final numbers, with honest error bars. |
| **6. Figures** | 6 new figures, including a diagram of the loop. | Old figures used wrong numbers and the dropped runaway index. |
| **7. Paper** | Rewrote `paper/main.tex`. New title: *"Measuring Reinforcement-Learning Bias Under Self-Contaminated Observations."* Two new citations, checked against the publishers. | The story changed. Every number is now generated from the results, never typed by hand. |
| **8. Final check** | 42/42 tests pass, simulator calibration 5/5 pass, paper check script passes. | To make sure everything holds together. |

---

## 5. The new results

*400k training steps. "Seeds" = number of separately trained models. Tested on 30 test seeds. "[a, b]" is a 95% confidence range.*

### E1: How unfair is PPO, and does it at least catch more crime?

| Policy | Seeds | Disparity × oracle | True-crime coverage | Recorded coverage |
|---|---|---|---|---|
| Greedy oracle (reference) | — | **1** (raw disparity 2.78) | 5.37% | 5.67% |
| Uniform | — | 0.36 | 4.55% | 4.55% |
| PredPol (EWMA) | — | 1.41 | 5.48% | 5.95% |
| Random forest | — | 1.36 | 5.47% | 5.91% |
| **PPO (naive)** | 5 | **3.91 [2.98, 5.08]** | **4.77%** | 6.72% |
| PPO that sees + is rewarded on true crime | 5 | 1.22 [1.11, 1.35] | 4.90% | 5.71% |

**In plain words:**
- PPO puts about **4×** more patrol on the poorest districts than "patrol where the need is" would, and about **2.8×** more than PredPol.
- Yet it **catches less real crime** than the oracle, PredPol or the random forest. It looks great on its own scoreboard (recorded crime) but does worse on what actually matters.
- The 5 PPO training runs range from **2.6× to 5.3×**. The AI is unstable, and we report that openly.
- **What PPO learned:** β ≈ **3.1** (chase hot spots hard), w ≈ **+1.28** (an extra deliberate push toward poor districts), μ ≈ **0.14** (almost no even split).

**The old headline changed:** 3.7× (old data) → **3.9×** (final). The earlier 2.8× was from just 2 runs.

### Is it the AI, or the goal? (fixed-rule search)

We tried **192 fixed dial settings** with no AI involved.

| Fixed rule | × oracle | True coverage | Recorded coverage |
|---|---|---|---|
| Best at *recorded* coverage (β=4, w=+2) | **13.0** | 4.75% | **7.46%** |
| Best at *true* coverage (β=1.5, w=0) | 2.2 | **5.52%** | 6.26% |
| PPO, for comparison | 3.9 | 4.77% | 6.72% |

**In plain words:**
- A simple fixed rule beats PPO at PPO's own goal (7.46% vs 6.72%), so **PPO is under-trained**.
- Chasing recorded crime harder makes things **even more unfair** (13×). So the bias comes from **the goal** ("cover recorded crime"), not from AI as such.
- **Caveat:** the best rule for true coverage beats the oracle (5.52% vs 5.37%). So the oracle is a *fairness reference* ("patrol where the need is"), **not** the best possible policy.

### E2: Does PPO pollute its own data?

| | PPO | PredPol | Oracle | Uniform |
|---|---|---|---|---|
| **Proxy gap** (recorded − true coverage) | **1.96 pts** | 0.47 | 0.30 | 0 |
| **Over-recording** (how much more of the poorest fifth's crime gets recorded vs the richest fifth; 1 = equal) | **1.89** | 1.75 | 1.58 | 1.00 |

**In plain words:**
- PPO's scoreboard overstates how much real crime it covers by about **2 percentage points**, 4× more than any baseline. The gap grows during an episode.
- The data PPO learns from is tilted against poor districts.
- **Dropped:** the old "runaway index" (it doesn't measure feedback).
- **Tested but not used as evidence:** a new "self-induced signal". It comes out high for the oracle too (0.51 vs PPO 0.48), so it measures the detection effect, not PPO.

### E3: Does PPO over-patrol poor districts *beyond their actual crime*?

We compare each district's patrol share with its **need** (its long-run share of true crime under even patrol).

| Policy | Hardship effect (×10⁻³) | t-value | Training runs where it's positive |
|---|---|---|---|
| Oracle | −0.7 | −1.3 (none) | — |
| PredPol | −0.5 | −0.8 (none) | — |
| **PPO** | **+3.8** | **7.8** | **5 of 5** |
| Best fixed rule for recorded coverage | +10.3 | 8.7 | — |
| Parity penalty 0.02 | −12.0 | −19.8 | 0 of 5 (under-patrols poor districts) |
| Oracle-targeted penalty 0.15 | −5.0 | −10.9 | 0 of 3 |

**In plain words:**
- Yes. Even after accounting for how much crime they really have, PPO gives the poorest fifth of districts about **27 percentage points more patrol share than they need**, taken mostly from the middle districts.
- The effect is real but about **¼ the size** the old paper claimed (+0.017, p < 10⁻³⁰⁰).
- **Note:** an earlier check with only 2 runs suggested "not significant" (t = 1.05), and we nearly reworded E3. With 5 runs × 30 test seeds it is clearly significant, so the E3 claim stays, in a smaller, honest form.

### The 2×2: is it what PPO *sees*, or what it's *rewarded* on?

| PPO observes ↓ / rewarded on → | Recorded crime | True crime |
|---|---|---|
| Recorded crime only | **4.85×** | **1.38×** |
| Recorded + true crime | 2.20× | 1.20× |

**In plain words:**
- Changing the **reward** to true crime fixes most of the bias (4.85 → 1.38).
- Just *showing* it true crime helps less (4.85 → 2.20).
- **The reward is the bigger lever.**

### E4: Can a penalty fix it without overdoing it?

| Policy | × oracle | True coverage |
|---|---|---|
| PPO, no penalty | 3.91 | 4.77% |
| Old parity penalty 0.02 | 0.66 | 4.81% |
| Old parity penalty 0.06 | 0.34 | 4.49% |
| Old parity penalty 0.15 | 0.25 | 4.27% |
| Old parity penalty 0.4 | 0.11 | 3.39% |
| New oracle-targeted penalty 0.05 | 2.98 | 4.82% |
| **New oracle-targeted penalty 0.15** | **0.76 [0.69, 0.85]** | **4.90%** |
| New oracle-targeted penalty 0.5 | 0.53 | 4.79% |

**In plain words:**
- **Old penalties overcorrect:** even the lightest leaves poor districts *under*-patrolled for their crime, and heavy ones wreck coverage (3.39%).
- **The new penalty at 0.15** gets **closest to fair** (0.76×) and *raises* true coverage above plain PPO (4.90% vs 4.77%, statistically significant).
- It still isn't perfect: it ends a bit below the target, and 0.05 is too weak.

### E5: Remove the patrol-changes-detection effect

Three training runs each, tested on 20 new seeds. Each setting is compared with its own oracle.

| Setting | PPO × oracle | Penalised PPO × oracle | Over-recording |
|---|---|---|---|
| Normal (patrol raises detection) | **4.92 [4.05, 5.92]** | 0.33 | 2.01 |
| Perfect observation (all crime recorded) | 1.65 [1.51, 1.86] | 0.36 | 1.00 |
| **Constant detection** (66.7% everywhere, same data volume) | **1.46 [1.36, 1.59]** | 0.35 | 1.00 |

**In plain words:**
- When patrol no longer changes what gets recorded, PPO's extra unfairness drops from **~4.9× to ~1.5×**.
- The new constant-detection control proves this is due to the **feedback loop**, not just to "more data".
- About **1.5×** remains even without the loop.
- **Wording rule:** this is *strong evidence inside the simulator*, **not "proof of causality"** in real policing.

### Statistics

- All **19** planned comparisons were tested with **Holm correction** for multiple tests.
- **17 of 19** are significant. The two that aren't: the new penalty at 0.05 and at 0.5 don't change true coverage compared with plain PPO.
- The full table is Table 7 of the paper and `results/final/contrasts.csv`.

---

## 6. Old vs new at a glance

| Topic | Before | After |
|---|---|---|
| Headline | PPO disparity **8.1** (1 run only, bug) | PPO **3.9× the oracle** [3.0, 5.1], 5 runs × 30 test seeds |
| Training | 150k / 400k mixed | 400k for all |
| Test seeds | 10 | 30 (E5: 20, was 6) |
| Fairness yardstick | raw disparity (1 = fair) | **× oracle** (1 = patrol matches need) |
| E2 evidence | runaway index | proxy gap + over-recording |
| E3 | +0.017, p < 10⁻³⁰⁰ | +0.0038, t = 7.8, 5/5 runs, need-controlled |
| E4 | parity penalty "removes" bias | parity penalty **overcorrects**; new targeted penalty is best |
| E5 | 1 run, perfect observation only | 3 runs, + constant-detection control |
| Oracle | "upper bound" | "fairness reference", not the best policy |
| New experiments | — | 2×2 attribution; fixed-rule search; learned dial values |

---

## 7. Honest limitations (also in the paper)

1. It's a **simulation**. Detection and deterrence aren't calibrated to real reporting rates.
2. Each "fifth" of districts has only **4–5 districts**.
3. **PPO is unstable** across runs (2.6× to 5.3×), and E5 has only 3 runs.
4. PPO only controls a 3-dial rule and **isn't fully optimised** (a fixed rule beats it at its own goal).
5. Population is assumed **equal** across districts.
6. The oracle is a **fairness reference**, not the most efficient policy.
7. Simulated criminals **don't adapt** to police.

---

## 8. Where things are

| What | Where |
|---|---|
| Audit of the old version | `PIVOT_AUDIT.md` |
| Final results (CSV) | `results/final/` |
| New paper | `paper/main.tex` (+ `numbers.tex`, `tables/`, `refs.bib`) |
| Old paper | `paper/archive/main_pre_pivot.tex` |
| New figures | `figures/fig_arch`, `fig_e1_disparity`, `fig_e2_contamination`, `fig_e3_need_residual`, `fig_e4_frontier`, `fig_e5_interventions` |
| Team progress log | https://claude.ai/code/artifact/2af34038-9a94-40f2-8bca-4a29b8656121 |

**To regenerate everything after a change** (Windows, from the repo root):

```
PYTHONPATH=. .venv-win/Scripts/python.exe scripts/summarise_400k.py
PYTHONPATH=. .venv-win/Scripts/python.exe scripts/make_paper_assets.py
.venv-win/Scripts/python.exe scripts/check_paper.py
```

Never type numbers into the paper by hand. Rerun these scripts instead.

---

## 9. What's left

- [ ] Compile the paper in **Overleaf** (no LaTeX on the laptop). Upload `main.tex`, `refs.bib`, `numbers.tex`, `tables/` and the figure PDFs into one folder.
- [ ] Team reads and edits the new draft.
- [ ] Set up a shared **git remote** so everyone can see the `pivot` branch.
- [ ] Optional: more training runs to tighten the error bars; the Lagrangian method as a robustness check.
