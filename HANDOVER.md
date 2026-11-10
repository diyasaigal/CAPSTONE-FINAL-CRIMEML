# JASSS revision: handover

Written 2026-11-10, on branch `jasss`. This is for the teammate who has the JASSS `.tex` source and will finish the revision. Read it together with:

- the change list, *JASSS Paper: Changes To Be Made* (items A1–A20, B1–B19, C1–C10, Part D, and the readiness gate);
- `JASSS_PROGRESS.md`, which records how Phases 0 and 1 were decided;
- `scripts/summarise_jasss.py`, whose docstring defines every number in `results/jasss/summary/`.

## 1. Where things stand

| Phase | Status |
|---|---|
| Phase 0: analysis on old rollouts (A1–A5) | Done. Those numbers are **obsolete** for the paper, because the generator changed at Gate 1. Use them only to see how decisions were made. |
| Phase 1: generator (A6–A9, A15) | Done and frozen. Tag `jasss-frozen-v1`, config `configs/jasss_base.yaml`. |
| Phase 2: core reruns (A10–A14, A17, E1/E5 at each contrast and agent count) | **Done.** All 380 models trained, evaluated, verified and summarised. Results are in `results/jasss/summary/`. |
| Phase 3: extensions | A17 done (part of Phase 2). A15 and A16 skipped by decision. **A18, A19 and C3 are open.** |
| Phase 4: paper | **Open.** Everything is in §5 below. |
| Phase 5: archive and submit | **Open.** See §5. |

### What was run
- **380 PPO models.** 10 training seeds (0–9) × 38 policy families, 400,000 steps each. Each is evaluated on the same 30 evaluation seeds (200–229).
- **20 settings:**
  - base, plus constant and perfect detection;
  - detection gain low/high;
  - deterrence weak/strong;
  - baseline reporting 0.15, 0.40, 0.50, 0.60 and 0.70, with saturation held at p_report + detect_gain = 0.85 so q never exceeds 1;
  - crime contrast low/high;
  - 500 and 1,000 agents;
  - a constant-detection twin of each of the last four.
- **Rule-based policies in every setting:**
  - uniform, random, EWMA hot-spot ("PredPol"), random-forest hot-spot;
  - the GreedyOracle (the latent-crime-proportional reference);
  - NeedProportional (the untreated-crime reference, A12);
  - no patrol, and the EWMA-like rule (1, 0, 0).
- **Grid-search optimum rules** (192-rule grid, 3 search seeds) for:
  - base, const and perfect;
  - contrast low/high and 500/1,000 agents.
- **The work was split across three laptops.** Diya ran A10, A11 and A17; Dennis ran A13 and A14; Ashmith ran A6A7.
  - Every model was checked: its settings equal its specification, it ran the full 400k steps, and it was evaluated on all 30 seeds.
  - **All evaluations were then redone on one machine (Diya's).** A spot check showed Dennis's models reproduce bit for bit. Ashmith's differed only around the 7th significant digit (a different CPU), and were re-evaluated anyway.
  - `results/jasss/trained_on.csv` lists the machine that trained each model, and any resume point.
- **Code version:** every model was trained on commit `b1d6890`. Later commits change only the launcher, the list of settings that get a rule grid, and the summary. None touch the simulator, the environment or training.

## 2. Files

| What | Where | In git? |
|---|---|---|
| Frozen generator config | `configs/jasss_base.yaml`, `crimesim/experiments/jasss.py` | yes |
| Run script, all settings and model specs | `scripts/jasss_runs.py` (launcher `scripts/run_jasss.ps1`) | yes |
| Summary script | `scripts/summarise_jasss.py` | yes |
| **Central table** (one row per setting × policy) | `results/jasss/summary/central.csv` | yes |
| Contrasts, Holm over 41 | `results/jasss/summary/contrasts.csv` | yes |
| Per-model provenance (A3) | `results/jasss/summary/provenance.csv` | yes |
| Per training seed / per episode | `per_seed.csv`, `episodes.csv` | yes |
| Crime prevented per hardship quintile | `quintiles.csv` | yes |
| Headline numbers for the abstract and readiness gate | `results/jasss/summary/headline.json` | yes |
| References per setting (oracle, need vector) | `results/jasss/refs/*.json` | yes |
| Rule grids | `results/jasss/rules/<setting>_grid.csv` | yes |
| Run logs | `results/jasss/run.log` (Diya), `run_dennis.log`, `run_ashmith.log` | yes |
| Real crime counts per cell, district and community area | `data/region_district_g20.npz`, `data/ca_lookup_g20.npz` | yes |
| Trained models and their rollouts (1.8 GB) | `runs/jasss/` | **no**: on Diya's laptop. The 10 base PPO models are in the handover zip (see below). |
| Rule-based rollouts (0.6 GB) | `results/jasss/eval/` | **no**: on Diya's laptop |
| Crime records (120 MB) | `data/processed_crime.csv` | **no**: in the handover zip (see below) |

To regenerate every summary file from the cached rollouts (about 10 minutes, no simulation):
```
PYTHONPATH=. python scripts/summarise_jasss.py
```
To redo any training or rollouts, run `scripts/jasss_runs.py` (see its docstring). It is resumable and skips finished work.

### Files that are not in git: the handover zip
Git leaves out large files (`runs/`, `*.pkl`, `data/processed_crime.csv` are in `.gitignore`). Diya sends the two parts the remaining work needs as one zip, **`CrimeML_handover_data.zip` (60 MB)**:

| In the zip | Size | What it is | Needed for |
|---|---|---|---|
| `data/processed_crime.csv` | 120 MB unzipped | The real crime records (details below) | A19 if you go beyond the cached counts; C6 data section; anything by hour, year or crime type; rebuilding the map caches |
| `runs/jasss/base__ppo_s0` … `base__ppo_s9` | 48 MB | The 10 naive-PPO models in the base setting. Each folder has `ppo_model.zip` (the policy), `obs_rms.npz` (observation normalisation, needed to load the policy), `config.yaml` (the exact config) and `eval_e200n30.pkl` (its 30 evaluation rollouts) | A2 (PPO trajectories, B11) |
| `HANDOVER_DATA_SHA256.txt` | | A SHA-256 checksum for every file | checking the copy is intact |

**Installing it.** Unzip it **at the repo root**, so the paths land exactly as listed (`data/processed_crime.csv`, `runs/jasss/base__ppo_s0/...`). In PowerShell, from the repo root:
```
tar -x -f C:\path\to\CrimeML_handover_data.zip
Get-Content HANDOVER_DATA_SHA256.txt | ForEach-Object { $h, $f = $_ -split '  '; if ((Get-FileHash $f -Algorithm SHA256).Hash.ToLower() -ne $h) { "BAD: $f" } }
```
The second line prints nothing if every file is intact. Then delete `HANDOVER_DATA_SHA256.txt`. All of these paths are git-ignored, so they cannot be committed by accident. **Do not** remove them from `.gitignore`: the crime file is too big for GitHub (100 MB limit).

**The crime records** (`data/processed_crime.csv`):
- **What:** 439,527 violent-crime records for Chicago, one row per incident. They are dated from 1 Jan 2020 to 23 Jan 2026 and were downloaded around 30 Jan 2026 (the latest "Updated On"). There are 23 police districts and 77 community areas.
- **Source:** the City of Chicago Data Portal, "Crimes – 2001 to Present"; the columns match that dataset exactly. **Confirm the dataset ID and download date with Diya before citing it in C6.**
- **Processing** (`legacy/preprocess.py`):
  - it keeps only BATTERY, ASSAULT, ROBBERY, HOMICIDE and CRIM SEXUAL ASSAULT;
  - it drops rows with no coordinates;
  - it adds `Hour` and normalised coordinates.
- **Per year:**

  | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 |
  |---|---|---|---|---|---|---|
  | 67,972 | 69,275 | 71,236 | 78,302 | 79,023 | 70,285 | 3,434 |

- **By type:** battery 256,948; assault 127,610; robbery 50,874; homicide 4,026; criminal sexual assault 69. The last count is tiny. The likely cause is that the portal now files these cases under "CRIMINAL SEXUAL ASSAULT", a name the filter does not keep. **Check this, and state it in the paper.**
- **What the simulator uses from it:** the code never reads this file during a run. It is read once to build the two map caches in `data/`, which *are* in git:
  - `region_district_g20.npz`, built by `crimesim/abm/regions.py`. It holds the 20×20 grid, each cell's district, real counts per cell (`real_grid`; 192 non-empty cells; 438,764 records inside the bounding box) and per district (`real_region`), plus population;
  - `ca_lookup_g20.npz`, built by `scripts/build_ca_lookup.py`. It holds real counts per community area (`real_ca`).

  The map caches are already built, so **the simulator, all training and the summary run without this file**.
- **Needed directly only for:**
  - A19 at a resolution or grouping the caches don't hold;
  - real-data patterns by hour, year or type (for example, the real night/day ratio 1.18 in C2/B3);
  - C6's record counts (given above).
- **If you delete the caches, the next run rebuilds them from this file.** Without the file, that run fails.

**Using the 10 models for A2.**
- **Steps 1–2** (action trajectories, SD within an episode, correlation with the EWMA dispersion and the hour) need **only the rollouts**:
  - `eval_e200n30.pkl` is a pandas DataFrame with one row per (evaluation seed, shift);
  - its columns are `beta`, `w`, `mu`, `ewma_cv`, `hour`, `recorded_coverage`, `latent_coverage`, `disparity_cum`, and others;
  - load it with `pd.read_pickle(...)`. No simulation is needed.
- **Step 3** (PPO's mean action run as a fixed rule):
  - compute each seed's mean (β, w, μ) from the same pickle;
  - roll it out as a fixed rule with `jasss_config("base")` and `region_for(cfg)` on evaluation seeds 200–229;
  - this also needs no model.
- **The models themselves** (`ppo_model.zip` + `obs_rms.npz`) are only needed to roll PPO out again, for example with extra logging. Load them the way `scripts/jasss_runs.py` does in its evaluation stage.

**Not in the zip** (ask Diya if needed):
- the other 370 models and every rule-based rollout (2.4 GB in total). Nothing left in §5 needs them, because every number from them is already in `results/jasss/summary/`;
- for the CoMSES archive (C7), ship the code, configs and summary CSVs rather than the models: every model can be regenerated with `scripts/jasss_runs.py`. If the editors want the models, Diya can upload the full `runs/jasss/` folder (1.8 GB) to Zenodo or Google Drive.

### Names in the code vs names for the paper

| Code (`family` / `setting`) | Paper |
|---|---|
| `oracle` | **latent-crime-proportional reference** (B5: never "need-proportional") |
| `need_prop` | **untreated-crime reference** (A12), which is "need" |
| `ppo` (base) | A10 cell 1, naive PPO |
| `rewlat` | cell 2 (rewarded on true crime) on base; cell 6 on `const` |
| `obslat` | cell 3 (observes true crime) |
| `oracleppo` | cell 4 (both levers) |
| `const/ppo`, `perfect/ppo` | cells 5 and 7 |
| `par<w>` | parity penalty, weight w |
| `penref<w>` | oracle-targeted penalty |
| `penneed0.15` | penalty targeting the untreated-crime reference |
| `lag` | Lagrangian PPO |
| `w0`, `wneg`, `w0_nosocio` | A11: w = 0; w ≤ 0; w = 0 without socio-economic observations |
| `ewma_hotspot` | PredPol-style EWMA hot-spot |
| `best_rule_recorded` / `best_rule_latent` | grid-search optimum for recorded / true coverage |
| `x_oracle` | x reference (geometric mean over episodes, paired by evaluation seed) |
| `prevented_pct` | crime prevented, % of the paired no-patrol run |

## 3. Headline results

Base setting, 10 training seeds, 30 evaluation seeds. Intervals are 95% hierarchical bootstrap (training seeds × evaluation seeds, 4,000 replicates, bootstrap seed 0). Contrast p-values use 20,000 replicates and are Holm-adjusted over all 41 contrasts. **`p_Holm = 0.002` is the smallest value the bootstrap can show,** so write it as p < 0.01.

**The one canonical PPO number (B1):**
- naive PPO is **5.74× the reference [5.17, 6.41]**;
- median 5.85, range 4.27–7.06 across the 10 seeds;
- against the untreated-crime reference it is 4.72×.

The old draft's 3.91×, 4.85× and 4.92× are all obsolete.

**A10, the common experiment** (`central.csv`):

| Cell | x reference [95% CI] | Crime prevented % | True coverage |
|---|---|---|---|
| 1 base PPO | 5.74 [5.17, 6.41] | 29.10 | 0.049 |
| 2 base, true-crime reward | 1.78 [1.65, 1.91] | 30.19 | 0.052 |
| 3 base, observes true crime | 2.76 [2.26, 3.49] | 28.47 | 0.048 |
| 4 base, both levers | 1.31 [1.24, 1.41] | 29.31 | 0.050 |
| 5 constant detection | 1.47 [1.39, 1.56] | 30.20 | 0.052 |
| 6 constant, true-crime reward | 1.52 [1.40, 1.66] | 30.38 | 0.052 |
| 7 perfect detection | 1.50 [1.45, 1.56] | 30.42 | 0.053 |
| reference (base) | 1.00 | 30.73 | 0.057 |
| PredPol (base) | 1.38 [1.36, 1.39] | 32.24 | 0.058 |
| grid optimum, recorded (base) | 7.95 [7.47, 8.34] | 34.76 | 0.053 |
| parity 0.06 (base) | 0.28 [0.25, 0.32] | 26.23 | 0.045 |

What this shows:
- **The detection channel drives most of the excess.** Base PPO is 3.91× constant-detection PPO [3.48, 4.38].
- **The clean baseline (cell 6 vs cell 5) shows no difference** (p = 0.48). Once the loop is broken, rewarding true crime changes nothing.
- **Perfect vs constant detection:** the old 1.65× vs 1.46× gap is gone at 10 seeds (1.50 vs 1.47, intervals overlap). The two hypotheses in A10 need not be tested; say the gap was seed noise.
- **The mentor's question:** breaking the loop also helps on crime, not just disparity. Constant-detection PPO prevents 1.1 points more crime than base PPO [0.83, 1.38], and perfect-detection PPO 1.3 points more [1.07, 1.59]. True coverage rises from 0.049 to 0.052–0.053.

**A1, crime prevented (Gate 0 must be re-checked: important):**
- Base PPO prevents **1.6 points less** crime than the reference [−1.89, −1.38]. The A1 decision rule therefore says **keep the cost claim**.
- **But concentration in itself does not cost crime in this model.**
  - The grid optimum for recorded crime is the most concentrated policy (7.95×), yet it prevents the **most** crime: 34.8%, 5.7 points more than PPO.
  - PPO with w = 0 is 3.09× and prevents 34.0%.
  - Deterrence is linear in patrol, so how much crime is prevented depends on *where* patrol goes, not on how even it is.
  - **The defensible claim:** PPO prevents less crime than the reference and much less than a fixed rule optimising the same proxy. That is an inefficiency of the learned policy. It is not "concentration costs crime".
- **The cost claim is not robust across settings.** At 500 agents, PPO prevents 30.68% [30.34, 31.00] against the reference's 30.42% [30.31, 30.53]: no deficit. At 1,000 agents PPO is 0.8 points below. At the low and high contrasts it is 2.0 and 2.4 points below.
  - Under the A1 rule, the claim has to be stated per setting, or weakened. **This is a team decision.**
- **By hardship quintile** (`quintiles.csv`), base PPO prevents 6.3% in the least disadvantaged quintile and 52.3% in the most. The reference's figures are 18.0% and 39.4%.

**A11, w switched off:**
- PPO with w = 0 is **3.09× the reference [2.83, 3.37]**, so **64% of the log excess remains** with the tilt removed (`headline.json`: `w_off_share_of_log_excess`).
- With w ≤ 0 it is 1.69×. With w = 0 and no socio-economic observations it is 2.92×.
- So recorded crime alone carries most of the skew, and w is a design choice that adds to it (B12, Section 3.4).

**A12, two references:** every ordering is the same against the untreated-crime reference, and the levels fall by a factor of about 0.82. Base PPO is 4.72×.

**A13, penalties** (crime prevented in brackets):

| Penalty | x reference | Crime prevented |
|---|---|---|
| none (PPO) | 5.74× | 29.1% |
| oracle-targeted 0.05 | 1.94× | 28.4% |
| oracle-targeted 0.15 | 0.43× | 27.2% |
| oracle-targeted 0.5 | 0.35× | 26.7% |
| parity 0.02 / 0.06 / 0.15 / 0.4 | 0.42 / 0.28 / 0.20 / 0.12× | 27.2 / 26.2 / 25.3 / 22.7% |
| untreated-crime-targeted 0.15 | 0.55× of its own target | 27.6% |
| **Lagrangian** | **1.06× [1.00, 1.12]** (not different from the reference, p = 0.20) | 27.2% (−1.9 points vs PPO) |

- **The oracle-targeted penalty overshoots** from 0.15 upward, ending below the reference. The old draft said it lands on the reference.
  - The penalty is one-sided and acts per shift: `w · max(0, ln disparity_shift − ln ref)`, where ref is the reference's *mean per-shift* disparity (`oracle_shift` in `results/jasss/refs/base.json`). See `crimesim/config.py` and `crimesim/rl/constrained_ppo.py`.
  - Keeping every shift under a ceiling pulls the average well below it. The x-reference is scored on whole-episode disparity, so the overshoot follows from the design. Say this in E4.
- **The Lagrangian is the only method that lands on the reference.** Every penalty costs crime prevented, and parity costs the most (B9).

**A6/A7, contrast and agent count** (excess = naive PPO over constant-detection PPO in the same world):

| World | Latent contrast | PPO x reference | Excess vs constant detection |
|---|---|---|---|
| contrast_low | ~1.8× (realistic) | 2.18 [1.95, 2.45] | **1.82× [1.62, 2.06]** |
| base | 3.1× | 5.74 | 3.91× [3.48, 4.38] |
| contrast_high | 4.3× (pre-revision world) | 6.38 | 3.43× [2.94, 4.01] |
| 500 agents | 3.8× | 9.88 | 5.90× [5.11, 6.62] |
| 1,000 agents | 4.6× | 10.60 | 6.14× [5.74, 6.57] |

- **The excess holds at the realistic contrast**, but it shrinks to 1.8×. Report that honestly (A6).
- **It is not monotone in contrast:** high is below base.
- **It grows with agent count, so it is not small-count noise.** A7 says: "if it changes, move the headline to the larger N". **That is a team decision.** Note that crime volume also grows with N, because there was no re-tuning (see `crimesim/experiments/jasss.py`).

**A14 and A17, sensitivity** (PPO x reference, 10 seeds each):
- detection gain 0.25 / 0.80: 3.43 / 7.05;
- deterrence slope 0.10 / 0.30: 10.28 / 5.25;
- baseline reporting 0.15 / 0.40 / 0.50 / 0.60 / 0.70: 11.11 / 4.00 / 3.02 / 2.59 / 2.07.

**Inside the survey range (0.41–0.57), PPO is about 3–4×** (B7). Base reporting (0.30) is below that range, so say so.

## 4. Things the paper must disclose
1. **Generator changes at Gate 1** (`JASSS_PROGRESS.md`, Phase 1):
   - hardship counted once;
   - population weighting (ACS 2023);
   - the night factor on both crime paths (`night_interaction`);
   - a 100-shift warm-up;
   - 500 and 1,000 agents **not** re-tuned.
2. **Calibration gate changes:**
   - night/day must be ≥ 1.10 (measured 1.17; real data 1.18);
   - the elasticity floor was lowered from 20% to 10% (measured 18.7%), with the reason why.
3. **A bug fix in parallel training:** environment seeds used to repeat across the parallel environments (`crimesim/env/policing_env.py`, `reset`). Every model in the paper was trained after the fix.
4. **Training was split across three machines on one commit.** Evaluation was redone on one machine. 14 models were resumed from a checkpoint after an interruption; the optimiser state was restored, see `trained_on.csv`.
5. **Displacement (A15) and the noisy-need penalty (A16) were not done.** Put both in the limitations.
6. **Rule grids** are scored on 3 search seeds and evaluated on the 30 evaluation seeds. Call them the "grid-search optimum within the box", not the exact optimum (B6, A18).

## 5. What is left, and how

### Step 0: bring in your files
1. Put the JASSS `.tex`, `.bib` and class in `paper/jasss/`.
2. Put the scripts behind the validation, calibration grid, OAT/LHS, concave curve, strong loop and rule boxes in `scripts/validation/`.
3. Commit on `jasss`.

### Analyses still to run (no PPO training needed)
Any script that scores rollouts **must** load the region with `region_for(cfg)` and the config with `jasss_config(...)` (`crimesim/experiments/jasss.py`). **Do not use `load_region_info(20)` and `configs/base.yaml`, which is the old world.** Exposure per capita depends on population.

- **A2: PPO trajectories on the new models.**
  - `scripts/ppo_actions.py` still reads the old `runs/ppo_s*` with `base.yaml`.
  - Point it at `runs/jasss/base__ppo_s0..9`, with config `jasss_config("base")` and `region_for(cfg)`, then rerun it. The 10 models are in the handover zip (§2). The simplest port reads each model's `eval_e200n30.pkl` instead of rolling it out again.
  - Rollouts already log β, w and μ per shift.
- **A4: mean-field error.**
  - `scripts/analytic_model.py` rolls fixed rules with `base.yaml` and `load_region_info`. Switch it to `jasss_config(...)` and `region_for`, then run it.
  - Then run `scripts/meanfield_error.py` and add your concave-curve rows.
  - A7 also wants the mean-field fit at 500 and 1,000 agents. Run the same with `jasss_config("agents_500")` and `jasss_config("agents_1000")`.
  - This regenerates Table 4, Figure 4 and the Section 5.4 sentence (B2).
- **A18: a wider rule box.** Extend `rule_grid` in `scripts/jasss_runs.py`:
  - β up to 6–8 and w up to 3–4, on a finer grid;
  - score on 10 seeds (use `REF_SEEDS`, not the evaluation seeds);
  - check whether concentration jumps near β ≈ 5.
  - Write the output to a new folder, so the cached 3-seed grids are not overwritten.
- **A19: null-model validation.** Compute the Spearman against real counts for:
  - hardship rank alone;
  - crime ∝ hardship;
  - district means spread uniformly over cells;
  - the simulator,

  at district and cell level. Use only the 192 cells with real crime.
  - The real counts are already in git: `region_for(cfg).real_grid` (cells) and `.real_region` (districts).
  - `data/processed_crime.csv` (handover zip) is needed only to group the records differently.
- **C3: a larger Latin hypercube.** Use 150 points on `jasss_base.yaml` and report PRCCs with bootstrap intervals and p-values. No training is needed.
- **Optional, cheap:** prevented-crime contrasts for the A6/A7 settings, if the paper quotes them as tests. Add them to `CONTRASTS` in `scripts/summarise_jasss.py`; Holm then changes for all of them.

### Paper (Phase 4), in the change list's order
- **B1–B4, numbers:**
  - B1: use §3's single PPO number, and label the seeds in every caption.
  - B2: from A4.
  - B3: one night/day definition (hourly, night = 20:00–02:00; 1.17 vs real 1.18).
  - B4: every interval comes from `summarise_jasss.py` with bootstrap seed 0. Write p as "< 0.01".
- **B5, the reference renamed everywhere** (see the name table in §2). Add the sentence that orderings are unchanged under the untreated-crime reference, with the level shift.
- **B6/B7, overclaims and survey range.** Use §3's A14/A17 numbers.
- **B8, why an ABM.** **B9, E4 reframed** with the A13 table, and the Lagrangian in a table. **B10, E3 shortened.**
- **B11, PPO's role.** From A2: Phase 0 found PPO near bang-bang, tracking the time of day. Re-confirm on the new models.
- **B12, new results:** prevented crime, the contrast/N subsection, w off, the second reference.
- **B13, limitations:**
  - the recorded 1.8× is not a latent figure;
  - w is a design choice;
  - N;
  - baseline reporting below the survey range;
  - no displacement;
  - the cost claim depends on the setting (§3).
- **B16–B19, housekeeping:** cross-references, appendices, Figure 1, and verifying every reference.
- **C1, ODD:** Learning, Prediction and Collectives. Runtime: 65–80 min per model on a 4-core laptop, about 10 min on a fast desktop, 0.6–0.9 min per evaluation. Versions are in `requirements-jasss.txt`.
- **C2, validation:** pattern-oriented, plus A19.
- **C4/C5, literature and deterrence anchor.** Verify every citation.
- **C6, data section:**
  - the ACS 2023 population source is in `data/ca_population_acs2023.csv`, from City of Chicago portal t68z-cikk;
  - add the crime records' source, years and counts. They are in §2 under "The crime records", including the note on the sexual-assault type name.
- **D, tables and figures:**
  - the central table (D.3) is `central.csv` restricted to the A10 rows;
  - every figure must show each training run as a point (`per_seed.csv`);
  - new figures: prevented crime by quintile (`quintiles.csv`), and excess against contrast and against N (§3 table).
- **B14 abstract, B15 conclusions: last.** Lead with the mechanism. Give one PPO number with its seed count, crime prevented, and the realistic-contrast result.

### Phase 5
- **A20:**
  - a `scripts/reproduce_all.py` that chains `jasss_runs.py` → `summarise_jasss.py` → figures → PDF;
  - pinned environment: `requirements-jasss.txt`, Python 3.11.9;
  - all seeds listed in one file;
  - a config hash beside results (each model's `config.yaml` is already saved next to it);
  - GitHub Actions running `pytest tests`. **There are now 61 tests, not 49:** update the text.
- **C7:** CoMSES archive with an anonymous peer-review link.
- **C8:** check the live JASSS policies.
- **C9:** anonymisation. The git history, `results/jasss/trained_on.csv` and the run logs contain first names and Windows user paths. Strip them from the archive.
- **C10:** cover letter.

### Readiness gate, current status

| Line | Status |
|---|---|
| One base-PPO number everywhere, with seed count | number ready (5.74×, 10 seeds); text not yet edited |
| Every number regenerates from one command on one tagged commit | results yes (`summarise_jasss.py`); `reproduce_all.py` and the tag still to do |
| No "need-proportional" label on the reference | to do (B5) |
| Crime prevented for every policy in the main table | ready (`prevented_pct` in `central.csv`) |
| Excess at the realistic contrast, N ≥ 500 and w off | ready (§3) |
| No blank cross-reference, empty appendix or placeholder | to do (B16–B18) |
| Abstract matches evidence | to do (B14) |
| Anonymous manuscript, metadata and archive | to do (C9) |

## 6. Practical notes
- **Running long jobs on Windows:**
  - start them from your own PowerShell window, not from an editor or assistant session; those can be stopped under memory pressure;
  - keep the laptop plugged in with sleep off;
  - use 3 workers on 16 GB RAM if anything else is open.
- **The resume save can fail on Windows** with `PermissionError ... resume_tmp -> resume`. This happened 3 times on Dennis's laptop, probably antivirus or indexing holding the folder. Rerunning the same command recovers. If you train again, add a short retry around that rename in `crimesim/rl/ppo.py`.
- **`results/jasss/run.log` is git-ignored** by the global `run.log` rule, so it was force-added. Append-only logs are fine to commit again.
- **The `incoming/` folder on Diya's laptop is a backup** of the teammates' original uploads. It is not in git.
