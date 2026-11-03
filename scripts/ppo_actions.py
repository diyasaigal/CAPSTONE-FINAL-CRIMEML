import crimesim._env_setup  # noqa: F401
"""JASSS Phase 0, item A2: does naive PPO adapt within an episode?

Table 7 reports only PPO's mean (beta, w, mu). PPO may change its action every
shift, but if its mean action run as a constant rule scores the same, PPO is in
effect a fixed rule. For each naive-PPO training seed this script

  1. rolls the model out on eval seeds 200-229 with the action logged per shift;
  2. measures how much the action moves: SD within an episode, SD of the
     episode means, and the correlation of each parameter with the dispersion
     of the recorded-crime EWMA the rule reads and with the hour of day;
  3. runs the seed's mean action as a fixed rule on the same eval seeds and
     compares recorded coverage, true coverage and disparity x GreedyOracle.

Writes ``results/phase0/``: actions_<run>.pkl, ppo_actions.csv,
ppo_vs_mean_rule.csv, and figures/fig_ppo_actions.{pdf,png}.

    PYTHONPATH=. python scripts/ppo_actions.py --workers 4
"""
import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("results/phase0")
RUNS = Path("runs")
SEEDS = list(range(200, 230))
TSEEDS = [0, 1, 2, 3, 4]
PARAMS = ["beta", "w", "mu"]


def _roll(job):
    kind, key = job
    from crimesim.config import load_config
    from crimesim.eval.rollout import rollout
    cfg = load_config("configs/base.yaml")
    if kind == "ppo":
        cache = OUT / f"actions_ppo_s{key}.pkl"
        if not cache.exists():
            from stable_baselines3 import PPO
            from crimesim.rl.ppo import load_obs_norm
            run = RUNS / f"ppo_s{key}"
            rollout(PPO.load(run / "ppo_model"), cfg, SEEDS, keep_arrays=True,
                    obs_norm=load_obs_norm(run / "obs_rms.npz")).to_pickle(cache)
    else:
        sys.path.insert(0, str(Path(__file__).parent))
        from rule_search import FixedRule
        tseed, (b, w, mu) = key
        cache = OUT / f"actions_meanrule_s{tseed}.pkl"
        if not cache.exists():
            rollout(FixedRule(b, w, mu), cfg, SEEDS, keep_arrays=True).to_pickle(cache)
    return str(cache)


def action_stats(df: pd.DataFrame) -> dict:
    ep = df.groupby("seed")
    row = {}
    for p in PARAMS:
        row[f"{p}_mean"] = df[p].mean()
        row[f"{p}_sd_within"] = ep[p].std().mean()
        row[f"{p}_sd_between"] = ep[p].mean().std()
        row[f"{p}_corr_ewma_cv"] = df[p].corr(df.ewma_cv, method="spearman")
        # share of the action's variance explained by hour of day (eta squared)
        hm = df.groupby("hour")[p].transform("mean")
        row[f"{p}_eta2_hour"] = ((hm - df[p].mean()) ** 2).sum() / max(((df[p] - df[p].mean()) ** 2).sum(), 1e-12)
    return row


def figure(acts: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = {"beta": r"concentration $\beta$", "w": r"hardship tilt $w$", "mu": r"uniform mix $\mu$"}
    lims = {"beta": (0, 4), "w": (-2, 2), "mu": (0, 1)}
    fig, axes = plt.subplots(2, 3, figsize=(10, 5), constrained_layout=True)
    for j, p in enumerate(PARAMS):
        top, bot = axes[0, j], axes[1, j]
        bins = np.linspace(*lims[p], 21)
        for s, df in acts.items():
            prof = df.groupby("hour")[p].mean()
            top.plot(prof.index, prof.values, lw=1.5, marker="o", ms=3, label=f"seed {s}")
            bot.hist(df[p], bins=bins, histtype="step", lw=1.3, density=True)
        top.set_title(labels[p], fontsize=10)
        top.set_ylim(*lims[p])
        top.set_xlabel("hour of day at the end of the shift")
        bot.set_xlabel(f"{labels[p]} chosen in a shift")
        bot.set_yticks([])
    axes[0, 0].set_ylabel("mean over episodes")
    axes[1, 0].set_ylabel("share of shifts")
    axes[0, 0].legend(fontsize=7, frameon=False)
    for ext in ("pdf", "png"):
        fig.savefig(f"figures/fig_ppo_actions.{ext}", dpi=200)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    with ProcessPoolExecutor(args.workers) as ex:
        list(ex.map(_roll, [("ppo", s) for s in TSEEDS]))
    acts = {s: pd.read_pickle(OUT / f"actions_ppo_s{s}.pkl") for s in TSEEDS}
    stats = pd.DataFrame([{"tseed": s, **action_stats(df)} for s, df in acts.items()])
    stats.to_csv(OUT / "ppo_actions.csv", index=False)

    means = {s: tuple(df[PARAMS].mean()) for s, df in acts.items()}
    with ProcessPoolExecutor(args.workers) as ex:
        list(ex.map(_roll, [("rule", (s, means[s])) for s in TSEEDS]))

    from crimesim.abm.regions import load_region_info
    from crimesim.eval.oracle_relative import episode_table
    region = load_region_info(20)
    oracle = pd.read_pickle(Path("results") / "baselines_e200_201n30.pkl").query("policy == 'oracle'")
    ref = episode_table(oracle, region).groupby("seed").disparity.mean()

    rows = []
    for s in TSEEDS:
        for kind, df in [("ppo", acts[s]), ("mean_rule", pd.read_pickle(OUT / f"actions_meanrule_s{s}.pkl"))]:
            t = episode_table(df, region)
            dlog = np.log(t.disparity) - np.log(t.seed.map(ref))
            rows.append({"tseed": s, "policy": kind, "x_oracle": float(np.exp(dlog.mean())),
                         "rec_cov": t.rec_cov.mean(), "lat_cov": t.lat_cov.mean(),
                         **dict(zip(PARAMS, means[s]))})
    cmp = pd.DataFrame(rows)
    cmp.to_csv(OUT / "ppo_vs_mean_rule.csv", index=False)
    figure(acts)

    pd.set_option("display.width", 220)
    print(stats.round(3).T.to_string())
    print(cmp.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
