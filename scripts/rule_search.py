import crimesim._env_setup  # noqa: F401
"""Non-RL baseline: the best FIXED patrol rule, found by grid search.

PPO acts through a 3-parameter allocation rule (``PolicingEnv.action_to_deployment``):
patrol shares proportional to (EWMA recorded crime)^beta * exp(w * z_hardship),
mixed with uniform by mu. Here the parameters are held constant for a whole
episode and searched on a grid, scored on SEARCH seeds that are disjoint from
every eval set.

Questions it answers (PIVOT_AUDIT.md, decision 5):
  * Is PPO under-optimised? If a fixed rule beats PPO on PPO's own reward
    (recorded coverage), PPO has not converged to the best policy in its class.
  * Is the bias specific to RL? The rule that maximises recorded coverage is
    evaluated with the same oracle-relative metrics as PPO. If it is just as
    skewed, the proxy objective, not the learner, produces the disparity.

Outputs ``results/rule_search/``: grid.csv (every rule on the search seeds) and
eval_e200n10.csv (best rules re-evaluated on eval seeds 200-209 next to the
cached baselines). No training.
"""
import argparse
import itertools
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS = Path("results")
OUT = RESULTS / "rule_search"


class FixedRule:
    """Constant action in the env's rule action space (looks like an SB3 model to ``rollout``)."""

    def __init__(self, beta: float, w: float, mu: float):
        self.beta, self.w, self.mu = beta, w, mu
        self.name = f"rule_b{beta:g}_w{w:g}_m{mu:g}"
        self._a = np.array([beta / 2.0 - 1.0, w / 2.0, 2.0 * mu - 1.0], np.float32)

    def predict(self, obs, deterministic=True):
        return self._a, None


def _score(args):
    beta, w, mu, seeds = args
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    from crimesim.config import load_config
    from crimesim.eval.rollout import rollout
    df = rollout(FixedRule(beta, w, mu), load_config("configs/base.yaml"), seeds)
    ep = df.groupby("seed")[["recorded_coverage", "latent_coverage", "disparity_cum"]].agg(
        {"recorded_coverage": "mean", "latent_coverage": "mean", "disparity_cum": "last"})
    return {"beta": beta, "w": w, "mu": mu,
            "rec_cov": ep.recorded_coverage.mean(), "rec_cov_sd": ep.recorded_coverage.std(),
            "lat_cov": ep.latent_coverage.mean(), "disparity": ep.disparity_cum.mean()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--search-seeds", type=int, nargs="+", default=[1000, 1001, 1002])
    ap.add_argument("--eval-seeds", type=int, nargs="+", default=list(range(200, 210)))
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    betas = [0, 0.5, 1, 1.5, 2, 2.5, 3, 4]
    ws = [-2, -1, -0.5, 0, 0.5, 1, 1.5, 2]
    mus = [0, 0.25, 0.5]
    grid_csv = OUT / "grid.csv"
    if grid_csv.exists():
        grid = pd.read_csv(grid_csv)
    else:
        jobs = [(b, w, m, args.search_seeds) for b, w, m in itertools.product(betas, ws, mus)]
        with ProcessPoolExecutor(args.workers) as ex:
            grid = pd.DataFrame(list(ex.map(_score, jobs)))
        grid.to_csv(grid_csv, index=False)
    print(f"{len(grid)} rules on search seeds {args.search_seeds}")
    print("\ntop 5 by recorded coverage (PPO's reward):")
    print(grid.nlargest(5, "rec_cov").round(4).to_string(index=False))
    print("\ntop 5 by latent coverage (the real objective):")
    print(grid.nlargest(5, "lat_cov").round(4).to_string(index=False))

    # re-evaluate the winners on the eval seeds, scored like every other policy
    from crimesim.abm.regions import load_region_info
    from crimesim.config import load_config
    from crimesim.eval.oracle_relative import (add_reference_shift, episode_table,
                                               reference_disparity, relative_summary)
    from crimesim.eval.rollout import rollout
    region = load_region_info(20)
    tag = f"e{args.eval_seeds[0]}_{args.eval_seeds[1]}n{len(args.eval_seeds)}"
    base = pd.read_pickle(RESULTS / f"baselines_{tag}.pkl")
    picks = {"best_rule_recorded": grid.loc[grid.rec_cov.idxmax()],
             "best_rule_latent": grid.loc[grid.lat_cov.idxmax()],
             "rule_ewma_like": grid[(grid.beta == 1) & (grid.w == 0) & (grid.mu == 0)].iloc[0]}
    tabs = [episode_table(g, region, family=p, tseed=-1) for p, g in base.groupby("policy", sort=False)]
    for label, r in picks.items():
        df = rollout(FixedRule(r.beta, r.w, r.mu), load_config("configs/base.yaml"),
                     args.eval_seeds, keep_arrays=True)
        tabs.append(episode_table(df, region, family=f"{label} (b={r.beta:g}, w={r.w:g}, mu={r.mu:g})",
                                  tseed=-1))
    eps = pd.concat(tabs, ignore_index=True)
    ref = reference_disparity(eps[eps.family == "oracle"])
    summ = relative_summary(add_reference_shift(eps, ref), ["family"], ref)
    summ.to_csv(OUT / f"eval_e{args.eval_seeds[0]}n{len(args.eval_seeds)}.csv", index=False)
    pd.set_option("display.width", 220)
    print(f"\n== eval seeds {args.eval_seeds[0]}-{args.eval_seeds[-1]}: x GreedyOracle disparity ==")
    print(summ[["family", "x_ref", "x_ref_lo", "x_ref_hi", "disparity", "lat_cov", "rec_cov",
                "proxy_gap", "need_tv"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
