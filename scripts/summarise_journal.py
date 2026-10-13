import crimesim._env_setup  # noqa: F401
"""Tables for the journal-readiness runs (scripts/journal_runs.py). No training.

Reads the cached rollouts and writes results/journal/:
  optimisation.csv     naive PPO at 400k (seeds 0-2), the four hyper-parameter
                       variants, the 1.2M run and the Lagrangian check: x oracle
                       with hierarchical-bootstrap CI, coverage, proxy gap
  per_model.csv        one row per trained model (reward vs disparity scatter)
  learning_curve.csv   every snapshot of ppo_long_s*: steps, recorded coverage, x oracle
  reward_disparity.json  rank correlation between reward and disparity
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

from crimesim.eval.oracle_relative import (add_reference_shift, episode_table,
                                           reference_disparity, relative_summary)

sys.path.insert(0, str(Path(__file__).parent))
from journal_runs import EVAL_TAG, HPARAMS, LAG_ALPHA, SEEDS, SNAP_TAG  # noqa: E402
from summarise_400k import RESULTS, RUNS, region  # noqa: E402

OUT = RESULTS / "journal"
FAMILIES = {"ppo": [0, 1, 2], **{k: SEEDS for k in HPARAMS}, "ppo_long": SEEDS,
            f"lag_a{LAG_ALPHA}": SEEDS}
ALL_PPO = {"ppo": [0, 1, 2, 3, 4], **{k: SEEDS for k in HPARAMS}, "ppo_long": SEEDS}


def _oracle_table() -> pd.DataFrame:
    base = pd.read_pickle(RESULTS / f"baselines_{EVAL_TAG}.pkl")
    return episode_table(base[base.policy == "oracle"], region, family="oracle", tseed=-1)


def _table(fam: str, s: int, path: Path) -> pd.DataFrame | None:
    return episode_table(pd.read_pickle(path), region, family=fam, tseed=s) if path.exists() else None


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 220)
    orc = _oracle_table()
    ref = reference_disparity(orc)

    tabs = [t for fam, seeds in ALL_PPO.items() for s in seeds
            if (t := _table(fam, s, RUNS / f"{fam}_s{s}" / f"eval_{EVAL_TAG}.pkl")) is not None]
    lag = f"lag_a{LAG_ALPHA}"
    tabs += [t for s in SEEDS if (t := _table(lag, s, RUNS / f"{lag}_s{s}" / f"eval_{EVAL_TAG}.pkl")) is not None]
    eps = add_reference_shift(pd.concat(tabs, ignore_index=True), ref)

    sub = eps[[s in FAMILIES[f] for f, s in zip(eps.family, eps.tseed)]]
    opt = relative_summary(sub, ["family"], ref)
    opt.to_csv(OUT / "optimisation.csv", index=False)
    print("== optimisation check (x GreedyOracle, eval seeds 200-229) ==")
    print(opt[["family", "n_train_seeds", "x_ref", "x_ref_lo", "x_ref_hi", "rec_cov", "lat_cov",
               "proxy_gap"]].round(4).to_string(index=False))

    per = eps[eps.family != lag].groupby(["family", "tseed"])[["dlog", "rec_cov", "lat_cov"]].mean().reset_index()
    per["x_ref"] = np.exp(per.dlog)
    per.to_csv(OUT / "per_model.csv", index=False)
    rho, p = sps.spearmanr(per.rec_cov, per.dlog)
    info = {"n_models": len(per), "spearman_reward_vs_log_disparity": float(rho), "p": float(p)}

    rows = []
    for s in SEEDS:
        snaps = [("ppo_long", int(d.name), d / f"eval_{SNAP_TAG}.pkl")
                 for d in (RUNS / f"ppo_long_s{s}" / "checkpoints").glob("*")]
        snaps.append(("ppo_400k", 400_000, RUNS / f"ppo_s{s}" / f"eval_{SNAP_TAG}.pkl"))  # the separate 400k model
        for fam, steps, path in sorted(snaps):
            t = _table(fam, s, path)
            if t is None:
                continue
            t = add_reference_shift(t, ref)
            rows.append({"family": fam, "tseed": s, "steps": steps,
                         "rec_cov": t.rec_cov.mean(), "lat_cov": t.lat_cov.mean(),
                         "dlog": t.dlog.mean(), "x_ref": float(np.exp(t.dlog.mean()))})
    if rows:
        lc = pd.DataFrame(rows)
        lc.to_csv(OUT / "learning_curve.csv", index=False)
        cur = lc[lc.family == "ppo_long"]
        mean = cur.groupby("steps")[["rec_cov", "lat_cov", "dlog"]].mean()
        mean["x_ref"] = np.exp(mean.dlog)
        print("\n== learning curve (mean over training seeds, eval seeds 200-209) ==")
        print(mean.round(4).to_string())
        r2, p2 = sps.spearmanr(cur.rec_cov, cur.dlog)
        info.update(n_snapshots=len(cur), spearman_snapshots=float(r2), p_snapshots=float(p2))
    (OUT / "reward_disparity.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
