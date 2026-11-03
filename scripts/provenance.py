import crimesim._env_setup  # noqa: F401
"""JASSS Phase 0, item A3: one row per trained model, with where it is used.

The same base PPO appears as 3.91x (Table 6, 5 seeds, eval 200-229), 4.85x
(Tables 8 and 10, seeds 0-2, eval 200-229) and 4.92x (Table 11, seeds 0-2,
eval 300-319). This table makes every model's own value visible.

Sources (no training, no new rollouts except what phase0_prevented.py cached):
  * every ``runs/<name>/eval_e200_201n30.pkl``, scored against the GreedyOracle on
    eval seeds 200-229, with crime prevented against the paired no-patrol run;
  * ``results/e5_400k_e300n20_episodes.csv`` and ``..._sens_episodes.csv`` for the
    E5 and sensitivity models (eval seeds 300-319, each against its own oracle).

Writes ``results/phase0/provenance.csv`` and ``provenance_families.csv``
(geometric mean, median and range over training seeds).

    PYTHONPATH=. python scripts/provenance.py
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd

from crimesim.abm.regions import load_region_info
from crimesim.config import load_config
from crimesim.eval.oracle_relative import episode_table, iter_episodes

RUNS, RESULTS, OUT = Path("runs"), Path("results"), Path("results/phase0")
TAG = "e200_201n30"

# which paper table each family feeds (JASSS draft numbering)
SOURCE = {"ppo": "T6 (seeds 0-4); T8, T10 (seeds 0-2)", "ppo_oracle": "T6, T10", "ppo_obslat": "T10",
          "ppo_rewlat": "T10", "pen_p": "T6", "penref_w": "T6", "lag_a": "text (Lagrangian)",
          "ppo_long": "T8", "ppo_lr": "T8", "ppo_ent0": "T8", "ppo_bigbatch": "T8"}


def source(fam: str) -> str:
    for k, v in sorted(SOURCE.items(), key=lambda kv: -len(kv[0])):
        if fam.startswith(k):
            return v
    return ""


def main():
    region = load_region_info(20)
    groups = region.quantile_groups("hardship", 5)
    base = pd.read_pickle(RESULTS / f"baselines_{TAG}.pkl")
    ref = episode_table(base[base.policy == "oracle"], region).groupby("seed").disparity.mean()
    nop = pd.read_pickle(OUT / f"no_patrol_{TAG}.pkl")
    nop_lat = pd.Series({int(ep.seed.iloc[0]): ep.latent_total.mean() for ep in iter_episodes(nop)})

    rows = []
    for run in sorted(RUNS.glob(f"*/eval_{TAG}.pkl")):
        name = run.parent.name
        m = re.match(r"(.+)_s(\d+)$", name)
        if not m or name.startswith("abl_"):
            continue
        fam, tseed = m.group(1), int(m.group(2))
        cfg = load_config(run.parent / "config.yaml")
        df = pd.read_pickle(run)
        t = episode_table(df, region)
        lat = pd.Series({int(ep.seed.iloc[0]): ep.latent_total.mean() for ep in iter_episodes(df)})
        prev = 100 * (nop_lat[lat.index] - lat) / nop_lat[lat.index]
        rows.append({"family": fam, "tseed": tseed, "algo": cfg.rl.algo, "steps": cfg.rl.total_timesteps,
                     "setting": "base", "eval_seeds": "200-229", "reference": "GreedyOracle, base",
                     "x_ref": float(np.exp((np.log(t.disparity) - np.log(t.seed.map(ref))).mean())),
                     "rec_cov": t.rec_cov.mean(), "lat_cov": t.lat_cov.mean(),
                     "prevented_pct": prev.mean(), "source": source(fam)})

    for f, label in [("e5_400k_e300n20_episodes.csv", "T11"), ("e5_400k_e300n20_sens_episodes.csv", "T12")]:
        e = pd.read_csv(RESULTS / f)
        e = e[e.kind != "oracle"]
        for (variant, kind, tseed), g in e.groupby(["variant", "kind", "tseed"]):
            if label == "T12" and variant in ("base",):
                continue        # same models as T11's base row
            rows.append({"family": f"e5_{kind}" if label == "T11" else f"sens_{kind}", "tseed": int(tseed),
                         "algo": "ppo" if kind == "ppo" else "penalized_ppo", "steps": 400_000,
                         "setting": variant, "eval_seeds": "300-319", "reference": f"GreedyOracle, {variant}",
                         "x_ref": float(np.exp(g.dlog.mean())), "rec_cov": g.rec_cov.mean(),
                         "lat_cov": g.lat_cov.mean(), "prevented_pct": np.nan, "source": label})

    prov = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    prov.to_csv(OUT / "provenance.csv", index=False)

    fams = (prov.groupby(["family", "setting", "eval_seeds"], sort=False)
            .agg(n=("tseed", "size"), seeds=("tseed", lambda s: " ".join(map(str, sorted(s)))),
                 x_geo=("x_ref", lambda x: float(np.exp(np.log(x).mean()))),
                 x_median=("x_ref", "median"), x_min=("x_ref", "min"), x_max=("x_ref", "max"),
                 prevented=("prevented_pct", "mean"), source=("source", "first"))
            .reset_index())
    fams.to_csv(OUT / "provenance_families.csv", index=False)
    pd.set_option("display.width", 220)
    print(fams.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
