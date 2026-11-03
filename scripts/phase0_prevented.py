import crimesim._env_setup  # noqa: F401
"""JASSS Phase 0, item A1: total crime prevented per policy.

Coverage share is not crime prevented: a district that patrol suppresses loses
crime and therefore loses share. Here every policy is compared with a paired
no-patrol run on the same evaluation seed (zero deployment: no deterrence,
detection at baseline; warm-up under proportional patrol as for every policy):

    prevented_pct = 100 * (latent_no_patrol - latent_policy) / latent_no_patrol

per episode, city-wide and per hardship quintile, then averaged with the same
hierarchical bootstrap as the disparity ratios. Reads the caches used by
``summarise_400k.py`` (no training) and writes ``results/phase0/``:

  prevented.csv           one row per family: prevented %, CI, total latent per shift
  prevented_quintile.csv  prevented % per hardship quintile (1 = least disadvantaged)
  prevented_episodes.csv  one row per (family, training seed, eval seed)

    PYTHONPATH=. python scripts/phase0_prevented.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from crimesim.config import load_config
from crimesim.eval.oracle_relative import iter_episodes
from crimesim.eval.rollout import rollout
from crimesim.eval.stats import hierarchical_bootstrap
from crimesim.rl.baselines import NoPatrolPolicy

sys.path.insert(0, str(Path(__file__).parent))
import summarise_400k as s4  # noqa: E402

OUT = Path("results/phase0")
N_Q = 5


def no_patrol_frame() -> pd.DataFrame:
    cache = OUT / f"no_patrol_{s4.TAG}.pkl"
    if not cache.exists():
        rollout(NoPatrolPolicy(), load_config("configs/base.yaml"), s4.SEEDS,
                keep_arrays=True).to_pickle(cache)
    return pd.read_pickle(cache)


def latent_by_episode(df: pd.DataFrame, groups: np.ndarray) -> pd.DataFrame:
    """Per episode: mean latent crime per shift, city-wide and per quintile."""
    rows = []
    for ep in iter_episodes(df):
        lat = np.stack(ep["latent_region"].to_numpy()).mean(0)
        rows.append({"seed": int(ep["seed"].iloc[0]), "latent": lat.sum(),
                     **{f"latent_q{g + 1}": lat[groups == g].sum() for g in range(N_Q)}})
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    groups = s4.region.quantile_groups("hardship", N_Q)
    base = latent_by_episode(no_patrol_frame(), groups).groupby("seed").mean()

    tabs = []
    for (fam, tseed), df in s4.load_frames().items():
        t = latent_by_episode(df, groups)
        for col in ["latent", *(f"latent_q{g + 1}" for g in range(N_Q))]:
            ref = t["seed"].map(base[col])
            t[col.replace("latent", "prev")] = 100 * (ref - t[col]) / ref
        tabs.append(t.assign(family=fam, tseed=tseed))
    eps = pd.concat(tabs, ignore_index=True)
    eps.to_csv(OUT / "prevented_episodes.csv", index=False)

    def boot(g, col):
        M = g.pivot_table(index="tseed", columns="seed", values=col, aggfunc="mean").to_numpy()
        return hierarchical_bootstrap(M, n_boot=s4.N_BOOT, seed=0)

    rows, qrows = [], []
    for fam, g in eps.groupby("family", sort=False):
        m, lo, hi = boot(g, "prev")
        per_seed = g.groupby("tseed").prev.mean()
        rows.append({"family": fam, "n_train_seeds": g.tseed.nunique() if g.tseed.max() >= 0 else 0,
                     "latent_per_shift": g.latent.mean(), "prevented_pct": m,
                     "prevented_lo": lo, "prevented_hi": hi,
                     "seed_min": per_seed.min(), "seed_max": per_seed.max()})
        qrows.append({"family": fam, **{f"q{q + 1}": g[f"prev_q{q + 1}"].mean() for q in range(N_Q)}})
    out = pd.DataFrame(rows).sort_values("prevented_pct", ascending=False)
    out.to_csv(OUT / "prevented.csv", index=False)
    pd.DataFrame(qrows).to_csv(OUT / "prevented_quintile.csv", index=False)

    # Gate 0: does concentrating patrol prevent less crime than the reference?
    pairs = [("ppo", "oracle"), ("ppo", "ewma_hotspot"), ("ppo", "uniform"),
             ("best_rule_recorded", "oracle"), ("ppo_oracle", "ppo"), ("penref_w0.15", "ppo"),
             ("pen_p0.06", "oracle")]
    con = pd.DataFrame([s4.contrast(eps, a, b, col="prev") for a, b in pairs])
    con.to_csv(OUT / "prevented_contrasts.csv", index=False)
    print(con.round(4).to_string(index=False))

    pd.set_option("display.width", 200)
    print(f"no-patrol latent crime per shift: {base.latent.mean():.2f}")
    print(out.round(3).to_string(index=False))
    print(pd.DataFrame(qrows).round(2).to_string(index=False))


if __name__ == "__main__":
    main()
