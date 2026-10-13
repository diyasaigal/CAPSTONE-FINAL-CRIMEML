import crimesim._env_setup  # noqa: F401
"""Plan step 5: does any conclusion depend on choosing the GreedyOracle as reference?

1. Same episodes (results/final), four references: GreedyOracle, Uniform, the
   EWMA-like fixed rule (status-quo hot-spot policing), and the fixed rule with
   the highest latent coverage. Because dlog is paired on eval seeds, switching
   the reference subtracts the same per-seed constant from every policy, so
   orderings and paired contrasts are unchanged by construction; only the level
   (which policies sit above 1) moves. The table shows how far.
2. Two reference-free measures of over-concentration:
     need_tv     total-variation distance between patrol share and latent-crime share
     top_excess  patrol share of the top hardship quintile / its latent-crime share
                 (1 = patrol in proportion to true crime)

Writes results/final/reference_robustness.csv and reference_free.csv. No training.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from crimesim.eval.oracle_relative import iter_episodes
from crimesim.eval.stats import hierarchical_bootstrap

sys.path.insert(0, str(Path(__file__).parent))
from summarise_400k import OUT, load_frames, region  # noqa: E402

REFS = {"oracle": "GreedyOracle", "uniform": "Uniform", "rule_ewma_like": "EWMA-like rule",
        "best_rule_latent": "best fixed rule (latent)"}
SHOW = ["ppo", "ppo_oracle", "ppo_obslat", "ppo_rewlat", "best_rule_recorded", "ewma_hotspot",
        "rf_hotspot", "oracle", "uniform", "rule_ewma_like", "best_rule_latent",
        "pen_p0.06", "penref_w0.15"]


def _ci(g: pd.DataFrame, col: str):
    M = g.pivot_table(index="tseed", columns="seed", values=col, aggfunc="mean").to_numpy()
    return hierarchical_bootstrap(M, n_boot=2000)


def top_excess(frames: dict) -> pd.DataFrame:
    groups = region.quantile_groups("hardship", 5)
    top = groups == groups.max()
    rows = []
    for (fam, s), df in frames.items():
        for ep in iter_episodes(df):
            dep = np.stack(ep["deployment"].to_numpy()).sum(0)
            lat = np.stack(ep["latent_region"].to_numpy()).sum(0)
            rows.append({"family": fam, "tseed": s, "seed": int(ep["seed"].iloc[0]),
                         "top_excess": (dep[top].sum() / dep.sum()) / (lat[top].sum() / lat.sum())})
    return pd.DataFrame(rows)


def main():
    pd.set_option("display.width", 220)
    eps = pd.read_csv(OUT / "episodes.csv")
    rows = []
    for ref_fam, ref_label in REFS.items():
        ref = eps[eps.family == ref_fam].groupby("seed").disparity.mean()
        e = eps.assign(dl=np.log(eps.disparity) - np.log(eps.seed.map(ref)))
        for fam in SHOW:
            m, lo, hi = _ci(e[e.family == fam], "dl")
            rows.append({"reference": ref_label, "family": fam,
                         "x_ref": np.exp(m), "x_ref_lo": np.exp(lo), "x_ref_hi": np.exp(hi)})
    rr = pd.DataFrame(rows)
    rr.to_csv(OUT / "reference_robustness.csv", index=False)
    wide = rr.pivot(index="family", columns="reference", values="x_ref").loc[SHOW, list(REFS.values())]
    print("== x reference (geometric mean of paired ratios) ==")
    print(wide.round(3).to_string())
    order = {r: list(wide[r].sort_values().index) for r in wide.columns}
    print("ordering identical across references:", len({tuple(v) for v in order.values()}) == 1)

    te = top_excess(load_frames())
    free = []
    for fam in SHOW:
        g_te = te[te.family == fam]
        g_tv = eps[eps.family == fam]
        m1, lo1, hi1 = _ci(g_te, "top_excess")
        m2, lo2, hi2 = _ci(g_tv, "need_tv")
        free.append({"family": fam, "top_excess": m1, "top_excess_lo": lo1, "top_excess_hi": hi1,
                     "need_tv": m2, "need_tv_lo": lo2, "need_tv_hi": hi2})
    free = pd.DataFrame(free)
    free.to_csv(OUT / "reference_free.csv", index=False)
    print("\n== reference-free over-concentration ==")
    print(free.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
