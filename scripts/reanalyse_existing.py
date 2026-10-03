import crimesim._env_setup  # noqa: F401
"""Pivot Phase 2 -- re-analyse EXISTING trajectories with the corrected metrics.

No training. Reads the cached per-shift evaluations and writes new tables to
``results/reanalysis/`` (old results/ files are left untouched). The only new
rollouts are rule-based GreedyOracle references for the E5 variants, which
never had them.

Metric definitions live in ``crimesim.eval.oracle_relative`` (E1/E2/E4/E5) and
``crimesim.eval.incidence.need_controlled_incidence`` (E3); see PIVOT_AUDIT.md
section B.

RL results are reported separately per training budget: s0-s2 were trained for
150k steps, s3-s4 for 400k, and the two must not be pooled.
"""
from pathlib import Path

import pandas as pd

from crimesim.abm.regions import load_region_info
from crimesim.config import load_config
from crimesim.eval.incidence import build_panel, incidence_regression, long_run_need, need_controlled_incidence
from crimesim.eval.oracle_relative import (add_reference_shift, episode_table,
                                           reference_disparity, relative_summary)
from crimesim.eval.rollout import rollout
from crimesim.eval.stats import ols
from crimesim.rl.baselines import GreedyOracle

RUNS, RESULTS = Path("runs"), Path("results")
OUT = RESULTS / "reanalysis"
TAG = "e200_201n10"
FAMS = ["ppo", "ppo_oracle", "pen_p0.02", "pen_p0.06", "pen_p0.15", "pen_p0.4"]
BUDGETS = {"150k": [0, 1, 2], "400k": [3, 4]}
region = load_region_info(20)


def load_rl(fam: str, s: int) -> pd.DataFrame:
    return pd.read_pickle(RUNS / f"{fam}_s{s}" / f"eval_{TAG}.pkl")


# ------------------------------------------------------------------ E1 / E2 / E4
def main_tables(base: pd.DataFrame):
    bt = pd.concat([episode_table(g, region, family=p, budget="-", tseed=-1)
                    for p, g in base.groupby("policy", sort=False)])
    ref = reference_disparity(bt[bt.family == "oracle"])
    rl = pd.concat([episode_table(load_rl(f, s), region, family=f, budget=b, tseed=s)
                    for f in FAMS for b, seeds in BUDGETS.items() for s in seeds])
    eps = add_reference_shift(pd.concat([bt, rl], ignore_index=True), ref)
    summ = relative_summary(eps, ["family", "budget"], ref)
    per_seed = eps.groupby(["family", "budget", "tseed"], sort=False)[
        ["disparity", "dlog", "need_tv", "lat_cov", "rec_cov", "proxy_gap", "rec_distortion"]].mean()
    return eps, summ, per_seed


# ------------------------------------------------------------------ E3
def _old_spec_clustered(df: pd.DataFrame) -> dict:
    """The published E3 specification, but with episode-clustered SEs."""
    from crimesim.eval.oracle_relative import episode_ids
    df = df.reset_index(drop=True).assign(_ep=lambda d: episode_ids(d))
    rows = []
    for ep_id, g in df.groupby("_ep"):
        p = build_panel(g.assign(policy="x"), region)
        rows.append(p.assign(ep=ep_id))
    P = pd.concat(rows, ignore_index=True)
    z = lambda s: (s - s.mean()) / s.std(ddof=0)
    res = ols(P.over_policing.to_numpy(), {"hardship": z(P.hardship), "latent_share": z(P.latent_share),
                                           "poverty": z(P.poverty)}, cluster=P.ep.to_numpy())
    return res.get("hardship")


def e3(base: pd.DataFrame) -> pd.DataFrame:
    need = long_run_need(base[base.policy == "uniform"])
    frames = {(p, "-"): base[base.policy == p] for p in ["ewma_hotspot", "rf_hotspot", "oracle", "random"]}
    for f in FAMS:
        for b, seeds in BUDGETS.items():
            frames[(f, b)] = pd.concat([load_rl(f, s) for s in seeds], ignore_index=True)
    rows = []
    for (fam, b), df in frames.items():
        new = need_controlled_incidence(df, region, need)
        old = _old_spec_clustered(df)
        rows.append({"family": fam, "budget": b, "n_episodes": new["n_clusters"],
                     "old_spec_coef": old["coef"], "old_spec_t_clustered": old["t"],
                     "need_ctrl_coef": new["hardship"]["coef"], "need_ctrl_se": new["hardship"]["se"],
                     "need_ctrl_t": new["hardship"]["t"], "need_ctrl_p": new["hardship"]["p"]})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ E5
E5_OVER = {
    "base": {}, "perfect_obs": {"detection": {"p_report": 1.0, "detect_gain": 0.0}},
    "high_detect_gain": {"detection": {"detect_gain": 0.8}},
    "weak_deterrence": {"detection": {"deterrence_slope": 0.10}},
    "strong_deterrence": {"detection": {"deterrence_slope": 0.30}},
    "no_socio_obs": {"env": {"include_socioeconomic_obs": False}},
}


def e5() -> pd.DataFrame:
    raw = pd.read_pickle(RESULTS / "e5_ablations_raw.pkl")
    seeds = sorted(raw.seed.unique())
    out = []
    for v, over in E5_OVER.items():
        cache = OUT / f"e5_oracle_{v}_e{seeds[0]}n{len(seeds)}.pkl"
        if cache.exists():
            orc = pd.read_pickle(cache)
        else:
            orc = rollout(GreedyOracle(), load_config("configs/base.yaml", **over), seeds, keep_arrays=True)
            orc.to_pickle(cache)
        ot = episode_table(orc, region, variant=v, kind="oracle", tseed=-1)
        ref = reference_disparity(ot)
        tabs = [ot] + [episode_table(g, region, variant=v, kind=k, tseed=int(g.tseed.iloc[0]))
                       for k, g in raw[raw.variant == v].groupby("kind")]
        eps = add_reference_shift(pd.concat(tabs, ignore_index=True), ref)
        s = relative_summary(eps, ["variant", "kind"], ref)
        s["per_episode_disparity"] = [" ".join(f"{x:.2f}" for x in eps[eps.kind == k].disparity)
                                      for k in s.kind]
        out.append(s)
    return pd.concat(out, ignore_index=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 240)
    base = pd.read_pickle(RESULTS / f"baselines_{TAG}.pkl")
    cols = ["family", "budget", "n_train_seeds", "x_ref", "x_ref_lo", "x_ref_hi", "ratio_of_means",
            "disparity", "need_tv", "lat_cov", "rec_cov", "proxy_gap", "rec_distortion"]

    eps, summ, per_seed = main_tables(base)
    eps.to_csv(OUT / "episodes_e200n10.csv", index=False)
    summ.to_csv(OUT / "e1_e4_oracle_relative.csv", index=False)
    per_seed.to_csv(OUT / "e1_per_training_seed.csv")
    print("== E1/E2/E4: x GreedyOracle disparity, eval seeds 200-209 ==")
    print(summ[cols].round(4).to_string(index=False))
    print("\n== per training seed ==")
    print(per_seed.round(4).to_string())

    e3t = e3(base)
    e3t.to_csv(OUT / "e3_need_controlled.csv", index=False)
    print("\n== E3: hardship coefficient, episode-clustered ==")
    print(e3t.round(4).to_string(index=False))

    e5t = e5()
    e5t.to_csv(OUT / "e5_oracle_relative.csv", index=False)
    print("\n== E5: each variant vs its own GreedyOracle (1 training seed, 120k, 6 eval seeds) ==")
    print(e5t[["variant", "kind", "x_ref", "x_ref_lo", "x_ref_hi", "disparity", "per_episode_disparity",
               "lat_cov", "proxy_gap", "rec_distortion"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
