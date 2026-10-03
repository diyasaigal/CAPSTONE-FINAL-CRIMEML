import crimesim._env_setup  # noqa: F401
"""Pivot step 2 -- re-analyse EXISTING trajectories with the corrected metrics.

No training. Reads the cached per-shift evaluations and writes new tables to
``results/reanalysis/`` (old results/ files are left untouched). The only new
rollouts are rule-based GreedyOracle / Uniform references for the E5 variants,
which never had them.

Metrics (see PIVOT_AUDIT.md section B):
  * disparity  -- cumulative exposure ratio, top / bottom hardship quintile
  * dlog       -- log(disparity) - log(oracle disparity) on the SAME eval seed;
                  "x oracle" = exp(mean dlog), CI by hierarchical bootstrap
                  (training seeds, then eval seeds)
  * need_tv    -- total-variation distance between cumulative patrol share and
                  the episode's own latent-crime share, over all 22 districts
  * proxy_gap  -- recorded coverage - latent coverage
  * rec_distortion -- (recorded/latent share in Q5) / (same in Q1): how much
                  more the top-hardship quintile is over-represented in the data
                  the agent sees. 1 = undistorted.
  * E3         -- hardship coefficient on (patrol share - long-run need) with
                  need and poverty controls and episode-clustered SEs

RL results are reported separately per training budget: s0-s2 were trained for
150k steps, s3-s4 for 400k, and the two must not be pooled.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from crimesim.abm.regions import load_region_info
from crimesim.config import load_config
from crimesim.eval.rollout import rollout
from crimesim.rl.baselines import GreedyOracle, UniformPolicy

RUNS, RESULTS = Path("runs"), Path("results")
OUT = RESULTS / "reanalysis"
TAG = "e200_201n10"
FAMS = ["ppo", "ppo_oracle", "pen_p0.02", "pen_p0.06", "pen_p0.15", "pen_p0.4"]
BUDGET = {0: "150k", 1: "150k", 2: "150k", 3: "400k", 4: "400k"}
B = 4000
rng = np.random.default_rng(0)

region = load_region_info(20)
groups = region.quantile_groups("hardship", 5)
Q1, Q5 = groups == 0, groups == groups.max()
pw = region.pop_weight


def episodes(df):
    df = df.reset_index(drop=True)
    s = df["shift"].to_numpy()
    st = np.where(np.r_[True, np.diff(s) <= 0])[0]
    for a, b in zip(st, np.r_[st[1:], len(df)]):
        yield df.iloc[a:b]


def ep_metrics(ep) -> dict:
    dep = np.stack(ep["deployment"].to_numpy()).sum(0)
    lat = np.stack(ep["latent_region"].to_numpy()).sum(0)
    rec = np.stack(ep["recorded_region"].to_numpy()).sum(0)
    ds, ls, rs = dep / dep.sum(), lat / lat.sum(), rec / rec.sum()
    expo = lambda m: ds[m].sum() / (pw[m].sum() / pw.sum())
    distort = lambda m: rs[m].sum() / ls[m].sum()
    return {"seed": int(ep["seed"].iloc[0]),
            "disparity": expo(Q5) / expo(Q1),
            "need_tv": 0.5 * np.abs(ds - ls).sum(),
            "lat_cov": ep["latent_coverage"].mean(),
            "rec_cov": ep["recorded_coverage"].mean(),
            "rec_distortion": distort(Q5) / distort(Q1)}


def ep_table(df, **labels) -> pd.DataFrame:
    return pd.DataFrame([dict(labels, **ep_metrics(e)) for e in episodes(df)])


def add_dlog(t: pd.DataFrame, oracle: pd.Series) -> pd.DataFrame:
    t["dlog"] = np.log(t.disparity) - np.log(t.seed.map(oracle))
    t["proxy_gap"] = t.rec_cov - t.lat_cov
    return t


def hboot(t: pd.DataFrame, col: str = "dlog"):
    """mean and 95% CI, resampling training seeds then eval seeds."""
    piv = t.pivot_table(index="tseed", columns="seed", values=col)
    M = piv.to_numpy()
    ti = rng.integers(0, M.shape[0], (B, M.shape[0]))
    ei = rng.integers(0, M.shape[1], (B, M.shape[1]))
    bs = np.array([M[np.ix_(a, b)].mean() for a, b in zip(ti, ei)])
    return M.mean(), *np.percentile(bs, [2.5, 97.5])


def summarise(t: pd.DataFrame, keys) -> pd.DataFrame:
    rows = []
    for k, g in t.groupby(keys, sort=False):
        m, lo, hi = hboot(g)
        rows.append(dict(zip(keys, k if isinstance(k, tuple) else (k,)),
                         n_train_seeds=g.tseed.nunique(), n_episodes=len(g),
                         x_oracle=np.exp(m), x_lo=np.exp(lo), x_hi=np.exp(hi),
                         dlog=m, ratio_of_means=np.nan,
                         disparity=g.disparity.mean(), need_tv=g.need_tv.mean(),
                         lat_cov=g.lat_cov.mean(), rec_cov=g.rec_cov.mean(),
                         proxy_gap=g.proxy_gap.mean(),
                         rec_distortion=g.rec_distortion.mean()))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ E1 / E2 / E4
def main_tables():
    base = pd.read_pickle(RESULTS / f"baselines_{TAG}.pkl")
    bt = pd.concat([ep_table(g, family=p, tseed=-1) for p, g in base.groupby("policy", sort=False)])
    oracle = bt[bt.family == "oracle"].set_index("seed")["disparity"]
    bt = add_dlog(bt, oracle).assign(budget="-")
    rl = pd.concat([ep_table(pd.read_pickle(RUNS / f"{f}_s{s}" / f"eval_{TAG}.pkl"),
                             family=f, tseed=s, budget=BUDGET[s])
                    for f in FAMS for s in range(5)])
    rl = add_dlog(rl, oracle)
    eps = pd.concat([bt, rl], ignore_index=True)
    summ = summarise(eps, ["family", "budget"])
    for i, r in summ.iterrows():
        g = eps[(eps.family == r.family) & (eps.budget == r.budget)]
        summ.loc[i, "ratio_of_means"] = g.disparity.mean() / oracle.loc[g.seed.unique()].mean()
    per_seed = eps.groupby(["family", "budget", "tseed"], sort=False)[
        ["disparity", "dlog", "need_tv", "lat_cov", "rec_cov", "proxy_gap", "rec_distortion"]].mean()
    return eps, summ, per_seed, base


# ------------------------------------------------------------------ E3
def _z(x):
    x = np.asarray(x, float)
    return (x - x.mean()) / (x.std() + 1e-12)


def clustered_ols(y, X, cl):
    Xm = np.column_stack([np.ones(len(y))] + [_z(c) for c in X])
    XtXi = np.linalg.pinv(Xm.T @ Xm)
    b = XtXi @ Xm.T @ y
    e = y - Xm @ b
    meat = np.zeros((Xm.shape[1],) * 2)
    for g in np.unique(cl):
        s = Xm[cl == g].T @ e[cl == g]
        meat += np.outer(s, s)
    G, (n, k) = len(np.unique(cl)), Xm.shape
    V = XtXi @ meat @ XtXi * G / (G - 1) * (n - 1) / (n - k)
    return b[1], np.sqrt(V[1, 1]), G


def e3(base):
    uni = base[base.policy == "uniform"]
    need = np.stack(uni["latent_region"].to_numpy()).sum(0)
    need = need / need.sum()
    frames = {(p, "-"): base[base.policy == p] for p in ["ewma_hotspot", "rf_hotspot", "oracle", "random"]}
    for f in FAMS:
        for budget, seeds in [("150k", [0, 1, 2]), ("400k", [3, 4])]:
            frames[(f, budget)] = pd.concat(
                [pd.read_pickle(RUNS / f"{f}_s{s}" / f"eval_{TAG}.pkl").assign(_t=s) for s in seeds])
    rows = []
    for (fam, budget), df in frames.items():
        parts, k = [], 0
        for ep in episodes(df):
            k += 1
            ep = ep[ep["shift"] >= 0.3 * ep["shift"].max()]
            dep = np.stack(ep["deployment"].to_numpy())
            dep = dep / dep.sum(1, keepdims=True)
            lat = np.stack(ep["latent_region"].to_numpy())
            lat = lat / np.clip(lat.sum(1, keepdims=True), 1e-9, None)
            T, R = dep.shape
            parts.append(pd.DataFrame({"ep": k, "y_lat": (dep - lat).ravel(),
                                       "y_need": (dep - need).ravel(), "lat": lat.ravel(),
                                       "need": np.tile(need, T), "hard": np.tile(region.hardship, T),
                                       "pov": np.tile(region.poverty, T)}))
        P = pd.concat(parts, ignore_index=True)
        b0, s0, G = clustered_ols(P.y_lat.to_numpy(), [P.hard, P.lat, P.pov], P.ep.to_numpy())
        b1, s1, _ = clustered_ols(P.y_need.to_numpy(), [P.hard, P.need, P.pov], P.ep.to_numpy())
        rows.append({"family": fam, "budget": budget, "n_episodes": G,
                     "old_spec_coef": b0, "old_spec_t_clustered": b0 / s0,
                     "need_ctrl_coef": b1, "need_ctrl_se": s1, "need_ctrl_t": b1 / s1})
    return pd.DataFrame(rows), need


# ------------------------------------------------------------------ E5
E5_OVER = {
    "base": {}, "perfect_obs": {"detection": {"p_report": 1.0, "detect_gain": 0.0}},
    "high_detect_gain": {"detection": {"detect_gain": 0.8}},
    "weak_deterrence": {"detection": {"deterrence_slope": 0.10}},
    "strong_deterrence": {"detection": {"deterrence_slope": 0.30}},
    "no_socio_obs": {"env": {"include_socioeconomic_obs": False}},
}


def e5():
    raw = pd.read_pickle(RESULTS / "e5_ablations_raw.pkl")
    seeds = sorted(raw.seed.unique())
    tabs = []
    for v, over in E5_OVER.items():
        cache = OUT / f"e5_oracle_{v}_e{seeds[0]}n{len(seeds)}.pkl"
        if cache.exists():
            orc = pd.read_pickle(cache)
        else:
            orc = rollout(GreedyOracle(), load_config("configs/base.yaml", **over), seeds, keep_arrays=True)
            orc.to_pickle(cache)
        ot = ep_table(orc, variant=v, kind="oracle", tseed=-1)
        o = ot.set_index("seed")["disparity"]
        tabs.append(add_dlog(ot, o))
        for kind, g in raw[raw.variant == v].groupby("kind"):
            tabs.append(add_dlog(ep_table(g, variant=v, kind=kind, tseed=int(g.tseed.iloc[0])), o))
    eps = pd.concat(tabs, ignore_index=True)
    rows = []
    for (v, kind), g in eps.groupby(["variant", "kind"], sort=False):
        d = g.dlog.to_numpy()
        bs = rng.choice(d, (B, len(d))).mean(1)
        rows.append({"variant": v, "kind": kind, "n_episodes": len(g),
                     "oracle_disparity": g.disparity.mean() if kind == "oracle" else np.nan,
                     "disparity": g.disparity.mean(),
                     "x_oracle": np.exp(d.mean()),
                     "x_lo": np.exp(np.percentile(bs, 2.5)), "x_hi": np.exp(np.percentile(bs, 97.5)),
                     "per_episode_disparity": " ".join(f"{x:.2f}" for x in g.disparity),
                     "lat_cov": g.lat_cov.mean(), "proxy_gap": g.proxy_gap.mean(),
                     "rec_distortion": g.rec_distortion.mean()})
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 220)
    eps, summ, per_seed, base = main_tables()
    eps.to_csv(OUT / "episodes_e200n10.csv", index=False)
    summ.to_csv(OUT / "e1_e4_oracle_relative.csv", index=False)
    per_seed.to_csv(OUT / "e1_per_training_seed.csv")
    print("== E1/E2/E4: oracle-relative, eval seeds 200-209 ==")
    print(summ.drop(columns=["n_episodes"]).round(4).to_string(index=False))
    print("\n== per training seed ==")
    print(per_seed.round(4).to_string())

    e3t, need = e3(base)
    e3t.to_csv(OUT / "e3_need_controlled.csv", index=False)
    print("\n== E3: hardship coefficient, episode-clustered ==")
    print(e3t.round(4).to_string(index=False))

    e5t = e5()
    e5t.to_csv(OUT / "e5_oracle_relative.csv", index=False)
    print("\n== E5: each variant vs its own GreedyOracle (1 training seed, 120k, 6 eval seeds) ==")
    print(e5t.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
