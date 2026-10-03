"""E3 -- socioeconomic incidence of the feedback loop.

Question: after accounting for where crime *actually* is, does a policy still
concentrate patrol on higher-hardship districts?

For each (policy, seed, shift, region) we form

    over_policing = deployment_share - latent_share

and regress it on district hardship, controlling for latent_share (and,
optionally, poverty). A positive, significant hardship coefficient means the
policy over-polices disadvantaged districts *beyond what their true crime
justifies* -- the signature of a discriminatory loop.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from crimesim.abm.regions import RegionInfo
from crimesim.eval.oracle_relative import iter_episodes
from crimesim.eval.stats import ols


def build_panel(df: pd.DataFrame, region: RegionInfo, burn_in: float = 0.3) -> pd.DataFrame:
    rows = []
    tmax = df["shift"].max()
    for (policy, seed), g in df.groupby(["policy", "seed"]):
        g = g[g["shift"] >= burn_in * tmax]
        for _, r in g.iterrows():
            dep = np.asarray(r["deployment"], float)
            lat = np.asarray(r["latent_region"], float)
            dep_s = dep / max(dep.sum(), 1e-9)
            lat_s = lat / max(lat.sum(), 1e-9)
            for j in range(region.n):
                rows.append({
                    "policy": policy, "seed": seed, "shift": int(r["shift"]), "region": j,
                    "over_policing": dep_s[j] - lat_s[j],
                    "latent_share": lat_s[j],
                    "hardship": region.hardship[j],
                    "poverty": region.poverty[j],
                })
    return pd.DataFrame(rows)


def incidence_regression(panel: pd.DataFrame, policy: str, *, with_poverty: bool = False) -> dict:
    p = panel[panel["policy"] == policy]
    X = {"hardship": _z(p["hardship"]), "latent_share": _z(p["latent_share"])}
    if with_poverty:
        X["poverty"] = _z(p["poverty"])
    res = ols(p["over_policing"].to_numpy(), X)
    return {"policy": policy, "hardship": res.get("hardship"),
            "latent_share": res.get("latent_share"), "r2": res.r2, "n": res.n,
            "_result": res}


def exposure_by_hardship_quintile(df: pd.DataFrame, region: RegionInfo, policy: str,
                                  n_q: int = 5) -> np.ndarray:
    """Mean enforcement-per-capita in each hardship quintile (cumulative, final shift)."""
    g = region.quantile_groups("hardship", n_q)
    p = df[df["policy"] == policy]
    cum = np.zeros(region.n)
    for _, r in p.iterrows():
        cum += np.asarray(r["deployment"], float)
    dep_share = cum / cum.sum()
    exposure = dep_share / region.pop_weight
    return np.array([exposure[g == q].mean() for q in range(n_q)])


def long_run_need(df: pd.DataFrame) -> np.ndarray:
    """Latent-crime share per region pooled over all rows of ``df``.

    Pass the rollouts of a neutral policy (Uniform) so that "need" is not itself
    shaped by the policy being tested.
    """
    lat = np.stack(df["latent_region"].to_numpy()).sum(0).astype(float)
    return lat / lat.sum()


def need_controlled_incidence(df: pd.DataFrame, region: RegionInfo, need: np.ndarray, *,
                              burn_in: float = 0.3) -> dict:
    """E3 with a need control and episode-clustered SEs (PIVOT_AUDIT.md B4).

    DV per (episode, shift, region) is patrol share minus long-run need; the
    regressors are z-scored hardship, need and poverty. The per-shift latent
    share used by ``incidence_regression`` is a noisy proxy for need, which
    leaves hardship (corr ~0.7 with need) to absorb the remainder. HC1 SEs over
    region-shifts also treat thousands of correlated rows as independent; here
    every episode is one cluster.
    """
    ys, hard, nd, pov, cl = [], [], [], [], []
    for k, ep in enumerate(iter_episodes(df)):
        ep = ep[ep["shift"] >= burn_in * ep["shift"].max()]
        dep = np.stack(ep["deployment"].to_numpy()).astype(float)
        dep = dep / np.clip(dep.sum(1, keepdims=True), 1e-9, None)
        T = len(dep)
        ys.append((dep - need[None, :]).ravel())
        hard.append(np.tile(region.hardship, T))
        nd.append(np.tile(need, T))
        pov.append(np.tile(region.poverty, T))
        cl.append(np.full(dep.size, k))
    res = ols(np.concatenate(ys), {"hardship": _z(np.concatenate(hard)),
                                   "need": _z(np.concatenate(nd)),
                                   "poverty": _z(np.concatenate(pov))},
              cluster=np.concatenate(cl))
    return {"hardship": res.get("hardship"), "n_clusters": len(cl), "n": res.n,
            "r2": res.r2, "_result": res}


def _z(s):
    s = np.asarray(s, float)
    return (s - s.mean()) / (s.std() + 1e-9)
