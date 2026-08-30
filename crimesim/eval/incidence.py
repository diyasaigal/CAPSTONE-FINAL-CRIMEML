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
    last = p[p["shift"] == p["shift"].max()]
    cum = np.zeros(region.n)
    for _, r in p.iterrows():
        cum += np.asarray(r["deployment"], float)
    dep_share = cum / cum.sum()
    exposure = dep_share / region.pop_weight
    return np.array([exposure[g == q].mean() for q in range(n_q)])


def _z(s):
    s = np.asarray(s, float)
    return (s - s.mean()) / (s.std() + 1e-9)
