"""E4 -- efficiency vs equity Pareto frontier for the fairness-constrained policy.

Given a set of trained Lagrangian policies (varying ``alpha``) plus the
unconstrained PPO and the baselines, evaluate each on held-out ABM seeds and
place them in (disparity, latent-crime) space. A point is Pareto-efficient if no
other point has both lower disparity and lower latent crime.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from crimesim.eval.rollout import summarise


def frontier_table(df: pd.DataFrame) -> pd.DataFrame:
    """Efficiency axis = latent-crime coverage (higher is better); equity axis =
    enforcement disparity (lower is better). A policy is Pareto-efficient if no
    other has both higher coverage and lower disparity."""
    s = summarise(df)[["policy", "lat_cov", "lat_cov_ci", "latent", "latent_ci",
                       "disparity", "disparity_ci", "gini"]].copy()
    # minimise (disparity, -lat_cov)
    pts = np.column_stack([s["disparity"].to_numpy(), -s["lat_cov"].to_numpy()])
    dominated = np.zeros(len(s), bool)
    for i in range(len(s)):
        for j in range(len(s)):
            if i != j and (pts[j] <= pts[i]).all() and (pts[j] < pts[i]).any():
                dominated[i] = True
                break
    s["pareto_efficient"] = ~dominated
    return s.sort_values("disparity").reset_index(drop=True)


def knee_point(frontier: pd.DataFrame) -> dict:
    """Among Pareto-efficient points, the one closest to the ideal (min disparity,
    min latent) after min-max normalising each axis."""
    f = frontier[frontier["pareto_efficient"]].copy()
    if len(f) < 2:
        return {} if f.empty else f.iloc[0].to_dict()
    d = (f["disparity"] - f["disparity"].min()) / (f["disparity"].max() - f["disparity"].min() + 1e-9)
    c = (f["lat_cov"].max() - f["lat_cov"]) / (f["lat_cov"].max() - f["lat_cov"].min() + 1e-9)
    f = f.assign(_dist=np.hypot(d, c))
    return f.loc[f["_dist"].idxmin()].to_dict()
