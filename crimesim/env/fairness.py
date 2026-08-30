"""Fairness / feedback-loop metrics on a patrol allocation.

Central quantity: **enforcement exposure per capita** for each region,

    exposure_r = deployment_share_r / population_share_r

where population share is proxied by the region's share of *real* violent crime.
``exposure_r == 1`` means the region is policed exactly in proportion to its
size; a runaway loop drives exposure up in the most disadvantaged regions.
"""
from __future__ import annotations

import numpy as np

from crimesim.abm.regions import RegionInfo
from crimesim.config import FairnessConfig


def exposure_per_capita(deployment: np.ndarray, pop_weight: np.ndarray) -> np.ndarray:
    dep_share = deployment / max(deployment.sum(), 1e-9)
    return dep_share / np.clip(pop_weight, 1e-9, None)


def gini(x: np.ndarray) -> float:
    x = np.sort(np.asarray(x, float))
    n = len(x)
    if x.sum() == 0:
        return 0.0
    idx = np.arange(1, n + 1)
    return float((2 * (idx * x).sum() - (n + 1) * x.sum()) / (n * x.sum()))


def _group_exposure(deployment, pop_weight, groups, g):
    m = groups == g
    dep = deployment[m].sum() / max(deployment.sum(), 1e-9)
    pop = pop_weight[m].sum() / max(pop_weight.sum(), 1e-9)
    return dep / max(pop, 1e-9)


class FairnessMeter:
    """Per-shift and cumulative fairness metrics for one rollout."""

    def __init__(self, cfg: FairnessConfig, region: RegionInfo):
        self.cfg = cfg
        self.alpha = float(cfg.alpha)           # disparity ceiling for the Lagrangian cost
        self.pop_weight = region.pop_weight
        self.groups = region.quantile_groups(cfg.protected_attr, cfg.n_quantiles)
        self.hi = int(self.groups.max())        # most-disadvantaged group
        self.lo = 0                             # least-disadvantaged group
        self.reset()

    def reset(self) -> None:
        self.cum_deploy = np.zeros(len(self.pop_weight))

    def _disparity(self, deployment: np.ndarray) -> float:
        if self.cfg.metric == "gini":
            return gini(exposure_per_capita(deployment, self.pop_weight))
        if self.cfg.metric == "max_min":
            e = exposure_per_capita(deployment, self.pop_weight)
            return float(e.max() / max(e.mean(), 1e-9))
        hi = _group_exposure(deployment, self.pop_weight, self.groups, self.hi)
        lo = _group_exposure(deployment, self.pop_weight, self.groups, self.lo)
        return float(hi / max(lo, 1e-9))

    def update(self, deployment: np.ndarray) -> dict:
        self.cum_deploy += deployment
        shift = self._disparity(deployment)
        cum = self._disparity(self.cum_deploy)
        # Lagrangian cost: hinge on the *cumulative* disparity exceeding the
        # ceiling alpha. Once disparity_cum <= alpha the cost and its gradient are
        # zero, so there is no incentive to push disparity below alpha -- the
        # policy settles at disparity ~ alpha rather than collapsing to 0.
        # Hinge on the *cumulative* disparity exceeding the ceiling alpha. With one
        # PPO rollout == one deployment episode (see configs/base.yaml), the rollout
        # mean of this cost tracks the end-of-deployment enforcement pattern we
        # report, and there is no gradient once disparity_cum <= alpha (so the
        # policy settles near alpha rather than collapsing to 0).
        cost = max(0.0, cum - self.alpha)
        return {
            "disparity_shift": shift,
            "disparity_cum": cum,
            "gini_cum": gini(exposure_per_capita(self.cum_deploy, self.pop_weight)),
            "cost": cost,
        }
