"""E2 -- quantify the runaway feedback loop over deployment time.

Consumes a per-shift DataFrame from ``rollout(..., keep_arrays=True)`` (possibly
several eval seeds and several trained seeds pooled under one policy name) with
array columns ``deployment``, ``latent_region``, ``recorded_region``.

Metrics, each a per-shift trajectory averaged over episodes and lightly smoothed:
  * ``enforcement_gini``    -- Gini of cumulative deployment across regions.
  * ``latent_coverage``     -- share of latent crime occurring under patrol.
  * ``deploy_latent_emd``   -- 1-D earth-mover distance between the deployment and
     latent crime-share distributions (how far patrol is from where crime is).
  * ``recorded_latent_emd`` -- EMD between recorded-crime and latent-crime shares.
  * ``runaway_index``       -- partial correlation between this shift's deployment
     and cumulative past deployment, controlling for current latent crime.
     CAUTION (PIVOT_AUDIT.md F3): any static allocation scores ~1.0 here, so
     it measures persistence, not feedback; do not report it as a loop measure.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from crimesim.env.fairness import gini
from crimesim.eval.oracle_relative import iter_episodes


def _emd(a: np.ndarray, b: np.ndarray) -> float:
    a = a / max(a.sum(), 1e-9)
    b = b / max(b.sum(), 1e-9)
    return float(np.sum(np.abs(np.cumsum(a - b))))


def _partial_corr(x, y, z) -> float:
    Z = np.column_stack([np.ones_like(z), z])
    rx = x - Z @ np.linalg.lstsq(Z, x, rcond=None)[0]
    ry = y - Z @ np.linalg.lstsq(Z, y, rcond=None)[0]
    if rx.std() < 1e-9 or ry.std() < 1e-9:
        return 0.0
    return float(np.corrcoef(rx, ry)[0, 1])


def feedback_trajectories(df: pd.DataFrame, smooth: int = 9) -> pd.DataFrame:
    rows = []
    for ep_id, ep in enumerate(iter_episodes(df)):
        dep = np.stack(ep["deployment"].to_numpy())
        lat = np.stack(ep["latent_region"].to_numpy())
        rec = np.stack(ep["recorded_region"].to_numpy())
        cum = np.cumsum(dep, axis=0)
        T = len(ep)
        for t in range(T):
            ds = dep[t] / max(dep[t].sum(), 1e-9)
            row = {
                "ep": ep_id, "shift": t,
                "enforcement_gini": gini(cum[t]),
                "latent_coverage": float((ds * lat[t]).sum() / max(lat[t].sum(), 1e-9)),
                "deploy_latent_emd": _emd(dep[t], lat[t]),
                "recorded_latent_emd": _emd(rec[t], lat[t]),
                "runaway_index": (_partial_corr(dep[t], cum[t - 1], lat[t])
                                  if t >= 5 else np.nan),
            }
            rows.append(row)
    traj = (pd.DataFrame(rows).groupby("shift").mean(numeric_only=True)
            .drop(columns="ep").reset_index())
    if smooth > 1:
        for c in traj.columns:
            if c != "shift":
                traj[c] = traj[c].rolling(smooth, center=True, min_periods=1).mean()
    return traj


def summary_row(df: pd.DataFrame) -> dict:
    traj = feedback_trajectories(df, smooth=1)
    tmax = traj["shift"].max()
    late = traj[traj["shift"] >= 0.6 * tmax]
    early = traj[traj["shift"] <= 0.2 * tmax]
    return {
        "enforcement_gini_final": float(late["enforcement_gini"].mean()),
        "enforcement_gini_rise": float(late["enforcement_gini"].mean()
                                       - early["enforcement_gini"].mean()),
        "latent_coverage_late": float(late["latent_coverage"].mean()),
        "deploy_latent_emd_late": float(late["deploy_latent_emd"].mean()),
        "runaway_index_late": float(late["runaway_index"].mean()),
    }
