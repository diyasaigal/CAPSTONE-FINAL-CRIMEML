"""Episode-level metrics and oracle-relative disparity (pivot, PIVOT_AUDIT.md section B).

Raw exposure disparity mixes two things: where crime actually is (high-hardship
districts have more of it) and how a policy distorts patrol beyond that. We
therefore score each policy against a reference policy, by default the
need-proportional GreedyOracle, on the SAME evaluation seed:

    dlog = log(disparity_policy) - log(disparity_reference)

and report ``x_reference = exp(mean dlog)`` (a geometric-mean ratio). The CI
comes from a hierarchical bootstrap over training seeds and eval seeds. Eval
seeds stay paired with the reference, so the reference's own seed-to-seed
variability is part of the interval rather than treated as a fixed constant.

Per-episode quantities (all over the whole episode, cumulative):
  * ``disparity``      -- exposure of the top hardship quantile / bottom quantile,
                          identical to ``FairnessMeter``'s final ``disparity_cum``
  * ``need_tv``        -- total-variation distance between patrol share and the
                          episode's own latent-crime share, over ALL regions
  * ``lat_cov``, ``rec_cov`` -- mean latent / recorded coverage per shift
  * ``proxy_gap``      -- rec_cov - lat_cov (how much the optimised proxy
                          overstates the real objective)
  * ``rec_distortion`` -- (recorded / latent share in the top quantile) divided by
                          the same in the bottom quantile; 1 = the data the agent
                          sees is not tilted by where it patrolled
"""
from __future__ import annotations

from typing import Iterator

import numpy as np
import pandas as pd

from crimesim.abm.regions import RegionInfo
from crimesim.env.fairness import _group_exposure
from crimesim.eval.stats import hierarchical_bootstrap


def episode_ids(df: pd.DataFrame) -> np.ndarray:
    """1-based episode id per row; a new episode starts wherever ``shift`` does not increase."""
    s = df["shift"].to_numpy()
    return np.cumsum(np.r_[True, s[1:] <= s[:-1]])


def iter_episodes(df: pd.DataFrame) -> Iterator[pd.DataFrame]:
    """Yield each contiguous episode of a (possibly pooled) per-shift frame."""
    df = df.reset_index(drop=True)
    ids = episode_ids(df)
    starts = np.flatnonzero(np.r_[True, np.diff(ids) != 0])
    for a, b in zip(starts, np.r_[starts[1:], len(df)]):
        yield df.iloc[a:b]


def episode_metrics(ep: pd.DataFrame, groups: np.ndarray, pop_weight: np.ndarray) -> dict:
    dep = np.stack(ep["deployment"].to_numpy()).sum(0).astype(float)
    lat = np.stack(ep["latent_region"].to_numpy()).sum(0).astype(float)
    rec = np.stack(ep["recorded_region"].to_numpy()).sum(0).astype(float)
    hi, lo = int(groups.max()), 0
    disparity = (_group_exposure(dep, pop_weight, groups, hi)
                 / max(_group_exposure(dep, pop_weight, groups, lo), 1e-9))
    dep_s = dep / max(dep.sum(), 1e-9)
    lat_s = lat / max(lat.sum(), 1e-9)
    rec_s = rec / max(rec.sum(), 1e-9)

    def over_recording(g):
        m = groups == g
        return rec_s[m].sum() / lat_s[m].sum() if lat_s[m].sum() > 0 else np.nan

    lat_cov = float(ep["latent_coverage"].mean())
    rec_cov = float(ep["recorded_coverage"].mean())
    return {"seed": int(ep["seed"].iloc[0]),
            "disparity": float(disparity),
            "need_tv": float(0.5 * np.abs(dep_s - lat_s).sum()),
            "lat_cov": lat_cov, "rec_cov": rec_cov, "proxy_gap": rec_cov - lat_cov,
            "rec_distortion": float(over_recording(hi) / over_recording(lo))}


def episode_table(df: pd.DataFrame, region: RegionInfo, *, attr: str = "hardship",
                  n_q: int = 5, **labels) -> pd.DataFrame:
    """One row per episode of ``df`` with ``episode_metrics`` plus constant ``labels``."""
    groups = region.quantile_groups(attr, n_q)
    return pd.DataFrame([dict(labels, **episode_metrics(ep, groups, region.pop_weight))
                         for ep in iter_episodes(df)])


def reference_disparity(table: pd.DataFrame) -> pd.Series:
    """Reference disparity per eval seed (mean over episodes sharing a seed)."""
    return table.groupby("seed")["disparity"].mean()


def add_reference_shift(table: pd.DataFrame, reference: pd.Series) -> pd.DataFrame:
    """Add ``dlog`` = log disparity minus log reference disparity on the same eval seed."""
    missing = sorted(set(table["seed"]) - set(reference.index))
    if missing:
        raise ValueError(f"reference has no rollout for eval seeds {missing}")
    return table.assign(dlog=np.log(table["disparity"])
                        - np.log(table["seed"].map(reference).astype(float)))


def relative_summary(table: pd.DataFrame, keys: list[str], reference: pd.Series, *,
                     tseed_col: str = "tseed", n_boot: int = 4000, seed: int = 0) -> pd.DataFrame:
    """Per group of ``keys``: x_reference with hierarchical-bootstrap CI and episode means.

    ``table`` must already carry ``dlog`` (see ``add_reference_shift``). Rows
    without a training seed (rule-based policies) should use a constant in
    ``tseed_col``; the bootstrap then resamples eval seeds only.
    """
    rows = []
    for k, g in table.groupby(keys, sort=False):
        M = g.pivot_table(index=tseed_col, columns="seed", values="dlog", aggfunc="mean")
        m, lo, hi = hierarchical_bootstrap(M.to_numpy(), n_boot=n_boot, seed=seed)
        seeds = g["seed"].unique()
        rows.append({**dict(zip(keys, k if isinstance(k, tuple) else (k,))),
                     "n_train_seeds": g[tseed_col].nunique(), "n_eval_seeds": len(seeds),
                     "n_episodes": len(g),
                     "x_ref": np.exp(m), "x_ref_lo": np.exp(lo), "x_ref_hi": np.exp(hi),
                     "dlog": m, "dlog_lo": lo, "dlog_hi": hi,
                     "ratio_of_means": g["disparity"].mean() / reference.loc[seeds].mean(),
                     **g[["disparity", "need_tv", "lat_cov", "rec_cov", "proxy_gap",
                          "rec_distortion"]].mean().to_dict()})
    return pd.DataFrame(rows)
