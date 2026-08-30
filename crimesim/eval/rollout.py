"""Evaluation harness: run any policy on the PolicingEnv across held-out ABM
seeds and collect per-shift ground-truth outcomes.

A *policy* is either a rule-based object from ``crimesim.rl.baselines`` (has
``reset()`` and ``act(env)->deployment``) or a Stable-Baselines3 model (has
``predict(obs)``). Use :func:`rollout` for both.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from crimesim.config import Config
from crimesim.env.policing_env import PolicingEnv


def _is_sb3(policy) -> bool:
    return hasattr(policy, "predict") and not hasattr(policy, "act")


def rollout(policy, cfg: Config, seeds, *, keep_arrays: bool = False,
           vecnorm=None, obs_norm=None) -> pd.DataFrame:
    """Return a per-shift DataFrame over all ``seeds``.

    Latent crime is always the ground-truth outcome; ``reward_on`` only affects
    what a *trained* policy optimised, not what we measure here.
    """
    env = PolicingEnv(cfg, eval_mode=True)
    rows = []
    for seed in seeds:
        obs, _ = env.reset(seed=int(seed))
        if hasattr(policy, "reset"):
            policy.reset()
        done = False
        t = 0
        while not done:
            if _is_sb3(policy):
                a = obs
                if obs_norm is not None:
                    a = obs_norm(obs)
                elif vecnorm is not None:
                    a = vecnorm.normalize_obs(obs)
                action, _ = policy.predict(np.asarray(a, np.float32), deterministic=True)
                obs, r, term, trunc, info = env.step(action)
            else:
                deployment = policy.act(env)
                obs, r, term, trunc, info = env.step_with_deployment(deployment)
            done = term or trunc
            rec = {
                "policy": getattr(policy, "name", policy.__class__.__name__),
                "seed": int(seed), "shift": t,
                "latent_total": info["latent_total"],
                "recorded_total": info["recorded_total"],
                "recorded_coverage": info.get("recorded_coverage", float("nan")),
                "latent_coverage": info.get("latent_coverage", float("nan")),
                "disparity_shift": info["disparity_shift"],
                "disparity_cum": info["disparity_cum"],
                "gini_cum": info["gini_cum"],
                "cost": info["cost"],
            }
            if keep_arrays:
                rec["latent_region"] = info["latent_region"].copy()
                rec["recorded_region"] = info["recorded_region"].copy()
                rec["deployment"] = info["deployment"].copy()
                rec["intensity_region"] = info["intensity_region"].copy()
            rows.append(rec)
            t += 1
    return pd.DataFrame(rows)


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    """Per-policy episode-level means with 95% CIs (over seeds)."""
    ep = df.groupby(["policy", "seed"]).agg(
        latent_per_shift=("latent_total", "mean"),
        recorded_per_shift=("recorded_total", "mean"),
        recorded_coverage=("recorded_coverage", "mean"),
        latent_coverage=("latent_coverage", "mean"),
        disparity_shift_mean=("disparity_shift", "mean"),
        disparity_cum_final=("disparity_cum", "last"),
        gini_cum_final=("gini_cum", "last"),
    ).reset_index()

    def ci(s):
        s = np.asarray(s, float)
        return 1.96 * s.std(ddof=1) / np.sqrt(len(s)) if len(s) > 1 else 0.0

    out = ep.groupby("policy").agg(
        latent=("latent_per_shift", "mean"),
        latent_ci=("latent_per_shift", ci),
        recorded=("recorded_per_shift", "mean"),
        lat_cov=("latent_coverage", "mean"),
        lat_cov_ci=("latent_coverage", ci),
        rec_cov=("recorded_coverage", "mean"),
        disparity=("disparity_cum_final", "mean"),
        disparity_ci=("disparity_cum_final", ci),
        disparity_shift=("disparity_shift_mean", "mean"),
        gini=("gini_cum_final", "mean"),
        gini_ci=("gini_cum_final", ci),
        n_seeds=("seed", "nunique"),
    ).reset_index()

    # efficiency vs the uniform reference (share of extra crime prevented)
    if "uniform" in out["policy"].values:
        u = out.loc[out.policy == "uniform", "latent"].iloc[0]
        out["crime_reduction_vs_uniform_pct"] = 100 * (u - out["latent"]) / u
    return out.sort_values("latent").reset_index(drop=True)
