"""Publication figures. Matplotlib only (no seaborn), matching the legacy style.

Each function takes tidy DataFrames from the eval modules and writes a PDF to
``figures/``.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from crimesim.config import FIGURES_DIR

plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3,
                     "figure.dpi": 140, "savefig.bbox": "tight"})

LABELS = {"uniform": "Uniform", "random": "Random", "ewma_hotspot": "PredPol (EWMA)",
          "rf_hotspot": "RandomForest", "ppo": "PPO (naive)", "ppo_fair": "PPO + fairness",
          "ppo_oracle": "PPO oracle", "oracle": "Oracle",
          "pen_p0.02": "PPO+pen .02", "pen_p0.06": "PPO+pen .06",
          "pen_p0.15": "PPO+pen .15", "pen_p0.4": "PPO+pen .4"}


def _save(fig, name):
    FIGURES_DIR.mkdir(exist_ok=True)
    p = FIGURES_DIR / name
    fig.savefig(p)
    plt.close(fig)
    return p


def fig_efficiency_equity(summary: pd.DataFrame, out="fig_efficiency_equity.pdf"):
    """Scatter: enforcement disparity (x) vs latent-crime coverage (y). The ideal
    corner is top-left: high coverage of where crime actually is, at parity."""
    fig, ax = plt.subplots(figsize=(5, 4))
    for _, r in summary.iterrows():
        ax.errorbar(r["disparity"], r["lat_cov"], yerr=r.get("lat_cov_ci", 0),
                    fmt="o", ms=7, capsize=3, label=LABELS.get(r["policy"], r["policy"]))
    ax.set_xlabel("Enforcement disparity (disadvantaged / advantaged quintile)")
    ax.set_ylabel("Latent-crime coverage (share of true crime under patrol)")
    ax.axvline(1.0, color="k", lw=0.8, ls=":")
    ax.legend(fontsize=7, ncol=2)
    ax.set_title("Efficiency vs equity")
    return _save(fig, out)


def fig_feedback_trajectories(traj_by_policy: dict[str, pd.DataFrame],
                              out="fig_feedback_trajectories.pdf"):
    metrics = [("enforcement_gini", "Cumulative enforcement Gini"),
               ("latent_coverage", "Latent-crime coverage"),
               ("deploy_latent_emd", "Patrol vs latent-crime EMD"),
               ("runaway_index", "Runaway index (partial corr.)")]
    fig, axes = plt.subplots(2, 2, figsize=(9, 6), sharex=True)
    for ax, (m, title) in zip(axes.ravel(), metrics):
        for pol, t in traj_by_policy.items():
            if m in t:
                ax.plot(t["shift"], t[m], label=LABELS.get(pol, pol))
        ax.set_title(title)
        ax.set_xlabel("Deployment shift")
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("Runaway feedback over the deployment horizon")
    return _save(fig, out)


def fig_incidence(exposure_by_quintile: dict[str, np.ndarray],
                  out="fig_incidence.pdf"):
    fig, ax = plt.subplots(figsize=(5.5, 4))
    q = np.arange(1, len(next(iter(exposure_by_quintile.values()))) + 1)
    for pol, e in exposure_by_quintile.items():
        ax.plot(q, e, "o-", label=LABELS.get(pol, pol))
    ax.axhline(1.0, color="k", lw=0.8, ls=":")
    ax.set_xlabel("District hardship quintile (5 = most disadvantaged)")
    ax.set_ylabel("Enforcement exposure per capita")
    ax.set_xticks(q)
    ax.legend(fontsize=7)
    ax.set_title("Who gets policed, by neighbourhood disadvantage")
    return _save(fig, out)


def fig_pareto(frontier: pd.DataFrame, out="fig_pareto.pdf"):
    fig, ax = plt.subplots(figsize=(5.4, 4.2))
    eff = frontier[frontier["pareto_efficient"]].sort_values("disparity")
    ax.plot(eff["disparity"], eff["lat_cov"], "-", color="0.6", zorder=1)
    fr = frontier.sort_values("disparity").reset_index(drop=True)
    for i, r in fr.iterrows():
        star = bool(r["pareto_efficient"])
        ax.scatter(r["disparity"], r["lat_cov"], marker="*" if star else "o",
                   s=130 if star else 55, color="C3" if r["policy"] == "ppo" else "C0",
                   zorder=3 if r["policy"] == "ppo" else 2)
        dy = 8 if i % 2 == 0 else -12
        ax.annotate(LABELS.get(r["policy"], r["policy"]), (r["disparity"], r["lat_cov"]),
                    fontsize=6.5, xytext=(3, dy), textcoords="offset points")
    ax.set_xlabel("Enforcement disparity (disadvantaged / advantaged quintile)")
    ax.set_ylabel("Latent-crime coverage")
    ax.set_title("Efficiency–equity frontier (★ = Pareto-efficient)")
    return _save(fig, out)


def fig_lambda_history(hist: np.ndarray, out="fig_lambda_history.pdf"):
    fig, ax1 = plt.subplots(figsize=(5, 3.2))
    t, cost, lam = hist[:, 0], hist[:, 1], hist[:, 2]
    ax1.plot(t, lam, color="C0"); ax1.set_ylabel("lambda", color="C0")
    ax2 = ax1.twinx()
    ax2.plot(t, cost, color="C1", alpha=0.7); ax2.set_ylabel("mean cost", color="C1")
    ax1.set_xlabel("timesteps")
    ax1.set_title("Dual ascent on the fairness multiplier")
    return _save(fig, out)
