import crimesim._env_setup  # noqa: F401
"""Pivot Phases 4-5: final E1-E4 tables at 400k and the statistics audit.

Reads the caches written by ``scripts/overnight.py`` (no training):
  * rule-based baselines on eval seeds 200-229 (``results/baselines_e200_201n30.pkl``)
  * every 400k model's rollouts (``runs/<name>/eval_e200_201n30.pkl``)
  * the E5 episode table (``results/e5_400k_e300n20_episodes.csv``)
and rolls out three fixed rules from ``scripts/rule_search.py`` on the same seeds.

Writes ``results/final/`` (nothing else in results/ is touched):
  e1_e4_summary.csv     every policy family x GreedyOracle (hierarchical-bootstrap CI)
  per_training_seed.csv one row per (family, training seed)
  e2_feedback.csv       proxy gap, recording distortion, self-induced signal
  e3_need_controlled.csv hardship coefficient, episode-clustered, plus a seed-level bootstrap
  attribution_2x2.csv   what PPO observes x what it is rewarded on
  penalty_sweep.csv     parity penalty vs oracle-referenced penalty (|dlog| vs latent coverage)
  contrasts.csv         pre-registered paired contrasts with Holm-adjusted p-values
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from crimesim.abm.regions import load_region_info
from crimesim.config import load_config
from crimesim.eval.feedback_metrics import self_induced_signal
from crimesim.eval.incidence import long_run_need, need_controlled_incidence
from crimesim.eval.oracle_relative import (add_reference_shift, episode_table, iter_episodes,
                                           reference_disparity, relative_summary)
from crimesim.eval.rollout import rollout

sys.path.insert(0, str(Path(__file__).parent))
from rule_search import FixedRule  # noqa: E402

RUNS, RESULTS = Path("runs"), Path("results")
OUT = RESULTS / "final"
TAG = "e200_201n30"
SEEDS = list(range(200, 230))
FIVE, THREE = [0, 1, 2, 3, 4], [0, 1, 2]
RL = {"ppo": FIVE, "ppo_oracle": FIVE, "ppo_obslat": THREE, "ppo_rewlat": THREE,
      **{f"pen_p{p}": FIVE for p in [0.02, 0.06, 0.15, 0.4]},
      **{f"penref_w{w}": THREE for w in [0.05, 0.15, 0.5]}}
RULES = {"best_rule_recorded": (4, 2, 0), "best_rule_latent": (1.5, 0, 0), "rule_ewma_like": (1, 0, 0)}
region = load_region_info(20)
N_BOOT = 4000


# ------------------------------------------------------------------ data
def load_frames() -> dict:
    """(family, tseed) -> per-shift rollouts; tseed = -1 for rule-based policies."""
    frames = {(p, -1): g for p, g in pd.read_pickle(RESULTS / f"baselines_{TAG}.pkl").groupby("policy")}
    for label, (b, w, mu) in RULES.items():
        cache = OUT / f"rule_{label}_{TAG}.pkl"
        if not cache.exists():
            rollout(FixedRule(b, w, mu), load_config("configs/base.yaml"), SEEDS,
                    keep_arrays=True).to_pickle(cache)
        frames[(label, -1)] = pd.read_pickle(cache)
    for fam, seeds in RL.items():
        for s in seeds:
            frames[(fam, s)] = pd.read_pickle(RUNS / f"{fam}_s{s}" / f"eval_{TAG}.pkl")
    return frames


def episodes(frames: dict) -> tuple[pd.DataFrame, pd.Series]:
    tabs = []
    for (fam, s), df in frames.items():
        t = episode_table(df, region, family=fam, tseed=s)
        t["self_induced"] = [self_induced_signal(ep) for ep in iter_episodes(df)]
        tabs.append(t)
    eps = pd.concat(tabs, ignore_index=True)
    ref = reference_disparity(eps[eps.family == "oracle"])
    return add_reference_shift(eps, ref), ref


# ------------------------------------------------------------------ statistics
def _boot_family(g: pd.DataFrame, col: str, rng, ci) -> np.ndarray:
    """Replicates of a family's mean of ``col``: resample its training seeds, eval seeds given by ``ci``."""
    M = g.pivot_table(index="tseed", columns="seed", values=col, aggfunc="mean")
    M = M.reindex(columns=SEEDS).to_numpy()
    ri = rng.integers(0, len(M), (len(ci), len(M)))
    return np.array([np.nanmean(M[np.ix_(a, b)]) for a, b in zip(ri, ci)]), np.nanmean(M)


def contrast(eps: pd.DataFrame, a: str, b: str, col: str = "dlog", seed: int = 0) -> dict:
    """Mean(a) - mean(b) of ``col``. Eval seeds are resampled jointly (paired);
    training seeds independently within each family. Two-sided bootstrap p."""
    rng = np.random.default_rng(seed)
    ci = rng.integers(0, len(SEEDS), (N_BOOT, len(SEEDS)))
    ra, ma = _boot_family(eps[eps.family == a], col, rng, ci)
    rb, mb = _boot_family(eps[eps.family == b], col, rng, ci)
    d = ra - rb
    lo, hi = np.percentile(d, [2.5, 97.5])
    p = min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))
    return {"a": a, "b": b, "metric": col, "diff": ma - mb, "lo": lo, "hi": hi, "p_boot": max(p, 1 / N_BOOT)}


def holm(p: np.ndarray) -> np.ndarray:
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for k, i in enumerate(order):
        running = max(running, (len(p) - k) * p[i])
        adj[i] = min(1.0, running)
    return adj


# ------------------------------------------------------------------ tables
def e3(frames: dict) -> pd.DataFrame:
    need = long_run_need(frames[("uniform", -1)])
    rows = []
    fams = dict.fromkeys(f for f, _ in frames)
    for fam in fams:
        keys = [k for k in frames if k[0] == fam]
        pooled = need_controlled_incidence(pd.concat([frames[k] for k in keys], ignore_index=True),
                                           region, need)
        row = {"family": fam, "n_train_seeds": len(keys) if keys[0][1] >= 0 else 0,
               "n_episodes": pooled["n_clusters"], **{f"pooled_{k}": v for k, v in pooled["hardship"].items()}}
        if len(keys) > 1:    # seed level: one coefficient per training seed
            per = np.array([need_controlled_incidence(frames[k], region, need)["hardship"]["coef"]
                            for k in keys])
            boot = np.random.default_rng(0).choice(per, (N_BOOT, len(per))).mean(1)
            row.update(seed_coefs=" ".join(f"{c:.4f}" for c in per),
                       seed_boot_lo=np.percentile(boot, 2.5), seed_boot_hi=np.percentile(boot, 97.5),
                       seeds_positive=f"{(per > 0).sum()}/{len(per)}")
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    frames = load_frames()
    eps, ref = episodes(frames)
    eps.to_csv(OUT / "episodes.csv", index=False)

    summ = relative_summary(eps, ["family"], ref, n_boot=N_BOOT)
    summ = summ.merge(eps.groupby("family").self_induced.mean(), on="family")
    summ["abs_dlog"] = summ.dlog.abs()
    summ.to_csv(OUT / "e1_e4_summary.csv", index=False)
    print("== E1/E4: x GreedyOracle disparity, eval seeds 200-229, 400k ==")
    print(summ[["family", "n_train_seeds", "x_ref", "x_ref_lo", "x_ref_hi", "disparity", "need_tv",
                "lat_cov", "rec_cov", "proxy_gap"]].round(4).to_string(index=False))

    per = (eps[eps.tseed >= 0].groupby(["family", "tseed"])
           [["disparity", "dlog", "need_tv", "lat_cov", "rec_cov", "proxy_gap", "self_induced"]].mean())
    per["x_ref"] = np.exp(per.dlog)
    per.to_csv(OUT / "per_training_seed.csv")
    spread = per.groupby("family").x_ref.agg(["min", "max", "std"])
    print("\n== per-training-seed x_ref spread ==")
    print(spread.round(3).to_string())

    e2 = summ[["family", "proxy_gap", "rec_distortion", "self_induced", "need_tv"]]
    e2.to_csv(OUT / "e2_feedback.csv", index=False)
    print("\n== E2: feedback metrics ==")
    print(e2.round(4).to_string(index=False))

    e3t = e3(frames)
    e3t.to_csv(OUT / "e3_need_controlled.csv", index=False)
    print("\n== E3: hardship coefficient on (patrol share - need), episode-clustered ==")
    print(e3t.round(4).to_string(index=False))

    # 2x2 on the three training seeds every cell has
    sub = eps[eps.family.isin(["ppo", "ppo_obslat", "ppo_rewlat", "ppo_oracle"]) & eps.tseed.isin(THREE)]
    att = relative_summary(sub, ["family"], ref, n_boot=N_BOOT)
    att.insert(1, "observes", att.family.map({"ppo": "recorded", "ppo_rewlat": "recorded",
                                              "ppo_obslat": "recorded+latent", "ppo_oracle": "recorded+latent"}))
    att.insert(2, "rewarded_on", att.family.map({"ppo": "recorded", "ppo_obslat": "recorded",
                                                 "ppo_rewlat": "latent", "ppo_oracle": "latent"}))
    att.to_csv(OUT / "attribution_2x2.csv", index=False)
    print("\n== 2x2 attribution (training seeds 0-2) ==")
    print(att[["family", "observes", "rewarded_on", "x_ref", "x_ref_lo", "x_ref_hi", "lat_cov",
               "rec_cov", "proxy_gap"]].round(4).to_string(index=False))

    sweep = summ[summ.family.str.startswith(("pen", "ppo", "oracle", "uniform"))
                 & ~summ.family.isin(["ppo_obslat", "ppo_rewlat"])]
    sweep = sweep[["family", "x_ref", "x_ref_lo", "x_ref_hi", "abs_dlog", "lat_cov", "rec_cov"]]
    sweep.to_csv(OUT / "penalty_sweep.csv", index=False)
    print("\n== penalty sweep ==")
    print(sweep.round(4).to_string(index=False))

    # pre-registered contrasts, Holm across the whole set
    C = [contrast(eps, "ppo", "oracle"),
         contrast(eps, "ppo", "ewma_hotspot"), contrast(eps, "ppo", "rf_hotspot"),
         contrast(eps, "ppo", "best_rule_recorded"),
         contrast(eps, "ppo_obslat", "ppo"), contrast(eps, "ppo_rewlat", "ppo"),
         contrast(eps, "ppo_oracle", "ppo"), contrast(eps, "ppo_oracle", "oracle"),
         contrast(eps, "ppo", "oracle", "lat_cov"), contrast(eps, "ppo", "uniform", "lat_cov"),
         contrast(eps, "ppo", "best_rule_recorded", "rec_cov"),
         *[contrast(eps, f"penref_w{w}", "oracle") for w in [0.05, 0.15, 0.5]],
         *[contrast(eps, f"penref_w{w}", "ppo", "lat_cov") for w in [0.05, 0.15, 0.5]]]
    ct = pd.DataFrame(C)
    e5 = pd.read_csv(RESULTS / "e5_400k_e300n20_episodes.csv")
    e5c = []
    for v in ["perfect_obs", "const_detect"]:      # E5: difference in dlog, paired on eval seeds 300-319
        a = e5[(e5.variant == v) & (e5.kind == "ppo")].groupby("tseed").dlog.mean()
        b = e5[(e5.variant == "base") & (e5.kind == "ppo")].groupby("tseed").dlog.mean()
        rng = np.random.default_rng(0)
        d = rng.choice(a, (N_BOOT, len(a))).mean(1) - rng.choice(b, (N_BOOT, len(b))).mean(1)
        e5c.append({"a": f"E5 {v} ppo", "b": "E5 base ppo", "metric": "dlog", "diff": a.mean() - b.mean(),
                    "lo": np.percentile(d, 2.5), "hi": np.percentile(d, 97.5),
                    "p_boot": max(2 * min((d <= 0).mean(), (d >= 0).mean()), 1 / N_BOOT)})
    ct = pd.concat([ct, pd.DataFrame(e5c)], ignore_index=True)
    ct["p_holm"] = holm(ct.p_boot.to_numpy())
    ct["ratio"] = np.where(ct.metric == "dlog", np.exp(ct["diff"]), np.nan)
    ct.to_csv(OUT / "contrasts.csv", index=False)
    print("\n== contrasts (dlog: exp(diff) = ratio of disparities; Holm over all rows) ==")
    print(ct.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
