import crimesim._env_setup  # noqa: F401
"""JASSS Phase 2 summary: the central table, crime prevented, and the contrasts.

Reads only cached rollouts (no training, no new simulation):
  * every model's evaluation  runs/jasss/<setting>__<family>_s<seed>/eval_e200n30.pkl
  * rule-based policies       results/jasss/eval/<setting>/<label>_e200n30.pkl
written by ``scripts/jasss_runs.py`` (stages ``train`` and ``baselines``).

Every policy is scored in its own setting, against that setting's own references,
on the same 30 evaluation seeds (200-229), paired by eval seed:

  x_oracle   exp(mean dlog), dlog = log disparity - log disparity of GreedyOracle
             (the latent-crime-proportional reference) on the same eval seed
  x_need     the same against NeedProportional (patrol in proportion to the
             untreated-crime need vector of results/jasss/refs/<setting>.json)
  prevented  100 * (latent_no_patrol - latent_policy) / latent_no_patrol, per
             episode, paired with NoPatrol on the same eval seed; city-wide and
             per hardship quintile (q1 = least disadvantaged)

Intervals: hierarchical bootstrap over training seeds x eval seeds, 4000
replicates (20000 for the contrasts, so p is resolved below 0.01 after Holm),
bootstrap seed 0 everywhere (A20). Training-seed spread: median,
min and max of each training seed's own x_oracle.

Writes results/jasss/summary/:
  episodes.csv     one row per (setting, family, training seed, eval seed)
  per_seed.csv     one row per (setting, family, training seed)
  provenance.csv   one row per trained model (A3): run, x_oracle, coverages, prevented,
                   steps, eval seeds and the machine that trained it (results/jasss/trained_on.csv)
  central.csv      one row per (setting, family): the central table (Part D.3)
  quintiles.csv    crime prevented per hardship quintile, per (setting, family)
  contrasts.csv    the pre-specified contrasts, Holm-adjusted over the whole set
  headline.json    the numbers the abstract and readiness gate quote

    PYTHONPATH=. python scripts/summarise_jasss.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from crimesim.abm.regions import region_for
from crimesim.eval.oracle_relative import episode_metrics, iter_episodes
from crimesim.eval.stats import hierarchical_bootstrap

sys.path.insert(0, str(Path(__file__).parent))
import jasss_runs as J  # noqa: E402

RUNS = Path("runs") / J.RUN_DIR
EVAL = J.OUT / "eval"
OUT = J.OUT / "summary"
SEEDS = J.EVAL_SEEDS
N_BOOT, BOOT_SEED, N_Q = 4000, 0, 5
# contrasts: more replicates, so the smallest p (1 / N_BOOT_CONTRAST) stays well below 0.01 after Holm
N_BOOT_CONTRAST = 20_000
BASELINES = ["uniform", "random", "ewma_hotspot", "rf_hotspot", "oracle", "need_prop", "no_patrol",
             "rule_ewma_like", "best_rule_recorded", "best_rule_latent"]


# ------------------------------------------------------------------ data
def frames_for(setting: str, specs) -> dict:
    """(family, tseed) -> per-shift rollouts of ``setting``; tseed = -1 for rule-based policies."""
    frames, missing = {}, []
    for label in BASELINES:
        p = EVAL / setting / f"{label}_{J.EVAL_TAG}.pkl"
        if p.exists():
            frames[(label, -1)] = pd.read_pickle(p)
        elif not label.startswith("best_rule"):          # the rule grids exist for 3 settings only
            missing.append(str(p))
    for _, s, fam, _, _, seed in specs:
        if s != setting:
            continue
        p = RUNS / f"{s}__{fam}_s{seed}" / f"eval_{J.EVAL_TAG}.pkl"
        if p.exists():
            frames[(fam, seed)] = pd.read_pickle(p)
        else:
            missing.append(str(p))
    if missing:
        raise SystemExit(f"{setting}: {len(missing)} cached rollouts missing, e.g. {missing[:3]}\n"
                         "run scripts/jasss_runs.py (stages train, baselines) first")
    return frames


def episodes(setting: str, frames: dict) -> pd.DataFrame:
    region = region_for(J.setting_cfg(setting))
    groups = region.quantile_groups("hardship", N_Q)
    rows = []
    for (fam, tseed), df in frames.items():
        for ep in iter_episodes(df):
            lat = np.stack(ep["latent_region"].to_numpy()).mean(0)
            rows.append({"setting": setting, "family": fam, "tseed": tseed,
                         **episode_metrics(ep, groups, region.pop_weight),
                         "latent": lat.sum(), **{f"latent_q{g + 1}": lat[groups == g].sum() for g in range(N_Q)}})
    eps = pd.DataFrame(rows)

    def per_seed(fam, col):
        ref = eps[eps.family == fam].groupby("seed")[col].mean()
        if set(eps.seed) - set(ref.index):
            raise ValueError(f"{setting}: {fam} lacks eval seeds {sorted(set(eps.seed) - set(ref.index))}")
        return eps.seed.map(ref)

    # NoPatrol deploys nobody: its disparity is 0 and it has no x_oracle
    log_disp = np.log(eps.disparity.where(eps.disparity > 0))
    eps["dlog"] = log_disp - np.log(per_seed("oracle", "disparity"))
    eps["dlog_need"] = log_disp - np.log(per_seed("need_prop", "disparity"))
    for col in ["latent", *(f"latent_q{g + 1}" for g in range(N_Q))]:
        ref = per_seed("no_patrol", col)
        eps[col.replace("latent", "prev")] = 100 * (ref - eps[col]) / ref
    return eps


# ------------------------------------------------------------------ statistics
def boot(g: pd.DataFrame, col: str) -> tuple[float, float, float]:
    if g[col].isna().all():
        return np.nan, np.nan, np.nan
    M = g.pivot_table(index="tseed", columns="seed", values=col, aggfunc="mean").reindex(columns=SEEDS)
    return hierarchical_bootstrap(M.to_numpy(), n_boot=N_BOOT, seed=BOOT_SEED)


def central(eps: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (setting, fam), g in eps.groupby(["setting", "family"], sort=False):
        m, lo, hi = boot(g, "dlog")
        mn, lon, hin = boot(g, "dlog_need")
        pv, plo, phi = boot(g, "prev")
        seed_x = np.exp(g.groupby("tseed").dlog.mean())
        rows.append({"setting": setting, "family": fam,
                     "n_train_seeds": g.tseed.nunique() if g.tseed.max() >= 0 else 0,
                     "n_eval_seeds": g.seed.nunique(),
                     "x_oracle": np.exp(m), "x_oracle_lo": np.exp(lo), "x_oracle_hi": np.exp(hi),
                     "x_oracle_median": seed_x.median(), "x_oracle_min": seed_x.min(),
                     "x_oracle_max": seed_x.max(),
                     "x_need": np.exp(mn), "x_need_lo": np.exp(lon), "x_need_hi": np.exp(hin),
                     "prevented_pct": pv, "prevented_lo": plo, "prevented_hi": phi,
                     **g[["disparity", "lat_cov", "rec_cov", "proxy_gap", "need_tv", "rec_distortion",
                          "latent"]].mean().to_dict()})
    return pd.DataFrame(rows)


def _replicates(g: pd.DataFrame, col: str, rng, ci) -> tuple[np.ndarray, float]:
    """Bootstrap replicates of a family's mean of ``col``: its training seeds resampled
    independently, the eval seeds given by ``ci`` (shared, so contrasts stay paired)."""
    M = g.pivot_table(index="tseed", columns="seed", values=col, aggfunc="mean").reindex(columns=SEEDS)
    M = M.to_numpy()
    ri = rng.integers(0, len(M), (len(ci), len(M)))
    return np.array([np.nanmean(M[np.ix_(a, b)]) for a, b in zip(ri, ci)]), float(np.nanmean(M))


def contrast(eps: pd.DataFrame, a: tuple, b: tuple, col: str = "dlog", item: str = "") -> dict:
    """mean(a) - mean(b) of ``col`` for (setting, family) pairs a and b; two-sided bootstrap p."""
    rng = np.random.default_rng(BOOT_SEED)
    ci = rng.integers(0, len(SEEDS), (N_BOOT_CONTRAST, len(SEEDS)))
    pick = lambda k: eps[(eps.setting == k[0]) & (eps.family == k[1])]  # noqa: E731
    ra, ma = _replicates(pick(a), col, rng, ci)
    rb, mb = _replicates(pick(b), col, rng, ci)
    d = ra - rb
    p = min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))
    out = {"item": item, "a": "/".join(a), "b": "/".join(b), "metric": col, "diff": ma - mb,
           "lo": float(np.percentile(d, 2.5)), "hi": float(np.percentile(d, 97.5)),
           "p_boot": max(p, 1 / N_BOOT_CONTRAST)}
    if col.startswith("dlog"):
        out.update(ratio=np.exp(out["diff"]), ratio_lo=np.exp(out["lo"]), ratio_hi=np.exp(out["hi"]))
    return out


def holm(p: np.ndarray) -> np.ndarray:
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for k, i in enumerate(order):
        running = max(running, (len(p) - k) * p[i])
        adj[i] = min(1.0, running)
    return adj


# Pre-specified contrasts. In "dlog", exp(diff) is a ratio of disparities; in "prev",
# diff is in percentage points of crime prevented. "Excess" of the detection channel
# = naive PPO in a setting vs naive PPO in the same world with constant detection.
B = "base"
CONTRASTS = [
    # A10: the common experiment
    ("A10", (B, "ppo"), (B, "oracle"), "dlog"),
    ("A10", (B, "ppo"), (B, "ewma_hotspot"), "dlog"),
    ("A10", (B, "ppo"), (B, "rf_hotspot"), "dlog"),
    ("A10", (B, "ppo"), (B, "best_rule_recorded"), "dlog"),
    ("A10", (B, "obslat"), (B, "ppo"), "dlog"),
    ("A10", (B, "rewlat"), (B, "ppo"), "dlog"),
    ("A10", (B, "oracleppo"), (B, "ppo"), "dlog"),
    ("A10", (B, "oracleppo"), (B, "oracle"), "dlog"),
    ("A10", (B, "ppo"), ("const", "ppo"), "dlog"),
    ("A10", (B, "ppo"), ("perfect", "ppo"), "dlog"),
    ("A10", ("const", "rewlat"), ("const", "ppo"), "dlog"),
    ("A10", (B, "par0.06"), (B, "ppo"), "dlog"),
    # A1: crime prevented, each A10 cell against the reference of its detection setting
    *[("A1", (s, f), (s, "oracle"), "prev") for s, f in
      [(B, "ppo"), (B, "rewlat"), (B, "obslat"), (B, "oracleppo"),
       ("const", "ppo"), ("const", "rewlat"), ("perfect", "ppo")]],
    ("A1", (B, "ppo"), (B, "uniform"), "prev"),
    ("A1", (B, "ppo"), (B, "best_rule_recorded"), "prev"),
    ("A1", (B, "oracleppo"), (B, "ppo"), "prev"),
    # the mentor's question: does breaking the loop also prevent more crime? (no-patrol
    # crime is the same in the three detection settings, so these are comparable)
    ("A1", ("const", "ppo"), (B, "ppo"), "prev"),
    ("A1", ("perfect", "ppo"), (B, "ppo"), "prev"),
    # A11: the hardship tilt switched off
    ("A11", (B, "w0"), (B, "ppo"), "dlog"),
    ("A11", (B, "wneg"), (B, "ppo"), "dlog"),
    ("A11", (B, "w0_nosocio"), (B, "ppo"), "dlog"),
    ("A11", (B, "w0"), (B, "oracle"), "dlog"),
    # A12/A13: penalties against the oracle, and what they cost in crime prevented
    *[("A13", (B, f"penref{w}"), (B, "oracle"), "dlog") for w in (0.05, 0.15, 0.5)],
    *[("A13", (B, f"penref{w}"), (B, "ppo"), "prev") for w in (0.05, 0.15, 0.5)],
    ("A13", (B, "penneed0.15"), (B, "need_prop"), "dlog_need"),
    ("A13", (B, "lag"), (B, "oracle"), "dlog"),
    ("A13", (B, "lag"), (B, "ppo"), "prev"),
    # A6/A7: the detection-channel excess at the realistic contrast and at more agents
    *[("A6A7", (v, "ppo"), (f"{v}_const", "ppo"), "dlog")
      for v in ("contrast_low", "contrast_high", "agents_500", "agents_1000")],
]


# ------------------------------------------------------------------ main
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    specs = J.model_specs(J.TRAIN_SEEDS)
    tabs = []
    for setting in J.SETTINGS:
        tabs.append(episodes(setting, frames_for(setting, specs)))
        print(f"{setting}: {tabs[-1].family.nunique()} policies", flush=True)
    eps = pd.concat(tabs, ignore_index=True)
    eps.to_csv(OUT / "episodes.csv", index=False)

    per_seed = (eps.groupby(["setting", "family", "tseed"])
                .agg(dlog=("dlog", "mean"), dlog_need=("dlog_need", "mean"), prev=("prev", "mean"),
                     lat_cov=("lat_cov", "mean"), rec_cov=("rec_cov", "mean")).reset_index())
    per_seed["x_oracle"] = np.exp(per_seed.dlog)
    per_seed.to_csv(OUT / "per_seed.csv", index=False)

    # A3 provenance: one row per trained model, with the machine that trained it
    prov = per_seed[per_seed.tseed >= 0].copy()
    prov["run"] = [J.run_name(s, f, int(t)) for s, f, t in zip(prov.setting, prov.family, prov.tseed)]
    prov = prov.assign(eval_seeds=f"{SEEDS[0]}-{SEEDS[-1]}", reference="oracle of the same setting")
    trained_on = J.OUT / "trained_on.csv"
    if trained_on.exists():
        prov = prov.merge(pd.read_csv(trained_on), on="run", how="left")
    prov.to_csv(OUT / "provenance.csv", index=False)

    cen = central(eps)
    cen.to_csv(OUT / "central.csv", index=False)
    q = (eps.groupby(["setting", "family"], sort=False)[[f"prev_q{g + 1}" for g in range(N_Q)]]
         .mean().reset_index())
    q.to_csv(OUT / "quintiles.csv", index=False)

    ct = pd.DataFrame([contrast(eps, a, b, col, item) for item, a, b, col in CONTRASTS])
    ct["p_holm"] = holm(ct.p_boot.to_numpy())
    ct.to_csv(OUT / "contrasts.csv", index=False)

    row = lambda s, f: cen[(cen.setting == s) & (cen.family == f)].iloc[0]  # noqa: E731
    con = lambda a, b, m: ct[(ct.a == a) & (ct.b == b) & (ct.metric == m)].iloc[0]  # noqa: E731
    head = {
        "base_ppo": {k: float(row(B, "ppo")[k]) for k in
                     ["x_oracle", "x_oracle_lo", "x_oracle_hi", "x_oracle_median", "x_oracle_min",
                      "x_oracle_max", "x_need", "prevented_pct", "prevented_lo", "prevented_hi"]},
        "base_ppo_n_train_seeds": int(row(B, "ppo").n_train_seeds),
        "n_eval_seeds": len(SEEDS),
        "prevented_pct": {f"{s}/{f}": float(row(s, f).prevented_pct)
                          for s, f in [(B, "ppo"), (B, "oracle"), (B, "uniform"), (B, "need_prop"),
                                       (B, "best_rule_recorded")]},
        "excess_vs_const": {a.split("/")[0]: {k: float(con(a, b, "dlog")[k])
                                              for k in ["ratio", "ratio_lo", "ratio_hi", "p_holm"]}
                            for a, b in [("base/ppo", "const/ppo"),
                                         *[(f"{v}/ppo", f"{v}_const/ppo") for v in
                                           ("contrast_low", "contrast_high", "agents_500", "agents_1000")]]},
        "w_off": {k: float(con("base/w0", "base/ppo", "dlog")[k]) for k in ["ratio", "ratio_lo", "ratio_hi", "p_holm"]},
        # A11: share of PPO's log excess over the oracle that remains with w pinned to 0
        "w_off_share_of_log_excess": float(np.log(row(B, "w0").x_oracle) / np.log(row(B, "ppo").x_oracle)),
        "bootstrap": {"n_boot": N_BOOT, "n_boot_contrasts": N_BOOT_CONTRAST, "seed": BOOT_SEED},
        "n_contrasts_holm": len(ct),
    }
    (OUT / "headline.json").write_text(json.dumps(head, indent=1))

    pd.set_option("display.width", 220)
    show = ["setting", "family", "n_train_seeds", "x_oracle", "x_oracle_lo", "x_oracle_hi",
            "x_oracle_min", "x_oracle_max", "prevented_pct", "lat_cov", "rec_cov"]
    print("\n== central table (x oracle, 95% hierarchical-bootstrap CI; prevented % vs no patrol) ==")
    print(cen[show].round(3).to_string(index=False))
    print(f"\n== contrasts (Holm over all {len(ct)}) ==")
    print(ct[["item", "a", "b", "metric", "diff", "lo", "hi", "p_holm"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
