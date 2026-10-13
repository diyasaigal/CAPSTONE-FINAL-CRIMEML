import crimesim._env_setup  # noqa: F401
"""Plan step 6: check the mean-field model (crimesim/theory.py) against the simulator.

For each observation variant (base, const_detect, perfect_obs) and each
concentration exponent beta, a fixed rule (w = 0, mu = 0) is rolled out in the
agent-based simulator on seeds 1000-1005 (disjoint from every eval set) next
to that variant's GreedyOracle and Uniform. The model needs one input from the
simulator: the untreated crime rate per district, estimated once from the base
Uniform rollout as latent / (1 - kappa). Everything else is the config.

Writes results/theory/: sim.csv (per rule), compare.csv (model vs simulator),
summary.json (elasticities, fit). No training.
"""
import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

OUT = Path("results/theory")
BETAS = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
VARIANTS = ["base", "const_detect", "perfect_obs"]
SEEDS = list(range(1000, 1006))


def _cfg(variant: str):
    from crimesim.config import load_config
    import run_ablations
    return load_config("configs/base.yaml", **run_ablations._merge(run_ablations.VARIANTS[variant], {}))


def _sim(job):
    variant, kind, beta = job
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    from crimesim.abm.regions import load_region_info
    from crimesim.eval.oracle_relative import episode_table
    from crimesim.eval.rollout import rollout
    from crimesim.rl.baselines import GreedyOracle, UniformPolicy
    from rule_search import FixedRule
    pol = {"rule": lambda: FixedRule(beta, 0.0, 0.0), "oracle": GreedyOracle, "uniform": UniformPolicy}[kind]()
    df = rollout(pol, _cfg(variant), SEEDS, keep_arrays=True)
    t = episode_table(df, load_region_info(20), variant=variant, kind=kind, beta=beta)
    t["latent_region"] = [np.stack(g.latent_region.to_numpy()).mean(0).tolist()
                          for _, g in df.groupby("seed", sort=False)]
    return t


def simulate(workers: int) -> pd.DataFrame:
    cache = OUT / "sim.pkl"
    if cache.exists():
        return pd.read_pickle(cache)
    jobs = [(v, "rule", b) for v in VARIANTS for b in BETAS]
    jobs += [(v, k, np.nan) for v in VARIANTS for k in ("oracle", "uniform")]
    with ProcessPoolExecutor(workers) as ex:
        sim = pd.concat(list(ex.map(_sim, jobs)), ignore_index=True)
    sim.to_pickle(cache)
    return sim


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    from crimesim.abm.regions import load_region_info
    from crimesim import theory as th
    region = load_region_info(20)
    groups = region.quantile_groups("hardship", 5)
    sim = simulate(args.workers)
    sim.drop(columns="latent_region").to_csv(OUT / "sim.csv", index=False)

    base = _cfg("base")
    uni = sim[(sim.variant == "base") & (sim.kind == "uniform")]
    lam = np.mean(uni.latent_region.tolist(), axis=0) / th.keep(1.0, base.detection)
    floor = base.env.min_share

    rows, info = [], {"lambda_top_over_bottom": float(lam[groups == groups.max()].mean()
                                                      / lam[groups == groups.min()].mean())}
    for v in VARIANTS:
        det = _cfg(v).detection
        eps = th.elasticity(1.0, det, "recorded")
        info[v] = {"elasticity_recorded_u1": eps, "elasticity_latent_u1": th.elasticity(1.0, det, "latent")}
        orc_m = th.fixed_point(lam, det, beta=1.0, signal="latent", floor=0.0)
        d_orc_m = th.group_disparity(orc_m["shares"], groups)
        s = sim[sim.variant == v]
        orc_s = s[s.kind == "oracle"].set_index("seed").disparity
        for b in BETAS:
            fp = th.fixed_point(lam, det, beta=b, floor=floor)
            r = s[(s.kind == "rule") & (s.beta == b)]
            dl = np.log(r.disparity.to_numpy()) - np.log(orc_s.loc[r.seed].to_numpy())
            rows.append({"variant": v, "beta": b, "converged": fp["converged"],
                         "amplification_local": 1.0 / (1.0 - b * eps) if b * eps < 1 else np.inf,
                         "model_disparity": th.group_disparity(fp["shares"], groups),
                         "model_x_oracle": th.group_disparity(fp["shares"], groups) / d_orc_m,
                         "sim_disparity": r.disparity.mean(),
                         "sim_x_oracle": float(np.exp(dl.mean())),
                         "sim_x_oracle_lo": float(np.exp(dl.mean() - 1.96 * dl.std(ddof=1) / np.sqrt(len(dl)))),
                         "sim_x_oracle_hi": float(np.exp(dl.mean() + 1.96 * dl.std(ddof=1) / np.sqrt(len(dl)))),
                         "model_rec_cov": th.coverage(fp["shares"], fp["recorded"]),
                         "model_lat_cov": th.coverage(fp["shares"], fp["latent"]),
                         "sim_rec_cov": r.rec_cov.mean(), "sim_lat_cov": r.lat_cov.mean(),
                         "model_max_intensity": fp["intensity"].max()})
        info[v]["oracle_disparity_model"] = d_orc_m
        info[v]["oracle_disparity_sim"] = float(orc_s.mean())
    cmp = pd.DataFrame(rows)
    a, b = np.log(cmp.model_x_oracle), np.log(cmp.sim_x_oracle)
    info["fit"] = {"r2_log_x_oracle": float(1 - ((a - b) ** 2).sum() / ((b - b.mean()) ** 2).sum()),
                   "corr": float(np.corrcoef(a, b)[0, 1]),
                   "mean_abs_log_error": float(np.abs(a - b).mean())}
    for v in VARIANTS:
        g = cmp[cmp.variant == v]
        info[v]["beta_max_rec_cov_model"] = float(g.loc[g.model_rec_cov.idxmax(), "beta"])
        info[v]["beta_max_rec_cov_sim"] = float(g.loc[g.sim_rec_cov.idxmax(), "beta"])
        info[v]["beta_max_lat_cov_model"] = float(g.loc[g.model_lat_cov.idxmax(), "beta"])
        info[v]["beta_max_lat_cov_sim"] = float(g.loc[g.sim_lat_cov.idxmax(), "beta"])
    cmp.to_csv(OUT / "compare.csv", index=False)
    (OUT / "summary.json").write_text(json.dumps(info, indent=2))
    pd.set_option("display.width", 250)
    print(cmp.round(3).to_string(index=False))
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
