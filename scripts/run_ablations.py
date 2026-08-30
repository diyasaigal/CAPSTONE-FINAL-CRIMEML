import crimesim._env_setup  # noqa: F401
"""E5 -- robustness / ablations.

For each variant we train a naive PPO and one mid-alpha Lagrangian PPO, evaluate
on held-out seeds, and record the headline quantities: latent crime, enforcement
disparity, and the E3 hardship-incidence coefficient. Resumable.

Key control: ``perfect_obs`` sets p_report=1 so patrol no longer inflates the
record -- the feedback loop should then collapse (a falsification check).
"""
import argparse
import dataclasses
import json
from pathlib import Path

import numpy as np
import pandas as pd

from crimesim.config import load_config
from crimesim.abm.regions import load_region_info
from crimesim.eval.incidence import build_panel, incidence_regression
from crimesim.eval.rollout import rollout, summarise
from crimesim.rl.baselines import make_baselines
from crimesim.experiments.registry import ensure_trained, eval_policy, RUNS

RESULTS = Path("results")

VARIANTS = {
    "base":            {},
    "perfect_obs":     {"detection": {"p_report": 1.0, "detect_gain": 0.0}},
    "low_detect_gain": {"detection": {"detect_gain": 0.25}},
    "high_detect_gain":{"detection": {"detect_gain": 0.8}},
    "weak_deterrence": {"detection": {"deterrence_slope": 0.10}},
    "strong_deterrence":{"detection": {"deterrence_slope": 0.30}},
    "no_socio_obs":    {"env": {"include_socioeconomic_obs": False}},
    "budget_10":       {"env": {"n_units": 10}},
    "budget_40":       {"env": {"n_units": 40}},
    "history_1":       {"env": {"history": 1}},
    "history_8":       {"env": {"history": 8}},
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--eval-seeds", type=int, nargs="+", default=list(range(300, 308)))
    ap.add_argument("--timesteps", type=int, default=150000)
    ap.add_argument("--penalty", type=float, default=0.06)
    ap.add_argument("--only", nargs="+", default=None)
    args = ap.parse_args()
    RESULTS.mkdir(exist_ok=True)
    region = load_region_info(20)
    tag = f"abl_e{args.eval_seeds[0]}n{len(args.eval_seeds)}"

    rows = []
    variants = {k: VARIANTS[k] for k in (args.only or VARIANTS)}
    for vname, over in variants.items():
        for s in args.seeds:
            for kind, extra in [("ppo", {"rl": {"algo": "ppo"}}),
                                ("fair", {"rl": {"algo": "penalized_ppo"},
                                          "fairness": {"penalty": args.penalty}})]:
                merged = _merge(over, {"rl": {"seed": s, "total_timesteps": args.timesteps},
                                       "env": {"seed": s}}, extra)
                cfg = load_config("configs/base.yaml", **merged)
                name = f"abl_{vname}_{kind}_s{s}"
                print(f"train {name}", flush=True)
                out = ensure_trained(name, cfg, cfg.rl.algo)
                ecfg = load_config("configs/base.yaml", **_merge(over, {}))
                df = eval_policy(name, out, ecfg, args.eval_seeds, tag)
                df = df.assign(variant=vname, kind=kind, tseed=s)
                rows.append(df)

    alldf = pd.concat(rows, ignore_index=True)
    alldf.to_pickle(RESULTS / "e5_ablations_raw.pkl")

    summ = []
    for (vname, kind), g in alldf.groupby(["variant", "kind"]):
        s = summarise(g.assign(policy=f"{vname}_{kind}")).iloc[0]
        panel = build_panel(g.assign(policy="x"), region)
        try:
            coef = incidence_regression(panel, "x")["hardship"]["coef"]
        except Exception:
            coef = np.nan
        summ.append({"variant": vname, "kind": kind, "latent": s["latent"],
                     "latent_ci": s["latent_ci"], "disparity": s["disparity"],
                     "gini": s["gini"], "hardship_incidence_coef": coef})
    out = pd.DataFrame(summ).sort_values(["variant", "kind"])
    out.to_csv(RESULTS / "e5_ablations.csv", index=False)
    print("\n== E5 ==\n", out.to_string(index=False))


def _merge(*dicts):
    m: dict = {}
    for d in dicts:
        for k, v in d.items():
            if isinstance(v, dict):
                m.setdefault(k, {}).update(v)
            else:
                m[k] = v
    return m


if __name__ == "__main__":
    main()
