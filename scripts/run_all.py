import crimesim._env_setup  # noqa: F401  (must precede numpy/torch)
"""Full paper pipeline: train every policy, evaluate on held-out ABM seeds,
compute E1-E4, write results/ + figures/. Resumable (skips finished runs).

    python scripts/run_all.py --train-seeds 0 1 2 3 4 --eval-seeds 20 ... 34 \
        --alphas 0.3 0.7 1.5 3.0 --timesteps 400000
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from crimesim.config import load_config
from crimesim.abm.regions import load_region_info
from crimesim.eval import figures as F
from crimesim.eval.feedback_metrics import feedback_trajectories, summary_row
from crimesim.eval.incidence import build_panel, incidence_regression, exposure_by_hardship_quintile
from crimesim.eval.pareto import frontier_table, knee_point
from crimesim.eval.rollout import summarise
from crimesim.experiments.registry import (RUNS, ensure_trained, eval_baselines,
                                           eval_policy, pool_by_family, rl_specs)

RESULTS = Path("results")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--train-seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--eval-seeds", type=int, nargs="+",
                    default=list(range(100, 110)))
    ap.add_argument("--penalties", type=float, nargs="+", default=[0.02, 0.06, 0.15, 0.4],
                    help="fairness penalty weights for the frontier sweep")
    ap.add_argument("--timesteps", type=int, default=None)
    args = ap.parse_args()

    RESULTS.mkdir(exist_ok=True)
    specs = rl_specs(args.config, args.train_seeds, args.penalties)
    mid_pen = args.penalties[len(args.penalties) // 2]

    # ---- train ----
    for i, (name, (cfg, algo)) in enumerate(specs.items(), 1):
        if args.timesteps:
            cfg = replace_timesteps(cfg, args.timesteps)
        print(f"[{i}/{len(specs)}] train {name}", flush=True)
        ensure_trained(name, cfg, algo)

    # ---- evaluate ----
    eval_cfg = load_config(args.config)
    tag = "e" + "_".join(map(str, args.eval_seeds[:2])) + f"n{len(args.eval_seeds)}"
    frames = [eval_baselines(eval_cfg, args.eval_seeds, RESULTS, tag)]
    for name, (cfg, algo) in specs.items():
        ecfg = eval_cfg
        if name.startswith("ppo_oracle"):
            ecfg = load_config(args.config, env={"reward_on": "latent_coverage",
                                                 "observe_latent": True})
        frames.append(eval_policy(name, RUNS / name, ecfg, args.eval_seeds, tag))
    df = pd.concat(frames, ignore_index=True)
    dfp = pool_by_family(df)
    dfp.to_pickle(RESULTS / "all_shifts.pkl")

    region = load_region_info(eval_cfg.abm.grid)

    # ---- E1: efficiency & equity ----
    e1 = summarise(dfp)
    e1.to_csv(RESULTS / "e1_summary.csv", index=False)
    F.fig_efficiency_equity(e1)
    cols = ["policy", "lat_cov", "lat_cov_ci", "rec_cov", "latent", "recorded",
            "disparity", "disparity_shift", "gini"]
    print("\n== E1 ==\n", e1[[c for c in cols if c in e1]].to_string(index=False))

    # ---- E2: feedback trajectories ----
    fair_mid = f"pen_p{mid_pen}"
    traj = {p: feedback_trajectories(dfp[dfp.policy == p])
            for p in ["uniform", "ewma_hotspot", "ppo", fair_mid]
            if p in dfp.policy.unique()}
    F.fig_feedback_trajectories(traj)
    e2 = {p: summary_row(dfp[dfp.policy == p]) for p in dfp.policy.unique()}
    pd.DataFrame(e2).T.to_csv(RESULTS / "e2_feedback.csv")
    print("\n== E2 ==\n", pd.DataFrame(e2).T.to_string())

    # ---- E3: incidence ----
    panel = build_panel(dfp, region)
    e3 = {}
    for p in dfp.policy.unique():
        try:
            r = incidence_regression(panel, p, with_poverty=True)
            e3[p] = {"hardship_coef": r["hardship"]["coef"], "hardship_p": r["hardship"]["p"],
                     "r2": r["r2"]}
        except Exception:
            pass
    pd.DataFrame(e3).T.to_csv(RESULTS / "e3_incidence.csv")
    exq = {p: exposure_by_hardship_quintile(dfp, region, p)
           for p in ["uniform", "ewma_hotspot", "ppo", "oracle"] if p in dfp.policy.unique()}
    if fair_mid in dfp.policy.unique():
        exq["ppo_fair"] = exposure_by_hardship_quintile(dfp, region, fair_mid)
    F.fig_incidence(exq)
    print("\n== E3 ==\n", pd.DataFrame(e3).T.to_string())

    # ---- E4: Pareto ----
    fam = ["uniform", "ewma_hotspot", "rf_hotspot", "oracle", "ppo", "ppo_oracle"]
    fam += [f"pen_p{p}" for p in args.penalties]
    front = frontier_table(dfp[dfp.policy.isin(fam)])
    front.to_csv(RESULTS / "e4_pareto.csv", index=False)
    F.fig_pareto(front)
    knee = knee_point(front)
    (RESULTS / "e4_knee.json").write_text(json.dumps(
        {k: (float(v) if isinstance(v, (int, float, np.floating)) else str(v))
         for k, v in knee.items()}, indent=2))
    print("\n== E4 Pareto ==\n", front.to_string(index=False))
    print("knee:", knee.get("policy"))

    print(f"\nfigures -> {Path('figures').resolve()}   results -> {RESULTS.resolve()}")


def replace_timesteps(cfg, n):
    import dataclasses
    return dataclasses.replace(cfg, rl=dataclasses.replace(cfg.rl, total_timesteps=n))


if __name__ == "__main__":
    main()
