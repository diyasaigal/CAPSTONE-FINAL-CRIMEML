import crimesim._env_setup  # noqa
"""Recompute E1-E4 tables and figures from results/all_shifts.pkl without
retraining or re-evaluating. Use after changing an eval/figure module.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from crimesim.abm.regions import load_region_info
from crimesim.eval import figures as F
from crimesim.eval.feedback_metrics import feedback_trajectories, summary_row
from crimesim.eval.incidence import build_panel, exposure_by_hardship_quintile, incidence_regression
from crimesim.eval.pareto import frontier_table, knee_point
from crimesim.eval.rollout import summarise

RESULTS = Path("results")
PENALTIES = [0.02, 0.06, 0.15, 0.4]
MID = f"pen_p{PENALTIES[len(PENALTIES) // 2]}"

dfp = pd.read_pickle(RESULTS / "all_shifts.pkl")
region = load_region_info(20)
pol = list(dfp.policy.unique())

e1 = summarise(dfp)
e1.to_csv(RESULTS / "e1_summary.csv", index=False)
F.fig_efficiency_equity(e1)
print("== E1 ==\n", e1[["policy", "lat_cov", "lat_cov_ci", "rec_cov", "disparity",
                        "disparity_shift", "gini"]].to_string(index=False))

traj = {p: feedback_trajectories(dfp[dfp.policy == p])
        for p in ["uniform", "ewma_hotspot", "ppo", MID] if p in pol}
F.fig_feedback_trajectories(traj)
e2 = pd.DataFrame({p: summary_row(dfp[dfp.policy == p]) for p in pol}).T
e2.to_csv(RESULTS / "e2_feedback.csv")
print("\n== E2 ==\n", e2.to_string())

panel = build_panel(dfp, region)
e3 = {}
for p in pol:
    try:
        r = incidence_regression(panel, p, with_poverty=True)
        e3[p] = {"hardship_coef": r["hardship"]["coef"], "hardship_p": r["hardship"]["p"],
                 "r2": r["r2"]}
    except Exception:
        pass
pd.DataFrame(e3).T.to_csv(RESULTS / "e3_incidence.csv")
exq = {p: exposure_by_hardship_quintile(dfp, region, p)
       for p in ["uniform", "ewma_hotspot", "ppo", "oracle"] if p in pol}
if MID in pol:
    exq["ppo_fair"] = exposure_by_hardship_quintile(dfp, region, MID)
F.fig_incidence(exq)
print("\n== E3 ==\n", pd.DataFrame(e3).T.to_string())

fam = ["uniform", "ewma_hotspot", "rf_hotspot", "oracle", "ppo", "ppo_oracle"]
fam += [f"pen_p{p}" for p in PENALTIES]
front = frontier_table(dfp[dfp.policy.isin(fam)])
front.to_csv(RESULTS / "e4_pareto.csv", index=False)
F.fig_pareto(front)
knee = knee_point(front)
(RESULTS / "e4_knee.json").write_text(json.dumps(
    {k: (float(v) if isinstance(v, (int, float, np.floating)) else str(v))
     for k, v in knee.items()}, indent=2))
print("\n== E4 ==\n", front.to_string(index=False), "\nknee:", knee.get("policy"))
print("\nfigures ->", Path("figures").resolve())
