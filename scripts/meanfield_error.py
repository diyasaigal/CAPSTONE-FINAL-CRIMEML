"""JASSS Phase 0, item A4: how close is the mean-field model to the simulator?

Reads ``results/theory/compare.csv`` (written by ``scripts/analytic_model.py``)
and reports, per rule-setting pair and overall, the absolute percentage error
of the model on two scales:

  * raw disparity (top / bottom hardship quintile exposure), and
  * disparity x each one's own GreedyOracle. The model's oracle is not the
    simulator's oracle, so this error also carries the error in the reference.

Also the R^2 of log(x oracle) and the pair count. Constant detection and
perfect observation are the same equations in the model (detection does not
depend on patrol), so the number of distinct model curves is reported too.
Writes ``results/phase0/meanfield_error.csv`` and ``meanfield_error.json``.

    python scripts/meanfield_error.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("results/phase0")


def ape(model, sim) -> pd.Series:
    return 100 * (model / sim - 1).abs()


def r2(model, sim) -> float:
    a, b = np.log(model), np.log(sim)
    return float(1 - ((a - b) ** 2).sum() / ((b - b.mean()) ** 2).sum())


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    c = pd.read_csv("results/theory/compare.csv")
    summary = json.loads(Path("results/theory/summary.json").read_text())
    c["ape_disparity"] = ape(c.model_disparity, c.sim_disparity)
    c["ape_x_oracle"] = ape(c.model_x_oracle, c.sim_x_oracle)
    c.to_csv(OUT / "meanfield_error.csv", index=False)

    def block(g):
        return {"pairs": len(g),
                "disparity_mean_pct": g.ape_disparity.mean(), "disparity_max_pct": g.ape_disparity.max(),
                "x_oracle_mean_pct": g.ape_x_oracle.mean(), "x_oracle_max_pct": g.ape_x_oracle.max(),
                "r2_log_disparity": r2(g.model_disparity, g.sim_disparity),
                "r2_log_x_oracle": r2(g.model_x_oracle, g.sim_x_oracle)}

    base = summary["base"]
    out = {"all": block(c),
           "by_variant": {v: block(g) for v, g in c.groupby("variant")},
           "distinct_model_curves": int(c.groupby("variant").model_disparity
                                        .apply(lambda s: tuple(s.round(6))).nunique()),
           "oracle_disparity_model": base["oracle_disparity_model"],
           "oracle_disparity_sim": base["oracle_disparity_sim"],
           "oracle_error_pct": 100 * (base["oracle_disparity_model"] / base["oracle_disparity_sim"] - 1)}
    (OUT / "meanfield_error.json").write_text(json.dumps(out, indent=2))

    print(c[["variant", "beta", "model_disparity", "sim_disparity", "ape_disparity",
             "model_x_oracle", "sim_x_oracle", "ape_x_oracle"]].round(2).to_string(index=False))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
