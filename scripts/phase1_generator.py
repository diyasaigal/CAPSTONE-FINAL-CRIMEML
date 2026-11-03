import crimesim._env_setup  # noqa: F401
"""JASSS Phase 1: measure the generator under candidate settings (no RL).

For every combination of the ``--grid`` values (on top of ``configs/base.yaml``
and any ``--fixed`` overrides) and each seed, the generator runs under
proportional patrol (intensity 1, as in the env warm-up) for ``--warmup`` shifts
and then ``--shifts`` measured shifts. Reported per setting (mean over seeds):

  latent        latent crime per shift
  contrast      latent crime per resident, top / bottom hardship quintile (A6)
  night_day     crime per night hour / per day hour, hours 20-02 (A5)
  drift_pct     linear trend of crime per shift, % of the mean per 100 shifts (A9)
  count_cv      coefficient of variation of a district's crime per shift (A7)
  gate_ok       all five calibration checks pass on this ABM config

Appends to ``results/phase1/generator_sweep.csv``. Example:

    PYTHONPATH=. python scripts/phase1_generator.py --tag recovery \\
        --fixed abm.hardship_in_risk=false abm.population_weighted=true \\
        --grid abm.stress_recovery=0.18,0.12,0.09,0.06
"""
import argparse
import itertools
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("results/phase1")


def _parse(v: str):
    low = v.lower()
    if low in ("true", "false"):
        return low == "true"
    try:
        return int(v)
    except ValueError:
        return float(v)


def _overrides(pairs: dict) -> dict:
    out: dict = {}
    for k, v in pairs.items():
        sec, key = k.split(".")
        out.setdefault(sec, {})[key] = v
    return out


def _one(job):
    pairs, seed, warmup, shifts = job
    from crimesim.abm.calibration import night_day_ratio
    from crimesim.abm.generator import CrimeGenerator
    from crimesim.abm.regions import region_for
    from crimesim.config import load_config
    cfg = load_config("configs/base.yaml", **_overrides(pairs))
    region = region_for(cfg)
    gen = CrimeGenerator(cfg.abm, cfg.detection, region)
    gen.reset(np.random.default_rng(seed))
    patrol = cfg.env.n_units * region.pop_weight
    for _ in range(warmup):
        gen.run_shift(patrol)
    lat, hours, counts = [], [], []
    for _ in range(shifts):
        out = gen.run_shift(patrol)
        lat.append(out["latent_region"])
        hours.extend(out["sub_hours"]); counts.extend(out["sub_counts"])
    lat = np.array(lat)                                   # (shifts, R)
    g = region.quantile_groups("hardship", 5)
    per_res = lat.sum(0) / region.pop_weight              # crime per unit population share
    pw = region.pop_weight
    q = [lat.sum(0)[g == k].sum() / pw[g == k].sum() for k in (0, 4)]
    tot = lat.sum(1)
    slope = np.polyfit(np.arange(shifts), tot, 1)[0]
    night, day = night_day_ratio(hours, counts)
    cv = (lat.std(0) / np.clip(lat.mean(0), 1e-9, None)).mean()
    return {**pairs, "seed": seed, "latent": tot.mean(), "contrast": q[1] / q[0],
            "night_day": night / day, "drift_pct": 100 * 100 * slope / tot.mean(),
            "count_cv": cv, "per_res_max_min": per_res.max() / per_res.min()}


def _gate(pairs):
    from crimesim.abm.calibration import run_calibration
    from crimesim.config import load_config
    cfg = load_config("configs/base.yaml", **_overrides(pairs))
    r = run_calibration(6, verbose=False, cfg=cfg.abm)
    night, day = r["checks"]["night_gt_day"][0]
    return {**pairs, "gate_ok": r["ok"], "gate_night_day": night / day,
            "gate_elasticity": r["elasticity_pct"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--fixed", nargs="*", default=[])
    ap.add_argument("--grid", nargs="*", default=[])
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--shifts", type=int, default=140)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--no-gate", action="store_true")
    a = ap.parse_args()

    fixed = {k: _parse(v) for k, v in (s.split("=") for s in a.fixed)}
    keys = [s.split("=")[0] for s in a.grid]
    vals = [[_parse(x) for x in s.split("=")[1].split(",")] for s in a.grid]
    settings = [{**fixed, **dict(zip(keys, combo))} for combo in itertools.product(*vals)]
    jobs = [(s, seed, a.warmup, a.shifts) for s in settings for seed in range(a.seeds)]

    t0 = time.time()
    with ProcessPoolExecutor(a.workers) as ex:
        rows = list(ex.map(_one, jobs))
        gates = [] if a.no_gate else list(ex.map(_gate, settings))
    df = pd.DataFrame(rows)
    cols = list(settings[0].keys())
    summ = df.groupby(cols, sort=False).agg(
        latent=("latent", "mean"), contrast=("contrast", "mean"), night_day=("night_day", "mean"),
        drift_pct=("drift_pct", "mean"), drift_sd=("drift_pct", "std"),
        count_cv=("count_cv", "mean")).reset_index()
    if gates:
        summ = summ.merge(pd.DataFrame(gates), on=cols)
    summ.insert(0, "tag", a.tag)
    summ["warmup"], summ["shifts"], summ["seeds"] = a.warmup, a.shifts, a.seeds
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "generator_sweep.csv"
    summ.to_csv(path, mode="a", header=not path.exists(), index=False)
    pd.set_option("display.width", 250)
    print(summ.drop(columns=["tag", "warmup", "shifts", "seeds"]).round(3).to_string(index=False))
    print(f"{len(jobs)} runs in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
