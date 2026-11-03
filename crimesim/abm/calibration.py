"""Calibration gate for the vectorised port of ``crime_model_v5``.

Run ``python -m crimesim.abm.calibration``. Checks that the NumPy generator
reproduces the qualitative signatures the Phase-2 paper relies on, before any RL
is built on top of it:

  1. SDT elasticity   -- raising community hardship clearly raises violent crime
     (at least +10%, at most +70%, Low -> High). The floor was 20%, set around
     v5's reported +42.6%; that figure includes v5's double count of hardship
     (stress and risk term), so the JASSS revision, which counts it once
     (``hardship_in_risk: false``), lowered it to "clearly positive".
  2. Emergence        -- from a spatially uniform risk grid, crime forms spatial
     clusters (high max/mean concentration).
  3. Routine-activity -- night hours (20-02) carry at least 10% more crime per
     hour than day hours (real Chicago records: 1.18). Was "night > day", which
     a flat profile of 1.01 passed.
  4. Monotonicity     -- Low < Medium < High total crime.

Exits non-zero if any check fails.
"""
from __future__ import annotations

import dataclasses
import sys

import numpy as np

from crimesim.abm.generator import CrimeGenerator
from crimesim.abm.regions import RegionInfo, load_region_info, with_population
from crimesim.config import ABMConfig, DetectionConfig

SCENARIOS = {"Low": 0.0, "Medium": 0.1, "High": 0.2}
ELASTICITY_MIN = 10.0     # % crime rise Low -> High
NIGHT_DAY_MIN = 1.10      # crime per night hour / per day hour


def _shift_region(reg: RegionInfo, delta: float) -> RegionInfo:
    return dataclasses.replace(
        reg,
        hardship=np.clip(reg.hardship + delta, 0, 1),
        unemployment=np.clip(reg.unemployment + delta, 0, 1),
    )


def is_night(hour) -> np.ndarray:
    """The model's night window (hours 20-02), as in ``dynamics.crime_events``."""
    hour = np.asarray(hour)
    return (hour >= 20) | (hour <= 2)


def night_day_ratio(hours, counts) -> tuple[float, float]:
    """Mean crime per night hour and per day hour, from hourly (sub-step) counts.

    The single definition used by the gate and by the validation against real
    records, so the two report the same quantity. (The gate used to classify a
    whole 4-hour shift by the hour it ended, which counts e.g. hours 17-20 as night.)
    """
    hours, counts = np.asarray(hours), np.asarray(counts, float)
    n = is_night(hours)
    return float(counts[n].mean()), float(counts[~n].mean())


def _run(cfg, det, reg, seed, n_shifts=90):
    gen = CrimeGenerator(cfg, det, reg)
    gen.reset(np.random.default_rng(seed))
    acc = np.zeros((cfg.grid, cfg.grid))
    total = 0.0
    hours, counts = [], []
    for _ in range(n_shifts):
        out = gen.run_shift(None)
        acc += out["latent_cell"]
        total += out["latent_cell"].sum()
        hours.extend(out["sub_hours"]); counts.extend(out["sub_counts"])
    night_rate, day_rate = night_day_ratio(hours, counts)
    return {"total": total, "acc": acc,
            "night_rate": night_rate, "day_rate": day_rate,
            "concentration": float(acc.max() / acc.mean()),
            "nonzero_frac": float((acc > 0).mean())}


def run_calibration(n_seeds: int = 8, verbose: bool = True, cfg: ABMConfig | None = None) -> dict:
    """Gate on ``cfg`` (default: the original model's values, ``ABMConfig()``).
    Pass the experiments' ABM settings (``--config configs/base.yaml``) to gate
    the configuration the results actually use."""
    cfg = cfg or ABMConfig()
    reg0 = with_population(load_region_info(cfg.grid), cfg.population_weighted)
    det = DetectionConfig(deterrence_slope=0.0)

    totals, results = {}, {}
    for name, delta in SCENARIOS.items():
        reg = _shift_region(reg0, delta)
        runs = [_run(cfg, det, reg, s) for s in range(n_seeds)]
        totals[name] = np.array([r["total"] for r in runs])
        results[name] = runs

    lo, hi = totals["Low"].mean(), totals["High"].mean()
    elasticity = 100.0 * (hi - lo) / lo
    conc = np.mean([r["concentration"] for r in results["Low"]])
    nz = np.mean([r["nonzero_frac"] for r in results["Low"]])
    night = np.mean([r["night_rate"] for r in results["Low"]])
    day = np.mean([r["day_rate"] for r in results["Low"]])

    checks = {
        "sdt_elasticity_pct": (elasticity, ELASTICITY_MIN <= elasticity <= 70.0),
        "monotonic_low_med_high": (
            (totals["Low"].mean(), totals["Medium"].mean(), totals["High"].mean()),
            totals["Low"].mean() < totals["Medium"].mean() < totals["High"].mean()),
        "spatial_clustering_ratio": (conc, conc >= 4.0),
        "not_degenerate_coverage": (nz, 0.4 <= nz <= 0.999),
        "night_gt_day": ((night, day), night >= NIGHT_DAY_MIN * day),
    }

    if verbose:
        print("=== CrimeGenerator calibration ===")
        print(f"  Low/Med/High total crime (mean over {n_seeds} seeds): "
              f"{lo:,.0f} / {totals['Medium'].mean():,.0f} / {hi:,.0f}")
        print(f"  SDT elasticity (Low->High): +{elasticity:.1f}%   [v5 reported +42.6%]")
        print(f"  spatial clustering max/mean: {conc:.1f}   nonzero cells: {nz:.1%}")
        print(f"  night vs day crime per hour: {night:,.1f} vs {day:,.1f} (ratio {night / day:.2f})")
        for k, (val, ok) in checks.items():
            print(f"    [{'PASS' if ok else 'FAIL'}] {k}: {val}")

    return {"ok": all(ok for _, ok in checks.values()), "checks": checks,
            "elasticity_pct": elasticity}


if __name__ == "__main__":
    import argparse
    from crimesim.config import load_config
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", help="gate this config's ABM settings instead of the code defaults")
    ap.add_argument("--seeds", type=int, default=8)
    a = ap.parse_args()
    cfg = load_config(a.config).abm if a.config else None
    sys.exit(0 if run_calibration(a.seeds, cfg=cfg)["ok"] else 1)
