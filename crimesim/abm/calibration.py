"""Calibration gate for the vectorised port of ``crime_model_v5``.

Run ``python -m crimesim.abm.calibration``. Checks that the NumPy generator
reproduces the qualitative signatures the Phase-2 paper relies on, before any RL
is built on top of it:

  1. SDT elasticity   -- raising community hardship raises violent crime, by a
     margin in the same ballpark as v5's reported +42.6% (Low -> High).
  2. Emergence        -- from a spatially uniform risk grid, crime forms spatial
     clusters (high max/mean concentration).
  3. Routine-activity -- night shifts carry more crime than daytime shifts.
  4. Monotonicity     -- Low < Medium < High total crime.

Exits non-zero if any check fails.
"""
from __future__ import annotations

import dataclasses
import sys

import numpy as np

from crimesim.abm.generator import CrimeGenerator
from crimesim.abm.regions import RegionInfo, load_region_info
from crimesim.config import ABMConfig, DetectionConfig

SCENARIOS = {"Low": 0.0, "Medium": 0.1, "High": 0.2}


def _shift_region(reg: RegionInfo, delta: float) -> RegionInfo:
    return dataclasses.replace(
        reg,
        hardship=np.clip(reg.hardship + delta, 0, 1),
        unemployment=np.clip(reg.unemployment + delta, 0, 1),
    )


def _run(cfg, det, reg, seed, n_shifts=90):
    gen = CrimeGenerator(cfg, det, reg)
    gen.reset(np.random.default_rng(seed))
    acc = np.zeros((cfg.grid, cfg.grid))
    total = night_c = night_n = day_c = day_n = 0.0
    for _ in range(n_shifts):
        out = gen.run_shift(None)
        acc += out["latent_cell"]
        total += out["latent_cell"].sum()
        if out["hour"] >= 20 or out["hour"] <= 2:
            night_c += out["latent_cell"].sum(); night_n += 1
        else:
            day_c += out["latent_cell"].sum(); day_n += 1
    return {"total": total, "acc": acc,
            "night_rate": night_c / max(night_n, 1), "day_rate": day_c / max(day_n, 1),
            "concentration": float(acc.max() / acc.mean()),
            "nonzero_frac": float((acc > 0).mean())}


def run_calibration(n_seeds: int = 8, verbose: bool = True) -> dict:
    reg0 = load_region_info(20)
    cfg = ABMConfig()
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
        "sdt_elasticity_pct": (elasticity, 20.0 <= elasticity <= 70.0),
        "monotonic_low_med_high": (
            (totals["Low"].mean(), totals["Medium"].mean(), totals["High"].mean()),
            totals["Low"].mean() < totals["Medium"].mean() < totals["High"].mean()),
        "spatial_clustering_ratio": (conc, conc >= 4.0),
        "not_degenerate_coverage": (nz, 0.4 <= nz <= 0.999),
        "night_gt_day": ((night, day), night > day),
    }

    if verbose:
        print("=== CrimeGenerator calibration ===")
        print(f"  Low/Med/High total crime (mean over {n_seeds} seeds): "
              f"{lo:,.0f} / {totals['Medium'].mean():,.0f} / {hi:,.0f}")
        print(f"  SDT elasticity (Low->High): +{elasticity:.1f}%   [v5 reported +42.6%]")
        print(f"  spatial clustering max/mean: {conc:.1f}   nonzero cells: {nz:.1%}")
        print(f"  night vs day crime rate/shift: {night:,.0f} vs {day:,.0f}")
        for k, (val, ok) in checks.items():
            print(f"    [{'PASS' if ok else 'FAIL'}] {k}: {val}")

    return {"ok": all(ok for _, ok in checks.values()), "checks": checks,
            "elasticity_pct": elasticity}


if __name__ == "__main__":
    sys.exit(0 if run_calibration()["ok"] else 1)
