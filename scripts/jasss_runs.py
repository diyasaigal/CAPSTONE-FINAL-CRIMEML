import crimesim._env_setup  # noqa: F401
"""JASSS Phase 2: every model of the revision, on the frozen generator.

One queue, one set of seeds: training seeds 0-9, evaluation seeds 200-229,
reference seeds 1000-1009 (penalty targets, need vector), rule-search seeds
1000-1002. Every setting is ``configs/jasss_base.yaml`` + a generator variant
(``crimesim.experiments.jasss``) + detection overrides, and every policy is
scored later against references computed in its own setting.

Stages (all resumable: finished work is cached on disk and skipped, and a model
interrupted mid-training continues from its last resume point, written every
25k steps; rerun the same command after any interruption):
  refs       per setting, on reference seeds: GreedyOracle disparity (per shift
             and cumulative), need vector = mean latent crime per district under
             uniform patrol, NeedProportional disparity -> results/jasss/refs/
  rules      fixed-rule grid search (192 rules) for the detection settings
             -> results/jasss/rules/
  train      train each model, then evaluate it on the evaluation seeds
             -> runs/jasss/<setting>__<family>_s<seed>/
  baselines  rule-based policies, references and the best fixed rules on the
             evaluation seeds, per setting -> results/jasss/eval/<setting>/

Groups (change-list items): A10 common experiment (detection x observation x
reward), A11 hardship tilt off, A13 penalties, A14 sensitivity, A6A7 contrast
and agent count, A17 baseline reporting. Detection sweeps hold detection at
saturation at 0.85 (p_report + detect_gain), so q never exceeds 1.

    PYTHONPATH=. python scripts/jasss_runs.py --dry-run
    PYTHONPATH=. python scripts/jasss_runs.py --timing
    PYTHONPATH=. python scripts/jasss_runs.py --workers 4                # everything
    PYTHONPATH=. python scripts/jasss_runs.py --groups A10 A11 --seeds 0 1 2 3 4
"""
import argparse
import itertools
import json
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("results/jasss")
RUN_DIR = "jasss"
STEPS = 400_000
TRAIN_SEEDS = list(range(10))
EVAL_SEEDS, EVAL_TAG = list(range(200, 230)), "e200n30"
REF_SEEDS = list(range(1000, 1010))
SEARCH_SEEDS = [1000, 1001, 1002]
SAT = 0.85                                    # p_report + detect_gain in the base setting
RESUME_EVERY = 25_000                         # an interrupted model loses at most this many steps

DET_CONST = {"detection": {"p_report": 0.667, "detect_gain": 0.0}}
DET_PERFECT = {"detection": {"p_report": 1.0, "detect_gain": 0.0}}


def _report(p):
    return {"detection": {"p_report": p, "detect_gain": round(SAT - p, 4)}}


# setting -> (generator variant, overrides)
SETTINGS = {
    "base": ("base", {}),
    "const": ("base", DET_CONST),
    "perfect": ("base", DET_PERFECT),
    "gain_low": ("base", {"detection": {"detect_gain": 0.25}}),
    "gain_high": ("base", {"detection": {"detect_gain": 0.80}}),
    "deter_weak": ("base", {"detection": {"deterrence_slope": 0.10}}),
    "deter_strong": ("base", {"detection": {"deterrence_slope": 0.30}}),
    **{f"report_{p:.2f}": ("base", _report(p)) for p in (0.15, 0.40, 0.50, 0.60, 0.70)},
    **{v: (v, {}) for v in ("contrast_low", "contrast_high", "agents_500", "agents_1000")},
    **{f"{v}_const": (v, DET_CONST) for v in ("contrast_low", "contrast_high", "agents_500", "agents_1000")},
}
RULE_SETTINGS = ["base", "const", "perfect"]

LATENT_REWARD = {"env": {"reward_on": "latent_coverage"}}
SEES_LATENT = {"env": {"observe_latent": True}}


def _merge(*dicts) -> dict:
    out: dict = {}
    for d in dicts:
        for sec, vals in d.items():
            out.setdefault(sec, {}).update(vals)
    return out


def setting_cfg(setting: str, *overrides):
    from crimesim.experiments.jasss import jasss_config
    variant, det = SETTINGS[setting]
    return jasss_config(variant, **_merge(det, *overrides))


# ------------------------------------------------------------------ references
def refs(setting: str) -> dict:
    """Reference quantities of ``setting`` on the reference seeds (cached)."""
    path = OUT / "refs" / f"{setting}.json"
    if path.exists():
        return json.loads(path.read_text())
    from crimesim.eval.rollout import rollout
    from crimesim.rl.baselines import GreedyOracle, NeedProportional, UniformPolicy
    cfg = setting_cfg(setting)
    ora = rollout(GreedyOracle(), cfg, REF_SEEDS, keep_arrays=False)
    uni = rollout(UniformPolicy(), cfg, REF_SEEDS, keep_arrays=True)
    need = np.stack(uni.latent_region.to_numpy()).mean(0)
    nee = rollout(NeedProportional(need), cfg, REF_SEEDS, keep_arrays=False)
    out = {"setting": setting, "seeds": REF_SEEDS,
           "oracle_shift": float(ora.disparity_shift.mean()),
           "oracle_cum": float(ora.groupby("seed").disparity_cum.last().mean()),
           "need": need.tolist(),
           "need_shift": float(nee.disparity_shift.mean()),
           "need_cum": float(nee.groupby("seed").disparity_cum.last().mean())}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1))
    return out


# ------------------------------------------------------------------ models
def model_specs(seeds) -> list[tuple]:
    """(group, setting, family, overrides, algo, seed) for every model."""
    base = refs("base")
    par = lambda w: {"rl": {"algo": "penalized_ppo"}, "fairness": {"penalty": w}}  # noqa: E731
    excess = lambda w, ref: {"rl": {"algo": "penalized_ppo"},  # noqa: E731
                             "fairness": {"penalty": w, "penalty_mode": "excess_log_ref",
                                          "ref_disparity": round(ref, 4)}}
    fams = [
        # A10: detection x observation x reward, plus the parity agent per detection setting
        ("A10", "base", "ppo", {}),
        ("A10", "base", "rewlat", LATENT_REWARD),
        ("A10", "base", "obslat", SEES_LATENT),
        ("A10", "base", "oracleppo", _merge(SEES_LATENT, LATENT_REWARD)),
        ("A10", "const", "ppo", {}),
        ("A10", "const", "rewlat", LATENT_REWARD),
        ("A10", "perfect", "ppo", {}),
        *[("A10", s, "par0.06", par(0.06)) for s in ("base", "const", "perfect")],
        # A11: hardship tilt
        ("A11", "base", "w0", {"env": {"w_mode": "zero"}}),
        ("A11", "base", "wneg", {"env": {"w_mode": "nonpos"}}),
        ("A11", "base", "w0_nosocio", {"env": {"w_mode": "zero", "include_socioeconomic_obs": False}}),
        # A13: penalties (parity 0.06 is in A10)
        *[("A13", "base", f"par{w}", par(w)) for w in (0.02, 0.15, 0.4)],
        *[("A13", "base", f"penref{w}", excess(w, base["oracle_shift"])) for w in (0.05, 0.15, 0.5)],
        ("A13", "base", "penneed0.15", excess(0.15, base["need_shift"])),
        ("A13", "base", "lag", {"rl": {"algo": "lagrangian_ppo"},
                                "fairness": {"alpha": round(base["oracle_cum"], 4)}}),
        # A14: sensitivity of naive PPO
        *[("A14", s, "ppo", {}) for s in ("gain_low", "gain_high", "report_0.15", "report_0.50",
                                          "deter_weak", "deter_strong")],
        # A6 / A7: crime contrast and agent count, with and without the detection channel
        *[("A6A7", s, "ppo", {}) for v in ("contrast_low", "contrast_high", "agents_500", "agents_1000")
          for s in (v, f"{v}_const")],
        # A17: baseline reporting inside the survey range (0.50 is in A14)
        *[("A17", f"report_{p:.2f}", "ppo", {}) for p in (0.40, 0.60, 0.70)],
    ]
    return [(g, s, f, o, o.get("rl", {}).get("algo", "ppo"), seed)
            for g, s, f, o in fams for seed in seeds]


def run_name(setting, family, seed) -> str:
    return f"{RUN_DIR}/{setting}__{family}_s{seed}"


def _train_job(job, steps=STEPS):
    group, setting, family, over, algo, seed = job
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    from crimesim.config import load_config
    from crimesim.experiments.registry import RUNS, ensure_trained, eval_policy
    name = run_name(setting, family, seed)
    t0 = time.time()
    try:
        cfg = setting_cfg(setting, over, {"rl": {"seed": seed, "total_timesteps": steps, "algo": algo,
                                                 "resume_every": RESUME_EVERY},
                                          "env": {"seed": seed}})
        existed = (RUNS / name / "ppo_model.zip").exists()
        out = ensure_trained(name, cfg, algo)
        t1 = time.time()
        eval_policy(name, out, load_config(out / "config.yaml"), EVAL_SEEDS, EVAL_TAG)
        return name, "cached" if existed else "trained", t1 - t0, time.time() - t1, None
    except Exception:
        return name, "FAILED", time.time() - t0, 0.0, traceback.format_exc()


# ------------------------------------------------------------------ rules and baselines
def _rule_score(args):
    setting, beta, w, mu = args
    cache = OUT / "rules" / setting / f"b{beta:g}_w{w:g}_m{mu:g}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    sys.path.insert(0, str(Path(__file__).parent))
    from rule_search import FixedRule
    from crimesim.eval.rollout import rollout
    df = rollout(FixedRule(beta, w, mu), setting_cfg(setting), SEARCH_SEEDS)
    ep = df.groupby("seed").agg(rec=("recorded_coverage", "mean"), lat=("latent_coverage", "mean"),
                                disp=("disparity_cum", "last"))
    row = {"setting": setting, "beta": beta, "w": w, "mu": mu, "rec_cov": float(ep.rec.mean()),
           "lat_cov": float(ep.lat.mean()), "disparity": float(ep.disp.mean())}
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(row))
    return row


def rule_grid(setting: str, workers: int) -> pd.DataFrame:
    path = OUT / "rules" / f"{setting}_grid.csv"
    if path.exists():
        return pd.read_csv(path)
    betas = [0, 0.5, 1, 1.5, 2, 2.5, 3, 4]
    ws = [-2, -1, -0.5, 0, 0.5, 1, 1.5, 2]
    mus = [0, 0.25, 0.5]
    jobs = [(setting, b, w, m) for b, w, m in itertools.product(betas, ws, mus)]
    with ProcessPoolExecutor(workers) as ex:
        grid = pd.DataFrame(list(ex.map(_rule_score, jobs)))
    path.parent.mkdir(parents=True, exist_ok=True)
    grid.to_csv(path, index=False)
    return grid


def _baseline_job(args):
    setting, label, spec = args
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    path = OUT / "eval" / setting / f"{label}_{EVAL_TAG}.pkl"
    if path.exists():
        return str(path)
    sys.path.insert(0, str(Path(__file__).parent))
    from rule_search import FixedRule
    from crimesim.eval.rollout import rollout
    from crimesim.rl import baselines as B
    if spec[0] == "rule":
        pol = FixedRule(*spec[1])
    elif spec[0] == "need":
        pol = B.NeedProportional(np.array(spec[1]))
    else:
        pol = {"uniform": B.UniformPolicy, "random": B.RandomPolicy, "ewma_hotspot": B.EWMAHotspotPolicy,
               "rf_hotspot": B.RandomForestHotspotPolicy, "oracle": B.GreedyOracle,
               "no_patrol": B.NoPatrolPolicy}[spec[0]]()
    df = rollout(pol, setting_cfg(setting), EVAL_SEEDS, keep_arrays=True).assign(policy=label)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_pickle(path)
    return str(path)


def baseline_jobs(settings) -> list:
    jobs = []
    for s in settings:
        r = refs(s)
        for name in ("uniform", "random", "ewma_hotspot", "rf_hotspot", "oracle", "no_patrol"):
            jobs.append((s, name, (name,)))
        jobs.append((s, "need_prop", ("need", r["need"])))
        jobs.append((s, "rule_ewma_like", ("rule", (1, 0, 0))))
        grid_path = OUT / "rules" / f"{s}_grid.csv"
        if grid_path.exists():
            g = pd.read_csv(grid_path)
            for label, row in [("best_rule_recorded", g.loc[g.rec_cov.idxmax()]),
                               ("best_rule_latent", g.loc[g.lat_cov.idxmax()])]:
                jobs.append((s, label, ("rule", (float(row.beta), float(row.w), float(row.mu)))))
    return jobs


# ------------------------------------------------------------------ driver
def log(msg):
    print(f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}", flush=True)


def timing(steps: int) -> None:
    """Train one naive PPO for ``steps`` and time one evaluation episode."""
    from crimesim.config import load_config
    from crimesim.eval.rollout import rollout
    from crimesim.experiments.registry import RUNS
    from crimesim.rl.ppo import load_obs_norm, train_ppo
    from stable_baselines3 import PPO
    cfg = setting_cfg("base", {"rl": {"seed": 0, "total_timesteps": steps}, "env": {"seed": 0}})
    out = RUNS / "jasss_timing"
    t0 = time.time()
    _, vn = train_ppo(cfg, save_dir=out)
    vn.close()
    dt = time.time() - t0
    t1 = time.time()
    rollout(PPO.load(out / "ppo_model"), load_config(out / "config.yaml"), [200],
            obs_norm=load_obs_norm(out / "obs_rms.npz"))
    ep = time.time() - t1
    per_model = dt * STEPS / steps / 60 + ep * len(EVAL_SEEDS) / 60
    log(f"training {steps} steps: {dt:.0f}s -> {dt * STEPS / steps / 60:.0f} min per 400k model (one process)")
    log(f"one evaluation episode: {ep:.1f}s -> {ep * len(EVAL_SEEDS) / 60:.1f} min per model")
    log(f"about {per_model:.0f} min per model alone; with 4 workers expect ~1.3-1.5x that per model")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--groups", nargs="*", default=["A10", "A11", "A13", "A14", "A6A7", "A17"])
    ap.add_argument("--seeds", type=int, nargs="*", default=TRAIN_SEEDS)
    ap.add_argument("--stages", nargs="*", default=["refs", "rules", "train", "baselines"])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--timing", action="store_true")
    ap.add_argument("--timing-steps", type=int, default=20_000)
    a = ap.parse_args()

    if a.timing:
        timing(a.timing_steps)
        return

    specs = [j for j in model_specs(a.seeds) if j[0] in a.groups]
    settings = list(dict.fromkeys(s for _, s, *_ in specs))
    if a.dry_run:
        df = pd.DataFrame(specs, columns=["group", "setting", "family", "over", "algo", "seed"])
        print(df.groupby(["group", "setting", "family"]).size().to_string())
        print(f"\n{len(specs)} models in {len(settings)} settings; groups {a.groups}; seeds {a.seeds}")
        return

    if "refs" in a.stages:
        with ProcessPoolExecutor(a.workers) as ex:
            list(ex.map(refs, settings))
        log(f"references ready for {len(settings)} settings")
    if "rules" in a.stages:
        for s in [s for s in RULE_SETTINGS if s in settings]:
            g = rule_grid(s, a.workers)
            log(f"rule grid {s}: best recorded {g.loc[g.rec_cov.idxmax(), ['beta', 'w', 'mu']].tolist()}, "
                f"best latent {g.loc[g.lat_cov.idxmax(), ['beta', 'w', 'mu']].tolist()}")
    failed = []
    if "train" in a.stages:
        log(f"{len(specs)} models, {a.workers} workers")
        with ProcessPoolExecutor(a.workers) as ex:
            futs = [ex.submit(_train_job, j) for j in specs]
            for i, f in enumerate(as_completed(futs), 1):
                name, status, dt, de, err = f.result()
                log(f"{i}/{len(specs)} {name}: {status}, train {dt / 60:.1f} min, eval {de / 60:.1f} min")
                if err:
                    failed.append(name)
                    log(err)
    if "baselines" in a.stages:
        jobs = baseline_jobs(settings)
        with ProcessPoolExecutor(a.workers) as ex:
            list(ex.map(_baseline_job, jobs))
        log(f"{len(jobs)} baseline rollouts ready")
    log(f"done; failed: {failed or 'none'}")


if __name__ == "__main__":
    main()
