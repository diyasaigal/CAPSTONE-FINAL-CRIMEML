import crimesim._env_setup  # noqa: F401
"""Overnight batch for the pivot (PIVOT_AUDIT.md section F): train every model
still missing, in parallel, then evaluate. Resumable -- rerun it after an
interruption and finished models / rollouts are reused (``ensure_trained`` and
``eval_policy`` cache on disk and refuse a model trained with another config).

Training queue, in priority order (all 400k steps):
  1. E5 (Phase 3): base / perfect_obs / const_detect x {ppo, fair} x seeds 0-2
  2. main families (E1-E4): seeds 0-2 retrained at 400k (3-4 already exist)
  3. 2x2 attribution: what PPO observes x what it is rewarded on, seeds 0-2
       ppo_obslat = sees latent crime, rewarded on recorded coverage
       ppo_rewlat = sees recorded only, rewarded on latent coverage
     (ppo and ppo_oracle fill the other two cells)
  4. oracle-referenced penalty sweep: w in {0.05, 0.15, 0.5} x seeds 0-2,
     penalty = w * max(0, ln disparity_shift - ln ref), ref = the GreedyOracle's
     mean per-shift disparity on reference seeds 1000-1009 (disjoint from eval)

Evaluation: E5 summary on eval seeds 300-319 (scripts/run_ablations.py), and
every main / 2x2 / penalty model plus the rule-based baselines on 200-229.
Only new cache files are written; nothing in results/ is overwritten.

    PYTHONPATH=. python scripts/overnight.py --workers 6
"""
import argparse
import json
import subprocess
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from crimesim.config import load_config
from crimesim.experiments.registry import RUNS, ensure_trained, eval_baselines, eval_policy, rl_specs
from crimesim.eval.rollout import rollout
from crimesim.rl.baselines import GreedyOracle

sys.path.insert(0, str(Path(__file__).parent))
import run_ablations  # noqa: E402

RESULTS = Path("results")
BASE = "configs/base.yaml"
STEPS = 400_000
SEEDS = [0, 1, 2]
PENALTIES = [0.02, 0.06, 0.15, 0.4]
REF_WEIGHTS = [0.05, 0.15, 0.5]
E5_VARIANTS = ["base", "perfect_obs", "const_detect"]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def oracle_reference(seeds=range(1000, 1010)) -> float:
    cache = RESULTS / "oracle_ref_e1000n10.json"
    if cache.exists():
        return json.loads(cache.read_text())["disparity_shift_mean"]
    df = rollout(GreedyOracle(), load_config(BASE), list(seeds))
    ref = {"seeds": list(seeds), "disparity_shift_mean": float(df.disparity_shift.mean()),
           "disparity_cum_final_mean": float(df.groupby("seed").disparity_cum.last().mean())}
    cache.write_text(json.dumps(ref, indent=2))
    return ref["disparity_shift_mean"]


def build_queue(ref: float) -> list[tuple[str, object, str]]:
    q = list(run_ablations.train_specs(E5_VARIANTS, SEEDS, STEPS, 0.06).items())
    for name, (cfg, algo) in rl_specs(BASE, SEEDS, PENALTIES).items():
        q.append((name, (load_config(BASE, **_with_steps(cfg)), algo)))
    for s in SEEDS:
        common = {"rl": {"seed": s, "total_timesteps": STEPS, "algo": "ppo"}}
        q.append((f"ppo_obslat_s{s}", (load_config(
            BASE, env={"seed": s, "observe_latent": True, "reward_on": "captured"}, **common), "ppo")))
        q.append((f"ppo_rewlat_s{s}", (load_config(
            BASE, env={"seed": s, "observe_latent": False, "reward_on": "latent_coverage"}, **common), "ppo")))
        for w in REF_WEIGHTS:
            q.append((f"penref_w{w}_s{s}", (load_config(
                BASE, env={"seed": s},
                rl={"seed": s, "total_timesteps": STEPS, "algo": "penalized_ppo"},
                fairness={"penalty": w, "penalty_mode": "excess_log_ref",
                          "ref_disparity": round(ref, 4)}), "penalized_ppo")))
    return [(n, c, a) for n, (c, a) in q]


def _with_steps(cfg) -> dict:
    d = cfg.to_dict()
    d["rl"]["total_timesteps"] = STEPS
    return d


def _train(job):
    name, cfg, algo = job
    t0 = time.time()
    try:
        existed = (RUNS / name / "ppo_model.zip").exists()
        ensure_trained(name, cfg, algo)
        return name, time.time() - t0, "cached" if existed else "trained", None
    except Exception:
        return name, time.time() - t0, "FAILED", traceback.format_exc()


def _eval(job):
    name, cfg, seeds, tag = job
    t0 = time.time()
    try:
        ecfg = load_config(BASE, env={"observe_latent": cfg.env.observe_latent,
                                      "reward_on": cfg.env.reward_on})
        eval_policy(name, RUNS / name, ecfg, seeds, tag)
        return name, time.time() - t0, None
    except Exception:
        return name, time.time() - t0, traceback.format_exc()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--skip-train", action="store_true")
    args = ap.parse_args()
    RESULTS.mkdir(exist_ok=True)

    ref = oracle_reference()
    log(f"GreedyOracle reference disparity (per shift, seeds 1000-1009): {ref:.4f}")
    queue = build_queue(ref)
    log(f"{len(queue)} models in queue, {args.workers} workers")

    failed = []
    if not args.skip_train:
        with ProcessPoolExecutor(args.workers) as ex:
            futs = [ex.submit(_train, j) for j in queue]
            for i, f in enumerate(as_completed(futs), 1):
                name, dt, status, err = f.result()
                log(f"train {i}/{len(queue)} {name}: {status} in {dt / 60:.1f} min")
                if err:
                    failed.append(name)
                    log(err)

    log("E5 evaluation and summary (eval seeds 300-319)")
    subprocess.run([sys.executable, str(Path(__file__).parent / "run_ablations.py"),
                    "--only", *E5_VARIANTS], check=False)

    seeds = list(range(200, 230))
    tag = "e200_201n30"
    log(f"baselines on eval seeds 200-229")
    eval_baselines(load_config(BASE), seeds, RESULTS, tag)
    jobs = [(n, c, seeds, tag) for n, c, _ in queue if not n.startswith("abl_") and n not in failed]
    jobs += [(n, load_config(RUNS / n / "config.yaml"), seeds, tag)
             for n in [f"{f}_s{s}" for f in ["ppo", "ppo_oracle", *(f"pen_p{p}" for p in PENALTIES)]
                       for s in (3, 4)]]
    with ProcessPoolExecutor(args.workers) as ex:
        for i, f in enumerate(as_completed([ex.submit(_eval, j) for j in jobs]), 1):
            name, dt, err = f.result()
            log(f"eval {i}/{len(jobs)} {name}: {'FAILED' if err else 'ok'} in {dt:.0f}s")
            if err:
                failed.append(name)
                log(err)
    log(f"done; failed: {failed or 'none'}")


if __name__ == "__main__":
    main()
