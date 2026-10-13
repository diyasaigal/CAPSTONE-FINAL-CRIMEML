import crimesim._env_setup  # noqa: F401
"""Journal-readiness batch (plan steps 3, 4 and 7). Resumable like overnight.py.

Training queue (longest first):
  1. ppo_long_s{0,1,2}   naive PPO for 1.2M steps, snapshot every 100k
                         -> learning curve of reward vs disparity
  2. hyper-parameter check, 400k, seeds 0-2 (base: lr 3e-4, ent 0.01, 2048 steps/update)
       ppo_lr1e-3, ppo_lr1e-4, ppo_ent0, ppo_bigbatch (8192 steps/update, batch 512)
  3. sensitivity: naive PPO under each detection / observation variant of
     run_ablations.VARIANTS, 400k, seeds 0-2
  4. lag_a2.78_s{0,1,2}  Lagrangian PPO with the disparity ceiling at the
     GreedyOracle's cumulative disparity (robustness check for the penalty method)

Evaluation: every main-setting model on eval seeds 200-229 (same as results/final),
every snapshot on 200-209, and the sensitivity variants through
``run_ablations.py --kinds ppo`` on 300-319, each scored against its own oracle.

    PYTHONPATH=. python scripts/journal_runs.py --workers 8
"""
import argparse
import subprocess
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from crimesim.config import load_config
from crimesim.experiments.registry import RUNS, ensure_trained, eval_policy

sys.path.insert(0, str(Path(__file__).parent))
import run_ablations  # noqa: E402

BASE = "configs/base.yaml"
SEEDS = [0, 1, 2]
STEPS = 400_000
LONG_STEPS, SNAP = 1_200_000, 100_000
HPARAMS = {"ppo_lr1e-3": {"learning_rate": 1e-3},
           "ppo_lr1e-4": {"learning_rate": 1e-4},
           "ppo_ent0": {"ent_coef": 0.0},
           "ppo_bigbatch": {"n_steps": 8192, "batch_size": 512}}
SENS = ["low_detect_gain", "high_detect_gain", "low_p_report", "high_p_report",
        "weak_deterrence", "strong_deterrence", "no_socio_obs"]
LAG_ALPHA = 2.78          # GreedyOracle disparity_cum on eval seeds (results/final)
EVAL_SEEDS, EVAL_TAG = list(range(200, 230)), "e200_201n30"
SNAP_SEEDS, SNAP_TAG = list(range(200, 210)), "e200_201n10"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def build_queue() -> list[tuple[str, object, str]]:
    q = []
    for s in SEEDS:
        q.append((f"ppo_long_s{s}", load_config(
            BASE, env={"seed": s},
            rl={"seed": s, "algo": "ppo", "total_timesteps": LONG_STEPS, "checkpoint_every": SNAP}), "ppo"))
    for name, hp in HPARAMS.items():
        for s in SEEDS:
            q.append((f"{name}_s{s}", load_config(
                BASE, env={"seed": s}, rl={"seed": s, "algo": "ppo", "total_timesteps": STEPS, **hp}), "ppo"))
    for name, (cfg, algo) in run_ablations.train_specs(SENS, SEEDS, STEPS, 0.06, ("ppo",)).items():
        q.append((name, cfg, algo))
    for s in SEEDS:
        q.append((f"lag_a{LAG_ALPHA}_s{s}", load_config(
            BASE, env={"seed": s}, rl={"seed": s, "algo": "lagrangian_ppo", "total_timesteps": STEPS},
            fairness={"alpha": LAG_ALPHA}), "lagrangian_ppo"))
    return q


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
    name, out, seeds, tag = job
    t0 = time.time()
    try:
        eval_policy(name, out, load_config(BASE), seeds, tag)
        return name, time.time() - t0, None
    except Exception:
        return name, time.time() - t0, traceback.format_exc()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--skip-train", action="store_true")
    args = ap.parse_args()
    queue = build_queue()
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

    main_models = [n for n, _, _ in queue if not n.startswith("abl_") and n not in failed]
    jobs = [(n, RUNS / n, EVAL_SEEDS, EVAL_TAG) for n in main_models]
    for s in SEEDS:
        for d in sorted((RUNS / f"ppo_long_s{s}" / "checkpoints").glob("*")):
            jobs.append((f"ppo_long_s{s}@{d.name}", d, SNAP_SEEDS, SNAP_TAG))
    for s in SEEDS:                                   # the 400k models, for the same curve
        jobs.append((f"ppo_s{s}", RUNS / f"ppo_s{s}", SNAP_SEEDS, SNAP_TAG))
    log(f"{len(jobs)} evaluations")
    with ProcessPoolExecutor(args.workers) as ex:
        for i, f in enumerate(as_completed([ex.submit(_eval, j) for j in jobs]), 1):
            name, dt, err = f.result()
            log(f"eval {i}/{len(jobs)} {name}: {'FAILED' if err else 'ok'} in {dt:.0f}s")
            if err:
                failed.append(name)
                log(err)

    log("sensitivity evaluation (eval seeds 300-319, own-variant oracle)")
    subprocess.run([sys.executable, str(Path(__file__).parent / "run_ablations.py"),
                    "--only", "base", *SENS, "--kinds", "ppo", "--out-tag", "_sens"], check=False)
    log(f"done; failed: {failed or 'none'}")


if __name__ == "__main__":
    main()
