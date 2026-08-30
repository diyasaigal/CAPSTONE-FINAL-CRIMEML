import crimesim._env_setup  # noqa: F401  (must precede numpy/torch)
"""Fast end-to-end sanity check: train a short naive PPO and a fairness-penalised
PPO, compare to the baselines. Not a paper run -- just checks the story appears.
"""
import argparse
import time
from pathlib import Path

import pandas as pd

from crimesim.abm.regions import load_region_info
from crimesim.config import load_config
from crimesim.eval.incidence import build_panel, incidence_regression
from crimesim.eval.rollout import rollout, summarise
from crimesim.rl.baselines import make_baselines
from crimesim.rl.constrained_ppo import train_penalized_ppo
from crimesim.rl.ppo import train_ppo

pd.set_option("display.width", 240); pd.set_option("display.max_columns", 30)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--timesteps", type=int, default=150_000)
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--penalty", type=float, default=0.06)
    args = ap.parse_args()

    cfg = load_config("configs/base.yaml",
                      rl={"total_timesteps": args.timesteps, "seed": 0}, env={"seed": 0})
    pen_cfg = load_config("configs/base.yaml",
                          rl={"total_timesteps": args.timesteps, "seed": 0, "algo": "penalized_ppo"},
                          env={"seed": 0}, fairness={"penalty": args.penalty})

    t0 = time.time()
    ppo_model, ppo_vn = train_ppo(cfg, verbose=0)
    print(f"[t+{time.time()-t0:.0f}s] naive PPO trained", flush=True)
    pen_model, pen_vn, _ = train_penalized_ppo(pen_cfg, verbose=0)
    print(f"[t+{time.time()-t0:.0f}s] penalised PPO trained", flush=True)

    ev = load_config("configs/base.yaml", env={"shifts_per_episode": 150, "warmup_shifts": 20})
    seeds = range(1000, 1000 + args.seeds)
    frames = [rollout(p, ev, seeds=seeds, keep_arrays=True) for p in make_baselines().values()]
    frames.append(rollout(ppo_model, ev, seeds=seeds, vecnorm=ppo_vn, keep_arrays=True).assign(policy="ppo"))
    frames.append(rollout(pen_model, ev, seeds=seeds, vecnorm=pen_vn, keep_arrays=True).assign(policy="ppo_fair"))
    df = pd.concat(frames, ignore_index=True)

    print("\n=== E1: coverage & fairness (held-out seeds) ===")
    print(summarise(df)[["policy", "rec_cov", "lat_cov", "lat_cov_ci", "disparity",
                         "disparity_shift", "gini"]].to_string(index=False))

    panel = build_panel(df, load_region_info(20))
    print("\n=== E3: over-policing ~ hardship | latent share ===")
    for pol in ["uniform", "ewma_hotspot", "ppo", "ppo_fair", "oracle"]:
        h = incidence_regression(panel, pol)["hardship"]
        print(f"  {pol:13s} hardship coef={h['coef']:+.4f} (p={h['p']:.2g})")

    Path("results").mkdir(exist_ok=True)
    df.to_pickle("results/quicktest.pkl")
    print("\nsaved results/quicktest.pkl")


if __name__ == "__main__":
    main()
