import crimesim._env_setup  # noqa: F401  (must precede numpy/torch)
"""Train one patrol-allocation policy.

    python scripts/train.py --config configs/ppo_unconstrained.yaml --out runs/ppo
    python scripts/train.py --config configs/base.yaml --penalty 0.06 --out runs/pen_p0.06
"""
import argparse
from pathlib import Path

from crimesim.config import load_config
from crimesim.rl.constrained_ppo import train_lagrangian_ppo, train_penalized_ppo
from crimesim.rl.ppo import train_ppo


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/base.yaml")
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--penalty", type=float, default=None, help="fairness penalty weight -> penalized_ppo")
    ap.add_argument("--alpha", type=float, default=None, help="disparity ceiling -> lagrangian_ppo")
    ap.add_argument("--timesteps", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--verbose", type=int, default=1)
    args = ap.parse_args()

    over: dict = {"rl": {}, "env": {}, "fairness": {}}
    if args.penalty is not None:
        over["rl"]["algo"] = "penalized_ppo"; over["fairness"]["penalty"] = args.penalty
    if args.alpha is not None:
        over["rl"]["algo"] = "lagrangian_ppo"; over["fairness"]["alpha"] = args.alpha
    if args.timesteps is not None:
        over["rl"]["total_timesteps"] = args.timesteps
    if args.seed is not None:
        over["rl"]["seed"] = args.seed; over["env"]["seed"] = args.seed
    over = {k: v for k, v in over.items() if v}
    cfg = load_config(args.config, **over)

    if cfg.rl.algo == "penalized_ppo":
        train_penalized_ppo(cfg, save_dir=args.out, verbose=args.verbose)
    elif cfg.rl.algo == "lagrangian_ppo":
        _, _, hist = train_lagrangian_ppo(cfg, save_dir=args.out, verbose=args.verbose)
        print(f"final lambda: {hist[-1][2]:.3f}" if hist else "no dual updates")
    else:
        train_ppo(cfg, save_dir=args.out, verbose=args.verbose)
    print(f"saved to {Path(args.out).resolve()}")


if __name__ == "__main__":
    main()
